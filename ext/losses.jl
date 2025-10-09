"""
    select_loss_function()
Helper function to pass a valid loss function to Training.single_train_step.
Selects a loss function based on the provided physics-informed loss function, in the standard workflow generated with create_physics_loss.

# Arguments
- `PI_loss`: Physics-informed loss function (default is a zero loss function)
"""
function select_loss_function(PI_loss::Function=create_physics_loss(nothing); subsampling::Int=1, α::Float32=0.5f0)
    function loss_function(model::Lux.AbstractLuxLayer, ps::NamedTuple, st::NamedTuple, (u_t1, target_data)::Tuple{AbstractArray, AbstractArray}; α=α)
        u_net = StatefulLuxLayer{true}(model, ps, st)
        data_loss = mse_loss_function(u_net, target_data, u_t1; subsampling=subsampling)
        physics_loss = PI_loss(u_net, u_t1)
        loss = (1 - α) * physics_loss + α * data_loss
        return (loss,
            (st),
            (;physics_loss, data_loss)
        )
    end
    return loss_function
end
"""
    physics_informed_loss_QG3(u::StatefulLuxLayer, q_0::AbstractArray)
Compute residual loss for QG3 equation.

# Arguments
- `u`: Neural network (StatefulLuxLayer)
- `q_0`: Initial state (input data)

# Returns
- Residual loss (mean squared residual)

# Requirements
- Precomputed right-hand side `rhs` (global variable) computed using QG3.QG3MM_gpu
- Precomputed time step `dt` (global variable) in QG3 units
"""
@views function physics_informed_loss_QG3(u::StatefulLuxLayer, q_0::AbstractArray)
    q_pred = u(q_0)
    ∂u_∂t = (q_pred .- q_0) ./ dt
    residual = ∂u_∂t .- rhs
    return mean(abs2, residual)
end

"""
    QG3_loss_function(model, ps, st, (u_t1, target_data); α=0.5f0)
Combined physics-data loss function for QG3.

# Arguments
- `model`: Lux model
- `ps`: Model parameters
- `st`: Model state
- `u_t1`: Input data
- `target_data`: Training targets
- `α`: Loss weighting (0.5 = equal weighting)

# Returns
- Tuple containing:
    - Total loss
    - Updated state
    - Named tuple with loss components (physics_loss, data_loss)
"""
function QG3_loss_function(model::AbstractLuxLayer, ps::NamedTuple, st::NamedTuple, (u_t1, target_data)::Tuple{AbstractArray, AbstractArray}; α::Float32=0.5f0)
    u_net = StatefulLuxLayer{true}(model, ps, st)
    data_loss = mse_loss_function(u_net, target_data, u_t1)
    physics_loss = physics_informed_loss_QG3(u_net, u_t1)
    loss = (1 - α) * physics_loss + α * data_loss
    return (loss,
        (st),
        (;physics_loss, data_loss)
    )
end
"""
    QG3_Physics_Parameters(dt::Float64, rhs::AbstractArray)
Create a struct to hold parameters for QG3 physics loss.

# Fields
- `dt`: Time step (scalar)
- `qg3p`: QG3 model parameters 
- `S`: precomputed forcing
- `ggsh`: Gaussian grid to spherical harmonics transform
- `shgg`: Spherical harmonics to Gaussian grid transform
- `μ`: Mean from data normalization
- `σ`: Standard deviation from data normalization
"""
struct QG3_Physics_Parameters
    dt::Float64
    qg3p::QG3.QG3Model
    S::AbstractArray
    ggsh::QG3.GaussianGridtoSHTransform
    shgg::QG3.SHtoGaussianGridTransform
    μ::Real
    σ::Real
end
"""
    create_QG3_physics_loss()
Helper function to create a QG3 physics loss function.

# Arguments
- `params`: parameters struct, pass nothing to create a zero loss function.
"""
function create_QG3_physics_loss(params::QG3_Physics_Parameters)
    @views function QG3_physics_loss(u::StatefulLuxLayer, q_0::AbstractArray{T,4}) where T<:Real
        q_pred = u(q_0) .* params.σ .+ params.μ
        #bc_loss = mean(abs2, q_pred[:,1:1,:,:] .- q_pred[:,end:end,:,:])
        q_0_denormalized = q_0 .* params.σ .+ params.μ
        ∂u_∂t = (q_pred .- q_0_denormalized) ./ params.dt
        ∂u_∂t = permutedims(∂u_∂t, (3, 1, 2, 4)) 
        q_pred_perm = permutedims(q_pred, (3, 1, 2, 4))
        q_pred_perm = transform_SH(q_pred_perm, params.ggsh)
        
        # Process each batch sample individually
        results = map(sample -> QG3.QG3MM_gpu(sample, (params.qg3p, params.S), (0,1)), eachslice(q_pred_perm; dims=4))
        
        # Combine results without splatting
        rhs_new = reduce((acc, x) -> cat(acc, x; dims=4), results)
        #rhs = cat(results...; dims=4)
        
        rhs_new = transform_grid(rhs_new, params.shgg)

        #rhs_final = permutedims(rhs_new, (2, 3, 1, 4))
        residual = ∂u_∂t .- rhs_new 
        return mean(abs2, residual) 
    end    
end

function create_QG3_physics_loss(::Nothing)
    function QG3_physics_loss(u::StatefulLuxLayer, q_0::AbstractArray)
        return 0.f0
    end
    return QG3_physics_loss
end

function mse_loss_function_QG3(u::StatefulLuxLayer, target::AbstractArray, u_t1::AbstractArray)
    return MSELoss()(u(u_t1), target)
end
"""
    select_QG3_loss_function(PI_loss::Function=create_QG3_physics_loss(nothing); SFNO::Bool=false)
Helper function to pass a valid QG3 loss function to Training.single_train_step.
Selects a loss function based on the provided physics-informed loss function, in the standard workflow generated with create_QG3_physics_loss.

# Arguments
- `PI_loss`: Physics-informed loss function (default is a zero loss function)
"""
function select_QG3_loss_function(PI_loss::Function=create_QG3_physics_loss(nothing))
    function QG3_loss_function(model::Lux.AbstractLuxLayer, ps::NamedTuple, st::NamedTuple, (u_t1, target_data)::Tuple{AbstractArray, AbstractArray}; α::Float32=0.5f0)
        u_net = StatefulLuxLayer{true}(model, ps, st)
        data_loss = mse_loss_function_QG3(u_net, target_data, u_t1)
        physics_loss = PI_loss(u_net, u_t1)
        loss = (1 - α) * physics_loss + α * data_loss
        return (loss,
            (st),
            (;physics_loss, data_loss)
        )
    end
    return QG3_loss_function
end
