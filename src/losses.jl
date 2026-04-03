struct SpectralPhysicsLossParameters{T<:Real, A<:AbstractArray}
    ν::T
    Δt::Int
    L::T
    x_σ::T
    x_μ::T
    t_step_length::T
    ik::A
    ik2::A
end

function SpectralPhysicsLossParameters(ν::Real, Δt::Int, L::Real, x_σ::Real, x_μ::Real,
                                        t_step_length::Real, nx::Int;
                                        T::Type{<:Real}=Float32, device=gpu_device())
    k_max   = nx ÷ 2
    k       = T(2π) / T(L) .* T.(0:k_max)
    ik_cpu  = Complex{T}.(im .* k)
    ik2_cpu = Complex{T}.(-k .^ 2)
    ik  = device(ik_cpu)
    ik2 = device(ik2_cpu)
    return SpectralPhysicsLossParameters{T, typeof(ik)}(
        T(ν), Δt, T(L), T(x_σ), T(x_μ), T(t_step_length), ik, ik2)
end

function SpectralPhysicsLossParameters(ν::Real, Δt::Int, L::Real,
                                        t_step_length::Real, nx::Int;
                                        T::Type{<:Real}=Float32, device=gpu_device())
    return SpectralPhysicsLossParameters(ν, Δt, L, one(T), zero(T), t_step_length, nx; T=T, device=device)
end

function Base.convert(::Type{SpectralPhysicsLossParameters{T}}, p::SpectralPhysicsLossParameters) where T<:Real
    return SpectralPhysicsLossParameters{T, typeof(p.ik)}(
        T(p.ν), p.Δt, T(p.L), T(p.x_σ), T(p.x_μ), T(p.t_step_length), p.ik, p.ik2)
end
# 1. spectral_derivative: uses precomputed ik/ik2 from params, operates on full array
function spectral_derivative(u::AbstractArray, params::SpectralPhysicsLossParameters)
    nx  = size(u, 1)
    ik  = reshape(params.ik,  length(params.ik),  ntuple(_->1, ndims(u)-1)...)
    ik2 = reshape(params.ik2, length(params.ik2), ntuple(_->1, ndims(u)-1)...)
    u_h = rfft(u, 1)
    ux  = irfft(ik  .* u_h, nx, 1)
    uxx = irfft(ik2 .* u_h, nx, 1)
    return ux, uxx
end

function ChainRulesCore.rrule(::typeof(spectral_derivative), u::AbstractArray,
                               params::SpectralPhysicsLossParameters)
    ux, uxx = spectral_derivative(u, params)
    function spectral_derivative_pullback(Δ)
        Δux, Δuxx = Δ
        nx  = size(u, 1)
        ik  = reshape(params.ik,  length(params.ik),  ntuple(_->1, ndims(u)-1)...)
        ik2 = reshape(params.ik2, length(params.ik2), ntuple(_->1, ndims(u)-1)...)
        Δu = irfft(conj(ik)  .* rfft(real.(Δux),  1), nx, 1) .+
             irfft(conj(ik2) .* rfft(real.(Δuxx), 1), nx, 1)
        return NoTangent(), Δu, NoTangent()
    end
    return (ux, uxx), spectral_derivative_pullback
end

# 2. spatial_derivative: operates on full array
function spatial_derivative(u::AbstractArray, params::SpectralPhysicsLossParameters)
    ux, uxx = spectral_derivative(u, params)
    return u .* ux, uxx
end

# 3. spatial_derivatives_batch: no eachslice, pass full array directly
function spatial_derivatives_batch(states::AbstractArray, params::SpectralPhysicsLossParameters)
    return spatial_derivative(states, params)
end

# 4. physics_loss 4D branch: compute spatial derivs on FULL u_t2, slice residual only
function create_physics_loss(params::SpectralPhysicsLossParameters)
    function physics_loss(u::StatefulLuxLayer, u_t1::AbstractArray)
        u_t2 = u(u_t1)
        u_t2 = u_t2 .* params.x_σ .+ params.x_μ
        u_t1 = u_t1 .* params.x_σ .+ params.x_μ
        if ndims(u_t1) == 3
            ∂u_∂t        = (u_t2 .- u_t1) ./ params.t_step_length
            ∂f_∂x, ∂u_∂xx = spatial_derivative(u_t2, params)
        else
            ∂u_∂t         = (u_t2[:, 3:end, :, :] .- u_t2[:, 1:end-2, :, :]) ./ (2 * params.t_step_length)
            ∂f_∂x, ∂u_∂xx = spatial_derivatives_batch(u_t2, params)   # full u_t2, no slice
            # trim spatial results to match interior time points
            ∂f_∂x  = ∂f_∂x[:,  2:end-1, :, :]
            ∂u_∂xx = ∂u_∂xx[:, 2:end-1, :, :]
        end
        return mean(abs2, ∂u_∂t .+ ∂f_∂x .- params.ν .* ∂u_∂xx)
    end
    return physics_loss
end

function physics_loss(u_t2::AbstractArray, u_t1::AbstractArray, params::SpectralPhysicsLossParameters)
         # Access params via the captured struct
        u_t2 = u_t2 .* params.x_σ .+ params.x_μ
        u_t1 = u_t1 .* params.x_σ .+ params.x_μ
        if ndims(u_t1) == 3
            ∂u_∂t = (u_t2 .- u_t1) ./ params.t_step_length
            #boundary_residual = u_t2[1:1,:,:] .- u_t2[end:end,:,:]
            ∂f_∂x, ∂u_∂xx = spatial_derivative(u_t2, params)
            boundary_residual = 0
        else 
            #forward_diff_first = (u_t2[:, 2:2, :, :] .- u_t2[:, 1:1, :, :]) ./ params.t_step_length
            #forward_diff_last = (u_t2[:, end:end, :, :] .- u_t2[:, end-1:end-1, :, :]) ./ params.t_step_length
            ∂u_∂t= (u_t2[:, 3:end, :, :] .- u_t2[:, 1:end-2, :, :]) ./ (2 * params.t_step_length)
            #∂u_∂t = cat(forward_diff_first, central_diff, forward_diff_last; dims=2) 
            ∂f_∂x, ∂u_∂xx = spatial_derivatives_batch(u_t2, params)
            ∂f_∂x  = ∂f_∂x[:,  2:end-1, :, :]
            ∂u_∂xx = ∂u_∂xx[:, 2:end-1, :, :]
            boundary_residual = 0
        end
        return mean(abs2, ∂u_∂t .+ ∂f_∂x .- (params.ν .* ∂u_∂xx)) + mean(abs2, boundary_residual)
    end
"""
    FDPhysicsLossParameters(ν::T, N_t::Int, t_max::T, t_min::T, Δt::T, x_σ::T, x_μ::T, M1_gpu::AbstractArray, M2_gpu::AbstractArray)

Create a struct to hold parameters for finite difference physics loss.

# Fields
- `ν`: Viscosity (scalar)
- 't_step_length`: Time step length (scalar)
- `M1_gpu`: Second derivative FD matrix (GPU array)
- `M2_gpu`: First derivative FD matrix (GPU array)
"""
struct FDPhysicsLossParameters{T} 
    ν::T
    Δt::Int
    t_step_length::T
    x_σ::T
    x_μ::T
    M1_gpu::AbstractArray
    M2_gpu::AbstractArray
end

function FDPhysicsLossParameters(ν::Real, Δt::Int, t_step_length::Real, x_σ::Real, x_μ::Real, 
                                  M1_gpu::AbstractArray, M2_gpu::AbstractArray; T::Type{<:Real}=Float32)
    return FDPhysicsLossParameters{T}(T(ν), Δt, T(t_step_length), T(x_σ), T(x_μ), M1_gpu, M2_gpu)
end

function FDPhysicsLossParameters(ν::Real, Δt::Int, t_step_length::Real,
                                  M1_gpu::AbstractArray, M2_gpu::AbstractArray; T::Type{<:Real}=Float32)
    return FDPhysicsLossParameters(ν, Δt, t_step_length, one(T), zero(T), M1_gpu, M2_gpu; T=T)
end

function create_physics_loss(params::FDPhysicsLossParameters)
    # Capture `params` in the closure
    function BurgersFD_physics_loss(u::StatefulLuxLayer, u_t1::AbstractArray)
        # Access params via the captured struct
        u_t2 = u(u_t1)
        u_t2 = u_t2 .* params.x_σ .+ params.x_μ
        u_t1 = u_t1 .* params.x_σ .+ params.x_μ
        if ndims(u_t1) == 3
            ∂u_∂t = (u_t2 .- u_t1) ./ params.t_step_length
            #boundary_residual = u_t2[1:1,:,:] .- u_t2[end:end,:,:]
            boundary_residual = 0
            ∂f_∂x, ∂u_∂xx = spatial_derivatives_batch(u_t2, params)
        else 
            #forward_diff_first = (u_t2[:, 2:2, :, :] .- u_t2[:, 1:1, :, :]) ./ params.t_step_length
            #forward_diff_last = (u_t2[:, end:end, :, :] .- u_t2[:, end-1:end-1, :, :]) ./ params.t_step_length
            ∂u_∂t= (u_t2[:, 3:end, :, :] .- u_t2[:, 1:end-2, :, :]) ./ (2 * params.t_step_length)
            #∂u_∂t = cat(forward_diff_first, central_diff, forward_diff_last; dims=2)
            #boundary_residual = u_t2[1:1,:,:,:] .- u_t2[end:end,:,:,:]
            boundary_residual = 0
            ∂f_∂x, ∂u_∂xx = spatial_derivatives_batch(u_t2[:,2:end-1,:,:], params)
        end
        
        return mean(abs2, ∂u_∂t .+ ∂f_∂x .- (params.ν .* ∂u_∂xx)) + mean(abs2, boundary_residual)
    end
    return BurgersFD_physics_loss
end

function physics_loss(u_t2::AbstractArray, u_t1::AbstractArray, params::FDPhysicsLossParameters)
         # Access params via the captured struct
        u_t2 = u_t2 .* params.x_σ .+ params.x_μ
        u_t1 = u_t1 .* params.x_σ .+ params.x_μ
        if ndims(u_t1) == 3
            ∂u_∂t = (u_t2 .- u_t1) ./ params.t_step_length
            #boundary_residual = u_t2[1:1,:,:] .- u_t2[end:end,:,:]
            ∂f_∂x, ∂u_∂xx = spatial_derivatives_batch(u_t2, params)
            boundary_residual = 0
        else 
            #forward_diff_first = (u_t2[:, 2:2, :, :] .- u_t2[:, 1:1, :, :]) ./ params.t_step_length
            #forward_diff_last = (u_t2[:, end:end, :, :] .- u_t2[:, end-1:end-1, :, :]) ./ params.t_step_length
            ∂u_∂t= (u_t2[:, 3:end, :, :] .- u_t2[:, 1:end-2, :, :]) ./ (2 * params.t_step_length)
            #∂u_∂t = cat(forward_diff_first, central_diff, forward_diff_last; dims=2) 
            ∂f_∂x, ∂u_∂xx = spatial_derivatives_batch(u_t2[:,2:end-1,:,:], params)
            boundary_residual = 0
        end
        return mean(abs2, ∂u_∂t .+ ∂f_∂x .- (params.ν .* ∂u_∂xx)) + mean(abs2, boundary_residual)
    end
"""
    mse_loss_function(u::StatefulLuxLayer, target::AbstractArray, xt::AbstractArray)

Standard mean squared error loss.

# Arguments
- `u`: Neural network
- `target`: Ground truth values
- `u_t1`: Network inputs

# Returns
- MSE between network output and target
"""
function mse_loss_function(u::StatefulLuxLayer, target::AbstractArray{T,3}, u_t1::AbstractArray{T,3}; subsampling::Int=1) where T
    return MSELoss()(u(u_t1)[1:subsampling:end,:,:], target[1:subsampling:end,:,:])
end
function mse_loss_function(u::StatefulLuxLayer, target::AbstractArray{T,4}, u_t1::AbstractArray{T,4}; subsampling::Int=1) where T
    return MSELoss()(u(u_t1)[1:subsampling:end,1:subsampling:end,:,:], target[1:subsampling:end,1:subsampling:end,:,:])
end
function create_physics_loss(::Nothing)
    function physics_loss(u::StatefulLuxLayer, u_t1::AbstractArray)
        return 0.f0
    end
    return physics_loss
end 
    
"""
    autoregressive_loss(model::StatefulLuxLayer, (u0, target)::Tuple{AbstractArray, AbstractArray}, n_steps::Int, params::FDPhysicsLossParameters, λ::Float32)

Compute autoregressive loss for a model over multiple time steps.
# Arguments 
- `model`: StatefulLuxLayer model
- `u0`: Initial state (input data)
- `target`: Target data for comparison
- `n_steps`: Number of time steps to propagate
- `params`: FDPhysicsLossParameters struct containing physics parameters
- `λ`: Weighting factor for physics loss
# Returns
- Total loss combining data loss and physics-informed loss
"""
function autoregressive_loss(
    model::StatefulLuxLayer,
    (u0,target)::Tuple{AbstractArray,AbstractArray},
    n_steps::Int,
    params::FDPhysicsLossParameters,
    λ::Float32
)
    function autoregressive_propagate(model, u0::AbstractArray, n_steps::Int)
        # Initialize immutable tuple with initial state
        states = (u0,)
        
        # Build state sequence through tuple concatenation
        for _ in 1:n_steps
            u_next = model(states[end])
            states = (states..., u_next)  # Immutable update
        end
        
        # Combine into single array
        return cat(states...; dims=4)
    end
    
    states_arr = autoregressive_propagate(model, u0, n_steps)

    # Vectorized time derivative calculation
    Δt = params.t_step_length / n_steps
    forward_diff = (states_arr[:, :, :, 2:end] .- states_arr[:, :, :, 1:end-1]) ./ Δt
    central_diff = (states_arr[:, :, :, 3:end] .- states_arr[:, :, :, 1:end-2]) ./ (2Δt)
    du_dt = cat(forward_diff[:, :, :, 1:1], central_diff, forward_diff[:, :, :, end:end]; dims=4)

    # Batched physics calculations
    ∂f_∂x_batch, ∂u_∂xx_batch = spatial_derivatives_batch(states_arr[:, :, :, 1:end], params)
    
    # Vectorized residuals and boundary loss
    residuals = du_dt .+ ∂f_∂x_batch .- (params.ν .* ∂u_∂xx_batch)
    boundary_loss = mean(abs2, states_arr[:, 1:1, :, 2:end] .- states_arr[:, end:end, :, 2:end])
    physics_loss = mean(abs2, residuals) + boundary_loss

    # Data loss calculation
    data_loss = mean(abs2, states_arr[:, :, :, end] .- target)

    return data_loss + λ * physics_loss
end

function spatial_derivatives_batch(states::AbstractArray, params::FDPhysicsLossParameters)
    dims = size(states)
    
    # Maintain batch dimensions for matrix operations
    f = states.^2 ./ 2
    ∂f_∂x = reshape(params.M2_gpu * reshape(f, dims[1], :), dims)
    ∂u_∂xx = reshape(params.M1_gpu * reshape(states, dims[1], :), dims)
    return ∂f_∂x, ∂u_∂xx
end

function create_autoregressive_loss_function(params::FDPhysicsLossParameters)
   function loss_function(model, ps, st, batch)
    u_net = StatefulLuxLayer{true}(model, ps, st)
    u0, target = batch
    loss = autoregressive_loss(
        u_net,
        (u0, target),
        3,  # n_steps
        params,
        0.1f0
    )
    return (loss, (st), ())
    end 
    return loss_function
end
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