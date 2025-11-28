############################
# QG3 Physics Parameters
############################
struct QG3_Physics_Parameters
    dt::Real
    qg3p::QG3.QG3Model
    S::AbstractArray
    ggsh::QG3.GaussianGridtoSHTransform
    shgg::QG3.SHtoGaussianGridTransform
    μ::Union{Real, AbstractArray}
    σ::Union{Real, AbstractArray}
    weights::AbstractArray
end
"""
$(TYPEDSIGNATURES)

Helper constructor to pass as empty default to train_model
"""
function QG3_Physics_Parameters(;n_lat=32, modes=21, batch_size=1, gpu::Bool=true)
    pars = qg3pars_constructor_helper(modes, n_lat)
    if !gpu
        QG3.gpuoff()
    end
    ggsh = QG3.GaussianGridtoSHTransform(pars, N_batch=batch_size)
    shgg = QG3.SHtoGaussianGridTransform(pars, N_batch=batch_size)
    qg3p = CUDA.@allowscalar QG3Model(pars)
    #geom_weights = QG3.togpu(reshape(pars.μ, 1, : , 1, 1))
    quad_weights = QG3.togpu(reshape(QG3.compute_GaussWeights(pars), :, 1, 1, 1))
    #weights = geom_weights .* quad_weights
    S = CUDA.@allowscalar QG3.reorder_SH_gpu(QG3.zeros_SH(pars),pars)
    dt = σ = 1
    μ = 0
    QG3.gpuon()
    return QG3_Physics_Parameters(dt, qg3p, S, ggsh, shgg, μ, σ, quad_weights)
end 

function QG3_Physics_Parameters(pars::QG3.QG3ModelParameters; batch_size=1, gpu::Bool=true)
    if !gpu
        QG3.gpuoff()
    end
    ggsh = QG3.GaussianGridtoSHTransform(pars, N_batch=batch_size)
    shgg = QG3.SHtoGaussianGridTransform(pars, N_batch=batch_size)
    qg3p = CUDA.@allowscalar QG3Model(pars)
    #geom_weights = QG3.togpu(reshape(pars.μ, :,1 , 1, 1))
    quad_weights = QG3.togpu(reshape(QG3.compute_GaussWeights(pars), :, 1, 1, 1))
    #weights = geom_weights .* quad_weights
    S = CUDA.@allowscalar QG3.reorder_SH_gpu(QG3.zeros_SH(pars),pars)
    dt = σ = 1
    μ = 0
    QG3.gpuon()
    return QG3_Physics_Parameters(dt, qg3p, S, ggsh, shgg, μ, σ, quad_weights) 
end

function QG3_Physics_Parameters(dt::Real,
                                qg3p::QG3.QG3Model,
                                S::AbstractArray,
                                μ::Union{Real, AbstractArray},
                                σ::Union{Real, AbstractArray};
                                batch_size::Int=1,
                                gpu::Bool=true)
    if !gpu
        QG3.gpuoff()
    end                            
    ggsh = QG3.GaussianGridtoSHTransform(qg3p.p, N_batch=batch_size)
    shgg = QG3.SHtoGaussianGridTransform(qg3p.p, N_batch=batch_size)

    #geom_weights = QG3.togpu(reshape(qg3p.p.μ, :, 1, 1, 1))
    quad_weights = QG3.togpu(reshape(QG3.compute_GaussWeights(qg3p.p), :, 1, 1, 1))
    #weights = geom_weights .* quad_weights
    QG3.gpuon()
    return QG3_Physics_Parameters(dt, qg3p, S, ggsh, shgg, μ, σ, quad_weights)
end

function QG3_Physics_Parameters(dt::Real,
    qg3p::QG3.QG3Model,
    S::AbstractArray,
    ggsh::QG3.GaussianGridtoSHTransform,
    shgg::QG3.SHtoGaussianGridTransform,
    μ::Union{Real, AbstractArray},
    σ::Union{Real, AbstractArray};
    gpu::Bool=true)
    if !gpu
        QG3.gpuoff()
    end
    #geom_weights = QG3.togpu(reshape(qg3p.p.μ, :, 1, 1, 1))
    quad_weights = QG3.togpu(reshape(QG3.compute_GaussWeights(qg3p.p), :, 1, 1, 1))
    #weights = geom_weights .* quad_weights
    QG3.gpuon()
    return QG3_Physics_Parameters(dt, qg3p, S, ggsh, shgg, μ, σ, quad_weights)
end
############################
# Base Loss Components
############################

# Simple MSE loss (baseline)
mse_loss_function_QG3(u, target, input) =
    Lux.MSELoss()(u(input), target)

# Geometrically weighted MSE losses
function geometric_mse_loss_function_QG3(output::AbstractArray, target::AbstractArray, pars::QG3_Physics_Parameters)
        w = pars.weights
        geom_cw_loss = sqrt.(sum(@.((output - target)^2 * w), dims=(1,2)) ./
                            sum(@.( target^2 * w), dims=(1,2)))
        return sum(geom_cw_loss) / length(geom_cw_loss)
end
function geometric_mse_loss_function_QG3(u, target, input, pars::QG3_Physics_Parameters)
    @views begin
        w = pars.weights
        u_pred = u(input)
        geom_cw_loss = sqrt.(sum(@.((u_pred - target)^2 * w), dims=(1,2)) ./
                            sum(@.( target^2 * w), dims=(1,2)))
        return sum(geom_cw_loss) / length(geom_cw_loss)    
    end
end

# Physics-informed residual loss
function physics_informed_loss_QG3(u, q_0, pars::QG3_Physics_Parameters)
    @views begin
        σ = pars.σ
        μ = pars.μ
        channelwise = !(isa(σ, Real) && isa(μ, Real))
        q_pred = ESM_PINO.denormalize_data(u(q_0), μ, σ, channelwise=channelwise)
        q_0_denorm = ESM_PINO.denormalize_data(q_0, μ, σ, channelwise=channelwise)

        ∂u_∂t = (q_pred .- q_0_denorm) ./ pars.dt
        ∂u_∂t = permutedims(∂u_∂t, (3, 1, 2, 4))

        q_pred_SH = QG3.transform_SH(permutedims(q_pred, (3, 1, 2, 4)), pars.ggsh)
        rhs_list = map(x -> QG3.QG3MM_gpu(x, (pars.qg3p, pars.S), (0,1)), eachslice(q_pred_SH; dims=4))
        rhs = reduce((acc, x) -> cat(acc, x; dims=4), rhs_list)
        rhs_grid = QG3.transform_grid(rhs, pars.shgg)
        residual = ∂u_∂t .- rhs_grid
        return mean(abs2, residual)    
    end
end


############################
# Factory: Loss Composer
############################

"""
    make_QG3_loss(pars::QG3_Physics_Parameters;
                  α=0.5f0,
                  use_physics::Bool=true,
                  geometric::Bool=false)

Create a composite QG3 loss function suitable for Lux training.
Returns a callable `(model, ps, st, (input, target)) -> (loss, st, metrics)`.
"""
function make_QG3_loss(pars::QG3_Physics_Parameters;
                       α::Float32=0.5f0,
                       use_physics::Bool=true,
                       geometric::Bool=true)

    # choose data loss variant
    data_loss_fun = geometric ?
        ((u, y, x) -> geometric_mse_loss_function_QG3(u, y, x, pars)) :
        mse_loss_function_QG3

    # choose physics-informed component
    physics_loss_fun = use_physics ?
        ((u, x) -> physics_informed_loss_QG3(u, x, pars)) :
        ((u, x) -> 0f0)

    # build Lux-compatible loss function
    function QG3_loss_function(model::Lux.AbstractLuxLayer,
                               ps::NamedTuple,
                               st::NamedTuple,
                               (input, target)::Tuple{AbstractArray, AbstractArray})
        u_net = Lux.StatefulLuxLayer{true}(model, ps, st)

        data_loss = data_loss_fun(u_net, target, input)
        physics_loss = physics_loss_fun(u_net, input)
        total_loss = α * data_loss + (1 - α) * physics_loss

        return (total_loss,
                (st),
                (; data_loss, physics_loss))
    end

    return QG3_loss_function
end


############################
# Optional: Autoregressive Extension
############################
"""
    make_autoregressive_loss(QG3_loss::Function; steps::Int, sequential::Bool=true)

Create an autoregressive loss function that rolls out predictions over `steps`
and accumulates the loss defined by `QG3_loss`.

# Arguments
- `QG3_loss`: A loss function of the form `(model, ps, st, (input, target)) -> (loss, st, details)`
- `steps`: Number of autoregressive rollout steps
- `sequential`: If true, predictions are fed sequentially (standard autoregressive); if false,
  all predictions are computed and compared in batch for efficiency.

# Returns
- A loss function `(model, ps, st, (u_t1, targets)) -> (loss, st, details)`
"""
function make_autoregressive_loss(QG3_loss::Function; steps::Int, sequential::Bool=false)
    function autoregressive_loss(model::Lux.AbstractLuxLayer, ps::NamedTuple, st::NamedTuple,
                                 (u_t1, targets)::Tuple{AbstractArray, AbstractArray})
        u_net = Lux.StatefulLuxLayer{true}(model, ps, st)
        total_loss = 0f0
        current_input = u_t1

        if sequential
            # Sequential rollout (standard)
            for step in 1:steps
                current_output = u_net(current_input)
                step_loss, _, _ = QG3_loss(model, ps, st, (current_input, targets[:,:,:,step,:]))
                total_loss += step_loss
                current_input = current_output
            end
        total_loss /= steps
        else
            preds = (u_t1,) #rewrite for case where you have different in_channels and out_channels
            for step in 2:steps
                next_pred = u_net(preds[end])
                preds = (preds..., next_pred) 
            end
            
            preds_stack = permutedims(cat(preds..., dims=5),(1,2,3,5,4))
            
            # target and preds_stack should have shape (Nx, Ny, Nz, steps, Nb)
            step_losses = map(step -> begin
                l, _, _ = QG3_loss(model, ps, st, (preds_stack[:,:,:,step,:], targets[:,:,:,step,:]))
                l
            end, 1:steps)
            total_loss = mean(step_losses)
        end

        return (total_loss, (st), (; total_loss))
    end
    return autoregressive_loss
end

