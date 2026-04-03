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
    geom_weights = QG3.togpu(reshape(cos.(pars.lats), :, 1, 1, 1))
    quad_weights = QG3.togpu(reshape(QG3.compute_GaussWeights(pars), :, 1, 1, 1))
    weights = geom_weights .* quad_weights
    S = CUDA.@allowscalar QG3.reorder_SH_gpu(QG3.zeros_SH(pars), pars)
    dt = σ = 1
    μ = 0
    QG3.gpuon()
    return QG3_Physics_Parameters(dt, qg3p, S, ggsh, shgg, μ, σ, weights)
end 

function QG3_Physics_Parameters(pars::QG3.QG3ModelParameters; batch_size=1, gpu::Bool=true)
    if !gpu
        QG3.gpuoff()
    end
    ggsh = QG3.GaussianGridtoSHTransform(pars, N_batch=batch_size)
    shgg = QG3.SHtoGaussianGridTransform(pars, N_batch=batch_size)
    qg3p = CUDA.@allowscalar QG3Model(pars)
    geom_weights = QG3.togpu(reshape(cos.(pars.lats), :, 1, 1, 1))
    quad_weights = QG3.togpu(reshape(QG3.compute_GaussWeights(pars), :, 1, 1, 1))
    weights = geom_weights .* quad_weights
    S = CUDA.@allowscalar QG3.reorder_SH_gpu(QG3.zeros_SH(pars),pars)
    dt = σ = 1
    μ = 0
    QG3.gpuon()
    return QG3_Physics_Parameters(dt, qg3p, S, ggsh, shgg, μ, σ, weights) 
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

    geom_weights = QG3.togpu(reshape(cos.(qg3p.p.lats), :, 1, 1, 1))
    quad_weights = QG3.togpu(reshape(QG3.compute_GaussWeights(qg3p.p), :, 1, 1, 1))
    weights = geom_weights .* quad_weights
    QG3.gpuon()
    if gpu
        weights = QG3.togpu(weights)
        pars = qg3p.p
        qg3p = CUDA.@allowscalar QG3Model(pars)
    end
    return QG3_Physics_Parameters(dt, qg3p, S, ggsh, shgg, μ, σ, weights)
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
    geom_weights = QG3.togpu(reshape(cos.(qg3p.p.lats), :, 1, 1, 1))
    quad_weights = QG3.togpu(reshape(QG3.compute_GaussWeights(qg3p.p), :, 1, 1, 1))
    weights = geom_weights .* quad_weights
    QG3.gpuon()
    if gpu
        weights = QG3.togpu(weights)
        pars = qg3p.p
        qg3p = CUDA.@allowscalar QG3Model(pars)
    end
    return QG3_Physics_Parameters(dt, qg3p, S, ggsh, shgg, μ, σ, weights)
end

# Base Loss Components


# Simple MSE loss (baseline)
mse_loss_function_QG3(u, target, input) =
    Lux.MSELoss()(u(input), target)

# Geometrically weighted MSE losses
function geometric_mse_loss_function_QG3(output::AbstractArray, target::AbstractArray, pars::QG3_Physics_Parameters)
    @views begin
        w = pars.weights
        geom_cw_loss = sqrt.((sum(@.((output - target)^2 * w), dims=(1,2))) ./
                            sum(@.( target^2 * w), dims=(1,2))) 
        return sum(geom_cw_loss) / length(geom_cw_loss)   
    end      
end
function geometric_mse_loss_function_QG3(u, target::AbstractArray{T,4}, input::AbstractArray{T,4}, pars::QG3_Physics_Parameters) where T
    @views begin
        w = pars.weights
        u_pred = u(input)
        geom_cw_loss = sqrt.((sum(@.((u_pred - target)^2 * w), dims=(1,2))) ./
                            sum(@.( target^2 * w), dims=(1,2)))
        return sum(geom_cw_loss) / length(geom_cw_loss)    
    end
end
function angular_power_spectrum_batch_gpu(A::AbstractArray{T,4}, k_indices) where T
    # A is (n_dim1, n_lat, n_lon, n_batch)
    n_dim1, L, _, n_batch = size(A)
    
    # Map over k_indices to compute power spectrum for each
    ps_arrays = map(k_indices) do k_idx
        l = k_idx - 1
        fac = T(1 / (2 * l + 1))
        
        # Sum over all m values
        ps = mapreduce(+, -l:l) do m
            il = l + 1 - abs(m)
            im = m < 0 ? 2*abs(m) : 2*m + 1
            A[:, il, im, :] .^ 2
        end
        
        fac .* ps  # Shape: (n_dim1, n_batch)
    end
    
    # Stack into 3D array and reshape
    # Each element in ps_arrays is (n_dim1, n_batch)
    stacked = cat([reshape(ps, n_dim1, n_batch, 1) for ps in ps_arrays]..., dims=3)
    
    # Permute to (n_k, n_dim1, n_batch) and reshape to (n_k, n_samples)
    stacked = permutedims(stacked, (3, 1, 2))
    return reshape(stacked, length(k_indices), n_dim1 * n_batch)
end

function spectral_loss_function_QG3(u, target::AbstractArray{T,4}, input::AbstractArray{T,4}, pars::QG3_Physics_Parameters; k_min::Int=1, k_max::Int=500) where T
    @assert k_min < k_max "k_min must be smaller than k_max"
    u_pred = u(input)
    u_pred_sh = QG3.transform_SH(permutedims(u_pred, (3, 1, 2, 4)), pars.ggsh)
    target_sh = QG3.transform_SH(permutedims(target, (3, 1, 2, 4)), pars.ggsh)
    
    if k_max > size(u_pred_sh,2)
        k_indices = 1:k_min
    else
        k_indices = vcat(1:k_min, k_max:size(u_pred_sh, 2))
    end
    
    # Compute on GPU
    u_pred_ps = angular_power_spectrum_batch_gpu(u_pred_sh, k_indices)
    target_ps = angular_power_spectrum_batch_gpu(target_sh, k_indices)
    
    spectral_loss = sqrt.(sum((u_pred_ps .- target_ps).^2, dims=1))
    return sum(spectral_loss) / length(spectral_loss)
end

# Physics-informed residual loss
function physics_informed_loss_QG3(u, q_0::AbstractArray{T,4}, pars::QG3_Physics_Parameters; bc=true, state_eq=true) where T
    @views begin
        σ = pars.σ
        μ = pars.μ
        q_pred = u(q_0)
        if size(q_pred,3) == 3
            if bc
                length_scale = 2π .* pars.qg3p.p.distance_unit .* pars.qg3p.cosϕ ./ pars.qg3p.p.N_lons
                bc_term = mean(abs2, (q_pred[:,end,:,:] .- q_pred[:,1,:,:]) ./ length_scale)
            end
            q_pred_denorm = ESM_PINO.denormalize_data(q_pred, μ, σ)
            q_0_denorm = ESM_PINO.denormalize_data(q_0, μ, σ)

            ∂u_∂t = (q_pred_denorm .- q_0_denorm) ./ pars.dt
            ∂u_∂t = permutedims(∂u_∂t, (3, 1, 2, 4))

            q_0_SH = QG3.transform_SH(permutedims(q_0_denorm, (3, 1, 2, 4)), pars.ggsh)
            rhs_list = map(x -> QG3.QG3MM_gpu(x, (pars.qg3p, pars.S), (0,1)), eachslice(q_0_SH; dims=4)) #maybe use q_0 SH?
            rhs = reduce((acc, x) -> cat(acc, x; dims=4), rhs_list)
            rhs_grid = QG3.transform_grid(rhs, pars.shgg)
            residual = ∂u_∂t .- rhs_grid
            if bc 
                return mean(abs2, residual) + bc_term
            else
                return mean(abs2, residual)
            end
        elseif size(q_pred, 3) == 6
            if bc
                bc_term = mean(abs2, q_pred[:,end,:,:] .- q_pred[:,1,:,:])
            end
            q_pred_denorm = ESM_PINO.denormalize_data(q_pred, μ, σ)
            if state_eq 
                ψ_pred_denorm = QG3.transform_grid(QG3.qprimetoψ(pars.qg3p, QG3.transform_SH(permutedims(q_pred_denorm[:,:,1:3,:], (3,1,2,4)), pars.ggsh)), pars.shgg)
                state_eq_term = mean(abs2, permutedims(ψ_pred_denorm,(2,3,1,4)) .- q_pred_denorm[:,:,4:6,:])
            end
            q_0_denorm = ESM_PINO.denormalize_data(q_0, μ, σ)

            ∂q_∂t = (q_pred_denorm .- q_0_denorm)[:,:,1:3,:] ./ pars.dt
            ∂q_∂t = permutedims(∂q_∂t, (3, 1, 2, 4))
            q_0_new = q_0_denorm[:,:,1:3,:]
            q_0_SH = QG3.transform_SH(permutedims(q_0_new, (3, 1, 2, 4)), pars.ggsh)
            rhs_list = map(x -> QG3.QG3MM_gpu(x, (pars.qg3p, pars.S), (0,1)), eachslice(q_0_SH; dims=4)) #maybe use q_0 SH?
            rhs = reduce((acc, x) -> cat(acc, x; dims=4), rhs_list)
            rhs_grid = QG3.transform_grid(rhs, pars.shgg)
            residual = ∂q_∂t .- rhs_grid
            return mean(abs2, residual) + (bc ? bc_term : 0f0) + (state_eq ? state_eq_term : 0f0)
        end
    end   
end
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
                       β::Float32=0.3f0,
                       use_physics::Bool=false,
                       geometric::Bool=true,
                       spectral::Bool=false)

    # choose data loss variant
    data_loss_fun = geometric ?
        ((u, y, x) -> geometric_mse_loss_function_QG3(u, y, x, pars)) :
        mse_loss_function_QG3

    # choose physics-informed component
    physics_loss_fun = use_physics ?
        ((u, x) -> physics_informed_loss_QG3(u, x, pars)) :
        ((u, x) -> 0f0)
    spectral_loss_fun = spectral ?
        ((u,y,x) -> spectral_loss_function_QG3(u,y,x,pars)) :
        ((u,y,x) -> 0f0) 
    # build Lux-compatible loss function
    function QG3_loss_function(model::Lux.AbstractLuxLayer,
                               ps::NamedTuple,
                               st::NamedTuple,
                               (input, target)::Tuple{AbstractArray, AbstractArray})
        u_net = Lux.StatefulLuxLayer{true}(model, ps, st)

        data_loss = data_loss_fun(u_net, target, input)
        spectral_loss = spectral_loss_fun(u_net, target, input)
        physics_loss = physics_loss_fun(u_net, input)
        total_loss = α * data_loss + β * spectral_loss + (1 - α - β) * physics_loss

        return (total_loss,
                (st),
                (; data_loss, physics_loss, spectral_loss))
    end

    return QG3_loss_function
end

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
function make_autoregressive_loss(QG3_loss::Function; steps::Int=2)
    function autoregressive_loss(model::Lux.AbstractLuxLayer, ps::NamedTuple, st::NamedTuple,
                                 (u_t1, targets)::Tuple{AbstractArray, AbstractArray})
        u_net = Lux.StatefulLuxLayer{true}(model, ps, st)
       
        preds = (u_t1,) #rewrite for case where you have different in_channels and out_channels
        for step in 2:steps
            next_pred = u_net(preds[end])
            preds = (preds..., next_pred) 
        end
        preds_stack = permutedims(cat(preds..., dims=5),(1,2,3,5,4))
        @views begin
            # target and preds_stack should have shape (Nx, Ny, Nz, steps, Nb)
            step_losses = map(step -> begin
            l, _, s = QG3_loss(model, ps, st, (preds_stack[:,:,:,step,:], targets[:,:,:,step,:]))
            l, s.data_loss, s.physics_loss, s.spectral_loss
            end, 1:steps)
            total_losses, data_losses, physics_losses, spectral_losses  = [collect(x) for x in zip(step_losses...)] 
            total_loss= mean(total_losses)
            physics_loss = mean(physics_losses)
            data_loss = mean(data_losses)
            spectral_loss = mean(spectral_losses)
            return (total_loss, (st), (; data_loss, physics_loss, spectral_loss))
        end
    end
    return autoregressive_loss
end

function autoregressive_loss(trained_u, x_batch::AbstractArray, target_batch::AbstractArray, pars::QG3_Physics_Parameters; n_steps::Int=2, geometric::Bool=true)
    preds = (x_batch,) #rewrite for case where you have different in_channels and out_channels
    for step in 2:n_steps
        next_pred = trained_u(preds[end])
        preds = (preds..., next_pred) 
    end
    preds_stack = permutedims(cat(preds..., dims=5),(1,2,3,5,4))
    if geometric
        QG3_loss = (u,y,x) -> geometric_mse_loss_function_QG3(u, y, x, pars)
    else
        QG3_loss = mse_loss_function_QG3
    end
    @views begin
        # target and preds_stack should have shape (Nx, Ny, Nz, steps, Nb)
        step_losses = map(step -> begin
            QG3_loss(trained_u, target_batch[:,:,:,step,:], preds_stack[:,:,:,step,:])
        end, 1:n_steps)
        return mean(step_losses)
    end
    
end

