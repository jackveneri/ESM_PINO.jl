"""
    transfer_SFNO_model(model, qg3ppars; batch_size=default_batch_size)

Construct a new SFNO model with the same architecture/parameters as `model`, but adapted to new 
discretization (`qg3ppars`) and batch size. Preserves spectral modes, channels, and other 
hyperparameters from the original model.

# Arguments
- `model::SFNO`: Source model whose architecture/hyperparameters will be copied.
- `qg3ppars`: New problem parameters (e.g., grid resolution) for the target model.

# Keywords
- `batch_size::Int`: (optional) Desired batch size. Defaults to the original model's batch size 
  (extracted from `model.sfno_blocks...FT_4d.plan.input_size[4]`).

# Returns
- `superres_model::SFNO`: New model configured for the target specifications.

# Example
```julia
# Original model (batch_size=32)
model = SFNO(orig_pars, batch_size=32, ...)

# Train model
ps, st = ...

# Adapted model (batch_size=64, new grid params)
new_model = transfer_SFNO_model(model, new_pars; batch_size=64)

# Perform inference using learned parameters
output = new_model(x, ps, st)
```
"""
#rewrite this
function transfer_SFNO_model(model, qg3ppars; 
    batch_size=model.sfno_blocks.model.spherical_kernel.spherical_conv.ggsh.FT_4d.plan.input_size[4])
    superres_model = SFNO(qg3ppars,
        batch_size = batch_size,
        modes = model.sfno_blocks.model.spherical_kernel.spherical_conv.modes,
        in_channels = model.lifting.layers.layer_1.in_chs,
        out_channels = model.projection.layers.layer_2.out_chs, #watch out as you might have more than 2 layers
        hidden_channels = model.sfno_blocks.model.spherical_kernel.spherical_conv.hidden_channels,
        n_layers = model.sfno_blocks.nrepeats,
        lifting_channel_ratio=model.lifting_channel_ratio,
        projection_channel_ratio=model.projection_channel_ratio,
        channel_mlp_expansion=model.sfno_blocks.model.channel_mlp.expansion_factor,
        activation = model.sfno_blocks.model.spherical_kernel.activation,
        positional_embedding = model.embedding == NoOpLayer() ? "no_grid" : "grid",
        gpu = Base.unwrap_unionall(typeof(model.sfno_blocks.model.spherical_kernel.spherical_conv.ggsh)).parameters[end],
        zsk = model.sfno_blocks.model.spherical_kernel.spherical_conv.zsk,
        inner_skip = model.sfno_blocks.model.skip,
        oouter_skip = model.outer_skip
        )
    return superres_model
end

function qg3pars_constructor_helper(L::Int, n_lat::Int; n_lon::Int=2*n_lat, iters::Int=100, tol::Real=1e-8,NF::Type{<:AbstractFloat}=Float32)
    lats, lons  = (ESM_PINO.gaussian_grid(n_lat; n_lon=n_lon, iters=iters, tol=tol))
    lats, lons = NF.(lats), NF.(lons)
    LS = h = zeros(NF, n_lat, n_lon)
    QG3ModelParameters(L, lats, lons, LS, h)
end

"""
Precompute the symmetric remapping structure for indices [-l..l] -> [-c..c+1).
Returns a NamedTuple with:
    remap_fixed :: Vector{Int}
    mask        :: CuArray or Array{eltype}
    new_indices :: Vector{Int}
"""
function remap_plan(l::Integer, c::Integer, T::Type{<:Number}=Float32; gpu::Bool=true)
    old_indices = vcat(0, collect(1:l), collect(-1:-1:-l))
    new_indices = vcat(0, collect(1:c), collect(-1:-1:-c+1))

    remap = [findfirst(==(idx), old_indices) === nothing ? 0 : findfirst(==(idx), old_indices)
             for idx in new_indices]
    remap_fixed = [r == 0 ? 1 : r for r in remap]

    mask_vec = T.(remap .!= 0)
    mask = reshape(mask_vec, (1, 1, length(mask_vec), 1))

    if gpu
        mask = cu(mask)
    end

    (; remap_fixed, mask, new_indices)
end


"""
Apply a precomputed remap plan along a given dimension (default: dim=3).
Works with CPU and GPU arrays and is Zygote-friendly.
"""
function remap_symmetric_dim(A::AbstractArray, plan; dim::Integer=3, fill_value=nothing)
    fill_value = isnothing(fill_value) ? zero(eltype(A)) : fill_value

    nd = ndims(A)
    @assert 1 ≤ dim ≤ nd "Invalid dimension"

    inds = ntuple(i -> :, nd)
    inds = Base.setindex(inds, plan.remap_fixed, dim)
    C = A[inds...]  # gather

    # broadcast mask
    mask = plan.mask
    if isa(A, CuArray) && !isa(mask, CuArray)
        mask = cu(mask)
    end

    if fill_value == zero(eltype(A))
        return C .* mask, plan.new_indices
    else
        return C .* mask .+ fill_value .* (1 .- mask), plan.new_indices
    end
end

#=
l, c = 2, 4
plan = remap_plan(l, c, Float32)
A = rand(Float32, 2, 2, 2l+1, 3)
B, idx = remap_symmetric_dim(A, plan)
@show size(B), idx

using Zygote

l, c = 2, 4
plan_gpu = remap_plan(l, c, Float32; device=:gpu)
A_gpu = cu(rand(Float32, 2, 2, 5, 3))

B_gpu, _ = remap_symmetric_dim(A_gpu, plan_gpu)
println(size(B_gpu))

# Test gradient
loss(A) = sum(remap_symmetric_dim(A, plan_gpu)[1])
gs = Zygote.gradient(loss, A_gpu)
println(size(gs[1]))  # matches A_gpu
=#

function train_model(
    x::AbstractArray, 
    target::AbstractArray, 
    pars::QG3ModelParameters;
    seed::Int=0,
    maxiters::Int=20, 
    batchsize::Int=256,
    modes::Int=pars.L,
    in_channels::Int=3,
    out_channels::Int=3,
    hidden_channels::Int=256,
    n_layers::Int=4,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=2.0,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid",
    inner_skip::Bool=true,
    outer_skip::Bool=true,
    zsk=false,
    use_norm::Bool=false,
    downsampling_factor::Int=2,
    lr_0::Float32=0.001f0, 
    parameters::Union{Nothing, QG3_Physics_Parameters}=nothing)
    
    rng = Random.default_rng(seed)
    
    dataloader = DataLoader((x, target); batchsize=batchsize, shuffle=true) |> gdev
    ggsh = QG3.GaussianGridtoSHTransform(pars, N_batch=batchsize)
    shgg = QG3.SHtoGaussianGridTransform(pars, N_batch=batchsize)
    # Create the model
    sfno = SFNO(
        ggsh,
        shgg,
        modes = modes,
        in_channels=in_channels, 
        out_channels=out_channels, 
        n_layers=n_layers,
        hidden_channels=hidden_channels, 
        positional_embedding=positional_embedding,
        lifting_channel_ratio=lifting_channel_ratio,
        projection_channel_ratio=projection_channel_ratio,
        channel_mlp_expansion=channel_mlp_expansion,
        activation=activation,
        inner_skip=inner_skip,
        outer_skip=outer_skip,
        zsk=zsk,
        downsampling_factor=downsampling_factor,
        use_norm=use_norm
    )

    ps, st = Lux.setup(rng, sfno) |> gdev
    
    # Rest of training setup remains similar
    
    opt = Optimisers.ADAM(lr_0)
    lr = i -> CosAnneal(lr_0, typeof(lr_0)(0), maxiters)(i)
    train_state = Training.TrainState(sfno, ps, st, opt)
    
    if !isnothing(parameters)
        par_train = ESM_PINOQG3.QG3_Physics_Parameters(
                parameters.dt,
                parameters.qg3p,
                parameters.S,
                QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=batchsize),
                QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=batchsize),
                parameters.μ,
                parameters.σ
        )
    else
        par_train = nothing
    end
    physics_loss = ESM_PINOQG3.create_QG3_physics_loss(par_train)
    loss_function = ESM_PINOQG3.select_QG3_loss_function(physics_loss)
    total_loss_tracker = ntuple(_ -> Lag(Float32, 32), 1)[1]

    iter = 1
    for (x, target_data) in Iterators.cycle(dataloader)
        Optimisers.adjust!(train_state, lr(iter))
        _, loss, stats, train_state = Training.single_train_step!(AutoZygote(), loss_function, (x, target_data), train_state)
        
        fit!(total_loss_tracker, Float32(loss))
        #fit!(physics_loss_tracker, Float32(stats.physics_loss))
        #fit!(data_loss_tracker, Float32(stats.data_loss))

        mean_loss = mean(OnlineStats.value(total_loss_tracker))
        #mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
        #mean_data_loss = mean(OnlineStats.value(data_loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))
        
        if iter % 5 == 0 || iter == maxiters
            #@printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f \
            #     (%.9f) \t Data Loss: %.9f (%.9f)\n" iter maxiters loss mean_loss stats.physics_loss mean_physics_loss stats.data_loss mean_data_loss
            @printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \n" iter maxiters loss mean_loss
                        
            GC.gc()
        end
        
        iter += 1
        
        if iter > maxiters
            break
        end
    end
    return StatefulLuxLayer{true}(sfno, train_state.parameters, train_state.states), loss_function
end

function fine_tuning(x::AbstractArray, 
    target::AbstractArray, 
    model,
    ps::NamedTuple,
    st::NamedTuple;
    n_steps::Int=2,
    maxiters::Int=5, 
    lr_0::Float32=0.00001f0, 
    parameters::Union{Nothing, QG3_Physics_Parameters}=nothing)
    batchsize = #retrieve from model
        Base.unwrap_unionall(typeof(model.sfno_blocks.model.spherical_kernel.spherical_conv.ggsh)).parameters[end].FT_4d.plan.input_size[4]
    dataloader = DataLoader((x, target); batchsize=batchsize, shuffle=true) |> gdev

    ps = ps |> gdev
    st = st |> gdev
    
    # Rest of training setup remains similar
    
    opt = Optimisers.ADAM(lr_0)
    train_state = Training.TrainState(sfno, ps, st, opt)
    
    if !isnothing(parameters)
        par_train = ESM_PINOQG3.QG3_Physics_Parameters(
                parameters.dt,
                parameters.qg3p,
                parameters.S,
                QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=batchsize),
                QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=batchsize),
                parameters.μ,
                parameters.σ
        )
    else
        par_train = nothing
    end
    physics_loss = ESM_PINOQG3.create_QG3_physics_loss(par_train)
    loss_function = ESM_PINOQG3.select_QG3_loss_function(physics_loss)
    autoregrssive_loss = ESM_PINOQG3.autoregressive_loss(n_steps) #check arguments here
    total_loss_tracker = ntuple(_ -> Lag(Float32, 32), 1)[1]

    iter = 1
    for (x, target_data) in Iterators.cycle(dataloader)
        Optimisers.adjust!(train_state, lr(iter))
        _, loss, stats, train_state = Training.single_train_step!(AutoZygote(), loss_function, (x, target_data), train_state)
        
        fit!(total_loss_tracker, Float32(loss))
        #fit!(physics_loss_tracker, Float32(stats.physics_loss))
        #fit!(data_loss_tracker, Float32(stats.data_loss))

        mean_loss = mean(OnlineStats.value(total_loss_tracker))
        #mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
        #mean_data_loss = mean(OnlineStats.value(data_loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))
        
        if iter % 5 == 0 || iter == maxiters
            #@printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f \
            #     (%.9f) \t Data Loss: %.9f (%.9f)\n" iter maxiters loss mean_loss stats.physics_loss mean_physics_loss stats.data_loss mean_data_loss
            @printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \n" iter maxiters loss mean_loss
                        
            GC.gc()
        end
        
        iter += 1
        
        if iter > maxiters
            break
        end
    end
    return StatefulLuxLayer{true}(sfno, train_state.parameters, train_state.states), loss_function
end