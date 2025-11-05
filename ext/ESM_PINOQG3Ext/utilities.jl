"""
$(TYPEDSIGNATURES)
Construct a new `SFNO` model that replicates the architecture and parameters of an existing
`model`, but adapts them to a new discretization (`qg3ppars`) and batch size.  
This function preserves all spectral modes, channels, and hyperparameters while adjusting
the internal transform plans to match the new grid configuration.

# Arguments
- `model::SFNO`: Source SFNO model whose architecture and parameters will be cloned.
- `qg3ppars`: Target problem parameters (e.g., grid, spectral resolution).

# Keywords
- `batch_size::Int`: Optional new batch size. Defaults to the batch size inferred from
  `model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.plan.ggsh.FT_4d.plan.input_size[4]`.

# Returns
- `SFNO`: A new model instance with the same architecture as `model`, configured for the
  target discretization and batch size.

# Example
```julia
# Original model (batch size = 32)
model = SFNO(orig_pars; batch_size=32, ...)

# Transfer model to a finer grid and larger batch
new_model = transfer_SFNO_model(model, new_pars; batch_size=64)

# Forward pass with transferred weights
ŷ = new_model(x, ps, st)
```
"""
function transfer_SFNO_model(model::ESM_PINO.SFNO, qg3ppars::QG3ModelParameters; 
    batch_size::Int=model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.plan.ggsh.FT_4d.plan.input_size[4]
)
    superres_model = SFNO(qg3ppars,
        batch_size = batch_size,
        modes = model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.modes,
        in_channels = model.lifting.layers.layer_1.in_chs,
        out_channels = model.projection.layers.layer_2.out_chs, #watch out as you might have more than 2 layers
        hidden_channels = model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.hidden_channels,
        n_layers = length(model.sfno_blocks.layers),
        lifting_channel_ratio=model.lifting_channel_ratio,
        projection_channel_ratio=model.projection_channel_ratio,
        channel_mlp_expansion=model.sfno_blocks.layers.layer_1.channel_mlp.expansion_factor,
        activation = model.sfno_blocks.layers.layer_1.spherical_kernel.activation,
        positional_embedding = model.embedding == NoOpLayer() ? "no_grid" : "grid",
        gpu = Base.unwrap_unionall(typeof(model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.ggsh)).parameters[end],
        zsk = model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.zsk,
        inner_skip = model.sfno_blocks.layers.layer_1.skip,
        outer_skip = model.outer_skip,
        use_norm= model.sfno_blocks.layer.layer_1.norm == NoOpLayer() ? false : true,
        downsampling_factor= model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.plan.ggsh.FT_4d.plan.input_size[2] ÷ model.sfno_blocks.layers.layer_2.spherical_kernel.spherical_conv.plan.ggsh.FT_4d.plan.input_size[2]
        )
    return superres_model
end
"""
$(TYPEDSIGNATURES)

Helper function to hook the constructor for QG3ModelParameters using a Gaussian grid.
Generates latitude/longitude points and initializes empty topography and land/sea mask.
Used mainly to handle SH transforms.

# Arguments

-`L::Int`: Spectral truncation level (maximum degree).

- `n_lat::Int`: Number of Gaussian latitudes.

# Keywords

- `n_lon::Int=2*n_lat`: Number of longitudes (default: twice the latitude count).

- `iters::Int=100`: Maximum number of iterations for Gaussian grid convergence.

- `tol::Real=1e-8`: Convergence tolerance.

- `NF::Type{<:AbstractFloat}=Float32`: Number format for outputs.

# Returns

- `QG3ModelParameters`: Model parameters including grid coordinates, topography (h),
and land-sea mask (LS).

# Example
```julia
pars = qg3pars_constructor_helper(42, 64)
```
"""

function qg3pars_constructor_helper(L::Int, n_lat::Int; n_lon::Int=2*n_lat, iters::Int=100, tol::Real=1e-8,NF::Type{<:AbstractFloat}=Float32)
    lats, lons  = (ESM_PINO.gaussian_grid(n_lat; n_lon=n_lon, iters=iters, tol=tol))
    lats, lons = NF.(lats), NF.(lons)
    LS = h = zeros(NF, n_lat, n_lon)
    QG3ModelParameters(L, lats, lons, LS, h)
end
"""
$(TYPEDSIGNATURES)
Precompute a symmetric remapping plan between index ranges [-l..l] and [-c..c+1).
This is used to ensure correct ordering and slicing of spectral coefficients on gpu

# Arguments

-`l::Integer`: Original spectral truncation (maximum mode index).

-`c::Integer`: half-length of gpu-array rows.

-`T::Type{<:Number}=Float32`: Element type for masks and indices.

# Keywords

- `gpu::Bool=true`: Whether to allocate the mask on the GPU (CuArray).

# Returns

NamedTuple with:

- `remap_fixed::Vector{Int}` — Mapped indices (0 replaced with 1).

- `mask::AbstractArray` — Binary mask (CPU or GPU) for valid indices.

- `new_indices::Vector{Int}` — Target symmetric indices.

# Example
```julia
plan = remap_plan(42, 63; gpu=false)
```
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
$(TYPEDSIGNATURES)

Apply a precomputed symmetric remapping plan along a given array dimension.
Works on both CPU and GPU arrays and is compatible with Zygote for AD.

# Arguments

-`A::AbstractArray`: Input tensor to be remapped.

-`plan`: Remapping structure returned by remap_plan.

# Keywords

-`dim::Integer=3`: Dimension along which to apply the remap.

-`fill_value`: Optional value for indices outside the valid range. Defaults to zero(eltype(A)).

# Returns

- `(C, new_indices)`: Tuple with the remapped array and the list of new indices.

# Example
```julia
plan = remap_plan(42, 63)
A_new, idx = remap_symmetric_dim(A, plan; dim=3)
```
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
"""
    $(TYPEDSIGNATURES)

Train an `SFNO` model with the possibility using a combined data-driven (simple MSE or geometrically weighted)
and physics-informed loss. This function initializes the model, optimizer, and training
loop, and performs iterative optimization of the model parameters.  

Both standard data loss and an optional physics-informed term are tracked during training.

# Arguments
- `x::AbstractArray`: Input training data tensor.
- `target::AbstractArray`: Target (ground truth) data tensor.
- `pars::QG3ModelParameters`: Model configuration including grid and spectral parameters.

# Keywords
- `seed::Int=0`: Random seed for reproducibility.
- `maxiters::Int=20`: Number of training iterations.
- `batchsize::Int=256`: Mini-batch size for training.
- `modes::Int=pars.L`: Spectral truncation level.
- `in_channels::Int=3`: Number of input channels.
- `out_channels::Int=3`: Number of output channels.
- `hidden_channels::Int=256`: Width of the hidden feature layers.
- `n_layers::Int=4`: Number of SFNO layers.
- `lifting_channel_ratio::Int=2`: Ratio of lifting layer expansion.
- `projection_channel_ratio::Int=2`: Ratio of projection layer contraction.
- `channel_mlp_expansion::Number=2.0`: Expansion factor in channel MLP blocks.
- `activation`: Activation function used in SFNO blocks (default: `NNlib.gelu`).
- `positional_embedding::AbstractString="grid"`: Type of positional embedding (`"grid"` or `"no_grid"`).
- `inner_skip::Bool=true`: Whether to enable residual connections inside SFNO blocks.
- `outer_skip::Bool=true`: Whether to enable skip connections between lifting output and projection input.
- `zsk::Bool=false`: Use zonally symmetric kernel formulation if `true`.
- `use_norm::Bool=false`: Apply normalization layers inside SFNO blocks.
- `downsampling_factor::Int=2`: Ratio of downsampling between layers.
- `lr_0::Float64=1e-3`: Initial learning rate for the optimizer.
- `parameters::QG3_Physics_Parameters=QG3_Physics_Parameters(pars, batch_size=batchsize)`: Physical parameters used in the QG3 loss.
- `use_physics::Bool=true`: Whether to include the physics-informed loss component.
- `geometric::Bool=true`: Use geometrically weighted formulation for the data loss.
- `α::Float32=0.7f0`: Weighting factor between physics loss and data loss.

# Returns
- `StatefulLuxLayer{true}`: Trained SFNO model containing learned parameters and internal state.

# Example
```julia
# Initialize parameters and data
pars = qg3pars_constructor_helper(42, 64)
x, y = generate_training_data(pars)

# Train SFNO model
trained_model = train_model(x, y, pars; maxiters=100, batchsize=128, lr_0=5e-4)

# Perform inference
pred = trained_model(x)
```
"""
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
    lr_0::Float64=1e-3, 
    parameters::QG3_Physics_Parameters=QG3_Physics_Parameters(pars, batch_size=batchsize),
    use_physics=true, 
    geometric=true,
    α=0.7f0)
    
    rng = Random.default_rng(seed)
    
    dataloader = DataLoader((x, target); batchsize=batchsize, shuffle=true) |> gpu_device()
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
        use_norm=use_norm,
        )

    ps, st = Lux.setup(rng, sfno) |> gpu_device()
    @assert st isa NamedTuple "Model state should be a NamedTuple"
    # Rest of training setup remains similar
    
    opt = Optimisers.ADAM(lr_0)
    lr = i -> CosAnneal(lr_0, typeof(lr_0)(0), maxiters)(i)
    train_state = Training.TrainState(sfno, ps, st, opt)
    
    if !isnothing(parameters)
        par_train = QG3_Physics_Parameters(
                parameters.dt,
                parameters.qg3p,
                parameters.S,
                QG3.GaussianGridtoSHTransform(pars, N_batch=batchsize),
                QG3.SHtoGaussianGridTransform(pars, N_batch=batchsize),
                parameters.μ,
                parameters.σ
        )
    else
        par_train = nothing
    end
    QG3_loss = make_QG3_loss(par_train;geometric=geometric, use_physics=use_physics, α=α)
    total_loss_tracker, physics_loss_tracker, data_loss_tracker = ntuple(_ -> Lag(Float32, 32), 3)

    iter = 1
    for (x, target_data) in Iterators.cycle(dataloader)
        Optimisers.adjust!(train_state, lr(iter))
        _, loss, stats, train_state = Training.single_train_step!(AutoZygote(), QG3_loss, (x, target_data), train_state)
        
        fit!(total_loss_tracker, Float32(loss))
        fit!(physics_loss_tracker, Float32(stats.physics_loss))
        fit!(data_loss_tracker, Float32(stats.data_loss))

        mean_loss = mean(OnlineStats.value(total_loss_tracker))
        mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
        mean_data_loss = mean(OnlineStats.value(data_loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))
        
        if iter % 1 == 0 || iter == maxiters
            @printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f \
                 (%.9f) \t Data Loss: %.9f (%.9f)\n" iter maxiters loss mean_loss stats.physics_loss mean_physics_loss stats.data_loss mean_data_loss
            #@printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \n" iter maxiters loss mean_loss
                        
            GC.gc()
        end
        
        iter += 1
        
        if iter > maxiters
            break
        end
    end
    return StatefulLuxLayer{true}(sfno, train_state.parameters, train_state.states)
end
"""
    $(TYPEDSIGNATURES)

Fine-tune a pretrained `SFNO` model using an autoregressive (AR) loss function.  
This procedure is typically applied after initial training to improve multi-step
forecast accuracy.

The function performs a short fine-tuning loop with autoregressive supervision,
optionally including a physics-informed loss component.

# Arguments
- `x::AbstractArray`: Input data tensor.
- `target::AbstractArray`: Target data tensor with shape `(lat, lon, channels, batch, time)`.
- `model`: Pretrained `SFNO` model to be fine-tuned.
- `ps::NamedTuple`: Model parameters (from previous training).
- `st::NamedTuple`: Model internal state.

# Keywords
- `n_steps::Int=2`: Number of autoregressive steps in the loss function.
- `maxiters::Int=5`: Maximum number of fine-tuning iterations.
- `lr_0::Float64=1e-5`: Learning rate for fine-tuning.
- `parameters::QG3_Physics_Parameters=QG3_Physics_Parameters()`: Physical parameters used in the loss.
- `use_physics::Bool=true`: Include physics-informed component in the loss if `true`.
- `geometric::Bool=true`: Use geometric formulation of the physics loss.
- `α::Float32=0.7f0`: Weighting factor between physics and data loss terms.

# Returns
- `StatefulLuxLayer{true}`: Fine-tuned model instance with updated parameters and states.

# Notes
- The target tensor must have five dimensions, with the number of autoregressive steps as the fifth dimension.
- The time dimension (`size(target, 5)`) must match the number of autoregressive steps (`n_steps`).

# Example
```julia
# Fine-tune a pretrained SFNO model for multi-step forecasting
ft_model = fine_tuning(x_val, y_val, pretrained_model, ps, st;
                       n_steps=3, maxiters=10, lr_0=1e-5)

# Evaluate fine-tuned model
pred = ft_model(x_val)
```
"""
function fine_tuning(x::AbstractArray, 
    target::AbstractArray, 
    model,
    ps::NamedTuple,
    st::NamedTuple;
    n_steps::Int=2,
    maxiters::Int=5, 
    lr_0::Float64=1e-5, 
    parameters::QG3_Physics_Parameters=QG3_Physics_Parameters(),
    use_physics=true,
    geometric=true, 
    α=0.7f0)
    batchsize = #retrieve from model
        model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.plan.ggsh.FT_4d.plan.input_size[4]
    @assert length(size(target)) == 5 "Target data must have 5 dimensions (lat, lon, channel, batch, time)"
    target = permutedims(target, (1,2,3,5,4)) # make sure target is (lat, lon, channel, time, batch)
    @assert size(target, 4) == n_steps "Target data must be pre-computed to have the same 5th dimension size as the number of autoregressive steps computed in the loss"
    dataloader = DataLoader((x, target); batchsize=batchsize, shuffle=true) |> gpu_device()

    ps = ps |> gpu_device()
    st = st |> gpu_device()
    
    # Rest of training setup remains similar
    
    opt = Optimisers.ADAM(lr_0)
    train_state = Training.TrainState(model, ps, st, opt)
    
    if !isnothing(parameters)
        par_train = QG3_Physics_Parameters(
                parameters.dt,
                parameters.qg3p,
                parameters.S,
                QG3.GaussianGridtoSHTransform(QG3.tocpu(parameters.qg3p.p), N_batch=batchsize),
                QG3.SHtoGaussianGridTransform(QG3.tocpu(parameters.qg3p.p), N_batch=batchsize),
                parameters.μ,
                parameters.σ
        )
    else
        par_train = nothing
    end
    
    QG3_loss = make_QG3_loss(par_train, geometric=geometric, use_physics=use_physics, α=α)
    AR_loss = make_autoregressive_loss(QG3_loss; steps=n_steps, sequential=false)
    total_loss_tracker = ntuple(_ -> Lag(Float32, 32), 1)[1]

    iter = 1
    for (x, target_data) in Iterators.cycle(dataloader)
        _, loss, stats, train_state = Training.single_train_step!(AutoZygote(), AR_loss, (x, target_data), train_state)
        
        fit!(total_loss_tracker, Float32(loss))
        #fit!(physics_loss_tracker, Float32(stats.physics_loss))
        #fit!(data_loss_tracker, Float32(stats.data_loss))

        mean_loss = mean(OnlineStats.value(total_loss_tracker))
        #mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
        #mean_data_loss = mean(OnlineStats.value(data_loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))
        
        if iter % 1 == 0 || iter == maxiters
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
    return StatefulLuxLayer{true}(model, train_state.parameters, train_state.states)
end
"""
    $(TYPEDSIGNATURES)

Convert a 4D tensor of sequential data `(lat, lon, channels, batch)` into a 5D tensor
suitable for autoregressive training or evaluation.  
The function constructs overlapping sequences along the batch dimension, each containing
`time_steps` consecutive snapshots.

# Arguments
- `data::AbstractArray{T,4}`: Input data tensor with dimensions `(lat, lon, channels, batch)`.
- `time_steps::Int`: Number of consecutive time steps to include in each sequence.

# Returns
- `AbstractArray{T,5}`: A 5D tensor of shape `(lat, lon, channels, time_steps, n_sequences)`,
  where `n_sequences = batch - time_steps + 1`.

# Notes
- The resulting array can be used as autoregressive training targets for multi-step prediction.
- Sequences are created by sliding a window of length `time_steps` along the batch axis.

# Example
```julia
# Input: 4D array with 10 time samples
data = rand(Float32, 64, 128, 3, 10)

# Stack into 5D sequences of 4 time steps each
seq_data = stack_time_steps(data, 4)

@assert size(seq_data) == (64, 128, 3, 4, 7)
```
"""
function stack_time_steps(data::AbstractArray{T,4}, time_steps::Int) where T
    lat, lon, channels, batch_size = size(data)
    n_sequences = batch_size - time_steps + 1
    
    # Create 5D array using comprehension
    result = cat(
        [data[:, :, :, i:i+time_steps-1] for i in 1:n_sequences]...,
        dims=5
    )
    
    # Permute dimensions to get the desired shape
    return permutedims(result, (1, 2, 3, 5, 4))
end
