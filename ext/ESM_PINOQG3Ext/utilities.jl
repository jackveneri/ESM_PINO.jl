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
    projection_layers = model.projection.layers
    last_layer_index = length(projection_layers)  # or fieldcount(typeof(projection_layers))
    out_channels = projection_layers[last_layer_index].out_chs
    superres_model = SFNO(qg3ppars,
        batch_size = batch_size,
        modes = model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.modes,
        in_channels = model.lifting.layers.layer_1.in_chs,
        out_channels = out_channels,
        hidden_channels = model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.hidden_channels,
        n_layers = length(model.sfno_blocks.layers),
        lifting_channel_ratio=model.lifting_channel_ratio,
        projection_channel_ratio=model.projection_channel_ratio,
        channel_mlp_expansion=model.sfno_blocks.layers.layer_1.channel_mlp.expansion_factor,
        activation = model.sfno_blocks.layers.layer_1.activation,
        positional_embedding = model.embedding == NoOpLayer() ? "no_grid" : "grid",
        gpu = Base.unwrap_unionall(typeof(model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.plan.ggsh)).parameters[end],
        operator_type = model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.operator_type,
        inner_skip = model.sfno_blocks.layers.layer_1.skip,
        outer_skip = model.outer_skip,
        use_norm = model.sfno_blocks.layers.layer_1.spherical_kernel.norm == NoOpLayer() ? false : true,
        #downsampling_factor= model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.plan.ggsh.FT_4d.plan.input_size[2] ÷ model.sfno_blocks.layers.layer_2.spherical_kernel.spherical_conv.plan.ggsh.FT_4d.plan.input_size[2]
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

function qg3pars_constructor_helper(L::Int, qg3ppars::QG3.QG3ModelParameters; NF::Type{<:AbstractFloat}=Float32)
    _, lons  = (ESM_PINO.gaussian_grid(qg3ppars.N_lats; n_lon=qg3ppars.N_lons))
    lats = qg3ppars.lats
    lats, lons = NF.(lats), NF.(lons)
    LS = h = zeros(NF, qg3ppars.N_lats, qg3ppars.N_lons)
    QG3ModelParameters(L, lats, lons, LS, h)
end

"""
    remap_array_components(arr::AbstractArray{T,4}, l::Int, c::Int) where T

Remap a 4D array from ordering (0,-1,1,-2,2,...,-l,l) to (0,1,2,...,l,...,c,-1,-2,...,-l,...,-(c-1)).

# Arguments
- `arr`: 4D array of size (i, j, 2l+1, k) with third dimension in order: 0,-1,1,-2,2,...,-l,l
- `l`: Original maximum index (third dimension has 2l+1 elements)
- `c`: New maximum index (output third dimension will have 2c elements)

# Returns
- 4D array of size (i, j, 2c, k) with reordered and padded third dimension

# Examples
```julia
# CPU example
arr = randn(3, 4, 5, 2)  # l=2, so 2*2+1=5
result = remap_array_components(arr, 2, 4)  # c=4, output size (3,4,8,2)

# GPU example
arr_gpu = CuArray(randn(3, 4, 5, 2))
result_gpu = remap_array_components(arr_gpu, 2, 4)
```
"""
function remap_array_components(arr::AbstractArray{T,4}, l::Int, c::Int) where T
    @assert c > l "c must be greater than l"
    @assert size(arr, 3) == 2l + 1 "Third dimension must have size 2l+1"
    
    i, j, _, k = size(arr)
    
    # Create output array with zeros (compatible with GPU)
    out = similar(arr, i, j, 2c, k)
    fill!(out, zero(T))
    
    # Map from old ordering to new ordering
    # Old: 0, -1, 1, -2, 2, ..., -l, l (indices 1 to 2l+1)
    # New: 0, 1, 2, ..., l, ..., c, -1, -2, ..., -l, ..., -(c-1) (indices 1 to 2c)
    
    # Element 0: position 1 → position 1
    out[:, :, 1, :] = arr[:, :, 1, :]
    
    # Positive elements: 1,2,...,l at old positions 3,5,7,...,2l+1
    # go to new positions 2,3,...,l+1
    for m in 1:l
        old_idx = 2m + 1
        new_idx = m + 1
        out[:, :, new_idx, :] = arr[:, :, old_idx, :]
    end
    
    # Negative elements: -1,-2,...,-l at old positions 2,4,6,...,2l
    # go to new positions c+1, c+2, ..., c+l
    for m in 1:l
        old_idx = 2m
        new_idx = c + m + 1
        out[:, :, new_idx, :] = arr[:, :, old_idx, :]
    end
    
    # Positions l+2 to c and c+l+1 to 2c remain zero (padding)
    
    return out
end

# Define custom reverse-mode rule for Zygote compatibility
function Zygote.rrule(::typeof(remap_array_components), arr::AbstractArray{T,4}, l::Int, c::Int) where T
    result = remap_array_components(arr, l, c)
    
    function remap_pullback(Δ)
        # Allocate gradient for input array
        ∇arr = similar(arr)
        fill!(∇arr, zero(T))
        
        # Reverse the mapping
        # Element 0
        ∇arr[:, :, 1, :] = Δ[:, :, 1, :]
        
        # Positive elements (were at odd positions 3,5,7,...)
        for m in 1:l
            old_idx = 2m + 1
            new_idx = m + 1
            ∇arr[:, :, old_idx, :] = Δ[:, :, new_idx, :]
        end
        
        # Negative elements (were at even positions 2,4,6,...)
        for m in 1:l
            old_idx = 2m
            new_idx = c + m + 1
            ∇arr[:, :, old_idx, :] = Δ[:, :, new_idx, :]
        end
        
        return (NoTangent(), ∇arr, NoTangent(), NoTangent())
    end
    
    return result, remap_pullback
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
    in_channels::Int=size(x, 3),
    out_channels::Int=size(target, 3),
    hidden_channels::Int=256,
    n_layers::Int=4,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=2.0,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid",
    inner_skip::Bool=true,
    outer_skip::Bool=true,
    operator_type::Symbol = :driscoll_healy,
    use_norm::Bool=false,
    downsampling_factor::Int=2,
    lr_0::Float64=2e-3, 
    parameters::QG3_Physics_Parameters=QG3_Physics_Parameters(pars, batch_size=batchsize),
    use_physics=true, 
    geometric=true,
    α=0.7f0)
    
    rng = Random.default_rng(seed)
    
    dataloader = DataLoader((x, target); batchsize=batchsize, shuffle=true) |> gpu_device()
    # Create the model
    sfno = SFNO(pars,
        batch_size=batchsize,
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
        operator_type=operator_type,
        downsampling_factor=downsampling_factor,
        use_norm=use_norm,
        )

    ps, st = Lux.setup(rng, sfno) |> gpu_device()
    @assert st isa NamedTuple "Model state should be a NamedTuple"
    # Rest of training setup remains similar
    
    opt = Optimisers.ADAM(lr_0)
    lr = i -> CosAnneal(lr_0, typeof(lr_0)(0), 20)(i)
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
    QG3_loss = make_QG3_loss(par_train; geometric=geometric, use_physics=use_physics, α=α)
    total_loss_tracker, physics_loss_tracker, data_loss_tracker = ntuple(_ -> Lag(Float32, 32), 3)

    iter = 1
    for (x, target_data) in Iterators.cycle(dataloader)
        #Optimisers.adjust!(train_state, lr(iter))
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

function load_precomputed_data(; N_sims=1000, root=dirname(@__DIR__), res="t42")
    if !isdir(string(root, "/data/"))
        error("Data directory not found at $(string(root, "/data/")). Please ensure precomputed data is available.")
    end
    if res != "t21" && res != "t42"
        error("Resolution $(res) not recognized. Supported resolutions are 't21' and 't42'.")
    end
    @load string(root, "/data/", res, "-precomputed-p.jld2") qg3ppars
    qg3ppars = qg3ppars
    qg3p = CUDA.@allowscalar QG3Model(qg3ppars)
    @load string(root, "/data/",res, "-precomputed-S.jld2") S
    if res == "t42"
        S = CUDA.@allowscalar QG3.reorder_SH_gpu(S, qg3ppars)
    else
        S = QG3.togpu(S)
    end 
    # initial conditions for streamfunction and vorticity
    @load string(root,"/data/", res, "_qg3_data_SH_CPU.jld2") q
    q = QG3.reorder_SH_gpu(q[:,:,:,1:N_sims+2], qg3ppars)
    ψ = QG3.qprimetoψ(qg3p, q)
    solψ = permutedims(QG3.transform_grid_data(ψ, qg3p),(2,3,1,4))
    solu = permutedims(QG3.transform_grid_data(q, qg3p),(2,3,1,4))
    
    return qg3ppars, qg3p, S, solψ, solu
end

N_sims = 1000

"""
    preprocess_data(;
        noise_level::Real=0.0,
        normalize::Bool=true,
        channelwise::Bool=false,
        to_gpu::Bool=false,
        noise_type::Symbol=:gaussian,
        slice_range::Union{Nothing, UnitRange, Tuple}=nothing
    )

Preprocess simulation data with normalization and noise injection options.

# Keyword Arguments
- `noise_level::Real=0.0`: Standard deviation of noise to add (0.0 = no noise)
- `normalize::Bool=true`: Whether to normalize the data
- `channelwise::Bool=false`: If true, normalize each channel independently
- `to_gpu::Bool=false`: If true, transfer data to GPU after preprocessing
- `noise_type::Symbol=:gaussian`: Type of noise (:gaussian, :uniform, :salt_pepper)
- `slice_range::Union{Nothing, UnitRange, Tuple}=nothing`: Optional range(s) to slice data
  - Can be a single UnitRange for the batch dimension
  - Can be a Tuple of ranges for multiple dimensions: (lat_range, lon_range, channel_range, batch_range)

# Returns
- `q_0`: Initial conditions (preprocessed)
- `q_evolved`: Evolved states (preprocessed)
- `μ`: Mean(s) used for normalization (scalar or vector)
- `σ`: Std(s) used for normalization (scalar or vector)
- `normalization_params`: NamedTuple with normalization metadata

# Examples
```julia
# Basic usage - no preprocessing
q_0, q_evolved, μ, σ, params = preprocess_data(noise_level=0.0, normalize=false)

# Normalize globally with noise
q_0, q_evolved, μ, σ, params = preprocess_data(noise_level=0.01, normalize=true)

# Channel-wise normalization with noise and GPU transfer
q_0, q_evolved, μ, σ, params = preprocess_data(
    noise_level=0.01, 
    normalize=true, 
    channelwise=true,
    to_gpu=true
)

# Slice data: take first 100 samples from batch dimension
q_0, q_evolved, μ, σ, params = preprocess_data(
    slice_range=1:100,
    normalize=true
)

# Advanced slicing: specify ranges for all dimensions
q_0, q_evolved, μ, σ, params = preprocess_data(
    slice_range=(1:48, 1:96, 1:3, 1:100),  # (lat, lon, channel, batch)
    normalize=true
)
```
"""
function preprocess_data(;
    normalize::Bool=true,
    channelwise::Bool=false,
    to_gpu::Bool=true,
    noise_level::Real=0.01, 
    noise_type::Symbol=:gaussian, 
    relative::Bool=true,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    slice_range::Union{Nothing, UnitRange, Tuple}=nothing
)
    
    # Load raw data
    qg3ppars, qg3p, S, solψ, solu = load_precomputed_data()
    
    # Apply slicing if requested
    if !isnothing(slice_range)
        if slice_range isa UnitRange
            # Single range = slice batch dimension only
            solu = solu[:, :, :, slice_range]
        elseif slice_range isa Tuple
            # Multiple ranges for different dimensions
            if length(slice_range) == 4
                lat_r, lon_r, ch_r, batch_r = slice_range
                solu = solu[lat_r, lon_r, ch_r, batch_r]
            else
                throw(ArgumentError("slice_range tuple must have 4 elements (lat, lon, channel, batch)"))
            end
        end
    end
    
    # Store original data info
    data_shape = size(solu)
    n_total_samples = size(solu, 4)
    
    # Normalization
    μ, σ = 0, 1
    if normalize
        solu, μ, σ = ESM_PINO.normalize_data(solu, channelwise=channelwise)
        @info "Data normalized" channelwise μ_type=typeof(μ) σ_type=typeof(σ)
    else
        @info "Normalization skipped"
    end
    
    # Split into initial conditions and evolved states
    # Ensure we have at least 2 time steps
    if n_total_samples < 2
        throw(ArgumentError("Need at least 2 time steps in data, got $n_total_samples"))
    end
    
    # q_0: initial conditions (all samples except last)
    q_0 = solu[:, :, :, 1:end-1]
    
    # q_evolved: evolved states (all samples except first)
    q_evolved = solu[:, :, :, 2:end]
    
    # Add noise if requested
    if noise_level > 0
        q_0 = Array(q_0)  # Ensure it's on CPU for noise injection
        q_evolved = Array(q_evolved)
        
        q_0 = ESM_PINO.add_noise(q_0, noise_level=noise_level, noise_type=noise_type, relative=relative, rng=rng)
        q_evolved = ESM_PINO.add_noise(q_evolved, noise_level=noise_level, noise_type=noise_type, relative=relative, rng=rng)
        
        @info "Noise added" noise_level noise_type
    else
        @info "No noise added"
    end
    
    # GPU transfer if requested
    if to_gpu
        q_0 = QG3.togpu(q_0)
        q_evolved = QG3.togpu(q_evolved)
        @info "Data transferred to GPU"
    else
        @info "Data kept on CPU"
    end
    
    # Store preprocessing metadata
    normalization_params = (
        normalized=normalize,
        channelwise=channelwise,
        μ=μ,
        σ=σ,
        noise_level=noise_level,
        noise_type=noise_type,
        original_shape=data_shape,
        n_pairs=size(q_0, 4),
        on_gpu=to_gpu
    )
    
    return q_0, q_evolved, μ, σ, normalization_params
end
"""
    preprocess_data(solu::AbstractArray;
        noise_level::Real=0.0,
        normalize::Bool=true,
        channelwise::Bool=false,
        to_gpu::Bool=false,
        noise_type::Symbol=:gaussian,
        slice_range::Union{Nothing, UnitRange, Tuple}=nothing
    )

Preprocess simulation data with normalization and noise injection options.

#Arguments
-`solu::AbstractArray`: Raw simulation data array of shape (lat, lon, channel, time)

# Keyword Arguments
- `noise_level::Real=0.0`: Standard deviation of noise to add (0.0 = no noise)
- `normalize::Bool=true`: Whether to normalize the data
- `channelwise::Bool=false`: If true, normalize each channel independently
- `to_gpu::Bool=false`: If true, transfer data to GPU after preprocessing
- `noise_type::Symbol=:gaussian`: Type of noise (:gaussian, :uniform, :salt_pepper)
- `slice_range::Union{Nothing, UnitRange, Tuple}=nothing`: Optional range(s) to slice data
  - Can be a single UnitRange for the batch dimension
  - Can be a Tuple of ranges for multiple dimensions: (lat_range, lon_range, channel_range, batch_range)

# Returns
- `q_0`: Initial conditions (preprocessed)
- `q_evolved`: Evolved states (preprocessed)
- `μ`: Mean(s) used for normalization (scalar or vector)
- `σ`: Std(s) used for normalization (scalar or vector)
- `normalization_params`: NamedTuple with normalization metadata

# Examples
```julia
# Basic usage - no preprocessing
q_0, q_evolved, μ, σ, params = preprocess_data(noise_level=0.0, normalize=false)

# Normalize globally with noise
q_0, q_evolved, μ, σ, params = preprocess_data(noise_level=0.01, normalize=true)

# Channel-wise normalization with noise and GPU transfer
q_0, q_evolved, μ, σ, params = preprocess_data(
    noise_level=0.01, 
    normalize=true, 
    channelwise=true,
    to_gpu=true
)

# Slice data: take first 100 samples from batch dimension
q_0, q_evolved, μ, σ, params = preprocess_data(
    slice_range=1:100,
    normalize=true
)

# Advanced slicing: specify ranges for all dimensions
q_0, q_evolved, μ, σ, params = preprocess_data(
    slice_range=(1:48, 1:96, 1:3, 1:100),  # (lat, lon, channel, batch)
    normalize=true
)
```
"""
function preprocess_data(solu::AbstractArray;
    normalize::Bool=true,
    channelwise::Bool=false,
    to_gpu::Bool=true,
    noise_level::Real=0.01, 
    noise_type::Symbol=:gaussian, 
    relative::Bool=true,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    slice_range::Union{Nothing, UnitRange, Tuple}=nothing
)
    @assert ndims(solu) == 4 "Input data solu must be a 4D array (lat, lon, channel, time)"
    # Apply slicing if requested
    if !isnothing(slice_range)
        if slice_range isa UnitRange
            # Single range = slice batch dimension only
            solu = solu[:, :, :, slice_range]
        elseif slice_range isa Tuple
            # Multiple ranges for different dimensions
            if length(slice_range) == 4
                lat_r, lon_r, ch_r, batch_r = slice_range
                solu = solu[lat_r, lon_r, ch_r, batch_r]
            else
                throw(ArgumentError("slice_range tuple must have 4 elements (lat, lon, channel, batch)"))
            end
        end
    end
    
    # Store original data info
    data_shape = size(solu)
    n_total_samples = size(solu, 4)
    
    # Normalization
    μ, σ = 0, 1
    if normalize
        solu, μ, σ = ESM_PINO.normalize_data(solu, channelwise=channelwise)
        @info "Data normalized" channelwise μ_type=typeof(μ) σ_type=typeof(σ)
    else
        @info "Normalization skipped"
    end
    
    # Split into initial conditions and evolved states
    # Ensure we have at least 2 time steps
    if n_total_samples < 2
        throw(ArgumentError("Need at least 2 time steps in data, got $n_total_samples"))
    end
    
    # q_0: initial conditions (all samples except last)
    q_0 = solu[:, :, :, 1:end-1]
    
    # q_evolved: evolved states (all samples except first)
    q_evolved = solu[:, :, :, 2:end]
    
    # Add noise if requested
    if noise_level > 0
        q_0 = Array(q_0)  # Ensure it's on CPU for noise injection
        q_evolved = Array(q_evolved)
        
        q_0 = ESM_PINO.add_noise(q_0, noise_level=noise_level, noise_type=noise_type, relative=relative, rng=rng)
        q_evolved = ESM_PINO.add_noise(q_evolved, noise_level=noise_level, noise_type=noise_type, relative=relative, rng=rng)
        
        @info "Noise added" noise_level noise_type
    else
        @info "No noise added"
    end
    
    # GPU transfer if requested
    if to_gpu
        q_0 = QG3.togpu(q_0)
        q_evolved = QG3.togpu(q_evolved)
        @info "Data transferred to GPU"
    else
        @info "Data kept on CPU"
    end
    
    # Store preprocessing metadata
    normalization_params = (
        normalized=normalize,
        channelwise=channelwise,
        μ=μ,
        σ=σ,
        noise_level=noise_level,
        noise_type=noise_type,
        original_shape=data_shape,
        n_pairs=size(q_0, 4),
        on_gpu=to_gpu
    )
    
    return q_0, q_evolved, μ, σ, normalization_params
end
