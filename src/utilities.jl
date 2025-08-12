function add_noise(data::AbstractArray; 
                  noise_level::Real=0.1, 
                  noise_type::Symbol=:gaussian, 
                  relative::Bool=true,
                  rng::AbstractRNG=Random.GLOBAL_RNG)
    
    # Validate noise level
    noise_level < 0 && throw(ArgumentError("noise_level must be non-negative"))
    
    # Determine floating point type for calculations
    F = float(eltype(data))
    
    # Calculate scaling factor (standard deviation for relative noise)
    scale = one(F)
    if relative
        if !isempty(data)
            s = std(data)
            scale = iszero(s) ? one(F) : F(s)
        end
    end
    
    F_noise_level = F(noise_level)
    scaled_level = F_noise_level * scale

    # Generate noise based on type
    if noise_type == :gaussian
        noise = randn(rng, F, size(data)) .* scaled_level
    elseif noise_type == :uniform
        noise = (rand(rng, F, size(data)) .- F(0.5)) .* (2 * scaled_level)
    else
        error("Unsupported noise type: $noise_type. Use :gaussian or :uniform.")
    end

    return data .+ noise
end

function normalize_data(data)
    μ = mean(data)
    σ = std(data)
    return (data .- μ) ./ σ, μ, σ
end

function denormalize_data(data, μ, σ)
    return data .* σ .+ μ
end
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
function transfer_SFNO_model(model, qg3ppars; batch_size=model.sfno_blocks.layer.spherical_kernel.spherical_conv.ggsh.FT_4d.plan.input_size[4])
    superres_model = SFNO(qg3ppars,
        batch_size = batch_size,
        modes = model.sfno_blocks.layer.spherical_kernel.spherical_conv.modes,
        in_channels = model.lifting.layers.layer_1.in_chs,
        out_channels = model.projection.layers.layer_2.out_chs, #watch out as you might have more than 2 layers
        hidden_channels = model.sfno_blocks.layer.spherical_kernel.spherical_conv.hidden_channels,
        n_layers = model.sfno_blocks.repeats,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        channel_mlp_expansion=0.5,
        activation = model.sfno_blocks.layer.spherical_kernel.activation,
        positional_embedding = model.embedding == NoOpLayer() ? "no_grid" : "grid",
        )
    return superres_model
end