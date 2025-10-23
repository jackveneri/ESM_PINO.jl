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
function transfer_SFNO_model(model, qg3ppars; batch_size=model.sfno_blocks.model.spherical_kernel.spherical_conv.ggsh.FT_4d.plan.input_size[4])
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