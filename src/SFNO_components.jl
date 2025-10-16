"""
$(TYPEDSIGNATURES)
Empty layer to test extension documentation.
"""
struct SphericalKernel{T} <: Lux.AbstractLuxLayer
    spatial_conv::Lux.Conv
    spherical_conv::SphericalConv{T}  
    activation::Function
end
# Default behavior: throw or warn if used without an extension
Lux.initialparameters(rng::AbstractRNG, layer::SphericalKernel{T}) where T =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::SphericalKernel{T}) where T =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::SphericalKernel{T}, ps, st, x) where T =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")

"""
$(TYPEDSIGNATURES)
Empty layer to test extension documentation.
"""
struct SFNO_Block{T} <: Lux.AbstractLuxLayer
    spherical_kernel :: SphericalKernel{T}
    channel_mlp :: ChannelMLP
    channels :: Int
    skip :: Bool
end

# Default behavior: throw or warn if used without an extension
Lux.initialparameters(rng::AbstractRNG, layer::SFNO_Block{T}) where T =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::SFNO_Block{T}) where T =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::SFNO_Block{T}, ps, st, x) where T =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")