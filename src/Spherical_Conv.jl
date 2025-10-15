"""
    AbstractSphericalConv <: Lux.AbstractLuxLayer

Abstract supertype for spherical convolution layers.

Concrete implementations are provided by extensions:
- `ESM_PINOQG3Ext.SphericalConv`: QG3-based transforms.
- `ESM_PINOSpeedyWeatherExt.SphericalConv`: SpeedyWeather transforms.

Load the corresponding extension to use a specific implementation.
"""
abstract type AbstractSphericalConv <: Lux.AbstractLuxLayer end

"""
    SphericalConv{T}

Empty layer to test extension documentation.
"""
struct SphericalConv{T} <: AbstractSphericalConv
    transform::T
end

# Default behavior: throw or warn if used without an extension
Lux.initialparameters(rng::AbstractRNG, layer::SphericalConv) =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::SphericalConv) =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::SphericalConv, ps, st, x) =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")