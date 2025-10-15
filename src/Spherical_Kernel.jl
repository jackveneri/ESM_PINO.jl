"""
    AbstractSphericalKernel <: Lux.AbstractLuxLayer

Abstract supertype for spherical kernel layers.

Concrete implementations are provided by extensions:
- `ESM_PINOQG3Ext.SphericalKernel`: QG3-based transforms.
- `ESM_PINOSpeedyWeatherExt.SphericalKernel`: SpeedyWeather transforms.

Load the corresponding extension to use a specific implementation.
"""
abstract type AbstractSphericalKernel <: Lux.AbstractLuxLayer end

"""
    SphericalKernel{T}

Empty layer to test extension documentation.
"""
struct SphericalKernel{T} <: AbstractSphericalKernel
    transform::T
end

# Default behavior: throw or warn if used without an extension
Lux.initialparameters(rng::AbstractRNG, layer::SphericalKernel) =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::SphericalKernel) =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::SphericalKernel, ps, st, x) =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")
