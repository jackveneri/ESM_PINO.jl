"""
    AbstractSFNOBlock <: Lux.AbstractLuxLayer

Abstract supertype for SFNO_Block layers.

Concrete implementations are provided by extensions:
- `ESM_PINOQG3Ext.SFNO_Block`: QG3-based transforms.
- `ESM_PINOSpeedyWeatherExt.SFNO_Block`: SpeedyWeather transforms.

Load the corresponding extension to use a specific implementation.
"""
abstract type AbstractSFNOBlock <: Lux.AbstractLuxLayer end

"""
    SFNO_Block{T}

Empty layer to test extension documentation.
"""
struct SFNO_Block{T} <: AbstractSFNOBlock
    transform::T
end

# Default behavior: throw or warn if used without an extension
Lux.initialparameters(rng::AbstractRNG, layer::SFNO_Block) =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::SFNO_Block) =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::SFNO_Block, ps, st, x) =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")
