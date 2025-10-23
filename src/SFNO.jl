"""
$(TYPEDSIGNATURES)
Empty layer to test extension documentation.
"""
struct SFNO{F,S,T} <: Lux.AbstractLuxContainerLayer{(:embedding, :lifting, :sfno_blocks, :projection)}
    embedding ::Union{Lux.NoOpLayer, GridEmbedding2D}
    lifting ::Lux.AbstractLuxLayer
    sfno_blocks ::Lux.RepeatedLayer{F,S,SFNO_Block{T}} #rewrite with  Lux.Chain to handle different block structures
    projection ::Lux.AbstractLuxLayer
    outer_skip :: Bool
    lifting_channel_ratio::Int
    projection_channel_ratio::Int
end
# Default behavior: throw or warn if used without an extension
Lux.initialparameters(rng::AbstractRNG, layer::SFNO{F,S,T}) where {F,S,T} =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::SFNO{F,S,T}) where {F,S,T} =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::SFNO{F,S,T}, ps, st, x) where {F,S,T} =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")
