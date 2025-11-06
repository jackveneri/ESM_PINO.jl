"""
$(TYPEDSIGNATURES)
Empty layer to test extension documentation.
"""
struct SFNO{E, L, B, P, Q} <: Lux.AbstractLuxContainerLayer{(:embedding, :lifting, :sfno_blocks, :projection)}
    embedding :: E
    lifting :: L
    sfno_blocks :: B
    projection :: P
    ext_type :: Q
    outer_skip :: Bool
    lifting_channel_ratio::Int
    projection_channel_ratio::Int
end

# Constructor for your specific field types
function SFNO(
    embedding::Union{Lux.NoOpLayer, GridEmbedding2D},
    lifting::Lux.AbstractLuxLayer,
    sfno_blocks::Lux.Chain,
    projection::Lux.AbstractLuxLayer,
    ext_type::Type,
    outer_skip::Bool,
    lifting_channel_ratio::Int,
    projection_channel_ratio::Int
)
    return SFNO{typeof(embedding), typeof(lifting), typeof(sfno_blocks), typeof(projection), typeof(ext_type)}(
        embedding, lifting, sfno_blocks, projection, outer_skip, 
        lifting_channel_ratio, projection_channel_ratio
    )
end
# Default behavior: throw or warn if used without an extension
Lux.initialparameters(rng::AbstractRNG, layer::SFNO{E, L, B, P, Q}) where {E, L, B, P, Q} =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::SFNO{E, L, B, P, Q}) where {E, L, B, P, Q} =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::SFNO{E, L, B, P, Q}, ps, st, x) where {E, L, B, P, Q} =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")
