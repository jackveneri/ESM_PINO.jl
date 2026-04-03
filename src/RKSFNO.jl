struct RKSFNO{E, L, B, P, Q} <: Lux.AbstractLuxContainerLayer{(:sfno,)}
    sfno::SFNO{E, L, B, P, Q}
end

Lux.initialparameters(rng::AbstractRNG, layer::RKSFNO{E, L, B, P, Q}) where {E, L, B, P, Q} =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::RKSFNO{E, L, B, P, Q}) where {E, L, B, P, Q} =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::RKSFNO{E, L, B, P, Q}, ps, st, x) where {E, L, B, P, Q} =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")
