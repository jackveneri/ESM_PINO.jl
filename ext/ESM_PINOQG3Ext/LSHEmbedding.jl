struct LSHEmbedding{T} <: Lux.AbstractLuxLayer
    boundaries::Vector{Vector{T}}
    qg3ppars::QG3.QG3ModelParameters{T}
end

function LSHEmbedding(qg3ppars::QG3.QG3ModelParameters{T}, boundaries::Vector{Vector{T}} = [[T(0), T(1)], [T(0), T(1)]]) where {T}
    return LSHEmbedding(boundaries, qg3ppars)
end

function Lux.initialparameters(rng::AbstractRNG, layer::LSHEmbedding)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, layer::LSHEmbedding)
    LS_max = maximum(layer.qg3ppars.LS)
    LS_Embedding = (reshape(layer.qg3ppars.LS, size(layer.qg3ppars.LS)..., 1, 1) ./ LS_max) .* (layer.boundaries[1][2] - layer.boundaries[1][1]) .+ layer.boundaries[1][1]
    H_max = maximum(layer.qg3ppars.h)
    H_Embedding = (reshape(layer.qg3ppars.h, size(layer.qg3ppars.h)..., 1, 1) ./ H_max) .* (layer.boundaries[2][2] - layer.boundaries[2][1]) .+ layer.boundaries[2][1]
    return (LS_Embedding=LS_Embedding, H_Embedding=H_Embedding) 
end

function (layer::LSHEmbedding)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    spatial_dims = size(x)[1:2]
    @assert size(st.LS_Embedding)[1:2] == spatial_dims "The spatial dimensions of the input data do not match the dimensions of the LS embedding. Check your model configuration."
    LS_embed_expanded = repeat(st.LS_Embedding, 1, 1, 1, size(x,4))
    H_embed_expanded = repeat(st.H_Embedding, 1, 1, 1, size(x,4))
    return cat(x, LS_embed_expanded, H_embed_expanded; dims=3), st
end