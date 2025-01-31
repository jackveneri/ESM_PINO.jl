struct NewRepeatedLayer{L} <: Lux.AbstractLuxLayer
    layer::L
    repeats::Int
end

function Lux.setup(rng::AbstractRNG, layer::NewRepeatedLayer)
    # Initialize parameters and states for each repeated layer
    ps = []
    st = []
    for i in 1:layer.repeats
        ps_layer, st_layer = Lux.setup(rng, layer.layer)
        push!(ps, ps_layer)
        push!(st, st_layer)
    end

    # Create named tuples with explicit keys (e.g., :block_1, :block_2, etc.)
    block_keys = Tuple(Symbol("block_$i") for i in 1:layer.repeats)
    parameters = NamedTuple{block_keys}(ps)
    state = NamedTuple{block_keys}(st)

    return parameters, state
end

function Lux.apply(layer::NewRepeatedLayer, x, ps, st::NamedTuple)
    updated_st = NamedTuple()
    # Apply each repeated layer in sequence
    for key in keys(ps)
        x, new_block_st = Lux.apply(layer.layer, x, ps[key], st[key])
        # Rebuild the NamedTuple with the updated state for this block
        updated_st = merge(updated_st, NamedTuple{(key,)}((new_block_st,)))
    end
    return x, updated_st
end