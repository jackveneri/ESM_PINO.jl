struct SpectralPositionEmbedding{R,S,T,FT,U,TU,I,onGPU} <: Lux.AbstractLuxLayer
    shgg::QG3.SHtoGaussianGridTransform{R,S,T,FT,U,TU,I,onGPU}
    channels::Int
    inverse_linear_indices
    remap_plan
end

function SpectralPositionEmbedding(qg3ppars::QG3ModelParameters, channels::Int)
    shgg = QG3.SHtoGaussianGridTransform(qg3ppars, channels, N_batch=1)
    return SpectralPositionEmbedding(shgg, channels)
end

function SpectralPositionEmbedding(shgg::QG3.SHtoGaussianGridTransform{R,S,T,FT,U,TU,I,onGPU}, channels::Int) where {R,S,T,FT,U,TU,I,onGPU}
    i, m, n, b = channels, size(shgg.P,2), size(shgg.P,3), 1

    # Compute shifts for each column (inverse direction)
    shifts = (1:n) .÷ 2
    
    # Create source row indices (shifted in opposite direction)
    rows = 1:m
    source_rows = mod1.(rows .+ shifts', m)
    
    # Reshape for broadcasting
    source_rows_4d = reshape(source_rows, 1, m, n, 1)
    
    # Create index arrays
    i_idx = reshape(1:i, i, 1, 1, 1)
    n_idx = reshape(1:n, 1, 1, n, 1)
    b_idx = reshape(1:b, 1, 1, 1, b)
    
    # Broadcast gather indices - ensure everything stays Int
    i_indices = Int.(i_idx .+ zeros(Int, 1, m, n, b))
    row_indices = Int.(source_rows_4d .+ zeros(Int, i, 1, 1, b))
    col_indices = Int.(n_idx .+ zeros(Int, i, m, 1, b))
    b_indices = Int.(b_idx .+ zeros(Int, i, m, n, 1))
    
    # Create linear indices
    inverse_linear_indices = @. i_indices + 
                        (row_indices - 1) * i +
                        (col_indices - 1) * (i * m) +
                        (b_indices - 1) * (i * m * n)
    remap_plan = create_remap_plan(floor(Int, sqrt(channels)), size(shgg.P,3) ÷ 2)
    return SpectralPositionEmbedding{R,S,T,FT,U,TU,I,onGPU}(shgg, channels, inverse_linear_indices, remap_plan)    
end

function Lux.initialparameters(rng::AbstractRNG, layer::SpectralPositionEmbedding)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, layer::SpectralPositionEmbedding{R,S,T,FT,U,TU,I,onGPU}) where {R,S,T,FT,U,TU,I,onGPU}
    pos_embed_freq = zeros(R, size(layer.shgg.P,2), size(layer.shgg.P,3), layer.channels, 1)
    for i in 1:layer.channels
        l = Int(floor(sqrt(i-1)))
        m = Int(i - 1 - l^2 - l)  
        if m < 0
            pos_embed_freq[l+1,-2*m,Int(i),1] = (l + m) % 2 == 0 ? 1f0 : -1f0
        else
            pos_embed_freq[l+1,1+2*m,Int(i),1] = (l + m) % 2 == 0 ? 1f0 : -1f0
        end   
    end
    pos_embed_freq = permutedims(pos_embed_freq, (3,1,2,4)) 
    reorder_pos_embed_freq = reorderQG3_indexes_4d(pos_embed_freq, layer.inverse_linear_indices)
    if onGPU
        reorder_pos_embed_freq_slice = reorder_pos_embed_freq[:,:,1:2*floor(Int, sqrt(layer.channels)) + 1,:]
        reorder_pos_embed_freq = QG3.togpu(remap_array_components_fast_v2(reorder_pos_embed_freq_slice, layer.remap_plan))
    end
    pos_embed = QG3.transform_grid(reorder_pos_embed_freq, layer.shgg)
    pos_embed = permutedims(pos_embed, (2,3,1,4))
    pos_embed = pos_embed ./ maximum(abs.(pos_embed), dims=(1,2))
    return (pos_embed=pos_embed,)   
end

function(layer::SpectralPositionEmbedding)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    return x .+ st.pos_embed, st
end

function Lux.apply(layer::SpectralPositionEmbedding, x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    return x .+ st.pos_embed, st
end