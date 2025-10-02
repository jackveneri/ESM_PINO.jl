struct NoOpLayer <: Lux.AbstractLuxLayer end

Lux.initialparameters(rng::AbstractRNG, ::NoOpLayer) = NamedTuple()
Lux.initialstates(rng::AbstractRNG, ::NoOpLayer) = NamedTuple()
(l::NoOpLayer)(x::AbstractArray, ps::NamedTuple, st::NamedTuple) = (x, st)
LuxCore.parameterlength(::NoOpLayer) = 0

"""
    FourierNeuralOperator

A layer that combines the Fourier Neural Operator (FNO) with positional embeddings, spectral kernels, and channel MLPs.

## Arguments
- `in_channels::Int`: Number of input channels.
- `out_channels::Int`: Number of output channels.
- `hidden_channels::Int`: Number of hidden channels in the FNO blocks.
- `n_modes::NTuple{N,Integer}`: Number of Fourier modes in each dimension (default is (16, 16)).
- `n_layers::Int`: Number of FNO blocks (default is 4).
- `lifting_channel_ratio::Int`: Ratio for the lifting layer (default is 2). 
- `projection_channel_ratio::Int`: Ratio for the projection layer (default is 2).
- `channel_mlp_expansion::Number`: Expansion factor for the channel MLP (default is 0.5).
- `activation`: Activation function (default is `NNlib.gelu`).
- `positional_embedding::AbstractString`: Type of positional embedding to use (default is "grid"). Options are "grid", "no_grid", "grid1D", "no_grid1D", "grid3D", and "no_grid3D".

## Returns
- `FourierNeuralOperator`: A layer that combines the Fourier Neural Operator with the specified configurations.
"""
struct FourierNeuralOperator <: Lux.AbstractLuxContainerLayer{(:embedding, :lifting, :fno_blocks, :projection)}
    embedding ::Union{NoOpLayer, GridEmbedding2D, GridEmbedding1D, GridEmbedding3D}
    lifting ::Lux.AbstractLuxLayer
    fno_blocks 
    projection ::Lux.AbstractLuxLayer
end

function FourierNeuralOperator(;
    in_channels::Int,
    out_channels::Int,
    hidden_channels::Int=32,
    n_modes::NTuple{N,Integer}=(16, 16),
    n_layers::Int=4,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=2,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid",
) where N
    n_dim = length(n_modes)
    embedding = nothing
    if positional_embedding in ["grid","no_grid"]
        if positional_embedding == "grid" 
            embedding = GridEmbedding2D()
            in_channels += n_dim
        else
            embedding = NoOpLayer()
        end
        lifting = Chain(
            Conv((1, 1), in_channels => Int(lifting_channel_ratio * hidden_channels), activation),
            Conv((1, 1), Int(lifting_channel_ratio * hidden_channels) => hidden_channels, activation),
        )
        
        projection = Chain(
            Conv((1, 1), hidden_channels => Int(projection_channel_ratio * hidden_channels), activation),
            Conv((1, 1), Int(projection_channel_ratio * hidden_channels) => out_channels, identity),
        )
        
        fno_blocks = RepeatedLayer(FNO_Block(hidden_channels, n_modes; expansion_factor=channel_mlp_expansion, activation=activation), repeats=Val(n_layers))
    else 
        if positional_embedding in ["grid1D", "no_grid1D"]
            if positional_embedding == "grid1D"
                embedding = GridEmbedding1D()
                in_channels += n_dim
            else
                embedding = NoOpLayer()
            end
            
            lifting = Chain(
                Conv((1,), in_channels => Int(lifting_channel_ratio * hidden_channels), activation),
                Conv((1,), Int(lifting_channel_ratio * hidden_channels) => hidden_channels, activation),
            )
            
            projection = Chain(
                Conv((1,), hidden_channels => Int(projection_channel_ratio * hidden_channels), activation),
                Conv((1,), Int(projection_channel_ratio * hidden_channels) => out_channels, identity),
            )
            
            fno_blocks = RepeatedLayer(FNO_Block1D(hidden_channels, n_modes; expansion_factor=channel_mlp_expansion, activation=activation), repeats=Val(n_layers))
        else
            if positional_embedding in ["grid3D", "no_grid3D"]
                if positional_embedding == "grid3D"
                    embedding = GridEmbedding3D()
                    in_channels += n_dim
                else
                    embedding = NoOpLayer()
                end
                
                lifting = Chain(
                    Conv((1, 1, 1), in_channels => Int(lifting_channel_ratio * hidden_channels), activation),
                    Conv((1, 1, 1), Int(lifting_channel_ratio * hidden_channels) => hidden_channels, activation),
                )
                
                projection = Chain(
                    Conv((1, 1, 1), hidden_channels => Int(projection_channel_ratio * hidden_channels), activation),
                    Conv((1, 1, 1), Int(projection_channel_ratio * hidden_channels) => out_channels, identity),
                )
                
                fno_blocks = RepeatedLayer(FNO_Block3D(hidden_channels, n_modes; expansion_factor=channel_mlp_expansion, activation=activation), repeats=Val(n_layers))
            else
            throw(ArgumentError("Invalid positional embedding type. Supported arguments are 'grid' and 'grid1D'."))
            end
        end
    end
    return FourierNeuralOperator(embedding, lifting, fno_blocks, projection)
end

function Lux.initialparameters(rng::AbstractRNG, layer::FourierNeuralOperator)
    ps_embedding = isnothing(layer.embedding) ? NamedTuple() : Lux.initialparameters(rng, layer.embedding)
    ps_lifting = Lux.initialparameters(rng, layer.lifting)
    ps_fno_blocks = Lux.initialparameters(rng, layer.fno_blocks)
    ps_projection = Lux.initialparameters(rng, layer.projection)
    return (
        embedding=ps_embedding,
        lifting=ps_lifting,
        fno_blocks=ps_fno_blocks,
        projection=ps_projection,
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::FourierNeuralOperator)
    st_embedding = isnothing(layer.embedding) ? NamedTuple() : Lux.initialstates(rng, layer.embedding)
    st_lifting = Lux.initialstates(rng, layer.lifting)
    st_fno_blocks = Lux.initialstates(rng, layer.fno_blocks)
    st_projection = Lux.initialstates(rng, layer.projection)
    return (
        embedding=st_embedding,
        lifting=st_lifting,
        fno_blocks=st_fno_blocks,
        projection=st_projection,
    )
end

function (layer::FourierNeuralOperator)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    if !isnothing(layer.embedding)
        x, st_embedding = layer.embedding(x, ps.embedding, st.embedding)
    else
        st_embedding = st.embedding
    end
    
    x, st_lifting = layer.lifting(x, ps.lifting, st.lifting)
    x, st_fno_blocks = layer.fno_blocks(x, ps.fno_blocks, st.fno_blocks)
    x, st_projection = layer.projection(x, ps.projection, st.projection)

    return x, (embedding=st_embedding, lifting=st_lifting, fno_blocks=st_fno_blocks, projection=st_projection)
end
