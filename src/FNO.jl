struct NoOpLayer <: Lux.AbstractLuxLayer end

Lux.initialparameters(rng::AbstractRNG, ::NoOpLayer) = NamedTuple()
Lux.initialstates(rng::AbstractRNG, ::NoOpLayer) = NamedTuple()
(l::NoOpLayer)(x::AbstractArray, ps::NamedTuple, st::NamedTuple) = (x, st)
LuxCore.parameterlength(::NoOpLayer) = 0

"""
    FourierNeuralOperator <: Lux.AbstractLuxContainerLayer

A Fourier Neural Operator (FNO) container that optionally includes positional embeddings,
lifting and projection convolutions, and a stack of FNO blocks.

# Arguments
- `in_channels::Int`: Number of input channels.
- `out_channels::Int`: Number of output channels.
- `hidden_channels::Int=32`: Number of hidden channels used inside FNO blocks.
- `n_modes::NTuple{N,Int}=(16, 16)`: Number of retained Fourier modes per spatial dimension.
- `n_layers::Int=4`: Number of FNO blocks to stack.
- `lifting_channel_ratio::Int=2`: Channel expansion ratio used in the lifting layer.
- `projection_channel_ratio::Int=2`: Channel expansion ratio used in the projection layer.
- `channel_mlp_expansion::Number=2`: Expansion factor inside ChannelMLP of each block.
- `activation=NNlib.gelu`: Activation function used in conv layers.
- `positional_embedding`::AbstractString="grid": Choice of positional embedding:
"grid", "no_grid" => 2D variants (GridEmbedding2D or NoOpLayer)
"grid1D", "no_grid1D" => 1D variants (GridEmbedding1D or NoOpLayer)
"grid3D", "no_grid3D" => 3D variants (GridEmbedding3D or NoOpLayer)

# Fields
- `embedding`: Positional embedding layer (a GridEmbeddingND or NoOpLayer).
- `lifting`: Lifting convolution(s) mapping in_channels -> hidden_channels.
- `fno_blocks`: Repeated stack of FNO blocks appropriate to dimensionality.
- `projection`: Projection convolution(s) mapping hidden_channels -> out_channels.

# Examples

Example (2D data with grid embedding):
```julia
using Lux, Random, NNlib

rng = Random.default_rng()

layer = FourierNeuralOperator(
    in_channels=3,
    out_channels=2,
    hidden_channels=32,
    n_modes=(12, 12),
    n_layers=4,
    positional_embedding="grid"
)

ps = Lux.initialparameters(rng, layer)
st = Lux.initialstates(rng, layer)

# Input tensor (H, W, C, Batch)
x = randn(Float32, 64, 64, 3, 10)

y, st_new = layer(x, ps, st)
@show size(y)   # expect (64, 64, 2, 10)
```
```julia
Example (1D data without grid embedding):

layer1d = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=16,
    n_modes=(8,),
    n_layers=3,
    positional_embedding="no_grid1D"
)

x1 = randn(Float32, 128, 1, 5)   # (L, C, Batch)
y1, _ = layer1d(x1,
    Lux.initialparameters(rng, layer1d),
    Lux.initialstates(rng, layer1d)
)
@show size(y1)   # expect (128, 1, 5)
```
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
