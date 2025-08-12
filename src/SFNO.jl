"""
    SFNO

A layer that combines the Spherical Fourier Neural Operator (SFNO) with positional embeddings, spectral kernels, and channel MLPs.

## Arguments
- `in_channels::Int`: Number of input channels.
- `out_channels::Int`: Number of output channels.
- `hidden_channels::Int`: Number of hidden channels in the FNO blocks.
- `pars::QG3ModelParameters`: Parameters for the QG3 model.
- `n_layers::Int`: Number of FNO blocks (default is 4).
- `lifting_channel_ratio::Int`: Ratio for the lifting layer (default is 2). 
- `projection_channel_ratio::Int`: Ratio for the projection layer (default is 2).
- `channel_mlp_expansion::Number`: Expansion factor for the channel MLP (default is 0.5).
- `activation`: Activation function (default is `NNlib.gelu`).
- `positional_embedding::AbstractString`: Type of positional embedding to use (default is "grid"). Options are "grid", "no_grid".

## Returns
- `SFNO`: A layer that combines the Spherical Fourier Neural Operator with the specified configurations.
"""
struct SFNO <: Lux.AbstractLuxContainerLayer{(:embedding, :lifting, :sfno_blocks, :projection)}
    embedding ::Union{NoOpLayer, GridEmbedding2D}
    lifting ::Lux.AbstractLuxLayer
    sfno_blocks ::NewRepeatedLayer{SFNO_Block}
    projection ::Lux.AbstractLuxLayer
end

function SFNO(pars::QG3ModelParameters;
    batch_size::Int=1,
    modes::Int=pars.L,
    in_channels::Int,
    out_channels::Int,
    hidden_channels::Int=32,
    n_layers::Int=4,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=0.5,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid"
) 
    embedding = nothing
    if positional_embedding in ["grid","no_grid"]
        if positional_embedding == "grid" 
            embedding = GridEmbedding2D()
            in_channels += 2
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
        
        sfno_blocks = NewRepeatedLayer(SFNO_Block(hidden_channels, pars; modes=modes, batch_size=batch_size, expansion_factor=channel_mlp_expansion, activation=activation), n_layers)
    
    else
            throw(ArgumentError("Invalid positional embedding type. Supported arguments are 'grid' and 'no_grid'."))
    end
    return SFNO(embedding, lifting, sfno_blocks, projection)
end

function SFNO(pars::QG3ModelParameters,
    ggsh::QG3.GaussianGridtoSHTransform,
    shgg::QG3.SHtoGaussianGridTransform;
    modes::Int=pars.L,
    in_channels::Int,
    out_channels::Int,
    hidden_channels::Int=32,
    n_layers::Int=4,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=0.5,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid"
)
   embedding = nothing
    if positional_embedding in ["grid","no_grid"]
        if positional_embedding == "grid" 
            embedding = GridEmbedding2D()
            in_channels += 2
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
        
        sfno_blocks = NewRepeatedLayer(SFNO_Block(hidden_channels, pars, ggsh, shgg; modes=modes, expansion_factor=channel_mlp_expansion, activation=activation), n_layers)
    
    else
            throw(ArgumentError("Invalid positional embedding type. Supported arguments are 'grid' and 'no_grid'."))
    end
    return SFNO(embedding, lifting, sfno_blocks, projection) 
end

function Lux.initialparameters(rng::AbstractRNG, layer::SFNO)
    ps_embedding = isnothing(layer.embedding) ? NamedTuple() : Lux.initialparameters(rng, layer.embedding)
    ps_lifting = Lux.initialparameters(rng, layer.lifting)
    ps_sfno_blocks = Lux.initialparameters(rng, layer.sfno_blocks)
    ps_projection = Lux.initialparameters(rng, layer.projection)
    return (
        embedding=ps_embedding,
        lifting=ps_lifting,
        sfno_blocks=ps_sfno_blocks,
        projection=ps_projection,
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::SFNO)
    st_embedding = isnothing(layer.embedding) ? NamedTuple() : Lux.initialstates(rng, layer.embedding)
    st_lifting = Lux.initialstates(rng, layer.lifting)
    st_sfno_blocks = Lux.initialstates(rng, layer.sfno_blocks)
    st_projection = Lux.initialstates(rng, layer.projection)
    return (
        embedding=st_embedding,
        lifting=st_lifting,
        sfno_blocks=st_sfno_blocks,
        projection=st_projection,
    )
end

function (layer::SFNO)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    if !isnothing(layer.embedding)
        x, st_embedding = layer.embedding(x, ps.embedding, st.embedding)
    else
        st_embedding = st.embedding
    end
    
    x, st_lifting = layer.lifting(x, ps.lifting, st.lifting)
    x, st_sfno_blocks = layer.sfno_blocks(x, ps.sfno_blocks, st.sfno_blocks)
    x, st_projection = layer.projection(x, ps.projection, st.projection)

    return x, (embedding=st_embedding, lifting=st_lifting, sfno_blocks=st_sfno_blocks, projection=st_projection)
end
#=
using JLD2
@load string(@__DIR__, "/data/t21-precomputed-p.jld2") qg3ppars
#pre-compute the model 
qg3ppars = qg3ppars
x = rand(Float32, 32, 64, 3, 10)
model = SFNO(qg3ppars,
    batch_size=size(x, 4),
    modes = 30,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=0.5,
    activation=NNlib.gelu,
    positional_embedding="no_grid",
)
model = SFNO(qg3ppars, QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=size(x, 4)), QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=size(x, 4)),
    modes = 15,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=0.5,
    activation=NNlib.gelu,
    positional_embedding="no_grid",
)
gdev = gpu_device()
rnd = Random.default_rng(0)
ps, st = Lux.setup(rnd, model) |> gdev
x = x |> gdev
model(x, ps, st)[1]

using Zygote
grad = Zygote.gradient(ps -> sum(model(x, ps, st)[1]), ps)

@load string(@__DIR__, "/data/t42-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
model = SFNO(qg3ppars,
    batch_size=size(x, 4),
    modes = model.sfno_blocks.layer.spherical_kernel.spherical_conv.modes,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=0.5,
    activation=NNlib.gelu,
    positional_embedding="no_grid",
)
x = rand(Float32, 64, 128, 3, 10) |> gdev
model(x, ps, st)
=#