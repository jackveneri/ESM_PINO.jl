using Lux, Random, QG3, NNlib
"""
    SFNO

Spherical Fourier Neural Operator (SFNO) layer combining positional embeddings, spectral kernels, and channel MLPs.

This layer implements the SFNO architecture on the sphere, optionally using Zonal Symmetric Kernels (ZSK)
following the approach described in **Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere**,
https://arxiv.org/abs/2204.06408.

# Arguments (primary constructor with `pars::QG3ModelParameters`)
- `pars::QG3ModelParameters`: Model parameters defining the spherical grid and maximum spherical harmonic degree `L`.
- `batch_size::Int=1`: Number of samples in a batch.
- `modes::Int=pars.L`: Maximum number of spherical harmonic modes to use.
- `in_channels::Int`: Number of input channels.
- `out_channels::Int`: Number of output channels.
- `hidden_channels::Int=32`: Number of hidden channels in the SFNO blocks.
- `n_layers::Int=4`: Number of SFNO blocks.
- `lifting_channel_ratio::Int=2`: Expansion ratio for the lifting layer.
- `projection_channel_ratio::Int=2`: Expansion ratio for the projection layer.
- `channel_mlp_expansion::Number=2.0`: Expansion factor for channel MLPs in each SFNO block.
- `activation`: Activation function (default is `NNlib.gelu`).
- `positional_embedding::AbstractString="grid"`: Type of positional embedding. Options: `"grid"` or `"no_grid"`.
- `inner_skip::Bool=true`: If true, use skip connections inside each SFNO block.
- `outer_skip::Bool=true`: If true, apply residual connection from lifting output to projection output.
- `gpu::Bool=true`: If true, computations are performed on GPU.
- `zsk::Bool=false`: If true, use Zonal Symmetric Kernels, enforcing longitudinal symmetry.

# Arguments (secondary constructor with `ggsh` and `shgg`)
- `ggsh::QG3.GaussianGridtoSHTransform`: Precomputed grid-to-SH transform.
- `shgg::QG3.SHtoGaussianGridTransform`: Precomputed SH-to-grid transform.
- Other keyword arguments are the same as for the primary constructor, except modes which default is set to `ggsh.output_size[1]`. Also, no need to specify `batch_size` or `gpu` as these are handled in the transforms.

# Returns
- `SFNO`: A Lux-compatible container layer.

# Details
- Constructs lifting, SFNO blocks, and projection layers compatible with Lux.jl.
- Positional embeddings are appended if `positional_embedding="grid"`.
- Supports both CPU and GPU execution.
- Zonal Symmetric Kernels (ZSK) reduce the number of parameters and improve stability on spherical domains.

# Example
```julia
using Lux, QG3, Random, NNlib, LuxCUDA

# Load precomputed QG3 parameters
qg3ppars = QG3.load_precomputed_params()[2]

# Input: [lat, lon, channels, batch]
x = rand(Float32, 32, 64, 3, 10)

# Construct SFNO layer using primary constructor
model1 = SFNO(qg3ppars;
    batch_size=size(x, 4),
    modes=30,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2.0,
    positional_embedding="no_grid",
    outer_skip=true,
    gpu=false
)

# Construct SFNO layer using secondary constructor
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=size(x,4))
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=size(x,4))
model2 = SFNO(ggsh, shgg;
    modes=15,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2.0,
    positional_embedding="no_grid",
    outer_skip=true,
    zsk=true
)

# Setup parameters and state
rng = Random.default_rng(0)
ps, st = Lux.setup(rng, model1)

# Forward pass
y, st = model1(x, ps, st)

# Compute gradients
using Zygote
gr = Zygote.gradient(ps -> sum(model1(x, ps, st)[1]), ps)
"""
struct SFNO <: Lux.AbstractLuxContainerLayer{(:embedding, :lifting, :sfno_blocks, :projection)}
    embedding ::Union{NoOpLayer, GridEmbedding2D}
    lifting ::Lux.AbstractLuxLayer
    sfno_blocks ::Lux.AbstractLuxLayer
    projection ::Lux.AbstractLuxLayer
    outer_skip :: Bool
    lifting_channel_ratio::Int
    projection_channel_ratio::Int
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
    channel_mlp_expansion::Number=2.0,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid",
    inner_skip::Bool=true,
    outer_skip::Bool=true,
    gpu=true,
    zsk=false
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
            Conv((1, 1), in_channels => Int(lifting_channel_ratio * hidden_channels), activation, cross_correlation=true, init_weight=kaiming_normal; init_bias=zeros32),
            Conv((1, 1), Int(lifting_channel_ratio * hidden_channels) => hidden_channels, identity, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32),
        )
        
        projection = Chain(
            Conv((1, 1), hidden_channels => Int(projection_channel_ratio * hidden_channels), activation,cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32),
            Conv((1, 1), Int(projection_channel_ratio * hidden_channels) => out_channels, identity, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32),
        )
        
        sfno_blocks = Lux.RepeatedLayer(SFNO_Block(hidden_channels, pars; modes=modes, batch_size=batch_size, expansion_factor=channel_mlp_expansion, activation=activation, skip=inner_skip, gpu=gpu, zsk=zsk), repeats=Val(n_layers))
    
    else
            throw(ArgumentError("Invalid positional embedding type. Supported arguments are 'grid' and 'no_grid'."))
    end
    return SFNO(embedding, lifting, sfno_blocks, projection, outer_skip, lifting_channel_ratio, projection_channel_ratio)
end

function SFNO(
    ggsh::QG3.GaussianGridtoSHTransform,
    shgg::QG3.SHtoGaussianGridTransform;
    modes::Int=ggsh.output_size[1],
    in_channels::Int,
    out_channels::Int,
    hidden_channels::Int=32,
    n_layers::Int=4,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=2.0,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid",
    inner_skip::Bool=true,
    outer_skip::Bool=true,
    zsk=false
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
            Conv((1, 1), in_channels => Int(lifting_channel_ratio * hidden_channels), activation, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32),
            Conv((1, 1), Int(lifting_channel_ratio * hidden_channels) => hidden_channels, identity, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32),
        )
        
        projection = Chain(
            Conv((1, 1), hidden_channels => Int(projection_channel_ratio * hidden_channels), activation, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32),
            Conv((1, 1), Int(projection_channel_ratio * hidden_channels) => out_channels, identity, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32),
        )
        
        sfno_blocks = Lux.RepeatedLayer(SFNO_Block(hidden_channels, ggsh, shgg; modes=modes, expansion_factor=channel_mlp_expansion, activation=activation, skip=inner_skip, zsk=zsk), repeats=Val(n_layers))
    
    else
            throw(ArgumentError("Invalid positional embedding type. Supported arguments are 'grid' and 'no_grid'."))
    end
    return SFNO(embedding, lifting, sfno_blocks, projection, outer_skip, lifting_channel_ratio, projection_channel_ratio) 
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
    residual = x
    x, st_sfno_blocks = layer.sfno_blocks(x, ps.sfno_blocks, st.sfno_blocks)
    if layer.outer_skip
        x_out = x + residual
    else
        x_out = x
    end
    x, st_projection = layer.projection(x_out, ps.projection, st.projection)
    
    return x, (embedding=st_embedding, lifting=st_lifting, sfno_blocks=st_sfno_blocks, projection=st_projection)
end
#=
using JLD2, Lux, Random, QG3, NNlib, LuxCUDA, ChainRulesCore
@load string(dirname(@__DIR__), "/data/t21-precomputed-p.jld2") qg3ppars
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
    channel_mlp_expansion=2,
    activation=NNlib.gelu,
    positional_embedding="no_grid",
)
model = SFNO(QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=size(x, 4)), QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=size(x, 4)),
    modes = 15,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2,
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

@load string(dirname(@__DIR__), "/data/t42-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
model = SFNO(qg3ppars,
    batch_size=size(x, 4),
    modes = model.sfno_blocks.model.spherical_kernel.spherical_conv.modes,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2,
    activation=NNlib.gelu,
    positional_embedding="no_grid",
)
x = rand(Float32, 64, 128, 3, 10) |> gdev
model(x, ps, st)
=#