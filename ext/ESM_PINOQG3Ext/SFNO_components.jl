"""
    SphericalKernel{P,F}

Combines a SphericalConv layer with a 1x1 convolution in parallel, followed by an activation function.
Expects input in (spatial..., channel, batch) format.
# Arguments
- `hidden_channels`: Number of channels
- `pars`: Precomputed QG3 model parameters (QG3ModelParameters)
- `activation`: Activation function applied after combining spatial and spectral branches (default: `NNlib.gelu`)
- `modes`: Number of spherical harmonic modes to retain (default: `pars.L`)
- `batch_size`: Batch size for transforms (default: 1)
- `gpu`: Whether to use GPU (default: true)
- `zsk`: Whether to use Zonal Symmetric Kernels (ZSK) (default: false)

#Returns
- `SphericalKernel`: A Lux-compatible layer operating on 4D arrays `[lat,

# Fields
- `spatial_conv::P`: 1x1 convolution operating directly in the spatial domain
- `spherical_conv::SphericalalConv`: Spherical convolution layer
- `activation::F`: Elementwise activation function

# Details
- The input is processed in parallel by a 1x1 convolution and a spherical convolution
- Outputs from both branches are summed and passed through the activation
- Useful for mixing local (spatial) and global (spectral) information
"""
struct SphericalKernel{P,F} <: Lux.AbstractLuxLayer
    spatial_conv::P  # 1x1 convolution
    spherical_conv::SphericalConv
    activation::F    # Activation function
end

function SphericalKernel(hidden_channels::Int, pars::QG3.QG3ModelParameters, activation=NNlib.gelu; modes::Int=pars.L, batch_size::Int=1, gpu::Bool=true, zsk::Bool=false) 
    conv = Lux.Conv((1,1), hidden_channels => hidden_channels, pad=0, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32)
    spherical = SphericalConv(pars, hidden_channels; modes=modes, batch_size=batch_size, gpu=gpu, zsk=zsk)
    return SphericalKernel(conv, spherical, activation)
end
"""
    function SphericalKernel(hidden_channels::Int, ggsh::QG3.GaussianGridtoSHTransform, shgg::QG3.SHtoGaussianGridTransform, activation=NNlib.gelu; modes::Int=ggsh.output_size[1], zsk::Bool=false)

Construct a SphericalKernel layer using precomputed transforms.

# Arguments
- `hidden_channels::Int`: Number of channels
- `ggsh::GaussianGridtoSHTransform`: Transformation from Gaussian grid to spherical harmonics
- `shgg::SHtoGaussianGridTransform`: Transformation from spherical harmonics back to Gaussian grid
- `activation`: Activation function applied after combining spatial and spectral branches (default: `NNlib.gelu`)
- `modes::Int=ggsh.output_size[1]`: Number of spherical harmonic modes to retain (default: `ggsh.output_size[1]`)
- `zsk::Bool=false`: Whether to use Zonal Symmetric Kernels (ZSK) (default: false)

# Returns
- `SphericalKernel`: A Lux-compatible layer operating on 4D arrays `[lat, lon, channels, batch]`.

# Fields
- `spatial_conv::P`: 1x1 convolution operating directly in the spatial domain
- `spherical_conv::SphericalalConv`: Spherical convolution layer
- `activation::F`: Elementwise activation function
"""
function SphericalKernel(hidden_channels::Int, ggsh::QG3.GaussianGridtoSHTransform, shgg::QG3.SHtoGaussianGridTransform, activation=NNlib.gelu; modes::Int=ggsh.output_size[1], zsk::Bool=false)
    conv = Lux.Conv((1,1), hidden_channels => hidden_channels, pad=0, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32)
    spherical = SphericalConv(hidden_channels, ggsh, shgg, modes, zsk=zsk)
    return SphericalKernel(conv, spherical, activation)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::SphericalKernel)
    ps_conv = Lux.initialparameters(rng, layer.spatial_conv)
    ps_spherical = Lux.initialparameters(rng, layer.spherical_conv)
    return (spatial=ps_conv, spherical=ps_spherical)
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::SphericalKernel)
    st_conv = Lux.initialstates(rng, layer.spatial_conv)
    st_spherical = Lux.initialstates(rng, layer.spherical_conv)
    return (spatial=st_conv, spherical=st_spherical)
end

function (layer::SphericalKernel)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spatial, st_spatial = layer.spatial_conv(x, ps.spatial, st.spatial)
    x_spherical, st_spherical = layer.spherical_conv(x, ps.spherical, st.spherical)
    x_out = layer.activation.(x_spatial .+ x_spherical)
    return x_out, (spatial=st_spatial, spherical=st_spherical)
end
#=
using JLD2, Lux, Random, QG3, NNlib, LuxCUDA, ChainRulesCore
include("SphericalConvTypeSpec.jl")
include("FNO_components.jl")
@load string(dirname(@__DIR__), "/data/t21-precomputed-p.jld2") qg3ppars


#pre-compute the model 
qg3ppars = qg3ppars
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
x = rand(Float32, 32, 64, 32, 1)
model = SphericalKernel(32, qg3ppars, batch_size=size(x,4), modes=30)
model = SphericalKernel(32, ggsh, shgg, modes=30)
gdev = gpu_device()
rnd = Random.default_rng(0)
ps, st = Lux.setup(rnd, model) |> gdev 
x = x |> gdev
model(x, ps, st)

using Zygote
gr = Zygote.gradient(ps -> sum(model(x, ps, st)[1]), ps)

@load string(dirname(@__DIR__), "/data/t42-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
model = SphericalKernel(32, qg3ppars, batch_size=size(x,4), modes=model.spherical_conv.modes)
x = rand(Float32, 64, 128, 32, 1) |> gdev
model(x, ps, st)
=#
"""
    SFNO_Block

A block that combines a spherical kernel with a channel MLP.
Expects input in (spatial..., channel, batch) format.

# Arguments
- `channels::Int`: Number of input/output channels
- `pars::QG3ModelParameters`: Precomputed QG3 model parameters (QG3ModelParameters)
- `modes::Int=pars.L`: Number of spherical harmonic modes to retain (default: `pars.L`)
- `batch_size::Int=1`: Batch size for transforms (default: 1)
- `expansion_factor::Real=2.0`: Expansion factor for the ChannelMLP (default: 2.0)
- `activation`: Activation function applied after combining spatial and spectral branches (default: `NNlib.gelu`)
- `skip::Bool=true`: Whether to include a skip connection (default: true)
- `gpu::Bool=true`: Whether to use GPU (default: true)
- `zsk::Bool=false`: Whether to use Zonal Symmetric Kernels (ZSK) (default: false)

# Returns
- `SFNO_Block`: A Lux-compatible layer operating on 4D arrays `[lat, lon, channels, batch]`.

# Fields
- `spherical_kernel::SphericalKernel`: Spherical kernel layer
- `channel_mlp::ChannelMLP`: Channel-wise MLP layer
- `channels::Int`: Number of input/output channels
- `skip::Bool`: Whether to include a skip connection

# Details
- The input is processed by a SphericalKernel followed by a ChannelMLP
- If `skip` is true, the input is added to the output (residual connection)


"""
struct SFNO_Block <: Lux.AbstractLuxLayer
    spherical_kernel :: SphericalKernel
    channel_mlp :: ChannelMLP
    channels :: Int
    skip :: Bool
end

function SFNO_Block(channels::Int, pars::QG3.QG3ModelParameters; modes::Int=pars.L, batch_size::Int=1, expansion_factor::Real=2.0, activation=NNlib.gelu, skip::Bool=true, gpu::Bool=true, zsk::Bool=false)
    spherical_kernel = SphericalKernel(channels, pars, activation; modes=modes, batch_size=batch_size, gpu=gpu, zsk=zsk)
    channel_mlp = ChannelMLP(channels, expansion_factor=expansion_factor, activation=activation)
    return SFNO_Block(spherical_kernel, channel_mlp, channels, skip)
end
"""
    SFNO_Block

A block that combines a spherical kernel with a channel MLP.
Expects input in (spatial..., channel, batch) format.

# Arguments
- `channels::Int`: Number of input/output channels
- `ggsh::GaussianGridtoSHTransform`: Transformation from Gaussian grid to spherical harmonics
- `shgg::SHtoGaussianGridTransform`: Transformation from spherical harmonics back to Gaussian
- `modes::Int=ggsh.output_size[1]`: Number of spherical harmonic modes to retain (default: `ggsh.output_size[1]`)
- `expansion_factor::Real=2.0`: Expansion factor for the ChannelMLP (default: 2.0)
- `activation`: Activation function applied after combining spatial and spectral branches (default: `NNlib.gelu`)
- `skip::Bool=true`: Whether to include a skip connection (default: true)
- `zsk::Bool=false`: Whether to use Zonal Symmetric Kernels (ZSK) (default: false)

# Returns
- `SFNO_Block`: A Lux-compatible layer operating on 4D arrays `[lat, lon, channels, batch]`.

# Fields
- `spherical_kernel::SphericalKernel`: Spherical kernel layer
- `channel_mlp::ChannelMLP`: Channel-wise MLP layer
- `channels::Int`: Number of input/output channels
- `skip::Bool`: Whether to include a skip connection

# Details
- The input is processed by a SphericalKernel followed by a ChannelMLP
- If `skip` is true, the input is added to the output (residual connection)


"""
function SFNO_Block(channels::Int, ggsh::QG3.GaussianGridtoSHTransform, shgg::QG3.SHtoGaussianGridTransform; modes::Int = ggsh.output_size[1], expansion_factor::Real=2.0, activation=NNlib.gelu, skip::Bool=true, zsk::Bool=false)
    spherical_kernel = SphericalKernel(channels, ggsh, shgg, activation; modes=modes, zsk=zsk)
    channel_mlp = ChannelMLP(channels, expansion_factor=expansion_factor, activation=activation)
    return SFNO_Block(spherical_kernel, channel_mlp, channels, skip)
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::SFNO_Block)
    ps_spherical = Lux.initialparameters(rng, block.spherical_kernel)
    ps_channel = Lux.initialparameters(rng, block.channel_mlp)
    return (spherical_kernel=ps_spherical, channel_mlp=ps_channel)
end

function Lux.initialstates(rng::Random.AbstractRNG, block::SFNO_Block)
    st_spherical = Lux.initialstates(rng, block.spherical_kernel)
    st_channel = Lux.initialstates(rng, block.channel_mlp)
    return (spherical_kernel=st_spherical, channel_mlp=st_channel)
end

function (sfno_block::SFNO_Block)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spherical, st_spherical = sfno_block.spherical_kernel(x, ps.spherical_kernel, st.spherical_kernel)
    x_mlp, st_channel = sfno_block.channel_mlp(x_spherical, ps.channel_mlp, st.channel_mlp)
    if sfno_block.skip
        x_out = x + x_mlp
    else
        x_out = x_mlp
    end
    return x_out, (spherical_kernel=st_spherical, channel_mlp=st_channel)
end
#=
using JLD2
@load string(dirname(@__DIR__), "/data/t21-precomputed-p.jld2") qg3ppars


#pre-compute the model 

qg3ppars = qg3ppars
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
x = rand(Float32, 32, 64, 32, 1)
model = SFNO_Block(32, qg3ppars, batch_size=size(x,4), modes=30)
model = SFNO_Block(32, ggsh, shgg, modes=30)
gdev = gpu_device()
rnd = Random.default_rng(0)
ps, st = Lux.setup(rnd, model) |> gdev
x = x |> gdev
model(x, ps, st)

using Zygote
gr = Zygote.gradient(ps -> sum(model(x, ps, st)[1]), ps)

@load string(dirname(@__DIR__), "/data/t42-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
model = SFNO_Block(32, qg3ppars, batch_size=size(x,4), modes=model.spherical_kernel.spherical_conv.modes)
x = rand(Float32, 64, 128, 32, 1) |> gdev
model(x, ps, st)
=#