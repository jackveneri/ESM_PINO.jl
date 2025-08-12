"""
    SphericalKernel{P,F}

Combines a SphericalConv layer with a 1x1 convolution in parallel, followed by an activation function.
Expects input in (spatial..., channel, batch) format.
"""
struct SphericalKernel{P,F} <: Lux.AbstractLuxLayer
    spatial_conv::P  # 1x1 convolution
    spherical_conv::SphericalConv
    activation::F    # Activation function
end

function SphericalKernel(hidden_channels::Int, ch::Pair{<:Integer,<:Integer}, pars::QG3ModelParameters, activation=NNlib.gelu; modes::Int=pars.L, batch_size::Int=1) 
    in_ch, out_ch = ch
    conv = Conv((1,1), in_ch => out_ch, pad=0)
    spherical = SphericalConv(pars, hidden_channels; modes=modes, batch_size=batch_size)
    return SphericalKernel(conv, spherical, activation)
end

function SphericalKernel(hidden_channels::Int, ch::Pair{<:Integer,<:Integer}, pars::QG3.QG3ModelParameters, ggsh::QG3.GaussianGridtoSHTransform, shgg::QG3.SHtoGaussianGridTransform, activation=NNlib.gelu; modes::Int=pars.L)
    in_ch, out_ch = ch
    conv = Conv((1,1), in_ch => out_ch, pad=0)
    if ggsh.FT_4d.plan.input_size[1] == hidden_channels
        spherical = SphericalConv(pars, hidden_channels, modes, ggsh, shgg)
    else
        println("Warning: SphericalConv hidden channels do not match the input channels of the GaussianGridtoSHTransform. Falling back to safe constructor")
        spherical = SphericalConv(pars, hidden_channels; batch_size=ggsh.FT_4d.plan.input_size[4], modes=modes)
    end
    return SphericalKernel(conv, spherical, activation)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SphericalKernel)
    ps_conv = Lux.initialparameters(rng, layer.spatial_conv)
    ps_spherical = Lux.initialparameters(rng, layer.spherical_conv)
    return (spatial=ps_conv, spherical=ps_spherical)
end

function Lux.initialstates(rng::AbstractRNG, layer::SphericalKernel)
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
using JLD2
@load string(@__DIR__, "/data/t21-precomputed-p.jld2") qg3ppars


#pre-compute the model 
qg3ppars = qg3ppars
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
x = rand(Float32, 32, 64, 32, 1)
model = SphericalKernel(32, 32 => 32, qg3ppars, batch_size=size(x,4), modes=30)
gdev = gpu_device()
rnd = Random.default_rng(0)
ps, st = Lux.setup(rnd, model) |> gdev
x = x |> gdev
model(x, ps, st)

using Zygote
gr = Zygote.gradient(ps -> sum(model(x, ps, st)[1]), ps)

@load string(@__DIR__, "/data/t42-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
model = SphericalKernel(32, 32 => 32, qg3ppars, batch_size=size(x,4), modes=model.spherical_conv.modes)
x = rand(Float32, 64, 128, 32, 1) |> gdev
model(x, ps, st)
=#
"""
    SFNO_Block

A block that combines a spectral kernel with a channel MLP. 
"""
struct SFNO_Block <: Lux.AbstractLuxLayer
    spherical_kernel :: SphericalKernel
    channel_mlp :: ChannelMLP
    pars :: QG3ModelParameters
    channels :: Int
end

function SFNO_Block(channels::Int, pars::QG3ModelParameters; modes::Int=pars.L, batch_size::Int=1, expansion_factor=0.5, activation=NNlib.gelu)
    spherical_kernel = SphericalKernel(channels, channels => channels, pars, activation;modes=modes, batch_size=batch_size)
    channel_mlp = ChannelMLP(channels, expansion_factor=expansion_factor, activation=activation)
    return SFNO_Block(spherical_kernel, channel_mlp, pars, channels)
end

function SFNO_Block(channels::Int, pars::QG3.QG3ModelParameters, ggsh::QG3.GaussianGridtoSHTransform, shgg::QG3.SHtoGaussianGridTransform; modes::Int=pars.L, expansion_factor=0.5, activation=NNlib.gelu)
    spherical_kernel = SphericalKernel(channels, channels => channels, pars, ggsh, shgg, activation; modes=modes)
    channel_mlp = ChannelMLP(channels, expansion_factor=expansion_factor, activation=activation)
    return SFNO_Block(spherical_kernel, channel_mlp, pars, channels)
end

function Lux.initialparameters(rng::AbstractRNG, block::SFNO_Block)
    ps_spherical = Lux.initialparameters(rng, block.spherical_kernel)
    ps_channel = Lux.initialparameters(rng, block.channel_mlp)
    return (spherical_kernel=ps_spherical, channel_mlp=ps_channel)
end

function Lux.initialstates(rng::AbstractRNG, block::SFNO_Block)
    st_spherical = Lux.initialstates(rng, block.spherical_kernel)
    st_channel = Lux.initialstates(rng, block.channel_mlp)
    return (spherical_kernel=st_spherical, channel_mlp=st_channel)
end

function (sfno_block::SFNO_Block)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spherical, st_spherical = sfno_block.spherical_kernel(x, ps.spherical_kernel, st.spherical_kernel)
    x_mlp, st_channel = sfno_block.channel_mlp(x_spherical, ps.channel_mlp, st.channel_mlp)
    return x_mlp, (spherical_kernel=st_spherical, channel_mlp=st_channel)
end
#=
using JLD2
@load string(@__DIR__, "/data/t21-precomputed-p.jld2") qg3ppars


#pre-compute the model 
qg3ppars = qg3ppars
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
x = rand(Float32, 32, 64, 32, 1)
model = SFNO_Block(32, qg3ppars, batch_size=size(x,4), modes=30)
gdev = gpu_device()
rnd = Random.default_rng(0)
ps, st = Lux.setup(rnd, model) |> gdev
x = x |> gdev
model(x, ps, st)

using Zygote
gr = Zygote.gradient(ps -> sum(model(x, ps, st)[1]), ps)

@load string(@__DIR__, "/data/t42-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
model = SFNO_Block(32, qg3ppars, batch_size=size(x,4), modes=model.spherical_kernel.spherical_conv.modes)
x = rand(Float32, 64, 128, 32, 1) |> gdev
model(x, ps, st)
=#