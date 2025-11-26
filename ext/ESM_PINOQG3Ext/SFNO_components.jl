"""
$(TYPEDSIGNATURES)
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
-`norm::Union{Lux.InstanceNorm, Lux.NoOpLayer}`: Optional normalization layer

# Details
- The input is processed in parallel by a 1x1 convolution and a spherical convolution
- Outputs from both branches are summed and passed through the activation
- Useful for mixing local (spatial) and global (spectral) information
"""
function ESM_PINO.SphericalKernel(hidden_channels::Int, pars::QG3.QG3ModelParameters;
                                    use_norm=false, 
                                    modes::Int=pars.L, 
                                    batch_size::Int=1, 
                                    gpu::Bool=true, 
                                    operator_type::Symbol = :driscoll_healy, # Changed from zsk::Bool
                                    inner_mixing::Bool=false,
                                    bias::Bool=false
                                    )  
    if inner_mixing
        conv = Lux.Conv((1,1), hidden_channels => hidden_channels, identity, cross_correlation=true, init_weight=kaiming_normal, use_bias=bias, init_bias=zeros32)
    else
        conv= Lux.NoOpLayer()
    end
    spherical = ESM_PINO.SphericalConv(pars, hidden_channels; modes=modes, batch_size=batch_size, gpu=gpu, operator_type=operator_type)
    if use_norm
        # InstanceNorm expects (H, W, C, B) format by default in Lux
        norm = Lux.InstanceNorm(hidden_channels, epsilon=1f-6, affine=true)
    else
        norm = Lux.NoOpLayer()
    end
    return ESM_PINO.SphericalKernel(conv, spherical, norm)
end
"""
$(TYPEDSIGNATURES)
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
-`norm::Union{Lux.InstanceNorm, Lux.NoOpLayer}`: Optional normalization layer
"""
function ESM_PINO.SphericalKernel(hidden_channels::Int, ggsh::QG3.GaussianGridtoSHTransform, shgg::QG3.SHtoGaussianGridTransform;
                                    use_norm=false, 
                                    modes::Int=ggsh.output_size[1], 
                                    operator_type::Symbol = :driscoll_healy, # Changed from zsk::Bool
                                    inner_mixing::Bool=false,
                                    bias::Bool=false
                                    )  
    if inner_mixing
        conv = Lux.Conv((1,1), hidden_channels => hidden_channels, identity, cross_correlation=true, init_weight=kaiming_normal, use_bias=bias, init_bias=zeros32)
    else
        conv= Lux.NoOpLayer()
    end
    spherical = ESM_PINO.SphericalConv(hidden_channels, ggsh, shgg, modes, operator_type=operator_type)
    if use_norm
        # InstanceNorm expects (H, W, C, B) format by default in Lux
        norm = Lux.InstanceNorm(hidden_channels, epsilon=1f-6, affine=true)
    else
        norm = Lux.NoOpLayer()
    end
    return ESM_PINO.SphericalKernel(conv, spherical, norm)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::ESM_PINO.SphericalKernel{ESM_PINOQG3})
    ps_spherical = Lux.initialparameters(rng, layer.spherical_conv)
    ps_norm = Lux.initialparameters(rng, layer.norm)
    ps_conv = Lux.initialparameters(rng, layer.spatial_conv)
    return (spatial=ps_conv, norm=ps_norm, spherical=ps_spherical)
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::ESM_PINO.SphericalKernel{ESM_PINOQG3})
    st_conv = Lux.initialstates(rng, layer.spatial_conv)
    st_norm = Lux.initialstates(rng, layer.norm)
    st_spherical = Lux.initialstates(rng, layer.spherical_conv)
    return (spatial=st_conv, norm=st_norm, spherical=st_spherical)
end

function (layer::ESM_PINO.SphericalKernel{ESM_PINOQG3})(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spherical, res_spherical, st_spherical = layer.spherical_conv(x, ps.spherical, st.spherical)
    x_spherical, st_norm = layer.norm(x_spherical, ps.norm, st.norm)
    x_spatial, st_spatial = layer.spatial_conv(res_spherical, ps.spatial, st.spatial) #here I the downsampled retransformed input, could add that with no downsampling you just take the input
    if layer.spatial_conv != Lux.NoOpLayer()
        x_out = x_spatial .+ x_spherical
    else
        x_out = x_spherical
    end
    return x_out, res_spherical, (spatial=st_spatial, norm=st_norm, spherical=st_spherical)
end
function Lux.apply(layer::ESM_PINO.SphericalKernel{ESM_PINOQG3}, x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spherical, res_spherical, st_spherical = layer.spherical_conv(x, ps.spherical, st.spherical)
    x_spherical, st_norm = layer.norm(x_spherical, ps.norm, st.norm)
    x_spatial, st_spatial = layer.spatial_conv(res_spherical, ps.spatial, st.spatial) #here I the downsampled retransformed input, could add that with no downsampling you just take the input
    if layer.spatial_conv != Lux.NoOpLayer()
        x_out = x_spatial .+ x_spherical
    else
        x_out = x_spherical
    end
    return x_out, res_spherical, (spatial=st_spatial, norm=st_norm, spherical=st_spherical)
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
$(TYPEDSIGNATURES)
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
-`activation::Function`: Activation function applied after the block

# Details
- The input is processed by a SphericalKernel followed by a ChannelMLP
- If `skip` is true, the input is added to the output (residual connection)
"""
function ESM_PINO.SFNO_Block(
    channels::Int, 
    pars::QG3.QG3ModelParameters; 
    modes::Int=pars.L, 
    batch_size::Int=1, 
    expansion_factor::Real=2.0, 
    activation=NNlib.gelu, 
    skip::Bool=true, 
    gpu::Bool=true, 
    operator_type::Symbol = :driscoll_healy, # Changed from zsk::Bool
    use_norm::Bool=false,
    soft_gating=false,
    bias::Bool=false
    )
    spherical_kernel = ESM_PINO.SphericalKernel(channels, pars; use_norm=use_norm, modes=modes, batch_size=batch_size, gpu=gpu, operator_type=operator_type, bias=bias)
    channel_mlp = ChannelMLP(channels, expansion_factor=expansion_factor, activation=activation, soft_gating=soft_gating, bias=bias)
    return ESM_PINO.SFNO_Block(spherical_kernel, channel_mlp, channels, skip, activation)
end
"""
$(TYPEDSIGNATURES)
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
-`activation::Function`: Activation function applied after the block

# Details
- The input is processed by a SphericalKernel followed by a ChannelMLP
- If `skip` is true, the input is added to the output (residual connection)


"""
function ESM_PINO.SFNO_Block(
    channels::Int, 
    ggsh::QG3.GaussianGridtoSHTransform, 
    shgg::QG3.SHtoGaussianGridTransform; 
    modes::Int = 0, 
    expansion_factor::Real=2.0, 
    activation=NNlib.gelu, 
    skip::Bool=true, 
    operator_type::Symbol = :driscoll_healy,
    use_norm::Bool=false,
    soft_gating=false,
    bias::Bool=false
    )
    spherical_kernel = ESM_PINO.SphericalKernel(channels, ggsh, shgg; use_norm=use_norm, modes=modes, operator_type=operator_type, bias=bias)
    channel_mlp = ChannelMLP(channels, expansion_factor=expansion_factor, activation=activation, soft_gating=soft_gating, bias=bias)
    return ESM_PINO.SFNO_Block(spherical_kernel, channel_mlp, channels, skip, activation)
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::ESM_PINO.SFNO_Block{ESM_PINOQG3})
    ps_spherical = Lux.initialparameters(rng, block.spherical_kernel)
    ps_channel = Lux.initialparameters(rng, block.channel_mlp)
    return (spherical_kernel=ps_spherical, channel_mlp=ps_channel)
end

function Lux.initialstates(rng::Random.AbstractRNG, block::ESM_PINO.SFNO_Block{ESM_PINOQG3})
    st_spherical = Lux.initialstates(rng, block.spherical_kernel)
    st_channel = Lux.initialstates(rng, block.channel_mlp)
    return (spherical_kernel=st_spherical, channel_mlp=st_channel)
end

function (sfno_block::ESM_PINO.SFNO_Block{ESM_PINOQG3})(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spherical, res_spherical, st_spherical = sfno_block.spherical_kernel(x, ps.spherical_kernel, st.spherical_kernel)
    x_mlp, st_channel = sfno_block.channel_mlp(x_spherical, ps.channel_mlp, st.channel_mlp)
    if sfno_block.skip
        x_out = res_spherical + x_mlp
    else
        x_out = x_mlp
    end
    return x_out, (spherical_kernel=st_spherical, channel_mlp=st_channel)
end
function Lux.apply(sfno_block::ESM_PINO.SFNO_Block{ESM_PINOQG3}, x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spherical, res_spherical, st_spherical = sfno_block.spherical_kernel(x, ps.spherical_kernel, st.spherical_kernel)
    x_mlp, st_channel = sfno_block.channel_mlp(x_spherical, ps.channel_mlp, st.channel_mlp)
    if sfno_block.skip
        x_out = res_spherical + x_mlp
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