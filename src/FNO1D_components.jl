"""
    SpectralKernel1D{P,F}
    Combines a SpectralConv layer with a 1x1 convolution in parallel, followed by an activation function.
Expects input in (spatial, channel, batch) format.
"""
struct SpectralKernel1D{P,F} <: Lux.AbstractLuxLayer
    spatial_conv::P  # 1x1 convolution
    spectral_conv::SpectralConv
    activation::F    # Activation function
end

function SpectralKernel1D(ch::Pair{<:Integer,<:Integer}, modes::NTuple{1,Integer}, activation=NNlib.gelu) 
    in_ch, out_ch = ch
    conv = Conv((1,), in_ch => out_ch, pad=0)
    spectral = SpectralConv(in_ch, out_ch, modes)
    return SpectralKernel1D(conv, spectral, activation)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SpectralKernel1D)
    ps_conv = Lux.initialparameters(rng, layer.spatial_conv)
    ps_spectral = Lux.initialparameters(rng, layer.spectral_conv)
    return (spatial=ps_conv, spectral=ps_spectral)
end

function Lux.initialstates(rng::AbstractRNG, layer::SpectralKernel1D)
    st_conv = Lux.initialstates(rng, layer.spatial_conv)
    st_spectral = Lux.initialstates(rng, layer.spectral_conv)
    return (spatial=st_conv, spectral=st_spectral)
end

function (layer::SpectralKernel1D)(x, ps, st::NamedTuple)
    x_spatial, st_spatial = layer.spatial_conv(x, ps.spatial, st.spatial)
    x_spectral, st_spectral = layer.spectral_conv(x, ps.spectral, st.spectral)
    x_out = layer.activation.(x_spatial .+ x_spectral)
    return x_out, (spatial=st_spatial, spectral=st_spectral)
end



"""
    SoftGating1D(channels::Int)

A soft gating layer that applies per-channel multiplicative scaling.
"""
struct SoftGating1D <: Lux.AbstractLuxLayer
    channels::Int
end

function Lux.initialparameters(rng::AbstractRNG, layer::SoftGating1D)
    weight = ones(Float32, 1, layer.channels, 1)
    return (weight=weight,)
end

function Lux.initialstates(rng::AbstractRNG, layer::SoftGating1D)
    return NamedTuple()
end

function (layer::SoftGating1D)(x, ps, st::NamedTuple)
    return ps.weight .* x, st
end



"""
    ChannelMLP1D(channels::Int; expansion_factor=0.5, activation=gelu)

Implements a channel-wise MLP with a skip connection.
"""
struct ChannelMLP1D{M,S} <: Lux.AbstractLuxLayer
    mlp::M
    skip::S
end

function ChannelMLP1D(channels::Int; expansion_factor=0.5, activation=gelu)
    hidden_ch = Int(expansion_factor * channels)
    mlp = Chain(
        Conv((1,), channels => hidden_ch, activation),
        Conv((1,), hidden_ch => channels)
    )
    skip = SoftGating1D(channels)
    return ChannelMLP1D(mlp, skip)
end

function Lux.initialparameters(rng::AbstractRNG, layer::ChannelMLP1D)
    ps_mlp = Lux.initialparameters(rng, layer.mlp)
    ps_skip = Lux.initialparameters(rng, layer.skip)
    return (mlp=ps_mlp, skip=ps_skip)
end

function Lux.initialstates(rng::AbstractRNG, layer::ChannelMLP1D)
    st_mlp = Lux.initialstates(rng, layer.mlp)
    st_skip = Lux.initialstates(rng, layer.skip)
    return (mlp=st_mlp, skip=st_skip)
end

function (layer::ChannelMLP1D)(x, ps, st)
    y_mlp, st_mlp = layer.mlp(x, ps.mlp, st.mlp)
    y_skip, st_skip = layer.skip(x, ps.skip, st.skip)
    return y_mlp .+ y_skip, (mlp=st_mlp, skip=st_skip)
end



struct GridEmbedding1D <: Lux.AbstractLuxLayer
    boundaries::Vector{Float32}
end

GridEmbedding1D() = GridEmbedding1D([0f0, 1f0])

function (layer::GridEmbedding1D)(x, ps, st)
    spatial_size, channels, batch_size = size(x)
    grid = reshape(LinRange(layer.boundaries..., spatial_size), (spatial_size, 1))
    #grid = stack([grid for _ in 1:batch_size]; dims=length(size(x)))
    grid = repeat(grid, outer = (1, 1, batch_size)) |> get_device(x)
    return cat(x, grid; dims=length(size(x))-1), st
end

function Lux.initialparameters(rng::AbstractRNG, layer::GridEmbedding1D)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, layer::GridEmbedding1D)
    return NamedTuple()
end

ChainRulesCore.@non_differentiable (layer::GridEmbedding1D)(::Any)

"""
    FNO_Block1D{N}

A block that combines a spectral kernel with a channel MLP. 
"""
struct FNO_Block1D{N} <: Lux.AbstractLuxLayer
    spectral_kernel :: SpectralKernel1D
    channel_mlp :: ChannelMLP1D
    channels :: Int
    modes :: NTuple{N, Int}
end

function FNO_Block1D(channels, modes; expansion_factor=0.5, activation=NNlib.gelu)
    spectral_kernel = SpectralKernel1D(channels => channels, modes, activation)
    channel_mlp = ChannelMLP1D(channels, expansion_factor=expansion_factor, activation=activation)
    return FNO_Block1D(spectral_kernel, channel_mlp, channels, modes)
end

function Lux.initialparameters(rng::AbstractRNG, block::FNO_Block1D)
    ps_spectral = Lux.initialparameters(rng, block.spectral_kernel)
    ps_channel = Lux.initialparameters(rng, block.channel_mlp)
    return (spectral_kernel=ps_spectral, channel_mlp=ps_channel)
end

function Lux.initialstates(rng::AbstractRNG, block::FNO_Block1D)
    st_spectral = Lux.initialstates(rng, block.spectral_kernel)
    st_channel = Lux.initialstates(rng, block.channel_mlp)
    return (spectral_kernel=st_spectral, channel_mlp=st_channel)
end

function (fno_block::FNO_Block1D)(x, ps, st::NamedTuple)
    x, st_spectral = fno_block.spectral_kernel(x, ps.spectral_kernel, st.spectral_kernel)
    x, st_channel = fno_block.channel_mlp(x, ps.channel_mlp, st.channel_mlp)
    return x, (spectral_kernel=st_spectral, channel_mlp=st_channel)
end

