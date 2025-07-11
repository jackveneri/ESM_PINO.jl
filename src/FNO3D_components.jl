"""
    SpectralKernel3D{P,F}

Combines a SpectralConv layer with a 1x1 convolution in parallel, followed by an activation function.
Expects input in (spatial..., channel, batch) format.
"""
struct SpectralKernel3D{P,F} <: Lux.AbstractLuxLayer
    spatial_conv::P  # 1x1 convolution
    spectral_conv::SpectralConv
    activation::F    # Activation function
end

function SpectralKernel3D(ch::Pair{<:Integer,<:Integer}, modes::NTuple{3,Integer}, activation=NNlib.gelu) 
    in_ch, out_ch = ch
    conv = Conv((1,1,1), in_ch => out_ch, pad=0)
    spectral = SpectralConv(in_ch, out_ch, modes)
    return SpectralKernel3D(conv, spectral, activation)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SpectralKernel3D)
    ps_conv = Lux.initialparameters(rng, layer.spatial_conv)
    ps_spectral = Lux.initialparameters(rng, layer.spectral_conv)
    return (spatial=ps_conv, spectral=ps_spectral)
end

function Lux.initialstates(rng::AbstractRNG, layer::SpectralKernel3D)
    st_conv = Lux.initialstates(rng, layer.spatial_conv)
    st_spectral = Lux.initialstates(rng, layer.spectral_conv)
    return (spatial=st_conv, spectral=st_spectral)
end

function (layer::SpectralKernel3D)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spatial, st_spatial = layer.spatial_conv(x, ps.spatial, st.spatial)
    x_spectral, st_spectral = layer.spectral_conv(x, ps.spectral, st.spectral)
    x_out = layer.activation.(x_spatial .+ x_spectral)
    return x_out, (spatial=st_spatial, spectral=st_spectral)
end
#=
x = randn(Float32, 64, 64, 64, 1, 1) |> gdev
model = SpectralKernel(1=>1, (16, 16, 16))
ps, st = Lux.setup(rng, model) |> gdev
model(x, ps, st)[1]
=#

"""
    SoftGating3D(channels::Int)

A soft gating layer that applies per-channel multiplicative scaling.
"""
struct SoftGating3D <: Lux.AbstractLuxLayer
    channels::Int
end

function Lux.initialparameters(rng::AbstractRNG, layer::SoftGating3D)
    weight = ones(Float32,1, 1, 1, layer.channels, 1)
    return (weight=weight,)
end

function Lux.initialstates(rng::AbstractRNG, layer::SoftGating3D)
    return NamedTuple()
end

function (layer::SoftGating3D)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    return ps.weight .* x, st
end

"""
    ChannelMLP3D(channels::Int; expansion_factor=0.5, activation=gelu)

Implements a channel-wise MLP with a skip connection.
"""
struct ChannelMLP3D{M,S} <: Lux.AbstractLuxLayer
    mlp::M
    skip::S
end

function ChannelMLP3D(channels::Int; expansion_factor::Number=0.5, activation=NNlib.gelu)
    hidden_ch = Int(expansion_factor * channels)
    mlp = Chain(
        Conv((1, 1, 1), channels => hidden_ch, activation),
        Conv((1, 1, 1), hidden_ch => channels)
    )
    skip = SoftGating3D(channels)
    return ChannelMLP3D(mlp, skip)
end

function Lux.initialparameters(rng::AbstractRNG, layer::ChannelMLP3D)
    ps_mlp = Lux.initialparameters(rng, layer.mlp)
    ps_skip = Lux.initialparameters(rng, layer.skip)
    return (mlp=ps_mlp, skip=ps_skip)
end

function Lux.initialstates(rng::AbstractRNG, layer::ChannelMLP3D)
    st_mlp = Lux.initialstates(rng, layer.mlp)
    st_skip = Lux.initialstates(rng, layer.skip)
    return (mlp=st_mlp, skip=st_skip)
end

function (layer::ChannelMLP3D)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    y_mlp, st_mlp = layer.mlp(x, ps.mlp, st.mlp)
    y_skip, st_skip = layer.skip(x, ps.skip, st.skip)
    return y_mlp + y_skip, (mlp=st_mlp, skip=st_skip)
end

#=
x = randn(Float32, 64, 64, 64, 1, 1) |> gdev
model = ChannelMLP(1, expansion_factor=2)
ps, st = Lux.setup(rng, model) |> gdev
model(x, ps, st)[1]
=#

"""
    GridEmbedding3D(grid_boundaries=[[0f0, 1f0], [0f0, 1f0], [0f0, 1f0]])

Positional embedding that appends a normalized 3D coordinate grid to input data.
"""
struct GridEmbedding3D <: Lux.AbstractLuxLayer
    boundaries_x::Vector{Float32}
    boundaries_y::Vector{Float32}
    boundaries_z::Vector{Float32}
end

GridEmbedding3D(grid_boundaries::Vector{Vector{Float32}} = [[0f0, 1f0], [0f0, 1f0], [0f0, 1f0]]) =
    GridEmbedding3D(grid_boundaries[1], grid_boundaries[2], grid_boundaries[3])

function Lux.initialparameters(rng::AbstractRNG, layer::GridEmbedding3D)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, layer::GridEmbedding3D)
    return NamedTuple()
end

function (layer::GridEmbedding3D)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    depth, height, width, channels, batch_size = size(x)
    
    # Handle singular dimensions
    x_range = depth == 1 ? [layer.boundaries_x[1]] : LinRange(layer.boundaries_x..., depth)
    y_range = height == 1 ? [layer.boundaries_y[1]] : LinRange(layer.boundaries_y..., height)
    z_range = width == 1 ? [layer.boundaries_z[1]] : LinRange(layer.boundaries_z..., width)
    
    # Create 3D grids
    grid_x = reshape(x_range, (depth, 1, 1)) |> get_device(x)
    grid_x = repeat(grid_x, 1, height, width)
    
    grid_y = reshape(y_range, (1, height, 1)) |> get_device(x)
    grid_y = repeat(grid_y, depth, 1, width)
    
    grid_z = reshape(z_range, (1, 1, width)) |> get_device(x)
    grid_z = repeat(grid_z, depth, height, 1)
    
    # Add channel and batch dimensions
    grid_x = reshape(grid_x, (depth, height, width, 1, 1))
    grid_x = repeat(grid_x, outer=(1, 1, 1, 1, batch_size)) |> get_device(x)
    
    grid_y = reshape(grid_y, (depth, height, width, 1, 1))
    grid_y = repeat(grid_y, outer=(1, 1, 1, 1, batch_size)) |> get_device(x)
    
    grid_z = reshape(grid_z, (depth, height, width, 1, 1))
    grid_z = repeat(grid_z, outer=(1, 1, 1, 1, batch_size)) |> get_device(x)
    
    # Concatenate grids with input
    embedded = cat(x, grid_x, grid_y, grid_z, dims=4)
    return embedded, st
end


ChainRulesCore.@non_differentiable (layer::GridEmbedding3D)(::Any)

"""
    FNO_Block3D

A block that combines a spectral kernel with a channel MLP. 
"""
struct FNO_Block3D <: Lux.AbstractLuxLayer
    spectral_kernel :: SpectralKernel3D
    channel_mlp :: ChannelMLP3D
    channels :: Int
    modes :: NTuple{3, Int}
end

function FNO_Block3D(channels::Int, modes::NTuple{3,Integer}; expansion_factor::Number=0.5, activation=NNlib.gelu) 
    spectral_kernel = SpectralKernel3D(channels => channels, modes, activation)
    channel_mlp = ChannelMLP3D(channels, expansion_factor=expansion_factor, activation=activation)
    return FNO_Block3D(spectral_kernel, channel_mlp, channels, modes)
end

function Lux.initialparameters(rng::AbstractRNG, block::FNO_Block3D)
    ps_spectral = Lux.initialparameters(rng, block.spectral_kernel)
    ps_channel = Lux.initialparameters(rng, block.channel_mlp)
    return (spectral_kernel=ps_spectral, channel_mlp=ps_channel)
end

function Lux.initialstates(rng::AbstractRNG, block::FNO_Block3D)
    st_spectral = Lux.initialstates(rng, block.spectral_kernel)
    st_channel = Lux.initialstates(rng, block.channel_mlp)
    return (spectral_kernel=st_spectral, channel_mlp=st_channel)
end

function (fno_block::FNO_Block3D)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spectral, st_spectral = fno_block.spectral_kernel(x, ps.spectral_kernel, st.spectral_kernel)
    x_mlp, st_channel = fno_block.channel_mlp(x_spectral, ps.channel_mlp, st.channel_mlp)
    return x_mlp, (spectral_kernel=st_spectral, channel_mlp=st_channel)
end
#=
gdev = gpu_device()
rng = Random.default_rng(0)
x = randn(Float32, 64, 64, 64, 1, 1) |> gdev
model = FNO_Block(1, (16, 16, 16), expansion_factor=2)
ps, st = Lux.setup(rng, model) |> gdev
model(x, ps, st)[1]
=#