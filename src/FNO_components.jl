"""
    SpectralConv{T,N}

Spectral convolution layer for Fourier Neural Operator in Lux.jl.
Expects input in (spatial..., channel, batch) format.
# Arguments
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `modes`: Tuple specifying number of low-frequency modes to retain along each spatial dimension
- `T`: Data type for weights (default: ComplexF32)
- `N`: Number of spatial dimensions (inferred from length of `modes`)
# Fields
- `in_channels::Int`: Number of input channels
- `out_channels::Int`: Number of output channels
- `modes::NTuple{N,Int}`: Number of low-frequency modes to retain along each spatial dimension
# Details
- Uses FFT to transform input to frequency domain, applies learned complex weights to low-frequency modes, and transforms back to spatial domain
- Pads output back to original spatial dimensions after truncation
- Weights are initialized with Glorot-like scaling

"""
struct SpectralConv{T,N} <: Lux.AbstractLuxLayer
    in_channels::Int
    out_channels::Int
    modes::NTuple{N,Int}
end

function SpectralConv(in_channels::Integer, out_channels::Integer, modes::NTuple{N,Integer}) where N
    return SpectralConv{ComplexF32, N}(in_channels, out_channels, modes)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SpectralConv{T,N}) where {T,N}
    in_ch, out_ch, modes = layer.in_channels, layer.out_channels, layer.modes
    init_std = sqrt(2f0 / (in_ch + out_ch))  # Standard Glorot-like scaling
    weight = init_std * randn(rng, T, modes..., out_ch, in_ch)
    return (weight=weight,)
end

function Lux.initialstates(rng::AbstractRNG, layer::SpectralConv)
    return NamedTuple()  # No internal state needed
end

"""
    low_pass(x_ft, modes)

Apply a low-pass filter to a Fourier-transformed array by retaining only the lowest frequency modes.

# Arguments
- `x_ft`: A Fourier-transformed array with at least 2 trailing dimensions
- `modes`: A tuple or array specifying the number of low-frequency modes to keep along each leading dimension

# Returns
- A view of the input array `x_ft` containing only the specified low-frequency modes, preserving the last two dimensions in full

# Details
The function creates a view that selects the first `modes[i]` elements along each leading dimension `i`,
while keeping all elements of the last two dimensions. This effectively implements a low-pass filter
in Fourier space by truncating high-frequency modes.
"""
function low_pass(x_ft::AbstractArray, modes)
    return view(x_ft, map(d -> 1:d, modes)..., :, :)
end

"""
    apply_pattern(x_tr::AbstractArray{T,N}, weights::AbstractArray{T,3}) where {T,N}

Apply learned weight patterns to truncated Fourier coefficients.

# Arguments
- `x_tr::AbstractArray{T,N}`: Truncated Fourier coefficients after low-pass filtering, with shape (modes..., in_channels, batch)
- `weights::AbstractArray{T,4}`: Complex-valued learned weights with shape (modes..., out_channels, in_channels)

# Returns
- Weighted Fourier coefficients with shape (modes..., out_channels, batch)
"""
function apply_pattern(x_tr::AbstractArray{T,N}, weights::AbstractArray{T}) where {T,N}
    x_expanded = reshape(x_tr, size(x_tr)[1:N-2]..., 1, size(x_tr)[N-1:N]...)
    return dropdims(sum(weights .* x_expanded, dims=N), dims=N)
end

"""
    expand_pad_dims(pad_dims::Dims{N}) where {N}

Convert N-dimensional padding specification into format required for NNlib's pad_constant function.

# Arguments
- `pad_dims::Dims{N}`: Tuple of N integers specifying the total padding needed along each dimension

# Returns
- `NTuple{2N,Int}`: Tuple of 2N integers specifying padding for both sides of each dimension,
  where padding is applied only at the end of each dimension (start padding is always 0)
"""
function expand_pad_dims(pad_dims::Dims{N}) where {N}
    return ntuple(i -> isodd(i) ? 0 : pad_dims[i ÷ 2], 2N)
end
@non_differentiable expand_pad_dims(::Any)

function (layer::SpectralConv)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_ft = fft(x, 1:ndims(x)-2)  # Apply Fourier transform on spatial dimensions
    x_tr = low_pass(x_ft, layer.modes)  # Truncate high frequencies
    x_p = apply_pattern(x_tr, ps.weight)  # Apply learned spectral filters
    
    # Pad back to original shape
    pad_dims = size(x_ft)[1:(end - 2)] .- size(x_p)[1:(end - 2)]
    x_padded = NNlib.pad_constant(x_p, expand_pad_dims(pad_dims), false; dims=ntuple(identity, ndims(x_p) - 2))
    
    # Inverse Fourier transform to return to spatial domain
    return real(ifft(x_padded, 1:ndims(x_padded)-2)), st
end

"""
    SpectralKernel{P,F}

Combines a SpectralConv layer with a 1x1 convolution in parallel, followed by an activation function.  
Expects input in (spatial..., channel, batch) format.

# Arguments
- `in_ch`: Number of input channels
- `out_ch`: Number of output channels
- `modes`: Tuple specifying number of low-frequency modes to retain in the spectral branch
- `activation`: Activation function applied after combining spatial and spectral branches (default: `NNlib.gelu`)

# Fields
- `spatial_conv::P`: 1x1 convolution operating directly in the spatial domain
- `spectral_conv::SpectralConv`: Spectral convolution layer
- `activation::F`: Elementwise activation function

# Details
- The input is processed in parallel by a 1x1 convolution and a spectral convolution
- Outputs from both branches are summed and passed through the activation
- Useful for mixing local (spatial) and global (spectral) information
"""
struct SpectralKernel{P,F} <: Lux.AbstractLuxLayer
    spatial_conv::P  # 1x1 convolution
    spectral_conv::SpectralConv
    activation::F    # Activation function
end

function SpectralKernel(in_ch::Integer, out_ch::Integer , modes::NTuple{2,Integer}, activation=NNlib.gelu) 
    conv = Conv((1,1), in_ch => out_ch, pad=0, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32)
    spectral = SpectralConv(in_ch, out_ch, modes)
    return SpectralKernel(conv, spectral, activation)
end

function SpectralKernel(ch::Pair{<:Integer,<:Integer}, modes::NTuple{2,Integer}, activation=NNlib.gelu) 
    in_ch, out_ch = ch
    conv = Conv((1,1), in_ch => out_ch, pad=0, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32)
    spectral = SpectralConv(in_ch, out_ch, modes)
    return SpectralKernel(conv, spectral, activation)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SpectralKernel)
    ps_conv = Lux.initialparameters(rng, layer.spatial_conv)
    ps_spectral = Lux.initialparameters(rng, layer.spectral_conv)
    return (spatial=ps_conv, spectral=ps_spectral)
end

function Lux.initialstates(rng::AbstractRNG, layer::SpectralKernel)
    st_conv = Lux.initialstates(rng, layer.spatial_conv)
    st_spectral = Lux.initialstates(rng, layer.spectral_conv)
    return (spatial=st_conv, spectral=st_spectral)
end

function (layer::SpectralKernel)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spatial, st_spatial = layer.spatial_conv(x, ps.spatial, st.spatial)
    x_spectral, st_spectral = layer.spectral_conv(x, ps.spectral, st.spectral)
    x_out = layer.activation.(x_spatial .+ x_spectral)
    return x_out, (spatial=st_spatial, spectral=st_spectral)
end

"""
    SoftGating(channels::Int)

A soft gating layer that applies per-channel multiplicative scaling.  
Expects input in (height, width, channels, batch) format.

# Arguments
- `channels`: Number of channels in the input

# Fields
- `channels::Int`: Number of channels

# Details
- Learns a single scalar weight per channel
- Weights are initialized to 1.0 (identity scaling)
- Useful for lightweight residual or skip connections
"""
struct SoftGating <: Lux.AbstractLuxLayer
    channels::Int
end

function Lux.initialparameters(rng::AbstractRNG, layer::SoftGating)
    weight = ones(Float32, 1, 1, layer.channels, 1)
    return (weight=weight,)
end

function Lux.initialstates(rng::AbstractRNG, layer::SoftGating)
    return NamedTuple()
end

function (layer::SoftGating)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    return ps.weight .* x, st
end

"""
    ChannelMLP(channels::Int; expansion_factor=2.0, activation=gelu)

Implements a channel-wise MLP with a skip connection.  
Expects input in (height, width, channels, batch) format.

# Arguments
- `channels`: Number of input/output channels
- `expansion_factor`: Factor to expand hidden layer size (default: 2.0)
- `activation`: Nonlinear activation function in hidden layer (default: `NNlib.gelu`)

# Fields
- `mlp::M`: Two-layer Conv-based MLP with hidden dimension = `expansion_factor * channels`
- `skip::S`: Skip connection implemented as a `SoftGating` layer
- `expansion_factor::Number`: Factor controlling hidden dimension size

# Details
- Expands channels with a 1x1 convolution, applies nonlinearity, then projects back
- Adds gated skip connection to stabilize training
- Functions similarly to a feed-forward block in transformers
"""
struct ChannelMLP{M,S} <: Lux.AbstractLuxLayer
    mlp::M
    skip::S
    expansion_factor::Number
end

function ChannelMLP(channels::Int; expansion_factor=2.0, activation=NNlib.gelu)
    hidden_ch = Int(expansion_factor * channels)
    mlp = Chain(
        Conv((1, 1), channels => hidden_ch, activation, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32),
        Conv((1, 1), hidden_ch => channels, cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32)
    )
    skip = SoftGating(channels)
    return ChannelMLP(mlp, skip, expansion_factor)
end

function Lux.initialparameters(rng::AbstractRNG, layer::ChannelMLP)
    ps_mlp = Lux.initialparameters(rng, layer.mlp)
    ps_skip = Lux.initialparameters(rng, layer.skip)
    return (mlp=ps_mlp, skip=ps_skip)
end

function Lux.initialstates(rng::AbstractRNG, layer::ChannelMLP)
    st_mlp = Lux.initialstates(rng, layer.mlp)
    st_skip = Lux.initialstates(rng, layer.skip)
    return (mlp=st_mlp, skip=st_skip)
end

function (layer::ChannelMLP)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    y_mlp, st_mlp = layer.mlp(x, ps.mlp, st.mlp)
    y_skip, st_skip = layer.skip(x, ps.skip, st.skip)
    return y_mlp + y_skip, (mlp=st_mlp, skip=st_skip)
end

"""
    meshgrid(x, y)

Generates a 2D meshgrid from vectors `x` and `y`.
"""
function meshgrid(x, y)
    return (repeat(x, 1, length(y)), repeat(y', length(x), 1))
end

"""
    GridEmbedding2D(grid_boundaries=[[0f0, 1f0], [0f0, 1f0]])

Positional embedding that appends normalized 2D coordinates to the input.  
Expects input in (height, width, channels, batch) format.

# Arguments
- `grid_boundaries`: Vector of two intervals `[x_min, x_max]`, `[y_min, y_max]` specifying coordinate range along each axis

# Fields
- `boundaries_x::Vector{Float32}`: Range boundaries for x-coordinate
- `boundaries_y::Vector{Float32}`: Range boundaries for y-coordinate

# Details
- Constructs a 2D meshgrid of coordinates normalized to `[x_min, x_max] × [y_min, y_max]`
- Repeats coordinate grids across batch dimension
- Concatenates `grid_x` and `grid_y` as extra channels to the input
"""
struct GridEmbedding2D <: Lux.AbstractLuxLayer
    boundaries_x::Vector{Float32}
    boundaries_y::Vector{Float32}
end

GridEmbedding2D(grid_boundaries::Vector{Vector{Float32}} = [[0f0, 1f0], [0f0, 1f0]]) =
    GridEmbedding2D(grid_boundaries[1], grid_boundaries[2])

ChainRulesCore.@non_differentiable LinRange{Float32,Int64}(::Float32, ::Float32, ::Int64)

function Lux.initialparameters(rng::AbstractRNG, layer::GridEmbedding2D)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, layer::GridEmbedding2D)
    return NamedTuple()
end

function (layer::GridEmbedding2D)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    height, width, channels, batch_size = size(x)
    x_range = height == 1 ? [layer.boundaries_x[1]] : LinRange(layer.boundaries_x..., height)
    y_range = width == 1 ? [layer.boundaries_y[1]] : LinRange(layer.boundaries_y..., width)
    grid_x, grid_y = meshgrid(x_range, y_range)
    grid_x = reshape(grid_x, (height, width, 1, 1))
    grid_x = repeat(grid_x, outer = (1, 1, 1, batch_size)) |> get_device(x) 
    grid_y = reshape(grid_y, (height, width, 1, 1))
    grid_y = repeat(grid_y, outer = (1, 1, 1, batch_size)) |> get_device(x)
    return cat(x, grid_x, grid_y, dims=length(size(x))-1), st
end

ChainRulesCore.@non_differentiable (layer::GridEmbedding2D)(::Any)

"""
    FNO_Block(channels::Int, modes::NTuple{2,Int}; expansion_factor=2, activation=gelu)

A block that combines a SpectralKernel with a ChannelMLP.  
Expects input in (height, width, channels, batch) format.

# Arguments
- `channels`: Number of input/output channels
- `modes`: Tuple specifying number of low-frequency modes for the spectral convolution
- `expansion_factor`: Factor controlling hidden dimension size in ChannelMLP (default: 2)
- `activation`: Nonlinear activation function (default: `NNlib.gelu`)

# Fields
- `spectral_kernel::SpectralKernel`: Combines spectral and spatial convolutions
- `channel_mlp::ChannelMLP`: Channel-wise MLP with skip connection
- `channels::Int`: Number of channels
- `modes::NTuple{2,Int}`: Retained Fourier modes

# Details
- Applies spectral kernel to mix global/local features
- Follows with a channel MLP for nonlinear channel mixing
- Forms the core computational unit of a Fourier Neural Operator
"""
struct FNO_Block{T} <: Lux.AbstractLuxLayer
    spectral_kernel :: SpectralKernel{T,N} where N
    channel_mlp :: ChannelMLP
    channels :: Int
    modes :: NTuple{2, Int}
end

function FNO_Block(channels::Int, modes::NTuple{2,Int}; expansion_factor=2, activation=NNlib.gelu)
    spectral_kernel = SpectralKernel(channels => channels, modes, activation)
    channel_mlp = ChannelMLP(channels, expansion_factor=expansion_factor, activation=activation)
    return FNO_Block(spectral_kernel, channel_mlp, channels, modes)
end

function Lux.initialparameters(rng::AbstractRNG, block::FNO_Block)
    ps_spectral = Lux.initialparameters(rng, block.spectral_kernel)
    ps_channel = Lux.initialparameters(rng, block.channel_mlp)
    return (spectral_kernel=ps_spectral, channel_mlp=ps_channel)
end

function Lux.initialstates(rng::AbstractRNG, block::FNO_Block)
    st_spectral = Lux.initialstates(rng, block.spectral_kernel)
    st_channel = Lux.initialstates(rng, block.channel_mlp)
    return (spectral_kernel=st_spectral, channel_mlp=st_channel)
end

function (fno_block::FNO_Block)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spectral, st_spectral = fno_block.spectral_kernel(x, ps.spectral_kernel, st.spectral_kernel)
    x_mlp, st_channel = fno_block.channel_mlp(x_spectral, ps.channel_mlp, st.channel_mlp)
    return x_mlp, (spectral_kernel=st_spectral, channel_mlp=st_channel)
end

