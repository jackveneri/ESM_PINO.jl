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
    gain::Real
end

function SpectralConv(in_channels::Integer, out_channels::Integer, modes::NTuple{N,Integer}; gain::Real=2.0) where N
    return SpectralConv{ComplexF32, N}(in_channels, out_channels, modes, gain)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SpectralConv{T,N}) where {T,N}
    in_ch, out_ch, modes = layer.in_channels, layer.out_channels, layer.modes
    init_std = sqrt(T(layer.gain) / in_ch )  # Standard Kaiming-like scaling
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

function Base.show(io::IO, layer::SpectralConv{T,N}) where {T,N}
    print(io, "SpectralConv(", layer.in_channels, "→", layer.out_channels, ", modes=", layer.modes, ", weight number format=", T, ")")
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
struct SpectralKernel{T,N} <: Lux.AbstractLuxContainerLayer{(:spatial_conv,:spectral_conv,:norm)}
    spatial_conv::Union{Lux.Conv, Lux.NoOpLayer}  
    spectral_conv::SpectralConv{T,N}
    norm::Union{Lux.InstanceNorm, Lux.NoOpLayer}
end

function SpectralKernel(in_ch::Integer, out_ch::Integer, modes::NTuple{N,Integer};
                        inner_mixing::Bool=false,
                        use_norm::Bool=false,
                        bias::Bool=false) where N
    if inner_mixing 
        conv = Conv(ntuple(_ -> 1, N), in_ch => out_ch, pad=0, cross_correlation=true, init_weight=kaiming_normal(gain=1.0), use_bias=bias, init_bias=zeros32)
    else
        conv = Lux.NoOpLayer()
    end
    spectral = SpectralConv(in_ch, out_ch, modes)
    if use_norm
        # InstanceNorm expects (H, W, C, B) format by default in Lux
        norm = Lux.InstanceNorm(out_ch, epsilon=1f-6, affine=true)
    else
        norm = Lux.NoOpLayer()
    end
    return SpectralKernel(conv, spectral, norm)
end

function SpectralKernel(ch::Pair{<:Integer,<:Integer}, modes::NTuple{N,Integer}; inner_mixing::Bool=false,
                        use_norm::Bool=false,
                        bias::Bool=false) where N
    in_ch, out_ch = ch
    return SpectralKernel(in_ch, out_ch, modes; inner_mixing=inner_mixing,
                        use_norm=use_norm,
                        bias=bias)    
end

function Lux.initialparameters(rng::AbstractRNG, layer::SpectralKernel)
    ps_conv = Lux.initialparameters(rng, layer.spatial_conv)
    ps_spectral = Lux.initialparameters(rng, layer.spectral_conv)
    ps_norm = Lux.initialparameters(rng, layer.norm)
    return (spatial=ps_conv, norm=ps_norm, spectral=ps_spectral)
end

function Lux.initialstates(rng::AbstractRNG, layer::SpectralKernel)
    st_conv = Lux.initialstates(rng, layer.spatial_conv)
    st_spectral = Lux.initialstates(rng, layer.spectral_conv)
    st_norm = Lux.initialstates(rng, layer.norm)
    return (spatial=st_conv, norm=st_norm, spectral=st_spectral)
end

function (layer::SpectralKernel)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    x_spatial, st_spatial = layer.spatial_conv(x, ps.spatial, st.spatial)
    x_spectral, st_spectral = layer.spectral_conv(x, ps.spectral, st.spectral)
    x_spectral, st_norm = layer.norm(x_spectral, ps.norm, st.norm)
    if layer.spatial_conv != Lux.NoOpLayer()
        x_out = x_spatial .+ x_spectral
    else
        x_out = x_spectral
    end
    return x_out, (spatial=st_spatial, norm=st_norm, spectral=st_spectral)
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
    spatial_dims::Int
    
    # Constructor with default spatial_dims=2 for backward compatibility
    SoftGating(channels::Int, spatial_dims::Int=2) = new(channels, spatial_dims)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SoftGating)
    weight_shape = ntuple(i -> i <= layer.spatial_dims ? 1 : 
                         (i == layer.spatial_dims + 1 ? layer.channels : 1), 
                         layer.spatial_dims + 2)
    weight = ones(Float32, weight_shape)
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
struct ChannelMLP{M,S} <: Lux.AbstractLuxContainerLayer{(:mlp, :skip)}
    mlp::M
    skip::S
    expansion_factor::Number
end

function ChannelMLP(channels::Int; spatial_dims::Int=2, expansion_factor=2.0, 
                    activation=NNlib.gelu, soft_gating=true, bias=false)
    hidden_ch = Int(expansion_factor * channels)
    kernel_size = ntuple(_ -> 1, spatial_dims)
    
    mlp = Chain(
        Conv(kernel_size, channels => hidden_ch, activation, 
             cross_correlation=true, init_weight=kaiming_normal, init_bias=zeros32),
        Conv(kernel_size, hidden_ch => channels, identity, 
             cross_correlation=true, init_weight=kaiming_normal(gain=1), 
             use_bias=bias, init_bias=zeros32)
    )
    
    if soft_gating
        skip = SoftGating(channels, spatial_dims)
    else
        skip = Lux.NoOpLayer()    
    end
    
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
    if layer.skip != Lux.NoOpLayer()
        return y_mlp + y_skip, (mlp=st_mlp, skip=st_skip)
    else
        return y_mlp, (mlp=st_mlp, skip=st_skip)
    end
end
struct GridEmbedding <: Lux.AbstractLuxLayer
    boundaries::Vector{Vector{Float32}}
    spatial_dims::Int
end

# Backward compatible constructor - defaults to 2D
function GridEmbedding(grid_boundaries::Vector{Vector{Float32}} = [[0f0, 1f0], [0f0, 1f0]])
    spatial_dims = length(grid_boundaries)
    return GridEmbedding(grid_boundaries, spatial_dims)
end

# Explicit constructor with spatial_dims
function GridEmbedding(spatial_dims::Int, boundaries::Vector{Vector{Float32}})
    @assert length(boundaries) == spatial_dims "Number of boundaries must match spatial_dims"
    return GridEmbedding(boundaries, spatial_dims)
end

ChainRulesCore.@non_differentiable LinRange{Float32,Int64}(::Float32, ::Float32, ::Int64)

function Lux.initialparameters(rng::AbstractRNG, layer::GridEmbedding)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, layer::GridEmbedding)
    return NamedTuple()
end

function (layer::GridEmbedding)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    # Get dimensions
    dims = size(x)
    spatial_sizes = dims[1:layer.spatial_dims]
    batch_size = dims[end]
    
    # Create coordinate grids
    grid_arrays = ntuple(layer.spatial_dims) do i
        # Create 1D coordinate array
        if spatial_sizes[i] == 1
            coord_1d = [layer.boundaries[i][1]]
        else
            coord_1d = collect(LinRange(layer.boundaries[i]..., spatial_sizes[i]))
        end
        
        # Reshape to have dimension i active, others singleton
        # For 2D: i=1 -> (H, 1), i=2 -> (1, W)
        shape_before = ntuple(j -> j == i ? spatial_sizes[i] : 1, layer.spatial_dims)
        basis = reshape(coord_1d, shape_before)
        
        # Use repeat to expand to full size - this is what your meshgrid function does
        # For (H, 1): repeat along dimension 2 W times
        # For (1, W): repeat along dimension 1 H times
        repeat_dims = ntuple(j -> j == i ? 1 : spatial_sizes[j], layer.spatial_dims)
        grid = repeat(basis, repeat_dims...)
        
        # Add channel and batch dimensions
        grid_4d = reshape(grid, (spatial_sizes..., 1, 1))
        grid_batched = repeat(grid_4d, ntuple(_ -> 1, layer.spatial_dims + 1)..., batch_size)
        
        return grid_batched |> get_device(x)
    end
    
    return cat(x, grid_arrays..., dims=layer.spatial_dims + 1), st
end

ChainRulesCore.@non_differentiable (layer::GridEmbedding)(::Any)

function Base.show(io::IO, layer::GridEmbedding)
    dim_names = ["x", "y", "z", "w"]  # Extend as needed
    ranges = []
    
    for (i, bounds) in enumerate(layer.boundaries)
        dim_name = i <= length(dim_names) ? dim_names[i] : "dim$i"
        range_str = length(bounds) == 2 ? 
            "($(bounds[1]):$(bounds[2]))" :
            "[$(join(bounds, ", "))]"
        push!(ranges, "$dim_name: $range_str")
    end
    
    print(io, "GridEmbedding($(layer.spatial_dims)D: $(join(ranges, ", ")))")
end
"""
$(TYPEDSIGNATURES)

Adaptable FNO_Block that works with any spatial dimensionality.

# Arguments
- `channels`: Number of input/output channels
- `modes`: Tuple specifying number of low-frequency modes for the spectral convolution
- `spatial_dims`: Number of spatial dimensions (inferred from length of `modes`)
- `expansion_factor`: Factor controlling hidden dimension size in ChannelMLP (default: 2)
- `activation`: Nonlinear activation function (default: `NNlib.gelu`)
- `use_norm`: Whether to use instance normalization after spectral kernel (default: false)
- `skip`: Whether to use residual connection (default: true)
- `soft_gating`: Whether to use soft gating in ChannelMLP (default: false)
- `bias`: Whether to use bias in convolutions (default: false)
"""
struct FNO_Block{T,N} <: Lux.AbstractLuxContainerLayer{(:spectral_kernel, :channel_mlp)}
    spectral_kernel::SpectralKernel{T,N}
    channel_mlp::ChannelMLP
    channels::Int
    skip::Bool
    activation::Function
end

function FNO_Block(channels::Int, modes::NTuple{N,Int}; 
                   expansion_factor::Real=2, 
                   activation=NNlib.gelu, 
                   use_norm::Bool=false,
                   skip::Bool=true,
                   soft_gating::Bool=false,
                   bias::Bool=false) where N
    spatial_dims = N
    spectral_kernel = SpectralKernel(channels => channels, modes;
                                    use_norm=use_norm, bias=bias)
    channel_mlp = ChannelMLP(channels; spatial_dims=spatial_dims, 
                            expansion_factor=expansion_factor, 
                            activation=activation, 
                            soft_gating=soft_gating, 
                            bias=bias)
    return FNO_Block(spectral_kernel, channel_mlp, channels, skip, activation)
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
    # Apply spectral kernel
    x_spectral, st_spectral = fno_block.spectral_kernel(x, ps.spectral_kernel, st.spectral_kernel)
    # Apply channel MLP
    x_mlp, st_channel = fno_block.channel_mlp(x_spectral, ps.channel_mlp, st.channel_mlp)
    if fno_block.skip
        x_out = x + x_mlp
    else
        x_out = x_mlp
    end
    return x_out, (spectral_kernel=st_spectral, channel_mlp=st_channel)
end
