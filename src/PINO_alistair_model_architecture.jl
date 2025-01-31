using Pkg
Pkg.activate(".")
using Lux, LuxCUDA, Random, Optimisers, Zygote, Statistics, MLUtils, ParameterSchedulers, Printf, CairoMakie, FFTW, NNlib, ChainRulesCore
using ComponentArrays  # Helps with structured parameter handling

include("NamedChain.jl")
"""
    SpectralConv{N}

Spectral convolution layer for Fourier Neural Operator in Lux.jl.
Expects input in (spatial..., channel, batch) format.
"""
struct SpectralConv{N} <: Lux.AbstractLuxLayer
    in_channels::Int
    out_channels::Int
    modes::NTuple{N,Int}
end

function SpectralConv(in_channels::Integer, out_channels::Integer, modes::NTuple{N,Integer}) where N
    return SpectralConv{ComplexF32, N}(in_channels, out_channels, modes)
end

function Lux.setup(rng::AbstractRNG, layer::SpectralConv)
    in_ch, out_ch, modes = layer.in_channels, layer.out_channels, layer.modes
    init_std = sqrt(2f0 / (in_ch + out_ch))  # Standard Glorot-like scaling

    weight = init_std * randn(rng, ComplexF32, modes..., out_ch, in_ch)

    parameters = (weight=weight,)
    state = NamedTuple()  # No internal state needed

    return parameters, state
end

function low_pass(x_ft, modes)
    return view(x_ft, map(d -> 1:d, modes)..., :, :)
end

function apply_pattern(x_tr::AbstractArray{T,N}, weights::AbstractArray{T}) where {T,N}
    x_expanded = reshape(x_tr, size(x_tr)[1:N-2]..., 1, size(x_tr)[N-1:N]...)
    return dropdims(sum(weights .* x_expanded, dims=N), dims=N)
end

function expand_pad_dims(pad_dims::Dims{N}) where {N}
    return ntuple(i -> isodd(i) ? 0 : pad_dims[i ÷ 2], 2N)
end
@non_differentiable expand_pad_dims(::Any)

function (layer::SpectralConv)( x, ps, st::NamedTuple)
    x_ft = fft(x, 1:ndims(x)-2)  # Apply Fourier transform on spatial dimensions
    x_tr = low_pass(x_ft, layer.modes)  # Truncate high frequencies
    x_p = apply_pattern(x_tr, ps.weight)  # Apply learned spectral filters
    
    # Pad back to original shape
    pad_dims = size(x_ft)[1:(end - 2)] .- size(x_p)[1:(end - 2)]
    x_padded = NNlib.pad_constant(x_p, expand_pad_dims(pad_dims), false; dims=ntuple(identity, ndims(x_p) - 2))
    
    # Inverse Fourier transform to return to spatial domain
    return real(ifft(x_padded, 1:ndims(x_padded)-2)), st
end
#example usage
#=
# Define layer
layer = SpectralConv(10 , 20, (16, 16))

# Initialize parameters and state
rng = Random.default_rng()
ps, st = Lux.setup(rng, layer)

# Example input (Batch size = 5, 10 input channels)
x = randn(Float32, 64, 64, 10, 5)  # (spatial_x, spatial_y, in_channels, batch)

# Forward pass
y, st_new = Lux.apply(layer, x, ps, st)
println("Output size: ", size(y))  # Should match (64, 64, out_channels, batch)
=#


"""
    SpectralKernel{P,F}

Combines a SpectralConv layer with a 1x1 convolution in parallel, followed by an activation function.
Expects input in (spatial..., channel, batch) format.
"""


using Lux: Conv

struct SpectralKernel{P,F} <: Lux.AbstractLuxLayer
    spatial_conv::P  # 1x1 convolution
    spectral_conv::SpectralConv
    activation::F    # Activation function
end

function SpectralKernel(ch::Pair{<:Integer,<:Integer}, modes::NTuple{N,Integer}, activation=NNlib.gelu) where N
    in_ch, out_ch = ch
    
    # Create 1x1 convolution
    conv = Conv((1,1), in_ch => out_ch, pad=0)
    
    # Create spectral convolution
    spectral = SpectralConv(in_ch, out_ch, modes)
    
    return SpectralKernel(conv, spectral, activation)
end

function Lux.setup(rng::AbstractRNG, layer::SpectralKernel)
    ps_conv, st_conv = Lux.setup(rng, layer.spatial_conv)
    ps_spectral, st_spectral = Lux.setup(rng, layer.spectral_conv)

    # Combine parameters and states
    parameters = (spatial=ps_conv, spectral=ps_spectral)
    state = (spatial=st_conv, spectral=st_spectral)

    return parameters, state
end

function (layer::SpectralKernel)(x, ps, st::NamedTuple)

    # Apply both convolutions
    x_spatial, st_spatial = (layer.spatial_conv)(x, ps.spatial, st.spatial)
    x_spectral, st_spectral = (layer.spectral_conv)(x, ps.spectral, st.spectral)

    # Combine results and apply activation
    x_out = layer.activation.(x_spatial .+ x_spectral)

    return x_out, (spatial=st_spatial, spectral=st_spectral)
end
#example usage
#=
# Define layer
layer = SpectralKernel(10 => 20, (16, 16))

# Initialize parameters and state
rng = Random.default_rng()
ps, st = Lux.setup(rng, layer)

# Example input (Batch size = 5, 10 input channels)
x = randn(Float32, 64, 64, 10, 5)  # (spatial_x, spatial_y, in_channels, batch)

# Forward pass
y, st_new = Lux.apply(layer, x, ps, st)
println("Output size: ", size(y))  # Should match (64, 64, out_channels, batch)
=#

"""
    SoftGating(channels::Int)

A soft gating layer that applies per-channel multiplicative scaling.

# Arguments
- `channels::Int`: Number of input/output channels.

# Behavior
- Learns a gating weight for each channel, initialized to ones.
- Applies element-wise multiplication to input.
"""
struct SoftGating <: Lux.AbstractLuxLayer
    channels::Int
end

# Initialize parameters and state
function Lux.setup(rng::AbstractRNG, layer::SoftGating)
    weight = ones(Float32, 1, 1, layer.channels, 1)  # Shape: (1, 1, channels, 1)
    parameters = (weight=weight,)
    state = NamedTuple()  # No state needed
    return parameters, state
end

# Forward pass
function (layer::SoftGating)(x, ps, st::NamedTuple)
    return ps.weight .* x, st  # Apply gating element-wise
end

#example usage
#=
# Define soft gating layer
layer = SoftGating(10)

# Initialize parameters and state
rng = Random.default_rng()
ps, st = Lux.setup(rng, layer)

# Example input (Batch size = 5, 10 channels)
x = randn(Float32, 64, 64, 10, 5)  # (spatial_x, spatial_y, channels, batch)

# Forward pass
y, st_new = Lux.apply(layer, x, ps, st)
println("Output size: ", size(y))  # Should match (64, 64, 10, 5)
=#
"""
    ChannelMLP(channels::Int; expansion_factor=0.5, activation=gelu)

Implements a channel-wise MLP with a skip connection.

# Arguments
- `channels::Int`: Number of input/output channels.
- `expansion_factor::Float`: Expansion factor for the hidden layer.
- `activation::Function`: Activation function for the hidden layer (default: `gelu`).

# Behavior
- Uses 1x1 convolutions for channel mixing.
- Applies a skip connection using a `SoftGating` layer.
"""
struct ChannelMLP{C,E,A} <: Lux.AbstractLuxLayer
    channels::C
    expansion_factor::E
    activation::A
end

# Provide a default constructor with keyword arguments
ChannelMLP(channels::Int; expansion_factor=0.5, activation=gelu) =
    ChannelMLP(channels, expansion_factor, activation)

# Initialize parameters and state
function Lux.setup(rng::AbstractRNG, layer::ChannelMLP)
    in_ch = layer.channels
    hidden_ch = Int(layer.expansion_factor * in_ch)
    
    # Define MLP structure
    mlp = Chain(
        Conv((1, 1), in_ch => hidden_ch, layer.activation),
        Conv((1, 1), hidden_ch => in_ch)
    )
    
    # Define SoftGating for skip connection
    skip = SoftGating(in_ch)

    # Initialize parameters and states for both submodules
    ps_mlp, st_mlp = Lux.setup(rng, mlp)
    ps_skip, st_skip = Lux.setup(rng, skip)
    
    # Merge parameters and states into single structures
    parameters = (mlp=ps_mlp, skip=ps_skip)
    state = (mlp=st_mlp, skip=st_skip)
    
    return parameters, state
end

# Forward pass
function (layer::ChannelMLP)(x, ps, st::NamedTuple)
    y_mlp, st_mlp = Lux.apply(
        Chain(
            Conv((1,1), layer.channels => Int(layer.expansion_factor * layer.channels), layer.activation),
            Conv((1,1), Int(layer.expansion_factor * layer.channels) => layer.channels)
        ),
        x, ps.mlp, st.mlp::NamedTuple
    )
    
    y_skip, st_skip = Lux.apply(SoftGating(layer.channels), x, ps.skip, st.skip::NamedTuple)
    
    return y_mlp + y_skip, (mlp=st_mlp, skip=st_skip)
end
#example usage
#=
# Define layer
layer = ChannelMLP(16)

# Initialize parameters and state
rng = Random.default_rng()
ps, st = Lux.setup(rng, layer)

# Example input (Batch size = 4, 16 channels)
x = randn(Float32, 32, 32, 16, 4)  # (spatial_x, spatial_y, channels, batch)

# Forward pass
y, st_new = Lux.apply(layer, x, ps, st)
println("Output size: ", size(y))  # Should match (32, 32, 16, 4)
=#
"""
meshgrid(x, y)
Generates a 2D meshgrid from vectors `x` and `y`.
"""
function meshgrid(x, y)
    return (repeat(x, 1, length(y)), repeat(y', length(x), 1))
end

"""
    GridEmbedding2D(grid_boundaries=[[0f0, 1f0], [0f0, 1f0]])

Positional embedding that appends a normalized 2D coordinate grid to input data.

# Arguments
- `grid_boundaries::Vector{Vector{Float32}}`: 
  A 2-element vector specifying `[x_boundaries, y_boundaries]`.

# Behavior
- Generates a positional embedding grid based on spatial dimensions.
- Appends the `x` and `y` coordinate grids to the input tensor.
"""
struct GridEmbedding2D <: Lux.AbstractLuxLayer
    boundaries_x::Vector{Float32}
    boundaries_y::Vector{Float32}
end

# Constructor with default boundaries
GridEmbedding2D(grid_boundaries::Vector{Vector{Float32}} = [[0f0, 1f0], [0f0, 1f0]]) =
    GridEmbedding2D(grid_boundaries[1], grid_boundaries[2])

# Prevent differentiation through LinRange construction
ChainRulesCore.@non_differentiable LinRange{Float32,Int64}(::Float32, ::Float32, ::Int64)

# Setup function (no trainable parameters, just state)
function Lux.setup(rng::AbstractRNG, layer::GridEmbedding2D)
    return NamedTuple(), NamedTuple()  # No trainable parameters or state
end

# Forward pass
function Lux.apply(layer::GridEmbedding2D, x, ps, st::NamedTuple)
    height, width, channels, batch_size = size(x)

    # Create grid
    x_range = LinRange(layer.boundaries_x..., height)
    y_range = LinRange(layer.boundaries_y..., width)
    grid_x, grid_y = meshgrid(x_range, y_range)

    # Reshape to match input format
    grid_x = reshape(grid_x, (height, width, 1, 1))
    grid_x = repeat(grid_x, outer = (1, 1, 1, batch_size)) 

    grid_y = reshape(grid_y, (height, width, 1, 1))
    grid_y = repeat(grid_y, outer = (1, 1, 1, batch_size)) 

    # Concatenate along channel dimension
    return cat(x, grid_x, grid_y, dims=length(size(x))-1), st
end

# Prevent differentiation through the entire layer
ChainRulesCore.@non_differentiable (layer::GridEmbedding2D)(::Any)


#example usage
#=
# Initialize Lux layer
embedding = GridEmbedding2D()

# Set up parameters and state
ps, st = Lux.setup(Random.default_rng(), embedding)

# Create dummy input tensor: (height=32, width=32, channels=16, batch_size=4)
x = randn(Float32, 32, 32, 16, 4)

# Apply the layer
y, _ = Lux.apply(embedding, x, ps, st)

println("Output size: ", size(y))  # Should be (32, 32, 18, 4)
=#
struct FNO_Block{N} <: Lux.AbstractLuxLayer
    spectral_kernel :: SpectralKernel
    channel_mlp :: ChannelMLP
    channels :: Int
    modes :: NTuple{N, Int}
end

function FNO_Block(channels, modes; expansion_factor=0.5, activation=NNlib.gelu)
    spectral_kernel = SpectralKernel(channels => channels, modes, activation)
    channel_mlp = ChannelMLP(channels, expansion_factor=expansion_factor, activation=activation)
    return FNO_Block(spectral_kernel, channel_mlp, channels, modes)
end

function Lux.setup(rng::AbstractRNG, block::FNO_Block)
    ps_spectral, st_spectral = Lux.setup(rng, block.spectral_kernel)
    ps_channel, st_channel = Lux.setup(rng, block.channel_mlp)
    return (spectral_kernel=ps_spectral, channel_mlp=ps_channel), (spectral_kernel=st_spectral, channel_mlp=st_channel)
end

function (fno_block::FNO_Block)(x, ps, st::NamedTuple)
    display(st)
    x, st_spectral = fno_block.spectral_kernel(x, ps.spectral_kernel, st.spectral_kernel)
    x, st_channel = fno_block.channel_mlp(x, ps.channel_mlp, st.channel_mlp)
    return x, (spectral_kernel=st_spectral, channel_mlp=st_channel)
end


#example usage
#=
model = FNO_Block(16, (16, 16))

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

x = randn(rng, Float32, 16, 16, 16, 4)

Lux.apply(model, x, ps, st) 
=#
"""
    FourierNeuralOperator

A layer that combines the Fourier Neural Operator (FNO) with positional embeddings, spectral kernels, and channel MLPs.
"""

struct FourierNeuralOperator <: Lux.AbstractLuxContainerLayer{(:embedding, :lifting, :fno_blocks, :projection)}
    embedding
    lifting
    fno_blocks 
    projection
end

function FourierNeuralOperator(;
    in_channels,
    out_channels,
    hidden_channels=32,
    n_modes=(16, 16),
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=0.5,
    activation=gelu,
    positional_embedding="grid",
)
    n_dim = length(n_modes)

    # Apply positional embedding if specified
    embedding = nothing
    if positional_embedding == "grid"
        embedding = GridEmbedding2D()
        in_channels += n_dim  # Add the number of grid dimensions to input channels
    end

    # Lifting block
    lifting = Chain(
        Conv((1, 1), in_channels => Int(lifting_channel_ratio * hidden_channels), activation),
        Conv((1, 1), Int(lifting_channel_ratio * hidden_channels) => hidden_channels, activation),
    )

    # Projection block
    projection = Chain(
        Conv((1, 1), hidden_channels => Int(projection_channel_ratio * hidden_channels), activation),
        Conv((1, 1), Int(projection_channel_ratio * hidden_channels) => out_channels, activation),
    )

    # Spectral Kernel and Channel MLP blocks
    fno_blocks = NewRepeatedLayer(FNO_Block(hidden_channels, n_modes; expansion_factor=channel_mlp_expansion, activation=activation), n_layers)

    # Combine all parts into a single model
    return FourierNeuralOperator(embedding, lifting, fno_blocks, projection)
end

function Lux.setup(rng::AbstractRNG, layer::FourierNeuralOperator)
    # Set up each component of the model
    ps_embedding, st_embedding = isnothing(layer.embedding) ? (nothing, nothing) : Lux.setup(rng, layer.embedding)
    ps_lifting, st_lifting = Lux.setup(rng, layer.lifting)
    ps_fno_blocks, st_fno_blocks = Lux.setup(rng, layer.fno_blocks)
    ps_projection, st_projection = Lux.setup(rng, layer.projection)

    # Combine parameters and states
    parameters = (
        embedding=ps_embedding,
        lifting=ps_lifting,
        fno_blocks=ps_fno_blocks,
        projection=ps_projection,
    )
    state = (
        embedding=st_embedding,
        lifting=st_lifting,
        fno_blocks=st_fno_blocks,
        projection=st_projection,
    )

    return parameters, state
end

function Lux.apply(layer::FourierNeuralOperator, x, ps, st::NamedTuple)
    # Apply embedding
    if !isnothing(layer.embedding)
        x, st_embedding = Lux.apply(layer.embedding, x, ps.embedding, st.embedding)
    else
        st_embedding = st.embedding
    end

    # Apply lifting
    x, st_lifting = Lux.apply(layer.lifting, x, ps.lifting, st.lifting)

    # Apply FNO blocks
    x, st_fno_blocks = Lux.apply(layer.fno_blocks, x, ps.fno_blocks, st.fno_blocks)

    # Apply projection
    x, st_projection = Lux.apply(layer.projection, x, ps.projection, st.projection)

    # Return output and updated state
    return x, (embedding=st_embedding, lifting=st_lifting, fno_blocks=st_fno_blocks, projection=st_projection)
end

# Initialize the Fourier Neural Operator with the desired parameters
fno = FourierNeuralOperator(in_channels=16, out_channels=1, hidden_channels=32, n_modes=(16, 16))

# Set up parameters and state
rng = Random.default_rng()
ps, st = Lux.setup(rng, fno)

# Example input tensor (height=32, width=32, channels=16, batch_size=4)
x = randn(Float32, 32, 32, 16, 4)

# Apply the FNO model
y, _ = Lux.apply(fno, x, ps, st)

println("Output size: ", size(y))  # Should match the output shape