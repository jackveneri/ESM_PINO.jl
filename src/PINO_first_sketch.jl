using Lux, LuxCUDA, Random, Optimisers, Zygote, Statistics, MLUtils, ParameterSchedulers, Printf, CairoMakie, FFTW, NNlib, ChainRulesCore
using ComponentArrays, OnlineStats  # Helps with structured parameter handling

include("NewRepeatedLayer.jl")

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

function Lux.initialparameters(rng::AbstractRNG, layer::SpectralConv)
    in_ch, out_ch, modes = layer.in_channels, layer.out_channels, layer.modes
    init_std = sqrt(2f0 / (in_ch + out_ch))  # Standard Glorot-like scaling
    weight = init_std * randn(rng, ComplexF32, modes..., out_ch, in_ch)
    return (weight=weight,)
end

function Lux.initialstates(rng::AbstractRNG, layer::SpectralConv)
    return NamedTuple()  # No internal state needed
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

function (layer::SpectralConv)(x, ps, st::NamedTuple)
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
"""
struct SpectralKernel{P,F} <: Lux.AbstractLuxLayer
    spatial_conv::P  # 1x1 convolution
    spectral_conv::SpectralConv
    activation::F    # Activation function
end

function SpectralKernel(ch::Pair{<:Integer,<:Integer}, modes::NTuple{N,Integer}, activation=NNlib.gelu) where N
    in_ch, out_ch = ch
    conv = Conv((1,1), in_ch => out_ch, pad=0)
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

function (layer::SpectralKernel)(x, ps, st::NamedTuple)
    x_spatial, st_spatial = layer.spatial_conv(x, ps.spatial, st.spatial)
    x_spectral, st_spectral = layer.spectral_conv(x, ps.spectral, st.spectral)
    x_out = layer.activation.(x_spatial .+ x_spectral)
    return x_out, (spatial=st_spatial, spectral=st_spectral)
end

"""
    SoftGating(channels::Int)

A soft gating layer that applies per-channel multiplicative scaling.
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

function (layer::SoftGating)(x, ps, st::NamedTuple)
    return ps.weight .* x, st
end

"""
    ChannelMLP(channels::Int; expansion_factor=0.5, activation=gelu)

Implements a channel-wise MLP with a skip connection.
"""
struct ChannelMLP{M,S} <: Lux.AbstractLuxLayer
    mlp::M
    skip::S
end

function ChannelMLP(channels::Int; expansion_factor=0.5, activation=gelu)
    hidden_ch = Int(expansion_factor * channels)
    mlp = Chain(
        Conv((1, 1), channels => hidden_ch, activation),
        Conv((1, 1), hidden_ch => channels)
    )
    skip = SoftGating(channels)
    return ChannelMLP(mlp, skip)
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

function (layer::ChannelMLP)(x, ps, st)
    y_mlp, st_mlp = layer.mlp(x, ps.mlp, st.mlp)
    y_skip, st_skip = layer.skip(x, ps.skip, st.skip)
    return y_mlp .+ y_skip, (mlp=st_mlp, skip=st_skip)
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

Positional embedding that appends a normalized 2D coordinate grid to input data.
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

function (layer::GridEmbedding2D)(x, ps, st::NamedTuple)
    height, width, channels, batch_size = size(x)
    x_range = height == 1 ? [layer.boundaries_x[1]] : LinRange(layer.boundaries_x..., height)
    y_range = width == 1 ? [layer.boundaries_y[1]] : LinRange(layer.boundaries_y..., width)
    grid_x, grid_y = meshgrid(x_range, y_range)
    grid_x = reshape(grid_x, (height, width, 1, 1))
    grid_x = repeat(grid_x, outer = (1, 1, 1, batch_size)) 
    grid_y = reshape(grid_y, (height, width, 1, 1))
    grid_y = repeat(grid_y, outer = (1, 1, 1, batch_size)) 
    return cat(x, grid_x, grid_y, dims=length(size(x))-1), st
end

ChainRulesCore.@non_differentiable (layer::GridEmbedding2D)(::Any)

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

function (fno_block::FNO_Block)(x, ps, st::NamedTuple)
    x, st_spectral = fno_block.spectral_kernel(x, ps.spectral_kernel, st.spectral_kernel)
    x, st_channel = fno_block.channel_mlp(x, ps.channel_mlp, st.channel_mlp)
    return x, (spectral_kernel=st_spectral, channel_mlp=st_channel)
end

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
    embedding = nothing
    if positional_embedding == "grid"
        embedding = GridEmbedding2D()
        in_channels += n_dim
    end

    lifting = Chain(
        Conv((1, 1), in_channels => Int(lifting_channel_ratio * hidden_channels), activation),
        Conv((1, 1), Int(lifting_channel_ratio * hidden_channels) => hidden_channels, activation),
    )

    projection = Chain(
        Conv((1, 1), hidden_channels => Int(projection_channel_ratio * hidden_channels), activation),
        Conv((1, 1), Int(projection_channel_ratio * hidden_channels) => out_channels, activation),
    )

    fno_blocks = NewRepeatedLayer(FNO_Block(hidden_channels, n_modes; expansion_factor=channel_mlp_expansion, activation=activation), n_layers)

    return FourierNeuralOperator(embedding, lifting, fno_blocks, projection)
end

function Lux.initialparameters(rng::AbstractRNG, layer::FourierNeuralOperator)
    ps_embedding = isnothing(layer.embedding) ? NamedTuple() : Lux.initialparameters(rng, layer.embedding)
    ps_lifting = Lux.initialparameters(rng, layer.lifting)
    ps_fno_blocks = Lux.initialparameters(rng, layer.fno_blocks)
    ps_projection = Lux.initialparameters(rng, layer.projection)
    return (
        embedding=ps_embedding,
        lifting=ps_lifting,
        fno_blocks=ps_fno_blocks,
        projection=ps_projection,
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::FourierNeuralOperator)
    st_embedding = isnothing(layer.embedding) ? NamedTuple() : Lux.initialstates(rng, layer.embedding)
    st_lifting = Lux.initialstates(rng, layer.lifting)
    st_fno_blocks = Lux.initialstates(rng, layer.fno_blocks)
    st_projection = Lux.initialstates(rng, layer.projection)
    return (
        embedding=st_embedding,
        lifting=st_lifting,
        fno_blocks=st_fno_blocks,
        projection=st_projection,
    )
end

function (layer::FourierNeuralOperator)(x, ps, st::NamedTuple)
    if !isnothing(layer.embedding)
        x, st_embedding = layer.embedding(x, ps.embedding, st.embedding)
    else
        st_embedding = st.embedding
    end

    x, st_lifting = layer.lifting(x, ps.lifting, st.lifting)
    x, st_fno_blocks = layer.fno_blocks(x, ps.fno_blocks, st.fno_blocks)
    x, st_projection = layer.projection(x, ps.projection, st.projection)

    return x, (embedding=st_embedding, lifting=st_lifting, fno_blocks=st_fno_blocks, projection=st_projection)
end

# Example usage
const gdev = gpu_device()
const cdev = cpu_device()
fno = FourierNeuralOperator(in_channels=1, out_channels=1, hidden_channels=8, n_modes=(16, 16))
rng = Random.default_rng()
ps, st = Lux.initialparameters(rng, fno), Lux.initialstates(rng, fno)
#expected shape of the input is (spatial..., in_channels, batch)
#note that spatial must be >= n_modes for the spectral convolution to work
x = randn(Float32, 16, 16, 1, 128)
y, _ = fno(x, ps, st)
#output shape is (spatial..., out_channels, batch)
println("Output size: ", size(y))

function mse_loss_function(u::StatefulLuxLayer, target::AbstractArray, xt::AbstractArray)
    return MSELoss()(u(xt), target)
end

function loss_function(model, ps, st,(xt, target_data))
    u_net = StatefulLuxLayer{true}(model, ps, st)
    data_loss = mse_loss_function(u_net, target_data, xt)
    return (data_loss,
        (st),
        (; data_loss)
    )
end

target = randn(Float32, 16, 16, 1, 128)

loss_function(fno, ps, st, (x, target))

function train_model(x, target; seed::Int=0,
    maxiters::Int=1000, hidden_channels::Int=8)
    rng = Random.default_rng(seed)
    fno = FourierNeuralOperator(in_channels=1, out_channels=1, hidden_channels=hidden_channels, n_modes=(16, 16))
    ps, st = Lux.setup(rng, fno) 
    
    dataloader = DataLoader((x, target); batchsize=32, shuffle=true) 

    train_state = Training.TrainState(fno, ps, st, Adam(0.001f0))

    loss_tracker, = ntuple(_ -> Lag(Float32, 32),1)
    iter = 1
    for (xt, target_data) in Iterators.cycle(dataloader)
        -, loss, stats, train_state = Training.single_train_step!(AutoZygote(), loss_function, (xt, target_data), train_state)
        
        fit!(loss_tracker, loss)
        
        mean_loss = mean(OnlineStats.value(loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))
        
        if iter % 1 == 0 || iter == maxiters
            @printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f)\n" iter maxiters loss mean_loss
        end
        
        iter += 1
        
        if iter > maxiters
            break
        end

    end
    return StatefulLuxLayer{true}(fno, cdev(train_state.parameters), cdev(train_state.states))
end

trained_model = train_model(x, target)
trained_u = Lux.testmode(StatefulLuxLayer{true}(trained_model.model, trained_model.ps, trained_model.st))