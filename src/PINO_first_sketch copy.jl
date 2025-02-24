using Lux, LuxCUDA, Random, Optimisers, Zygote, Statistics, MLUtils, ParameterSchedulers, Printf, CairoMakie, FFTW, NNlib, ChainRulesCore
using ComponentArrays  # Helps with structured parameter handling

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
struct ChannelMLP{C,E,A} <: Lux.AbstractLuxLayer
    channels::C
    expansion_factor::E
    activation::A
end

ChannelMLP(channels::Int; expansion_factor=0.5, activation=gelu) =
    ChannelMLP(channels, expansion_factor, activation)

function Lux.initialparameters(rng::AbstractRNG, layer::ChannelMLP)
    in_ch = layer.channels
    hidden_ch = Int(layer.expansion_factor * in_ch)
    mlp = Chain(
        Conv((1, 1), in_ch => hidden_ch, layer.activation),
        Conv((1, 1), hidden_ch => in_ch)
    )
    skip = SoftGating(in_ch)
    ps_mlp = Lux.initialparameters(rng, mlp)
    ps_skip = Lux.initialparameters(rng, skip)
    return (mlp=ps_mlp, skip=ps_skip)
end

function Lux.initialstates(rng::AbstractRNG, layer::ChannelMLP)
    in_ch = layer.channels
    hidden_ch = Int(layer.expansion_factor * in_ch)
    mlp = Chain(
        Conv((1, 1), in_ch => hidden_ch, layer.activation),
        Conv((1, 1), hidden_ch => in_ch)
    )
    skip = SoftGating(in_ch)
    st_mlp = Lux.initialstates(rng, mlp)
    st_skip = Lux.initialstates(rng, skip)
    return (mlp=st_mlp, skip=st_skip)
end

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
    ps_embedding = isnothing(layer.embedding) ? nothing : Lux.initialparameters(rng, layer.embedding)
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
    st_embedding = isnothing(layer.embedding) ? nothing : Lux.initialstates(rng, layer.embedding)
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
#=
fno = FourierNeuralOperator(in_channels=16, out_channels=1, hidden_channels=32, n_modes=(16, 16))
rng = Random.default_rng()
ps, st = Lux.initialparameters(rng, fno), Lux.initialstates(rng, fno)
x = randn(Float32, 32, 32, 16, 4)
y, _ = fno(x, ps, st)
println("Output size: ", size(y))
=#

using MAT

"""

Burgers' equation dataset from
[fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)

mapping between initial conditions to the solutions at the last point of time \
evolution in some function space.

u(x,0) -> u(x, time_end):

    * `a`: initial conditions u(x,0)
    * `u`: solutions u(x,t_end)
"""
filepath2 = "C:/Users/jackv/Downloads/Burgers_R10/burgers_data_R10.mat"

const N = 2048
const Δsamples = 2^3
const grid_size = div(2^13, Δsamples)
const T = Float32

file = matopen(filepath2)
x_data = reshape(T.(collect(read(file, "a")[1:N, 1:Δsamples:end])), 1, :, 1, N)
y_data = reshape(T.(collect(read(file, "u")[1:N, 1:Δsamples:end])), 1, :, 1, N)
#x = T.(collect(read(file, "a")[1:end, 1:end]))
#x_data = reshape(x, 1, 8192, 1, 2048)
#y = T.(collect(read(file, "u")[1:end, 1:end]))
#y_data = reshape(y, 1, 8192, 1, 2048)
close(file)

grid = reshape(T.(collect(range(0, 1; length=grid_size))'), :, grid_size , 1, 1)
time_grid = reshape(T.(collect(range(0, 1; length=2))'), 1, 2, 1, 1) 

const cdev = cpu_device()
const gdev = gpu_device()

#call FourierNeuralOperator here with the correct parameters
fno = FourierNeuralOperator(in_channels=1, out_channels=1, hidden_channels=8, n_modes=(1, 8), n_layers=2)
ps, st = Lux.setup(Random.default_rng(), fno) |> gdev;

x_data_dev = x_data |> gdev
y_data_dev = y_data |> gdev
grid_dev = grid |> gdev
time_grid_dev = time_grid |> gdev

# Define the Burgers' equation residual
function burgers_residual(model, x, t, ν)
    u = first(model((x, t), ps, st))
    u_x = gradient(x -> (model(x, t)), x)[1]  #define u(x,t)
    u_t = gradient(t -> (model(x, t)), t)[1]
    u_xx = gradient(x -> gradient(x -> model(x, t), x)[1], x)[1]
    residual = u_t + u .* u_x - ν * u_xx
    return residual
end

# Define the physics-informed loss
function physics_informed_loss(model, ps, st, x, t, ν)
    residual = burgers_residual(model, x, t, ν)
    return mean(residual .^ 2)
end

# Define the combined loss function
function combined_loss(model, ps, st, ((x, t), u_true), ν)
    # Data-driven loss
    u_pred, st = model(x, ps, st)
    loss = MSELoss()
    data_loss = loss(u_pred, u_true)
    
    # Physics-informed loss
    physics_loss = physics_informed_loss(model, ps, st, x, t, ν)
    
    # Weighted combination
    α = 0.5  # Weight for physics loss (can be tuned)
    return (1 - α) * data_loss + α * physics_loss, st
end

# Training loop
function train_model!(model, ps, st, data, ν; epochs=100)
    opt = Adam(0.0001f0)
    train_state = Training.TrainState(model, ps, st, opt)

    for epoch in 1:epochs
        # Compute loss and gradients
        loss, grad = Zygote.withgradient(ps -> combined_loss(model, ps, st, data, ν), train_state.parameters)
        
        # Update model parameters
        train_state = Training.update!(train_state, grad[1])
        
        # Print loss
        if epoch % 25 == 1 || epoch == epochs
            @printf("Epoch %d: loss = %.6e\n", epoch, loss)
        end
    end

    return train_state.parameters, train_state.states
end

# Train the model
ν = 0.01f0  # Viscosity coefficient
ps_trained, st_trained = train_model!(fno, ps, st, ((x_data_dev, grid_dev), y_data_dev), ν)

# Visualize the results
#pred = first(fno((x_data_dev, grid_dev), ps_trained, st_trained)) |> cdev
pred, _ = fno((x_data_dev, grid_dev), ps_trained, st_trained) 
pred = pred |> cdev  

fig = Figure(; size=(1024, 1024))
axs = [Axis(fig[i, j]) for i in 1:4, j in 1:4]
for i in 1:4, j in 1:4
    idx = i + (j - 1) * 4
    ax = axs[i, j]
    l1 = lines!(ax, vec(grid), pred[idx, :, 1])
    l2 = lines!(ax, vec(grid), y_data[idx, :, 1])

    i == 4 && (ax.xlabel = "x")
    j == 1 && (ax.ylabel = "u(x)")

    if i == 1 && j == 1
        axislegend(ax, [l1, l2], ["Predictions", "Ground Truth"])
    end
end
linkaxes!(axs...)
fig[0, :] = Label(fig, "Burgers Equation using PINO"; tellwidth=false, font=:bold)
fig
