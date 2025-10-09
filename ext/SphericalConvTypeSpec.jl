using Lux, Random, QG3, NNlib
"""
    SphericalConv(hidden_channels::Int, ggsh::GaussianGridtoSHTransform, shgg::SHtoGaussianGridTransform; modes::Int=ggsh.output_size[1], zsk::Bool=false)

Spherical convolution layer for functions on the sphere using spherical harmonics.  
Transforms data from Gaussian grid → spherical harmonics, applies learned weights, and transforms back.

# Arguments
- `hidden_channels::Int`: Number of input/output channels.
- `ggsh::GaussianGridtoSHTransform`: Transformation from Gaussian grid to spherical harmonics.
- `shgg::SHtoGaussianGridTransform`: Transformation from spherical harmonics back to Gaussian grid.
- `modes::Int=ggsh.output_size[1]`: Maximum number of spherical harmonic modes to use. If higher than `ggsh.output_size[1]`, it is truncated with a warning.
- `zsk::Bool=false`: If true, uses Zonal Symmetric Kernels (ZSK), reducing the number of free weights. It follows follows **Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere**, https://arxiv.org/abs/2204.06408.


# Returns
- `SphericalConv`: A Lux-compatible layer operating on 4D arrays `[lat, lon, channels, batch]`.

# Details
- Input is permuted internally to `[channels, lat, lon, batch]` for computation.
- Uses `ps.weight` for element-wise multiplication in spherical harmonic space.
- Supports padding to match the original spherical grid dimensions.
- Compatible with GPU and CPU (controlled externally in ggsh, shgg constructor).

# Example
```julia
using Random, Lux, QG3, NNlib

# Load precomputed spherical model parameters
qg3ppars = QG3.load_precomputed_params()[2]

# Create transforms
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)

# Initialize layer
layer = SphericalConv(32, ggsh, shgg, 30; zsk=true)

# Generate random input [lat, lon, channels, batch]
x = rand(Float32, 32, 64, 32, 1)

# Setup parameters and state
rng = Random.default_rng(0)
ps, st = Lux.setup(rng, layer)

# Forward pass
y, st = layer(x, ps, st)

# Compute gradient
using Zygote
gr = Zygote.gradient(ps -> sum(layer(x, ps, st)[1]), ps)
"""
struct SphericalConv{G,S} <: Lux.AbstractLuxLayer 
    hidden_channels::Int
    modes::Int
    ggsh::G  # GaussianGridtoSHTransform
    shgg::S  # SHtoGaussianGridTransform
    zsk::Bool  # Whether to use Zonal Symmetric Kernels (ZSK)
    # Inner constructor to enforce modes ≤ pars.L
    function SphericalConv(
            hidden_channels::Int,
            ggsh::QG3.GaussianGridtoSHTransform,
            shgg::QG3.SHtoGaussianGridTransform,
            modes::Int = ggsh.output_size[1];
            zsk::Bool = false
        )
        # Correct modes if necessary
        corrected_modes = min(modes, ggsh.output_size[1])
        if modes != corrected_modes
            @warn "modes ($modes) exceeds ggsh.output_size[1] ($(ggsh.output_size[1])). Setting modes = $(ggsh.output_size[1])."
        end
        # Create the struct with the corrected value
        new{QG3.GaussianGridtoSHTransform,QG3.SHtoGaussianGridTransform}(hidden_channels, corrected_modes, ggsh, shgg, zsk)
    end
end
"""
    SphericalConv(pars::QG3.QG3ModelParameters{T}, hidden_channels::Int; modes::Int=pars.L, batch_size::Int=1, gpu::Bool=true, zsk::Bool=false) where T

Construct a spherical convolution layer using precomputed model parameters.

# Arguments
- `pars::QG3.QG3ModelParameters{T}`: Model parameters defining the spherical grid resolution and maximum spherical harmonic degree `L`.
- `hidden_channels::Int`: Number of input/output channels.
- `modes::Int=pars.L`: Maximum number of spherical harmonic modes to use. If higher than `pars.L`, it is truncated with a warning.
- `batch_size::Int=1`: Number of samples in a batch (used for internal transforms).
- `gpu::Bool=true`: If true, computations are moved to GPU using `QG3.gpuon()`.
- `zsk::Bool=false`: If true, uses Zonal Symmetric Kernels (ZSK), reducing the number of free weights. ZSK enforces rotational symmetry along longitude and follows **Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere**, https://arxiv.org/abs/2204.06408.

# Returns
- `SphericalConv`: A Lux-compatible layer operating on 4D arrays `[lat, lon, channels, batch]`.

# Details
- Internally constructs `GaussianGridtoSHTransform` and `SHtoGaussianGridTransform` objects for the given `pars` and `hidden_channels`.
- Corrects the requested `modes` to not exceed `pars.L`.
- Supports both CPU and GPU computation.
- Zonal Symmetric Kernels (ZSK) reduce the number of learnable parameters by enforcing symmetry along longitude, improving stability for spherical dynamics.

# Example
```julia
using Random, Lux, QG3, NNlib

# Load precomputed parameters
qg3ppars = QG3.load_precomputed_params()[2]

# Initialize spherical convolution layer
layer = SphericalConv(qg3ppars, 32; modes=30, batch_size=1, gpu=false, zsk=true)

# Generate random input [lat, lon, channels, batch]
x = rand(Float32, 64, 128, 32, 1)

# Setup parameters and state
rng = Random.default_rng(0)
ps, st = Lux.setup(rng, layer)

# Forward pass
y, st = layer(x, ps, st)

# Compute gradient
using Zygote
gr = Zygote.gradient(ps -> sum(layer(x, ps, st)[1]), ps)
"""
function SphericalConv(
        pars::QG3.QG3ModelParameters{T},
        hidden_channels::Int;
        modes::Int = pars.L,
        batch_size::Int = 1,
        gpu::Bool = true,
        zsk ::Bool = false
    ) where T
    # Correct modes upfront (before inner constructor)
    corrected_modes = min(modes, pars.L)
    if modes != corrected_modes
        @warn "modes ($modes) exceeds pars.L ($(pars.L)). Setting modes = $(pars.L)."
    end
    # Proceed with construction
    if gpu
        QG3.gpuon()
    else
        QG3.gpuoff()
    end

    ggsh = QG3.GaussianGridtoSHTransform(pars, hidden_channels; N_batch=batch_size)
    shgg = QG3.SHtoGaussianGridTransform(pars, hidden_channels; N_batch=batch_size)
    SphericalConv(hidden_channels, ggsh, shgg, corrected_modes, zsk=zsk)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SphericalConv{G,S}) where {G,S}
    init_std = typeof(layer.ggsh).parameters[1](sqrt(2 / layer.hidden_channels))
    # Initialize 2D weights for spatial pattern (L × M)
    if layer.zsk == true
        weight = init_std * randn(rng, typeof(layer.ggsh).parameters[1], 1, layer.modes, 1, 1)
    else
        weight = init_std * randn(rng, typeof(layer.ggsh).parameters[1], 1, layer.modes, 2 * layer.modes - 1, 1)
    end
    return (weight=weight,)
end

Lux.initialstates(rng::AbstractRNG, layer::SphericalConv) = NamedTuple()

function (layer::SphericalConv{G,S})(x::AbstractArray{T,4}, ps::NamedTuple, st::NamedTuple) where {G,S,T} 
    @assert T == typeof(layer.ggsh).parameters[1] "Input type $T does not match model parameter type $(typeof(layer.ggsh).parameters[1]))"
    @views begin
    x_perm = permutedims(x, (3, 1, 2, 4))  # [channels, lat, lon, batch]
    x_tr = QG3.transform_SH(x_perm, layer.ggsh)
    # Type-stable element-wise multiplication with broadcast
    x_p = ps.weight .* x_tr[:, 1:layer.modes, 1:2*layer.modes-1, :]  
    x_pad = NNlib.pad_zeros(x_p, (0, 0, 0, size(x_tr, 2) - layer.modes, 0, size(x_tr, 3) - (2*layer.modes -1), 0, 0))
    x_out = QG3.transform_grid(x_pad, layer.shgg)
    x_out_perm = permutedims(x_out, (2, 3, 1, 4)) # [lat, lon, channels, batch]
    end  
    return x_out_perm, st
end
#=
using JLD2, Lux, Random, QG3, NNlib
@load string(dirname(@__DIR__), "/data/t21-precomputed-p.jld2") qg3ppars


# pre-compute the model 
qg3ppars = qg3ppars
QG3.gpuoff()
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
x = rand(Float32, 32, 64, 32, 1)
model = SphericalConv(32, ggsh, shgg, 30, zsk=true)
gdev = gpu_device()
rnd = Random.default_rng(0)
ps, st = Lux.setup(rnd, model)
x = x 
model(x, ps, st)

using Zygote
gr = Zygote.gradient(ps -> sum(model(x, ps, st)[1]), ps)

@load string(dirname(@__DIR__), "/data/t42-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
model = SphericalConv(qg3ppars, 32, modes=model.modes, batch_size=size(x,4), gpu=false)
x = rand(Float32, 64, 128, 32, 1)
model(x, ps, st)
=#