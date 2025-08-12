struct SphericalConv{T, G, S} <: Lux.AbstractLuxLayer
    pars::QG3.QG3ModelParameters{T}
    hidden_channels::Int
    modes::Int
    ggsh::G  # GaussianGridtoSHTransform
    shgg::S  # SHtoGaussianGridTransform

    # Inner constructor to enforce modes ≤ pars.L
    function SphericalConv(
            pars::QG3.QG3ModelParameters{T},
            hidden_channels::Int,
            modes::Int,
            ggsh::G,
            shgg::S
        ) where {T, G, S}
        # Correct modes if necessary
        corrected_modes = min(modes, pars.L)
        if modes != corrected_modes
            @warn "modes ($modes) exceeds pars.L ($(pars.L)). Setting modes = $(pars.L)."
        end
        # Create the struct with the corrected value
        new{T, G, S}(pars, hidden_channels, corrected_modes, ggsh, shgg)
    end
end

function SphericalConv(
        pars::QG3.QG3ModelParameters{T},
        hidden_channels::Int;
        modes::Int = pars.L,
        batch_size::Int = 1
    ) where T
    # Correct modes upfront (before inner constructor)
    corrected_modes = min(modes, pars.L)
    if modes != corrected_modes
        @warn "modes ($modes) exceeds pars.L ($(pars.L)). Setting modes = $(pars.L)."
    end
    # Proceed with construction
    ggsh = QG3.GaussianGridtoSHTransform(pars, hidden_channels; N_batch=batch_size)
    shgg = QG3.SHtoGaussianGridTransform(pars, hidden_channels; N_batch=batch_size)
    SphericalConv(pars, hidden_channels, corrected_modes, ggsh, shgg)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SphericalConv{T}) where T
    init_std = sqrt(T(1) / layer.hidden_channels)
    # Initialize 2D weights for spatial pattern (L × M)
    weight = init_std * randn(rng, T, 1, layer.modes, 2 * layer.modes - 1, 1)
    return (weight=weight,)
end

Lux.initialstates(rng::AbstractRNG, layer::SphericalConv) = NamedTuple()

function (layer::SphericalConv{T})(x::AbstractArray{T,4}, ps::NamedTuple, st::NamedTuple) where T
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
using JLD2
@load string(@__DIR__, "/data/t21-precomputed-p.jld2") qg3ppars


# pre-compute the model 
qg3ppars = qg3ppars
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
x = rand(Float32, 32, 64, 32, 1)
model = SphericalConv(qg3ppars, 32, batch_size=size(x,4), modes=30)
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
model = SphericalConv(qg3ppars, 32, model.modes, ggsh, shgg)
x = rand(Float32, 64, 128, 32, 1) |> gdev
model(x, ps, st)
=#