# ESM_PINO

[comment]: <> ([![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jackveneri.github.io/ESM_PINO.jl/stable/))
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jackveneri.github.io/ESM_PINO.jl/dev/)
[![Build Status](https://github.com/jackveneri/ESM_PINO.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jackveneri/ESM_PINO.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Installation

Install e.g. via `]add https://github.com/jackveneri/ESM_PINO.jl.git` and test the installation with `]test ESM_PINO`

## References

This package implements different Neural Operator architectures:

- The Fourier Neural Operator, following: [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- The Spherical Fourier Neural Operator, following: [Spherical Fourier Neural Operators:  Learning Stable Dynamics on the Sphere](http://arxiv.org/abs/2306.03838)
- Physics Informed loss functions, following: [Physics-Informed Neural Operator for Learning Partial Differential Equations](https://arxiv.org/abs/2111.03794)
- Implementation references in Python can be found at: <https://github.com/neuraloperator/physics_informed>, <https://github.com/NVIDIA/torch-harmonics>

## Example Scripts

- [FNO_training.jl](examples/FNO_training.jl), [SFNO_training.jl](examples/SFNO_training.jl), [FNO_PINO_training.jl](examples/FNO_PINO_training.jl), [SFNO_PINO_training.jl](examples/SFNO_PINO_training.jl) include training examples on data obtained from the QG3 model (see [QG3.jl](https://github.com/maximilian-gelbrecht/QG3.jl))
- [PINO_first_sketch.jl](examples/PINO_first_sketch.jl) includes training on data obtained from the 1D Burgers Equation (see [burgers_simulation_fd_schemes.jl](examples/burgers_simulation_FD_schemes.jl))

## About extensions

We rely on different backends to implement Spherical Harmonics transforms, and therefore we have two different SFNO constructors.

### QG3.jl

This is an unregistered package, we thank Maximilian Gelbrecht for providing it.
It is AD-compatible with the Zygote backend.
In our extension we rely on it not only to define an SFNO constructor, but also to implement different physics informed losses. To do so, we define internal types, and we therefore recommend calling such functions defining at the top of your script:

```julia
using ESM_PINO, QG3

const ESM_PINOQG3 = Base.get_extension(ESM_PINO, :ESM_PINOQG3Ext)

```
to be able to see them in the namespace. For more extensive documentation, see <https://jackveneri.github.io/ESM_PINO.jl/dev/extensions/QG3/#QG3-Extension>

### SpeedyWeather.jl

This package (available at <https://github.com/SpeedyWeather/SpeedyWeather.jl>) provides a more feature-rich implementation of Spherical Harmonics. It is designed to be AD-compatible using the Enzyme backend, but is currently still WIP

## Minimal Usage Example

FNO Example (2D data with grid embedding):

```julia
using Lux, Random, ESM_PINO

rng = Random.default_rng()

layer = FourierNeuralOperator(
    in_channels=3,
    out_channels=2,
    hidden_channels=32,
    n_modes=(12, 12),
    n_layers=4,
    positional_embedding="grid"
)

ps = Lux.initialparameters(rng, layer)
st = Lux.initialstates(rng, layer)

# Input tensor (H, W, C, Batch)
x = randn(Float32, 64, 64, 3, 10)

y, st_new = layer(x, ps, st)
@show size(y)   # expect (64, 64, 2, 10)
```

Another FNO example (1D data without grid embedding):

```julia
using Lux, Random, ESM_PINO

layer1d = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=16,
    n_modes=(8,),
    n_layers=3,
    positional_embedding="no_grid1D"
)

x1 = randn(Float32, 128, 1, 5)   # (L, C, Batch)
y1, _ = layer1d(x1,
    Lux.initialparameters(rng, layer1d),
    Lux.initialstates(rng, layer1d)
)
@show size(y1)   # expect (128, 1, 5)
```

SFNO usage example:

```julia
using Lux, QG3, Random, ESM_PINO, LuxCUDA

# Load precomputed QG3 parameters
qg3ppars = QG3.load_precomputed_params()[2]

# Input: [lat, lon, channels, batch]
x = rand(Float32, 32, 64, 3, 10)


# Construct SFNO layer using secondary constructor
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=size(x,4))
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=size(x,4))
model2 = SFNO(ggsh, shgg;
    modes=15,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2.0,
    positional_embedding="no_grid",
    outer_skip=true,
    zsk=true
)

# Setup parameters and state
rng = Random.default_rng(0)
ps, st = Lux.setup(rng, model2)

# Forward pass
y, st = model2(x, ps, st)

# Compute gradients
using Zygote
gr = Zygote.gradient(ps -> sum(model2(x, ps, st)[1]), ps)
```
