# ESM_PINO

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jackveneri.github.io/ESM_PINO.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jackveneri.github.io/ESM_PINO.jl/dev/)
[![Build Status](https://github.com/jackveneri/ESM_PINO.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jackveneri/ESM_PINO.jl/actions/workflows/CI.yml?query=branch%3Amaster)

## Installation

Install e.g. via `]add https://github.com/jackveneri/ESM_PINO.jl.git` and test the installation with `]test ESM_PINO`

## References

This package implements different Neural Operator architectures:

- The Fourier Neural Operator, following: [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- The Spherical Fourier Neural Operator, following: [Spherical Fourier Neural Operators:  Learning Stable Dynamics on the Sphere](http://arxiv.org/abs/2306.03838)
- Physics Informed loss functions, following: [Physics-Informed Neural Operator for Learning Partial Differential Equations](https://arxiv.org/abs/2111.03794)
- Implementation references in Python can be found at: <https://github.com/neuraloperator/physics_informed>, <https://github.com/NVIDIA/torch-harmonics>

## Examples

- `FNO_training.jl`, `SFNO_training.jl`, `FNO_PINO_training.jl`, `SFNO_PINO_training.jl` include training examples on data obtained from the QG3 model (see [QG3.jl](https://github.com/maximilian-gelbrecht/QG3.jl))
- `PINO_first_sketch.jl` includes training on data obtained from the 1D Burgers Equation (see <`burgers_simulation_fd_schemes.jl`>)