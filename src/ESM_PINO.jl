module ESM_PINO

using Lux, LuxCUDA, Random, Optimisers, Zygote, Statistics, MLUtils, ParameterSchedulers, FFTW, NNlib, ChainRulesCore, ComponentArrays, LinearAlgebra, SparseArrays
using AbstractFFTs, Adapt, CUDA
using Zygote:@adjoint 

include("FD_schemes.jl")
include("FNO_components.jl")
include("FNO1D_components.jl")
include("FNO3D_components.jl")
include("FNO.jl")
include("losses.jl")
include("utilities.jl")

abstract type AbstractSphericalConv <: Lux.AbstractLuxLayer end

struct SphericalConv{T} <: AbstractSphericalConv
    transform::T
end

# Default behavior: throw or warn if used without an extension
Lux.initialparameters(rng::AbstractRNG, layer::SphericalConv) =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::SphericalConv) =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::SphericalConv, ps, st, x) =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")


export ChannelMLP, SoftGating, GridEmbedding2D, FourierNeuralOperator
end
