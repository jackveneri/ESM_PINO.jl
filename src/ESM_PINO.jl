module ESM_PINO

using Lux, LuxCUDA, Random, Optimisers, Zygote, Statistics, MLUtils, ParameterSchedulers, FFTW, NNlib, ChainRulesCore, ComponentArrays, LinearAlgebra, SparseArrays
using AbstractFFTs, Adapt, CUDA
using Zygote:@adjoint 

include("NewRepeatedLayer.jl")
include("FD_schemes.jl")
include("FNO_components.jl")
include("FNO1D_components.jl")
include("FNO3D_components.jl")
include("FNO.jl")
include("losses.jl")

export FourierNeuralOperator, QG3_loss_function, loss_function_just_data, PINO_spectral_loss_function, PINO_FD_loss_function, Grid, BurgersFD2, BurgersFD
export SpectralPhysicsLossParameters, FDPhysicsLossParameters, create_physics_loss, select_loss_function, QG3_Physics_Parameters, create_QG3_physics_loss, select_QG3_loss_function
end
