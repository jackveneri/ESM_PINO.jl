module ESM_PINOQG3Ext

using ESM_PINO
using QG3
using ESM_PINO.Lux, ESM_PINO.LuxCore, ESM_PINO.Random, ESM_PINO.Zygote, ESM_PINO.NNlib, ESM_PINO.Statistics, ESM_PINO.CUDA, ESM_PINO.Printf, ESM_PINO.MLUtils, ESM_PINO.JLD2
using ESM_PINO.DocStringExtensions, ESM_PINO.Optimisers, ESM_PINO.ParameterSchedulers, ESM_PINO.OnlineStats, ESM_PINO.LuxCUDA, ESM_PINO.ChainRulesCore, ESM_PINO.OMEinsum
"""
    RemapPlan

Precomputed plan for efficient array remapping.
Stores source and destination indices to avoid recomputation.
"""
struct RemapPlan{T<:Integer}
    l::Int
    c::Int
    src_indices::Vector{T}  # Source indices in dimension 3
    dst_indices::Vector{T}  # Destination indices in dimension 3
    output_size_3::Int      # Output size in dimension 3 (2c)
    input_size_3::Int       # Input size in dimension 3 (2l+1)
end
struct ESM_PINOQG3
ggsh::QG3.GaussianGridtoSHTransform
shgg::QG3.SHtoGaussianGridTransform
linerar_indices::Array{Int,4}
inverse_linear_indices::Array{Int,4}
remap_plan::RemapPlan
end
export ESM_PINOQG3
include("ESM_PINOQG3Ext/gaussian_grid_utils.jl")
include("ESM_PINOQG3Ext/losses.jl")
include("ESM_PINOQG3Ext/utilities.jl")
include("ESM_PINOQG3Ext/SphericalConvTypeSpec.jl")
include("ESM_PINOQG3Ext/SFNO_components.jl")
include("ESM_PINOQG3Ext/SFNO.jl")

export qg3pars_constructor_helper
end
