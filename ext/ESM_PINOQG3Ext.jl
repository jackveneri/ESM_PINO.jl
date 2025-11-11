module ESM_PINOQG3Ext

using ESM_PINO
using QG3
using ESM_PINO.Lux, ESM_PINO.Random, ESM_PINO.Zygote, ESM_PINO.NNlib, ESM_PINO.Statistics, ESM_PINO.CUDA, ESM_PINO.Printf, ESM_PINO.MLUtils
using ESM_PINO.DocStringExtensions, ESM_PINO.Optimisers, ESM_PINO.ParameterSchedulers, ESM_PINO.OnlineStats, ESM_PINO.LuxCUDA, ESM_PINO.ChainRulesCore
struct ESM_PINOQG3
ggsh::QG3.GaussianGridtoSHTransform
shgg::QG3.SHtoGaussianGridTransform
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
