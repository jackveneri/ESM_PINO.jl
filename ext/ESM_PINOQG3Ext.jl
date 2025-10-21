module ESM_PINOQG3Ext

using ESM_PINO
using QG3
using ESM_PINO.Lux, ESM_PINO.Random, ESM_PINO.Zygote, ESM_PINO.NNlib, ESM_PINO.Statistics
using ESM_PINO.DocStringExtensions

struct ESM_PINOQG3
ggsh::QG3.GaussianGridtoSHTransform
shgg::QG3.SHtoGaussianGridTransform
end
export ESM_PINOQG3

include("ESM_PINOQG3Ext/SphericalConvTypeSpec.jl")
include("ESM_PINOQG3Ext/SFNO_components.jl")
include("ESM_PINOQG3Ext/losses.jl")
include("ESM_PINOQG3Ext/utilities.jl")
include("ESM_PINOQG3Ext/SFNO.jl")
end
