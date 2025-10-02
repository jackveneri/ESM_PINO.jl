module ESM_PINOSpeedyWeatherExt
using ESM_PINO
using SpeedyWeather
using SpeedyWeather.RingGrids
using SpeedyWeather.LowerTriangularArrays
using SpeedyWeather.SpeedyTransforms
using Lux, Random

include("gaussian_grid_utils.jl")
include("SphericalConvSpeedy.jl")

export SphericalConv
end