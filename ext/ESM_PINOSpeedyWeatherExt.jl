module ESM_PINOSpeedyWeatherExt
using ESM_PINO
using SpeedyWeather
using SpeedyWeather.RingGrids
using SpeedyWeather.LowerTriangularArrays
using SpeedyWeather.SpeedyTransforms
using Lux, Random

include("ESM_PINOSpeedyWeatherExt/gaussian_grid_utils.jl")
include("ESM_PINOSpeedyWeatherExt/SphericalConvSpeedy.jl")

export SphericalConv
end