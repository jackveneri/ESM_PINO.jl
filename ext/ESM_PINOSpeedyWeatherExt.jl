module ESM_PINOSpeedyWeatherExt
using ESM_PINO
using SpeedyWeather
using SpeedyWeather.RingGrids
using SpeedyWeather.LowerTriangularArrays
using SpeedyWeather.SpeedyTransforms

include("gaussian_grid_utils.jl")
include("SphericalConvSpeedy.jl")

export SphericalConv
end