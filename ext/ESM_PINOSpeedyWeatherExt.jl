module ESM_PINOSpeedyWeatherExt
using ESM_PINO
using SpeedyWeather
using SpeedyWeather.RingGrids
using SpeedyWeather.LowerTriangularArrays
using SpeedyWeather.SpeedyTransforms
using Lux, Random
using ESM_PINO.DocStringExtensions

struct ESM_PINOSpeedy
    spectral_transform :: SpectralTransform
    NF :: Type{<:AbstractFloat}
end
export ESM_PINOSpeedy
include("ESM_PINOSpeedyWeatherExt/gaussian_grid_utils.jl")
include("ESM_PINOSpeedyWeatherExt/SphericalConvSpeedy.jl")

end