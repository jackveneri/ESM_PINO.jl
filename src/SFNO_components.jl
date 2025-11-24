"""
    gaussian_latitudes(nlat::Int) -> Vector{Float64}

Compute Gaussian latitude points (in degrees) for a grid with `nlat` latitudes.
Uses Legendre polynomial roots for accurate Gaussian quadrature points.

# Arguments
- `nlat::Int`: Number of latitude points

# Returns
- `Vector{Float64}`: Latitude values in degrees from North to South pole
"""
function gaussian_latitudes(nlat::Int)
    # Handle edge case
    if nlat == 1
        return [0.0]
    end
    
    # Compute Gaussian latitudes using Legendre polynomial roots
    # Initial guess using Chebyshev nodes
    θ = [π * (4i - 1) / (4nlat + 2) for i in 1:nlat]
    
    # Newton-Raphson iteration to find roots of Legendre polynomial
    for iter in 1:10
        # Evaluate Legendre polynomial and its derivative
        P, dP = legendre_and_derivative(nlat, cos.(θ))
        
        # Newton-Raphson update
        for i in 1:nlat
            correction = P[i] / dP[i]
            θ[i] -= correction
        end
    end
    
    # Convert co-latitude to latitude in degrees
    lats = 90.0 .- rad2deg.(θ)
    
    return lats
end

"""
    legendre_and_derivative(n::Int, x::Vector{Float64}) -> Tuple{Vector{Float64}, Vector{Float64}}

Compute Legendre polynomial Pn and its derivative at points x.
"""
function legendre_and_derivative(n::Int, x::Vector{Float64})
    P = zeros(length(x))
    dP = zeros(length(x))
    
    for i in eachindex(x)
        # Initialize recursion
        P0, P1 = 1.0, x[i]
        dP0, dP1 = 0.0, 1.0
        
        # Recursion for Legendre polynomials
        for k in 2:n
            P_next = ((2k - 1) * x[i] * P1 - (k - 1) * P0) / k
            dP_next = dP0 + (2k - 1) * P1
            P0, P1 = P1, P_next
            dP0, dP1 = dP1, dP_next
        end
        
        P[i] = P1
        dP[i] = dP1
    end
    
    return P, dP
end

"""
    GaussianGridEmbedding2D(normalize_to::Vector{Vector{Float32}} = [[0f0, 1f0], [0f0, 1f0]])

Positional embedding using Gaussian grid coordinates that adapts to input dimensions.
Appends normalized latitude (Gaussian) and longitude (uniform) coordinates to the input.
Expects input in (height, width, channels, batch) format.

# Arguments
- `normalize_to`: Vector of two intervals `[lat_min, lat_max]`, `[lon_min, lon_max]` 
  specifying the normalization range for coordinates

# Fields
- `normalize_lat::Vector{Float32}`: Range boundaries for latitude normalization
- `normalize_lon::Vector{Float32}`: Range boundaries for longitude normalization

# Details
- Constructs Gaussian latitude grid based on input height
- Constructs uniform longitude grid based on input width
- Normalizes coordinates to specified ranges (default [0,1] × [0,1])
- Repeats coordinate grids across batch dimension
- Concatenates `grid_lat` and `grid_lon` as extra channels to the input
"""
struct GaussianGridEmbedding2D <: Lux.AbstractLuxLayer
    normalize_lat::Vector{Float32}
    normalize_lon::Vector{Float32}
end

GaussianGridEmbedding2D(normalize_to::Vector{Vector{Float32}} = [[0f0, 1f0], [0f0, 1f0]]) =
    GaussianGridEmbedding2D(normalize_to[1], normalize_to[2])

ChainRulesCore.@non_differentiable gaussian_latitudes(::Int)


function Lux.initialparameters(rng::AbstractRNG, layer::GaussianGridEmbedding2D)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, layer::GaussianGridEmbedding2D)
    return NamedTuple()
end

function (layer::GaussianGridEmbedding2D)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    height, width, channels, batch_size = size(x)
    
    # Generate Gaussian latitudes based on input height
    if height == 1
        lats_deg = [0.0]
    else
        lats_deg = gaussian_latitudes(height)
    end
    
    # Normalize latitudes from [-90, 90] to specified range
    lats_normalized = (lats_deg .+ 90.0) ./ 180.0  # First to [0, 1]
    lat_range = layer.normalize_lat[2] - layer.normalize_lat[1]
    lats_normalized = lats_normalized .* lat_range .+ layer.normalize_lat[1]
    
    # Generate uniform longitudes based on input width
    if width == 1
        lons_normalized = [layer.normalize_lon[1]]
    else
        lons = LinRange(0.f0, 360.f0, width + 1)[1:end-1]  # Exclude 360 (same as 0)
        lons_normalized = lons ./ 360.f0  # Normalize to [0, 1]
        lon_range = layer.normalize_lon[2] - layer.normalize_lon[1]
        lons_normalized = lons_normalized .* lon_range .+ layer.normalize_lon[1]
    end
    
    # Create meshgrid
    grid_lat, grid_lon = meshgrid(Float32.(lats_normalized), Float32.(lons_normalized))
    
    # Reshape and repeat for batch
    grid_lat = reshape(grid_lat, (height, width, 1, 1))
    grid_lat = repeat(grid_lat, outer = (1, 1, 1, batch_size)) |> get_device(x)
    
    grid_lon = reshape(grid_lon, (height, width, 1, 1))
    grid_lon = repeat(grid_lon, outer = (1, 1, 1, batch_size)) |> get_device(x)
    
    return cat(x, grid_lat, grid_lon, dims=length(size(x))-1), st
end

ChainRulesCore.@non_differentiable (layer::GaussianGridEmbedding2D)(::Any)

"""
$(TYPEDSIGNATURES)
Empty layer to test extension documentation.
"""
struct SphericalKernel{T} <: Lux.AbstractLuxLayer
    spatial_conv::Union{Lux.Conv, Lux.NoOpLayer}
    spherical_conv::SphericalConv{T}
    norm::Union{Lux.InstanceNorm, Lux.NoOpLayer}  
end
# Default behavior: throw or warn if used without an extension
Lux.initialparameters(rng::AbstractRNG, layer::SphericalKernel{T}) where T =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::SphericalKernel{T}) where T =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::SphericalKernel{T}, ps, st, x) where T =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")

"""
$(TYPEDSIGNATURES)
Empty layer to test extension documentation.
"""
struct SFNO_Block{T} <: Lux.AbstractLuxLayer
    spherical_kernel :: SphericalKernel{T}
    channel_mlp :: ChannelMLP
    channels :: Int
    skip :: Bool
    activation::Function
end

# Default behavior: throw or warn if used without an extension
Lux.initialparameters(rng::AbstractRNG, layer::SFNO_Block{T}) where T =
    error("No implementation of `initialparameters` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

           Lux.initialstates(rng::AbstractRNG, layer::SFNO_Block{T}) where T =
    error("No implementation of `initialstates` for this layer. \
           Load the appropriate extension (e.g., ESM_PINOQG3Ext or ESM_PINOSpeedyWeatherExt).")

Lux.apply(layer::SFNO_Block{T}, ps, st, x) where T =
    error("No `apply` method defined for this layer type. \
           Check if an extension providing it is loaded.")