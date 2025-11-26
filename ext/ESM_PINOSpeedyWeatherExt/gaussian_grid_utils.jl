"""
    GaussianGridInfo

Structure containing information about a Gaussian grid resolution.

# Fields
- `truncation::Int`: Spectral truncation number (e.g., 31 for T31)
- `nlat::Int`: Number of latitude points
- `nlon::Int`: Number of longitude points  
- `km_at_equator::Float64`: Approximate grid spacing at equator in km
- `deg_at_equator::Float64`: Approximate grid spacing at equator in degrees
- `description::String`: Human-readable description
"""
struct GaussianGridInfo
    truncation::Int
    nlat::Int
    nlon::Int
    km_at_equator::Float64
    deg_at_equator::Float64
    description::String
end

# Standard Gaussian grid resolutions based on ECMWF and common climate model standards
const GAUSSIAN_GRIDS = Dict{String, GaussianGridInfo}(
    # Standard resolutions from NCAR Climate Data Guide and ECMWF
    "T21"   => GaussianGridInfo(21,   32,   64,   625, 5.61, "Very coarse resolution"),
    "T31"   => GaussianGridInfo(31,   48,   96,   417, 3.75, "Coarse resolution"),
    "T42"   => GaussianGridInfo(42,   64,  128,   310, 2.79, "Medium-coarse resolution"),
    "T62"   => GaussianGridInfo(62,   94,  192,   210, 1.89, "Medium resolution"),
    "T63"   => GaussianGridInfo(63,   96,  192,   210, 1.88, "Medium resolution"),
    "T85"   => GaussianGridInfo(85,  128,  256,   155, 1.39, "Medium-fine resolution"),
    "T106"  => GaussianGridInfo(106, 160,  320,   125, 1.12, "Fine resolution"),
    "T159"  => GaussianGridInfo(159, 240,  480,    83, 0.75, "Fine resolution"),
    "T213"  => GaussianGridInfo(213, 320,  640,    62, 0.56, "Very fine resolution"),
    "T255"  => GaussianGridInfo(255, 256,  512,    60, 0.54, "Very fine resolution"),
    "T319"  => GaussianGridInfo(319, 480,  960,    41, 0.37, "High resolution"),
    "T382"  => GaussianGridInfo(382, 576, 1152,    38, 0.34, "High resolution"),
    "T511"  => GaussianGridInfo(511, 768, 1536,    29, 0.26, "Very high resolution"),
    "T639"  => GaussianGridInfo(639, 960, 1920,    23, 0.21, "Ultra high resolution"),
    "T799"  => GaussianGridInfo(799, 800, 1600,    25, 0.22, "Ultra high resolution"),
    "T1279" => GaussianGridInfo(1279, 1920, 3840,   14, 0.13, "Extreme high resolution"),
    "T2047" => GaussianGridInfo(2047, 3072, 6144,    9, 0.08, "Extreme high resolution"),
)

"""
    gaussian_resolution_to_grid(resolution::AbstractString) -> Tuple{Int, Int}

Convert a Gaussian grid resolution string (e.g., "T31", "T63") to (nlat, nlon) tuple.

# Arguments
- `resolution::AbstractString`: Grid resolution in format "TN" where N is truncation number

# Returns  
- `Tuple{Int, Int}`: (number of latitude points, number of longitude points)

# Examples
```julia
julia> gaussian_resolution_to_grid("T31")
(48, 96)

julia> gaussian_resolution_to_grid("T63")  
(96, 192)

julia> gaussian_resolution_to_grid("T255")
(256, 512)
```

# Throws
- `ArgumentError`: If resolution is not recognized
"""
function gaussian_resolution_to_grid(resolution::AbstractString)
    resolution_upper = uppercase(resolution)
    
    if haskey(GAUSSIAN_GRIDS, resolution_upper)
        grid_info = GAUSSIAN_GRIDS[resolution_upper]
        return (grid_info.nlat, grid_info.nlon)
    else
        # Try to parse as TN format and calculate dimensions
        if startswith(resolution_upper, "T") && length(resolution_upper) > 1
            try
                truncation = parse(Int, resolution_upper[2:end])
                return calculate_gaussian_grid_size(truncation)
            catch
                throw(ArgumentError("Invalid resolution format: $resolution"))
            end
        else
            throw(ArgumentError("Unknown resolution: $resolution. Use format 'TN' where N is truncation number."))
        end
    end
end

"""
    calculate_gaussian_grid_size(truncation::Int) -> Tuple{Int, Int}

Calculate Gaussian grid dimensions from spectral truncation number using standard formulas.

For a spectral truncation T, the standard relationships are:
- nlat = (truncation + 1) * 3 / 2  (for reduced grids, varies slightly)
- nlon = 2 * nlat  (for regular grids)

# Arguments
- `truncation::Int`: Spectral truncation number

# Returns
- `Tuple{Int, Int}`: (nlat, nlon)
"""
function calculate_gaussian_grid_size(truncation::Int)
    # Standard relationship for Gaussian grids
    # This is an approximation - actual grids may vary slightly
    nlat = round(Int, (truncation + 1) * 3 / 2)
    nlon = 2 * nlat
    return (nlat, nlon)
end

"""
    get_truncation_from_nlat(nlat::Int) -> Int

Retrieve the spectral truncation number for a given number of latitude points.

# Arguments
- `nlat::Int`: Number of latitude points

# Returns
- `Int`: Spectral truncation number (e.g., 31 for T31)

# Examples
```julia
julia> get_truncation_from_nlat(48)
31

julia> get_truncation_from_nlat(96)
63

julia> get_truncation_from_nlat(256)
255
```

# Throws
- `ArgumentError`: If nlat doesn't match any known Gaussian grid resolution
"""
function get_truncation_from_nlat(nlat::Int)
    # Search through known grids
    for (name, grid_info) in GAUSSIAN_GRIDS
        if grid_info.nlat == nlat
            return grid_info.truncation
        end
    end
    
    # If not found in standard grids, try to estimate using inverse formula
    # nlat ≈ (truncation + 1) * 3 / 2
    # Therefore: truncation ≈ (2 * nlat / 3) - 1
    estimated_truncation = round(Int, (2 * nlat / 3) - 1)
    
    # Verify the estimate by calculating back
    calculated_nlat, _ = calculate_gaussian_grid_size(estimated_truncation)
    
    if calculated_nlat == nlat
        @warn "nlat=$nlat not found in standard grids. Estimated truncation=$estimated_truncation"
        return estimated_truncation
    else
        throw(ArgumentError("Cannot determine truncation for nlat=$nlat. Not found in standard grids and estimation failed."))
    end
end