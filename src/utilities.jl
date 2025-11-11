"""
    add_noise(data::AbstractArray; noise_level::Real=0.1, noise_type::Symbol=:gaussian, relative::Bool=true, rng::AbstractRNG=Random.GLOBAL_RNG)

Add random noise to an array, supporting Gaussian or uniform distributions.

# Arguments
- `data::AbstractArray`: Input array to which noise will be added.
- `noise_level::Real=0.1`: Magnitude of the noise. Interpreted as standard deviation for Gaussian or half-width for uniform.
- `noise_type::Symbol=:gaussian`: Type of noise distribution. Options: `:gaussian` or `:uniform`.
- `relative::Bool=true`: If true, scale the noise level relative to the standard deviation of `data`.
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator.

# Returns
- `Array`: A copy of `data` with added noise.

# Details
- For `:gaussian` noise, samples are drawn from a normal distribution with mean 0 and specified standard deviation.
- For `:uniform` noise, samples are drawn uniformly from `[-noise_level, noise_level]`.
- If `relative=true`, the noise magnitude is scaled by the standard deviation of `data`.

# Example
```julia
julia> x = rand(10);

julia> y = add_noise(x; noise_level=0.05, noise_type=:gaussian);

julia> z = add_noise(x; noise_level=0.2, noise_type=:uniform, relative=false);
```
"""
function add_noise(data::AbstractArray; 
                  noise_level::Real=0.1, 
                  noise_type::Symbol=:gaussian, 
                  relative::Bool=true,
                  rng::AbstractRNG=Random.GLOBAL_RNG)
    
    # Validate noise level
    noise_level < 0 && throw(ArgumentError("noise_level must be non-negative"))
    
    # Determine floating point type for calculations
    F = float(eltype(data))
    
    # Calculate scaling factor (standard deviation for relative noise)
    scale = one(F)
    if relative
        if !isempty(data)
            s = std(data)
            scale = iszero(s) ? one(F) : F(s)
        end
    end
    
    F_noise_level = F(noise_level)
    scaled_level = F_noise_level * scale

    # Generate noise based on type
    if noise_type == :gaussian
        noise = randn(rng, F, size(data)) .* scaled_level
    elseif noise_type == :uniform
        noise = (rand(rng, F, size(data)) .- F(0.5)) .* (2 * scaled_level)
    else
        error("Unsupported noise type: $noise_type. Use :gaussian or :uniform.")
    end

    return data .+ noise
end

"""
    normalize_data(data)

Normalize an array to zero mean and unit variance.
# Arguments
-`data`: Input array.
# Returns
-`(normalized_data, μ, σ):` A tuple containing:
    normalized_data: The normalized array.
    μ: The mean of the original data.
    σ: The standard deviation of the original data.
"""
function normalize_data(data)
    μ = mean(data)
    σ = std(data)
    return (data .- μ) ./ σ, μ, σ
end

function denormalize_data(data, μ, σ)
    return data .* σ .+ μ
end
"""
    Generate Gaussian grid with proper Gaussian latitudes using Legendre polynomials.
    Returns latitudes in radians, sorted from North to South (decreasing order).
"""
function gaussian_grid(n_lat::Int; n_lon::Int=2*n_lat, iters::Int=100, tol::Real=1e-8)
    
    
    # Longitudes (equally spaced)
    lon_step = 2π / n_lon
    longitudes = range(0, 2π - lon_step, length=n_lon)
    
    # Gaussian latitudes (using Legendre polynomial roots)
    latitudes = compute_gaussian_latitudes(n_lat, iters=iters, tol=tol)
    
    return latitudes, longitudes 
end
"""
    Compute Gaussian latitudes using Newton's method to find roots of Legendre polynomials.
    Returns latitudes in radians, sorted from North to South (decreasing order).
"""
function compute_gaussian_latitudes(n_lat::Int; iters::Int=100, tol::Real=1e-8)
    
    # We'll store the cosine values (x) first, then convert to latitudes
    x_values = zeros(n_lat)
    
    # Initial guesses for the roots (Chebyshev nodes as starting points)
    for i in 1:n_lat
        # Initial guess using Chebyshev nodes
        x0 = cos(π * (i - 0.5) / n_lat)
        
        # Refine using Newton's method
        x = x0
        for _ in 1:iters
            p, dp = legendre_polynomial(n_lat, x)
            x -= p / dp
            abs(p) < tol && break
        end
        
        x_values[i] = x
    end
    
    # Convert from cosine of colatitude to latitude in radians
    # and sort from North Pole (π/2) to South Pole (-π/2)
    latitudes = asin.(sort(x_values, rev=true))
    
    return latitudes
end
"""
    Compute Legendre polynomial P_n(x) and its derivative using recurrence relation.
"""
function legendre_polynomial(n::Int, x::Float64)
    
    if n == 0
        return 1.0, 0.0
    elseif n == 1
        return x, 1.0
    end
    
    p_prev = 1.0
    p_curr = x
    dp_prev = 0.0
    dp_curr = 1.0
    
    for k in 2:n
        p_next = ((2k - 1) * x * p_curr - (k - 1) * p_prev) / k
        dp_next = ((2k - 1) * (p_curr + x * dp_curr) - (k - 1) * dp_prev) / k
        
        p_prev, p_curr = p_curr, p_next
        dp_prev, dp_curr = dp_curr, dp_next
    end
    
    return p_curr, dp_curr
end

function analyze_weights(ps, prefix="", indent=0)
    indent_str = "  " ^ indent
    
    for (key, value) in pairs(ps)
        current_path = isempty(prefix) ? string(key) : "$prefix.$key"
        
        if value isa AbstractArray && eltype(value) <: Number
            # We've reached actual numerical arrays (weights/biases)
            println("$(indent_str)$current_path:")
            println("$(indent_str)  Shape: $(size(value))")
            println("$(indent_str)  Mean: $(round(mean(value), digits=6))")
            println("$(indent_str)  Std: $(round(std(value), digits=6))")
            println("$(indent_str)  Min/Max: $(round(minimum(value), digits=6)) / $(round(maximum(value), digits=6))")
            println("$(indent_str)  L2 norm: $(round(norm(value), digits=6))")
            println("$(indent_str)  % zeros: $(round(100 * count(isapprox(0), value) / length(value), digits=2))%")
            println()
        elseif value isa NamedTuple || value isa Dict
            # Recurse into nested structures
            println("$(indent_str)$current_path/")
            analyze_weights(value, current_path, indent + 1)
        end
    end
end