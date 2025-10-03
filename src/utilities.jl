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
#Arguments
-`data`: Input array.
#Returns
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
