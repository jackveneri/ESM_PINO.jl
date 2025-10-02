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

function normalize_data(data)
    μ = mean(data)
    σ = std(data)
    return (data .- μ) ./ σ, μ, σ
end

function denormalize_data(data, μ, σ)
    return data .* σ .+ μ
end
