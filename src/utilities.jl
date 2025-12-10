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
    normalize_data(data; channelwise=false, dims=nothing)

Normalize data using mean and standard deviation.

# Arguments
- `data::AbstractArray`: Input data to normalize

# Keyword Arguments
- `channelwise::Bool=false`: If true, normalize each channel independently
- `dims::Union{Nothing, Int, Tuple}=nothing`: Dimensions to compute statistics over.
  If `nothing`, behavior depends on `channelwise` and data shape.

# Returns
- Normalized data, mean(s), std(s)

# Details
When `channelwise=true`:
- Attempts to detect 4D format (lat, lon, channel, batch)
- Computes per-channel statistics across spatial and batch dimensions
- Returns vectors of means and stds (one per channel)

When `channelwise=false`:
- Computes global statistics across all dimensions
- Returns scalar mean and std

# Examples
```julia
# Global normalization
data_norm, μ, σ = normalize_data(data)

# Channel-wise normalization (4D data)
data = randn(Float32, 48, 96, 5, 32)  # (lat, lon, channels, batch)
data_norm, μ_vec, σ_vec = normalize_data(data, channelwise=true)

# Custom dimensions
data_norm, μ, σ = normalize_data(data, dims=(1, 2, 4))  # Over lat, lon, batch
```
"""
function normalize_data(data::AbstractArray; channelwise::Bool=false, dims::Union{Nothing, Int, Tuple}=nothing)
    
    if channelwise
        ndims_data = ndims(data)
        
        if ndims_data == 4
            # Format: (lat, lon, channel, batch)
            # Compute statistics over dims (1, 2, 4) for each channel
            μ = mean(data, dims=(1, 2, 4))  # Shape: (1, 1, n_channels, 1)
            σ = std(data, dims=(1, 2, 4))   # Shape: (1, 1, n_channels, 1)
            
            # Avoid division by zero - use broadcasting-safe replacement
            σ_safe = ifelse.(σ .< eps(eltype(data)), one(eltype(data)), σ)
            
            # Check for near-zero std and warn (on CPU)
            if any(Array(σ) .< eps(eltype(data)))
                n_channels = size(data, 3)
                @warn "Some channels have near-zero std. Setting those to 1.0 to avoid division by zero."
            end
            
            # Normalize using broadcasting (GPU-friendly)
            normalized_data = (data .- μ) ./ σ_safe
            
            # Squeeze dimensions for return values
            μ_squeezed = dropdims(μ, dims=(1, 2, 4))
            σ_squeezed = dropdims(σ_safe, dims=(1, 2, 4))
            
            return normalized_data, μ_squeezed, σ_squeezed
            
        elseif ndims_data == 3
            # Try to determine format
            sizes = size(data)
            if sizes[3] < minimum(sizes[1:2])
                # Assume (lat, lon, channel) format
                # Compute statistics over dims (1, 2) for each channel
                μ = mean(data, dims=(1, 2))  # Shape: (1, 1, n_channels)
                σ = std(data, dims=(1, 2))   # Shape: (1, 1, n_channels)
                
                # Avoid division by zero
                σ_safe = ifelse.(σ .< eps(eltype(data)), one(eltype(data)), σ)
                
                if any(Array(σ) .< eps(eltype(data)))
                    @warn "Some channels have near-zero std. Setting those to 1.0."
                end
                
                # Normalize
                normalized_data = (data .- μ) ./ σ_safe
                
                # Squeeze dimensions
                μ_squeezed = dropdims(μ, dims=(1, 2))
                σ_squeezed = dropdims(σ_safe, dims=(1, 2))
                
                return normalized_data, μ_squeezed, σ_squeezed
            else
                @warn "3D data format unclear. Falling back to global normalization. Use dims argument for custom behavior."
                channelwise = false
            end
        else
            @warn "Channel-wise normalization requested but data has $ndims_data dimensions. Expected 3 or 4. Falling back to global normalization."
            channelwise = false
        end
    end
    
    # Global normalization (or fallback)
    if !isnothing(dims)
        # Custom dimensions specified
        μ = mean(data, dims=dims)
        σ = std(data, dims=dims)
        
        # Avoid division by zero using broadcasting
        σ_safe = ifelse.(σ .< eps(eltype(data)), one(eltype(data)), σ)
        
        return (data .- μ) ./ σ_safe, μ, σ_safe
    else
        # Global statistics (returns scalars)
        μ = mean(data)
        σ = std(data)
        
        # Check on CPU if needed
        σ_val = σ isa Number ? σ : Array(σ)[]
        if σ_val < eps(eltype(data))
            @warn "Data has near-zero std ($σ_val). Setting std to 1.0."
            σ = one(eltype(data))
        end
        
        return (data .- μ) ./ σ, μ, σ
    end
end
function normalize_data(data::AbstractArray, μ::Union{Real, AbstractArray},σ::Union{Real, AbstractArray})
    channelwise = !(isa(σ, Real) && isa(μ, Real))
    if channelwise
        # (lat, lon, channel, batch) format
        n_channels = size(data, 3)
        
        if length(μ) != n_channels || length(σ) != n_channels
            throw(ArgumentError("μ and σ length ($(length(μ))) must match number of channels ($n_channels)"))
        end
        normalized_channels = [
                (selectdim(data, 3, c).- μ[c]) ./ σ[c] 
                for c in 1:n_channels
            ]
            
        # Stack along the channel dimension
        return permutedims(cat(normalized_channels...; dims=4),(1,2,4,3))
    else
        return (data .- μ) ./ σ
    end    
end
"""
    denormalize_data(normalized_data, μ, σ; channelwise=false)

Reverse the normalization applied by `normalize_data`.

# Arguments
- `normalized_data::AbstractArray`: Normalized data
- `μ`: Mean(s) used for normalization (scalar or vector)
- `σ`: Std(s) used for normalization (scalar or vector)

# Keyword Arguments
- `channelwise::Bool=false`: Must match the mode used during normalization

# Returns
- Original scale data

# Examples
```julia
# Global denormalization
data_original = denormalize_data(data_norm, μ, σ)

# Channel-wise denormalization
data_original = denormalize_data(data_norm, μ_vec, σ_vec, channelwise=true)
```
"""
function denormalize_data(normalized_data::AbstractArray, μ, σ)
    channelwise = !(isa(σ, Real) && isa(μ, Real))
    
    if channelwise
        if !(μ isa AbstractVector && σ isa AbstractVector)
            throw(ArgumentError("For channelwise denormalization, μ and σ must be vectors"))
        end
        
        ndims_data = ndims(normalized_data)
        
        if ndims_data == 4
            n_channels = size(normalized_data, 3)
            
            if length(μ) != n_channels || length(σ) != n_channels
                throw(ArgumentError("μ and σ length ($(length(μ))) must match number of channels ($n_channels)"))
            end
            
            μ_reshaped = reshape(μ, 1, 1, :, 1)
            σ_reshaped = reshape(σ, 1, 1, :, 1)
            
            return normalized_data .* σ_reshaped .+ μ_reshaped
            
        elseif ndims_data == 3
            n_channels = size(normalized_data, 3)
            
            if length(μ) != n_channels || length(σ) != n_channels
                throw(ArgumentError("μ and σ length must match number of channels"))
            end
            
            μ_reshaped = reshape(μ, 1, 1, :)
            σ_reshaped = reshape(σ, 1, 1, :)
            
            return normalized_data .* σ_reshaped .+ μ_reshaped
            
        else
            throw(ArgumentError("Channel-wise denormalization expects 3D or 4D data"))
        end
    else
        return normalized_data .* σ .+ μ
    end
end
"""
    Generate Gaussian grid with proper Gaussian latitudes using Legendre polynomials.
    Returns latitudes in radians, sorted from North to South (decreasing order).
"""
function gaussian_grid(n_lat::Int; n_lon::Int=2*n_lat, iters::Int=100, tol::Real=1e-10)
    
    # Longitudes (equally spaced)
    if n_lon ==64
        longitudes = Float32[0.000000000000000,0.098174773156643,0.196349546313286,0.294524312019348,0.392699092626572,
                        0.490873843431473,0.589048624038696,0.687223374843597,0.785398185253143,0.883572936058044,
                        0.981747686862946,1.079922437667847,1.178097248077393,1.276272058486938,1.374446749687195,
                        1.472621560096741,1.570796370506287,1.668971061706543,1.767145872116089,1.865320682525635,
                        1.963495373725891,2.061670064926147,2.159844875335693,2.258019685745239,2.356194496154785,
                        2.454369306564331,2.552544116973877,2.650718688964844,2.748893499374390,2.847068309783936,
                        2.945243120193481,3.043417930603027,3.141592741012573,3.239767313003540,3.337942123413086,
                        3.436116933822632,3.534291744232178,3.632466554641724,3.730641365051270,3.828815937042236,
                        3.926990747451782,4.025165557861328,4.123340129852295,4.221515178680420,4.319689750671387,
                        4.417864799499512,4.516039371490479,4.614213943481445,4.712388992309570,4.810563564300537,
                        4.908738613128662,5.006913185119629,5.105088233947754,5.203262805938721,5.301437377929688,
                        5.399612426757812,5.497786998748779,5.595962047576904,5.694136619567871,5.792311191558838,
                        5.890486240386963,5.988660812377930,6.086835861206055,6.185010433197021]
    else
    lon_step = 2π / n_lon
    longitudes = range(0, 2π - lon_step, length=n_lon)
    end
    # Gaussian latitudes (using Legendre polynomial roots)
    latitudes = compute_gaussian_latitudes(n_lat, iters=iters, tol=tol)
    
    return latitudes, longitudes 
end

"""
    Compute Gaussian latitudes using Newton's method to find roots of Legendre polynomials.
    Returns latitudes in radians, sorted from North to South (decreasing order).
    If n_lat is a common value (32 or 64), precomputed roots are returned to ensure consistency with precomputed available QG3 data.
"""
function compute_gaussian_latitudes(n_lat::Int; iters::Int=100, tol::Real=1e-10)
    #for common n_lat values, return precomputed roots for QG3 compatibility
    if n_lat == 32
        @load string(dirname(@__DIR__),"/data/t21-precomputed-p.jld2") qg3ppars
        return qg3ppars.lats
    elseif n_lat == 64
        @load string(dirname(@__DIR__),"/data/t42-precomputed-p.jld2") qg3ppars
        return qg3ppars.lats
    else
        n = n_lat
        x_values = zeros(Float64, n)
        
        for i in 1:n
            #initial guess
            k = Float64(i)
            x0 = cos(π * (k - 0.25 + 1/(8*(2k-1))) / (n + 0.5))
            
            x = x0
            for iter in 1:iters
                p, dp = legendre_polynomial(n, x)
                if abs(p) < tol
                    break
                end
                delta = p / dp
                x_new = x - delta
                if x_new < -1.0 || x_new > 1.0
                    x_new = x - 0.5 * delta
                end
                x = x_new
            end
            x_values[i] = x
        end
        
        # Sort and convert to latitudes
        return asin.(sort(x_values, rev=true))
    end
end

"""
    Compute Legendre polynomial P_n(x) and its derivative using recurrence relation.
    This uses the standard normalization where P_n(1) = 1.
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
        # Recurrence relation for Legendre polynomials
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
            println("$(indent_str)  L2 norm: $(round(norm(value)/sqrt(length(value)), digits=6))")
            println("$(indent_str)  % zeros: $(round(100 * count(isapprox(0), value) / length(value), digits=2))%")
            println()
        elseif value isa NamedTuple || value isa Dict
            # Recurse into nested structures
            println("$(indent_str)$current_path/")
            analyze_weights(value, current_path, indent + 1)
        end
    end
end

function apply_n_times(f, x::AbstractArray, n::Int; m::Int=0, μ=0.0,  σ=1.0, channelwise::Bool=false)
    y = x
    snapshots = m > 0 ? Vector{typeof(x)}() : nothing
    save_steps = m > 0 ? round.(Int, range(1, n; length=m)) : Int[]
    
    for i in 1:n
        y = f(y)
        if i in save_steps
            push!(snapshots, copy(denormalize_data(y, μ, σ)))
        end
    end

    return m > 0 ? snapshots : y
end