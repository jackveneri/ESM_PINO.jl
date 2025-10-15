"""
    spectral_derivative(u::AbstractArray{T}, L::T) where T<:Real

Compute first and second spatial derivatives using FFT spectral methods.

# Arguments
- `u`: Input array (real-valued), assumed to be on GPU. First dimension is spatial.
- `L`: Domain length in spatial dimension.

# Returns
- `du`: First derivative (real array)
- `d2u`: Second derivative (real array)

# Notes
- Uses FFT/iFFT with wavenumbers from `compute_k`
- Assumes periodic boundary conditions
- Maintains input array type/location (GPU/CPU)
- Output derivatives are real-valued arrays
"""
function spectral_derivative(u::AbstractArray{T}, L::T) where T<:Real
    k = compute_k(u, L)  # Defined below
    
    u_hat = fft(u)
    # First derivative
    du_hat = im .* k .* u_hat
    du = real(ifft(du_hat))
    # Second derivative
    d2u_hat = -k.^2 .* u_hat
    d2u = real(ifft(d2u_hat))
    
    return du, d2u
end

"""
    compute_k(u::AbstractArray{T}, L::T) where T<:Real

Generate wavenumber array for spectral differentiation.

# Arguments
- `u`: Template array for dimensions
- `L`: Domain length

# Returns
- `k`: Wavenumber array on GPU, reshaped for broadcasting
# Details
- Handles even/odd array sizes differently
- Automatically converts to GPU array
- Returns array with singleton dimensions for ND broadcasting
"""
function compute_k(u::AbstractArray{T}, L::T) where T<:Real
    N = size(u, 1)  # Assuming first dimension is spatial
    freq = if iseven(N)
        [0:(N÷2-1); (-N÷2):-1]
    else
        n = (N-1) ÷ 2
        [0:n; -n:-1]
    end
    
    # Convert to GPU array with matching type
    freq_gpu = CUDA.cu(T.(freq))
    k = (2π/L) .* freq_gpu
    
    #Add singleton dimensions to match input dimensions
    return reshape(k, (size(k)..., ntuple(_->1, ndims(u)-1)...))
end

"""    
    dealias(u_hat::AbstractArray{Complex{T}}, L::T) where T<:Real

Apply 2/3 dealiasing filter to Fourier coefficients.

# Arguments
- `u_hat`: Fourier coefficients (complex array)
- `L`: Domain length (unused in current implementation)

# Returns
- Filtered coefficients with high frequencies zeroed

# Notes
- Implements 2/3 rule for anti-aliasing
- Creates mask directly on GPU
- Preserves array dimensions for broadcasting
"""
function dealias(u_hat::AbstractArray{Complex{T}}, L::T) where T<:Real
    N = size(u_hat, 1)
    cutoff = floor(Int, N/3)
    
    # Create mask directly on GPU
    mask = CUDA.cu([i <= cutoff for i in 1:N])
    
    #Add singleton dimensions for broadcasting
    mask = reshape(mask, (size(mask)..., ntuple(_->1, ndims(u_hat)-1)...))
    return u_hat .* mask
end
"""
    SpectralPhysicsLossParameters(ν::Float64, L::Float64, N_t::Int, t_max::Float64, t_min::Float64, Δt::Float64, x_σ::Float64, x_μ::Float64)

Create a struct to hold parameters for spectral physics loss.

# Fields
- `ν`: Viscosity (scalar)
- `L`: Domain size (scalar)
- `N_t`: Number of time steps (integer)
- `t_max`: Maximum time (scalar)
- `t_min`: Minimum time (scalar)
- `Δt`: Time step size (scalar)
- `x_σ`: Standard deviation for normalization (scalar)
- `x_μ`: Mean for normalization (scalar)
"""
struct SpectralPhysicsLossParameters
    ν::Float64
    Δt::Int
    L::Float64
    x_σ::Float64
    x_μ::Float64
    t_step_length::Float64
end

function SpectralPhysicsLossParameters(ν::Float64, L::Float32, t_step_length::Float64)
    return SpectralPhysicsLossParameters(ν, L, 1, 0, t_step_length)
end
"""
    create_physics_loss()

helper function to create a physics loss function.

# Arguments
- `params`: parameters struct, pass nothing to create a zero loss function.
"""
function create_physics_loss(params::SpectralPhysicsLossParameters)
    # Capture `params` in the closure
    @views function physics_loss(u::StatefulLuxLayer, u_t1::AbstractArray)
         # Access params via the captured struct
        u_t2 = u(u_t1)
        u_t2 = u_t2 .* params.x_σ .+ params.x_μ
        u_t1 = u_t1 .* params.x_σ .+ params.x_μ
        if ndims(u_t1) == 3
            ∂u_∂t = (u_t2 .- u_t1) ./ params.t_step_length
            boundary_residual = u_t2[1:1,:,:] .- u_t2[end:end,:,:]
            ∂f_∂x, ∂u_∂xx = spatial_derivative(u_t2, params)
        else 
            forward_diff_first = (u_t2[:, 2:2, :, :] .- u_t2[:, 1:1, :, :]) ./ params.t_step_length
            forward_diff_last = (u_t2[:, end:end, :, :] .- u_t2[:, end-1:end-1, :, :]) ./ params.t_step_length
            central_diff = (u_t2[:, 3:end, :, :] .- u_t2[:, 1:end-2, :, :]) ./ (2 * params.t_step_length)
            ∂u_∂t = cat(forward_diff_first, central_diff, forward_diff_last; dims=2)
            boundary_residual = u_t2[1:1,:,:,:] .- u_t2[end:end,:,:,:]
            u_t2_permute =  permutedims(u_t2, (1, 4, 3, 2)) 
            ∂f_∂x, ∂u_∂xx = spatial_derivatives_batch(u_t2, params)
            ∂f_∂x = permutedims(∂f_∂x, (1, 4, 3, 2))
            ∂u_∂xx = permutedims(∂u_∂xx, (1, 4, 3, 2))
        end
        return mean(abs2, ∂u_∂t .+ ∂f_∂x .- (params.ν .* ∂u_∂xx)) + mean(abs2, boundary_residual)
    end
    return physics_loss    
end

"""
    FDPhysicsLossParameters(ν::Float64, N_t::Int, t_max::Float64, t_min::Float64, Δt::Float64, x_σ::Float64, x_μ::Float64, M1_gpu::AbstractArray, M2_gpu::AbstractArray)

Create a struct to hold parameters for finite difference physics loss.

# Fields
- `ν`: Viscosity (scalar)
- 't_step_length`: Time step length (scalar)
- `M1_gpu`: Second derivative FD matrix (GPU array)
- `M2_gpu`: First derivative FD matrix (GPU array)
"""
struct FDPhysicsLossParameters
    ν::Float64
    Δt::Int
    t_step_length::Float32
    x_σ::Float64
    x_μ::Float64
    M1_gpu::AbstractArray
    M2_gpu::AbstractArray
end

function FDPhysicsLossParameters(ν::Float64, Δt::Int, t_step_length::Float32, M1_gpu::AbstractArray, M2_gpu::AbstractArray)
    return FDPhysicsLossParameters(ν, Δt, t_step_length, 1, 0, M1_gpu, M2_gpu)
end

function create_physics_loss(params::FDPhysicsLossParameters)
    # Capture `params` in the closure
    function BurgersFD_physics_loss(u::StatefulLuxLayer, u_t1::AbstractArray)
        # Access params via the captured struct
        u_t2 = u(u_t1)
        u_t2 = u_t2 .* params.x_σ .+ params.x_μ
        u_t1 = u_t1 .* params.x_σ .+ params.x_μ
        if ndims(u_t1) == 3
            ∂u_∂t = (u_t2 .- u_t1) ./ params.t_step_length
            boundary_residual = u_t2[1:1,:,:] .- u_t2[end:end,:,:]
        else 
            forward_diff_first = (u_t2[:, 2:2, :, :] .- u_t2[:, 1:1, :, :]) ./ params.t_step_length
            forward_diff_last = (u_t2[:, end:end, :, :] .- u_t2[:, end-1:end-1, :, :]) ./ params.t_step_length
            central_diff = (u_t2[:, 3:end, :, :] .- u_t2[:, 1:end-2, :, :]) ./ (2 * params.t_step_length)
            ∂u_∂t = cat(forward_diff_first, central_diff, forward_diff_last; dims=2)
            boundary_residual = u_t2[1:1,:,:,:] .- u_t2[end:end,:,:,:]
        end
        ∂f_∂x, ∂u_∂xx = spatial_derivatives_batch(u_t2, params)
        return mean(abs2, ∂u_∂t .+ ∂f_∂x .- (params.ν .* ∂u_∂xx)) + mean(abs2, boundary_residual)
    end
    return BurgersFD_physics_loss
end

"""
    mse_loss_function(u::StatefulLuxLayer, target::AbstractArray, xt::AbstractArray)

Standard mean squared error loss.

# Arguments
- `u`: Neural network
- `target`: Ground truth values
- `u_t1`: Network inputs

# Returns
- MSE between network output and target
"""
function mse_loss_function(u::StatefulLuxLayer, target::AbstractArray, u_t1::AbstractArray; subsampling::Int=1)
    return MSELoss()(u(u_t1)[1:subsampling:end,:,:], target[1:subsampling:end,:,:])
end

function create_physics_loss(::Nothing)
    function physics_loss(u::StatefulLuxLayer, u_t1::AbstractArray)
        return 0.f0
    end
    return physics_loss
end 
    
"""
    autoregressive_loss(model::StatefulLuxLayer, (u0, target)::Tuple{AbstractArray, AbstractArray}, n_steps::Int, params::FDPhysicsLossParameters, λ::Float32)

Compute autoregressive loss for a model over multiple time steps.
# Arguments 
- `model`: StatefulLuxLayer model
- `u0`: Initial state (input data)
- `target`: Target data for comparison
- `n_steps`: Number of time steps to propagate
- `params`: FDPhysicsLossParameters struct containing physics parameters
- `λ`: Weighting factor for physics loss
# Returns
- Total loss combining data loss and physics-informed loss
"""
function autoregressive_loss(
    model::StatefulLuxLayer,
    (u0,target)::Tuple{AbstractArray,AbstractArray},
    n_steps::Int,
    params::FDPhysicsLossParameters,
    λ::Float32
)
    function autoregressive_propagate(model, u0::AbstractArray, n_steps::Int)
        # Initialize immutable tuple with initial state
        states = (u0,)
        
        # Build state sequence through tuple concatenation
        for _ in 1:n_steps
            u_next = model(states[end])
            states = (states..., u_next)  # Immutable update
        end
        
        # Combine into single array
        return cat(states...; dims=4)
    end
    
    states_arr = autoregressive_propagate(model, u0, n_steps)

    # Vectorized time derivative calculation
    Δt = params.t_step_length / n_steps
    forward_diff = (states_arr[:, :, :, 2:end] .- states_arr[:, :, :, 1:end-1]) ./ Δt
    central_diff = (states_arr[:, :, :, 3:end] .- states_arr[:, :, :, 1:end-2]) ./ (2Δt)
    du_dt = cat(forward_diff[:, :, :, 1:1], central_diff, forward_diff[:, :, :, end:end]; dims=4)

    # Batched physics calculations
    ∂f_∂x_batch, ∂u_∂xx_batch = spatial_derivatives_batch(states_arr[:, :, :, 1:end], params)
    
    # Vectorized residuals and boundary loss
    residuals = du_dt .+ ∂f_∂x_batch .- (params.ν .* ∂u_∂xx_batch)
    boundary_loss = mean(abs2, states_arr[:, 1:1, :, 2:end] .- states_arr[:, end:end, :, 2:end])
    physics_loss = mean(abs2, residuals) + boundary_loss

    # Data loss calculation
    data_loss = mean(abs2, states_arr[:, :, :, end] .- target)

    return data_loss + λ * physics_loss
end

function spatial_derivatives_batch(states::AbstractArray, params::FDPhysicsLossParameters)
    dims = size(states)
    
    # Maintain batch dimensions for matrix operations
    ∂f_∂x = reshape(params.M2_gpu * reshape(states, dims[1], :), dims)
    ∂u_∂xx = reshape(params.M1_gpu * reshape(states, dims[1], :), dims)
    
    return ∂f_∂x, ∂u_∂xx
end
function spatial_derivatives_batch(states::AbstractArray, params::ESM_PINO.SpectralPhysicsLossParameters)
    
    results = map(x -> spatial_derivative(x, params), eachslice(states; dims=4))

    df_dx_batch = cat(map(res -> res[1], results)...; dims=4)
    d2u_dx2_batch = cat(map(res -> res[2], results)...; dims=4)
    return df_dx_batch, d2u_dx2_batch
end

function spatial_derivative(u_t2::AbstractArray, params::ESM_PINO.SpectralPhysicsLossParameters)
    _, d2u_dx2 = ESM_PINO.spectral_derivative(u_t2, params.L)
    f = u_t2.^2 ./ 2
    f_hat = fft(f)
    f_hat = ESM_PINO.dealias(f_hat, params.L) 
    k = ESM_PINO.compute_k(f, params.L)
    df_dx_hat = im .* k .* f_hat
    df_dx = real(ifft(df_dx_hat))
    return df_dx, d2u_dx2
end

function create_autoregressive_loss_function(params::FDPhysicsLossParameters)
   function loss_function(model, ps, st, batch)
    u_net = StatefulLuxLayer{true}(model, ps, st)
    u0, target = batch
    loss = autoregressive_loss(
        u_net,
        (u0, target),
        3,  # n_steps
        params,
        0.1f0
    )
    return (loss, (st), ())
    end 
    return loss_function
end
"""
    select_loss_function()
Helper function to pass a valid loss function to Training.single_train_step.
Selects a loss function based on the provided physics-informed loss function, in the standard workflow generated with create_physics_loss.

# Arguments
- `PI_loss`: Physics-informed loss function (default is a zero loss function)
"""
function select_loss_function(PI_loss::Function=create_physics_loss(nothing); subsampling::Int=1, α::Float32=0.5f0)
    function loss_function(model::Lux.AbstractLuxLayer, ps::NamedTuple, st::NamedTuple, (u_t1, target_data)::Tuple{AbstractArray, AbstractArray}; α=α)
        u_net = StatefulLuxLayer{true}(model, ps, st)
        data_loss = mse_loss_function(u_net, target_data, u_t1; subsampling=subsampling)
        physics_loss = PI_loss(u_net, u_t1)
        loss = (1 - α) * physics_loss + α * data_loss
        return (loss,
            (st),
            (;physics_loss, data_loss)
        )
    end
    return loss_function
end