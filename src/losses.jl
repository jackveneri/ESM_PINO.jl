using AbstractFFTs, LinearAlgebra, Statistics, Adapt, Lux, CUDA
using Zygote:@adjoint 

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
- `k`: Wt2venumber array on GPU, reshaped for broadcastingt2
# Det2ails
- Handles even/odd array sizes differently
- Autt2matically converts to GPU array
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
    physics_informed_spectralt2loss_function(u::StatefulLuxLayer, u_t1::AbstractArray)

Compute Burgers' equation residual loss using spectral methods.

# Arguments
- `u`: StatefulLuxLayer network
- `u_t1`: Input state at time t

# Returns
- Spectral formulation residual error (mean squared residual)

# Requirements
- Global parameters: `ν` (viscosity), `L` (domain size), `Δt` (timestep)
- `t_max`, `N_t` (time discretization parameters)

# Equations
Implements residual for Burgers' equation:
∂u/∂t + u∂u/∂x = ν∂²u/∂x²
"""
@views function physics_informed_spectral_loss_function(
    u::StatefulLuxLayer, u_t1::AbstractArray)
    u_t2 = u(u_t1)
    dt = (t_max * Δt)/(N_t-1)
    ∂u_∂t = (u_t2 - u_t1) / dt
    _, d2u_dx2 = spectral_derivative(u_t1, L)
    f = u_t1.^2 ./ 2
    f_hat = fft(f)
    f_hat = dealias(f_hat, L) 
    k = compute_k(f, L)
    df_dx_hat = im .* k .* f_hat
    df_dx = real(ifft(df_dx_hat))
    nonlinear_term = df_dx
    residual_spectral = ∂u_∂t .+ nonlinear_term .- ν .* d2u_dx2
    return mean(abs2, residual_spectral)
end

"""
    physics_informed_loss_function(u::StatefulLuxLayer, u_t1::AbstractArray)

Compute PDE residual loss using finite differences.

# Arguments
- `u`: Neural network
- `u_t1`: Input state

# Returns
- Physics loss based on equation residual (mean squared residual)

# Requirements
- Precomputed derivative matrices `M1_gpu`, `M2_gpu`
- Global parameters: `Δx`, `ν`, `N_t`, `ts`

# Notes
- Uses finite difference for derivatives
- Assumes specific array slicing pattern (`Δx` steps)
"""
@views function physics_informed_loss_function(
    u::StatefulLuxLayer, u_t1::AbstractArray)
    u_t2 = reshape(u(u_t1),size(u_t1)[1],:)
    u_t2 = u_t2[1:Δx:end, 1, :]
    u_t1 = u_t1[1:Δx:end, 1, :]
    ∂u_∂t = ((N_t-1)/ts[end]) .* (u_t2 .- u_t1) / Δt
    f = 0.5 .* (u_t1.^2) 
    ∂f_∂x =  M2_gpu * f
    ∂u_∂xx = M1_gpu * u_t1
    return mean(abs2, ∂u_∂t .+ ∂f_∂x .- (ν .* ∂u_∂xx))
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
function mse_loss_function(u::StatefulLuxLayer, target::AbstractArray, u_t1::AbstractArray)
    return MSELoss()(u(u_t1), target)
end


"""
    loss_function(model, ps, st, (u_t1, target_data); α=0.5f0)

Combined finite differences physics-data loss function.

# Arguments
- `model`: Lux model
- `ps`: Model parameters
- `st`: Model state
- `u_t1`: Input data
- `target_data`: Training targets
- `α`: Loss weighting (0.5 = equal weighting)

# Returns
- Tuple containing:
  - Total loss
  - Updated state
  - Named tuple with loss components (physics_loss, data_loss)

# Notes
- Uses `StatefulLuxLayer` for network state management
- Compatible with Zygote AD
"""
function PINO_FD_loss_function(model, ps, st,(u_t1, target_data); α=0.5f0)
    u_net = StatefulLuxLayer{true}(model, ps, st)
    data_loss = mse_loss_function(u_net, target_data, u_t1)
    physics_loss = physics_informed_loss_function(u_net, u_t1)
    loss = (1 - α) * physics_loss + α * data_loss
    return (loss,
        (st),
        (;physics_loss, data_loss)
    )
end

"""
    loss_function(model, ps, st, (u_t1, target_data); α=0.5f0)

Combined spectral physics-data loss function.

# Arguments
- `model`: Lux model
- `ps`: Model parameters
- `st`: Model state
- `u_t1`: Input data
- `target_data`: Training targets
- `α`: Loss weighting (0.5 = equal weighting)

# Returns
- Tuple containing:
  - Total loss
  - Updated state
  - Named tuple with loss components (physics_loss, data_loss)

# Notes
- Uses `StatefulLuxLayer` for network state management
- Compatible with Zygote AD
"""
function PINO_spectral_loss_function(model, ps, st,(u_t1, target_data); α=0.5f0)
    u_net = StatefulLuxLayer{true}(model, ps, st)
    data_loss = mse_loss_function(u_net, target_data, u_t1)
    physics_loss = physics_informeds_pectral_loss_function(u_net, u_t1)
    loss = (1 - α) * physics_loss + α * data_loss
    return (loss,
        (st),
        (;physics_loss, data_loss)
    )
end
"""
    loss_function_just_data(model, ps, st, (u_t1, target_data))

Data-only loss function variant.

# Arguments
- `model`: Lux model
- `ps`: Model parameters
- `st`: Model state
- `u_t1`: Input data
- `target_data`: Training targets

# Returns
- Tuple with:
  - Data loss
  - Updated state
  - Named tuple with loss components (physics_loss=0, data_loss)
"""
function loss_function_just_data(model, ps, st,(u_t1, target_data))
    u_net = StatefulLuxLayer{true}(model, ps, st)
    data_loss = mse_loss_function(u_net, target_data, u_t1)
    physics_loss = 0.f0
    loss = data_loss
    return (loss,
        (st),
        (;physics_loss, data_loss)
    )
end