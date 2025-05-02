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

# Retuu_t1ns
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
    
    #u_t1Add singleton dimensions to match input dimensions
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
    L::Float64
    N_t::Int
    t_max::Float64
    t_min::Float64
    Δt::Float64
    x_σ::Float64
    x_μ::Float64
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
        u_t1 = u_t1 .* params.x_σ .+ params.x_μ
        u_t2 = u_t2 .* params.x_σ .+ params.x_μ 
        dt = ((params.t_max - params.t_min) * params.Δt)/(params.N_t-1)
        ∂u_∂t = (u_t2 .- u_t1) / dt
        _, d2u_dx2 = spectral_derivative(u_t1, params.L)
        f = u_t1.^2 ./ 2
        f_hat = fft(f)
        f_hat = dealias(f_hat, params.L) 
        k = compute_k(f, params.L)
        df_dx_hat = im .* k .* f_hat
        df_dx = real(ifft(df_dx_hat))
        nonlinear_term = df_dx
        residual_spectral = ∂u_∂t .+ nonlinear_term .- params.ν .* d2u_dx2
        return mean(abs2, residual_spectral)
    end    
end

"""
    physics_informed_spectral_loss_function(u::StatefulLuxLayer, u_t1::AbstractArray)

Compute Burgers' equation residual loss using spectral methods.

# Arguments
- `u`: StatefulLuxLayer network
- `u_t1`: Input state at time t

# Returns
- Spectral formulation residual error (mean squared residual)

# Requirements
- Global parameters: `ν` (viscosity), `L` (domain size), `Δt` (timestep)
- `t_max`, `t_min`, `N_t` (time discretization parameters)

# Equations
Implements residual for Burgers' equation:
∂u/∂t + u∂u/∂x = ν∂²u/∂x²
"""
@views function physics_informed_spectral_loss_function(
    u::StatefulLuxLayer, u_t1::AbstractArray)
    u_t2 = u(u_t1)
    u_t1 = u_t1 .* x_σ .+ x_μ
    u_t2 = u_t2 .* x_σ .+ x_μ 
    dt = ((t_max - t_min) * Δt)/(N_t-1)
    ∂u_∂t = (u_t2 .- u_t1) / dt
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
    FDPhysicsLossParameters(ν::Float64, N_t::Int, t_max::Float64, t_min::Float64, Δt::Float64, x_σ::Float64, x_μ::Float64, M1_gpu::AbstractArray, M2_gpu::AbstractArray)
Create a struct to hold parameters for finite difference physics loss.

# Fields
- `ν`: Viscosity (scalar)
- `N_t`: Number of time steps (integer)
- `t_max`: Maximum time (scalar)
- `t_min`: Minimum time (scalar)
- `Δt`: Time step size (scalar)
- `x_σ`: Standard deviation for normalization (scalar)
- `x_μ`: Mean for normalization (scalar)
- `M1_gpu`: Second derivative FD matrix (GPU array)
- `M2_gpu`: First derivative FD matrix (GPU array)
"""
struct FDPhysicsLossParameters
    ν::Float64
    N_t::Int
    t_max::Float64
    t_min::Float64
    Δt::Float64
    x_σ::Float64
    x_μ::Float64
    M1_gpu::AbstractArray
    M2_gpu::AbstractArray
end

function create_physics_loss(params::FDPhysicsLossParameters)
    # Capture `params` in the closure
    function BurgersFD_physics_loss(u::StatefulLuxLayer, u_t1::AbstractArray)
        # Access params via the captured struct
        u_t2 = reshape(u(u_t1), size(u_t1)[1], :)
        u_t2 = u_t2[:, 1, :]
        u_t1 = u_t1[:, 1, :]
        u_t1 = u_t1 .* params.x_σ .+ params.x_μ
        u_t2 = u_t2 .* params.x_σ .+ params.x_μ
        ∂u_∂t = ((params.N_t-1)/(params.t_max - params.t_min)) .* (u_t2 .- u_t1) ./ params.Δt
        f = 0.5 .* (u_t1.^2) 
        ∂f_∂x =  params.M2_gpu * f
        ∂u_∂xx = params.M1_gpu * u_t1
        return mean(abs2, ∂u_∂t .+ ∂f_∂x .- (params.ν .* ∂u_∂xx))
    end
    return BurgersFD_physics_loss
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
- Global parameters: `ν`, `N_t`, `t_max`, `t_min`, `Δt`, `x_σ`, `x_μ`

# Notes
- Uses finite difference for derivatives
- Assumes specific array slicing pattern (`Δx` steps)
"""
@views function physics_informed_loss_function(
    u::StatefulLuxLayer, u_t1::AbstractArray)
    u_t2 = reshape(u(u_t1),size(u_t1)[1],:)
    u_t2 = u_t2[:, 1, :]
    u_t1 = u_t1[:, 1, :]
    u_t1 = u_t1 .* x_σ .+ x_μ
    u_t2 = u_t2 .* x_σ .+ x_μ
    ∂u_∂t = ((N_t-1)/(t_max - t_min)) .* (u_t2 .- u_t1) ./ Δt
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
function PINO_FD_loss_function(model::Lux.AbstractLuxLayer, ps::NamedTuple, st::NamedTuple,(u_t1, target_data)::Tuple{AbstractArray, AbstractArray}; α::Float32=0.5f0)
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
function PINO_spectral_loss_function(model::Lux.AbstractLuxLayer, ps::NamedTuple, st::NamedTuple,(u_t1, target_data)::Tuple{AbstractArray, AbstractArray}; α::Float32=0.5f0)
    u_net = StatefulLuxLayer{true}(model, ps, st)
    data_loss = mse_loss_function(u_net, target_data, u_t1)
    physics_loss = physics_informed_spectral_loss_function(u_net, u_t1)
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
function loss_function_just_data(model::Lux.AbstractLuxLayer, ps::NamedTuple, st::NamedTuple,(u_t1, target_data)::Tuple{AbstractArray, AbstractArray})
    u_net = StatefulLuxLayer{true}(model, ps, st)
    data_loss = mse_loss_function(u_net, target_data, u_t1)
    physics_loss = 0.f0
    loss = data_loss
    return (loss,
        (st),
        (;physics_loss, data_loss)
    )
end

function create_physics_loss(::Nothing)
    function physics_loss(u::StatefulLuxLayer, u_t1::AbstractArray)
        return 0.f0
    end
    return physics_loss
end 
    
"""
    select_loss_function()
Helper function to pass a valid loss function to Training.single_train_step.
Selects a loss function based on the provided physics-informed loss function, in the standard workflow generated with create_physics_loss.

# Arguments
- `PI_loss`: Physics-informed loss function (default is a zero loss function)
"""
function select_loss_function(PI_loss::Function=create_physics_loss(nothing))
    function loss_function(model::Lux.AbstractLuxLayer, ps::NamedTuple, st::NamedTuple, (u_t1, target_data)::Tuple{AbstractArray, AbstractArray}; α::Float32=0.5f0)
        u_net = StatefulLuxLayer{true}(model, ps, st)
        data_loss = mse_loss_function(u_net, target_data, u_t1)
        physics_loss = PI_loss(u_net, u_t1)
        loss = (1 - α) * physics_loss + α * data_loss
        return (loss,
            (st),
            (;physics_loss, data_loss)
        )
    end
    return loss_function
end
"""
    physics_informed_loss_QG3(u::StatefulLuxLayer, q_0::AbstractArray)
Compute residual loss for QG3 equation.

# Arguments
- `u`: Neural network (StatefulLuxLayer)
- `q_0`: Initial state (input data)

# Returns
- Residual loss (mean squared residual)

# Requirements
- Precomputed right-hand side `rhs` (global variable) computed using QG3.QG3MM_gpu
- Precomputed time step `dt` (global variable) in QG3 units
"""
@views function physics_informed_loss_QG3(u::StatefulLuxLayer, q_0::AbstractArray)
    q_pred = u(q_0)
    ∂u_∂t = (q_pred .- q_0) ./ dt
    residual = ∂u_∂t .- rhs
    return mean(abs2, residual)
end

"""
    QG3_loss_function(model, ps, st, (u_t1, target_data); α=0.5f0)
Combined physics-data loss function for QG3.

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
"""
function QG3_loss_function(model::AbstractLuxLayer, ps::NamedTuple, st::NamedTuple, (u_t1, target_data)::Tuple{AbstractArray, AbstractArray}; α::Float32=0.5f0)
    u_net = StatefulLuxLayer{true}(model, ps, st)
    data_loss = mse_loss_function(u_net, target_data, u_t1)
    physics_loss = physics_informed_loss_QG3(u_net, u_t1)
    loss = (1 - α) * physics_loss + α * data_loss
    return (loss,
        (st),
        (;physics_loss, data_loss)
    )
end
"""
    QG3_Physics_Parameters(dt::Float64, rhs::AbstractArray)
Create a struct to hold parameters for QG3 physics loss.

# Fields
- `dt`: Time step (scalar)
- `rhs`: Right-hand side of the QG3 equation (array)
"""
struct QG3_Physics_Parameters
    dt::Float64
    rhs::AbstractArray
end
"""
    create_QG3_physics_loss()
Helper function to create a QG3 physics loss function.

# Arguments
- `params`: parameters struct, pass nothing to create a zero loss function.
"""
function create_QG3_physics_loss(params::QG3_Physics_Parameters)
    # Capture `params` in the closure
    function QG3_physics_loss(u::StatefulLuxLayer, q_0::AbstractArray)
        # Access params via the captured struct
        q_pred = u(q_0)
        ∂u_∂t = (q_pred .- q_0) ./ params.dt
        residual = ∂u_∂t .- params.rhs
        return mean(abs2, residual)
    end    
end
function create_QG3_physics_loss(::Nothing)
    function QG3_physics_loss(u::StatefulLuxLayer, q_0::AbstractArray)
        return 0.f0
    end
    return QG3_physics_loss
end
"""
    select_QG3_loss_function(PI_loss::Function=create_QG3_physics_loss(nothing))
Helper function to pass a valid QG3 loss function to Training.single_train_step.
Selects a loss function based on the provided physics-informed loss function, in the standard workflow generated with create_QG3_physics_loss.

# Arguments
- `PI_loss`: Physics-informed loss function (default is a zero loss function)
"""
function select_QG3_loss_function(PI_loss::Function=create_QG3_physics_loss(nothing))
    function QG3_loss_function(model::Lux.AbstractLuxLayer, ps::NamedTuple, st::NamedTuple, (u_t1, target_data)::Tuple{AbstractArray, AbstractArray}; α::Float32=0.5f0)
        u_net = StatefulLuxLayer{true}(model, ps, st)
        data_loss = mse_loss_function(u_net, target_data, u_t1)
        physics_loss = PI_loss(u_net, u_t1)
        loss = (1 - α) * physics_loss + α * data_loss
        return (loss,
            (st),
            (;physics_loss, data_loss)
        )
    end
    return QG3_loss_function
end