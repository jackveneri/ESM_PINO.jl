using LinearAlgebra, SparseArrays
"""
    Grid{T}

Dicretization grid for 1d finite difference schemes. 

# Initialization 

    Grid(x)

with `x` array or range with constant spacing 

# Fields 

* `N`
* `x::AbstractVector{T}`
* `Δx::T`
"""
struct Grid{T}
    N 
    x::AbstractVector{T}
    Δx::T
end 

function Grid(x) 
    N = length(x)
    Δx = abs(x[2] - x[1])
    return Grid(N, x, Δx)
end

abstract type AbstractFiniteDifferencesScheme{T} end 

"""
    NeumannFD{T}

Finite Difference Scheme matrix with Neumann Boundary Conditions, so that the derivative at the boundaries is zero

"""
struct NeumannFD{T} <: AbstractFiniteDifferencesScheme{T}
    M::T
end 

function NeumannFD(T::DataType, n::Integer, Δx::Number=1)
    M = diagm(-1=>(-1*ones(T, n-1)),1=>ones(T, n-1))
    M[1,2] = T(0)
    M[n,n-1] = T(0)
    M ./= T(2*Δx)
    NeumannFD(M)
end 

NeumannFD(grid::Grid{T}) where T = NeumannFD(T, grid.N, grid.Δx)

(FD::AbstractFiniteDifferencesScheme{T})(x::AbstractVector) where T = FD.M * x

"""
    PeriodicFD{T}

Finite Difference Scheme matrix with Periodic Boundary Conditions

"""
struct PeriodicFD{T} <: AbstractFiniteDifferencesScheme{T}
    M::T
end 

function PeriodicFD(T::DataType, n::Integer, Δx::Number=1)
    M = diagm(-1=>(-1*ones(T, n-1)),1=>ones(T, n-1))
    M[1,n] = T(-1)
    M[n,1] = T(1)
    M ./= T(2*Δx)
    PeriodicFD(M)
end 

PeriodicFD(grid::Grid{T}) where T = PeriodicFD(T, grid.N, grid.Δx)

struct BurgersFD{T} <: AbstractFiniteDifferencesScheme{T}
    M::T
    M2::T
end

function BurgersFD(T::DataType, n::Integer, Δx::Number=1)
    #M is the matrix for the second derivative
    M = diagm(-1=>ones(T,n-1),1=>ones(T,n-1),0=>T(-2).*ones(T,n))
    M[1,n] = T(1)
    M[n,1] = T(1)
    M ./= T(Δx^2)
    #M2 is the matrix for the first derivative using central differences
    M2 = diagm(-1=>(-1*ones(T, n-1)),1=>ones(T, n-1))
    M2[1,n] = T(-1)
    M2[n,1] = T(1)
    M2 ./= T(2*Δx)
    BurgersFD(M, M2)
end

BurgersFD(grid::Grid{T}) where T = BurgersFD(T, grid.N, grid.Δx)

x = LinRange(0.f0, 1.f0, 10)
grid = Grid(x)
M1, M2 = BurgersFD(grid).M, BurgersFD(grid).M2
M1_sparse, M2_sparse = sparse(M1), sparse(M2)

using FFTW, Plots, Random

# Function to generate one realization of the initial condition u0 ~ N(0, 625(-Δ+25I)^(-2))
function sample_u0(N::Int; L=1.0)
    dx = L / N
    x = dx .* (0:N-1)
    # Fourier frequencies in standard FFT ordering
    k = [0:div(N,2); -div(N,2)+1:-1]
    k = Float64.(k)
    # Eigenvalues in Fourier space: for mode k, -Δ has eigenvalue (2πk/L)^2.
    lam = 625 ./ (((2π * k / L).^2 .+ 25).^2)
    sqrtlam = sqrt.(lam)
    # Sample independent standard normal random numbers for real and imaginary parts
    ζ = ComplexF64.(randn(N), randn(N))
    ζ .= ζ .* sqrtlam
    # Enforce conjugate symmetry so that the field is real.
    ζ[1] = real(ζ[1])
    if iseven(N)
        ζ[div(N,2)+1] = real(ζ[div(N,2)+1])
    end
    for j in 2:div(N,2)
        ζ[N - j + 2] = conj(ζ[j])
    end
    # Inverse FFT (note: FFTW.ifft returns an unnormalized transform)
    u0 = real(ifft(ζ)) * N
    return x, u0
end

using OrdinaryDiffEq

N = 128
L = 1.0
x, u0 = sample_u0(N, L=L)
g = Grid(x)
M1, M2 = BurgersFD(g).M, BurgersFD(g).M2
M1_sparse, M2_sparse = sparse(M1), sparse(M2)

function burgers!(du, u, p, t)
    ν = p
    f = 0.5 .* u.^2	
    du .= (ν .* M1_sparse * u) .- (M2_sparse * f)
end

tspan = (0.,15.)
prob = ODEProblem(burgers!, u0, tspan, [0.001])
sol = solve(prob, Tsit5());

ts = tspan[1]:0.1:tspan[2]
anim = Plots.@animate for t ∈ ts
    plot(x,sol(t),ylims=(minimum(u0),maximum(u0)),label="t=$t")
end

gif(anim, "burgers.gif",fps=10)

using OrdinaryDiffEq, JLD2

# Define simulation parameters
N_sim = 100
N = 128                      # Number of spatial grid points
tspan = (0.0, 3.0)
ts = collect(tspan[1]:tspan[2]/127:tspan[2])
N_t = length(ts)              # Number of time steps

# Preallocate a 3D array with dimensions (time, space, simulation)
results = Array{Float64}(undef, N_t, N, N_sim)

# Run the ensemble simulations and store each solution in the 3D array
for sim in 1:N_sim
    x, u0 = sample_u0(N, L=1.0)  # generate initial condition
    prob = ODEProblem(burgers!, u0, tspan, [0.001])
    sol = solve(prob, Tsit5())
    
    for (i, t) in enumerate(ts)
        results[i, :, sim] = sol(t)
    end
end

# Save the 3D array to disk for later use, for example using JLD2:
@save "burgers_results.jld2" results ts x
