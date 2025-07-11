root = dirname(@__DIR__)
using Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using ESM_PINO, FFTW, Plots, Random, JLD2, OrdinaryDiffEq, SparseArrays

function sample_initial_condition(N::Int; L::Float64=1.0)            
    dx = L / N         # Spatial resolution
    x = dx .* (0:N-1)  # Grid points on [0, L)

    # Compute Fourier wave numbers for a periodic domain:
    # For even N: 0, 1, 2, …, N/2, -N/2+1, …, -1.
    k = [0:div(N,2); -div(N,2)+1:-1]

    # Multiplier is the square root of the covariance eigenvalue:
    # sqrt(625/(k^2+25)^2) = 25/(k^2+25)
    multiplier = 25 ./ (k.^2 .+ 25)

    # Allocate Fourier coefficient array
    fhat = zeros(ComplexF64, N)

    # Mode 0 must be real
    fhat[1] = multiplier[1] * randn()

    half = div(N, 2)
    for j in 2:half
        noise = (randn() + im * randn()) / sqrt(2)
        fhat[j] = multiplier[j] * noise
        # Enforce Hermitian symmetry
        fhat[N - j + 2] = conj(fhat[j])
    end

    # For even N, the Nyquist mode must be real
    if iseven(N)
        fhat[half+1] = multiplier[half+1] * randn()
    end

    # Inverse FFT to get the spatial field.
    u0 = real(ifft(fhat))
    return x, u0
end

N = 512
L = 1.0
x, u0 = sample_initial_condition(N, L=L)
u0 *= 70
#x = push!(Array(x), L)
#u0 = push!(u0, u0[1])
g = ESM_PINO.Grid(x)
M1, M2 = ESM_PINO.BurgersFD(g).M, ESM_PINO.BurgersFD(g).M2
M1_sparse, M2_sparse = sparse(M1), sparse(M2)

function burgers!(du, u, p, t)
    ν, = p	
    f = 0.5 .* (u.^2)
    du .= (ν .* M1_sparse * u) .- (M2_sparse * f)
end

tspan = (0.,30.)
prob = ODEProblem(burgers!, u0, tspan, 0.001)
@time sol = solve(prob, Rodas5P(), dt=0.0001, reltol=1e-6, abstol=1e-8)

ts = tspan[1]:0.1:tspan[2]
anim = Plots.@animate for t ∈ ts
    plot(x,sol(t),ylims=(minimum(u0),maximum(u0)),label="t=$t")
end

gif(anim, "burgers.gif",fps=20)


# Define simulation parameters
N_sim = 100
N = 512          
tspan = (0.0, 3.0)
N_t = 1024
ts = collect(tspan[1]:tspan[2]/(N_t-1):tspan[2])

# Preallocate a 3D array with dimensions (time, space, simulation)
results = Array{Float64}(undef, N_t, N, N_sim)

# Run the ensemble simulations and store each solution in the 3D array
for sim in 1:N_sim
    x, u0 = sample_initial_condition(N, L=1.0)  # generate initial condition
    u0 = u0 .* 100  # scale the initial condition
    #x = push!(Array(x), 1.0)
    #u0 = push!(u0, u0[1])
    prob = ODEProblem(burgers!, u0, tspan, [0.001])
    sol = solve(prob, Rodas5P(), dt=0.0001, reltol=1e-6, abstol=1e-8)
    
    for (i, t) in enumerate(ts)
        results[i, :, sim] = sol(t)
    end
    if sim % 100 == 0
        println("Simulation $sim completed")
    end
end

# Save the 3D array to disk for later use, for example using JLD2:
@save string(root,"/burgers_results_test.jld2") results ts x
