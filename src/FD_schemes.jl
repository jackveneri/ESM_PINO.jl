"""
    Grid{T}

Discretization grid for 1D finite difference schemes.

# Initialization

`Grid(x)`

with `x` an array or range with constant spacing.

# Arguments
- `x::AbstractVector{T}`: Discretization points, assumed uniformly spaced.

# Fields
- `N::Int`: Number of grid points
- `x::AbstractVector{T}`: Coordinates of grid points
- `Δx::T`: Grid spacing, computed from `x`

# Details
- Computes `Δx` as the absolute difference between the first two grid points.
- Useful for constructing finite-difference scheme matrices.
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

Finite-difference operator for the first derivative with **Neumann boundary conditions**
(enforcing zero derivative at the boundaries).

# Initialization

`NeumannFD(T, n, Δx=1)`  
`NeumannFD(grid::Grid{T})`

# Arguments
- `T::DataType`: Element type (e.g. `Float64`, `Float32`)
- `n::Integer`: Number of grid points
- `Δx::Number`: Grid spacing (default: 1)
- `grid::Grid{T}`: Grid object containing `N` and `Δx`

# Fields
- `M::AbstractMatrix{T}`: Finite-difference matrix representing the derivative operator

# Details
- Implements central differences for the interior and modifies boundary rows to enforce zero slope.
- The returned operator approximates ∂/∂x with Neumann BCs.
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

Finite-difference operator for the first derivative with **periodic boundary conditions**.

# Initialization

`PeriodicFD(T, n, Δx=1)`  
`PeriodicFD(grid::Grid{T})`

# Arguments
- `T::DataType`: Element type
- `n::Integer`: Number of grid points
- `Δx::Number`: Grid spacing (default: 1)
- `grid::Grid{T}`: Grid object

# Fields
- `M::AbstractMatrix{T}`: Finite-difference matrix representing the derivative operator

# Details
- Implements central differences and wraps the stencil at the domain boundaries (periodic).
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

"""
    BurgersFD{T}

Finite-difference matrices for discretising the 1D Burgers equation using
**periodic boundary conditions**.

# Initialization

`BurgersFD(T, n, Δx=1)`  
`BurgersFD(grid::Grid{T})`

# Arguments
- `T::DataType`: Element type
- `n::Integer`: Number of grid points
- `Δx::Number`: Grid spacing
- `grid::Grid{T}`: Grid object

# Fields
- `M::AbstractMatrix{T}`: Second-derivative (Laplacian) matrix with periodic BCs
- `M2::AbstractMatrix{T}`: First-derivative (central differences) matrix with periodic BCs

# Details
- `M` approximates ∂²/∂x² with periodic wrap-around.
- `M2` approximates ∂/∂x using central differences and periodic wrap-around.
- Use these matrices for semi-discrete formulations of Burgers’ equation.
"""
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
    M2 = diagm(-1=>T(-1).*ones(T, n-1),1=>ones(T, n-1))
    M2[1,n] = T(-1)
    M2[n,1] = T(1)
    M2 ./= T(2*Δx)
    BurgersFD(M, M2)
end

BurgersFD(grid::Grid{T}) where T = BurgersFD(T, grid.N, grid.Δx)

"""
    BurgersFD_Dirichlet{T}

Finite-difference matrices for the 1D Burgers equation with **Dirichlet boundary conditions**.

# Initialization

`BurgersFD_Dirichlet(T, n, Δx=1)`  
`BurgersFD_Dirichlet(grid::Grid{T})`

# Arguments
- `T::DataType`: Element type
- `n::Integer`: Number of grid points
- `Δx::Number`: Grid spacing
- `grid::Grid{T}`: Grid object

# Fields
- `M::AbstractMatrix{T}`: Second-derivative matrix with Dirichlet enforcement at boundaries
- `M2::AbstractMatrix{T}`: First-derivative matrix with boundary rows/cols zeroed

# Details
- Boundary rows/columns are zeroed to reflect fixed-value (Dirichlet) conditions.
- Intended for Burgers’ problems with fixed boundary values (e.g., u=0 at domain ends).
"""
struct BurgersFD_Dirichlet{T} <: AbstractFiniteDifferencesScheme{T}
    M::T
    M2::T
end

function BurgersFD_Dirichlet(T::DataType, n::Integer, Δx::Number=1)
    #M is the matrix for the second derivative
    M = diagm(-1=>ones(T,n-1),1=>ones(T,n-1),0=>T(-2).*ones(T,n))
    M[1,:] .= 0
    M[end,:] .= 0
    M[:,1] .= 0
    M[:,end] .= 0
    M ./= T(Δx^2)
    #M2 is the matrix for the first derivative using central differences
    M2 = diagm(-1=>(-1*ones(T, n-1)),1=>ones(T, n-1))
    M2[1,:] .= 0
    M2[end,:] .= 0
    M2[:,1] .= 0
    M2[:,end] .= 0
    M2 ./= T(2*Δx)
    BurgersFD_Dirichlet(M, M2)
end

BurgersFD_Dirichlet(grid::Grid{T}) where T = BurgersFD_Dirichlet(T, grid.N, grid.Δx)
"""
    BurgersFD2{T}

Finite-difference matrices for the 1D Burgers equation with **periodic boundary conditions**
using a **backward-difference** discretisation for the convective (first-derivative) term.

# Initialization

`BurgersFD2(T, n, Δx=1)`  
`BurgersFD2(grid::Grid{T})`

# Arguments
- `T::DataType`: Element type
- `n::Integer`: Number of grid points
- `Δx::Number`: Grid spacing
- `grid::Grid{T}`: Grid object

# Fields
- `M::AbstractMatrix{T}`: Second-derivative (Laplacian) matrix with periodic BCs
- `M2::AbstractMatrix{T}`: First-derivative matrix using backward differences (periodic BCs)

# Details
- `M` is identical in form to BurgersFD's Laplacian (periodic).
- `M2` is a backward-difference approximation of ∂/∂x; can improve stability for convection-dominated flows.
"""
struct BurgersFD2{T} <: AbstractFiniteDifferencesScheme{T}
    M::T
    M2::T
end

function BurgersFD2(T::DataType, n::Integer, Δx::Number=1)
    #M is the matrix for the second derivative
    M = diagm(-1=>ones(T,n-1),1=>ones(T,n-1),0=>T(-2).*ones(T,n))
    M[1,n] = T(1)
    M[n,1] = T(1)
    M ./= T(Δx^2)
    #M2 is the matrix for the first derivative using backward differences
    M2 = diagm(-1=>T(-1).*ones(T, n-1),0=>ones(T, n))
    M2[1,n] = T(-1)
    M2 ./= T(Δx)
    BurgersFD2(M, M2)
end

BurgersFD2(grid::Grid{T}) where T = BurgersFD2(T, grid.N, grid.Δx)

