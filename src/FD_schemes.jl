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
    M2 = diagm(-1=>T(-1).*ones(T, n-1),1=>ones(T, n-1))
    M2[1,n] = T(-1)
    M2[n,1] = T(1)
    M2 ./= T(2*Δx)
    BurgersFD(M, M2)
end

BurgersFD(grid::Grid{T}) where T = BurgersFD(T, grid.N, grid.Δx)

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

