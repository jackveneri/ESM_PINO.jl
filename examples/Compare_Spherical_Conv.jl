"""
compare_spherical_conv.jl
=========================
Julia side of the SphericalConv equivalence test.

Loads the .npy arrays saved by compare_spherical_conv.py and reproduces
the same spectral contraction using your QG3 SHT backend, then compares
the outputs element-wise.

Steps
-----
1.  Load shared input / weight / reference output from Python (via NPZ/NPY).
2.  Run your SphericalConv forward pass (CPU path, driscoll_healy).
3.  Compare intermediate results (SHT coefficients, post-contraction,
    final grid output) with the reference values from torch-harmonics.

Requirements
------------
  - NPZ.jl  (or your preferred .npy reader, e.g. via JLD2 after conversion)
  - Your QG3 / ESM_PINO packages loaded in the environment

Adjust the `using` block and constructor calls to match your project setup.
"""
dir = @__DIR__
using Pkg
Pkg.activate(dir)
Pkg.instantiate()
# ─── Load packages ────────────────────────────────────────────────────────────
using NPZ               # pip-equivalent: ]add NPZ
using LinearAlgebra
# using QG3, ESM_PINO   # ← uncomment once environment is set up

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (must match compare_spherical_conv.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
const NLAT  = 8
const NLON  = 16
const MODES = 4
const C_IN  = 2
const C_OUT = 2
const BATCH = 1

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load reference arrays from Python
# ─────────────────────────────────────────────────────────────────────────────
# Adjust paths to wherever you saved the .npy files
root = dirname(dirname(dirname(@__DIR__)))
input_py    = npzread(string(root, "/input_x.npy"))        # (B, C, lat, lon) Float64
weight_py   = npzread(string(root, "/weight_W.npy"))       # (C_out, C_in, L) Float64
x_hat_ri_py = npzread(string(root, "/x_hat.npy"))         # (B, C, L, M, 2)  real/imag
x_conv_ri_py= npzread(string(root, "/x_conv_ri.npy"))     # (B, C_out, L, M, 2)
output_py   = npzread(string(root, "/output_x_out.npy"))  # (B, C_out, lat, lon)

println("Loaded reference arrays:")
println("  input_py    : ", size(input_py))
println("  weight_py   : ", size(weight_py))
println("  x_hat_ri_py : ", size(x_hat_ri_py))
println("  output_py   : ", size(output_py))

# ─────────────────────────────────────────────────────────────────────────────
# Helper: convert Python (B, C, lat, lon) → Julia (lat, lon, C, B)
# ─────────────────────────────────────────────────────────────────────────────
function py_to_julia_input(A::Array{T,4}) where T
    # Python: (B, C, lat, lon) → Julia: (lat, lon, C, B)
    return permutedims(A, (3, 4, 2, 1))
end

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Set up your Julia SphericalConv layer
#
#     Uncomment and adapt these lines once your packages are available.
#     Replace `pars` with actual QG3ModelParameters matching NLAT/NLON/MODES.
# ─────────────────────────────────────────────────────────────────────────────


using QG3, ESM_PINO, Random, Lux
ESM_PINOQG3 = Base.get_extension(ESM_PINO, :ESM_PINOQG3Ext)

pars  = ESM_PINOQG3.qg3pars_constructor_helper(MODES, NLAT, NF=Float64)   # fill in your constructor
layer = ESM_PINO.SphericalConv(pars, C_IN;
            modes=MODES,
            batch_size=BATCH,
            gpu=false,
            operator_type=:driscoll_healy,
            gain=2.0)

# Pre-initialise weights with the SAME values as Python
# weight_py shape (Python): (C_out, C_in, L) – already Julia-compatible axes
ps = (weight = Float64.(weight_py),)
st = Lux.initialstates(Random.default_rng(), layer)

# Convert Python input → Julia layout
x_jl = Float64.(py_to_julia_input(input_py))   # (lat, lon, C, B)

# Forward pass
x_out_jl, res_out_jl, _ = layer(x_jl, ps, st)
# x_out_jl shape: (lat, lon, C_out, B)

# Convert back to Python layout for comparison: (B, C, lat, lon)
x_out_jl_py_layout = permutedims(x_out_jl, (4, 3, 1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Intermediate diagnostic: compare SHT coefficients
#
#     Your SHT output after reorderQG3 should match x_hat from Python once
#     you account for the different coefficient layouts.
#
#     Python x_hat layout: (B, C, L, M) complex
#     Julia  x_tr  layout: (C, L, M, B) real   ← 2D real/imag in M dim
#
#     The reorderQG3 step applies circular column shifts to the M dimension.
#     You need to verify that after the reorder, Julia's real/imag pairs
#     correspond to Python's Re/Im for each (l, m) pair.
# ─────────────────────────────────────────────────────────────────────────────

function check_sht_layout(x_hat_ri_py, x_tr_jl)
    # x_hat_ri_py: (B, C, L, M, 2)  – Python real/imag
    # x_tr_jl:     (C, L, M, B)     – Julia real-valued (real/imag interleaved)
    #
    # Reconstruct complex from Python:
    x_hat_py_c = x_hat_ri_py[:, :, :, :, 1] .+ 1im .* x_hat_ri_py[:, :, :, :, 2]
    # x_hat_py_c: (B, C, L, M)
    
    # From Julia: for m=0, coeffs are all real. For m>0, real part is at col 2m-1,
    # imag part is at col 2m (this depends on your QG3 SHT convention).
    println("  SHT layout check: inspect manually")
    println("  Python x_hat[1,1,1,1] = ", x_hat_py_c[1,1,1,1])
    println("  Compare with Julia x_tr[1,1,1,1] (after reorder) = ??")
    return x_hat_py_c
end

# Reconstruct complex coeffs from saved Python arrays (for manual inspection)
x_hat_py_c = x_hat_ri_py[:, :, :, :, 1] .+ 1im .* x_hat_ri_py[:, :, :, :, 2]
println("\nPython x_hat (B=1, C=1, L=1, all M): ", x_hat_py_c[1, 1, 1, :])

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Pure-Julia spectral contraction (no SHT, just the einsum)
#
#     Given that SHT outputs are aligned, replicate the driscoll_healy step:
#         Python: einsum("bixy,kix->bkxy", x_hat, W)
#         Julia:  ein"ilmb,oil->olmb"(x_tr, W)
#
#     Both contract over (C_in=i) and (L=x in Python, l in Julia).
#     They are equivalent IF x_hat_py[b,i,l,m] == x_tr_jl[i,l,m,b].
# ─────────────────────────────────────────────────────────────────────────────

# Replicate Python einsum in Julia on Python's arrays
# x_hat_py_c: (B, C_in, L, M)
# weight_py:  (C_out, C_in, L)

function py_einsum_in_julia(x_hat, W)
    # Python: "bixy,kix->bkxy"  →  contract over i (C_in) and x (L)
    # x_hat: (B, C_in, L, M)
    # W:     (C_out, C_in, L)
    B, Ci, L, M = size(x_hat)
    Co = size(W, 1)
    out = zeros(eltype(x_hat), B, Co, L, M)
    for b in 1:B, k in 1:Co, x in 1:L, y in 1:M
        for i in 1:Ci
            out[b, k, x, y] += x_hat[b, i, x, y] * W[k, i, x]
        end
    end
    return out
end

# Apply on complex arrays (treat real weights as complex with imag=0)
W_complex = ComplexF64.(weight_py)
x_conv_ref = py_einsum_in_julia(x_hat_py_c, W_complex)

println("\nReference x_conv (B=1,C=1,L=1,all M): ", x_conv_ref[1, 1, 1, :])

x_conv_ri_c = x_conv_ri_py[:,:,:,:,1] .+ 1im .* x_conv_ri_py[:,:,:,:,2]
diff = maximum(abs.(x_conv_ref .- x_conv_ri_c))
println("Max diff between replicated and saved Python x_conv: ", diff)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Comparison function  (call after running the full Julia forward pass)
# ─────────────────────────────────────────────────────────────────────────────

function compare_outputs(output_julia, output_python; tol=1e-6)
    # output_julia:  (B, C, lat, lon)  or  (lat, lon, C, B) – adjust as needed
    # output_python: (B, C, lat, lon)
    diff = output_julia .- output_python
    max_abs = maximum(abs.(diff))
    rel_err = max_abs / (maximum(abs.(output_python)) + eps())
    println("\n── Output comparison ──────────────────────────")
    println("  Max absolute difference : ", max_abs)
    println("  Relative error          : ", rel_err)
    if max_abs < tol
        println("  ✓  Outputs match within tol=$(tol)")
    else
        println("  ✗  Outputs differ — check SHT coefficient layout and reorder logic")
        # Find largest discrepancy location
        idx = argmax(abs.(diff))
        println("  Largest diff at index: ", idx)
        println("  Julia value  : ", output_julia[idx])
        println("  Python value : ", output_python[idx])
    end
    return max_abs, rel_err
end

# ─── Uncomment once Julia layer is running ────────────────────────────────────
compare_outputs(x_out_jl_py_layout, output_py)

println("""
─────────────────────────────────────────────────────────────────
Next steps:
  1. Uncomment the layer construction block (section 2) and run.
  2. Call compare_outputs(x_out_jl_py_layout, output_py) at the end.
  3. If outputs differ, use section 3 to diagnose whether the mismatch
     is in the SHT step or the spectral contraction.
─────────────────────────────────────────────────────────────────
""")