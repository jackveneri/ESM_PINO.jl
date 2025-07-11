root = dirname(@__DIR__)
using Pkg
Pkg.activate(root)
Pkg.instantiate()
using ESM_PINO, JLD2, CairoMakie, Printf, Statistics, QG3, NetCDF, Dates, CFTime, Lux, CUDA, LuxCUDA, Random

gdev = gpu_device()
cdev = cpu_device()

@load string(root, "/data/t21-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
qg3p = CUDA.@allowscalar QG3Model(qg3ppars)
@load string(root, "/data/t21-precomputed-S.jld2") S
S = CuArray(S)


@load string(root,"/data/solq.jld2") solu
solu = CuArray(cat(solu...,dims=4))
ggsh2 = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=size(solu,4))
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=size(solu,4))
solu = QG3.transform_grid(solu, shgg)
solu = permutedims(solu,(2,3,1,4))
solu,  μ, σ = ESM_PINO.normalize_data(solu)

function create_QG3_physics_loss_test(params::ESM_PINO.QG3_Physics_Parameters)
    @views function QG3_physics_loss(q_1::AbstractArray{T,4}, q_0::AbstractArray{T,4}) where T<:Real
        q_pred = q_1 .* params.σ .+ params.μ
        bc_loss = mean(abs2, q_pred[:,1:1,:,:] .- q_pred[:,end:end,:,:])
        q_0_denormalized = q_0 .* params.σ .+ params.μ
        ∂u_∂t = (q_pred .- q_0_denormalized) ./ params.dt
        ∂u_∂t = permutedims(∂u_∂t, (3, 1, 2, 4)) 
        q_pred_perm = permutedims(q_pred, (3, 1, 2, 4))
        q_pred_perm_new = transform_SH(q_pred_perm, params.ggsh)
        
        # Process each batch sample individually
        results = map(sample -> QG3.QG3MM_gpu(sample, (params.qg3p, params.S), (0,1)), eachslice(q_pred_perm_new; dims=4))
        
        # Combine results without splatting
        rhs_new = reduce((acc, x) -> cat(acc, x; dims=4), results)
        #rhs = cat(results...; dims=4)
        
        rhs_newer = transform_grid(rhs_new, params.shgg)

        #rhs_final = permutedims(rhs_newer, (2, 3, 1, 4))
        residual = ∂u_∂t .- rhs_newer 
        return mean(abs2, residual) , bc_loss
    end    
end

q_1 = solu[:,:,:,2:end]
q_0 = solu[:,:,:,1:end-1]
@assert size(q_0,4)==size(q_1,4)
ggsh_test = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=size(q_0,4))
shgg_test = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=size(q_0,4))


params = ESM_PINO.QG3_Physics_Parameters(1, qg3p, S, ggsh_test, shgg_test, μ, σ)
test_loss = create_QG3_physics_loss_test(params)

using BenchmarkTools
@btime test_loss(q_1,q_0)