root = dirname(@__DIR__)
dir = @__DIR__
using Pkg
Pkg.activate(dir)
Pkg.instantiate()
using ESM_PINO, Printf, CUDA, OnlineStats, Lux, LuxCUDA, Random, Statistics, MLUtils, Optimisers, ParameterSchedulers, QG3, NetCDF, Dates, CFTime, JLD2

const ESM_PINOQG3 = Base.get_extension(ESM_PINO, :ESM_PINOQG3Ext)
const gdev = gpu_device()
const cdev = cpu_device()

@load string(root, "/data/t42-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
qg3p = CUDA.@allowscalar QG3Model(qg3ppars)
@load string(root, "/data/t42-precomputed-S.jld2") S
S = CUDA.@allowscalar QG3.reorder_SH_gpu(S, qg3ppars)

# initial conditions for streamfunction and vorticity
N_sims = 1000
@load string(root,"/data/t42_qg3_data_SH_CPU.jld2") q
q = QG3.reorder_SH_gpu(q[:,:,:,1:N_sims+2], qg3ppars)
solu = permutedims(QG3.transform_grid_data(q, qg3p),(2,3,1,4))
solu,  μ, σ = ESM_PINO.normalize_data(solu)
q_0 = solu[:,:,:,1:N_sims]
q_0 = CuArray(ESM_PINO.add_noise(Array(q_0)))
q_evolved = solu[:,:,:,2:N_sims+2]
q_evolved = CuArray(ESM_PINO.add_noise(Array(q_evolved)))

dt = 1 #QG3.p.time_unit
maxiters = 20
hidden_channels = 256
batch_size = 256

ggsh_loss = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=N_sims)
shgg_loss = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=N_sims)    

pars = ESM_PINOQG3.QG3_Physics_Parameters(dt, qg3p, S, ggsh_loss, shgg_loss, μ, σ)

trained_model = ESM_PINOQG3.train_model(q_0, q_evolved[:,:,:,1:N_sims], qg3ppars, 
                                            maxiters=maxiters, 
                                            hidden_channels=hidden_channels, 
                                            parameters=pars, 
                                            batchsize=batch_size,
                                            use_physics=true
                                            )
trained_model_architecture = trained_model.model
trained_model_ps = trained_model.ps
trained_model_st = trained_model.st

autoregressive_steps = 2
autoregressive_target = ESM_PINOQG3.stack_time_steps(q_evolved, autoregressive_steps)

fine_tuned_model = ESM_PINOQG3.fine_tuning(q_0, autoregressive_target, trained_model_architecture, trained_model_ps, trained_model_st, 
                                            parameters=pars, 
                                            use_physics=false,
                                            n_steps=autoregressive_steps
                                            )


model = fine_tuned_model.model
ps = cdev(fine_tuned_model.ps)
st = cdev(fine_tuned_model.st)
@save joinpath(root, "SFPINO_results.jld2") model ps st