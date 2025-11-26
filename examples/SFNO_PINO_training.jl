root = dirname(@__DIR__)
dir = @__DIR__
using Pkg
Pkg.activate(dir)
Pkg.instantiate()
using ESM_PINO, Printf, CUDA, OnlineStats, Lux, LuxCUDA, Random, Statistics, MLUtils, Optimisers, ParameterSchedulers, QG3, NetCDF, Dates, CFTime, JLD2

const ESM_PINOQG3 = Base.get_extension(ESM_PINO, :ESM_PINOQG3Ext)
const gdev = gpu_device()
const cdev = cpu_device()

dt = 1 #QG3.p.time_unit
maxiters = 500
hidden_channels = 20
batch_size = 10
N_sims = 1000

qg3ppars, qg3p, S, solψ, solu = ESM_PINOQG3.load_precomputed_data(root=root, N_sims=N_sims, res="t21")
#sol = cat(solψ, solu; dims=3)
q_0, q_evolved, μ, σ, _ = ESM_PINOQG3.preprocess_data(solu, normalize=true)

ggsh_loss = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=N_sims)
shgg_loss = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=N_sims)    

pars = ESM_PINOQG3.QG3_Physics_Parameters(dt, qg3p, S, ggsh_loss, shgg_loss, μ, σ)

trained_model = ESM_PINOQG3.train_model(q_0[:,:,:,1:N_sims], q_evolved[:,:,:,1:N_sims], qg3ppars, 
                                            maxiters=maxiters,
                                            downsampling_factor=1, 
                                            hidden_channels=hidden_channels, 
                                            parameters=pars, 
                                            batchsize=batch_size,
                                            use_norm=true,
                                            operator_type =:driscoll_healy,
                                            inner_skip=true,
                                            outer_skip=false,
                                            use_physics=true,
                                            geometric=true,
                                            positional_embedding = "gaussian_grid"
                                            )
trained_model_architecture = trained_model.model
trained_model_ps = trained_model.ps
trained_model_st = trained_model.st

autoregressive_steps = 2
autoregressive_target = ESM_PINOQG3.stack_time_steps(q_evolved, autoregressive_steps)

GC.gc()
CUDA.reclaim()

fine_tuned_model = ESM_PINOQG3.fine_tuning(q_0[:,:,:,1:N_sims], autoregressive_target, trained_model_architecture, trained_model_ps, trained_model_st, 
                                            parameters=pars, 
                                            use_physics=false,
                                            geometric=true,
                                            n_steps=autoregressive_steps,
                                            maxiters=125
                                            )


model = fine_tuned_model.model
ps = cdev(fine_tuned_model.ps)
st = cdev(fine_tuned_model.st)
@save joinpath(root, "models/SFPINO_results.jld2") model ps st