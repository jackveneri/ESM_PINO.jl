root = dirname(@__DIR__)
dir = @__DIR__
using Pkg
Pkg.activate(dir)
Pkg.instantiate()
using ESM_PINO, Printf, CUDA, OnlineStats, Lux, LuxCUDA, Random, Statistics, MLUtils, Optimisers, ParameterSchedulers, QG3, JLD2

const ESM_PINOQG3 = Base.get_extension(ESM_PINO, :ESM_PINOQG3Ext)
const gdev = gpu_device()
const cdev = cpu_device()

dt = 3 #QG3.p.time_unit
maxiters = 100
hidden_channels = 32
batch_size = 4
num_examples = num_valid = 256
N_sims = 3000
N_val = 300
gpu = false
modes = 32
autoregressive_steps = 2

if gpu
    QG3.gpuon()
else
    QG3.gpuoff()
end

qg3ppars, qg3p, S, solψ, solu = ESM_PINOQG3.load_precomputed_data(root=root, N_sims=N_sims+(2*autoregressive_steps*dt)+N_val, res="t42", gpu=gpu)
#sol = cat(solψ, solu; dims=3)
q_0_train, q_evolved_train, q_0_val, q_evolved_val, μ, σ, _ = ESM_PINOQG3.preprocess_data(solu, normalize=true, to_gpu=gpu, dt=dt, channelwise=true, train_fraction=N_sims/(N_sims+N_val)) 

autoregressive_target = ESM_PINOQG3.stack_time_steps(q_evolved_train, autoregressive_steps, dt=dt, N_sims=N_sims, gpu=gpu)
autoregressive_target_val = ESM_PINOQG3.stack_time_steps(q_evolved_val, autoregressive_steps, dt=dt, N_sims=N_val, gpu=gpu)

ggsh_loss = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=batch_size)
shgg_loss = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=batch_size)  

pars = ESM_PINOQG3.QG3_Physics_Parameters(dt, qg3p, S, ggsh_loss, shgg_loss, μ, σ, gpu=gpu)

trained_model = ESM_PINOQG3.train_model(q_0_train, q_evolved_train, q_0_val, q_evolved_val, qg3ppars, 
                                            nepochs=maxiters,
                                            downsampling_factor=2,
                                            modes=modes, 
                                            hidden_channels=hidden_channels, 
                                            parameters=pars, 
                                            batchsize=batch_size,
                                            use_norm=true,
                                            operator_type=:driscoll_healy,
                                            inner_skip=true,
                                            outer_skip=true,
                                            use_physics=true,
                                            geometric=true,
                                            spectral=false,
                                            positional_embedding = "no_grid",
                                            gpu=gpu,
                                            lr_0=2e-3,
                                            β=0.015f0 
                                            )
trained_model_architecture = trained_model.model
trained_model_ps = trained_model.ps
trained_model_st = trained_model.st

GC.gc()
CUDA.reclaim()

fine_tuned_model = ESM_PINOQG3.fine_tuning(q_0_train[:,:,:,1:size(autoregressive_target,4)], autoregressive_target, q_0_val[:,:,:,1:size(autoregressive_target_val,4)], autoregressive_target_val, trained_model_architecture, trained_model_ps, trained_model_st, 
                                            parameters=pars, 
                                            use_physics=false,
                                            geometric=true,
                                            n_steps=autoregressive_steps,
                                            nepochs=10,
                                            gpu=gpu
                                            )


model = trained_model.model
ps = cdev(trained_model.ps)
st = cdev(trained_model.st)
@save joinpath(root, "models/SFNO_results.jld2") model ps st dt N_sims μ σ