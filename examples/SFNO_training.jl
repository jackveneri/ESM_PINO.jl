root = dirname(@__DIR__)
dir = @__DIR__
using Pkg
Pkg.activate(dir)
Pkg.instantiate()
using ESM_PINO, Printf, CUDA, OnlineStats, Lux, LuxCUDA, Random, Statistics, MLUtils, Optimisers, ParameterSchedulers, QG3, NetCDF, Dates, CFTime, JLD2

const ESM_PINOQG3 = Base.get_extension(ESM_PINO, :ESM_PINOQG3Ext)
const gdev = gpu_device()
const cdev = cpu_device()

function train_model(x::AbstractArray, target::AbstractArray, pars::QG3ModelParameters, ggsh::QG3.GaussianGridtoSHTransform, shgg::QG3.SHtoGaussianGridTransform; seed::Int=0,
    maxiters::Int=3000, hidden_channels::Int=32, parameters::Union{Nothing, ESM_PINOQG3.QG3_Physics_Parameters}=nothing)
    
    rng = Random.default_rng(seed)
    batchsize = 100
    dataloader = DataLoader((x, target); batchsize=batchsize, shuffle=true) |> gdev
    ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=batchsize)
    shgg = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=batchsize)
    # Create the model
    sfno = SFNO(
        ggsh,
        shgg,
        in_channels=3, 
        out_channels=3, 
        n_layers=4,
        hidden_channels=hidden_channels, 
        positional_embedding="grid"
    )

    ps, st = Lux.setup(rng, sfno) |> gdev
    
    # Rest of training setup remains similar
    
    opt = Optimisers.ADAM(0.001f0)
    lr = i -> Exp(0.001f0, 0.999f0).(i)
    train_state = Training.TrainState(sfno, ps, st, opt)
    
    if !isnothing(parameters)
        par_train = ESM_PINOQG3.QG3_Physics_Parameters(
                parameters.dt,
                parameters.qg3p,
                parameters.S,
                QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=batchsize),
                QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=batchsize),
                parameters.μ,
                parameters.σ
        )
    else
        par_train = nothing
    end
    physics_loss = ESM_PINOQG3.create_QG3_physics_loss(par_train)
    loss_function = ESM_PINOQG3.select_QG3_loss_function(physics_loss)
    total_loss_tracker = ntuple(_ -> Lag(Float32, 32), 1)[1]

    iter = 1
    for (x, target_data) in Iterators.cycle(dataloader)
        Optimisers.adjust!(train_state, lr(iter))
        _, loss, stats, train_state = Training.single_train_step!(AutoZygote(), loss_function, (x, target_data), train_state)
        
        fit!(total_loss_tracker, Float32(loss))
        #fit!(physics_loss_tracker, Float32(stats.physics_loss))
        #fit!(data_loss_tracker, Float32(stats.data_loss))

        mean_loss = mean(OnlineStats.value(total_loss_tracker))
        #mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
        #mean_data_loss = mean(OnlineStats.value(data_loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))
        
        if iter % 5 == 0 || iter == maxiters
            #@printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f \
            #     (%.9f) \t Data Loss: %.9f (%.9f)\n" iter maxiters loss mean_loss stats.physics_loss mean_physics_loss stats.data_loss mean_data_loss
            @printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \n" iter maxiters loss mean_loss
                        
            GC.gc()
        end
        
        iter += 1
        
        if iter > maxiters
            break
        end
    end
    return StatefulLuxLayer{true}(sfno, train_state.parameters, train_state.states), loss_function
end


@load string(root, "/data/t42-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
qg3p = CUDA.@allowscalar QG3Model(qg3ppars)
@load string(root, "/data/t42-precomputed-S.jld2") S
S = CUDA.@allowscalar QG3.reorder_SH_gpu(S, qg3ppars)

# initial conditions for streamfunction and vorticity
N_sims = 1000
@load string(root,"/data/t42_qg3_data_SH_CPU.jld2") q
q = QG3.reorder_SH_gpu(q, qg3ppars)
solu = permutedims(QG3.transform_grid_data(q, qg3p),(2,3,1,4))
solu,  μ, σ = ESM_PINO.normalize_data(solu)
q_0 = solu[:,:,:,1:N_sims]
q_0 = CuArray(ESM_PINO.add_noise(Array(q_0)))
q_evolved = solu[:,:,:,2:N_sims+1]
q_evolved = CuArray(ESM_PINO.add_noise(Array(q_evolved)))
shgg2 = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=N_sims)
ggsh2 = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=N_sims)

dt = 1 #G3.p.time_unit
seed = 0
maxiters = 5000
hidden_channels = 64
rng = Random.default_rng(seed)
ggsh_train = QG3.GaussianGridtoSHTransform(qg3ppars, hidden_channels, N_batch=N_sims)
shgg_train = QG3.SHtoGaussianGridTransform(qg3ppars, hidden_channels, N_batch=N_sims)
ggsh_loss = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=N_sims)
shgg_loss = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=N_sims)    
sfno = SFNO(
        ggsh_train,
        shgg_train,
        in_channels=3, 
        out_channels=3, 
        n_layers=4, 
        hidden_channels=hidden_channels, 
        positional_embedding="grid"
)


ps, st = Lux.setup(rng, sfno) |> gdev
q_0 = q_0 |> gdev
q_evolved = q_evolved |> gdev
sfno(q_0, ps, st)[1]
GC.gc()
model = StatefulLuxLayer{true}(sfno, ps, st)
pars = ESM_PINOQG3.QG3_Physics_Parameters(dt, qg3p, S, ggsh_loss, shgg_loss, μ, σ)

PI_loss = ESM_PINOQG3.create_QG3_physics_loss(pars)
loss = ESM_PINOQG3.select_QG3_loss_function(PI_loss)

loss(sfno, ps, st, (q_0, q_evolved))

trained_model, loss_function = train_model(q_0, q_evolved, qg3ppars, ggsh_train, shgg_train; seed=0, maxiters=maxiters, hidden_channels=hidden_channels, parameters=nothing)

N_test = 100
shgg2 = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=N_test)
ggsh2 = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=N_test)
test_model = SFNO(
        ggsh2,
        shgg2,
        in_channels=3, 
        out_channels=3, 
        n_layers=4, 
        hidden_channels=hidden_channels, 
        positional_embedding="grid"
    )

trained_u = Lux.testmode(StatefulLuxLayer{true}(test_model, trained_model.ps, trained_model.st))

q_test = solu[:,:,:,N_sims+1:N_sims+N_test]
q_test = CuArray(ESM_PINO.add_noise(Array(q_test)))
q_test_evolved = (solu[:,:,:,N_sims+2:N_sims+N_test+1] .* σ .+ μ)
q_test_array = Float32.(q_test)
GC.gc()
q_pred = trained_u(q_test_array)
q_pred = (q_pred .* σ .+ μ) |> gdev
@printf "Relative L2 Error = %.5f \n"  (mean(abs2, q_pred .- q_test_evolved) / mean(abs2, q_test_evolved))
q_pred_sh = QG3.transform_SH(permutedims(q_pred,(3,1,2,4)), ggsh2)
q_test_evolved_sh = QG3.transform_SH(permutedims(q_test_evolved,(3,1,2,4)), ggsh2)
sf_plot_pred = transform_grid(qprimetoψ(qg3p, q_pred_sh), shgg2)
sf_plot_evolved = transform_grid(qprimetoψ(qg3p, q_test_evolved_sh), shgg2)

loss = zeros(size(sf_plot_evolved, 1))
for i in 1:size(sf_plot_evolved, 1)
    loss[i] = mean(abs2, sf_plot_pred[i,:,:,:] .- sf_plot_evolved[i,:,:,:])
    scale = mean(abs2, sf_plot_evolved[i,:,:,:])
    loss[i] = loss[i] / scale * 100
end


mistake = sf_plot_pred .- sf_plot_evolved
mistake = Array(mistake)
sf_plot_pred = Array(sf_plot_pred)

using CairoMakie, JLD2

my_theme = merge(theme_latexfonts(), theme_minimal())
set_theme!(my_theme, fontsize = 24, font = "Helvetica", color = :black)

for ilvl in 1:3
    # Set up figure and color limits
    clims_pred = (-1.1 * maximum(abs, sf_plot_pred[ilvl,:,:,:]), 
                  1.1 * maximum(abs, sf_plot_pred[ilvl,:,:,:]))
    clims_err = (-1.1 * maximum(abs, mistake[ilvl,:,:,:]), 
                 1.1 * maximum(abs, mistake[ilvl,:,:,:]))
    
    plot_times = 1:size(sf_plot_pred, 4)

    # Animation for predictions using Observables
    fig_pred = Figure()
    ax_pred = Axis(fig_pred[1, 1])
    data_obs_pred = Observable(permutedims(sf_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = heatmap!(ax_pred, data_obs_pred, colorrange=clims_pred, colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)
    
    CairoMakie.record(fig_pred, joinpath(root, "SFNO_prediction_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_pred[] =permutedims(sf_plot_pred[ilvl,:,:,it],(2,1))
        ax_pred.title = @sprintf("SFNO Prediction at time = %d - %.2f d", it, it * qg3p.p.time_unit)
    end

    # Animation for errors using Observables
    fig_err = Figure()
    ax_err = Axis(fig_err[1, 1])
    data_obs_err = Observable(permutedims(mistake[ilvl,:,:,1],(2,1)))    
    hm_err = heatmap!(ax_err, data_obs_err, colorrange=clims_err, colormap=:balance)
    Colorbar(fig_err[1, 2], hm_err)
    
    CairoMakie.record(fig_err, joinpath(root, "SFNO_error_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_err[] = permutedims(mistake[ilvl,:,:,it],(2,1))
        ax_err.title = @sprintf("SFNO Error at time = %d - %.2f d", it, it * qg3p.p.time_unit)    
    end

    @printf "SFNO Percentual Error L2 norm at level %1d: %.9f\n" ilvl loss[ilvl]
end

# Save results
ps = cdev(trained_model.ps)
st = cdev(trained_model.st)
@save joinpath(root, "SFNO_results.jld2") sf_plot_pred sf_plot_evolved mistake loss ps st 