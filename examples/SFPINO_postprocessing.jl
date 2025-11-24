root = dirname(@__DIR__)
dir = @__DIR__
using Pkg
Pkg.activate(dir)
Pkg.instantiate()
using ESM_PINO, JLD2, CairoMakie, Printf, Statistics, QG3, NetCDF, Dates, CFTime, Lux, CUDA, LuxCUDA, Random
QG3.gpuon()

const ESM_PINOQG3 = Base.get_extension(ESM_PINO, :ESM_PINOQG3Ext)
const gdev = gpu_device()
const cdev = cpu_device() 

model_string = "SFPINO"
GC.gc()
CUDA.reclaim()

@load string(root, "/data/t21-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
qg3p = CUDA.@allowscalar QG3Model(qg3ppars)

@load string(root,"/models/",model_string,"_results.jld2") model ps st
@load string(root,"/data/t21_qg3_data_SH_CPU.jld2") t q

ESM_PINO.analyze_weights(ps)

q = QG3.reorder_SH_gpu(q, qg3p.p)
solu = permutedims(QG3.transform_grid_data(q, qg3p),(2,3,1,4))
solu, μ, σ = ESM_PINO.normalize_data(solu)
#μ, σ = 0, 1
N_test = 100
shgg2 = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=N_test)
ggsh2 = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=N_test)
test_model = ESM_PINOQG3.transfer_SFNO_model(model, qg3ppars, batch_size=N_test)
ps, st = gdev(ps), gdev(st)

trained_u = Lux.testmode(StatefulLuxLayer{true}(test_model, ps, st))

q_test = solu[:,:,:,1001:1100]  |> gdev
q_test_evolved = ESM_PINO.denormalize_data(solu[:,:,:,1002:1101], μ, σ)

q_test_array = (Float32.(q_test))
GC.gc()
CUDA.reclaim()
q_pred = trained_u(q_test_array) .* σ .+ μ

one_step_L2_rel_err = mean(abs2, q_pred .- q_test_evolved) / mean(abs2, q_test_evolved)
println("One-step Relative L2 Error: ", one_step_L2_rel_err)

q_test_rollout = q_test_array[:, :, :, 1:1]

test_model_autoreg = ESM_PINOQG3.transfer_SFNO_model(model, qg3ppars, batch_size=1)

trained_u_autoreg = Lux.testmode(StatefulLuxLayer{true}(test_model_autoreg, ps, st))

q_pred_sh = QG3.transform_SH(permutedims(q_pred,(3,1,2,4)), ggsh2)
q_test_evolved_sh = QG3.transform_SH(permutedims(q_test_evolved,(3,1,2,4)), ggsh2)
q_plot_pred = Array(transform_grid(q_pred_sh, shgg2))
q_plot_evolved = Array(transform_grid(q_test_evolved_sh, shgg2))
sf_plot_pred = transform_grid(qprimetoψ(qg3p, q_pred_sh), shgg2)
#sf_plot_evolved = transform_grid(qprimetoψ(qg3p, permutedims(q_test_evolved_sh,(2,3,1,4))), shgg2)
sf_plot_evolved = transform_grid(qprimetoψ(qg3p, q_test_evolved_sh), shgg2)
loss_sf = zeros(size(sf_plot_evolved, 1))
loss_q = zeros(size(q_plot_evolved, 1))
for i in 1:size(sf_plot_evolved, 1)
    loss_sf[i] = mean(abs2, sf_plot_pred[i,:,:,:] .- sf_plot_evolved[i,:,:,:])
    scale = mean(abs2, sf_plot_evolved[i,:,:,:])
    loss_sf[i] = loss_sf[i] / scale * 100
end
for i in 1:size(q_plot_evolved, 1)
    loss_q[i] = mean(abs2, q_plot_pred[i,:,:,:] .- q_plot_evolved[i,:,:,:])
    scale = mean(abs2, q_plot_evolved[i,:,:,:])
    loss_q[i] = loss_q[i] / scale * 100
end

mistake_sf = sf_plot_pred .- sf_plot_evolved
mistake_q = q_plot_pred .- q_plot_evolved
mistake_sf = Array(mistake_q)
sf_plot_pred = Array(sf_plot_pred)

for ilvl in 1:3
        
    plot_times = 1:size(q_plot_pred, 4)
    clims = (-1.1 * maximum(abs.(q_plot_pred[ilvl,:,:,:])),1.1 * maximum(abs.(q_plot_pred[ilvl,:,:,:])))
    # Animation for predictions using Observables
    fig_pred = Figure()
    ax_pred = Axis(fig_pred[1, 1])
    data_obs_pred = Observable(permutedims(q_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = heatmap!(ax_pred, data_obs_pred, colorrange=clims, colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)
    
    CairoMakie.record(fig_pred, string(root, "/figures/", model_string,"_prediction_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_pred[] =permutedims(q_plot_pred[ilvl,:,:,it],(2,1))
        ax_pred.title = model_string * @sprintf(" Prediction at time = %d - %.2f d", it, it * qg3ppars.time_unit)
    end

    # Animation for errors using Observables
    fig_err = Figure()
    ax_err = Axis(fig_err[1, 1])
    data_obs_err = Observable(permutedims(mistake_q[ilvl,:,:,1],(2,1)))    
    hm_err = heatmap!(ax_err, data_obs_err, colorrange=clims, colormap=:balance)
    Colorbar(fig_err[1, 2], hm_err)
    
    CairoMakie.record(fig_err, string(root, "/figures/", model_string ,"_error_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_err[] = permutedims(mistake_q[ilvl,:,:,it],(2,1))
        ax_err.title = model_string * @sprintf(" Error at time = %d - %.2f d", it, it * qg3ppars.time_unit)    
    end

    @printf "%s Percentual Error L2 norm at level %1d: %.9f\n" model_string ilvl loss_q[ilvl]
end
sf_plot_evolved = Array(sf_plot_evolved)
#=
for ilvl in 1:3
    plot_times = 1:size(sf_plot_evolved, 4)
    fig_ev = Figure()
    ax_ev = Axis(fig_ev[1, 1])
    data_obs_ev = Observable(permutedims(sf_plot_evolved[ilvl,:,:,1],(2,1)))
    hm_ev = heatmap!(ax_ev, data_obs_ev, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_ev[1, 2], hm_ev)
    
    CairoMakie.record(fig_ev, string(root, "/figures/QG3_evo_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_ev[] =permutedims(sf_plot_evolved[ilvl,:,:,it],(2,1))
        ax_ev.title = @sprintf("Evolution at time = %d - %.2f d", it, it * time_scale)
    end
end
=#
long_rollout_iter = Int(round(10 / qg3ppars.time_unit))
snapshots = 100
time_scale_adjust = Int(round(long_rollout_iter / (snapshots-1)))
q_stable = ESM_PINO.apply_n_times(trained_u_autoreg, q_test_rollout, long_rollout_iter; m=snapshots, μ=μ,  σ=σ)
q_stable = cat(q_stable..., dims=4)
q_pred_sh = QG3.transform_SH(permutedims(q_stable,(3,1,2,4)), ggsh2)
q_plot_pred = Array(transform_grid(q_pred_sh, shgg2))
sf_plot_pred = transform_grid(qprimetoψ(qg3p, q_pred_sh), shgg2)
sf_plot_pred = Array(sf_plot_pred)

for ilvl in 1:3
    plot_times = 1:size(q_plot_pred, 4)
    clims = (-1.1 * maximum(abs.(q_plot_pred[ilvl,:,:,:])),1.1 * maximum(abs.(q_plot_pred[ilvl,:,:,:])))
    fig_pred = Figure()
    ax_pred = Axis(fig_pred[1, 1])
    data_obs_pred = Observable(permutedims(q_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = heatmap!(ax_pred, data_obs_pred, colorrange=clims, colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)

    CairoMakie.record(fig_pred, string(root, "/figures/",model_string,"_stability_fps20_sf_lvl$(ilvl).gif"), plot_times;
            framerate=5) do it
            # Update observable data
            data_obs_pred[] = permutedims(q_plot_pred[ilvl,:,:,it],(2,1))
            ax_pred.title = model_string * @sprintf(" Prediction at time = %d - %.2f days", it, it * qg3ppars.time_unit * time_scale_adjust)
        end
    end
