root = dirname(@__DIR__)
using Pkg
Pkg.activate(root)
Pkg.instantiate()
using ESM_PINO, JLD2, CairoMakie, Printf, Statistics, QG3, NetCDF, Dates, CFTime, Lux, CUDA, LuxCUDA, Random

gdev = gpu_device()
cdev = cpu_device()

model_string = "SFNO_PINO"

@load string(root, "/data/t21-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
qg3p = CUDA.@allowscalar QG3Model(qg3ppars)

@load string(root,"/data/solq.jld2") solu
solu = CuArray(cat(solu...,dims=4))
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=size(solu,4))
solu = QG3.transform_grid(solu, shgg)
solu = permutedims(solu,(2,3,1,4))
solu,  μ, σ = ESM_PINO.normalize_data(solu)

@load string(root,"/models/",model_string,"_results.jld2")  sf_plot_pred sf_plot_evolved mistake loss ps st
@load string(root, "/data/plotting_utils.jld2") clims time_scale
@load string(root,"/data/t21_qg3_data_SH_CPU.jld2") t q
q = QG3.reorder_SH_gpu(q, qg3p.p)
solu = permutedims(QG3.transform_grid_data(q, qg3p),(2,3,1,4))

N_test = 100
hidden_channels = 128
shgg2 = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=N_test)
ggsh2 = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=N_test)
test_model = SFNO(
        qg3ppars,
        ggsh2,
        shgg2,
        in_channels=3, 
        out_channels=3, 
        n_layers=4, 
        hidden_channels=hidden_channels, 
        positional_embedding="grid"
)
ps, st = gdev(ps), gdev(st)

trained_u = Lux.testmode(StatefulLuxLayer{true}(test_model, ps, st))


q_test = ((solu[:,:,:,1001:1100] .- μ ) ./ σ)  |> gdev
q_test_evolved = solu[:,:,:,1002:1101]

q_test_array = (Float32.(q_test))
GC.gc()
q_pred = trained_u(q_test_array) .* σ .+ μ
mean(abs2, q_pred .- q_test_evolved) / mean(abs2, q_test_evolved)

#using BenchmarkTools
q_test_rollout = q_test_array[:, :, :, 1:1]
function apply_n_times(f, x::AbstractArray, n::Int; m::Int=0)
    y = x
    snapshots = m > 0 ? Vector{typeof(x)}() : nothing
    save_steps = m > 0 ? round.(Int, range(1, n; length=m)) : Int[]
    
    for i in 1:n
        y = f(y)
        if i in save_steps
            push!(snapshots, copy(y) .* σ .+ μ)
        end
    end

    return m > 0 ? snapshots : y
end
shgg3 = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=1)
ggsh3 = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=1)
test_model_autoreg = SFNO(
        qg3ppars,
        ggsh3,
        shgg3,
        in_channels=3, 
        out_channels=3, 
        n_layers=4, 
        hidden_channels=hidden_channels, 
        positional_embedding="grid"
)
ps, st = gdev(ps), gdev(st)

trained_u_autoreg = Lux.testmode(StatefulLuxLayer{true}(test_model_autoreg, ps, st))
#=
GC.gc()
@btime apply_n_times($trained_u_autoreg, $q_test_rollout, $100)
GC.gc()
@btime trained_u($q_test_array)
GC.gc()
@btime test_model($q_test_array, $ps, $st)[1]
=#
q_pred_sh = QG3.transform_SH(permutedims(q_pred,(3,1,2,4)), ggsh2)
q_test_evolved_sh = QG3.transform_SH(permutedims(q_test_evolved,(3,1,2,4)), ggsh2)
sf_plot_pred = transform_grid(qprimetoψ(qg3p, q_pred_sh), shgg2)
#sf_plot_evolved = transform_grid(qprimetoψ(qg3p, permutedims(q_test_evolved_sh,(2,3,1,4))), shgg2)
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
        
    plot_times = 1:size(sf_plot_pred, 4)

    # Animation for predictions using Observables
    fig_pred = Figure()
    ax_pred = Axis(fig_pred[1, 1])
    data_obs_pred = Observable(permutedims(sf_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = heatmap!(ax_pred, data_obs_pred, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)
    
    CairoMakie.record(fig_pred, string(root, "/figures/", model_string,"_prediction_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_pred[] =permutedims(sf_plot_pred[ilvl,:,:,it],(2,1))
        ax_pred.title = model_string * @sprintf(" Prediction at time = %d - %.2f d", it, it * time_scale)
    end

    # Animation for errors using Observables
    fig_err = Figure()
    ax_err = Axis(fig_err[1, 1])
    data_obs_err = Observable(permutedims(mistake[ilvl,:,:,1],(2,1)))    
    hm_err = heatmap!(ax_err, data_obs_err, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_err[1, 2], hm_err)
    
    CairoMakie.record(fig_err, string(root, "/figures/", model_string ,"_error_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_err[] = permutedims(mistake[ilvl,:,:,it],(2,1))
        ax_err.title = model_string * @sprintf(" Error at time = %d - %.2f d", it, it * time_scale)    
    end

    @printf "%s Percentual Error L2 norm at level %1d: %.9f\n" model_string ilvl loss[ilvl]
end
sf_plot_evolved = Array(sf_plot_evolved)
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

long_rollout_iter = Int(round(20 / time_scale))
snapshots = 100
time_scale_adjust = Int(round(long_rollout_iter / (snapshots-1)))
q_stable = apply_n_times(trained_u_autoreg, q_test_rollout, long_rollout_iter; m=snapshots)
q_stable = cat(q_stable..., dims=4)
q_pred_sh = QG3.transform_SH(permutedims(q_stable,(3,1,2,4)), ggsh2)
sf_plot_pred = transform_grid(qprimetoψ(qg3p, q_pred_sh), shgg2)
sf_plot_pred = Array(sf_plot_pred)

for ilvl in 1:3
    plot_times = 1:size(sf_plot_pred, 4)
    fig_pred = Figure()
    ax_pred = Axis(fig_pred[1, 1])
    data_obs_pred = Observable(permutedims(sf_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = heatmap!(ax_pred, data_obs_pred, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)

    CairoMakie.record(fig_pred, string(root, "/figures/",model_string,"_stability_fps20_sf_lvl$(ilvl).gif"), plot_times;
            framerate=5) do it
            # Update observable data
            data_obs_pred[] = permutedims(sf_plot_pred[ilvl,:,:,it],(2,1))
            ax_pred.title = model_string * @sprintf(" Prediction at time = %d - %.2f days", it, it * time_scale * time_scale_adjust)
        end
    end
