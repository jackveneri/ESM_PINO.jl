root = dirname(@__DIR__)
using Pkg
Pkg.activate(root)
Pkg.instantiate()
using ESM_PINO, JLD2, CairoMakie, Printf, Statistics, QG3, NetCDF, Dates, CFTime, Lux, CUDA, LuxCUDA, Random

gdev = gpu_device()
cdev = cpu_device()

model_string = "FNO_PINO"

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
@load string(root, "/data/plotting_utils.jld2")
@load string(root,"/data/t21_qg3_data_SH_CPU.jld2") t q
q = QG3.reorder_SH_gpu(q, qg3p.p)
solu = permutedims(QG3.transform_grid_data(q, qg3p),(2,3,1,4))

N_test = 100
hidden_channels = 64
ggsh2 = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=N_test)
shgg2 = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=N_test)
test_model = FourierNeuralOperator(
    in_channels=3, 
    out_channels=3, 
    n_layers=4, 
    hidden_channels=hidden_channels, 
    n_modes=(15, 15),  # 2D Fourier modes (lat, lon)
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
sf_plot_evolved = Array(sf_plot_evolved)


my_theme = merge(theme_latexfonts(), theme_minimal())
set_theme!(my_theme, fontsize = 24, font = "Helvetica", color = :black)
#=
for ilvl in 1:3
    # Set up figure and color limits
    cl = (-1.1 * maximum(abs, sf_plot_evolved[ilvl,:,:,:]), 
                  1.1 * maximum(abs, sf_plot_evolved[ilvl,:,:,:]))
    clims[ilvl] = cl   
    plot_times = 1:size(sf_plot_pred, 4)

   # Animation for evolution using Observables
    fig = Figure()
    ax = Axis(fig[1, 1])
    data_obs = Observable(permutedims(sf_plot_evolved[ilvl,:,:,1],(2,1)))    
    hm = heatmap!(ax, data_obs, colorrange=cl, colormap=:balance)
    Colorbar(fig[1, 2], hm)
    
    CairoMakie.record(fig, string(root, "/figures/evo_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs[] = permutedims(sf_plot_evolved[ilvl,:,:,it],(2,1))
        ax.title = @sprintf("Evolution at time = %d - %.2f d", it, it * time_scale)    
    end

    
end
=#

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
    
    CairoMakie.record(fig_err, string(root, "/figures/",model_string, "_error_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_err[] = permutedims(mistake[ilvl,:,:,it],(2,1))
        ax_err.title = model_string * @sprintf(" Error at time = %d - %.2f d", it, it * time_scale)    
    end

    @printf "%s Percentual Error L2 norm at level %1d: %.9f\n" model_string ilvl loss[ilvl]
end

long_rollout_iter = Int(round(10 / time_scale))
snapshots = 100
time_scale_adjust = Int(round(long_rollout_iter / (snapshots-1)))
q_stable = q_test_evolved = apply_n_times(trained_u, q_test_rollout, long_rollout_iter; m=100)
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

    CairoMakie.record(fig_pred, string(root, "/figures/", model_string, "_stability_fps20_sf_lvl$(ilvl).gif"), plot_times;
            framerate=5) do it
            # Update observable data
            data_obs_pred[] = permutedims(sf_plot_pred[ilvl,:,:,it],(2,1))
            ax_pred.title =model_string * @sprintf(" Prediction at time = %d - %.2f days", it, it * time_scale * time_scale_adjust)    
        end
end
