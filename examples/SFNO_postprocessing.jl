root = dirname(@__DIR__)
dir = @__DIR__
using Pkg
Pkg.activate(dir)
Pkg.instantiate()
using ESM_PINO, JLD2, GLMakie, Printf, Statistics, QG3, NetCDF, Dates, CFTime, Lux, CUDA, LuxCUDA, Random, GeoMakie

# Determine device type and setup
function get_device(model)
    # Check if model parameters are on GPU
    gpu = typeof(model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.plan.ggsh).parameters[end]
    if gpu
        CUDA.functional() || error("Model is GPU-based but CUDA is not available")
        QG3.gpuon()
        return gpu_device()
    else
        QG3.gpuoff()
        return cpu_device()
    end
end

# Device-aware data transfer
function transfer_data(data, dev)
    if dev == gpu_device()
        return gdev(data)
    else
        return data
    end
end

function to_array(data, dev)
    if dev == gpu_device()
        return Array(data)
    else
        return data
    end
end

# Main execution
model_string = "SFNO"
GC.gc()
CUDA.reclaim()

@load string(root,"/models/",model_string,"_results.jld2") model ps st dt
dev = get_device(model)
const ESM_PINOQG3 = Base.get_extension(ESM_PINO, :ESM_PINOQG3Ext)

@load string(root, "/data/t42-precomputed-p.jld2") qg3ppars
qg3p = @CUDA.allowscalar QG3Model(qg3ppars)


@load string(root,"/data/t42_qg3_data_SH_CPU.jld2") t q

ESM_PINO.analyze_weights(ps)

q = QG3.reorder_SH_gpu(q, qg3p.p)
solu = permutedims(QG3.transform_grid_data(q, qg3p),(2,3,1,4))
solu, μ, σ = ESM_PINO.normalize_data(solu, channelwise=true)

N_test = 100
shgg2 = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=N_test)
ggsh2 = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=N_test)
test_model = ESM_PINOQG3.transfer_SFNO_model(model, qg3ppars, batch_size=N_test)
ps, st = transfer_data(ps, dev), transfer_data(st, dev)

trained_u = Lux.testmode(StatefulLuxLayer{true}(test_model, ps, st))

q_test, q_test_targets, _, _  = ESM_PINOQG3.preprocess_data(solu[:,:,:,1001:1100+dt], normalize=false, noise_level=0, dt=dt)
q_test, q_test_targets = transfer_data(q_test, dev), transfer_data(q_test_targets, dev)
q_test_evolved = ESM_PINO.denormalize_data(q_test_targets, μ, σ, channelwise=true)

q_test_array = transfer_data(Float32.(q_test), dev)
GC.gc()
CUDA.reclaim()
q_pred = ESM_PINO.denormalize_data(trained_u(q_test_array),μ, σ)

one_step_L2_rel_err = mean(abs2, to_array(q_pred, dev) .- to_array(q_test_evolved, dev)) / 
                      mean(abs2, to_array(q_test_evolved, dev))
println("One-step Relative L2 Error: ", one_step_L2_rel_err)

q_test_rollout = q_test_array[:, :, :, 1:1]

test_model_autoreg = ESM_PINOQG3.transfer_SFNO_model(model, qg3ppars, batch_size=1)
trained_u_autoreg = Lux.testmode(StatefulLuxLayer{true}(test_model_autoreg, ps, st))

q_pred_sh = QG3.transform_SH(permutedims(q_pred,(3,1,2,4)), ggsh2)
q_test_evolved_sh = QG3.transform_SH(permutedims(q_test_evolved,(3,1,2,4)), ggsh2)
q_plot_pred = to_array(transform_grid(q_pred_sh, shgg2), dev)
q_plot_evolved = to_array(transform_grid(q_test_evolved_sh, shgg2), dev)
sf_plot_pred = transform_grid(qprimetoψ(qg3p, q_pred_sh), shgg2)
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
mistake_sf = to_array(mistake_q, dev)
sf_plot_pred = to_array(sf_plot_pred, dev)

# Visualization functions
function create_animation(data, filename, title_template, clims, ilvl, dev, qg3ppars)
    plot_times = 1:size(data, 4)
    lons = range(-180,180,qg3ppars.N_lons)    # Longitude 
    lats = rad2deg.(qg3ppars.lats)    # Latitude 
    fig = Figure()
    ax = GeoAxis(fig[1, 1], dest="+proj=moll")
    data_obs = Observable(permutedims(data[ilvl,:,:,1],(2,1)))
    hm = GeoMakie.surface!(ax, lons, lats, data_obs, colorrange=clims, colormap=:balance)
    lines!(ax, GeoMakie.coastlines(ax), color=:black, overdraw=true)
    Colorbar(fig[1, 2], hm)
    
    GeoMakie.record(fig, filename, plot_times; framerate=20) do it
        data_obs[] = permutedims(data[ilvl,:,:,it],(2,1))
        ax.title = title_template * @sprintf(" at time = %d - %.2f d", it, it * qg3ppars.time_unit)
    end
end

for ilvl in 1:3
    clims = (-1.1 * maximum(abs.(q_plot_pred[ilvl,:,:,:])), 1.1 * maximum(abs.(q_plot_pred[ilvl,:,:,:])))
    
    # Prediction animation
    create_animation(
        q_plot_pred, 
        string(root, "/figures/", model_string, "_prediction_fps20_sf_lvl$(ilvl).gif"),
        model_string * " Prediction",
        clims, ilvl, dev, qg3ppars
    )
    
    # Error animation  
    create_animation(
        mistake_q,
        string(root, "/figures/", model_string, "_error_fps20_sf_lvl$(ilvl).gif"),
        model_string * " Error",
        clims, ilvl, dev, qg3ppars
    )

    @printf "%s Percentual Error L2 norm at level %1d: %.9f\n" model_string ilvl loss_q[ilvl]
end

sf_plot_evolved = to_array(sf_plot_evolved, dev)

# Long rollout
long_rollout_t_steps = Int(round(10 / qg3ppars.time_unit)) #10 days in QG3 time units 
long_rollout_iter = Int(long_rollout_t_steps ÷ dt)
snapshots = min(100, long_rollout_iter)
time_scale_adjust = Int(round(long_rollout_t_steps / (snapshots-1)))

ggsh3 = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=long_rollout_iter)
shgg3 = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=long_rollout_iter)

q_stable = ESM_PINO.apply_n_times(trained_u_autoreg, q_test_rollout, long_rollout_iter; m=snapshots, μ=μ, σ=σ)
q_stable = cat(q_stable..., dims=4)
q_pred_sh = QG3.transform_SH(permutedims(q_stable,(3,1,2,4)), ggsh3)
q_plot_pred = to_array(transform_grid(q_pred_sh, shgg3), dev)
sf_plot_pred = transform_grid(qprimetoψ(qg3p, q_pred_sh), shgg3)
sf_plot_pred = to_array(sf_plot_pred, dev)

for ilvl in 1:3
    plot_times = 1:size(q_plot_pred, 4)
    lons = range(-180,180,qg3ppars.N_lons)    # Longitude 
    lats = rad2deg.(qg3ppars.lats)    # Latitude 
    clims = (-1.1 * maximum(abs.(q_plot_pred[ilvl,:,:,:])), 1.1 * maximum(abs.(q_plot_pred[ilvl,:,:,:])))
    fig_pred = Figure()
    ax_pred = GeoAxis(fig_pred[1, 1], dest="+proj=moll")
    data_obs_pred = Observable(permutedims(q_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = GeoMakie.surface!(ax_pred,lons, lats, data_obs_pred, colorrange=clims, colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)

    GeoMakie.record(fig_pred, string(root, "/figures/", model_string, "_stability_fps20_sf_lvl$(ilvl).gif"), 
                     plot_times; framerate=5) do it
        data_obs_pred[] = permutedims(q_plot_pred[ilvl,:,:,it],(2,1))
        ax_pred.title = model_string * @sprintf(" Prediction at time = %d - %.2f days", it, 
                         it * qg3ppars.time_unit * time_scale_adjust)
    end
end