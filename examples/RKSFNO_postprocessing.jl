root = dirname(@__DIR__)
dir = @__DIR__
using Pkg
Pkg.activate(dir)
using ESM_PINO, JLD2, CairoMakie, Printf, Statistics, QG3, NetCDF, Dates, CFTime, Lux, CUDA, LuxCUDA, Random, GeoMakie, Plots

include(string(dir,"/plotting_utils.jl"))
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
        return data |> gpu_device()
    else
        return data
    end
end

function to_array(data, dev)
    if dev == gpu_device()
        return data |> cpu_device()
    else
        return data
    end
end

# Main execution
model_string = "RKSFNO"
GC.gc()
CUDA.reclaim()

@load string(root,"/models/",model_string,"_results.jld2") model ps st dt N_sims μ σ res
dev = get_device(model)
const ESM_PINOQG3 = Base.get_extension(ESM_PINO, :ESM_PINOQG3Ext)

gpu=typeof(model.sfno_blocks.layers.layer_1.spherical_kernel.spherical_conv.plan.ggsh).parameters[end]
qg3ppars, qg3p, S, solψ, solu = ESM_PINOQG3.load_precomputed_data(root=root, N_sims=5000, res=res, gpu=gpu)
velocity = ESM_PINOQG3.prepare_velocity_ML_data(solψ, qg3p, gpu=gpu)
sol = cat(solu, solψ; dims=3)

ESM_PINO.analyze_weights(ps)

sol = ESM_PINO.normalize_data(velocity, μ, σ)
N_test = 100
model_channels = model.lifting.layers.layer_1.in_chs
shgg2 = QG3.SHtoGaussianGridTransform(qg3ppars, model_channels, N_batch=N_test)
ggsh2 = QG3.GaussianGridtoSHTransform(qg3ppars, model_channels, N_batch=N_test)
test_model = ESM_PINOQG3.transfer_SFNO_model(model, qg3ppars, batch_size=N_test)
ps, st = transfer_data(ps, dev), transfer_data(st, dev)

trained_u = Lux.testmode(StatefulLuxLayer{true}(test_model, ps, st))
#you can call this on sol or sol[:,:,1:3,:] depending on e.g. model channels
function prepare_test_data(sol::AbstractArray{T,4}; N_sims=3000, N_val=300, dt=1, normalize=false, noise_level=0, dev=cpu_device(), μ=0, σ=1) where T
    sol_lt = sol[:,:,:, 1:N_sims]
    q_test, q_test_targets, _, _  = ESM_PINOQG3.preprocess_data(sol[:,:,:,N_sims+N_val+1:N_sims+N_val+N_test+dt], normalize=normalize, noise_level=noise_level, dt=dt, train_fraction=0)
    q_test, q_test_targets = transfer_data(q_test, dev), transfer_data(q_test_targets, dev)
    q_test_evolved = ESM_PINO.denormalize_data(q_test_targets, μ, σ)
    q_test_ltm = reshape(mean(sol_lt, dims=4),size(sol_lt)[1:3])

    q_test_array = transfer_data(Float32.(q_test), dev)
    GC.gc()
    CUDA.reclaim()
    return q_test_array, q_test_evolved, q_test_ltm
end
q_test_array, q_test_evolved, q_test_ltm = prepare_test_data(sol; N_sims=N_sims, N_val=300, dt=dt, normalize=false, dev=dev, μ=μ, σ=σ)
q_pred = ESM_PINO.denormalize_data(trained_u(q_test_array), μ, σ)

one_step_L2_rel_err = mean(abs2, to_array(q_pred, dev) .- to_array(q_test_evolved, dev)) / 
                      mean(abs2, to_array(q_test_evolved, dev))
println("One-step Relative L2 Error: ", one_step_L2_rel_err)

q_test_rollout = q_test_array[:, :, :, 2:2]

test_model_autoreg = ESM_PINOQG3.transfer_SFNO_model(model, qg3ppars, batch_size=1)
trained_u_autoreg = Lux.testmode(StatefulLuxLayer{true}(test_model_autoreg, ps, st))

function prepare_plot_data(q_pred::AbstractArray{T,4}, qg3p::QG3Model, ggsh2::QG3.GaussianGridtoSHTransform, shgg2::QG3.SHtoGaussianGridTransform; dev=cpu_device()) where T
    @assert size(q_pred,3) == size(q_test_evolved,3)
    if size(q_pred,3) == 6
    u_plot_pred = permutedims(q_pred[:, :,1:3,:], (3,1,2,4))
    v_plot_pred = permutedims(q_pred[:, :,4:6,:], (3,1,2,4))
    else
        error("Unexpected number of channels in prediction data")
    end
    return u_plot_pred, v_plot_pred           
end

u_plot_pred, v_plot_pred = prepare_plot_data(q_pred, qg3p, ggsh2, shgg2; dev=dev)
u_plot_evolved, v_plot_evolved = prepare_plot_data(q_test_evolved, qg3p, ggsh2, shgg2; dev=dev)
function compute_losses(u_plot_pred::AbstractArray{T,4}, u_plot_evolved::AbstractArray{T,4}, v_plot_pred::AbstractArray{T,4}, v_plot_evolved::AbstractArray{T,4}) where T
    loss_v = zeros(size(v_plot_evolved, 1))
    loss_u = zeros(size(u_plot_evolved, 1))
    for i in 1:size(v_plot_evolved, 1)
        loss_v[i] = mean(abs2, v_plot_pred[i,:,:,:] .- v_plot_evolved[i,:,:,:])
        scale = mean(abs2, v_plot_evolved[i,:,:,:])
        loss_v[i] = loss_v[i] / scale * 100
    end
    for i in 1:size(u_plot_evolved, 1)
        loss_u[i] = mean(abs2, u_plot_pred[i,:,:,:] .- u_plot_evolved[i,:,:,:])
        scale = mean(abs2, u_plot_evolved[i,:,:,:])
        loss_u[i] = loss_u[i] / scale * 100
    end
    return loss_u, loss_v
end
loss_u, loss_v = compute_losses(u_plot_pred, u_plot_evolved, v_plot_pred, v_plot_evolved)

mistake_u = to_array(u_plot_pred .- u_plot_evolved, dev)
mistake_v = to_array(v_plot_pred .- v_plot_evolved, dev)

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
    clims = (-1.1 * maximum(abs.(u_plot_pred[ilvl,:,:,:])), 1.1 * maximum(abs.(u_plot_pred[ilvl,:,:,:])))
    
    # Prediction animation
    create_animation(
        u_plot_pred, 
        string(root, "/figures/", model_string, "_prediction_fps20_sf_lvl$(ilvl).gif"),
        model_string * " Prediction",
        clims, ilvl, dev, qg3ppars
    )
    
    # Error animation  
    create_animation(
        mistake_u,
        string(root, "/figures/", model_string, "_error_fps20_sf_lvl$(ilvl).gif"),
        model_string * " Error",
        clims, ilvl, dev, qg3ppars
    )

    @printf "%s Percentual Error L2 norm at level %1d: %.9f\n" model_string ilvl loss_u[ilvl]
end

u_plot_evolved = to_array(u_plot_evolved, dev)
v_plot_evolved = to_array(v_plot_evolved, dev)

# Long rollout
n_days = 10
long_rollout_t_steps = Int(round(n_days / qg3ppars.time_unit)) #n_days in QG3 time units 
long_rollout_iter = Int(long_rollout_t_steps ÷ dt)
snapshots = min(100, long_rollout_iter)
time_scale_adjust = Int(round(long_rollout_t_steps / (snapshots-1)))

ggsh3 = QG3.GaussianGridtoSHTransform(qg3ppars, model_channels, N_batch=snapshots)
shgg3 = QG3.SHtoGaussianGridTransform(qg3ppars, model_channels, N_batch=snapshots)

q_stable = ESM_PINO.apply_n_times(trained_u_autoreg, q_test_rollout, long_rollout_iter; m=snapshots, μ=μ, σ=σ)
q_stable = cat(q_stable..., dims=4)
u_stable_plot_pred, v_stable_plot_pred = prepare_plot_data(q_stable, qg3p, ggsh3, shgg3; dev=dev)

for ilvl in 1:3
    plot_times = 1:size(u_stable_plot_pred, 4)
    lons = range(-180,180,qg3ppars.N_lons)    # Longitude 
    lats = rad2deg.(qg3ppars.lats)    # Latitude 
    clims = (-1.1 * maximum(abs.(u_stable_plot_pred[ilvl,:,:,:])), 1.1 * maximum(abs.(u_stable_plot_pred[ilvl,:,:,:])))
    fig_pred = Figure()
    ax_pred = GeoAxis(fig_pred[1, 1], dest="+proj=moll")
    data_obs_pred = Observable(permutedims(u_stable_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = GeoMakie.surface!(ax_pred,lons, lats, data_obs_pred, colorrange=clims, colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)

    GeoMakie.record(fig_pred, string(root, "/figures/", model_string, "_stability_fps20_sf_lvl$(ilvl).gif"), 
                     plot_times; framerate=5) do it
        data_obs_pred[] = permutedims(u_stable_plot_pred[ilvl,:,:,it],(2,1))
        ax_pred.title = model_string * @sprintf(" Prediction at time = %d - %.2f days", it, 
                         it * qg3ppars.time_unit * time_scale_adjust)
    end
end

acc_horizon = 30
acc = Vector{Vector{Float64}}(undef,3)
for ilvl in 1:3
    acc[ilvl] = ESM_PINOQG3.compute_ACC(q_stable[:,:,ilvl,1:acc_horizon], q_test_evolved[:,:,ilvl,dt:dt:dt*acc_horizon], qg3ppars, q_test_ltm[:,:,ilvl])
end
save(string(root, "/figures/", model_string, "_anomaly_correlation_coefficient.png"), Plots.plot(acc[1]))

function kinetic_energy_uv(u::AbstractArray{T,4}, v::AbstractArray{T,4}, m::QG3Model{T}) where T
    E_kin = transform_SH(T.(0.5 .* (u.^2 .+ v.^2)), QG3.GaussianGridtoSHTransform(m.p, N_batch=size(u,4)) )
    return E_kin[:,1,1,:]
end

function plot_kinetic_energy_uv(u::AbstractArray{T,4}, v::AbstractArray{T,4}, qg3p::QG3Model{T}; lvl=1) where T
    E_kin = kinetic_energy_uv(u, v, qg3p)
    Plots.plot(E_kin[lvl,:],
         title="Kinetic Energy at Level $(lvl)",
         xlabel="Time Step",
         ylabel="Kinetic Energy",
         legend=false)    
end

function plot_zonal_mean_velocity_uv(u::AbstractArray{T,4}, qg3p::QG3Model{T}; lvl=1, start_time=1, times=size(u,4)) where T
    zonal_min = Inf
    zonal_max = -Inf
    zonal_mean_velocities = zeros(Float32,size(u,1), qg3p.p.N_lats, size(u,4))
    for time = start_time:(start_time + times-1)
        zonal_mean_velocity = reshape(mean(u[:,:,:,time], dims=3), size(u)[1:2])
        zonal_min = min(zonal_min, minimum(zonal_mean_velocity[lvl,:]))
        zonal_max = max(zonal_max, maximum(zonal_mean_velocity[lvl,:]))
        zonal_mean_velocities[:,:,time - start_time + 1] = zonal_mean_velocity
    end
    anim = @animate for time in start_time:(start_time + times-1)
        p2 = Plots.plot(zonal_mean_velocities[lvl,:,time], qg3ppars.lats,
              title="Zonal Mean",
              xlabel="Velocity",
              ylabel="Latitude",
              legend=false,
              xlim=(zonal_min, zonal_max))
    end
    return anim    
end

zmv_anim = plot_zonal_mean_velocity_uv(u_stable_plot_pred, qg3p; lvl=1, start_time=1, times=size(u_stable_plot_pred,4))
gif(zmv_anim, string(root, "/figures/", model_string, "_zonal_mean_velocity_lvl1.gif"), fps=5)

save(string(root, "/figures/", model_string, "_kinetic_energy_stable.png"), plot_kinetic_energy_uv(u_stable_plot_pred, v_stable_plot_pred, qg3p; lvl=1))
