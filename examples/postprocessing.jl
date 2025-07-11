root = dirname(@__DIR__)
using Pkg
Pkg.activate(root)
Pkg.instantiate()
using ESM_PINO, JLD2, CairoMakie, Printf, Statistics
@load string(root, "/data/plotting_utils.jld2") clims time_scale 
my_theme = merge(theme_latexfonts(), theme_minimal())
set_theme!(my_theme, fontsize = 24, font = "Helvetica", color = :black)
@load string(root, "/models/FNO_results.jld2")  sf_plot_pred sf_plot_evolved mistake loss ps st

for ilvl in 1:3
    
    plot_times = 1:size(sf_plot_pred, 4)

    # Animation for predictions using Observables
    fig_pred = Figure()
    ax_pred = Axis(fig_pred[1, 1])
    data_obs_pred = Observable(permutedims(sf_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = heatmap!(ax_pred, data_obs_pred, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)
    
    CairoMakie.record(fig_pred, string(root, "/figures/FNO_prediction_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_pred[] = permutedims(sf_plot_pred[ilvl,:,:,it],(2,1))
        ax_pred.title = @sprintf("FNO Prediction at time = %d - %.2f d", it, it * time_scale)
    end

    # Animation for errors using Observables
    fig_err = Figure()
    ax_err = Axis(fig_err[1, 1])
    data_obs_err = Observable(permutedims(mistake[ilvl,:,:,1],(2,1)))
    hm_err = heatmap!(ax_err, data_obs_err, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_err[1, 2], hm_err)
    
    CairoMakie.record(fig_err, string(root, "/figures/FNO_error_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_err[] = permutedims(mistake[ilvl,:,:,it],(2,1))
        ax_err.title = @sprintf("FNO Error at time = %d - %.2f d", it, it * time_scale)    
    end

    # Calculate and print errors
    error2PINO = mean(abs2, mistake[ilvl,:,:,:])
    error1PINO = mean(abs, mistake[ilvl,:,:,:])
    @printf "FNO Percentual Error L2 norm at level %1d: %.9f\n" ilvl loss[ilvl]
end
@load "FNO_PINO_results.jld2"  sf_plot_pred sf_plot_evolved mistake loss ps st

for ilvl in 1:3
    
    plot_times = 1:size(sf_plot_pred, 4)

    # Animation for predictions using Observables
    fig_pred = Figure()
    ax_pred = Axis(fig_pred[1, 1])
    data_obs_pred = Observable(permutedims(sf_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = heatmap!(ax_pred, data_obs_pred, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)
    
    CairoMakie.record(fig_pred, string(root, "/figures/FNO_PINO_prediction_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_pred[] = permutedims(sf_plot_pred[ilvl,:,:,it],(2,1))
        ax_pred.title = @sprintf("FNO PINO Prediction at time = %d - %.2f d", it, it * time_scale)
    end

    # Animation for errors using Observables
    fig_err = Figure()
    ax_err = Axis(fig_err[1, 1])
    data_obs_err = Observable(permutedims(mistake[ilvl,:,:,1],(2,1)))
    hm_err = heatmap!(ax_err, data_obs_err, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_err[1, 2], hm_err)
    
    CairoMakie.record(fig_err, string(root, "/figures/FNO_PINO_error_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_err[] = permutedims(mistake[ilvl,:,:,it],(2,1))
        ax_err.title = @sprintf("FNO PINO Error at time = %d - %.2f d", it, it * time_scale)    
    end

    # Calculate and print errors
    error2PINO = mean(abs2, mistake[ilvl,:,:,:])
    error1PINO = mean(abs, mistake[ilvl,:,:,:])
    @printf "FNO PINO Percentual Error L2 norm at level %1d: %.9f\n" ilvl loss[ilvl]
end
@load "SFNO_results.jld2"  sf_plot_pred sf_plot_evolved mistake loss ps st

for ilvl in 1:3
    
    plot_times = 1:size(sf_plot_pred, 4)

    # Animation for predictions using Observables
    fig_pred = Figure()
    ax_pred = Axis(fig_pred[1, 1])
    data_obs_pred = Observable(permutedims(sf_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = heatmap!(ax_pred, data_obs_pred, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)
    
    CairoMakie.record(fig_pred, string(root, "/figures/SFNO_prediction_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_pred[] = permutedims(sf_plot_pred[ilvl,:,:,it],(2,1))
        ax_pred.title = @sprintf("SFNO Prediction at time = %d - %.2f d", it, it * time_scale)
    end

    # Animation for errors using Observables
    fig_err = Figure()
    ax_err = Axis(fig_err[1, 1])
    data_obs_err = Observable(permutedims(mistake[ilvl,:,:,1],(2,1)))
    hm_err = heatmap!(ax_err, data_obs_err, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_err[1, 2], hm_err)
    
    CairoMakie.record(fig_err, string(root, "/figures/SFNO_error_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_err[] = permutedims(mistake[ilvl,:,:,it],(2,1))
        ax_err.title = @sprintf("SFNO Error at time = %d - %.2f d", it, it * time_scale)    
    end

    # Calculate and print errors
    error2PINO = mean(abs2, mistake[ilvl,:,:,:])
    error1PINO = mean(abs, mistake[ilvl,:,:,:])
    @printf "SFNO Percentual Error L2 norm at level %1d: %.9f\n" ilvl loss[ilvl]
end

@load "SFNO_PINO_results.jld2"  sf_plot_pred sf_plot_evolved mistake loss ps st

for ilvl in 1:3
    
    plot_times = 1:size(sf_plot_pred, 4)

    # Animation for predictions using Observables
    fig_pred = Figure()
    ax_pred = Axis(fig_pred[1, 1])
    data_obs_pred = Observable(permutedims(sf_plot_pred[ilvl,:,:,1],(2,1)))
    hm_pred = heatmap!(ax_pred, data_obs_pred, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_pred[1, 2], hm_pred)
    
    CairoMakie.record(fig_pred, string(root, "/figures/SFNO_PINO_prediction_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_pred[] = permutedims(sf_plot_pred[ilvl,:,:,it],(2,1))
        ax_pred.title = @sprintf("SFNO PINO Prediction at time = %d - %.2f d", it, it * time_scale)
    end

    # Animation for errors using Observables
    fig_err = Figure()
    ax_err = Axis(fig_err[1, 1])
    data_obs_err = Observable(permutedims(mistake[ilvl,:,:,1],(2,1)))
    hm_err = heatmap!(ax_err, data_obs_err, colorrange=clims[ilvl,:], colormap=:balance)
    Colorbar(fig_err[1, 2], hm_err)
    
    CairoMakie.record(fig_err, string(root, "/figures/SFNO_PINO_error_fps20_sf_lvl$(ilvl).gif"), plot_times;
        framerate=20) do it
        # Update observable data
        data_obs_err[] = permutedims(mistake[ilvl,:,:,it],(2,1))
        ax_err.title = @sprintf("SFNO PINO Error at time = %d - %.2f d", it, it * time_scale)    
    end

    # Calculate and print errors
    error2PINO = mean(abs2, mistake[ilvl,:,:,:])
    error1PINO = mean(abs, mistake[ilvl,:,:,:])
    @printf "SFNO PINO Percentual Error L2 norm at level %1d: %.9f\n" ilvl loss[ilvl]
end
