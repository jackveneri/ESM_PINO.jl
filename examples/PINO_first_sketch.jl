using ESM_PINO, Printf, CairoMakie, JLD2, OnlineStats, Lux, Random, Statistics, MLUtils, Optimisers, ParameterSchedulers

const gdev = gpu_device()
const cdev = cpu_device()

# Example usage

#Load and prepare data (generated with burgers_simulation_FD_schemes.jl)
@load "burgers_results.jld2" results ts x
sim1_results = permutedims(results, (2, 1, 3))

# Extract raw data
const t_max = ts[end]
const t_min = ts[1]
t_index = 700
Δt = 1
Δx = 1
n_training = min(200, size(sim1_results, 3))  # Ensure at least 200 training samples
u_t1 = reshape(sim1_results[1:Δx:end, t_index, 1:n_training], Int(size(sim1_results)[1]/Δx), 1, n_training)
target = reshape(sim1_results[1:Δx:end, t_index + Δt, 1:n_training], Int(size(sim1_results)[1]/Δx), 1, n_training)
#noise_level = 0.1
#u_t1 = ESM_PINO.add_noise(u_t1, noise_level=noise_level)
#target = ESM_PINO.add_noise(target, noise_level=noise_level)
const N = size(u_t1)[1]
const L = 1.f0
const ν = 0.001

const N_t = size(sim1_results)[2]
t_step_length = Δt * (t_max - t_min) / (N_t - 1)
x_grid = L*(0:N-1) / N
g = ESM_PINO.Grid(x_grid)
M1, M2 = ESM_PINO.BurgersFD(g).M, ESM_PINO.BurgersFD(g).M2 
# M1_sparse, M2_sparse = sparse(M1), sparse(M2)
M1_gpu = gdev(M1) 
M2_gpu = gdev(M2) 

# Normalize input and target separately
x_normalized, x_μ, x_σ = ESM_PINO.normalize_data(u_t1)
target_normalized, target_μ, target_σ = ESM_PINO.normalize_data(target)

#also generate a test set using burgers_simulation_FD_schemes.jl
@load "burgers_results_test.jld2" results ts x
sim2_results = permutedims(results, (2, 1, 3))

# Extract raw data
x_test = (reshape(sim2_results[:, t_index, :], size(sim2_results)[1], 1, size(sim2_results)[3]))
#x_test = ESM_PINO.add_noise(x_test, noise_level=noise_level)
target_test = (reshape(sim2_results[:, t_index + Δt, :], size(sim2_results)[1], 1, size(sim2_results)[3]))
# Get normalized predictions
#x_test_normalized,_,_ = normalize_data(x_test)
x_test_normalized = (x_test .- x_μ)./ x_σ
#=
fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_modes=(16,), 
    positional_embedding="grid1D"
)
=#
#rng = Random.default_rng()
#ps, st = Lux.initialparameters(rng, fno), Lux.initialstates(rng, fno)
#expected shape of the input is (spatial..., in_channels, batch)
#note that spatial must be >= n_modes for the spectral convolution to work
#y = fno(x, ps, st)[1]
#output shape is (spatial..., out_channels, batch)

fd_params = ESM_PINO.FDPhysicsLossParameters(ν, Δt, t_step_length, x_σ, x_μ, M1_gpu, M2_gpu)
spectral_params = ESM_PINO.SpectralPhysicsLossParameters(ν, Δt, L, x_σ, x_μ, t_step_length)

# Define a dictionary to map loss functions to plot titles
const LOSS_TITLES = Dict(
    nothing => "Training with Data Loss Only",
    fd_params => "Training with Physics-Informed Loss (Finite Difference)",
    spectral_params => "Training with Physics-Informed Loss (Spectral)"
)


function train_model(x, target; seed::Int=0,
    maxiters::Int=3000, hidden_channels::Int=32, parameters=nothing, subsampling=1, α=0.5f0)
    
    rng = Random.default_rng(seed)
    
    fno = FourierNeuralOperator(in_channels=1, 
    out_channels=1, 
    n_layers=4, 
    hidden_channels=hidden_channels, 
    n_modes=(15,), 
    positional_embedding="grid1D"
    )
    ps, st = Lux.setup(rng, fno) |> gdev
    dataloader = DataLoader((x, target); batchsize=1, shuffle=false) |> gdev

    opt = Optimisers.ADAM(0.001f0)
    lr = i -> Exp(0.001f0, 0.993f0).(i)
    train_state = Training.TrainState(fno, ps, st, opt)

    par = parameters
    PI_loss = ESM_PINO.create_physics_loss(par)
    loss_function = ESM_PINO.select_loss_function(PI_loss, subsampling=subsampling, α=α)

    total_loss_tracker, physics_loss_tracker, data_loss_tracker = ntuple(_ -> Lag(Float32, 32), 3)
    iter = 1
    for (x, target_data) in Iterators.cycle(dataloader)
        Optimisers.adjust!(train_state, lr(iter-1))
        _, loss, stats, train_state = Training.single_train_step!(AutoZygote(), loss_function, (x, target_data), train_state)
        
        fit!(total_loss_tracker, Float32(loss))
        fit!(physics_loss_tracker, Float32(stats.physics_loss))
        fit!(data_loss_tracker, Float32(stats.data_loss))

        mean_loss = mean(OnlineStats.value(total_loss_tracker))
        mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
        mean_data_loss = mean(OnlineStats.value(data_loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))
        
        if iter % 5 == 0 || iter == maxiters
            @printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f \
                 (%.9f) \t Data Loss: %.9f (%.9f)\n" iter maxiters loss mean_loss stats.physics_loss mean_physics_loss stats.data_loss mean_data_loss
        end
        
        iter += 1
        
        if iter > maxiters
            break
        end

    end
    return StatefulLuxLayer{true}(fno, cdev(train_state.parameters), cdev(train_state.states)), loss_function
end

const loss_results = Dict{String, Tuple{Vector{Float64}, Float64, Float64, Int, Int}}()

for (par, title) in LOSS_TITLES

    trained_model, used_loss_function = train_model(x_normalized, target_normalized; parameters=par, hidden_channels=64, maxiters=1000, subsampling=1) 
    trained_u = Lux.testmode(StatefulLuxLayer{true}(trained_model.model, trained_model.ps, trained_model.st))

    #target_test_normalized, target_test_μ, target_test_σ = normalize_data(target_test)
    u_pred_normalized = trained_u(Float32.(x_test_normalized))
    # Convert back to physical scale
    u_pred = ESM_PINO.denormalize_data(u_pred_normalized, target_μ, target_σ)
    #u_pred = denormalize_data(u_pred_normalized, target_test_μ, target_test_σ)
    #u_pred = trained_u(Float32.(x_test))

    # Get the title from the dictionary
    #plot_title = get(LOSS_TITLES, used_loss_function, "Unknown Loss Function")
    N_plot = size(u_pred)[1]
    x_plot = L * (0:N_plot-1) / N_plot

    loss = zeros(size(target_test)[3])
    for i in range(1,size(target_test)[3])
        loss[i] = mean(abs2, u_pred[:, 1, i] - target_test[:, 1, i])
    end

    sorted_loss = sort(loss, rev=true)
    begin
        fig = Figure(size = (800, 600))
        ax = CairoMakie.Axis(fig[1,1],xlabel="Simulation", ylabel="Loss", title="$title - Loss per Simulation")
        lines!(ax, 1:size(loss)[1], sorted_loss, label="$title")
        axislegend(ax; )
    end
    display(fig)
    scale = mean(abs2, target_test)
    loss ./= scale  # Scale the loss by the mean squared value of the target
    result = Statistics.mean(loss)
    std_err = Statistics.std(loss)
    worst = argmax(loss)
    best = argmin(loss)

    begin
        my_theme = merge(theme_latexfonts(), theme_minimal())
        set_theme!(my_theme, fontsize = 18, font = "Helvetica", color = :black)
        fig1 = Figure(size = (900, 600))
        title_worst = @sprintf("%s - Simulation n.%d, worst loss: %.4g", title, worst, loss[worst])
        ax = CairoMakie.Axis(fig1[1, 1]; xlabel="x", ylabel="u",title = title_worst)
        empty!(ax) 
        lines!(ax, x_plot, u_pred[:, 1, worst], label="Predicted", color="blue")
        lines!(ax, x_plot, target_test[:, 1, worst], label="Target", color="red")
        #lines!(ax, x_plot, x_test[:, 1, worst], label="Previous State", color="green")
        axislegend(ax; )


        fig2 = Figure(size = (900, 600))
        title_best = @sprintf("%s - Simulation n.%d, best loss: %.4g", title, best, loss[best])
        ax2 = CairoMakie.Axis(fig2[1, 1]; xlabel="x", ylabel="u", title = title_best)
        empty!(ax2) 
        lines!(ax2, x_plot, u_pred[:, 1, best], label="Predicted", color="blue")
        lines!(ax2, x_plot, target_test[:, 1, best], label="Ground Truth", color="red", linestyle=:dash)
        lines!(ax2, x_plot, x_test[:, 1, best], label="Initial State", color="black", linestyle=:dot)
        axislegend(ax2; )
    end
    display(fig1)
    display(fig2)
    #save("$(title)_noise_lvl=$(noise_level)_best.svg", fig2)
    save("$(title)_worst.svg", fig1)
    save("$(title)_best.svg", fig2)

    loss_results[title] = (
        copy(loss),          # Full loss vector
        result,              # Mean loss
        std_err,            # Standard error of the loss
        worst,               # Index of worst case
        best                 # Index of best case
    )
end


for (par, title) in LOSS_TITLES
    percer = (loss_results[title][2])*100
    percstd = (loss_results[title][3])*100
    @printf("%s - mean squared relative error: %.3g ± %.3g %%\n", title, percer, percstd)
end

begin
    my_theme = merge(theme_latexfonts(), theme_minimal())
    set_theme!(my_theme, fontsize = 24, font = "Helvetica", color = :black)
    comparison_fig = Figure(size=(1600, 1000))  # Adjusted for vertical layout
    
    # First plot (top row)
    ax1 = CairoMakie.Axis(comparison_fig[1, 1], 
        title="Comparative Error Across Simulations",
        xlabel="Simulation Index", 
        ylabel=L"L_2 Error",
        yscale= log10,
        )
    
    # Second plot (bottom row)
    ax2 = CairoMakie.Axis(comparison_fig[2, 1], 
        title="Error Distribution Statistics",
        xticks=(1:length(loss_results), collect(keys(loss_results))),
        xlabel="Training Type",
        ylabel=L"L_2 Error",
        yscale= log10,
        )

    # Prepare data (same as before)
    categories = Int[]
    vals = Float64[]
    color_indices = Int[]  
    colors = 1:length(loss_results)
    colormap = :viridis
    colorrange = (1, length(loss_results))
    
    for (idx, (title, (loss, mean_loss, _, _))) in enumerate(loss_results)
        append!(categories, fill(idx, length(loss)))
        append!(vals, loss)
        append!(color_indices, fill(idx, length(loss)))
        lines!(ax1, 1:length(loss), loss, color=colors[idx], colormap=colormap, colorrange=colorrange, label=title)
    end

    # Boxplot (same as before)
    CairoMakie.boxplot!(ax2, categories, vals,
        color=color_indices, colormap=colormap,
        show_notch=false, whiskerwidth=0.4,
        strokecolor=:black, strokewidth=1,
        mediancolor=:white
    )

    # Add mean indicators
    for (idx, (title, (loss, mean_loss, _, _))) in enumerate(loss_results)
        CairoMakie.scatter!(ax2, [idx], [mean(loss)], 
            marker=:diamond, color=:black, strokecolor=:black, markersize=20)
    end

    # Add legend to first plot
    axislegend(ax1;
        position=:rt,
        title="Training Types",
        framecolor=:transparent,
        padding=(10, 10, 10, 10)
    )

    # Create dummy elements for boxplot legend
    # Black thicker line underneath (outline)
    median_outline = lines!(ax2, [0.0, 0.0], [0.0, 0.0], color=:black, linewidth=4, visible=false)

    # White thinner line on top
    median_line = lines!(ax2, [0.0, 0.0], [0.0, 0.0], color=:white, linewidth=2, visible=false)
    mean_marker = CairoMakie.scatter!(ax2, [0], [0], marker=:diamond, color=:black, strokecolor=:black, visible=false)

    # Add statistics legend to second plot
    
    axislegend(
        ax2,
        [median_outline, mean_marker],
        ["Median Value", "Mean Value"];
        title="Statistics",
        titleposition=:left,
        orientation=:horizontal,
        framevisible=false,
        position=:rt
    )
    

    # Adjust row proportions and spacing
    rowsize!(comparison_fig.layout, 1, Relative(0.4))  # Top plot height
    rowsize!(comparison_fig.layout, 2, Relative(0.6))  # Bottom plot height
    rowgap!(comparison_fig.layout, 15)  # Space between rows

    display(comparison_fig)
end

#save("comparison_fig_noise_level=$(noise_level).svg", comparison_fig)
#save("comparison_fig_dt=$(t_step_length).svg", comparison_fig)
save("comparison_fig_n_training=$(n_training).svg", comparison_fig)