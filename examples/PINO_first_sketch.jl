root = dirname(@__DIR__)
dir = @__DIR__
using Pkg
Pkg.activate(dir)
Pkg.instantiate()

using ESM_PINO, Printf, CairoMakie, JLD2, OnlineStats, Lux, Random, Statistics, MLUtils, Optimisers, ParameterSchedulers

const gdev = gpu_device()
const cdev = cpu_device()

# ===== Hyperparameters =====
dt = 2
Δx = 2
t_index = 100
hidden_channels = 32
batch_size = 10
num_examples = 500
num_valid = 100
N_sims = 500
N_val = 100
nepochs = 100
lr_0 = 2e-3
gpu = true
modes = (128,)

# ===== Load and prepare data =====
@load string(root, "/data/burgers_results.jld2") results ts x
sim1_results = permutedims(results, (2, 1, 3))

const t_max = ts[end]
const t_min = ts[1]
const N_t   = size(sim1_results, 2)
const L     = 1.f0
const ν     = 0.001

t_step_length = dt * (t_max - t_min) / (N_t - 1)

raw    = sim1_results[1:Δx:end, :, :]
const N = size(raw, 1)
x_grid = L * (0:N-1) / N

all_u      = Float32.(reshape(raw[:, t_index,      1:N_sims+N_val], N, 1, N_sims+N_val))
all_target = Float32.(reshape(raw[:, t_index + dt, 1:N_sims+N_val], N, 1, N_sims+N_val))

u_train_raw      = all_u[:, :, 1:N_sims]
target_train_raw = all_target[:, :, 1:N_sims]
u_val_raw        = all_u[:, :, N_sims+1:end]
target_val_raw   = all_target[:, :, N_sims+1:end]

x_normalized,     x_μ, x_σ = ESM_PINO.normalize_data(u_train_raw)
target_normalized            = ESM_PINO.normalize_data(target_train_raw, x_μ, x_σ)
x_val_normalized             = ESM_PINO.normalize_data(u_val_raw,        x_μ, x_σ)
target_val_normalized        = ESM_PINO.normalize_data(target_val_raw,   x_μ, x_σ)
#x_normalized = ESM_PINO.add_noise(x_normalized)
#target_normalized = ESM_PINO.add_noise(target_normalized)
#x_val_normalized = ESM_PINO.add_noise(x_val_normalized)
# ===== FD matrices =====
g      = ESM_PINO.Grid(x_grid)
M1_gpu = gdev(ESM_PINO.BurgersFD(g).M)
M2_gpu = gdev(ESM_PINO.BurgersFD(g).M2)

fd_params       = ESM_PINO.FDPhysicsLossParameters(ν, dt, t_step_length, x_σ, x_μ, M1_gpu, M2_gpu)
spectral_params = ESM_PINO.SpectralPhysicsLossParameters(ν, dt, L, x_σ, x_μ, t_step_length,N)

# ===== Loss configurations =====
const LOSS_TITLES = Dict(
    nothing         => "Data Loss Only",
    fd_params       => "Physics-Informed (FD)",
    spectral_params => "Physics-Informed (Spectral)",
)

const loss_results = Dict{String, Tuple{Vector{Float64}, Float64, Float64, Int, Int}}()

# ===== Load test data once =====
@load string(root, "/data/burgers_results_test.jld2") results ts x
sim2_results = permutedims(results, (2, 1, 3))
x_test_raw   = Float32.(reshape(sim2_results[1:Δx:end, t_index,      :], N, 1, :))
target_test  =          reshape(sim2_results[1:Δx:end, t_index + dt, :], N, 1, :)
x_test_norm  = ESM_PINO.normalize_data(x_test_raw, x_μ, x_σ)
#x_test_norm  = ESM_PINO.add_noise(x_test_norm)
x_plot       = L * (0:N-1) / N

for (par, title) in LOSS_TITLES
    @printf "\n===== %s =====\n" title

    rng = Random.default_rng(42)
    fno = FourierNeuralOperator(
        in_channels=1,
        out_channels=1,
        n_layers=4,
        hidden_channels=hidden_channels,
        n_modes=modes,
        use_norm=true,
        bias=false,
        positional_embedding=false,
    )
    ps, st = Lux.setup(rng, fno) |> gdev

    opt         = Optimisers.ADAM(lr_0)
    train_state = Training.TrainState(fno, ps, st, opt)

    PI_loss       = ESM_PINO.create_physics_loss(par)
    loss_function = ESM_PINO.select_loss_function(PI_loss; subsampling=1, α=0.5f0)

    total_loss_tracker, physics_loss_tracker, data_loss_tracker = ntuple(_ -> Lag(Float32, 32), 3)

    val_dataloader = DataLoader(
        (x_val_normalized, target_val_normalized);
        batchsize=batch_size, shuffle=false) |> gdev

    valid_loss_min    = Inf
    no_improv_counter = 0
    current_lr        = lr_0

    for epoch in 1:nepochs
        epoch_start = time()

        train_idx    = randperm(rng, N_sims)[1:num_examples]
        train_loader = DataLoader(
            (x_normalized[:, :, train_idx], target_normalized[:, :, train_idx]);
            batchsize=batch_size, shuffle=true) |> gdev

        acc_loss = acc_phys = acc_data = 0.0
        n_train  = 0

        for (x_batch, target_batch) in train_loader
            _, loss, stats, train_state = Training.single_train_step!(
                AutoZygote(), loss_function, (x_batch, target_batch), train_state)

            fit!(total_loss_tracker,   Float32(loss))
            fit!(physics_loss_tracker, Float32(stats.physics_loss))
            fit!(data_loss_tracker,    Float32(stats.data_loss))

            bs        = size(x_batch)[end]
            acc_loss += loss               * bs
            acc_phys += stats.physics_loss * bs
            acc_data += stats.data_loss    * bs
            n_train  += bs

            isnan(loss) && throw(ArgumentError("NaN loss at epoch $epoch"))
        end

        # Validation
        trained_u  = Lux.testmode(StatefulLuxLayer{true}(fno, train_state.parameters, train_state.states))
        valid_loss = 0.0
        n_valid    = 0
        for (xv, tv) in val_dataloader
            valid_loss += Lux.MSELoss()(trained_u(xv), tv) * size(xv)[end]
            n_valid    += size(xv)[end]
        end
        valid_loss /= n_valid

        @printf "Epoch [%3d/%3d] (%.2fs) | Train: %.6f (phys: %.6f, data: %.6f) | Val: %.6f\n" epoch nepochs (time()-epoch_start) (acc_loss/n_train) (acc_phys/n_train) (acc_data/n_train) valid_loss

        if valid_loss < valid_loss_min
            valid_loss_min    = valid_loss
            no_improv_counter = 0
        else
            no_improv_counter += 1
        end

        if no_improv_counter >= 3
            current_lr /= 2
            Optimisers.adjust!(train_state, current_lr)
            @printf "  → LR reduced to %.2e\n" current_lr
        end

        if no_improv_counter >= 10
            println("  → Early stopping triggered.")
            break
        end
    end

    # ===== Evaluation =====
    final_model = Lux.testmode(StatefulLuxLayer{true}(fno, cdev(train_state.parameters), cdev(train_state.states)))
    u_pred_norm = final_model(Float32.(x_test_norm))
    u_pred      = ESM_PINO.denormalize_data(u_pred_norm, x_μ, x_σ)

    scale    = mean(abs2, target_test)
    loss_vec = [mean(abs2, u_pred[:,1,i] .- target_test[:,1,i]) / scale
                for i in 1:size(target_test, 3)]

    result  = mean(loss_vec)
    std_err = std(loss_vec)
    worst   = argmax(loss_vec)
    best    = argmin(loss_vec)

    loss_results[title] = (loss_vec, result, std_err, worst, best)
    @printf "%s — mean relative MSE: %.3g ± %.3g %%\n" title result*100 std_err*100

    # ===== Per-configuration plots (worst/best) =====
    my_theme = merge(theme_latexfonts(), theme_minimal())
    set_theme!(my_theme, fontsize=18, font="Helvetica", color=:black)

    fig1 = Figure(size=(900, 600))
    title_worst = @sprintf("%s - Simulation n.%d, worst loss: %.4g", title, worst, loss_vec[worst])
    ax1 = CairoMakie.Axis(fig1[1, 1]; xlabel="x", ylabel="u", title=title_worst)
    lines!(ax1, x_plot, u_pred[:, 1, worst],       label="Predicted",     color=:blue)
    lines!(ax1, x_plot, target_test[:, 1, worst],  label="Target",        color=:red)
    lines!(ax1, x_plot, x_test_raw[:, 1, worst],   label="Initial State", color=:black, linestyle=:dot)
    axislegend(ax1)

    fig2 = Figure(size=(900, 600))
    title_best = @sprintf("%s - Simulation n.%d, best loss: %.4g", title, best, loss_vec[best])
    ax2 = CairoMakie.Axis(fig2[1, 1]; xlabel="x", ylabel="u", title=title_best)
    lines!(ax2, x_plot, u_pred[:, 1, best],        label="Predicted",     color=:blue)
    lines!(ax2, x_plot, target_test[:, 1, best],   label="Ground Truth",  color=:red, linestyle=:dash)
    lines!(ax2, x_plot, x_test_raw[:, 1, best],    label="Initial State", color=:black, linestyle=:dot)
    axislegend(ax2)

    display(fig1)
    display(fig2)
    save(string(root, "/$(title)_worst.svg"), fig1)
    save(string(root, "/$(title)_best.svg"),  fig2)

    # ===== Save model =====
    model   = final_model.model
    ps_save = final_model.ps
    st_save = final_model.st
    @save joinpath(root, "models/FNO_burgers_$(replace(title, ' '=>'_')).jld2") model ps_save st_save x_μ x_σ dt ν
end

# ===== Summary statistics =====
for (par, title) in LOSS_TITLES
    percer   = loss_results[title][2] * 100
    percstd  = loss_results[title][3] * 100
    @printf "%s — mean squared relative error: %.3g ± %.3g %%\n" title percer percstd
end

# ===== Comparison figure =====
begin
    my_theme = merge(theme_latexfonts(), theme_minimal())
    set_theme!(my_theme, fontsize=24, font="Helvetica", color=:black)
    comparison_fig = Figure(size=(1600, 1000))

    ax1 = CairoMakie.Axis(comparison_fig[1, 1],
        title="Comparative Error Across Simulations",
        xlabel="Simulation Index",
        ylabel=L"L_2 Error",
        yscale=log10)

    ax2 = CairoMakie.Axis(comparison_fig[2, 1],
        title="Error Distribution Statistics",
        xticks=(1:length(loss_results), collect(keys(loss_results))),
        xlabel="Training Type",
        ylabel=L"L_2 Error",
        yscale=log10)

    categories  = Int[]
    vals        = Float64[]
    color_indices = Int[]
    colors      = 1:length(loss_results)
    colormap    = :viridis
    colorrange  = (1, length(loss_results))

    for (idx, (title, (loss, mean_loss, _, _,_))) in enumerate(loss_results)
        append!(categories,    fill(idx, length(loss)))
        append!(vals,          loss)
        append!(color_indices, fill(idx, length(loss)))
        lines!(ax1, 1:length(loss), loss,
            color=colors[idx], colormap=colormap, colorrange=colorrange, label=title)
    end

    CairoMakie.boxplot!(ax2, categories, vals,
        color=color_indices, colormap=colormap,
        show_notch=false, whiskerwidth=0.4,
        strokecolor=:black, strokewidth=1,
        mediancolor=:white)

    for (idx, (title, (loss, _, _, _, _))) in enumerate(loss_results)
        CairoMakie.scatter!(ax2, [idx], [mean(loss)],
            marker=:diamond, color=:black, strokecolor=:black, markersize=20)
    end

    axislegend(ax1; position=:rt, title="Training Types", framecolor=:transparent, padding=(10,10,10,10))

    median_outline = lines!(ax2, [0.0, 0.0], [0.0, 0.0], color=:black, linewidth=4, visible=false)
    mean_marker    = CairoMakie.scatter!(ax2, [0], [0], marker=:diamond, color=:black, strokecolor=:black, visible=false)
    axislegend(ax2, [median_outline, mean_marker], ["Median Value", "Mean Value"];
        title="Statistics", titleposition=:left, orientation=:horizontal,
        framevisible=false, position=:rt)

    rowsize!(comparison_fig.layout, 1, Relative(0.4))
    rowsize!(comparison_fig.layout, 2, Relative(0.6))
    rowgap!(comparison_fig.layout, 15)

    display(comparison_fig)
    save(string(root, "/comparison_fig.svg"), comparison_fig)
end