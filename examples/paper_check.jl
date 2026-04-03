root = dirname(@__DIR__)
dir = @__DIR__
using Pkg
Pkg.activate(dir)
Pkg.instantiate()

using ESM_PINO, Printf, CairoMakie, JLD2, OnlineStats, Lux, Random, Statistics, MLUtils, Optimisers, ParameterSchedulers, CUDA

const gdev = gpu_device()
const cdev = cpu_device()

# ===== Hyperparameters =====
t_index  = 100
t_res    = 64
Δt       = 4
Δx       = 4
hidden_channels = 16
batch_size      = 4
num_examples    = 200
num_valid       = 100
N_sims          = 800
N_val           = 100
nepochs         = 100
lr_0            = 1e-3
α               = 0.5f0
gpu             = true
modes           = (50, 50)

# ===== Load and prepare training data =====
@load string(root, "/data/burgers_results.jld2") results ts x
sim1_results = permutedims(results, (2, 1, 3))

const t_max = ts[end]
const t_min = ts[1]
const N_t   = size(sim1_results, 2)
const L     = 1.f0
const ν     = 0.001

t_step_length = Δt * (t_max - t_min) / (N_t - 1)

N_x = Int(size(sim1_results, 1) / Δx)
x_grid = L * (0:N_x-1) / N_x
t_grid = ts[t_index+t_res*Δt:Δt:t_index+Δt*(2*t_res-1)] .- t_min

# Extract raw windows before any normalization
all_u = Float32.(reshape(
    sim1_results[1:Δx:end, t_index:Δt:t_index+(t_res-1)*Δt, :],
    N_x, t_res, 1, size(sim1_results, 3)))
all_target = Float32.(reshape(
    sim1_results[1:Δx:end, t_index+t_res*Δt:Δt:t_index+Δt*(2*t_res-1), :],
    N_x, t_res, 1, size(sim1_results, 3)))

# Split before normalizing
u_train_raw      = all_u[:, :, :, 1:N_sims]
target_train_raw = all_target[:, :, :, 1:N_sims]
u_val_raw        = all_u[:, :, :, N_sims+1:N_sims+N_val]
target_val_raw   = all_target[:, :, :, N_sims+1:N_sims+N_val]

# Compute statistics from training input only, apply everywhere
x_normalized,     x_μ, x_σ = ESM_PINO.normalize_data(u_train_raw)
target_normalized            = ESM_PINO.normalize_data(target_train_raw, x_μ, x_σ)
x_val_normalized             = ESM_PINO.normalize_data(u_val_raw,        x_μ, x_σ)
target_val_normalized        = ESM_PINO.normalize_data(target_val_raw,   x_μ, x_σ)

# FD matrices
const N = N_x
x_grid = L * (0:N-1) / N
t_grid = ts[t_index+t_res*Δt:Δt:t_index+Δt*(2*t_res-1)] .- t_min
x_plot = x_grid
t_plot = t_grid

g      = ESM_PINO.Grid(x_grid)
M1_gpu = gdev(ESM_PINO.BurgersFD(g).M)
M2_gpu = gdev(ESM_PINO.BurgersFD(g).M2)

fd_params       = ESM_PINO.FDPhysicsLossParameters(ν, Δt, t_step_length, x_σ, x_μ, M1_gpu, M2_gpu)
spectral_params = ESM_PINO.SpectralPhysicsLossParameters(ν, Δt, L, x_σ, x_μ, t_step_length,N)

# Load test data and normalize with training statistics
@load string(root, "/data/burgers_results_test.jld2") results ts x
sim2_results = permutedims(results, (2, 1, 3))

x_test      = Float32.(reshape(
    sim2_results[1:Δx:end, t_index:Δt:t_index+(t_res-1)*Δt, :],
    N_x, t_res, 1, size(sim2_results, 3)))
target_test = reshape(
    sim2_results[1:Δx:end, t_index+t_res*Δt:Δt:t_index+Δt*(2*t_res-1), :],
    N_x, t_res, 1, size(sim2_results, 3))
x_test_normalized = ESM_PINO.normalize_data(x_test, x_μ, x_σ)

# ===== Loss configurations =====
const LOSS_TITLES = Dict(
    nothing         => "Training with Data Loss Only",
    fd_params       => "Training with Physics-Informed Loss (Finite Difference)",
    spectral_params => "Training with Physics-Informed Loss (Spectral)",
)

const loss_results = Dict{String, Tuple{Vector{Float64}, Float64, Float64, Int, Int}}()

for (par, title) in LOSS_TITLES
    @printf "\n===== %s =====\n" title

    rng = Random.default_rng(42)
    fno = FourierNeuralOperator(
        in_channels=1,
        out_channels=1,
        n_layers=4,
        hidden_channels=hidden_channels,
        n_modes=modes,
        positional_embedding=false
    )
    ps, st = Lux.setup(rng, fno) |> gdev

    opt         = Optimisers.ADAM(lr_0)
    train_state = Training.TrainState(fno, ps, st, opt)

    val_dataloader = DataLoader(
        (x_val_normalized, target_val_normalized);
        batchsize=batch_size, shuffle=false) |> gdev

    # ===== Pre-training: 10 epochs with data loss only =====
    println("  Pre-training (data loss only)...")
    pretrain_loss_fn = ESM_PINO.select_loss_function(
        ESM_PINO.create_physics_loss(nothing); subsampling=1, α=1.0f0)

    for epoch in 1:10
        epoch_start  = time()
        train_idx    = randperm(rng, N_sims)[1:num_examples]
        train_loader = DataLoader(
            (x_normalized[:, :, :, train_idx], target_normalized[:, :, :, train_idx]);
            batchsize=batch_size, shuffle=true) |> gdev

        acc_loss = 0.0; n_train = 0
        for (x_batch, target_batch) in train_loader
            _, loss, _, train_state = Training.single_train_step!(
                AutoZygote(), pretrain_loss_fn, (x_batch, target_batch), train_state)
            isnan(loss) && throw(ArgumentError("NaN loss in pre-training epoch $epoch"))
            acc_loss += loss * size(x_batch)[end]
            n_train  += size(x_batch)[end]
        end

        valid_loss = 0.0; n_valid = 0
        trained_u  = Lux.testmode(StatefulLuxLayer{true}(fno, train_state.parameters, train_state.states))
        for (xv, tv) in val_dataloader
            valid_loss += Lux.MSELoss()(trained_u(xv), tv) * size(xv)[end]
            n_valid    += size(xv)[end]
        end

        @printf "  Pre-train Epoch [%2d/10] (%.2fs) | Train: %.6f | Val: %.6f\n" epoch (time()-epoch_start) (acc_loss/n_train) (valid_loss/n_valid)
    end

    # ===== Main training with selected loss =====
    println("  Main training...")
    PI_loss       = ESM_PINO.create_physics_loss(par)
    loss_function = ESM_PINO.select_loss_function(PI_loss; subsampling=1, α=α)

    total_loss_tracker, physics_loss_tracker, data_loss_tracker = ntuple(_ -> Lag(Float32, 32), 3)

    valid_loss_min    = Inf
    no_improv_counter = 0
    current_lr        = lr_0

    for epoch in 1:nepochs
        epoch_start  = time()
        train_idx    = randperm(rng, N_sims)[1:num_examples]
        train_loader = DataLoader(
            (x_normalized[:, :, :, train_idx], target_normalized[:, :, :, train_idx]);
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

        trained_u  = Lux.testmode(StatefulLuxLayer{true}(fno, train_state.parameters, train_state.states))
        valid_loss = 0.0; n_valid = 0
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
    u_pred_norm = final_model(Float32.(x_test_normalized))
    u_pred      = ESM_PINO.denormalize_data(u_pred_norm, x_μ, x_σ)

    scale    = mean(abs2, target_test)
    loss_vec = [mean(abs2, u_pred[:,:,1,i] .- target_test[:,:,1,i]) / scale
                for i in 1:size(target_test, 4)]

    result  = mean(loss_vec)
    std_err = std(loss_vec)
    worst   = argmax(loss_vec)
    best    = argmin(loss_vec)

    mistake = abs2.(u_pred[:,:,1,:] .- target_test[:,:,1,:]) ./ scale * 100

    loss_results[title] = (loss_vec, result, std_err, worst, best)
    @printf "%s — mean relative MSE: %.3g ± %.3g %%\n" title result*100 std_err*100

    # ===== Heatmap plots =====
    my_theme = merge(theme_latexfonts(), theme_minimal())
    set_theme!(my_theme, fontsize=18, font="Helvetica", color=:black)

    # Worst case
    begin
        fig1 = Figure(size=(900, 700))
        worst_pred   = u_pred[:, :, 1, worst]
        worst_target = target_test[:, :, 1, worst]
        vmin_w = min(minimum(worst_pred), minimum(worst_target))
        vmax_w = max(maximum(worst_pred), maximum(worst_target))

        ax1 = Axis(fig1[1, 1]; xlabel="x", ylabel="t", title="$title, worst - prediction")
        hm1 = heatmap!(ax1, x_plot, t_plot, worst_pred; colorrange=(vmin_w, vmax_w))
        Colorbar(fig1[1, 2], hm1; label="u", vertical=true)

        ax2 = Axis(fig1[2, 1]; xlabel="x", ylabel="t", title="$title, worst - target")
        hm2 = heatmap!(ax2, x_plot, t_plot, worst_target; colorrange=(vmin_w, vmax_w))
        Colorbar(fig1[2, 2], hm2; label="u", vertical=true)

        ax3 = Axis(fig1[3, 1]; xlabel="x", ylabel="t", title="$title, worst - error")
        hm3 = heatmap!(ax3, x_plot, t_plot, mistake[:,:,worst]; colormap=:dense)
        Colorbar(fig1[3, 2], hm3; label="percentual error", vertical=true)
    end

    # Best case
    begin
        fig2 = Figure(size=(900, 700))
        best_pred   = u_pred[:, :, 1, best]
        best_target = target_test[:, :, 1, best]
        vmin_b = min(minimum(best_pred), minimum(best_target))
        vmax_b = max(maximum(best_pred), maximum(best_target))

        ax4 = Axis(fig2[1, 1]; xlabel="x", ylabel="t", title="$title, best - prediction")
        hm4 = heatmap!(ax4, x_plot, t_plot, best_pred; colorrange=(vmin_b, vmax_b))
        Colorbar(fig2[1, 2], hm4; label="u", vertical=true)

        ax5 = Axis(fig2[2, 1]; xlabel="x", ylabel="t", title="$title, best - target")
        hm5 = heatmap!(ax5, x_plot, t_plot, best_target; colorrange=(vmin_b, vmax_b))
        Colorbar(fig2[2, 2], hm5; label="u", vertical=true)

        ax6 = Axis(fig2[3, 1]; xlabel="x", ylabel="t", title="$title, best - error")
        hm6 = heatmap!(ax6, x_plot, t_plot, mistake[:,:,best]; colormap=:dense)
        Colorbar(fig2[3, 2], hm6; label="percentual error", vertical=true)
    end

    display(fig1); display(fig2)
    save(string(root, "/$(title)_worst.png"), fig1)
    save(string(root, "/$(title)_best.png"),  fig2)

    # ===== Save model =====
    model   = final_model.model
    ps_save = final_model.ps
    st_save = final_model.st
    @save joinpath(root, "models/FNO_burgers_$(replace(title, ' '=>'_')).jld2") model ps_save st_save x_μ x_σ Δt ν
end

# ===== Summary statistics =====
for (par, title) in LOSS_TITLES
    percer  = loss_results[title][2] * 100
    percstd = loss_results[title][3] * 100
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

    categories    = Int[]
    vals          = Float64[]
    color_indices = Int[]
    colors        = 1:length(loss_results)
    colormap      = :viridis
    colorrange    = (1, length(loss_results))

    for (idx, (title, (loss, mean_loss, _, _, _))) in enumerate(loss_results)
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

    axislegend(ax1; position=:rt, title="Training Types",
        framecolor=:transparent, padding=(10,10,10,10))

    median_outline = lines!(ax2, [0.0,0.0], [0.0,0.0], color=:black, linewidth=4, visible=false)
    mean_marker    = CairoMakie.scatter!(ax2, [0], [0], marker=:diamond,
        color=:black, strokecolor=:black, visible=false)
    axislegend(ax2, [median_outline, mean_marker], ["Median Value", "Mean Value"];
        title="Statistics", titleposition=:left, orientation=:horizontal,
        framevisible=false, position=:rt)

    rowsize!(comparison_fig.layout, 1, Relative(0.4))
    rowsize!(comparison_fig.layout, 2, Relative(0.6))
    rowgap!(comparison_fig.layout, 15)

    display(comparison_fig)
    save(string(root, "/comparison_fig.png"), comparison_fig)
end

# ===== Derivative diagnostic =====
begin
    x_diag = range(0, stop=L*(N-1)/N, length=N)
    t_diag = range(0, stop=1, length=t_res)
    states = zeros(Float64, N, t_res, 1, 2)

    for i in 1:t_res
        states[:, i, 1, 1] = cos.(2π .* (x_diag .- t_diag[i]))
        states[:, i, 1, 2] = sin.(2π .* (x_diag .- t_diag[i]))
    end

    actual_∂f_∂x = cat(
        -2π .* sin.(2π .* (x_diag .- reshape(t_diag,1,:))) .* cos.(2π .* (x_diag .- reshape(t_diag,1,:))),
         2π .* sin.(2π .* (x_diag .- reshape(t_diag,1,:))) .* cos.(2π .* (x_diag .- reshape(t_diag,1,:)));
        dims=4)
    actual_∂u_∂xx = cat(
        -4π^2 .* cos.(2π .* (x_diag .- reshape(t_diag,1,:))),
        -4π^2 .* sin.(2π .* (x_diag .- reshape(t_diag,1,:)));
        dims=4)

    ∂f_∂x_fd, ∂u_∂xx_fd         = ESM_PINO.spatial_derivatives_batch(gdev(states), fd_params)
    ∂f_∂x_sp, ∂u_∂xx_sp         = ESM_PINO.spatial_derivatives_batch(CuArray(states), spectral_params)

    selected_t     = 16
    selected_batch = 2

    for (label, ∂f_∂x, ∂u_∂xx) in [("FD", ∂f_∂x_fd, ∂u_∂xx_fd), ("Spectral", ∂f_∂x_sp, ∂u_∂xx_sp)]
        fig_diag = Figure(size=(900, 500))
        ax_f  = Axis(fig_diag[1,1]; xlabel="x", title="∂f/∂x — $label")
        ax_u  = Axis(fig_diag[1,2]; xlabel="x", title="∂²u/∂x² — $label")

        lines!(ax_f, Array(∂f_∂x[:, selected_t, 1, selected_batch]),  label="Computed")
        lines!(ax_f, actual_∂f_∂x[:, selected_t, 1, selected_batch],  label="Actual", linestyle=:dash)
        axislegend(ax_f)

        lines!(ax_u, Array(∂u_∂xx[:, selected_t, 1, selected_batch]), label="Computed")
        lines!(ax_u, actual_∂u_∂xx[:, selected_t, 1, selected_batch], label="Actual", linestyle=:dash)
        axislegend(ax_u)

        display(fig_diag)
        save(string(root, "/derivative_check_$(label).png"), fig_diag)
    end
end

# ===== Physics loss sanity check =====
println("\n===== Physics Loss Sanity Check =====")

n_check = 10  # number of random samples to check
check_idx = randperm(Random.default_rng(0), N_sims)[1:n_check]

for (par, title) in LOSS_TITLES
    isnothing(par) && continue

    residuals = Float64[]
    for i in check_idx
        u_sample      = gdev(x_normalized[:, :, :, i:i])
        target_sample = gdev(target_normalized[:, :, :, i:i])
        res = ESM_PINO.physics_loss(target_sample, u_sample, par)
        push!(residuals, Float64(res))
    end

    @printf "%-55s | mean: %.3e | std: %.3e | min: %.3e | max: %.3e\n" title mean(residuals) std(residuals) minimum(residuals) maximum(residuals)
end

# Also check on a known analytical solution: u(x,t) = cos(2π(x - t))
# for which Burgers residual is exactly zero (inviscid, ν→0 limit it's not,
# but useful to check derivative scaling)
println("\n  Analytical check (cos wave, should be near zero only for ν≈0):")
x_diag = range(0, stop=L*(N-1)/N, length=N)
t_diag = range(0, stop=1, length=t_res)
u_analytical = Float32.(reshape(
    [cos(2π * (xi - ti)) for xi in x_diag, ti in t_diag],
    N, t_res, 1, 1))

# Denormalize is not needed here — analytical check is in physical space,
# so we pass x_μ=0, x_σ=1 via a temporary params with no normalization
fd_params_check       = ESM_PINO.FDPhysicsLossParameters(ν, Δt, t_step_length, 1.0, 0.0, M1_gpu, M2_gpu)
spectral_params_check = ESM_PINO.SpectralPhysicsLossParameters(ν, Δt, L, 1.0, 0.0, t_step_length,N)

for (par_check, label) in [(fd_params_check, "FD"), (spectral_params_check, "Spectral")]
    res = ESM_PINO.physics_loss(gdev(u_analytical), gdev(u_analytical), par_check)
    @printf "  %-10s analytical residual: %.3e\n" label Float64(res)
end
println("=======================================\n")

println("\n===== Physics Loss Sanity Check =====")
n_check = 10
check_idx = randperm(Random.default_rng(0), N_sims)[1:n_check]

# Build params with no normalization to check in physical space
fd_params_phys = ESM_PINO.FDPhysicsLossParameters(ν, Δt, t_step_length, 1.0, 0.0, M1_gpu, M2_gpu)
spectral_params_phys = ESM_PINO.SpectralPhysicsLossParameters(ν, Δt, L, 1.0, 0.0, t_step_length, N)

for (par, label) in [(fd_params_phys, "FD"), (spectral_params_phys, "Spectral")]
    residuals = Float64[]
    for i in check_idx
        # Use raw (physical) consecutive windows — u_t2 should follow from u_t1
        u1 = gdev(Float32.(all_u[:, :, :, i:i]))       # physical units, no normalization
        u2 = gdev(Float32.(all_target[:, :, :, i:i]))   # physical units, next time window
        res = ESM_PINO.physics_loss(u2, u1, par)
        push!(residuals, Float64(res))
    end
    @printf "%-12s | mean: %.3e | std: %.3e | min: %.3e | max: %.3e\n" label mean(residuals) std(residuals) minimum(residuals) maximum(residuals)
end

# Also check with normalized data using training params
println("  With normalization (should match training conditions):")
for (par, label) in [(fd_params, "FD"), (spectral_params, "Spectral")]
    residuals = Float64[]
    for i in check_idx
        u1 = gdev(x_normalized[:, :, :, i:i])
        u2 = gdev(target_normalized[:, :, :, i:i])
        res = ESM_PINO.physics_loss(u2, u1, par)
        push!(residuals, Float64(res))
    end
    @printf "  %-12s | mean: %.3e | std: %.3e | min: %.3e | max: %.3e\n" label mean(residuals) std(residuals) minimum(residuals) maximum(residuals)
end
println("=========================================\n")

println("\n===== Residual Term Decomposition =====")
n_check = 10
check_idx = randperm(Random.default_rng(0), N_sims)[1:n_check]

T = Float32
fd_params_phys       = ESM_PINO.FDPhysicsLossParameters(T(ν), Δt, T(t_step_length), T(1.0), T(0.0), M1_gpu, M2_gpu)
spectral_params_phys = ESM_PINO.SpectralPhysicsLossParameters(T(ν), Δt, T(L), T(1.0), T(0.0), T(t_step_length), N)

for (par, label) in [(fd_params_phys, "FD"), (spectral_params_phys, "Spectral")]
    dudt_norms   = Float64[]
    dfdx_norms   = Float64[]
    d2udx2_norms = Float64[]
    residuals    = Float64[]

    for i in check_idx
        u1 = gdev(Float32.(all_u[:, :, :, i:i]))
        u2 = gdev(Float32.(all_target[:, :, :, i:i]))

        # ∂u/∂t via central differences on interior points
        ∂u_∂t = (u2[:, 3:end, :, :] .- u2[:, 1:end-2, :, :]) ./ (2 * par.t_step_length)

        # spatial terms on interior time points
        ∂f_∂x, ∂u_∂xx = ESM_PINO.spatial_derivatives_batch(u2[:, 2:end-1, :, :], par)

        push!(dudt_norms,   Float64(mean(abs, ∂u_∂t)))
        push!(dfdx_norms,   Float64(mean(abs, ∂f_∂x)))
        push!(d2udx2_norms, Float64(mean(abs, ν .* ∂u_∂xx)))
        push!(residuals,    Float64(mean(abs2, ∂u_∂t .+ ∂f_∂x .- ν .* ∂u_∂xx)))
    end

    @printf "\n%s term magnitudes (mean |·|):\n" label
    @printf "  |∂u/∂t|       mean: %.3e  std: %.3e\n" mean(dudt_norms)   std(dudt_norms)
    @printf "  |∂f/∂x|       mean: %.3e  std: %.3e\n" mean(dfdx_norms)   std(dfdx_norms)
    @printf "  |ν·∂²u/∂x²|  mean: %.3e  std: %.3e\n" mean(d2udx2_norms) std(d2udx2_norms)
    @printf "  residual MSE  mean: %.3e  std: %.3e\n" mean(residuals)    std(residuals)
end

# Also print t_step_length and the actual time spacing in the data
@printf "\nt_step_length used in params: %.6f\n" t_step_length
@printf "t_grid[1]: %.6f, t_grid[end]: %.6f, span: %.6f\n" t_grid[1] t_grid[end] (t_grid[end]-t_grid[1])
@printf "Actual dt between consecutive snapshots in window: %.6f\n" (t_grid[2]-t_grid[1])
@printf "N_t=%d, t_max=%.3f, t_min=%.3f, Δt=%d\n" N_t t_max t_min Δt
println("=========================================\n")