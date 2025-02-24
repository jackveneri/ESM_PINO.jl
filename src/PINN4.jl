using ADTypes, Lux, Optimisers, Zygote, Random, Printf, Statistics, MLUtils, OnlineStats,
      CairoMakie
using LuxCUDA

CUDA.allowscalar(false)

const gdev = gpu_device()
const cdev = cpu_device()

const ν = 0.001f0

struct PINN2{U, V} <: Lux.AbstractLuxContainerLayer{(:u, :v)}
    u::U
    v::V
end

function create_mlp(act, hidden_dims)
    return Chain(
        Dense(2 => hidden_dims, act),
        Dense(hidden_dims => hidden_dims, act),
        Dense(hidden_dims => hidden_dims, act),
        Dense(hidden_dims => hidden_dims, act),
        Dense(hidden_dims => 1)
    )
end

function PINN2(; hidden_dims::Int=32)
    return PINN2(
        create_mlp(gelu, hidden_dims),
        create_mlp(gelu, hidden_dims),
    )
end

@views function physics_informed_loss_function(
    u::StatefulLuxLayer, v::StatefulLuxLayer, xt::AbstractArray)
∂u_∂xt = only(Zygote.gradient(sum ∘ u, xt))
∂u_∂x, ∂u_∂t = ∂u_∂xt[1:1, :], ∂u_∂xt[2:2, :]
∂v_∂x = only(Zygote.gradient(sum ∘ v, xt))[1:1, :]
v_xt = v(xt)
return (
    mean(abs2, ∂u_∂t .+ (u(xt) .* ∂u_∂x) .- (ν .* ∂v_∂x)) +
    mean(abs2, v_xt .- ∂u_∂x) 
)
end

function mse_loss_function(u::StatefulLuxLayer, target::AbstractArray, xt::AbstractArray)
    return MSELoss()(u(xt), target)
end

function loss_function(model, ps, st, (xt, target_data, xt_bc, target_bc))
    u_net = StatefulLuxLayer{true}(model.u, ps.u, st.u)
    v_net = StatefulLuxLayer{true}(model.v, ps.v, st.v)
    physics_loss = physics_informed_loss_function(u_net, v_net, xt)
    data_loss = mse_loss_function(u_net, target_data, xt)
    bc_loss = mse_loss_function(u_net, target_bc, xt_bc)
    loss = physics_loss + data_loss + bc_loss
    return (
        loss,
        (; u=u_net.st, v=v_net.st),
        (; physics_loss, data_loss, bc_loss)
    )
end

#import data here
using JLD2
@load "burgers_results.jld2" results ts x
sim1_results = results[:, :, 2]

begin
    N = size(sim1_results, 1)
    grid_len = 128
    grid_step = N ÷ grid_len

    x_grid = range(0.0f0, 1.0f0; length=grid_len)
    t_grid = range(0.0f0, 3.0f0; length=grid_len)
    xt = stack([[elem...] for elem in vec(collect(Iterators.product(x_grid, t_grid)))])

    target_data = sim1_results[1:grid_step:end, 1:grid_step:end]
    target_data = reshape(target_data', 1, :)

    bc_len = 128

    x = collect(range(0.0f0, 1.0f0; length=bc_len))
    t = collect(range(0.0f0, 3.0f0; length=bc_len))

    xt_bc = hcat(
        stack((x, zeros(Float32, bc_len)); dims=1),
        stack((zeros(Float32, bc_len), t); dims=1),
        stack((ones(Float32, bc_len), t); dims=1)
    )
    
    target_bc = hcat(
        sim1_results[1, :],
        sim1_results[:, 1],
        sim1_results[:, end]
    )
    
    target_bc = reshape(target_bc, 1, :)

    min_target_bc, max_target_bc = extrema(target_bc)
    min_data, max_data = extrema(target_data)
    min_pde_val, max_pde_val = min(min_data, min_target_bc), max(max_data, max_target_bc)

    xt = (xt .- minimum(xt)) ./ (maximum(xt) .- minimum(xt))
    xt_bc = (xt_bc .- minimum(xt_bc)) ./ (maximum(xt_bc) .- minimum(xt_bc))
    target_bc = (target_bc .- min_pde_val) ./ (max_pde_val - min_pde_val)
    target_data = (target_data .- min_pde_val) ./ (max_pde_val - min_pde_val)
end

function train_model(xt, target_data, xt_bc, target_bc; seed::Int=0,
    maxiters::Int=50000, hidden_dims::Int=32)
rng = Random.default_rng()
Random.seed!(rng, seed)

pinn = PINN2(; hidden_dims)
ps, st = Lux.setup(rng, pinn) |> gdev

bc_dataloader = DataLoader((xt_bc, target_bc); batchsize=32, shuffle=true) |> gdev
pde_dataloader = DataLoader((xt, target_data); batchsize=32, shuffle=true) |> gdev

train_state = Training.TrainState(pinn, ps, st, Adam(0.01f0))
lr = i -> i < 5000 ? 0.01f0 : (i < 10000 ? 0.001f0 : 0.0001f0)

total_loss_tracker, physics_loss_tracker, data_loss_tracker, bc_loss_tracker = ntuple(
    _ -> Lag(Float32, 32), 4)

iter = 1
for ((xt_batch, target_data_batch), (xt_bc_batch, target_bc_batch)) in zip(
    Iterators.cycle(pde_dataloader), Iterators.cycle(bc_dataloader))
    Optimisers.adjust!(train_state, lr(iter))

    _, loss, stats, train_state = Training.single_train_step!(
        AutoZygote(), loss_function, (
            xt_batch, target_data_batch, xt_bc_batch, target_bc_batch),
        train_state)

    fit!(total_loss_tracker, loss)
    fit!(physics_loss_tracker, stats.physics_loss)
    fit!(data_loss_tracker, stats.data_loss)
    fit!(bc_loss_tracker, stats.bc_loss)

    mean_loss = mean(OnlineStats.value(total_loss_tracker))
    mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
    mean_data_loss = mean(OnlineStats.value(data_loss_tracker))
    mean_bc_loss = mean(OnlineStats.value(bc_loss_tracker))

    isnan(loss) && throw(ArgumentError("NaN Loss Detected"))

    if iter % 500 == 1 || iter == maxiters
        @printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f \
                 (%.9f) \t Data Loss: %.9f (%.9f) \t BC \
                 Loss: %.9f (%.9f)\n" iter maxiters loss mean_loss stats.physics_loss mean_physics_loss stats.data_loss mean_data_loss stats.bc_loss mean_bc_loss
    end

    iter += 1
    iter ≥ maxiters && break
end

return StatefulLuxLayer{true}(
    pinn, cdev(train_state.parameters), cdev(train_state.states))
end

trained_model = train_model(xt, target_data, xt_bc, target_bc; hidden_dims=64)
trained_u = Lux.testmode(StatefulLuxLayer{true}(
trained_model.model.u, trained_model.ps.u, trained_model.st.u))

ts, xs = 0.0f0:3.0/(N-1):3.0f0, 0.0f0:1/(N-1):1.0f0
grid = stack([[elem...] for elem in vec(collect(Iterators.product(xs, ts)))])

u_real = sim1_results

grid_normalized = (grid .- minimum(grid)) ./ (maximum(grid) .- minimum(grid))
u_pred = reshape(trained_u(grid_normalized), length(xs), length(ts))
u_pred = u_pred .* (max_pde_val - min_pde_val) .+ min_pde_val

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="u")

    x_min, x_max = minimum(xs), maximum(xs)

    # Calculate global y limits for consistent scaling
    y_min = minimum([u_pred; u_real])
    y_max = maximum([u_pred; u_real])

    CairoMakie.record(fig, "prediction_vs_real.gif", 1:length(ts); framerate=20) do i
        empty!(ax)  # Clear previous frame content
        ax.title = "Time Evolution | t = $(round(ts[i], digits=3))"
        
        # Plot real solution and prediction
        lines!(ax, xs, u_real[i, :]; color=:blue, label="Real")
        lines!(ax, xs, u_pred[:, i]; color=:red, label="Prediction", linestyle=:dash)
        
        # Add legend and set axis limits
        axislegend(ax; position=:rb)
        CairoMakie.ylims!(ax, y_min, y_max)
        CairoMakie.xlims!(ax, x_min, x_max)
        
        return fig
    end

    fig
end