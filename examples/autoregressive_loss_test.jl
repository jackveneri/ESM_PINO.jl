using Lux, LuxCUDA, Zygote, ESM_PINO, Statistics, JLD2, Random, Printf, OnlineStats, Optimisers, ParameterSchedulers, FFTW, MLUtils

function autoregressive_loss(
    model::StatefulLuxLayer,
    (u0,target)::Tuple{AbstractArray,AbstractArray},
    n_steps::Int,
    params::Union{ESM_PINO.FDPhysicsLossParameters,ESM_PINO.SpectralPhysicsLossParameters},
    λ::Float32
)
    states_arr = autoregressive_propagate(model, u0, n_steps)
    pred = states_arr[:,:,:,end]  # Get the last state as prediction
    states_arr = states_arr .* params.x_σ .+ params.x_μ  # Denormalize states if necessary

    # Vectorized time derivative calculation
    Δt = params.t_step_length / n_steps
    forward_diff = (states_arr[:, :, :, end:end] .- states_arr[:, :, :, end-1:end-1]) ./ Δt
    central_diff = (states_arr[:, :, :, 3:end] .- states_arr[:, :, :, 1:end-2]) ./ (2Δt)
    du_dt = cat(central_diff, forward_diff; dims=4)

    # Batched physics calculations
    ∂f_∂x_batch, ∂u_∂xx_batch = spatial_derivatives_batch(states_arr[:, :, :, 2:end], params)
    # Vectorized residuals and boundary loss
    residuals = du_dt .+ ∂f_∂x_batch .- (params.ν .* ∂u_∂xx_batch)
    boundary_loss = mean(abs2, states_arr[1:1, :, :, 2:end] .- states_arr[end:end, :, :, 2:end])
    physics_loss = mean(abs2, residuals) + boundary_loss

    # Data loss calculation
    data_loss = mean(abs2, pred .- target)

    return (loss= ( 1 - λ ) * data_loss + λ * physics_loss, 
            physics_loss=physics_loss,
            boundary_loss=boundary_loss, 
            data_loss=data_loss)
end
function autoregressive_propagate(model, u0::AbstractArray, n_steps::Int)
    # Initialize immutable tuple with initial state
    states = (u0,)
    
    # Build state sequence through tuple concatenation
    for _ in 1:n_steps
        u_next = model(states[end])
        states = (states..., u_next)  # Immutable update
    end
    
    # Combine into single array
    return cat(states...; dims=4)
end

function spatial_derivatives_batch(states::AbstractArray, params::ESM_PINO.FDPhysicsLossParameters)
    dims = size(states)
    states_reshaped = reshape(states, size(states, 1), :)
    f = 0.5 .* (states_reshaped .^ 2)
    ∂f_∂x = reshape(params.M2_gpu * f, dims)
    ∂u_∂xx = reshape(params.M1_gpu * states_reshaped, dims)
    return ∂f_∂x, ∂u_∂xx
end

function spatial_derivatives_batch(states::AbstractArray, params::ESM_PINO.SpectralPhysicsLossParameters)
    
    results = map(x -> spatial_derivative(x, params), eachslice(states; dims=4))

    df_dx_batch = cat(map(res -> res[1], results)...; dims=4)
    d2u_dx2_batch = cat(map(res -> res[2], results)...; dims=4)
    return df_dx_batch, d2u_dx2_batch
end

function spatial_derivative(u_t2::AbstractArray, params::ESM_PINO.SpectralPhysicsLossParameters)
    _, d2u_dx2 = ESM_PINO.spectral_derivative(u_t2, params.L)
    f = u_t2.^2 ./ 2
    f_hat = fft(f)
    f_hat = ESM_PINO.dealias(f_hat, params.L) 
    k = ESM_PINO.compute_k(f, params.L)
    df_dx_hat = im .* k .* f_hat
    df_dx = real(ifft(df_dx_hat))
    return df_dx, d2u_dx2
end

const gdev = gpu_device()
const cdev = cpu_device()

function normalize_data(data)
    μ = mean(data)
    σ = std(data)
    return (data .- μ) ./ σ, μ, σ
end

function denormalize_data(data, μ, σ)
    return data .* σ .+ μ
end

@load "burgers_results.jld2" results ts x
sim1_results = Float32.(permutedims(results, (2, 1, 3)))


const t_max = ts[end]
const t_min = ts[1]
t_index = 900
Δt = 10
Δx = 1

u_t1 = reshape(sim1_results[1:Δx:end, t_index, :], Int(size(sim1_results)[1]/Δx), 1, size(sim1_results)[3])
#u_t1 = add_adaptive_noise(reshape(sim1_results[1:Δx:end, t_index, :], Int(size(sim1_results)[1]/Δx), 1, size(sim1_results)[3]), snr_dB=30)
x_normalized, x_μ, x_σ = normalize_data(u_t1)
target = reshape(sim1_results[1:Δx:end, t_index + Δt, :], Int(size(sim1_results)[1]/Δx), 1, size(sim1_results)[3])
target_normalized, target_μ, target_σ = normalize_data(target)
const N = size(u_t1)[1]
const L = 1.0
const ν = 0.001

const N_t = size(sim1_results)[2]
t_step_length = Δt * (t_max - t_min) / (N_t - 1)
x_grid = L*(0:N-1) / N
g = ESM_PINO.Grid(x_grid)
M1, M2 = ESM_PINO.BurgersFD(g).M, ESM_PINO.BurgersFD(g).M2 
# M1_sparse, M2_sparse = sparse(M1), sparse(M2)
M1_gpu = gdev(M1) 
M2_gpu = gdev(M2) 


#also generate a test set using burgers_simulation_FD_schemes.jl
@load "burgers_results_test.jld2" results ts x
sim2_results = Float32.(permutedims(results, (2, 1, 3)))


# Extract raw data
x_test = reshape(sim2_results[:, t_index, :], size(sim2_results)[1], 1, size(sim2_results)[3]) |> gdev
x_test_normalized = (x_test .- x_μ) ./ x_σ |> gdev
target_test = (reshape(sim2_results[:, t_index + Δt, :], size(sim2_results)[1], 1, size(sim2_results)[3])) |> gdev
# Get normalized predictions


fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    n_modes=(15,), 
    positional_embedding="grid1D"
)   

rng = Random.default_rng()
ps, st = Lux.setup(rng, fno) |> gdev
#fno(x_test, ps, st)[1]
test_model = Lux.testmode(StatefulLuxLayer{true}(fno, ps, st)) 
#expected shape of the input is (spatial..., in_channels, batch)
#note that spatial must be >= n_modes for the spectral convolution to work
#y = fno(x, ps, st)[1]
#output shape is (spatial..., out_channels, batch)

fd_params = ESM_PINO.FDPhysicsLossParameters(ν, Δt, t_step_length, x_σ, x_μ, M1_gpu, M2_gpu)
spectral_params = ESM_PINO.SpectralPhysicsLossParameters(ν, Δt, t_step_length, x_σ, x_μ, L)
n_steps = 2
autoregressive_loss(
    test_model,
    (x_test, target_test),
    n_steps,
    spectral_params,
    0.01f0
)

using Lux.Training, Optimisers, Zygote, MLUtils

# 1. Define wrapper function for training
function loss_function(model::AbstractLuxLayer, ps::NamedTuple, st::NamedTuple, batch::Tuple{AbstractArray, AbstractArray})
    u_net = StatefulLuxLayer{true}(model, ps, st)
    u0, target = batch
    loss = autoregressive_loss(
        u_net,
        (u0, target),
        n_steps,  # n_steps
        spectral_params,
        0.01f0
    )
    total_loss = loss[1]
    physics_loss = loss[2]
    boundary_loss = loss[3]
    data_loss = loss[4] 
    return (total_loss, (st), (;physics_loss, boundary_loss, data_loss))
end

# 2. Create training state
function prepare_training(model, ps, st)
    # Define optimizer
    optimizer = Optimisers.Adam(0.001f0)
    
    # Create training state
    training_state = Training.TrainState(
        model,
        ps,
        st,
        optimizer,
    
    )
    
    return training_state
end

# 3. Training loop
function train_model!(training_state, train_loader, num_epochs)
        iter = 1
        lr = i -> Exp(0.001f0, 0.993f0).(i)
        total_loss_tracker, physics_loss_tracker, boundary_loss_tracker, data_loss_tracker = ntuple(_ -> Lag(Float32, 32), 4)
    for (x, target_data) in Iterators.cycle(train_loader)
        Optimisers.adjust!(training_state, lr(iter))
        # Perform training step
        _, loss, stats, training_state = Training.single_train_step!(
            AutoZygote(),
            loss_function,
            (x, target_data),
            training_state
        )
        
        fit!(total_loss_tracker, Float32(loss))
        fit!(physics_loss_tracker, Float32(stats.physics_loss))
        fit!(boundary_loss_tracker, Float32(stats.boundary_loss))
        fit!(data_loss_tracker, Float32(stats.data_loss))

        mean_loss = mean(OnlineStats.value(total_loss_tracker))
        mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
        mean_boundary_loss = mean(OnlineStats.value(boundary_loss_tracker))
        mean_data_loss = mean(OnlineStats.value(data_loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))
        
        if iter % 5 == 0 || iter == num_epochs
            @printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f \
                 (%.9f) \t Boundary Loss: %.9f (%.9f) \t Data Loss: %.9f (%.9f)\n" iter num_epochs loss mean_loss stats.physics_loss mean_physics_loss stats.boundary_loss mean_boundary_loss stats.data_loss mean_data_loss
            #@printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \n" iter num_epochs loss mean_loss
            #GC.gc()
        end
        
        iter += 1
        
        if iter > num_epochs
            break
        end

    end
    return StatefulLuxLayer{true}(fno, training_state.parameters, training_state.states)

end

function train_model2!(training_state, train_loader, num_epochs)
        iter = 1
        lr = i -> Exp(0.001f0, 0.993f0).(i)
        total_loss_tracker = ntuple(_ -> Lag(Float32, 32), 1)[1]
    for (x, target_data) in Iterators.cycle(train_loader)
        Optimisers.adjust!(training_state, lr(iter))
        # Perform training step
        _, loss, stats, training_state = Training.single_train_step!(
            AutoZygote(),
            MSELoss(),
            (x, target_data),
            training_state
        )
        
        fit!(total_loss_tracker, Float32(loss))
        #fit!(physics_loss_tracker, Float32(stats.physics_loss))
        #fit!(boundary_loss_tracker, Float32(stats.boundary_loss))
        #fit!(data_loss_tracker, Float32(stats.data_loss))

        mean_loss = mean(OnlineStats.value(total_loss_tracker))
        #mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
        #mean_boundary_loss = mean(OnlineStats.value(boundary_loss_tracker))
        #mean_data_loss = mean(OnlineStats.value(data_loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))
        
        if iter % 5 == 0 || iter == num_epochs
            #@printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f \
            #     (%.9f) \t Boundary Loss: %.9f (%.9f) \t Data Loss: %.9f (%.9f)\n" iter num_epochs loss mean_loss stats.physics_loss mean_physics_loss stats.boundary_loss mean_boundary_loss stats.data_loss mean_data_loss
            @printf "Iteration: [%5d / %5d] \t Loss: %.9f (%.9f) \n" iter num_epochs loss mean_loss
            GC.gc()
        end
        
        iter += 1
        
        if iter > num_epochs
            break
        end

    end
    return StatefulLuxLayer{true}(fno, training_state.parameters, training_state.states)

end

# 4. Usage example
# Prepare training state
training_state = prepare_training(fno, ps, st)

# Create data loader
train_loader = DataLoader((x_normalized, target_normalized), batchsize=1, shuffle=false) |> gdev

# Run training
trained_model = train_model2!(training_state, train_loader, 1000)

training_state = prepare_training(fno, trained_model.ps, trained_model.st)

trained_model = train_model!(training_state, train_loader, 300)

trained_u = Lux.testmode(StatefulLuxLayer{true}(trained_model.model, trained_model.ps, trained_model.st))

function apply_n_times(f, x, n)
    y = x
    for _ in 1:n
        y = f(y)
    end
    return y
end

u_pred_normalized = apply_n_times(trained_u, x_test_normalized, n_steps)
u_pred = denormalize_data(u_pred_normalized, target_μ, target_σ)
title = "Burgers Autoregressive Loss Test"

N_plot = size(u_pred)[1]
x_plot = L*(0:N_plot-1) / N_plot

target_test = target_test |> gdev
loss = zeros(size(target_test)[3])

for i in range(1,size(target_test)[3])
    loss[i] = mean(abs2, u_pred[:, 1, i] - target_test[:, 1, i])
end

using CairoMakie
u_pred = Array(u_pred)
target_test = Array(target_test)
begin
    fig = Figure(size = (800, 600))
    ax = CairoMakie.Axis(fig[1,1],xlabel="Simulation", ylabel="Loss", title="$title - Loss per Simulation")
    lines!(ax, 1:size(loss)[1], loss, label="$title")
    axislegend(ax; )
end
display(fig)
result = mean(abs2, loss)

worst = argmax(loss)
best = argmin(loss)

begin
    fig1 = Figure(size = (900, 600))
    ax = CairoMakie.Axis(fig1[1, 1]; xlabel="x", ylabel="u",title = "$title - Simulation n.$(worst), worst loss achieved: $(loss[worst])")
    empty!(ax) 
    lines!(ax, x_plot, u_pred[:, 1, worst], label="Predicted", color="blue")
    lines!(ax, x_plot, target_test[:, 1, worst], label="Target", color="red")
    #lines!(ax, x_plot, x_test[:, 1, worst], label="Previous State", color="green")
    axislegend(ax; )


    fig2 = Figure(size = (900, 600))
    ax2 = CairoMakie.Axis(fig2[1, 1]; xlabel="x", ylabel="u", title = "$title - Simulation n.$(best), best loss achieved: $(loss[best])")
    empty!(ax2) 
    lines!(ax2, x_plot, u_pred[:, 1, best], label="Predicted", color="blue")
    lines!(ax2, x_plot, target_test[:, 1, best], label="Target", color="red")
    #lines!(ax2, x_plot, x_test[:, 1, best], label="Previous State", color="green")
    axislegend(ax2; )
end

display(fig1)
display(fig2)
save("$(title)_worst.svg", fig1)
save("$(title)_best.svg", fig2)

