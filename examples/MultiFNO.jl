using ESM_PINO, Printf, NetCDF, OnlineStats, QG3, CFTime, Dates, Lux, CUDA, Random, Optimisers, MLUtils, ParameterSchedulers

struct MultiFNO <: Lux.Lux.AbstractLuxContainerLayer{(:fno1, :fno2, :fno3)}
    fno1::FourierNeuralOperator
    fno2::FourierNeuralOperator
    fno3::FourierNeuralOperator
end

function Lux.initialparameters(rng::AbstractRNG, m::MultiFNO)
    ps1 = Lux.initialparameters(rng, m.fno1)
    ps2 = Lux.initialparameters(rng, m.fno2)
    ps3 = Lux.initialparameters(rng, m.fno3)
    return (fno1=ps1, fno2=ps2, fno3=ps3)
end

function Lux.initialstates(rng::AbstractRNG, m::MultiFNO)
    st1 = Lux.initialstates(rng, m.fno1)
    st2 = Lux.initialstates(rng, m.fno2)
    st3 = Lux.initialstates(rng, m.fno3)
    return (fno1=st1, fno2=st2, fno3=st3)
end

function (m::MultiFNO)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    # Split input into individual levels
    x1 = x[1:1, :, :, :, :]  # Level 1: (1, lat, lon, 1, batch)
    x2 = x[2:2, :, :, :, :]  # Level 2
    x3 = x[3:3, :, :, :, :]  # Level 3

    # Process each level with corresponding FNO
    y1, st1 = m.fno1(dropdims(x1; dims=1), ps.fno1, st.fno1)  # (lat, lon, 1, batch)
    y2, st2 = m.fno2(dropdims(x2; dims=1), ps.fno2, st.fno2)
    y3, st3 = m.fno3(dropdims(x3; dims=1), ps.fno3, st.fno3)

    # Concatenate outputs along level dimension
    y = cat(
        reshape(y1, 1, size(y1)...),  # (1, lat, lon, 1, batch)
        reshape(y2, 1, size(y2)...),
        reshape(y3, 1, size(y3)...);
        dims=1
    )

    # Update state
    new_st = (fno1=st1, fno2=st2, fno3=st3)
    return y, new_st
end

function train_model(x::AbstractArray, target::AbstractArray; seed::Int=0,
    maxiters::Int=3000, hidden_channels::Int=32, loss_function::Function=QG3_loss_function)
    
    rng = Random.default_rng(seed)
    
    # Create three 2D FNOs (one per atmospheric level)
    fno1 = FourierNeuralOperator(
        in_channels=1, 
        out_channels=1, 
        n_layers=3, 
        hidden_channels=hidden_channels, 
        n_modes=(32, 32),  # 2D Fourier modes (lat, lon)
        positional_embedding="grid"
    )
    fno2 = FourierNeuralOperator(
        in_channels=1, 
        out_channels=1, 
        n_layers=3, 
        hidden_channels=hidden_channels, 
        n_modes=(32, 32),
        positional_embedding="grid"
    )
    fno3 = FourierNeuralOperator(
        in_channels=1, 
        out_channels=1, 
        n_layers=3, 
        hidden_channels=hidden_channels, 
        n_modes=(32, 32),
        positional_embedding="grid"
    )
    
    # Wrap FNOs into multi-level model
    multi_fno = MultiFNO(fno1, fno2, fno3)
    ps, st = Lux.setup(rng, multi_fno) |> gdev
    
    # Rest of training setup remains similar
    dataloader = DataLoader((x, target); batchsize=1) |> gdev
    opt = Optimisers.ADAM(0.0005f0)
    lr = i -> Exp(0.001f0, 0.999f0).(i)
    train_state = Training.TrainState(multi_fno, ps, st, opt)
    
    total_loss_tracker, physics_loss_tracker, data_loss_tracker = ntuple(_ -> Lag(Float32, 32), 3)
    iter = 1
    for (x, target_data) in Iterators.cycle(dataloader)
        Optimisers.adjust!(train_state, lr(iter))
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
            GC.gc()
        end
        
        iter += 1
        
        if iter > maxiters
            break
        end
    end
    return StatefulLuxLayer{true}(multi_fno, train_state.parameters, train_state.states), loss_function
end

const gdev = gpu_device()
const cdev = cpu_device()

# Example usage
#=
function normalize_data(data)
    μ = mean(data)
    σ = std(data)
    return (data .- μ) ./ σ, μ, σ
end

function denormalize_data(data, μ, σ)
    return data .* σ .+ μ
end
=#
#import data
T = Float32

begin
        DIR = "data/"
        NAME = "ERA5-sf-t21q.nc"
        LSNAME = "land-t21.nc"
        ORONAME = "oro-t21.nc"

        LATNAME = "lat"
        LONNAME = "lon"

        lats = deg2rad.(T.(ncread(string(DIR,NAME),LATNAME)))
        lat_inds = 1:size(lats,1)

        ψ = ncread(string(DIR,NAME),"atmosphere_horizontal_streamfunction")[:,:,:,:]

        lvl = ncread(string(DIR,NAME),"level")
        lats = deg2rad.(T.(ncread(string(DIR,NAME),LATNAME)))[lat_inds]
        lons = deg2rad.(T.(ncread(string(DIR,NAME),LONNAME)))

        times = CFTime.timedecode( ncread(string(DIR,NAME),"time"),ncgetatt(string(DIR,NAME),"time","units"))

        summer_ind = [month(t) ∈ [6,7,8] for t ∈ times]
        winter_ind = [month(t) ∈ [12,1,2] for t ∈ times]

        LS = T.(permutedims(ncread(string(DIR,LSNAME),"var172")[:,:,1],[2,1]))[lat_inds,:]
        # Land see mask, on the same grid as lats and lons

        h = (T.(permutedims(ncread(string(DIR,ORONAME),"z")[:,:,1],[2,1]))[lat_inds,:] .* T.(ncgetatt(string(DIR,ORONAME), "z", "scale_factor"))) .+ T.(ncgetatt(string(DIR,ORONAME),"z","add_offset"))
        # orography, array on the same grid as lats and lons

        LEVELS = [200, 500, 800]

        ψ = togpu(ψ[:,:,level_index(LEVELS,lvl),:])
        ψ = permutedims(ψ, [3,2,1,4]) # level, lat, lon,
        ψ = T.(ψ[:,lat_inds,:,:])

        gridtype="gaussian"
end

L = 22 # T21 grid, truncate with l_max = 21

# pre-compute the model and normalize the data
qg3ppars = QG3ModelParameters(L, lats, lons, LS, h)

ψ = ψ ./ qg3ppars.ψ_unit

qg3p = CUDA.@allowscalar QG3Model(qg3ppars)
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, N_batch=size(ψ,4))
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, N_batch=100)

# stream function data in spherical domain
ψ_SH = QG3.transform_SH(ψ, ggsh)
#ψ_SH = QG3.transform_SH_data(ψ, qg3p)

# compute the forcing from winter data
S = QG3.compute_S_Roads(ψ_SH[:,:,:,winter_ind], qg3p)

# initial conditions for streamfunction and vorticity
N_sims = 1000
ψ_0 = ψ_SH[:,:,:,1:N_sims]
q_0 = QG3.ψtoqprime(qg3p, ψ_0)
ψ_evolved = ψ_SH[:,:,:,2:N_sims+1]
q_evolved = QG3.ψtoqprime(qg3p, ψ_evolved)

# Assuming q_0 is a 4D array (possibly a CuArray for GPU)
rhs = [QG3.QG3MM_gpu(q_0[:, :, :, i], (qg3p, S), (T(0), T(1))) for i in axes(q_0, 4)]
# Stack the results into a single 4D CuArray (x × y × z × batch_size)
rhs = cat(rhs...; dims=4)  # Explicitly concatenate along the 4th dimension

# Reshape to add a singleton 4th dimension (now 5D: x × y × z × 1 × batch_size)
rhs = reshape(rhs, size(q_0, 1), size(q_0, 2), size(q_0, 3), 1, size(q_0, 4))

ψ_0, q_0, ψ_evolved, q_evolved = map(x -> reshape(x, size(x,1), size(x,2), size(x,3), 1, size(x,4)), [ψ_0, q_0, ψ_evolved, q_evolved])

dt = times[2]-times[1]
tu = qg3ppars.time_unit
dt = Day(dt)
dt = dt.value / tu

seed = 0
maxiters = 3000
hidden_channels = 32
rng = Random.default_rng(seed)
    
# Create three 2D FNOs (one per atmospheric level)
fno1 = FourierNeuralOperator(
    in_channels=1, 
    out_channels=1, 
    n_layers=3, 
    hidden_channels=hidden_channels, 
    n_modes=(32, 32),  # 2D Fourier modes (lat, lon)
    positional_embedding="grid"
)
fno2 = FourierNeuralOperator(
    in_channels=1, 
    out_channels=1, 
    n_layers=3, 
    hidden_channels=hidden_channels, 
    n_modes=(32, 32),
    positional_embedding="grid"
)
fno3 = FourierNeuralOperator(
    in_channels=1, 
    out_channels=1, 
    n_layers=3, 
    hidden_channels=hidden_channels, 
    n_modes=(32, 32),
    positional_embedding="grid"
)

# Wrap FNOs into multi-level model
multi_fno = MultiFNO(fno1, fno2, fno3)
ps, st = Lux.setup(rng, multi_fno) |> gdev
q_0 = q_0 |> gdev

par = QG3_Physics_Parameters(dt, rhs)
physics_loss = create_QG3_physics_loss(par)
loss_function = select_QG3_loss_function(physics_loss)

multi_fno(q_0, ps, st)

trained_model, loss_function = train_model(q_0, q_evolved; seed=0, maxiters=3000, hidden_channels=32, loss_function=loss_function)

trained_u = Lux.testmode(StatefulLuxLayer{true}(trained_model.model, cdev(trained_model.ps), cdev(trained_model.st)))

ψ_test = ψ_SH[:,:,:,1001:1101]
q_test_pre = reshape(QG3.ψtoqprime(qg3p, ψ_test), size(ψ_test,1), size(ψ_test,2), size(ψ_test,3), 1, size(ψ_test,4))
q_test = q_test_pre[:,:,:,:,1:100]
q_test_evolved = q_test_pre[:,:,:,:,2:101]
q_test_array = Array(Float32.(q_test))
GC.gc()
q_pred = trained_u(q_test_array)
q_pred = q_pred |> gdev

sf_plot_pred = transform_grid(qprimetoψ(qg3p,reshape(q_pred, size(q_pred,1),size(q_pred,2),size(q_pred,3),size(q_pred,5)) ), shgg)
sf_plot_evolved = transform_grid(qprimetoψ(qg3p, reshape(q_test_evolved, size(q_test_evolved,1),size(q_test_evolved,2),size(q_test_evolved,3),size(q_test_evolved,5))), shgg)


err = sf_plot_pred .- sf_plot_evolved
err = Array(err)
sf_plot_pred = Array(sf_plot_pred)

using Plots
pyplot()


ilvl = 1
clims = (-1.1*maximum(abs, sf_plot_pred[ilvl,:,:,:]), 1.1*maximum(abs, sf_plot_pred[ilvl,:,:,:]))

plot_times = 1:size(q_test)[5]

anim = @animate for (iit,it) ∈ enumerate(plot_times)   
    
    Plots.heatmap(sf_plot_pred[ilvl,:,:,iit], c=:balance, title=string("time=",it,"   - ",it*qg3p.p.time_unit," d"), clims=clims)
end
gif(anim, "prediction_fps20_sf.gif", fps = 20)

clims = (-1.1*maximum(abs, err[ilvl,:,:,:]), 1.1*maximum(abs, err[ilvl,:,:,:]))

plot_times = 1:size(q_test)[5]

anim = @animate for (iit,it) ∈ enumerate(plot_times)   
    
    Plots.heatmap(err[ilvl,:,:,iit], c=:balance, title=string("time=",it,"   - ",it*qg3p.p.time_unit," d"), clims=clims)
end
gif(anim, "error_fps20_sf.gif", fps = 20)