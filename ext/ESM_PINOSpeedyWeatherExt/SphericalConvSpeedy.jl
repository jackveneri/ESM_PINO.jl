function SphericalConv(resolution::Int, hidden_channels::Int;
    NF::Type{<:AbstractFloat}=Float64,
    grid_res::Union{AbstractString,Nothing}=nothing,
    Grid=FullGaussianGrid,
    dealiasing=2,
    nlat_half = round(Int, dealiasing*(resolution+1)*3/8), 
    nlayers=hidden_channels,
    zsk::Bool=false
    )
    spectral_grid = zeros(LowerTriangularMatrix{Complex{NF}}, resolution+1, resolution+1)
    if !isnothing(grid_res)
        nlat_half = Int(gaussian_resolution_to_grid(grid_res)[1]/2)
    end
    S = SpectralTransform(spectral_grid, Grid=Grid, nlat_half=nlat_half, dealiasing=dealiasing, nlayers=nlayers)
    plan = ESM_PINOSpeedy(S, NF)
    return ESM_PINO.SphericalConv(hidden_channels, resolution, plan, zsk)
end

function Lux.initialparameters(rng::AbstractRNG, layer::ESM_PINO.SphericalConv{ESM_PINOSpeedy})
    init_std = layer.plan.NF(sqrt(1 / layer.hidden_channels))
    if layer.zsk == true
        weight = init_std * (randn(LowerTriangularArray{Complex{layer.plan.NF}}, layer.resolution + 1, 1, 1)) 
    else
        weight = init_std * (randn(LowerTriangularArray{Complex{layer.plan.NF}}, layer.resolution + 1, layer.resolution + 1, 1)) 
    end
    return (weight=weight,)
end

function Lux.initialstates(rng::AbstractRNG, layer::ESM_PINO.SphericalConv{ESM_PINOSpeedy})
    return NamedTuple()  # No internal state needed
end

function (layer::ESM_PINO.SphericalConv{ESM_PINOSpeedy})(x::AbstractArray, ps::NamedTuple, st::NamedTuple) 
    x_t = layer.plan.NF(x)
    lat, lon, lev, batch_size = size(x_t)

    x_res = reshape(x_t, (lat, lon, :))

    x_field = FullGaussianGrid(x_res, input_as=Matrix)
    x_tr = SpeedyTransforms.transform(x_field, layer.plan.spectral_transform)
    #x_tr = Array(x_tr)<
    x_p = x_tr .* ps.weight
    #x_p = LowerTriangularArray(x_p)
    x_out_field = SpeedyTransforms.transform(x_p, layer.plan.spectral_transform)
    
    x_out_data = map(i -> Matrix(x_out_field[:,i]), 1:size(x_out_field,2))
    x_out = reshape(cat(x_out_data..., dims=3), lat, lon, lev, batch_size)
    return  x_out, st
end
#=
T = Float32
levels = 1
x = rand(T, 32, 64, levels, 1)
y = x.^2  
model = SphericalConv(21, levels, NF=T, grid_res="T21")
rng = Random.default_rng(0)
ps, st = Lux.setup(rng, model)
y_new = model(x, ps, st)[1]
#using Lux.Training: compute_gradients, TrainState
#ts = TrainState(model, ps, st, Adam(0.001f0))
#grads, loss, stats, ts = compute_gradients(AutoEnzyme(), MSELoss(), (x, y), ts)


dataloader = DeviceIterator(cpu_device, zip(x, y)) 
train_state = Training.TrainState(model, ps, st, Adam(0.001f0))
function train_model(model, ps, st, dataloader)
        train_state = Training.TrainState(model, ps, st, Adam(0.001f0))

        for iteration in 1:1000
            for (i, (xᵢ, yᵢ)) in enumerate(dataloader)
                println(typeof(xᵢ))
                println(eltype(xᵢ))
                _, loss, _, train_state = Training.single_train_step!(
                    AutoEnzyme(), MSELoss(), (xᵢ, yᵢ), train_state)
                if (iteration % 100 == 0 || iteration == 1) && i == 1
                    @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, 1000, loss)
                end
            end
        end

        return train_state
end

for iteration in 1:1000
    for (i, (xᵢ, yᵢ)) in enumerate(dataloader)
        println(typeof(xᵢ))
        println(eltype(xᵢ))
        _, loss, _, train_state = Training.single_train_step!(
            AutoEnzyme(), MSELoss(), (xᵢ, yᵢ), train_state)
        if (iteration % 100 == 0 || iteration == 1) && i == 1
            @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, 1000, loss)
        end
    end
end
resolution = 21
hidden_channels = 32
Grid=FullGaussianGrid
dealiasing=2
nlat_half = 16
nlayers=hidden_channels
spectral_grid = zeros(LowerTriangularMatrix{Complex{T}}, resolution+1, resolution+1)
S = SpectralTransform(spectral_grid, Grid=Grid, nlat_half=nlat_half, dealiasing=dealiasing, nlayers=nlayers)
x = T.(x)
lat, lon, lev, batch = size(x)

x_res = reshape(x, (lat, lon, :))

x_field = FullGaussianField(x_res, input_as=Matrix)
x_tr = SpeedyTransforms.transform(x_field, S)
x_tr = Array(x_tr)
x_p = x_tr .* ps.weight
x_p = LowerTriangularArray(x_p)
x_out_field = SpeedyTransforms.transform(x_p, S)

x_out_data = map(i -> Matrix(x_out_field[:,i]), 1:size(x_out_field,2))
x_out = reshape(cat(x_out_data..., dims=3), lat, lon, lev, batch)
using Statistics
test_f = (x, ps) -> begin
    lat, lon, lev, batch = size(x)

    x_res = reshape(x, (lat, lon, :))

    x_field = FullGaussianField(x_res, input_as=Matrix)
    x_tr = SpeedyTransforms.transform(x_field, S)
    x_tr = Array(x_tr)
    x_p = x_tr .* ps
    x_p = LowerTriangularArray(x_p)
    x_out_field = SpeedyTransforms.transform(x_p, S)

    x_out_data = map(i -> Matrix(x_out_field[:,i]), 1:size(x_out_field,2))
    x_out = reshape(cat(x_out_data..., dims=3), lat, lon, lev, batch)
    return mean(abs2, x_out)
end
ps = randn(T, (resolution + 1, resolution + 1, 1))
test_f2(ps) = test_f(x, ps)
test_f2(ps)
dx = zeros(eltype(ps),size(ps))
autodiff(ReverseWithPrimal, test_f2, Active, Duplicated(ps, dx))
=#