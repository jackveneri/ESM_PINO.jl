struct SphericalConv{T, G, S} <: Lux.AbstractLuxLayer
    pars::QG3.QG3ModelParameters{T}
    hidden_channels::Int
    ggsh::G  # Concrete GaussianGridtoSHTransform type
    shgg::S  # Concrete SHtoGaussianGridTransform type
end

function SphericalConv(pars::QG3.QG3ModelParameters{T}, hidden_channels::Int; batch_size::Int=1) where T
    ggsh = QG3.GaussianGridtoSHTransform(pars, hidden_channels; N_batch=batch_size)
    shgg = QG3.SHtoGaussianGridTransform(pars, hidden_channels; N_batch=batch_size)
    SphericalConv{T, typeof(ggsh), typeof(shgg)}(pars, hidden_channels, ggsh, shgg)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SphericalConv{T}) where T
    init_std = sqrt(T(1) / layer.hidden_channels)
    # Initialize 2D weights for spatial pattern (L × M)
    weight = init_std * randn(rng, T, 1, layer.pars.L, layer.pars.M, 1)
    return (weight=weight,)
end

Lux.initialstates(rng::AbstractRNG, layer::SphericalConv) = NamedTuple()

function (layer::SphericalConv{T})(x::AbstractArray{T,4}, ps::NamedTuple, st::NamedTuple) where T
    @views begin
    x_perm = permutedims(x, (3, 1, 2, 4))  # [channels, lat, lon, batch]
    x_tr = QG3.transform_SH(x_perm, layer.ggsh)
    # Type-stable element-wise multiplication with broadcast
    x_p = ps.weight .* x_tr[:, 1:layer.pars.L, 1:layer.pars.M, :]  
    x_pad = NNlib.pad_zeros(x_p, (0, 0, 0, size(x_tr, 2) - layer.pars.L, 0, size(x_tr, 3) - layer.pars.M, 0, 0))
    x_out = QG3.transform_grid(x_pad, layer.shgg)
    x_out_perm = permutedims(x_out, (2, 3, 1, 4)) # [lat, lon, channels, batch]
    end  
    return x_out_perm, st
end
#=
using NetCDF, CFTime, Dates

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

        #ψ = ncread(string(DIR,NAME),"atmosphere_horizontal_streamfunction")[:,:,:,:]

        #lvl = ncread(string(DIR,NAME),"level")
        lats = deg2rad.(T.(ncread(string(DIR,NAME),LATNAME)))[lat_inds]
        lons = deg2rad.(T.(ncread(string(DIR,NAME),LONNAME)))

        times = CFTime.timedecode( ncread(string(DIR,NAME),"time"),ncgetatt(string(DIR,NAME),"time","units"))

        #summer_ind = [month(t) ∈ [6,7,8] for t ∈ times]
        #winter_ind = [month(t) ∈ [12,1,2] for t ∈ times]

        LS = T.(permutedims(ncread(string(DIR,LSNAME),"var172")[:,:,1],[2,1]))[lat_inds,:]
        # Land see mask, on the same grid as lats and lons

        h = (T.(permutedims(ncread(string(DIR,ORONAME),"z")[:,:,1],[2,1]))[lat_inds,:] .* T.(ncgetatt(string(DIR,ORONAME), "z", "scale_factor"))) .+ T.(ncgetatt(string(DIR,ORONAME),"z","add_offset"))
        # orography, array on the same grid as lats and lons

        #LEVELS = [200, 500, 800]

        #ψ = togpu(ψ[:,:,level_index(LEVELS,lvl),:])
        #ψ = permutedims(ψ, [3,2,1,4]) # level, lat, lon,
        #ψ = T.(ψ[:,lat_inds,:,:])

        #gridtype="gaussian"
end

L = 22 # T21 grid, truncate with l_max = 21

# pre-compute the model 
qg3ppars = QG3ModelParameters(L, lats, lons, LS, h)
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=1)
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=1)
x = rand(Float32, 32, 64, 32, 1)
model = SphericalConv(qg3ppars, 32, batch_size=size(x,4))
gdev = gpu_device()
rnd = Random.default_rng(0)
ps, st = Lux.setup(rnd, model) |> gdev
x = x |> gdev
model(x, ps, st)

using Zygote
gr = Zygote.gradient(ps -> sum(model(x, ps, st)[1]), ps)
=#
