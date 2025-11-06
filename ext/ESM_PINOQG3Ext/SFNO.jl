"""
$(TYPEDSIGNATURES)
Spherical Fourier Neural Operator (SFNO) layer combining positional embeddings, spectral kernels, and channel MLPs.

This layer implements the SFNO architecture on the sphere, optionally using Zonal Symmetric Kernels (ZSK)
following the approach described in [**Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere**](https://arxiv.org/abs/2204.06408).

# Arguments 
- `pars::QG3ModelParameters`: Model parameters defining the spherical grid and maximum spherical harmonic degree `L`.
- `batch_size::Int=1`: Number of samples in a batch.
- `modes::Int=pars.L`: Maximum number of spherical harmonic modes to use.
- `in_channels::Int`: Number of input channels.
- `out_channels::Int`: Number of output channels.
- `hidden_channels::Int=32`: Number of hidden channels in the SFNO blocks.
- `n_layers::Int=4`: Number of SFNO blocks.
- `lifting_channel_ratio::Int=2`: Expansion ratio for the lifting layer.
- `projection_channel_ratio::Int=2`: Expansion ratio for the projection layer.
- `channel_mlp_expansion::Number=2.0`: Expansion factor for channel MLPs in each SFNO block.
- `activation`: Activation function (default is `NNlib.gelu`).
- `positional_embedding::AbstractString="grid"`: Type of positional embedding. Options: `"grid"` or `"no_grid"`.
- `inner_skip::Bool=true`: If true, use skip connections inside each SFNO block.
- `outer_skip::Bool=true`: If true, apply residual connection from lifting output to projection output.
- `gpu::Bool=true`: If true, computations are performed on GPU.
- `zsk::Bool=false`: If true, use Zonal Symmetric Kernels, enforcing longitudinal symmetry.


# Returns
- `SFNO`: A Lux-compatible container layer.

# Details
- Constructs lifting, SFNO blocks, and projection layers compatible with Lux.jl.
- Positional embeddings are appended if `positional_embedding="grid"`.
- Supports both CPU and GPU execution.
- Zonal Symmetric Kernels (ZSK) reduce the number of parameters and improve stability on spherical domains.

# Example
```julia
using Lux, QG3, Random, NNlib, LuxCUDA

# Load precomputed QG3 parameters
qg3ppars = QG3.load_precomputed_params()[2]

# Input: [lat, lon, channels, batch]
x = rand(Float32, 32, 64, 3, 10)

# Construct SFNO layer using primary constructor
model1 = SFNO(qg3ppars;
    batch_size=size(x, 4),
    modes=30,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2.0,
    positional_embedding="no_grid",
    outer_skip=true,
    gpu=false
)


# Setup parameters and state
rng = Random.default_rng(0)
ps, st = Lux.setup(rng, model1)

# Forward pass
y, st = model1(x, ps, st)

# Compute gradients
using Zygote
gr = Zygote.gradient(ps -> sum(model1(x, ps, st)[1]), ps)
```
"""
function ESM_PINO.SFNO(pars::QG3.QG3ModelParameters;
    batch_size::Int=1,
    modes::Int=pars.L,
    in_channels::Int=3,
    out_channels::Int=3,
    hidden_channels::Int=32,
    n_layers::Int=4,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=2.0,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid",
    inner_skip::Bool=true,
    outer_skip::Bool=true,
    zsk::Bool=false,
    use_norm::Bool=false,
    downsampling_factor::Int=1,
    gpu::Bool=true
) 
    embedding = nothing
    if positional_embedding in ["grid","no_grid"]
        if positional_embedding == "grid" 
            embedding = GridEmbedding2D()
            in_channels += 2
        else
            embedding = Lux.NoOpLayer()
        end
        lifting = Lux.Chain(
            Lux.Conv(
                (1, 1), 
                in_channels => Int(lifting_channel_ratio * hidden_channels), 
                activation, 
                cross_correlation=true, 
                init_weight=kaiming_normal; 
                init_bias=zeros32),
            Lux.Conv(
                (1, 1), 
                Int(lifting_channel_ratio * hidden_channels) => hidden_channels, 
                identity, cross_correlation=true, 
                init_weight=kaiming_normal, 
                init_bias=zeros32),
        )
        
        projection = Lux.Chain(
            Lux.Conv(
                (1, 1), 
                hidden_channels => Int(projection_channel_ratio * hidden_channels), 
                activation,
                cross_correlation=true, 
                init_weight=kaiming_normal, 
                init_bias=zeros32),
            Lux.Conv(
                (1, 1), 
                Int(projection_channel_ratio * hidden_channels) => out_channels, 
                identity, 
                cross_correlation=true, 
                init_weight=kaiming_normal, 
                init_bias=zeros32),
        )
        new_modes = modes ÷ downsampling_factor
        if !gpu
            QG3.gpuoff()
        end 
        pars_outer_layers = qg3pars_constructor_helper(new_modes, pars.N_lats)
        pars_inner_layers = qg3pars_constructor_helper(new_modes, pars.N_lats ÷ downsampling_factor)
        ggsh_outer = QG3.GaussianGridtoSHTransform(pars_outer_layers, hidden_channels, N_batch=batch_size)
        shgg_outer = QG3.SHtoGaussianGridTransform(pars_outer_layers, hidden_channels, N_batch=batch_size)
        ggsh_inner = QG3.GaussianGridtoSHTransform(pars_inner_layers, hidden_channels, N_batch=batch_size)
        shgg_inner = QG3.SHtoGaussianGridTransform(pars_inner_layers, hidden_channels, N_batch=batch_size)
        blocks = []
        block1 = ESM_PINO.SFNO_Block(
            hidden_channels, 
            ggsh_outer, 
            shgg_inner; 
            modes=new_modes, 
            expansion_factor=channel_mlp_expansion, 
            activation=activation, 
            skip=inner_skip,
            zsk=zsk,
            use_norm=use_norm)
        push!(blocks, block1)
        for i in 2:n_layers-1
            blocki = ESM_PINO.SFNO_Block(
                hidden_channels, 
                ggsh_inner, 
                shgg_inner; 
                modes=new_modes,
                expansion_factor=channel_mlp_expansion, 
                activation=activation, 
                skip=inner_skip,  
                zsk=zsk,
                use_norm=use_norm)
            push!(blocks, blocki)
        end
        final_block = ESM_PINO.SFNO_Block(
            hidden_channels, 
            ggsh_inner, 
            shgg_outer; 
            modes=new_modes, 
            expansion_factor=channel_mlp_expansion, 
            activation=activation, 
            skip=inner_skip, 
            zsk=zsk,
            use_norm=use_norm)
        push!(blocks, final_block)
        sfno_blocks = Lux.Chain(blocks...)
        plan = ESM_PINOQG3(ggsh_outer, shgg_outer, pars_inner_layers) #dummy plan to satisfy type parameter
    else
            throw(ArgumentError("Invalid positional embedding type. Supported arguments are 'grid' and 'no_grid'."))
    end
    return ESM_PINO.SFNO(embedding, lifting, sfno_blocks, projection, plan, outer_skip, lifting_channel_ratio, projection_channel_ratio)
end
"""
$(TYPEDSIGNATURES)
Spherical Fourier Neural Operator (SFNO) layer combining positional embeddings, spectral kernels, and channel MLPs.

This layer implements the SFNO architecture on the sphere, optionally using Zonal Symmetric Kernels (ZSK)
following the approach described in [**Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere**](https://arxiv.org/abs/2204.06408).

# Arguments 
- `ggsh::QG3.GaussianGridtoSHTransform`: Precomputed grid-to-SH transform.
- `shgg::QG3.SHtoGaussianGridTransform`: Precomputed SH-to-grid transform.
- Other keyword arguments are the same as for the primary constructor, except modes which default is set to `ggsh.output_size[1]`. Also, no need to specify `batch_size` or `gpu` as these are handled in the transforms.

# Returns
- `SFNO`: A Lux-compatible container layer.

# Details
- Constructs lifting, SFNO blocks, and projection layers compatible with Lux.jl.
- Positional embeddings are appended if `positional_embedding="grid"`.
- Supports both CPU and GPU execution.
- Zonal Symmetric Kernels (ZSK) reduce the number of parameters and improve stability on spherical domains.

# Example
```julia
using Lux, QG3, Random, NNlib, LuxCUDA

# Load precomputed QG3 parameters
qg3ppars = QG3.load_precomputed_params()[2]

# Input: [lat, lon, channels, batch]
x = rand(Float32, 32, 64, 3, 10)


# Construct SFNO layer using secondary constructor
ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=size(x,4))
shgg = QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=size(x,4))
model2 = SFNO(ggsh, shgg;
    modes=15,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2.0,
    positional_embedding="no_grid",
    outer_skip=true,
    zsk=true
)

# Setup parameters and state
rng = Random.default_rng(0)
ps, st = Lux.setup(rng, model2)

# Forward pass
y, st = model2(x, ps, st)

# Compute gradients
using Zygote
gr = Zygote.gradient(ps -> sum(model2(x, ps, st)[1]), ps)
```
"""
function ESM_PINO.SFNO( #could become useless if I don't find a way to handle downsampling
    ggsh::QG3.GaussianGridtoSHTransform,
    shgg::QG3.SHtoGaussianGridTransform;
    modes::Int=ggsh.output_size[1],
    in_channels::Int=3,
    out_channels::Int=3,
    hidden_channels::Int=32,
    n_layers::Int=4,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=2.0,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid",
    inner_skip::Bool=true,
    outer_skip::Bool=true,
    zsk::Bool=false,
    use_norm::Bool=false,
    downsampling_factor::Int=1,
    gpu::Bool=true,
    batch_size::Int=ggsh.FT_4d.plan.input_size[4]
)
   embedding = nothing
    if positional_embedding in ["grid","no_grid"]
        if positional_embedding == "grid" 
            embedding = GridEmbedding2D()
            in_channels += 2
        else
            embedding = Lux.NoOpLayer()
        end
         lifting = Lux.Chain(
            Lux.Conv(
                (1, 1), 
                in_channels => Int(lifting_channel_ratio * hidden_channels), 
                activation, 
                cross_correlation=true, 
                init_weight=kaiming_normal, 
                init_bias=zeros32),
            Lux.Conv(
                (1, 1), 
                Int(lifting_channel_ratio * hidden_channels) => hidden_channels, 
                identity, 
                cross_correlation=true, 
                init_weight=kaiming_normal, 
                init_bias=zeros32),
        )
        
        projection = Lux.Chain(
            Lux.Conv(
                (1, 1), 
                hidden_channels => Int(projection_channel_ratio * hidden_channels), 
                activation, 
                cross_correlation=true, 
                init_weight=kaiming_normal, 
                init_bias=zeros32),
            Lux.Conv(
                (1, 1), 
                Int(projection_channel_ratio * hidden_channels) => out_channels, 
                identity, 
                cross_correlation=true, 
                init_weight=kaiming_normal, 
                init_bias=zeros32),
        )
        QG3.gpuoff()
        safe_modes = ggsh.output_size[1]
        N_lats = shgg.output_size[1]
        # Correct modes if necessary
        corrected_modes = 0
        if modes == 0
            corrected_modes = safe_modes
        else
            corrected_modes = min(modes, safe_modes)
        end
        if modes > corrected_modes
            @warn "modes ($modes) exceeds safe_modes ($(safe_modes)). Setting modes = $(safe_modes)."
        end
        new_modes = corrected_modes ÷ downsampling_factor 
        if gpu
            QG3.gpuon()
        end
        pars_outer_layers = qg3pars_constructor_helper(new_modes, N_lats)
        pars_inner_layers = qg3pars_constructor_helper(new_modes, N_lats ÷ downsampling_factor)
        ggsh_outer = QG3.GaussianGridtoSHTransform(pars_outer_layers, hidden_channels, N_batch=batch_size)
        shgg_outer = QG3.SHtoGaussianGridTransform(pars_outer_layers, hidden_channels, N_batch=batch_size)
        ggsh_inner = QG3.GaussianGridtoSHTransform(pars_inner_layers, hidden_channels, N_batch=batch_size)
        shgg_inner = QG3.SHtoGaussianGridTransform(pars_inner_layers, hidden_channels, N_batch=batch_size)
        blocks = []
        block1 = ESM_PINO.SFNO_Block(
            hidden_channels, 
            ggsh_outer, 
            shgg_inner; 
            modes=new_modes, 
            expansion_factor=channel_mlp_expansion, 
            activation=activation, 
            skip=inner_skip,
            zsk=zsk,
            use_norm=use_norm)
        push!(blocks, block1)
        for i in 2:n_layers-1
            blocki = ESM_PINO.SFNO_Block(
                hidden_channels, 
                ggsh_inner, 
                shgg_inner; 
                modes=new_modes,
                expansion_factor=channel_mlp_expansion, 
                activation=activation, 
                skip=inner_skip,  
                zsk=zsk,
                use_norm=use_norm)
            push!(blocks, blocki)
        end
        final_block = ESM_PINO.SFNO_Block(
            hidden_channels, 
            ggsh_inner, 
            shgg_outer; 
            modes=new_modes, 
            expansion_factor=channel_mlp_expansion, 
            activation=activation, 
            skip=inner_skip, 
            zsk=zsk,
            use_norm=use_norm)
        push!(blocks, final_block)
        sfno_blocks = Lux.Chain(blocks...)
        plan = ESM_PINOQG3(ggsh, shgg, pars_inner_layers) #dummy plan to satisfy type parameter    
    else
            throw(ArgumentError("Invalid positional embedding type. Supported arguments are 'grid' and 'no_grid'."))
    end
    return SFNO(embedding, lifting, sfno_blocks, projection, plan, outer_skip, lifting_channel_ratio, projection_channel_ratio) 
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::ESM_PINO.SFNO{E, L, B, P, ESM_PINOQG3}) where {E, L, B, P} 
    ps_embedding = isnothing(layer.embedding) ? NamedTuple() : Lux.initialparameters(rng, layer.embedding)
    ps_lifting = Lux.initialparameters(rng, layer.lifting)
    ps_sfno_blocks = Lux.initialparameters(rng, layer.sfno_blocks)
    ps_projection = Lux.initialparameters(rng, layer.projection)
    return (
        embedding=ps_embedding,
        lifting=ps_lifting,
        sfno_blocks=ps_sfno_blocks,
        projection=ps_projection,
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::ESM_PINO.SFNO{E, L, B, P, ESM_PINOQG3}) where {E, L, B, P} 
    st_embedding = isnothing(layer.embedding) ? NamedTuple() : Lux.initialstates(rng, layer.embedding)
    st_lifting = Lux.initialstates(rng, layer.lifting)
    st_sfno_blocks = Lux.initialstates(rng, layer.sfno_blocks)
    st_projection = Lux.initialstates(rng, layer.projection)
    return (
        embedding=st_embedding,
        lifting=st_lifting,
        sfno_blocks=st_sfno_blocks,
        projection=st_projection,
    )
end

function (layer::ESM_PINO.SFNO{E, L, B, P, ESM_PINOQG3})(x::AbstractArray, ps::NamedTuple, st::NamedTuple) where {E, L, B, P} 
    if !isnothing(layer.embedding)
        x, st_embedding = layer.embedding(x, ps.embedding, st.embedding)
    else
        st_embedding = st.embedding
    end
    
    x, st_lifting = layer.lifting(x, ps.lifting, st.lifting)
    residual = x
    x, st_sfno_blocks = layer.sfno_blocks(x, ps.sfno_blocks, st.sfno_blocks)
    if layer.outer_skip
        x_out = x + residual
    else
        x_out = x
    end
    x, st_projection = layer.projection(x_out, ps.projection, st.projection)
    
    return x, (embedding=st_embedding, lifting=st_lifting, sfno_blocks=st_sfno_blocks, projection=st_projection)
end
function Lux.apply(layer::ESM_PINO.SFNO{E, L, B, P, ESM_PINOQG3}, x::AbstractArray, ps::NamedTuple, st::NamedTuple) where {E, L, B, P} 
    if !isnothing(layer.embedding)
        x, st_embedding = layer.embedding(x, ps.embedding, st.embedding)
    else
        st_embedding = st.embedding
    end
    
    x, st_lifting = layer.lifting(x, ps.lifting, st.lifting)
    residual = x
    x, st_sfno_blocks = layer.sfno_blocks(x, ps.sfno_blocks, st.sfno_blocks)
    if layer.outer_skip
        x_out = x + residual
    else
        x_out = x
    end
    x, st_projection = layer.projection(x_out, ps.projection, st.projection)
    
    return x, (embedding=st_embedding, lifting=st_lifting, sfno_blocks=st_sfno_blocks, projection=st_projection)
end
#=
using JLD2, Lux, Random, QG3, NNlib, LuxCUDA, ChainRulesCore
@load string(dirname(@__DIR__), "/data/t21-precomputed-p.jld2") qg3ppars
#pre-compute the model 
qg3ppars = qg3ppars
x = rand(Float32, 32, 64, 3, 10)
model = SFNO(qg3ppars,
    batch_size=size(x, 4),
    modes = 30,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2,
    activation=NNlib.gelu,
    positional_embedding="no_grid",
)
model = SFNO(QG3.GaussianGridtoSHTransform(qg3ppars, 32, N_batch=size(x, 4)), QG3.SHtoGaussianGridTransform(qg3ppars, 32, N_batch=size(x, 4)),
    modes = 15,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2,
    activation=NNlib.gelu,
    positional_embedding="no_grid",
)
gdev = gpu_device()
rnd = Random.default_rng(0)
ps, st = Lux.setup(rnd, model) |> gdev
x = x |> gdev
model(x, ps, st)[1]

using Zygote
grad = Zygote.gradient(ps -> sum(model(x, ps, st)[1]), ps)

@load string(dirname(@__DIR__), "/data/t42-precomputed-p.jld2") qg3ppars
qg3ppars = qg3ppars
model = SFNO(qg3ppars,
    batch_size=size(x, 4),
    modes = model.sfno_blocks.model.spherical_kernel.spherical_conv.modes,
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2,
    activation=NNlib.gelu,
    positional_embedding="no_grid",
)
x = rand(Float32, 64, 128, 3, 10) |> gdev
model(x, ps, st)
=#