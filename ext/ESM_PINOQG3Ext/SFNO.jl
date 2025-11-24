"""
Helper function to build encoder/decoder layers with variable depth.

# Arguments
- `in_channels::Int`: Number of input channels
- `out_channels::Int`: Number of output channels  
- `hidden_channels::Int`: Number of hidden channels
- `n_layers::Int`: Number of layers (depth)
- `activation`: Activation function
- `bias::Bool`: Whether to use bias in convolutions

# Returns
- `Lux.Chain`: Sequential chain of convolutional layers
"""
function build_encoder_decoder(
    in_channels::Int,
    out_channels::Int,
    hidden_channels::Int,
    n_layers::Int,
    activation,
    bias::Bool
)
    if n_layers < 1
        throw(ArgumentError("n_layers must be at least 1"))
    end
    
    layers = []
    current_dim = in_channels
    
    # Build intermediate layers (all but the last)
    for l in 1:(n_layers - 1)
        # Conv layer with kaiming normal initialization (scale = sqrt(2/fan_in))
        push!(layers, Lux.Conv(
            (1, 1),
            current_dim => hidden_channels,
            activation,  
            cross_correlation=true,
            init_weight=kaiming_normal,
            #use_bias=bias,
            init_bias=zeros32
        ))
        current_dim = hidden_channels
    end
    
    # Final layer with different initialization (scale = sqrt(1/fan_in))
    final_init_weight = (rng, dims...) -> begin
        scale = sqrt(1.0f0 / current_dim)
        randn(rng, Float32, dims...) .* scale
    end
    
    push!(layers, Lux.Conv(
        (1, 1),
        current_dim => out_channels,
        identity,
        cross_correlation=true,
        init_weight=final_init_weight,
        use_bias=bias,
        init_bias=zeros32
    ))
    
    return Lux.Chain(layers...)
end

"""
Modified SFNO constructor with variable encoder/decoder depths.

# New Arguments
- `num_encoder_layers::Int=2`: Number of layers in the encoder (lifting)
- `num_decoder_layers::Int=2`: Number of layers in the decoder (projection)

# Notes
- When `num_encoder_layers=2` and `num_decoder_layers=2`, this behaves identically to the original implementation
- `lifting_channel_ratio` controls the hidden dimension ratio for the encoder
- `projection_channel_ratio` controls the hidden dimension ratio for the decoder
"""
function ESM_PINO.SFNO(pars::QG3.QG3ModelParameters;
    batch_size::Int=1,
    modes::Int=pars.L,
    in_channels::Int=3,
    out_channels::Int=3,
    hidden_channels::Int=32,
    n_layers::Int=4,
    num_encoder_layers::Int=1,
    num_decoder_layers::Int=1,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=2.0,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid",
    inner_skip::Bool=true,
    outer_skip::Bool=true,
    operator_type::Symbol=:driscoll_healy,
    use_norm::Bool=false,
    downsampling_factor::Int=1,
    gpu::Bool=true,
    soft_gating::Bool=false,
    bias::Bool=false
)
    if n_layers < 2
        throw(ArgumentError("n_layers must be at least 2"))
    end
    # Setup positional embedding
    if positional_embedding in ["grid", "no_grid", "gaussian_grid"]
        if positional_embedding == "grid"
            embedding = GridEmbedding2D()
            in_channels += 2
        elseif positional_embedding == "gaussian_grid"
            embedding = GaussianGridEmbedding2D()
            in_channels += 2
        else
            embedding = Lux.NoOpLayer()
        end
        
        # Build encoder (lifting) with variable depth
        encoder_hidden_dim = Int(hidden_channels * lifting_channel_ratio)
        lifting = build_encoder_decoder(
            in_channels,
            hidden_channels,
            encoder_hidden_dim,
            num_encoder_layers,
            activation,
            bias
        )
        
        # Build decoder (projection) with variable depth
        decoder_hidden_dim = Int(hidden_channels * projection_channel_ratio)
        projection = build_encoder_decoder(
            hidden_channels,
            out_channels,
            decoder_hidden_dim,
            num_decoder_layers,
            activation,
            bias
        )
        
        # Setup modes and transforms
        safe_modes = pars.N_lats ÷ downsampling_factor
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
        if !gpu
            QG3.gpuoff()
        end
        
        if downsampling_factor > 1
            pars_outer_layers = qg3pars_constructor_helper(corrected_modes, pars)
            pars_inner_layers = qg3pars_constructor_helper(corrected_modes, pars.N_lats ÷ downsampling_factor)
        else
            pars_outer_layers = pars_inner_layers = pars
        end
        
        ggsh_outer = QG3.GaussianGridtoSHTransform(pars_outer_layers, hidden_channels, N_batch=batch_size)
        shgg_outer = QG3.SHtoGaussianGridTransform(pars_outer_layers, hidden_channels, N_batch=batch_size)
        ggsh_inner = QG3.GaussianGridtoSHTransform(pars_inner_layers, hidden_channels, N_batch=batch_size)
        shgg_inner = QG3.SHtoGaussianGridTransform(pars_inner_layers, hidden_channels, N_batch=batch_size)
        
        # Build SFNO blocks
        blocks = []
        block1 = ESM_PINO.SFNO_Block(
            hidden_channels,
            ggsh_outer,
            shgg_inner;
            modes=corrected_modes,
            expansion_factor=channel_mlp_expansion,
            activation=activation,
            skip=inner_skip,
            operator_type=operator_type,
            use_norm=use_norm,
            soft_gating=soft_gating,
            bias=bias
        )
        push!(blocks, block1)
        
        for i in 2:n_layers-1
            blocki = ESM_PINO.SFNO_Block(
                hidden_channels,
                ggsh_inner,
                shgg_inner;
                modes=corrected_modes,
                expansion_factor=channel_mlp_expansion,
                activation=activation,
                skip=inner_skip,
                operator_type=operator_type,
                use_norm=use_norm,
                soft_gating=soft_gating,
                bias=bias
            )
            push!(blocks, blocki)
        end
        
        final_block = ESM_PINO.SFNO_Block(
            hidden_channels,
            ggsh_inner,
            shgg_outer;
            modes=corrected_modes,
            expansion_factor=channel_mlp_expansion,
            activation=activation,
            skip=inner_skip,
            operator_type=operator_type,
            use_norm=use_norm,
            soft_gating=soft_gating,
            bias=bias
        )
        push!(blocks, final_block)
        
        sfno_blocks = Lux.Chain(blocks...)
        plan = ESM_PINOQG3(ggsh_outer, shgg_outer)
    else
        throw(ArgumentError("Invalid positional embedding type. Supported arguments are 'grid', 'gaussian_grid' and 'no_grid'."))
    end
    
    return ESM_PINO.SFNO(
        embedding,
        lifting,
        sfno_blocks,
        projection,
        plan,
        outer_skip,
        lifting_channel_ratio,
        projection_channel_ratio
    )
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
function ESM_PINO.SFNO( 
    ggsh::QG3.GaussianGridtoSHTransform,
    shgg::QG3.SHtoGaussianGridTransform;
    in_channels::Int=3,
    out_channels::Int=3,
    hidden_channels::Int=32,
    n_layers::Int=4,
    num_encoder_layers::Int=1,
    num_decoder_layers::Int=1,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=2.0,
    activation=NNlib.gelu,
    positional_embedding::AbstractString="grid",
    inner_skip::Bool=true,
    outer_skip::Bool=true,
    operator_type::Symbol = :driscoll_healy, # New argument to specify operator type
    use_norm::Bool=false,
    downsampling_factor::Int=1,
    modes::Int=get_truncation_from_nlat(shgg.output_size[1]),
    gpu::Bool=true,
    batch_size::Int=typeof(ggsh).parameters[end] ? ggsh.FT_4d.plan.input_size[4] : ggsh.FT_4d.plan.sz[4],
    soft_gating::Bool=false,
    bias::Bool=false
)
        if n_layers < 2
        throw(ArgumentError("n_layers must be at least 2"))
    end
    # Setup positional embedding
    if positional_embedding in ["grid", "no_grid", "gaussian_grid"]
        if positional_embedding == "grid"
            embedding = GridEmbedding2D()
            in_channels += 2
        elseif positional_embedding == "gaussian_grid"
            embedding = GaussianGridEmbedding2D()
            in_channels += 2
        else
            embedding = Lux.NoOpLayer()
        end
        
        # Build encoder (lifting) with variable depth
        encoder_hidden_dim = Int(hidden_channels * lifting_channel_ratio)
        lifting = build_encoder_decoder(
            in_channels,
            hidden_channels,
            encoder_hidden_dim,
            num_encoder_layers,
            activation,
            bias
        )
        
        # Build decoder (projection) with variable depth
        decoder_hidden_dim = Int(hidden_channels * projection_channel_ratio)
        projection = build_encoder_decoder(
            hidden_channels,
            out_channels,
            decoder_hidden_dim,
            num_decoder_layers,
            activation,
            bias
        )
        N_lats = shgg.output_size[1]
        safe_modes = N_lats ÷ downsampling_factor
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
        if gpu
            QG3.gpuon()
        else
            QG3.gpuoff()
        end
        pars_outer_layers = qg3pars_constructor_helper(corrected_modes, N_lats)
        pars_inner_layers = qg3pars_constructor_helper(corrected_modes, N_lats ÷ downsampling_factor)
        ggsh_outer = QG3.GaussianGridtoSHTransform(pars_outer_layers, hidden_channels, N_batch=batch_size)
        shgg_outer = QG3.SHtoGaussianGridTransform(pars_outer_layers, hidden_channels, N_batch=batch_size)
        ggsh_inner = QG3.GaussianGridtoSHTransform(pars_inner_layers, hidden_channels, N_batch=batch_size)
        shgg_inner = QG3.SHtoGaussianGridTransform(pars_inner_layers, hidden_channels, N_batch=batch_size)
        blocks = []
        block1 = ESM_PINO.SFNO_Block(
            hidden_channels, 
            ggsh_outer, 
            shgg_inner; 
            modes=corrected_modes, 
            expansion_factor=channel_mlp_expansion, 
            activation=activation, 
            skip=inner_skip,
            operator_type=operator_type,
            use_norm=use_norm,
            soft_gating=soft_gating,
            bias=bias)
        push!(blocks, block1)
        for i in 2:n_layers-1
            blocki = ESM_PINO.SFNO_Block(
                hidden_channels, 
                ggsh_inner, 
                shgg_inner; 
                modes=corrected_modes,
                expansion_factor=channel_mlp_expansion, 
                activation=activation, 
                skip=inner_skip,  
                operator_type=operator_type,
                use_norm=use_norm,
                soft_gating=soft_gating,
                bias=bias)
            push!(blocks, blocki)
        end
        final_block = ESM_PINO.SFNO_Block(
            hidden_channels, 
            ggsh_inner, 
            shgg_outer; 
            modes=corrected_modes, 
            expansion_factor=channel_mlp_expansion, 
            activation=activation, 
            skip=inner_skip, 
            operator_type=operator_type,
            use_norm=use_norm,
            soft_gating=soft_gating,
            bias=bias)
        push!(blocks, final_block)
        sfno_blocks = Lux.Chain(blocks...)
        plan = ESM_PINOQG3(ggsh, shgg) #dummy plan to satisfy type parameter    
    else
            throw(ArgumentError("Invalid positional embedding type. Supported arguments are 'grid', 'gaussian_grid' and 'no_grid'."))
    end
    return SFNO(embedding, lifting, sfno_blocks, projection, plan, outer_skip, lifting_channel_ratio, projection_channel_ratio) 
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::ESM_PINO.SFNO{E, L, B, P, ESM_PINOQG3}) where {E, L, B, P} 
    ps_embedding = Lux.initialparameters(rng, layer.embedding)
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
    st_embedding = Lux.initialstates(rng, layer.embedding)
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
    residual = x
    x, st_embedding = layer.embedding(x, ps.embedding, st.embedding)
    x, st_lifting = layer.lifting(x, ps.lifting, st.lifting)
    x, st_sfno_blocks = layer.sfno_blocks(x, ps.sfno_blocks, st.sfno_blocks)
    
    x, st_projection = layer.projection(x, ps.projection, st.projection)
    if layer.outer_skip
        x = x + residual
    else
        x = x
    end
    return x, (embedding=st_embedding, lifting=st_lifting, sfno_blocks=st_sfno_blocks, projection=st_projection)
end
function Lux.apply(layer::ESM_PINO.SFNO{E, L, B, P, ESM_PINOQG3}, x::AbstractArray, ps::NamedTuple, st::NamedTuple) where {E, L, B, P} 
    residual = x
    x, st_embedding = layer.embedding(x, ps.embedding, st.embedding)
    x, st_lifting = layer.lifting(x, ps.lifting, st.lifting)
    x, st_sfno_blocks = layer.sfno_blocks(x, ps.sfno_blocks, st.sfno_blocks)
    
    x, st_projection = layer.projection(x, ps.projection, st.projection)
    if layer.outer_skip
        x = x + residual
    else
        x = x
    end
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