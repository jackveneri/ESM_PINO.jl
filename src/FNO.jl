"""
Helper function to build encoder/decoder layers with variable depth.

# Arguments
- `in_channels::Int`: Number of input channels
- `out_channels::Int`: Number of output channels  
- `hidden_channels::Int`: Number of hidden channels
- `n_layers::Int`: Number of layers (depth)
- `spatial_dims::Int`: Number of spatial dimensions
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
    spatial_dims::Int,
    activation,
    bias::Bool
)
    if n_layers < 1
        throw(ArgumentError("n_layers must be at least 1"))
    end
    
    kernel_size = ntuple(_ -> 1, spatial_dims)
    layers = []
    current_dim = in_channels
    
    # Build intermediate layers (all but the last)
    for l in 1:(n_layers - 1)
        push!(layers, Lux.Conv(
            kernel_size,
            current_dim => hidden_channels,
            activation,  
            cross_correlation=true,
            init_weight=kaiming_normal,
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
        kernel_size,
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
$(TYPEDSIGNATURES)

A Fourier Neural Operator (FNO) container that works with any spatial dimensionality.

# Arguments
- `in_channels::Int`: Number of input channels.
- `out_channels::Int`: Number of output channels.
- `hidden_channels::Int=32`: Number of hidden channels used inside FNO blocks.
- `n_modes::NTuple{N,Int}`: Number of retained Fourier modes per spatial dimension.
- `n_layers::Int=4`: Number of FNO blocks to stack.
- `num_encoder_layers::Int=2`: Number of layers in the encoder (lifting).
- `num_decoder_layers::Int=2`: Number of layers in the decoder (projection).
- `lifting_channel_ratio::Int=2`: Channel expansion ratio used in the lifting layer.
- `projection_channel_ratio::Int=2`: Channel expansion ratio used in the projection layer.
- `channel_mlp_expansion::Number=2`: Expansion factor inside ChannelMLP of each block.
- `activation=NNlib.gelu`: Activation function used in conv layers.
- `positional_embedding::Bool=true`: Whether to use GridEmbedding (default: true).
- `grid_boundaries::Vector{Vector{Float32}}`: Boundaries for each spatial dimension (default: [0,1] for each).
- `use_norm::Bool=false`: Whether to use normalization layers inside FNO blocks.
- `bias::Bool=false`: Whether to use bias in convolutions.

# Notes
- When `num_encoder_layers=2` and `num_decoder_layers=2`, this behaves like the original implementation
- The spatial dimensionality is automatically inferred from the length of `n_modes`

# Examples

Example (2D data with grid embedding):
```julia
using Lux, Random

rng = Random.default_rng()

layer = FourierNeuralOperator(
    in_channels=3,
    out_channels=2,
    hidden_channels=32,
    n_modes=(12, 12),
    n_layers=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    positional_embedding=true
)

ps = Lux.initialparameters(rng, layer)
st = Lux.initialstates(rng, layer)

# Input tensor (H, W, C, Batch)
x = randn(Float32, 64, 64, 3, 10)

y, st_new = layer(x, ps, st)
@show size(y)   # expect (64, 64, 2, 10)
```

Example (1D data without grid embedding):
```julia
layer1d = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=16,
    n_modes=(8,),
    n_layers=3,
    num_encoder_layers=1,
    num_decoder_layers=1,
    positional_embedding=false
)

x1 = randn(Float32, 128, 1, 5)   # (L, C, Batch)
y1, _ = layer1d(x1,
    Lux.initialparameters(rng, layer1d),
    Lux.initialstates(rng, layer1d)
)
@show size(y1)   # expect (128, 1, 5)
```

Example (3D data with grid embedding):
```julia
layer3d = FourierNeuralOperator(
    in_channels=2,
    out_channels=1,
    hidden_channels=24,
    n_modes=(8, 8, 8),
    n_layers=3,
    num_encoder_layers=2,
    num_decoder_layers=2,
    positional_embedding=true
)

x3 = randn(Float32, 32, 32, 32, 2, 4)   # (H, W, D, C, Batch)
y3, _ = layer3d(x3,
    Lux.initialparameters(rng, layer3d),
    Lux.initialstates(rng, layer3d)
)
@show size(y3)   # expect (32, 32, 32, 1, 4)
```
"""
struct FourierNeuralOperator{T,N,E} <: Lux.AbstractLuxContainerLayer{(:embedding, :lifting, :fno_blocks, :projection)}
    embedding::E
    lifting::Lux.AbstractLuxLayer
    fno_blocks::Lux.Chain{<:NamedTuple{<:Any,<:Tuple{Vararg{FNO_Block{T,N}}}}}
    projection::Lux.AbstractLuxLayer
    outer_skip::Bool
end

function FourierNeuralOperator(;
    in_channels::Int=3,
    out_channels::Int=3,
    hidden_channels::Int=32,
    n_modes::NTuple{N,Integer}=(16, 16),
    n_layers::Int=4,
    num_encoder_layers::Int=2,
    num_decoder_layers::Int=2,
    lifting_channel_ratio::Int=2,
    projection_channel_ratio::Int=2,
    channel_mlp_expansion::Number=2,
    activation=NNlib.gelu,
    inner_skip::Bool=true,
    outer_skip::Bool=true,
    positional_embedding::Bool=true,
    grid_boundaries::Union{Nothing,Vector{Vector{Float32}}}=nothing,
    use_norm::Bool=false,
    bias::Bool=false,
) where N
    
    spatial_dims = length(n_modes)
    
    # Create embedding layer and adjust input channels if needed
    if positional_embedding
        if isnothing(grid_boundaries)
            # Default boundaries: [0, 1] for each spatial dimension
            grid_boundaries = [Float32[0, 1] for _ in 1:spatial_dims]
        end
        embedding = GridEmbedding(grid_boundaries)
        adjusted_in_channels = in_channels + spatial_dims
    else
        embedding = Lux.NoOpLayer()
        adjusted_in_channels = in_channels
    end
    
    # Build encoder (lifting) with variable depth
    encoder_hidden_dim = Int(hidden_channels * lifting_channel_ratio)
    lifting = build_encoder_decoder(
        adjusted_in_channels,
        hidden_channels,
        encoder_hidden_dim,
        num_encoder_layers,
        spatial_dims,
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
        spatial_dims,
        activation,
        bias
    )
    
    # Create FNO blocks with optional normalization
    fno_blocks = Chain(
        [FNO_Block(
            hidden_channels, 
            n_modes; 
            expansion_factor=channel_mlp_expansion,
            skip=inner_skip, 
            activation=activation,
            use_norm=use_norm
        ) for _ in 1:n_layers]... 
    )
    
    return FourierNeuralOperator{ComplexF32,N,typeof(embedding)}(embedding, lifting, fno_blocks, projection, outer_skip)
end

function Lux.initialparameters(rng::AbstractRNG, layer::FourierNeuralOperator{T,N,E}) where {T,N,E}
    ps_embedding = Lux.initialparameters(rng, layer.embedding)
    ps_lifting = Lux.initialparameters(rng, layer.lifting)
    ps_fno_blocks = Lux.initialparameters(rng, layer.fno_blocks)
    ps_projection = Lux.initialparameters(rng, layer.projection)
    return (
        embedding=ps_embedding,
        lifting=ps_lifting,
        fno_blocks=ps_fno_blocks,
        projection=ps_projection,
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::FourierNeuralOperator{T,N,E}) where {T,N,E}
    st_embedding = Lux.initialstates(rng, layer.embedding)
    st_lifting = Lux.initialstates(rng, layer.lifting)
    st_fno_blocks = Lux.initialstates(rng, layer.fno_blocks)
    st_projection = Lux.initialstates(rng, layer.projection)
    return (
        embedding=st_embedding,
        lifting=st_lifting,
        fno_blocks=st_fno_blocks,
        projection=st_projection,
    )
end

function (layer::FourierNeuralOperator{T,N,E})(x::AbstractArray, ps::NamedTuple, st::NamedTuple) where {T,N,E}
    residual = x
    x, st_embedding = layer.embedding(x, ps.embedding, st.embedding)
    x, st_lifting = layer.lifting(x, ps.lifting, st.lifting)
    x, st_fno_blocks = layer.fno_blocks(x, ps.fno_blocks, st.fno_blocks)
    x, st_projection = layer.projection(x, ps.projection, st.projection)
    if layer.outer_skip
        x = x + residual
    end
    return x, (embedding=st_embedding, lifting=st_lifting, fno_blocks=st_fno_blocks, projection=st_projection)
end