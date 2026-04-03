function ESM_PINO.FourierNeuralOperator(qg3ppars::QG3.QG3ModelParameters;
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
    positional_embedding::String="spectral",
    grid_boundaries::Vector{Vector{Float32}}=[[0f0, 1f0], [0f0, 1f0]],
    use_norm::Bool=false,
    bias::Bool=false,
) where N
    
    spatial_dims = length(n_modes)
    
    # Create embedding layer and adjust input channels if needed
    if positional_embedding == "spectral"
        shgg = QG3.SHtoGaussianGridTransform(qg3ppars, hidden_channels, N_batch=1)
        embedding = SpectralPositionEmbedding(shgg, hidden_channels)
    elseif positional_embedding == "lsh"
        embedding = LSHEmbedding(grid_boundaries, qg3ppars)
        in_channels += 2    
    end
    
    # Build encoder (lifting) with variable depth
    encoder_hidden_dim = Int(hidden_channels * lifting_channel_ratio)
    lifting = ESM_PINO.build_encoder_decoder(
        in_channels,
        hidden_channels,
        encoder_hidden_dim,
        num_encoder_layers,
        spatial_dims,
        activation,
        bias
    )
    
    # Build decoder (projection) with variable depth
    decoder_hidden_dim = Int(hidden_channels * projection_channel_ratio)
    projection = ESM_PINO.build_encoder_decoder(
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
        [ESM_PINO.FNO_Block(
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

function (layer::FourierNeuralOperator{T,N,E})(x::AbstractArray{G,4}, ps::NamedTuple, st::NamedTuple) where {T,N,E<:ESM_PINOQG3Ext.LSHEmbedding,G}
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

function (layer::FourierNeuralOperator{T,N,E})(x::AbstractArray{G,4}, ps::NamedTuple, st::NamedTuple) where {T,N,E<:ESM_PINOQG3Ext.SpectralPositionEmbedding,G}
    residual = x
    x, st_lifting = layer.lifting(x, ps.lifting, st.lifting)
    x, st_embedding = layer.embedding(x, ps.embedding, st.embedding)
    x, st_fno_blocks = layer.fno_blocks(x, ps.fno_blocks, st.fno_blocks)
    x, st_projection = layer.projection(x, ps.projection, st.projection)
    if layer.outer_skip
        x = x + residual
    end
    return x, (embedding=st_embedding, lifting=st_lifting, fno_blocks=st_fno_blocks, projection=st_projection)
end