"""
    reorderQG3_indexes(A::AbstractMatrix)

Reorder a 2D array by applying circular shifts to each column.
Column j gets shifted by j÷2 positions downward.

GPU-compatible and Zygote-differentiable.
"""
function reorderQG3_indexes(A::AbstractMatrix{T}) where T
    m, n = size(A)
    l = m - 1
    
    # Compute shift amount for each column: j÷2
    shifts = (1:n) .÷ 2
    
    # Create a gathering index matrix
    # For each position (row, col), determine which source row to read from
    rows = 1:m
    # source_row[r, c] = mod1(r - shifts[c], m)
    source_rows = mod1.(rows .- shifts', m)
    
    # Create column indices matrix
    cols = 1:n
    col_matrix = repeat(cols', m, 1)
    
    # Convert to linear indices for gathering
    linear_indices = source_rows .+ (col_matrix .- 1) .* m
    
    # Gather values using linear indexing
    return reshape(A[linear_indices], m, n)
end

"""
    inverse_reorderQG3_indexes(A::AbstractMatrix)

Inverse of reorderQG3_indexes - shift each column upward by j÷2.

GPU-compatible and Zygote-differentiable.
"""
function inverse_reorderQG3_indexes(A::AbstractMatrix{T}) where T
    m, n = size(A)
    l = m - 1
    
    # Shift upward: use negative shifts
    shifts = (1:n) .÷ 2
    
    rows = 1:m
    # For inverse, we shift in opposite direction
    source_rows = mod1.(rows .+ shifts', m)
    
    cols = 1:n
    col_matrix = repeat(cols', m, 1)
    
    linear_indices = source_rows .+ (col_matrix .- 1) .* m
    
    return reshape(A[linear_indices], m, n)
end

"""
    reorderQG3_indexes_4d(A::AbstractArray{T,4})

Reorder a 4D array by applying the 2D reordering to dimensions 2 and 3.
Dimensions 1 and 4 are preserved.

GPU-compatible and Zygote-differentiable.
"""
function reorderQG3_indexes_4d(A::AbstractArray{T,4}) where T
    i, m, n, b = size(A)
    
    # Compute shifts for each column
    shifts = (1:n) .÷ 2
    
    # Create source row indices
    # Shape: (m, n) - for each position (row, col), which source row to read from
    rows = 1:m
    source_rows = mod1.(rows .- shifts', m)
    
    # Reshape for broadcasting over 4D array
    # Need shape (1, m, n, 1) to broadcast over dimensions 1 and 4
    source_rows_4d = reshape(source_rows, 1, m, n, 1)
    
    # Create index arrays for all dimensions
    i_idx = reshape(1:i, i, 1, 1, 1)
    n_idx = reshape(1:n, 1, 1, n, 1)
    b_idx = reshape(1:b, 1, 1, 1, b)
    
    # Broadcast gather indices - ensure everything stays Int
    i_indices = Int.(i_idx .+ zeros(Int, 1, m, n, b))
    row_indices = Int.(source_rows_4d .+ zeros(Int, i, 1, 1, b))
    col_indices = Int.(n_idx .+ zeros(Int, i, m, 1, b))
    b_indices = Int.(b_idx .+ zeros(Int, i, m, n, 1))
    
    # Create linear indices for the entire 4D array
    # Linear index formula: i + (row-1)*I + (col-1)*I*M + (b-1)*I*M*N
    linear_indices = @. i_indices + 
                        (row_indices - 1) * i +
                        (col_indices - 1) * (i * m) +
                        (b_indices - 1) * (i * m * n)
    
    return reshape(A[linear_indices], i, m, n, b)
end

"""
    inverse_reorderQG3_indexes_4d(A::AbstractArray{T,4})

Inverse of reorderQG3_indexes_4d.

GPU-compatible and Zygote-differentiable.
"""
function inverse_reorderQG3_indexes_4d(A::AbstractArray{T,4}) where T
    i, m, n, b = size(A)
    
    # Compute shifts for each column (inverse direction)
    shifts = (1:n) .÷ 2
    
    # Create source row indices (shifted in opposite direction)
    rows = 1:m
    source_rows = mod1.(rows .+ shifts', m)
    
    # Reshape for broadcasting
    source_rows_4d = reshape(source_rows, 1, m, n, 1)
    
    # Create index arrays
    i_idx = reshape(1:i, i, 1, 1, 1)
    n_idx = reshape(1:n, 1, 1, n, 1)
    b_idx = reshape(1:b, 1, 1, 1, b)
    
    # Broadcast gather indices - ensure everything stays Int
    i_indices = Int.(i_idx .+ zeros(Int, 1, m, n, b))
    row_indices = Int.(source_rows_4d .+ zeros(Int, i, 1, 1, b))
    col_indices = Int.(n_idx .+ zeros(Int, i, m, 1, b))
    b_indices = Int.(b_idx .+ zeros(Int, i, m, n, 1))
    
    # Create linear indices
    linear_indices = @. i_indices + 
                        (row_indices - 1) * i +
                        (col_indices - 1) * (i * m) +
                        (b_indices - 1) * (i * m * n)
    
    return reshape(A[linear_indices], i, m, n, b)
end

function ESM_PINO.SphericalConv(
        hidden_channels::Int,
        ggsh::QG3.GaussianGridtoSHTransform,
        shgg::QG3.SHtoGaussianGridTransform,
        modes::Int = 0;
        operator_type::Symbol = :driscoll_healy,  # Changed from zsk::Bool
        gain::Real=2.0
        )
    # Validate operator_type
    valid_operators = [:diagonal, :block_diagonal, :driscoll_healy]
    if !(operator_type in valid_operators)
        throw(ArgumentError("operator_type must be one of $valid_operators, got :$operator_type"))
    end
    
    if typeof(ggsh).parameters[end] == true
        safe_modes = min(shgg.output_size[1], size(ggsh.Pw,1))  
    else
        safe_modes = min(size(ggsh.Pw,2),size(shgg.P,2))
    end
    
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
    plan = ESM_PINOQG3(ggsh, shgg)
    gpu_columns = [1]
    for k in 1:corrected_modes-1
        push!(gpu_columns, [ggsh.FT_4d.i_imag+k+1, k+1]...)
    end
    # Store in layer
    ESM_PINO.SphericalConv{ESM_PINOQG3}(hidden_channels, corrected_modes, plan, operator_type, gpu_columns, gain)
end

function ESM_PINO.SphericalConv(
        pars::QG3.QG3ModelParameters{T},
        hidden_channels::Int;
        modes::Int = pars.L,
        batch_size::Int = 1,
        gpu::Bool = true,
        operator_type::Symbol = :driscoll_healy,  # Changed from zsk::Bool
        gain::Real=2.0 
        ) where T
    # Validate operator_type
    valid_operators = [:diagonal, :block_diagonal, :driscoll_healy]
    if !(operator_type in valid_operators)
        throw(ArgumentError("operator_type must be one of $valid_operators, got :$operator_type"))
    end
    
    # Correct modes upfront
    if gpu
        corrected_modes = min(modes, pars.N_lats)
    else
        corrected_modes = min(modes, pars.L)
    end
    if modes != corrected_modes
        @warn "modes ($modes) exceeds N_lats ($(pars.N_lats)). Setting modes = $(pars.N_lats)."
    end
    
    # GPU setup
    if gpu
        QG3.gpuon()
    else
        QG3.gpuoff()
    end

    ggsh = QG3.GaussianGridtoSHTransform(pars, hidden_channels; N_batch=batch_size)
    shgg = QG3.SHtoGaussianGridTransform(pars, hidden_channels; N_batch=batch_size)
    plan = ESM_PINOQG3(ggsh, shgg)
    
    gpu_columns = [1]
    for k in 1:corrected_modes-1
        push!(gpu_columns, [ggsh.FT_4d.i_imag+k+1, k+1]...)
    end

    # Store in layer
    ESM_PINO.SphericalConv{ESM_PINOQG3}(hidden_channels, corrected_modes, plan, operator_type, gpu_columns, gain)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::ESM_PINO.SphericalConv{ESM_PINOQG3})
    T_type = typeof(layer.plan.ggsh).parameters[1]
    init_std = T_type(sqrt(layer.gain / layer.hidden_channels))
    
    # Determine weight shape based on operator type
    if layer.operator_type == :diagonal
        # Shape: (out_channels, in_channels, lat_modes, lon_modes)
        # For your case with same in/out channels: (hidden_channels, hidden_channels, modes, 2*modes-1)
        weight = init_std * randn(rng, T_type, 
                                   layer.hidden_channels, 
                                   layer.hidden_channels, 
                                   layer.modes, 
                                   2 * layer.modes - 1)
        
    elseif layer.operator_type == :block_diagonal
        # Shape: (out_channels, in_channels, lat_modes, lon_modes, lon_modes)
        # Extra dimension for matrix multiplication along longitude
        weight = init_std * randn(rng, T_type,
                                   layer.hidden_channels,
                                   layer.hidden_channels,
                                   layer.modes,
                                   2 * layer.modes - 1,
                                   2 * layer.modes - 1)
        
    elseif layer.operator_type == :driscoll_healy
        # Shape: (out_channels, in_channels, lat_modes)
        # Shared across all longitude modes
        weight = init_std * randn(rng, T_type,
                                   layer.hidden_channels,
                                   layer.hidden_channels,
                                   layer.modes)
    else
        error("Unknown operator_type: $(layer.operator_type)")
    end
    
    return (weight=weight,)
end

Lux.initialstates(rng::Random.AbstractRNG, layer::ESM_PINO.SphericalConv{ESM_PINOQG3}) = NamedTuple()

function LuxCore.display_name(layer::ESM_PINO.SphericalConv{ESM_PINOQG3})
    # Extract grid sizes from the transforms
    n_lat_shgg = layer.plan.shgg.output_size[1]  
    n_lat_ggsh = size(layer.plan.ggsh.Pw, 1)
    sh_size_ggsh = size(layer.plan.ggsh.Pw)[2:3]
    sh_size_shgg = size(layer.plan.shgg.P)[2:3] 
    
    # Build the display string
    parts = String[]
    
    # Main layer info with hidden channels
    push!(parts, "SphericalConv($(layer.hidden_channels) channels")
    
    # Grid resolution information
    push!(parts, "grid: $(n_lat_ggsh)→$(sh_size_ggsh)SH$(sh_size_shgg)→$(n_lat_shgg)")
    
    # Modes information (only show if non-zero)
    if layer.modes > 0
        push!(parts, "modes: $(layer.modes)")
    end
    
    # Operator type
    push!(parts, "op: $(layer.operator_type)")
    
    return join(parts, ", ") * ")"
end

function Base.show(io::IO, layer::ESM_PINO.SphericalConv{T}) where T
    print(io, LuxCore.display_name(layer))
end

function (layer::ESM_PINO.SphericalConv{ESM_PINOQG3})(x::AbstractArray{T,4}, ps::NamedTuple, st::NamedTuple) where T
    @assert T == typeof(layer.plan.ggsh).parameters[1] "Input type $T does not match model parameter type $(typeof(layer.plan.ggsh).parameters[1]))"
    
        x_perm = permutedims(x, (3, 1, 2, 4))  # [channels, lat, lon, batch]
        x_tr = QG3.transform_SH(x_perm, layer.plan.ggsh)
        # x_tr shape: (in_channels, lat_modes, lon_modes, batch)
        
        if typeof(layer.plan.ggsh).parameters[end] == true 
            # GPU path
            x_extracted_cols = x_tr[:, :, layer.gpu_cols, :]
            x_extracted = reorderQG3_indexes_4d(x_extracted_cols)
            x_extracted = x_extracted[:,1:layer.modes,:,:]
            # Apply operator based on type
            if layer.operator_type == :diagonal
                # einsum: "bilm,oilm->bolm"
                x_p = ein"ilmb,oilm->olmb"(x_extracted, ps.weight)
                
            elseif layer.operator_type == :block_diagonal
                # einsum: "bilm,oilnm->boln"
                x_p = ein"ilmb,oilnm->olnb"(x_extracted, ps.weight)
                
            elseif layer.operator_type == :driscoll_healy
                # einsum: "bilm,oil->bolm"
                x_p = ein"ilmb,oil->olmb"(x_extracted, ps.weight)
            end
            
            # Remap and pad
            x_p = inverse_reorderQG3_indexes_4d(x_p)
            x_p = remap_array_components(x_p, layer.modes-1, size(layer.plan.shgg.P,3)÷2)
            x_pad = NNlib.pad_zeros(x_p, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, 0, 0, 0))    
        else
            # CPU path
            x_extracted = reorderQG3_indexes_4d(x_tr)
            x_extracted = x_tr[:, 1:layer.modes, 1:2*layer.modes-1, :]
            
            # Apply operator based on type
            if layer.operator_type == :diagonal
                # einsum: "bilm,oilm->bolm"
                x_p = ein"ilmb,oilm->olmb"(x_extracted, ps.weight)    
            elseif layer.operator_type == :block_diagonal
                # einsum: "bilm,oilnm->boln"
                x_p = ein"ilmb,oilnm->olnb"(x_extracted, ps.weight)
            elseif layer.operator_type == :driscoll_healy
                # einsum: "bilm,oil->bolm"
                x_p = ein"ilmb,oil->olmb"(x_extracted, ps.weight)
            end
            
            # Pad
            x_p = inverse_reorderQG3_indexes_4d(x_p)
            x_pad = NNlib.pad_zeros(x_p, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, size(layer.plan.shgg.P, 3) - (2*layer.modes-1), 0, 0))   
        end
        
        # Transform back to grid
        x_out = QG3.transform_grid(x_pad, layer.plan.shgg)
        res_out = QG3.transform_grid(x_tr, layer.plan.shgg)
        
        # Permute back to [lat, lon, channels, batch]
        x_out_perm = permutedims(x_out, (2, 3, 1, 4))
        res_out_perm = permutedims(res_out, (2, 3, 1, 4))
    
    return x_out_perm, res_out_perm, st
end
function Lux.apply(layer::ESM_PINO.SphericalConv{ESM_PINOQG3}, x::AbstractArray{T,4}, ps::NamedTuple, st::NamedTuple) where T
    @assert T == typeof(layer.plan.ggsh).parameters[1] "Input type $T does not match model parameter type $(typeof(layer.plan.ggsh).parameters[1]))"
    
        x_perm = permutedims(x, (3, 1, 2, 4))  # [channels, lat, lon, batch]
        x_tr = QG3.transform_SH(x_perm, layer.plan.ggsh)
        # x_tr shape: (in_channels, lat_modes, lon_modes, batch)
        
        if typeof(layer.plan.ggsh).parameters[end] == true 
            # GPU path
            x_extracted_cols = x_tr[:, :, layer.gpu_cols, :]
            x_extracted = reorderQG3_indexes_4d(x_extracted_cols)
            x_extracted = x_extracted[:,1:layer.modes,:,:]
            # Apply operator based on type
            if layer.operator_type == :diagonal
                # einsum: "bilm,oilm->bolm"
                x_p = ein"ilmb,oilm->olmb"(x_extracted, ps.weight)
                
            elseif layer.operator_type == :block_diagonal
                # einsum: "bilm,oilnm->boln"
                x_p = ein"ilmb,oilnm->olnb"(x_extracted, ps.weight)
                
            elseif layer.operator_type == :driscoll_healy
                # einsum: "bilm,oil->bolm"
                x_p = ein"ilmb,oil->olmb"(x_extracted, ps.weight)
            end
            
            # Remap and pad
            x_p = inverse_reorderQG3_indexes_4d(x_p)
            x_p = remap_array_components(x_p, layer.modes-1, size(layer.plan.shgg.P,3)÷2)
            x_pad = NNlib.pad_zeros(x_p, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, 0, 0, 0))    
        else
            # CPU path
            x_extracted = reorderQG3_indexes_4d(x_tr)
            x_extracted = x_tr[:, 1:layer.modes, 1:2*layer.modes-1, :]
            
            # Apply operator based on type
            if layer.operator_type == :diagonal
                # einsum: "bilm,oilm->bolm"
                x_p = ein"ilmb,oilm->olmb"(x_extracted, ps.weight)    
            elseif layer.operator_type == :block_diagonal
                # einsum: "bilm,oilnm->boln"
                x_p = ein"ilmb,oilnm->olnb"(x_extracted, ps.weight)
            elseif layer.operator_type == :driscoll_healy
                # einsum: "bilm,oil->bolm"
                x_p = ein"ilmb,oil->olmb"(x_extracted, ps.weight)
            end
            
            # Pad
            x_p = inverse_reorderQG3_indexes_4d(x_p)
            x_pad = NNlib.pad_zeros(x_p, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, size(layer.plan.shgg.P, 3) - (2*layer.modes-1), 0, 0))   
        end
        
        # Transform back to grid
        x_out = QG3.transform_grid(x_pad, layer.plan.shgg)
        res_out = QG3.transform_grid(x_tr, layer.plan.shgg)
        
        # Permute back to [lat, lon, channels, batch]
        x_out_perm = permutedims(x_out, (2, 3, 1, 4))
        res_out_perm = permutedims(res_out, (2, 3, 1, 4))
    
    return x_out_perm, res_out_perm, st
end