function ESM_PINO.SphericalConv(
        hidden_channels::Int,
        ggsh::QG3.GaussianGridtoSHTransform,
        shgg::QG3.SHtoGaussianGridTransform,
        modes::Int = 0;
        operator_type::Symbol = :driscoll_healy  # Changed from zsk::Bool
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
    ESM_PINO.SphericalConv{ESM_PINOQG3}(hidden_channels, corrected_modes, plan, operator_type, gpu_columns)
end

function ESM_PINO.SphericalConv(
        pars::QG3.QG3ModelParameters{T},
        hidden_channels::Int;
        modes::Int = pars.L,
        batch_size::Int = 1,
        gpu::Bool = true,
        operator_type::Symbol = :driscoll_healy  # Changed from zsk::Bool
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
    ESM_PINO.SphericalConv{ESM_PINOQG3}(hidden_channels, corrected_modes, plan, operator_type, gpu_columns)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::ESM_PINO.SphericalConv{ESM_PINOQG3})
    T_type = typeof(layer.plan.ggsh).parameters[1]
    init_std = T_type(sqrt(2 / layer.hidden_channels))
    
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

function (layer::ESM_PINO.SphericalConv{ESM_PINOQG3})(x::AbstractArray{T,4}, ps::NamedTuple, st::NamedTuple) where T
    @assert T == typeof(layer.plan.ggsh).parameters[1] "Input type $T does not match model parameter type $(typeof(layer.plan.ggsh).parameters[1]))"
    
    
        x_perm = permutedims(x, (3, 1, 2, 4))  # [channels, lat, lon, batch]
        x_tr = QG3.transform_SH(x_perm, layer.plan.ggsh)
        # x_tr shape: (in_channels, lat_modes, lon_modes, batch)
        
        if typeof(layer.plan.ggsh).parameters[end] == true 
            # GPU path
            x_extracted = x_tr[:, 1:layer.modes, layer.gpu_cols, :]
            
            # Apply operator based on type
            if layer.operator_type == :diagonal
                # einsum: "bilm,oilm->bolm"
                x_p = ein"ilmb,oilm->olmb"(x_extracted, ps.weight)
                x_res = x_extracted
                
            elseif layer.operator_type == :block_diagonal
                # einsum: "bilm,oilnm->boln"
                x_p = ein"ilmb,oilnm->olnb"(x_extracted, ps.weight)
                x_res = x_extracted
                
            elseif layer.operator_type == :driscoll_healy
                # einsum: "bilm,oil->bolm"
                x_p = ein"ilmb,oil->olmb"(x_extracted, ps.weight)
                x_res = x_extracted
            end
            
            # Remap and pad
            x_p = remap_array_components(x_p, layer.modes-1, size(layer.plan.shgg.P,3)÷2)
            x_pad = NNlib.pad_zeros(x_p, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, 0, 0, 0))
            
            x_res = remap_array_components(x_res, layer.modes-1, size(layer.plan.shgg.P,3)÷2)
            x_res_pad = NNlib.pad_zeros(x_res, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, 0, 0, 0))
            
        else
            # CPU path
            x_extracted = x_tr[:, 1:layer.modes, 1:2*layer.modes-1, :]
            
            # Apply operator based on type
            if layer.operator_type == :diagonal
                # einsum: "bilm,oilm->bolm"
                x_p = ein"ilmb,oilm->olmb"(x_extracted, ps.weight)
                x_res = x_extracted
                
            elseif layer.operator_type == :block_diagonal
                # einsum: "bilm,oilnm->boln"
                x_p = ein"ilmb,oilnm->olnb"(x_extracted, ps.weight)
                x_res = x_extracted
                
            elseif layer.operator_type == :driscoll_healy
                # einsum: "bilm,oil->bolm"
                x_p = ein"ilmb,oil->olmb"(x_extracted, ps.weight)
                x_res = x_extracted
            end
            
            # Pad
            x_pad = NNlib.pad_zeros(x_p, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, size(layer.plan.shgg.P, 3) - (2*layer.modes-1), 0, 0))
            x_res_pad = NNlib.pad_zeros(x_res, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, size(layer.plan.shgg.P, 3) - (2*layer.modes-1), 0, 0))
        end
        
        # Transform back to grid
        x_out = QG3.transform_grid(x_pad, layer.plan.shgg)
        res_out = QG3.transform_grid(x_res_pad, layer.plan.shgg)
        
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
            x_extracted = x_tr[:, 1:layer.modes, layer.gpu_cols, :]
            
            # Apply operator based on type
            if layer.operator_type == :diagonal
                # einsum: "bilm,oilm->bolm"
                x_p = ein"ilmb,oilm->olmb"(x_extracted, ps.weight)
                x_res = x_extracted
                
            elseif layer.operator_type == :block_diagonal
                # einsum: "bilm,oilnm->boln"
                x_p = ein"ilmb,oilnm->olnb"(x_extracted, ps.weight)
                x_res = x_extracted
                
            elseif layer.operator_type == :driscoll_healy
                # einsum: "bilm,oil->bolm"
                x_p = ein"ilmb,oil->olmb"(x_extracted, ps.weight)
                x_res = x_extracted
            end
            
            # Remap and pad
            x_p = remap_array_components(x_p, layer.modes-1, size(layer.plan.shgg.P,3)÷2)
            x_pad = NNlib.pad_zeros(x_p, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, 0, 0, 0))
            
            x_res = remap_array_components(x_res, layer.modes-1, size(layer.plan.shgg.P,3)÷2)
            x_res_pad = NNlib.pad_zeros(x_res, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, 0, 0, 0))
            
        else
            # CPU path
            x_extracted = x_tr[:, 1:layer.modes, 1:2*layer.modes-1, :]
            
            # Apply operator based on type
            if layer.operator_type == :diagonal
                # einsum: "bilm,oilm->bolm"
                x_p = ein"ilmb,oilm->olmb"(x_extracted, ps.weight)
                x_res = x_extracted
                
            elseif layer.operator_type == :block_diagonal
                # einsum: "bilm,oilnm->boln"
                x_p = ein"ilmb,oilnm->olnb"(x_extracted, ps.weight)
                x_res = x_extracted
                
            elseif layer.operator_type == :driscoll_healy
                # einsum: "bilm,oil->bolm"
                x_p = ein"ilmb,oil->olmb"(x_extracted, ps.weight)
                x_res = x_extracted
            end
            
            # Pad
            x_pad = NNlib.pad_zeros(x_p, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, size(layer.plan.shgg.P, 3) - (2*layer.modes-1), 0, 0))
            x_res_pad = NNlib.pad_zeros(x_res, (0, 0, 0, size(layer.plan.shgg.P, 2) - layer.modes, 0, size(layer.plan.shgg.P, 3) - (2*layer.modes-1), 0, 0))
        end
        
        # Transform back to grid
        x_out = QG3.transform_grid(x_pad, layer.plan.shgg)
        res_out = QG3.transform_grid(x_res_pad, layer.plan.shgg)
        
        # Permute back to [lat, lon, channels, batch]
        x_out_perm = permutedims(x_out, (2, 3, 1, 4))
        res_out_perm = permutedims(res_out, (2, 3, 1, 4))
     
    
    return x_out_perm, res_out_perm, st
end