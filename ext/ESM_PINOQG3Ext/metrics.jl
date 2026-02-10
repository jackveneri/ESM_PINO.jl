"""
    compute_ACC(X_pred, X_true, lats, ltm; l=1)

Compute the Anomaly Correlation Coefficient (ACC) for a variable at time-step l.

# Arguments
- `X_pred`: Predicted values, array with dimensions [lat, lon, time] or [lat, lon]
- `X_true`: True values, array with dimensions [lat, lon, time] or [lat, lon]
- `lats`: Latitude values in degrees for each latitude grid point (length NLat)
- `ltm`: Long-term mean to subtract (array with dimensions [lat, lon])

# Returns
- ACC value (scalar between -1 and 1)
"""
function compute_ACC(X_pred, X_true, lats, ltm)
    # Compute latitude weights once
    wLat_grid = QG3.togpu(LinearAlgebra.normalize(cos.(lats), 1))
    
    # Handle 2D case (single time slice)
    if ndims(X_pred) == 2
        X_pred_anom = X_pred .- ltm
        X_true_anom = X_true .- ltm
        
        numerator = sum(wLat_grid .* X_pred_anom .* X_true_anom)
        sum_pred_sq = sum(wLat_grid .* X_pred_anom.^2)
        sum_true_sq = sum(wLat_grid .* X_true_anom.^2)
        denominator = sqrt(sum_pred_sq * sum_true_sq)
        
        return numerator / denominator
    end
    
    # Compute all anomalies at once (broadcasting over time dimension)
    X_pred_anom = X_pred .- ltm
    X_true_anom = X_true .- ltm
    
    # Vectorized computation across time dimension
    # Reshape wLat_grid to broadcast correctly: [lat, lon, 1]
    wLat_grid_3d = reshape(wLat_grid, size(wLat_grid)..., 1)
    
    # Compute numerators for all time steps at once
    numerators = dropdims(sum(wLat_grid_3d .* X_pred_anom .* X_true_anom, dims=(1,2)), dims=(1,2))
    
    # Compute denominator terms for all time steps
    sum_pred_sq = dropdims(sum(wLat_grid_3d .* X_pred_anom.^2, dims=(1,2)), dims=(1,2))
    sum_true_sq = dropdims(sum(wLat_grid_3d .* X_true_anom.^2, dims=(1,2)), dims=(1,2))
    
    # Compute ACC for all time steps
    denominators = sqrt.(sum_pred_sq .* sum_true_sq)
    ACC_values = numerators ./ denominators
    
    # Convert to CPU if needed (CuArray -> Array)
    return Array(ACC_values)
end    

"""
    compute_ACC(X_pred, X_true, pars::QG3.QG3ModelParameters, ltm)

Compute the Anomaly Correlation Coefficient (ACC) for a variable at each time-step.

# Arguments
- `X_pred`: Predicted values, array with dimensions [lat, lon, time] or [lat, lon]
- `X_true`: True values, array with dimensions [lat, lon, time] or [lat, lon]
- `pars::QG3.QG3ModelParameters`: QG3 model parameters containing latitude weights
- `ltm`: Long-term mean to subtract (array with dimensions [lat, lon])

# Returns
- ACC value(s): scalar if input is 2D, vector of length `time` if input is 3D
"""
function compute_ACC(X_pred, X_true, pars::QG3.QG3ModelParameters, ltm)
    # Compute latitude weights once
    wLat_grid = QG3.togpu(LinearAlgebra.normalize(cos.(pars.lats), 1))
    
    # Handle 2D case (single time slice)
    if ndims(X_pred) == 2
        X_pred_anom = X_pred .- ltm
        X_true_anom = X_true .- ltm
        
        numerator = sum(wLat_grid .* X_pred_anom .* X_true_anom)
        sum_pred_sq = sum(wLat_grid .* X_pred_anom.^2)
        sum_true_sq = sum(wLat_grid .* X_true_anom.^2)
        denominator = sqrt(sum_pred_sq * sum_true_sq)
        
        return numerator / denominator
    end
    
    # Compute all anomalies at once (broadcasting over time dimension)
    X_pred_anom = X_pred .- ltm
    X_true_anom = X_true .- ltm
    
    # Vectorized computation across time dimension
    # Reshape wLat_grid to broadcast correctly: [lat, lon, 1]
    wLat_grid_3d = reshape(wLat_grid, size(wLat_grid)..., 1)
    
    # Compute numerators for all time steps at once
    numerators = dropdims(sum(wLat_grid_3d .* X_pred_anom .* X_true_anom, dims=(1,2)), dims=(1,2))
    
    # Compute denominator terms for all time steps
    sum_pred_sq = dropdims(sum(wLat_grid_3d .* X_pred_anom.^2, dims=(1,2)), dims=(1,2))
    sum_true_sq = dropdims(sum(wLat_grid_3d .* X_true_anom.^2, dims=(1,2)), dims=(1,2))
    
    # Compute ACC for all time steps
    denominators = sqrt.(sum_pred_sq .* sum_true_sq)
    ACC_values = numerators ./ denominators
    
    # Convert to CPU if needed (CuArray -> Array)
    return Array(ACC_values)
end