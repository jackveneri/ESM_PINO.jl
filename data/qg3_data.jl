dir = @__DIR__
using Pkg
Pkg.activate(dir)
Pkg.instantiate()

using QG3, JLD2, OrdinaryDiffEq, CUDA

"""
    load_data(name::String, path="data-files/", GPU::Bool)

Loads pre-computed data for the QG Model, e.g. for a `name=T42` grid.  
"""
function load_data(name::String, path="/../data/"; GPU::Bool=false)

    root_path = @__DIR__
    
    if name=="T42"
        @load string(root_path, path, "t42-precomputed-S.jld2") S
        @load string(root_path, path, "t42-precomputed-p.jld2") qg3ppars
        @load string(root_path, path, "t42-precomputed-sf.jld2") ψ_0
        @load string(root_path, path, "t42-precomputed-q.jld2") q_0
    elseif name=="T21"
        S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()
    else
        error("Unknown grid, only T21, T42 and T60 available")
    end

    if GPU
        S, qg3ppars, ψ_0, q_0 = QG3.reorder_SH_gpu(S, qg3ppars), togpu(qg3ppars), QG3.reorder_SH_gpu(ψ_0, qg3ppars), QG3.reorder_SH_gpu(q_0, qg3ppars)
    end
    return S, qg3ppars, ψ_0, q_0
end

"""
$(TYPEDSIGNATURES)
Compute training data from the QG3 Model.
"""
function compute_QG3_data(qg3p::QG3Model{T}, q_0, S, DT::Number; t_save_length::Number, t_transient=T(100), reltol=1e-5) where T
    DT = T(DT)
    t_save_length = T(t_save_length)
    t_end = t_transient + t_save_length

    prob = ODEProblem(QG3.QG3MM_gpu,q_0,(T(0.),t_end),[qg3p, S])
    sol = @time solve(prob, Tsit5(), dt=DT, saveat=t_transient:DT:t_end, reltol=reltol)
        
    q = QG3.reorder_SH_cpu(Array(sol), qg3p.p) # cpu for saving 
    t = sol.t 

    return (t, q)
end 

S, qg3ppars, ψ_0, q_0 = load_data("T42", GPU=true)
qg3p = @CUDA.allowscalar QG3Model(qg3ppars)
DT = 1 # time step in QG3 units
dataset_size = 6000
t,q = compute_QG3_data(qg3p, q_0, S, DT, t_save_length=DT*(dataset_size-1))

@save string(dir, "/t42_qg3_data_SH_CPU.jld2") t q DT