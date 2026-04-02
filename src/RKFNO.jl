struct RKFNO{T, N, E} <: Lux.AbstractLuxContainerLayer{(:fno,)}
    fno::FourierNeuralOperator{T, N, E}
    #function RKFNO(fno::FourierNeuralOperator{T, N, E}) where {T, N, E}
    #    return new{T, N, E}(fno)
    #end
end

function RKFNO(;kwargs...)
    @assert kwargs[:in_channels] == kwargs[:out_channels] "For RKFNO, in_channels must equal out_channels."
    @assert kwargs[:n_modes] !== nothing "n_modes must be specified for RKFNO."
    fno = FourierNeuralOperator(;kwargs...)
    return RKFNO{ComplexF32, length(kwargs[:n_modes]), typeof(fno.embedding)}(fno)
end

function (layer::RKFNO)(u, ps, st)
    k1, st_fno = layer.fno(u,            ps.fno, st.fno)
    k2, st_fno = layer.fno(u .+ k1./2f0, ps.fno, st_fno)
    k3, st_fno = layer.fno(u .+ k2./2f0, ps.fno, st_fno)
    k4, st_fno = layer.fno(u .+ k3,      ps.fno, st_fno)

    u_next = u .+ (k1 .+ 2f0.*k2 .+ 2f0.*k3 .+ k4) ./ 6f0
    return u_next, (fno=st_fno,)
end
