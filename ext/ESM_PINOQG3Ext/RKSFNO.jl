function ESM_PINO.RKSFNO(qg3ppars::QG3.QG3ModelParameters; kwargs...)
    @assert kwargs[:in_channels] == kwargs[:out_channels] "For RKSFNO, in_channels must equal out_channels."
    sfno = SFNO(qg3ppars; kwargs...)
    return RKSFNO{typeof(sfno.embedding), typeof(sfno.lifting), typeof(sfno.sfno_blocks), typeof(sfno.projection), typeof(sfno.ext_type)}(sfno)
end

function Lux.initialparameters(rng::AbstractRNG, layer::ESM_PINO.RKSFNO{E, L, B, P, ESM_PINOQG3}) where {E, L, B, P}
    ps_sfno = Lux.initialparameters(rng, layer.sfno)
    return (sfno=ps_sfno,)
end

function Lux.initialstates(rng::AbstractRNG, layer::ESM_PINO.RKSFNO{E, L, B, P, ESM_PINOQG3}) where {E, L, B, P}
    st_sfno = Lux.initialstates(rng, layer.sfno)
    return (sfno=st_sfno,)
end

function (layer::ESM_PINO.RKSFNO{E, L, B, P, ESM_PINOQG3})(u::AbstractArray, ps::NamedTuple, st::NamedTuple) where {E, L, B, P}
    k1, st_sfno = layer.sfno(u,            ps.sfno, st.sfno)
    k2, st_sfno = layer.sfno(u .+ k1./2f0, ps.sfno, st_sfno)
    k3, st_sfno = layer.sfno(u .+ k2./2f0, ps.sfno, st_sfno)
    k4, st_sfno = layer.sfno(u .+ k3,      ps.sfno, st_sfno)

    u_next = u .+ (k1 .+ 2f0.*k2 .+ 2f0.*k3 .+ k4) ./ 6f0
    return u_next, (sfno=st_sfno,)
end

function Lux.apply(layer::ESM_PINO.RKSFNO{E, L, B, P, ESM_PINOQG3}, u::AbstractArray, ps::NamedTuple, st::NamedTuple) where {E, L, B, P}
    k1, st_sfno = layer.sfno(u,            ps.sfno, st.sfno)
    k2, st_sfno = layer.sfno(u .+ k1./2f0, ps.sfno, st_sfno)
    k3, st_sfno = layer.sfno(u .+ k2./2f0, ps.sfno, st_sfno)
    k4, st_sfno = layer.sfno(u .+ k3,      ps.sfno, st_sfno)

    u_next = u .+ (k1 .+ 2f0.*k2 .+ 2f0.*k3 .+ k4) ./ 6f0
    return u_next, (sfno=st_sfno,)
end