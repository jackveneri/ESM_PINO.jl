function plot_zonal_mean_velocity(psi, qg3p::QG3.QG3Model; lvl=1, start_time=1, times=size(psi,4))
    zonal_min = Inf
    zonal_max = -Inf
    zonal_mean_velocities = zeros(Float32,size(psi,1), qg3p.p.N_lats, size(psi,4))
    for time = start_time:(start_time + times-1)
        velocity = QG3.u(psi[:,:,:,time], qg3p)
        
        zonal_mean_velocity = reshape(mean(velocity, dims=3), size(velocity)[1:2])
        zonal_min = min(zonal_min, minimum(zonal_mean_velocity[lvl,:]))
        zonal_max = max(zonal_max, maximum(zonal_mean_velocity[lvl,:]))
        zonal_mean_velocities[:,:,time - start_time + 1] = zonal_mean_velocity
    end
    anim = @animate for time in start_time:(start_time + times-1)
        p2 = Plots.plot(zonal_mean_velocities[lvl,:,time], qg3ppars.lats,
              title="Zonal Mean",
              xlabel="Velocity",
              ylabel="Latitude",
              legend=false,
              xlim=(zonal_min, zonal_max))
    end
    return anim    
end

function plot_kinetic_energy(psi, qg3p::QG3.QG3Model; lvl=1)
    k = map( psi -> QG3.kinetic_energy(psi, qg3p), eachslice(psi;dims=4))
    k = cat(k..., dims=4)
    k = reshape(k, 3, :)
    Plots.plot(k[lvl,:],
         title="Kinetic Energy at Level $(lvl)",
         xlabel="Time Step",
         ylabel="Kinetic Energy",
         legend=false)
end