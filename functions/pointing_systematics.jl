mutable struct pointings
    x::AbstractFloat
    y::AbstractFloat
    z::AbstractFloat
    θ::AbstractFloat # position of sky
    φ::AbstractFloat # position of sky
    Ω::Int           # Sky pixel index
    ψ::AbstractFloat # Crossing angle
    ϕ::AbstractFloat # HWP angle
    ξ::AbstractFloat # mod2pi(2ϕ) + ψ
end

function pointings(resol::Resolution, θ, φ, ψ, ϕ)
    vec = ang2vec(θ, φ)
    Ω = ang2pixRing(resol, θ, φ)
    ξ = mod2pi(2ϕ) + ψ
    return pointings(vec[1],vec[2],vec[3], θ, φ, Ω, ψ, ϕ, ξ)
end

function true_signal(p::pointings, maps::PolarizedHealpixMap, pixbuf, weightbuf)
    maps.i[p.Ω] + maps.q[p.Ω]*cos(2p.ξ) + maps.u[p.Ω]*sin(2p.ξ)
end

function true_signal(p::pointings, maps::PolarizedHealpixMap)
    maps.i[p.Ω] + maps.q[p.Ω]*cos(2p.ξ) + maps.u[p.Ω]*sin(2p.ξ)
end

function interp_signal(p::pointings, maps::PolarizedHealpixMap)
    i = interpolate(maps.i, p.θ, p.φ)
    q = interpolate(maps.q, p.θ, p.φ)
    u = interpolate(maps.u, p.θ, p.φ)
    return i + q*cos(2p.ξ) + u*sin(2p.ξ)
end

function interp_signal(p::pointings, maps::PolarizedHealpixMap, pixbuf, weightbuf)
    i = interpolate(maps.i, p.θ, p.φ, pixbuf, weightbuf)
    q = interpolate(maps.q, p.θ, p.φ, pixbuf, weightbuf)
    u = interpolate(maps.u, p.θ, p.φ, pixbuf, weightbuf)
    return i + q*cos(2p.ξ) + u*sin(2p.ξ)
end

function normarize!(resol::Resolution, maps::Array, hitmap::Array)
    N = size(maps)[1]
    M = size(maps)[2]
    if N != M
        for i in 1:N
            maps[i,1,:] ./= hitmap
        end
    end
    if N == M
        for i in 1:N
            for j in 1:N
                maps[i,j,:] ./= hitmap
            end
        end
    end
    return maps
end

function binned_mapmake(ss::ScanningStrategy, division::Int, inputinfo::Falcons.InputInfo, signal)
    w(ψ,ϕ) = @SMatrix [1 cos(2ψ+4ϕ) sin(2ψ+4ϕ)]
    resol = Resolution(ss.nside)
    npix = resol.numOfPixels
    chunk = Int(ss.duration / division)
    ω_hwp = rpm2angfreq(ss.hwp_rpm)
    total_signal = zeros(3, 1, npix)
    hitmap = zeros(Int64, npix)
    hitmatrix = zeros(3, 3, npix)
    outmap = zeros(3, 1, npix)
    progress = Progress(division)
    inputmap = inputinfo.Inputmap
    pixbuf = Array{Int}(undef, 4)
    weightbuf = Array{Float64}(undef, 4)
    BEGIN = 0
    @inbounds @views for i = 1:division
        END = i * chunk
        theta, phi, psi, time = get_pointings_tuple(ss, BEGIN, END)
        @views @inbounds for j = eachindex(length(ss.FP_theta))
            theta_j = theta[:,j]
            phi_j = phi[:,j]
            psi_j = psi[:,j]
            @inbounds @views for k = eachindex(time)
                t = time[k]
                p = pointings(resol, theta_j[k], phi_j[k], psi_j[k], mod2pi(ω_hwp*t))
                dₖ = signal(p, inputmap)
                total_signal[:, :, p.Ω] .+= @SMatrix [dₖ; dₖ*cos(2p.ψ+4p.ϕ); dₖ*sin(2p.ψ+4p.ϕ)]
                hitmap[p.Ω] += 1
                hitmatrix[:, :, p.Ω] .+= transpose(w(p.ψ, p.ϕ)) * w(p.ψ, p.ϕ)
            end
        end
        BEGIN = END
        next!(progress)
    end
    normarize!(resol, total_signal, hitmap)
    normarize!(resol, hitmatrix, hitmap)
    @inbounds @threads for j = 1:npix
        outmap[:, :, j] = hitmatrix[:, :, j] \ total_signal[:, :, j]
    end
    outmap = transpose([outmap[1,1,:] outmap[2,1,:] outmap[3,1,:]])
    return outmap, hitmap
end

function get_pointing_offset(ss::ScanningStrategy, division::Int, inputinfo::Falcons.InputInfo,; signal, tod_check=false, start_time=0, end_time=60*60)
    w(ψ,ϕ) = @SMatrix [1 cos(2ψ+4ϕ) sin(2ψ+4ϕ)]
    resol = Resolution(ss.nside)
    npix = resol.numOfPixels
    chunk = Int(ss.duration / division)
    ω_hwp = rpm2angfreq(ss.hwp_rpm)
    total_signal = zeros(3, 1, npix)
    hitmap = zeros(Int64, npix)
    hitmatrix = zeros(3, 3, npix)
    outmap = zeros(3, 1, npix)
    progress = Progress(division)
    inputmap = inputinfo.Inputmap
    pixbuf = Array{Int}(undef, 4)
    weightbuf = Array{Float64}(undef, 4)
    BEGIN = 0
    ss_err = deepcopy(ss)
    
    ss_err.FP_theta = ss.FP_theta .+ rad2deg.(inputinfo.Systematics.Pointing.offset_ρ)
    ss_err.FP_phi = ss.FP_phi .+ rad2deg.(inputinfo.Systematics.Pointing.offset_χ)
    
    @inbounds @views for i = 1:division
        END = i * chunk
        theta, phi, psi, time = get_pointings_tuple(ss, BEGIN, END)
        theta_e, phi_e, psi_e, time_e = get_pointings_tuple(ss_err, BEGIN, END)
        @views @inbounds for j = eachindex(ss.FP_theta)       
            theta_j = theta[:,j]
            phi_j = phi[:,j]
            psi_j = psi[:,j]
            theta_e_j = theta_e[:,j]
            phi_e_j = phi_e[:,j]
            psi_e_j = psi_e[:,j]
            @inbounds @views for k = eachindex(time)
                t = time[k]
                p = pointings(resol, theta_j[k], phi_j[k], psi_j[k], mod2pi(ω_hwp*t))
                p_err = pointings(resol, theta_e_j[k], phi_e_j[k], psi_e_j[k], mod2pi(ω_hwp*t))
                dₖ = signal(p_err, inputmap)
                total_signal[:, :, p.Ω] .+= @SMatrix [dₖ; dₖ*cos(2p.ξ); dₖ*sin(2p.ξ)]
                hitmatrix[:, :, p.Ω] .+= transpose(w(p.ψ, p.ϕ)) * w(p.ψ, p.ϕ)
                hitmap[p.Ω] += 1
            end
        end
        BEGIN = END
        next!(progress)
    end
    normarize!(resol, total_signal, hitmap)
    normarize!(resol, hitmatrix, hitmap)
    @inbounds @threads for j = 1:npix
        if det(hitmatrix[:, :, j]) != 0
            outmap[:, :, j] = hitmatrix[:, :, j] \ total_signal[:, :, j]
        end
    end
    outmap = transpose([outmap[1,1,:] outmap[2,1,:] outmap[3,1,:]])
    
    if tod_check == true
        theta, phi, psi, time = get_pointings_tuple(ss, start_time, end_time)
        theta_e, phi_e, psi_e, time_e = get_pointings_tuple(ss_err, start_time, end_time)
        tod_true = zeros(length(time), length(ss.FP_theta))
        tod_true_interp = zeros(length(time), length(ss.FP_theta))
        tod_err = zeros(length(time_e), length(ss_err.FP_theta))
        pol_ang = zeros(length(time), length(ss_err.FP_theta))
        for j in eachindex(ss.FP_theta)
            for k in eachindex(time)
                t = time[k]
                p = pointings(resol, theta[k,j], phi[k,j], psi[k,j], ω_hwp*t)
                p_err = pointings(resol, theta_e[k,j], phi_e[k,j], psi_e[k,j], ω_hwp*t)
                d_true = signal(p, inputmap)
                d_true_interp = interp_signal(p, inputmap)
                d_err = signal(p_err, inputmap)
                pol_ang[k,j] = p_err.ξ
                tod_true[k,j] = d_true
                tod_true_interp[k,j] = d_true_interp
                tod_err[k,j] = d_err
            end
        end
        return outmap, hitmap, theta, phi, psi, pol_ang, time, tod_true, tod_true_interp, tod_err
    end
    return outmap, hitmap
end

function get_pointing_offset_extend(ss::ScanningStrategy, division::Int, inputinfo::Falcons.InputInfo,; signal, tod_check=false, start_time=0, end_time=60*60, w)
    #w(ψ,ϕ) = @SMatrix [1 cos(2ψ+4ϕ) sin(2ψ+4ϕ)]
    matsize = length(w(0,0))
    resol = Resolution(ss.nside)
    npix = resol.numOfPixels
    chunk = Int(ss.duration / division)
    ω_hwp = rpm2angfreq(ss.hwp_rpm)
    total_signal = zeros(matsize, 1, npix)
    hitmap = zeros(Int64, npix)
    hitmatrix = zeros(matsize, matsize, npix)
    outmap = zeros(matsize, 1, npix)
    progress = Progress(division)
    inputmap = inputinfo.Inputmap
    pixbuf = Array{Int}(undef, 4)
    weightbuf = Array{Float64}(undef, 4)
    BEGIN = 0
    ss_err = deepcopy(ss)
    ss_err.FP_theta = ss.FP_theta .+ rad2deg.(inputinfo.Systematics.Pointing.offset_ρ)
    ss_err.FP_phi = ss.FP_phi .+ rad2deg.(inputinfo.Systematics.Pointing.offset_χ)
    @inbounds @views for i = 1:division
        END = i * chunk
        theta, phi, psi, time = get_pointings_tuple(ss, BEGIN, END)
        theta_e, phi_e, psi_e, time_e = get_pointings_tuple(ss_err, BEGIN, END)
        @views @inbounds for j = eachindex(ss.FP_theta)       
            theta_j = theta[:,j]
            phi_j = phi[:,j]
            psi_j = psi[:,j]
            theta_e_j = theta_e[:,j]
            phi_e_j = phi_e[:,j]
            psi_e_j = psi_e[:,j]
            @inbounds @views for k = eachindex(time)
                t = time[k]
                p = pointings(resol, theta_j[k], phi_j[k], psi_j[k], mod2pi(ω_hwp*t))
                p_err = pointings(resol, theta_e_j[k], phi_e_j[k], psi_e_j[k], mod2pi(ω_hwp*t))
                dₖ = signal(p_err, inputmap)
                #@show SMatrix(dₖ.*w(p.ψ, p.ϕ))
                #@show total_signal[:, :, p.Ω]
                total_signal[:, :, p.Ω] .+= SMatrix(transpose(dₖ.*w(p.ψ, p.ϕ)))
                hitmatrix[:, :, p.Ω] .+= transpose(w(p.ψ, p.ϕ)) * w(p.ψ, p.ϕ)
                hitmap[p.Ω] += 1
            end
        end
        BEGIN = END
        next!(progress)
    end
    normarize!(resol, total_signal, hitmap)
    normarize!(resol, hitmatrix, hitmap)
    @inbounds @threads for j = 1:npix
        if det(hitmatrix[:, :, j]) != 0
            outmap[:, :, j] = hitmatrix[:, :, j] \ total_signal[:, :, j]
        end
    end
    #outmap = transpose([outmap[1,1,:] outmap[2,1,:] outmap[3,1,:]])
    if tod_check == true
        theta, phi, psi, time = get_pointings_tuple(ss, start_time, end_time)
        theta_e, phi_e, psi_e, time_e = get_pointings_tuple(ss_err, start_time, end_time)
        tod_true = zeros(length(time), length(ss.FP_theta))
        tod_true_interp = zeros(length(time), length(ss.FP_theta))
        tod_err = zeros(length(time_e), length(ss_err.FP_theta))
        pol_ang = zeros(length(time), length(ss_err.FP_theta))
        for j in eachindex(ss.FP_theta)
            for k in eachindex(time)
                t = time[k]
                p = pointings(resol, theta[k,j], phi[k,j], psi[k,j], ω_hwp*t)
                p_err = pointings(resol, theta_e[k,j], phi_e[k,j], psi_e[k,j], ω_hwp*t)
                d_true = signal(p, inputmap)
                d_true_interp = interp_signal(p, inputmap)
                d_err = signal(p_err, inputmap)
                pol_ang[k,j] = p_err.ξ
                tod_true[k,j] = d_true
                tod_true_interp[k,j] = d_true_interp
                tod_err[k,j] = d_err
            end
        end
        return outmap, hitmap, theta, phi, psi, pol_ang, time, tod_true, tod_true_interp, tod_err
    end
    return outmap, hitmap
end


function gen_signalfield(resol::Resolution, maps::PolarizedHealpixMap)
    alm_i = hp.map2alm(maps.i.pixels)
    alm_q = hp.map2alm(maps.q.pixels)
    alm_u = hp.map2alm(maps.u.pixels)
    di = hp.alm2map_der1(alm_i, resol.nside)
    dq = hp.alm2map_der1(alm_q, resol.nside)
    du = hp.alm2map_der1(alm_u, resol.nside)
    return di,dq,du
end
    
function abs_point(p::pointings, m, II::Falcons.InputInfo)
    ∂I = @views m[1][3,p.Ω] - m[1][2,p.Ω]im
    ∂P = @views m[2][3,p.Ω] + m[3][2,p.Ω] - (m[2][2,p.Ω] - m[3][3,p.Ω])im
    ∂̄P = @views m[2][3,p.Ω] - m[3][2,p.Ω] + (m[2][2,p.Ω] + m[3][3,p.Ω])im
    I = II.Inputmap.i.pixels[p.Ω]
    P = II.Inputmap.q.pixels[p.Ω] + II.Inputmap.u.pixels[p.Ω]*im
    ρ = II.Systematics.Pointing.offset_ρ[1]
    χ = II.Systematics.Pointing.offset_χ[1]
    I1 = I - (ρ/2)*(ℯ^(im*(p.ψ+χ))*∂I + ℯ^(-im*(p.ψ+χ))*conj(∂I))
    P1 = (1/2) * (P*ℯ^(im*(2p.ψ+4p.ϕ))        - (ρ/2) * (ℯ^(im*(3p.ψ+4p.ϕ+χ))*∂P       + ℯ^(im*(p.ψ+4p.ϕ-χ))*∂̄P ))
    P2 = (1/2) * (conj(P)*ℯ^(-im*(2p.ψ+4p.ϕ)) - (ρ/2) * (ℯ^(im*(-p.ψ-4p.ϕ+χ))*conj(∂̄P) + ℯ^(im*(-3p.ψ-4p.ϕ-χ))*conj(∂P) ))
    return I1 + P1 + P2
end
