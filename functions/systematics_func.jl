using Healpix
using Falcons
using PyCall
using Base.Threads
using NPZ
using Statistics
using StaticArrays
using ProgressMeter

hp = pyimport("healpy")
np = pyimport("numpy")

function diff_gain_mapbase(hmap, inputinfo)
    II = inputinfo
    h̃ = @views hmap["h"]
    Imap = @views II.Inputmap.i.pixels
    Pmap = @views II.Inputmap.q.pixels .+ (II.Inputmap.u.pixels)im
    g_A = inputinfo.Systematics.Gain.offset[1]
    g_B = 0

    S₀  = @views @. (1/4) * (g_A - g_B) * Imap
    S₂  = @views @. (1/8) * Pmap * (2 + g_A + g_B)
    S₋₂ = @views @. conj(S₂)
    
    determ = @views @. 4/(abs2(h̃[:,4]) - 1)
    
    ₊₂Sᵈ  = @views @. h̃[:,2]*S₀ + h̃[:,4]*S₋₂ + S₂
    ₋₂Sᵈ  = @views @. conj(h̃[:,2])*S₀ + S₋₂ + conj(h̃[:,4])*S₂

    Pstar = @views @. determ * (conj(h̃[:,4]) * ₊₂Sᵈ - ₋₂Sᵈ)
    mapbase = @views [Imap.*0 real.(Pstar) -imag.(Pstar)] |> transpose;
end

function diff_pointing_mapbase(hmap, inputinfo)
    II = inputinfo
    res = Resolution(npix2nside(length(II.Inputmap.i.pixels)))
    Imap = @views II.Inputmap.i.pixels
    Pmap = @views II.Inputmap.q.pixels .+ (II.Inputmap.u.pixels)im
    alm_I = hp.map2alm(Imap)
    alm_Q = hp.map2alm(real.(Pmap))
    alm_U = hp.map2alm(imag.(Pmap))
    dI = hp.alm2map_der1(alm_I, res.nside)
    dQ = hp.alm2map_der1(alm_Q, res.nside)
    dU = hp.alm2map_der1(alm_U, res.nside)
    
    h̃ = @views hmap["h"]
    #replace!(h̃, NaN+NaN*im=>0)
    ζ = II.Systematics.Pointing.offset_ρ[1] * ℯ^(im * II.Systematics.Pointing.offset_χ[1])
    ∂I = @views dI[3,:] + dI[2,:]im
    ∂P = @views dQ[3,:] - dU[2,:] + (dQ[2,:] - dU[3,:])im
    ∂̄P = @views dQ[3,:] + dU[2,:] - (dQ[2,:] - dU[3,:])im 

    S₁  = @views @. -(1/8)*conj(ζ)*∂I - (1/16)*ζ*∂̄P
    S₋₁ = @views @. conj(S₁)

    S₂  = @views @. Pmap/4
    S₋₂ = @views @. conj(S₂)

    S₃  = @views @. -(1/16)*conj(ζ)*∂P
    S₋₃ = @views @. conj(S₃)

    ₊₂Sᵈ  = @views @. h̃[:,5]*S₋₃ + h̃[:,4]*S₋₂ + h̃[:,3]*S₋₁ + h̃[:,2]*0 + h̃[:,1]*S₁ + S₂ + conj(h̃[:,1])*S₃;
    ₋₂Sᵈ  = @views @. h̃[:,1]*S₋₃ + S₋₂ + conj(h̃[:,1])*S₋₁ + conj(h̃[:,2])*0 + conj(h̃[:,3])*S₁ + conj(h̃[:,4])*S₂ + conj(h̃[:,5])*S₃

    determ = @views @. 4/(abs2(h̃[:,4]) -1)
    Pstar = @views @. determ * (conj(h̃[:,4]) * ₊₂Sᵈ - ₋₂Sᵈ);
    mapbase_pointing = @views [Imap.*0 real.(Pstar) -imag.(Pstar)] |> transpose;
end

function forecasting(lmax, cl_obs,;
        path="/home/cmb/yusuket/program/MapData/CAMB/ClBB_PTEPini.npz", 
        rmin=1e-8,
        rmax=1e-1,
        rresol=1e5 |> Int,
        iter=3,
        verbose=false
    )
    gridOfr = range(rmin, rmax, length=rresol)
    cl_models = npzread(path)
    cl_lens = @views cl_models["lens"]
    cl_tens = @views cl_models["tensor"]
    ℓ = 2:lmax
    Nₗ = length(ℓ)
    delta_r = 0.0
    likelihood = 0
    @views @inbounds for j in 0:iter
        Nᵣ = length(gridOfr)
        likelihood = zeros(Nᵣ)
        @views @inbounds @threads for i in eachindex(gridOfr)
            Cl_hat = @. cl_obs[3:lmax+1] + cl_lens[3:lmax+1]
            Cl = @. gridOfr[i] * cl_tens[3:lmax+1] + cl_lens[3:lmax+1]
            likelihood[i] = sum(@.((-0.5)*(2*ℓ + 1)*((Cl_hat / Cl) + log(Cl) - ((2*ℓ - 1)/(2*ℓ + 1))*log(Cl_hat))))
        end
        likelihood = @views exp.(likelihood .- maximum(likelihood))
        maxid = findmax(likelihood)[2]
        delta_r = gridOfr[maxid]
        survey_range = @views [delta_r - delta_r*(0.5/(j+1)), delta_r + delta_r*(0.5/(j+1))]
        gridOfr = range(survey_range[1], survey_range[2], length=10000)
        if verbose == true
            println("*--------------------------- iter = $j ---------------------------*")
            println(" Δr                : $delta_r")
            println(" Next survey range : $survey_range")
        end
    end
    return delta_r, gridOfr, likelihood
end

function get_hnm_map(SS::ScanningStrategy,; division::Int, spin_n, spin_m)
    h = orientation_func_hwp
    resol = Resolution(SS.nside)
    npix = nside2npix(SS.nside)
    
    month = Int(SS.duration / division)
    ω_hwp = rpm2angfreq(SS.hwp_rpm)
    mmax = 3
    hit_map = zeros(Int64, npix)
    hₙₘ = zeros(Complex{Float64}, (length(spin_n), length(spin_m), npix))
    D_μ = zeros(Complex{Float64}, (3, 3, npix))
    BEGIN = 0
    p = Progress(division)
    @views @inbounds for i = 1:division
        END = i * month
        pix_tod, psi_tod, time_array = get_pointing_pixels(SS, BEGIN, END)
        @views @inbounds for j = eachindex(psi_tod[1,:])
            pix_tod_jth_det = pix_tod[:,j]
            psi_tod_jth_det = psi_tod[:,j]
            @views @inbounds @simd for k = eachindex(psi_tod[:,1])
                t = time_array[k]
                Ωₖ = pix_tod_jth_det[k]
                ψₖ = psi_tod_jth_det[k]
                ϕ_hwp = mod2pi(ω_hwp*t)
                hit_map[Ωₖ] += 1
                D_μ[:, :, Ωₖ] .+= hmat3D(ψₖ, ϕ_hwp)
                @views @inbounds for _n_ in eachindex(spin_n)
                    @views @inbounds for _m_ in eachindex(spin_m)
                        hₙₘ[_n_, _m_, Ωₖ] += h(spin_n[_n_], spin_m[_m_], ψₖ, ϕ_hwp)
                    end
                end
            end
        end
        BEGIN = END
        next!(p)
    end
    @views for _n_ in eachindex(spin_n)
        @views for _m_ in eachindex(spin_m)
            hₙₘ[_n_,_m_,:] ./= hit_map
        end
    end
    
    @views for i in 1:3
        @views for j in 1:3
            D_μ[i,j,:] ./= hit_map
        end
    end 
    outmap = Dict{String, Array}(
        "hitmap" => hit_map,
        "h" => hₙₘ,
        )
    return outmap, D_μ
end


orientation_func_hwp(n, m, ψⱼ, ϕⱼ) = ℯ^(-im*(n*ψⱼ + m*ϕⱼ))

function hmat3D(ψⱼ, ϕⱼ)
    h = orientation_func_hwp
    M = @SMatrix [
        1.0                       (1/2)*h(-2, -4, ψⱼ, ϕⱼ)   (1/2)*h(2, 4, ψⱼ, ϕⱼ)  ;
        (1/2)*h(-2, -4, ψⱼ, ϕⱼ)   (1/4)*h(-4, -8, ψⱼ, ϕⱼ)    1/4                   ;
        (1/2)*h(2, 4, ψⱼ, ϕⱼ)      1/4                      (1/4)*h(4, 8, ψⱼ, ϕⱼ)
    ]
end

function abs_point(Ω, ϕ, ψ, Imap, Pmap, dI, dQ, dU, II::Falcons.InputInfo)
    ∂I = @views dI[3,Ω] - dI[2,Ω]im
    ∂P = @views dQ[3,Ω] + dU[2,Ω] - (dQ[2,Ω] - dU[3,Ω])im
    ∂̄P = @views dQ[3,Ω] - dU[2,Ω] + (dQ[2,Ω] + dU[3,Ω])im
    I = Imap[Ω]
    P = Pmap[Ω] 
    ρ = II.Systematics.Pointing.offset_ρ[1]
    χ = II.Systematics.Pointing.offset_χ[1]
    I1 = I - (ρ/2)*(ℯ^(im*(ψ+χ))*∂I + ℯ^(-im*(ψ+χ))*conj(∂I))
    P1 = (1/2) * (P*ℯ^(im*(2ψ+4ϕ))        - (ρ/2) * (ℯ^(im*(3ψ+4ϕ+χ))*∂P       + ℯ^(im*(ψ+4ϕ-χ))*∂̄P ))
    P2 = (1/2) * (conj(P)*ℯ^(-im*(2ψ+4ϕ)) - (ρ/2) * (ℯ^(im*(-ψ-4ϕ+χ))*conj(∂̄P) + ℯ^(im*(-3ψ-4ϕ-χ))*conj(∂P) ))
    return I1 + P1 + P2
end

function pointing_offset_hwp(h̃, hitmat, inputinfo, N, M)
    #h̃ = outmap["h"];
    n(i) = findall(x->x==i, N)[1]
    m(i) = findall(x->x==i, M)[1]
    
    Imap = inputinfo.Inputmap.i.pixels
    Pmap = inputinfo.Inputmap.q.pixels .+ (inputinfo.Inputmap.u.pixels)im
    npix = length(Imap)

    alm_I = hp.map2alm(Imap)
    alm_Q = hp.map2alm(real.(Pmap))
    alm_U = hp.map2alm(imag.(Pmap))
    dI = hp.alm2map_der1(alm_I, npix2nside(length(inputinfo.Inputmap.i.pixels)))
    dQ = hp.alm2map_der1(alm_Q, npix2nside(length(inputinfo.Inputmap.i.pixels)))
    dU = hp.alm2map_der1(alm_U, npix2nside(length(inputinfo.Inputmap.i.pixels)));
    
    ρ = inputinfo.Systematics.Pointing.offset_ρ[1]
    χ = inputinfo.Systematics.Pointing.offset_χ[1]
    ∂I = @. dI[3,:] - dI[2,:]im
    ∂P = @. dQ[3,:] + dU[2,:] - (dQ[2,:] - dU[3,:])im
    ∂̄P = @. dQ[3,:] - dU[2,:] + (dQ[2,:] + dU[3,:])im

    S_00 = Imap
    S_10 = @. -(ρ/2)*ℯ^(im*χ)*∂I
    S_14 = @. -(ρ/4)*ℯ^(-im*χ)*∂̄P
    S_24 = @. Pmap/2
    S_34 = @. -(ρ/4)*ℯ^(im*χ)*∂P

    ₊₀₀Sᵈ = @views h̃[n(3),m(4),:]   .*conj.(S_34) .+ h̃[n(2),m(4),:].*conj.(S_24)  .+ h̃[n(1),m(4),:].*conj.(S_14) .+ 
            h̃[n(1),m(0),:]   .*conj.(S_10) .+ S_00                         .+ h̃[n(-1),m(0),:].*S_10       .+   
            h̃[n(-1),m(-4),:] .*S_14        .+ h̃[n(-2),m(-4),:].*S_24       .+ h̃[n(-3),m(-4),:].*S_34

    ₊₂₄Sᵈ  = @views h̃[n(5),m(8),:] .*conj.(S_34) .+ h̃[n(4),m(8),:].*conj.(S_24)  .+ h̃[n(3),m(8),:] .*conj.(S_14) .+
             h̃[n(3),m(4),:] .*conj.(S_10) .+ h̃[n(2),m(4),:].*S_00         .+ h̃[n(1),m(4),:] .*S_10        .+
             h̃[n(1),m(0),:] .*S_14        .+ S_24                         .+ h̃[n(-1),m(0),:].*S_34

    ₋₂₋₄Sᵈ = @views h̃[n(1),m(0),:]   .*conj.(S_34) .+ conj.(S_24)                .+ h̃[n(-1),m(0),:] .*conj.(S_14) .+
             h̃[n(-1),m(-4),:] .*conj.(S_10) .+ h̃[n(-2),m(-4),:].*S_00     .+ h̃[n(-3),m(-4),:].*S_10        .+
             h̃[n(-3),m(-8),:] .*S_14        .+ h̃[n(-4),m(-8),:].*S_24     .+ h̃[n(-5),m(-8),:].*S_34;
    signal = zeros(Complex{Float64}, (3, 1, npix))
    maps = zeros(Complex{Float64}, (3, 1, npix))

    signal[1,1,:] .= @views ₊₀₀Sᵈ
    signal[2,1,:] .= @views ₋₂₋₄Sᵈ/2
    signal[3,1,:] .= @views ₊₂₄Sᵈ/2
    @inbounds @threads for j = eachindex(signal[1,1,:])
        maps[:, :, j] = hitmat[:, :, j] \ signal[:, :, j]
    end
    return mapbase = [real.(maps[1,1,:]) real.(maps[2,1,:]) imag.(maps[2,1,:])] |> transpose
end


