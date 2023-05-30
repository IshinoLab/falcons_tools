using Falcons
using Healpix
using Base.Threads
using NPZ
using Statistics
using StaticArrays
using ProgressMeter
using DataFrames
using StatsBase
using PyCall
using Base.Threads
using NaNStatistics
using HDF5

hp = pyimport("healpy")
np = pyimport("numpy")
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

function mapmaking_pointing(SS::ScanningStrategy,; division::Int, InputInfo::Falcons.InputInfo)
    II = InputInfo
    Imap = @views II.Inputmap.i.pixels
    Pmap = @views II.Inputmap.q.pixels .+ (II.Inputmap.u.pixels)im
    alm_I = hp.map2alm(Imap)
    alm_Q = hp.map2alm(real.(Pmap))
    alm_U = hp.map2alm(imag.(Pmap))
    dI = hp.alm2map_der1(alm_I, ss.nside)
    dQ = hp.alm2map_der1(alm_Q, ss.nside)
    dU = hp.alm2map_der1(alm_U, ss.nside)
    
    resol = Resolution(SS.nside)
    npix = resol.numOfPixels
    
    month = Int(SS.duration / division)
    ω_hwp = rpm2angfreq(SS.hwp_rpm)
    d_μp_μ = zeros(Complex{Float64}, (3, 1, npix))
    hit_map = zeros(npix)
    D_μ = zeros(Complex{Float64}, (3, 3, npix))
    outmap = zeros(Complex{Float64}, (3, 1, npix))
    
    BEGIN = 0
    p = Progress(division)
    @inbounds @views for i = 1:division
        END = i * month
        theta_tod, phi_tod, psi_tod, time_array = get_pointings_tuple(SS, BEGIN, END)
        @views @inbounds for j = eachindex(psi_tod[1,:])
            theta_tod_jth_det = theta_tod[:,j]
            phi_tod_jth_det = phi_tod[:,j]
            psi_tod_jth_det = psi_tod[:,j]
            @inbounds @views for k = eachindex(time_array)
                t = time_array[k]
                θₖ = theta_tod_jth_det[k]
                ϕₖ = phi_tod_jth_det[k]
                ψₖ = psi_tod_jth_det[k]
                ϕ_hwp = mod2pi(ω_hwp*t)
                Ωₖ = ang2pixRing(resol, θₖ, ϕₖ)
                hit_map[Ωₖ] += 1
                
                D_μ[:, :, Ωₖ] .+= hmat3D(ψₖ, ϕ_hwp)
                dₖ = abs_point(Ωₖ, ϕ_hwp, ψₖ, Imap, Pmap, dI, dQ, dU, II)
                d_μp_μ[:, :, Ωₖ] .+= @SMatrix [dₖ; (1/2)*dₖ*ℯ^(im*(2ψₖ+4ϕ_hwp)); (1/2)*dₖ*ℯ^(-im*(2ψₖ+4ϕ_hwp))]
            end
        end
        BEGIN = END
        next!(p)
    end
    @views for i in 1:3
        d_μp_μ[i,1,:] ./= hit_map
        @views for j in 1:3
            D_μ[i,j,:] ./= hit_map
        end
    end
    
    @inbounds @threads for j = 1:npix
        outmap[:, :, j] = D_μ[:, :, j] \ d_μp_μ[:, :, j]
    end
    
    outmap = @views [real.(outmap[1,1,:]) real.(outmap[2,1,:]) imag.(outmap[2,1,:])] |> transpose #|> convert_maps
    return outmap, hit_map, d_μp_μ, D_μ
end

mutable struct scanfield{I<:Integer}
    hitmap::AbstractArray{I,1}
    h::Array
    quantify::DataFrame
    n::AbstractArray{I,1}
    m::AbstractArray{I,1}
    ss::ScanningStrategy
end

function get_scanfield(SS::ScanningStrategy,; division::Int, spin_n::AA, spin_m::AA) where {AA<:AbstractArray}
    h = orientation_func_hwp
    resol = Resolution(SS.nside)
    npix = nside2npix(SS.nside)
    chunk = Int(SS.duration / division)
    ω_hwp = rpm2angfreq(SS.hwp_rpm)
    hitmap = zeros(Int64, npix)
    hₙₘ = zeros(Complex{Float32}, (length(spin_n), length(spin_m), npix))
    BEGIN = 0
    p = Progress(division)
    @views @inbounds for i = 1:division
        END = i * chunk
        pix_tod, psi_tod, time_array = get_pointing_pixels(SS, BEGIN, END)
        @views @inbounds for j = eachindex(psi_tod[1,:])
            pix_tod_jth_det = pix_tod[:,j]
            psi_tod_jth_det = psi_tod[:,j]
            @views @inbounds @simd for k = eachindex(psi_tod[:,1])
                t = time_array[k]
                Ωₖ = pix_tod_jth_det[k]
                ψₖ = psi_tod_jth_det[k]
                ϕ_hwp = mod2pi(ω_hwp*t)
                hitmap[Ωₖ] += 1
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
            hₙₘ[_n_,_m_,:] ./= hitmap
        end
    end
    df = get_hnm_quantify(hₙₘ, spin_n, spin_m)
    return scanfield(hitmap, hₙₘ, df, spin_n, spin_m, ss)
end

function get_hnm_quantify(hₙₘ, spin_n, spin_m)
    df = DataFrame(n = repeat(spin_n, inner=length(spin_m)), 
                   m = repeat(spin_m, outer=length(spin_n)),
                   mean = zeros(length(spin_n)*length(spin_m)),
                   std = zeros(length(spin_n)*length(spin_m)),
    )
    for _n_ in spin_n
        for _m_ in spin_m
            row = findfirst((r -> r.n == _n_ && r.m == _m_), eachrow(df))
            abs2_field = abs2.(h_nm(hₙₘ, spin_n, spin_m, _n_, _m_))
            df[row, :mean] = mean(abs2_field)
            df[row, :std] = std(abs2_field)
        end
    end
    return df
end

function h_nm(field::scanfield, n::Int,m::Int)
    fn(i) = findall(x->x==i, field.n)[1]
    fm(i) = findall(x->x==i, field.m)[1]
    return field.h[fn(n), fm(m),:]
end

function h_nm(hₙₘ::Array, spin_n::Vector, spin_m::Vector, n::Int, m::Int)
    fn(i) = findall(x->x==i, spin_n)[1]
    fm(i) = findall(x->x==i, spin_m)[1]
    return hₙₘ[fn(n), fm(m),:]
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


function scanfield2hitmatrix(field::scanfield)
    res = Resolution(field.ss.nside)
    npix = res.numOfPixels
    hitmatrix = zeros(Complex{Float32}, (3, 3, npix))
    hitmatrix[1,1,:] .= ones(npix)
    hitmatrix[1,2,:] .= 1/2 .* h_nm(field, -2, -4)
    hitmatrix[1,3,:] .= 1/2 .* h_nm(field, 2, 4)
    
    hitmatrix[2,1,:] .= 1/2 .* h_nm(field, -2, -4)
    hitmatrix[2,2,:] .= 1/4 .* h_nm(field, -4, -8)
    hitmatrix[2,3,:] .= 1/4 .* ones(npix)
    
    hitmatrix[3,1,:] .= 1/2 .* h_nm(field, 2, 4) 
    hitmatrix[3,2,:] .= 1/4 .* ones(npix)
    hitmatrix[3,3,:] .= 1/4 .* h_nm(field, 4, 8)
    return hitmatrix
end
#=
function scanfield2hitmatrix(field::scanfield, ipix::Int)
    hitmatrix = Symmetric( @SMatrix [1.0                              0.5 .* h_nm(field, -2, -4)[ipix]  0.5 .* h_nm(field, 2, 4)[ipix]
                 0.5 .* h_nm(field, -2, -4)[ipix] 0.25 .* h_nm(field, -4, -8)[ipix] 0.25
                 0.5 .* h_nm(field, 2, 4)[ipix]   0.25                              0.25 .* h_nm(field, 4, 8)[ipix]
        ])
    return  hitmatrix
end
=#
function pointing_offset_field(field::scanfield, inputinfo::Falcons.InputInfo)
    Imap = inputinfo.Inputmap.i.pixels
    Pmap = inputinfo.Inputmap.q.pixels .+ (inputinfo.Inputmap.u.pixels)im
    npix = nside2npix(field.ss.nside)

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

    ₙₘS = [conj.(S_34), conj.(S_24), conj.(S_14), conj.(S_10), S_00, S_10, S_14, S_24, S_34]
    
    ₊₀₋ₙ₀₋ₘh̃ = [h_nm(field,3,4),  h_nm(field,2,4),       h_nm(field,1,4), 
                h_nm(field,1,0),   ones(ComplexF32,npix), h_nm(field,-1,0), 
                h_nm(field,-1,-4), h_nm(field,-2,-4),     h_nm(field,-3,-4)]
    
    ₋₂₋ₙ₋₄₋ₘh̃ = [h_nm(field,1,0),  ones(ComplexF32,npix),  h_nm(field,-1,0), 
                h_nm(field,-1,-4), h_nm(field,-2,-4),      h_nm(field,-3,-4), 
                h_nm(field,-3,-8), h_nm(field,-4,-8),      h_nm(field,-5,-8)]
    
    ₊₂₋ₙ₊₄₋ₘh̃ = [h_nm(field,5,8),  h_nm(field,4,8),       h_nm(field,3,8), 
                h_nm(field,3,4),   h_nm(field,2,4),       h_nm(field,1,4), 
                h_nm(field,1,0),   ones(ComplexF32,npix), h_nm(field,-1,0)]
    
    ₊₀₀Sᵈ  = [₊₀₋ₙ₀₋ₘh̃[i]  .* ₙₘS[i] for i in eachindex(ₙₘS)] |> sum
    ₋₂₋₄Sᵈ = [₋₂₋ₙ₋₄₋ₘh̃[i] .* ₙₘS[i] for i in eachindex(ₙₘS)] |> sum
    ₊₂₄Sᵈ  = [₊₂₋ₙ₊₄₋ₘh̃[i] .* ₙₘS[i] for i in eachindex(ₙₘS)] |> sum

    signal = zeros(Complex{Float32}, (3, 1, npix))
    maps = zeros(Complex{Float32}, (3, 1, npix))

    signal[1,1,:] .= @views ₊₀₀Sᵈ
    signal[2,1,:] .= @views ₋₂₋₄Sᵈ/2
    signal[3,1,:] .= @views ₊₂₄Sᵈ/2
    
    hitmatrix = scanfield2hitmatrix(field)
    
    @inbounds @threads for j = eachindex(maps[1,1,:])
        maps[:, :, j] = hitmatrix[:, :, j] \ signal[:, :, j]
    end
    return mapbase = [real.(maps[1,1,:]) real.(maps[2,1,:]) imag.(maps[2,1,:])] |> transpose
end


function healpy2julia(maps::PolarizedHealpixMap)
    [maps.i.pixels maps.q.pixels maps.u.pixels] |> transpose
end

function gen_input(args)
    nside = args["ss"].nside
    lmax = 3*nside - 1
    fwhm = args["fwhm"]
    g_off = 0
    rho = args["rho"]
    chi_deg=args["chi"]
    seed = args["seed"]
    inputmap_path = ""
    fwhm_arcmin = rad2deg(fwhm)*60
    if haskey(args, "inputmap_path")
        inputmap_path = args["inputmap_path"]
        origin_map = hp.read_map(inputmap_path, field=(0,1,2))
        origin_map = hp.smoothing(origin_map, fwhm=fwhm)
        println("Input map is loaded from requested path: \n$(inputmap_path)")
        println("Smoothed by gaussian beam (fwhm=$fwhm_arcmin)")
    end
    if inputmap_path == ""
        input_powerspectrum_path = "/home/cmb/yusuket/program/scan_strategy/optimization/alpha_beta_2D/spin_char/simple_model_spectrum.npz"
        power_spectrum_data = npzread(input_powerspectrum_path)
        np.random.seed(seed)
        origin_map = hp.synfast(power_spectrum_data["total"] |> transpose, nside, new=true, fwhm=fwhm)
        println("Input map is generated by model of power spectrum: \n$(input_powerspectrum_path)")
        println("Smoothed by gaussian beam (fwhm=$fwhm_arcmin)")
    end
    inputinfo = set_input(inputmap=origin_map)
    inputinfo.Systematics.Gain.offset = [g_off]
    inputinfo.Systematics.Pointing.offset_ρ = [rho]
    inputinfo.Systematics.Pointing.offset_χ = [chi_deg]
    return inputinfo
end

function forecasting(lmax, cl_obs,;
        path="/home/cmb/yusuket/program/MapData/CAMB/ClBB_PTEPini.npz", 
        rmin=1e-8,
        rmax=1e-1,
        rresol=1e5 |> Int,
        iter=3,
        verbose=false
    )
    #= `Cl_obs requwire B-mode power spectrum due to the systematics`=#
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
        @views @inbounds for i in eachindex(gridOfr)
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

mutable struct systematics_output{}
    delta_r
    gridOfr
    likelihood
    field::scanfield
    cl_sys#::AbstractArrays{T,6}
    sysmap#::
    args#::
end
    
function estimate_systematics(field::scanfield, sysfield, args,; verbose=false)
    if minimum(field.hitmap)>3
        @show minimum(field.hitmap)
        lmax = 3*field.ss.nside
        npix = nside2npix(field.ss.nside)
        residual_maps = zeros(3, npix)
        cl_sys = zeros(6, lmax)
        nreal = args["nreal"]
        lmax4likelihood = args["lmax4likelihood"]
        sysmap_out = false
        if args["sysmap_output"]
            sysmap_out = true
            sysmaps = zeros(nreal, 3, npix)
            cl_sys_tot = zeros(nreal, 6, lmax)
        end

        for i in 1:nreal
            args["seed"] = i
            if verbose == true
                print_dict(args)
            end
            inputinfo = gen_input(args)
            maps = sysfield(field, inputinfo)
            residual_maps = maps .- healpy2julia(inputinfo.Inputmap)
            
            cl_i = hp.anafast(residual_maps)
            cl_sys .+= cl_i
            if sysmap_out == true
                sysmaps[i, :,:] .= residual_maps
                cl_sys_tot[i,:,:] .= cl_i
            end
        end
        cl_sys .= cl_sys./nreal 
        delta_r, gridOfr, likelihood = forecasting(lmax4likelihood, cl_sys[3,:])
        if sysmap_out == true
            result = systematics_output(delta_r, gridOfr, likelihood, field, cl_sys_tot, sysmaps, args)
        else
            result = systematics_output(delta_r, gridOfr, likelihood, field, cl_sys, residual_maps, args)
        end
    else
        @warn "Since minimum value of hitmap is less than 3, the Stokes parameter is not estimated."
        result = systematics_output(NaN, [0], [0], field, [0], [0], args)
    end
    return result
end

function print_dict(d::Dict)
    for (key, value) in d
        println("$key => $value")
    end
end

function run_systematics(args)
    print_dict(args)
    println("")
    show_ss(args["ss"])
    flush(stdout)
    
    @time field = get_scanfield(args["ss"], division=args["division"], spin_n=args["spin_n"], spin_m=args["spin_m"]);
    println("Size of field.h : ", sizeof(field.h)/1e9, "GB")
    println("run `estimate_systematics`")
    
    @time result = estimate_systematics(field, args["sysfield"], args, verbose=false);
    println("======== Finish ========")
    return result
end

function create_h5_file(dir::AbstractString, n::Int, name::AbstractString, result::systematics_output)
    # "./name"ディレクトリのパスを作成
    dir_path = joinpath(dir, name)
    
    # "./name"ディレクトリが存在しない場合は作成
    if !isdir(dir_path)
        mkdir(dir_path)
    end
    
    # "./name/n.h5"のパスを作成
    h5_path = joinpath(dir_path, "output_$n.h5")
    ss_field_names = [String(Symbol(f)) for f in propertynames(ss)]    
    # "./name/n.h5"のファイルを作成
    h5open(h5_path, "w") do file
        write(file, "args/spin_n", result.args["spin_n"])
        write(file, "args/spin_m", result.args["spin_m"])
        write(file, "args/lmax4likelihood", result.args["lmax4likelihood"])
        write(file, "args/rho", result.args["rho"])
        write(file, "args/chi", result.args["chi"])
        write(file, "delta_r", result.delta_r)
        for f in propertynames(result.args["ss"])
            write(file, "args/ss/$f", getfield(result.args["ss"], f))
        end
        write(file, "quantify/hitmap_std", std(result.field.hitmap))
        write(file, "quantify/n", result.field.quantify.n) 
        write(file, "quantify/m", result.field.quantify.m) 
        write(file, "quantify/mean", result.field.quantify.mean) 
        write(file, "quantify/std", result.field.quantify.std) 
        if minimum(result.field.hitmap)<3
            @warn "The minimum of hitmap is less than 3 hits. The hitmap will be saved."
            write(file, "hitmap", Int32.(result.field.hitmap))
        end
    end 
end

v(α, β, ω_α, ω_β) = rad2deg( ω_α*sind(α+β) + ω_β*sind(β) )
logspace(a, b, c) = 10 .^(range(a,stop=b,length=c))
T_beta_lim(alpha, beta, T_alpha, dtheta, f_hwp) = 2π*T_alpha*sind(beta)/(dtheta*f_hwp*T_alpha-2π*sind(alpha+beta))
T_beta_lim(alpha, beta, T_alpha, dtheta, f_hwp, N) = 2π*N*T_alpha*sind(beta)/(dtheta*f_hwp*T_alpha-2π*N*sind(alpha+beta))

function gen_scan_parameter_space(alpha, T_alpha, FWHM, f_hwp, apb)
    dtheta = deg2rad(FWHM)
    beta = apb .- alpha
    alpha_grid = zeros(length(T_alpha), length(alpha))
    T_alpha_grid = zeros(length(T_alpha), length(alpha))
    T_spin_grid = zeros(length(T_alpha), length(alpha))
    for i in eachindex(T_alpha)
        for j in eachindex(alpha)
            alpha_grid[i,j] = alpha[j]
            T_alpha_grid[i,j] = T_alpha[i]
            T_spin_grid[i,j] = T_beta_lim(alpha[j], beta[j], T_alpha[i], dtheta, f_hwp)
        end
    end
    return  alpha_grid, T_alpha_grid, T_spin_grid
end

function gen_scan_parameter_space(alpha, T_alpha, FWHM, f_hwp, apb, N)
    #= N is number of rotation of HWP within the beam size =#
    dtheta = deg2rad(FWHM)
    beta = apb .- alpha
    alpha_grid = zeros(length(T_alpha), length(alpha))
    T_alpha_grid = zeros(length(T_alpha), length(alpha))
    T_spin_grid = zeros(length(T_alpha), length(alpha))
    for i in eachindex(T_alpha)
        for j in eachindex(alpha)
            alpha_grid[i,j] = alpha[j]
            T_alpha_grid[i,j] = T_alpha[i]
            T_spin_grid[i,j] = T_beta_lim(alpha[j], beta[j], T_alpha[i], dtheta, f_hwp, N)
        end
    end
    return  alpha_grid, T_alpha_grid, T_spin_grid
end
