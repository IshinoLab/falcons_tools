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
using StatsBase
using CSV
using LinearAlgebra

hp = pyimport("healpy")
np = pyimport("numpy")
ap = pyimport("astropy")
apc = pyimport("astropy.coordinates")

mutable struct latitudinal_data
    latitude::Vector
    psi::Vector
    time::Vector
    ss::ScanningStrategy
end 

mutable struct revisit_output
    std_on_latitude::Vector
    mean_revisit::Number
    ss::ScanningStrategy
end

function get_revisit_info(data::latitudinal_data)
    resol = Resolution(data.ss.nside)
    std_of_revisit_on_latitude = [std(diff(data.time[i])) * get_numOfpixels_in_ring(resol, i) 
        for i in eachindex(data.latitude)]
    mean_revisit_std = sum(std_of_revisit_on_latitude)/resol.numOfPixels
    return revisit_output(std_of_revisit_on_latitude, mean_revisit_std, ss)
end


function get_latitudinal_position(resol::Resolution, phi::Number)
    theta = zeros(resol.numOfPixels)
    for i in 1:resol.numOfPixels
        ang = pix2angRing(resol, i)
        theta[i] = ang[1]
    end
    theta = unique(theta)
    if length(theta) != resol.nsideTimesFour-1
        @error "The length of theta array which have unique value is not 4nside-1"
    end
    pix = zeros(Int64, resol.nsideTimesFour-1)
    for i in eachindex(pix)
        pix[i] = ang2pixRing(resol, theta[i], phi)
    end
    return theta, pix
end

function get_numOfpixels_in_ring(resol::Resolution, ith_ring::Int)
    if 0 > ith_ring 
        @error "ith_ring must be positive integer."
    end
    if ith_ring > resol.nsideTimesFour-1 
        @error "ith_ring must be smaller than 4nside-1"
    end
    nside = resol.nside
    npix = resol.numOfPixels
    n = ith_ring - 1
    a1 = 4
    d = 4
    if ith_ring < nside + 1
        start = 0.5*n*(2*a1+(n-1)*d)+1
        stop  = 0.5*ith_ring*(2*a1+(ith_ring-1)*d)
    elseif ith_ring < 3nside+1
        n_2 = ith_ring - nside
        n = nside - 1
        start = 0.5*nside*(2*a1+(nside-1)*d) + 4*(n_2-1)*nside + 1
        stop = 0.5*nside*(2*a1+(nside-1)*d) + 4*(n_2)*nside 
    else
        n_2 = 4nside-1 - ith_ring
        start = npix - 0.5*(n_2+1)*(2*a1+(n_2)*d) + 1
        stop = npix - 0.5*(n_2)*(2*a1+(n_2-1)*d)
    end 
    return Int(stop - start + 1)
end

function get_scaninfo_per_latitude(SS::ScanningStrategy,; division::Int, phi=π/2)
    resol = Resolution(SS.nside)
    npix = nside2npix(SS.nside)

    chunk = Int(SS.duration / division)
    ω_hwp = rpm2angfreq(SS.hwp_rpm)
    #lat = deg2rad.(0:180)
    #target_pix = [ang2pixRing(resol, lat[i], π/2) for i in eachindex(lat)]
    target_theta, target_pixels = get_latitudinal_position(resol, phi)
    
    psi_db = [Float64[] for i in eachindex(target_pixels)]
    time_db = [Float64[] for i in eachindex(target_pixels)]

    BEGIN = 0
    p = Progress(division)
    @views @inbounds for i = 1:division
        END = i * chunk
        pix_tod, psi_tod, time_array = get_pointing_pixels(SS, BEGIN, END)
        @views @inbounds for j = eachindex(psi_tod[1,:])
            pix_tod_jth_det = pix_tod[:,j]
            psi_tod_jth_det = ifelse(ω_hwp == 0.0, -psi_tod[:,j], psi_tod[:,j])
            @views @inbounds for k = eachindex(psi_tod[:,1])
                t = time_array[k]
                ipix = pix_tod_jth_det[k] 
                psi = 2ω_hwp*t - psi_tod_jth_det[k]
                @views @inbounds for l in eachindex(target_pixels)
                    if ipix == target_pixels[l]
                        push!(psi_db[l], psi)
                        push!(time_db[l], t)
                    end
                end
            end
        end
        BEGIN = END
        next!(p)
    end
    return latitudinal_data(target_pixels, psi_db, time_db, ss)
end

function create_h5_file(dir::AbstractString, n::Int, name::AbstractString, result::revisit_output)
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
        write(file, "std_on_latitude", result.std_on_latitude)
        write(file, "mean_revisit", result.mean_revisit)
        for f in propertynames(result.ss)
            write(file, "ss/$f", getfield(result.ss, f))
        end
    end 
end


#===   Compact sources observation   ===#

function diff_ang(A,a)
    x = zeros(size(A)[1])
    for i in eachindex(x)
        val = A[i,:] ⋅ a / (norm(A[i,:]) * norm(a))
        if val > 1
            val = 1
        end
        x[i] = acos(val)
    end
    x
end

function ang2xyz(theta, phi)
    xyz = zeros(length(theta), 3)
    for i in eachindex(theta)
        xyz[i,:] .= ang2vec(theta[i], phi[i])
    end
    xyz
end

py"""
def planet_pos(planet, start, duration, S):
    import astropy.time, astropy.units as u
    import numpy as np
    import healpy as hp
    from astropy.time import Time
    from astropy.coordinates import (
        ICRS,
        get_body_barycentric,
        BarycentricMeanEcliptic,
        solar_system_ephemeris,
    )
    
    t_width = ["2030-01-01T00:00:00", "2030-01-01T00:00:01"]
    
    t = Time([start], scale="tdb")
    T = Time(t_width, scale="tdb")
    one_sec = T[1] - T[0]
    
    time = t[0] + one_sec * np.linspace(0, duration, (duration * S) + 1 )
    
    solar_system_ephemeris.set("builtin")
    icrs_pos = get_body_barycentric(
        planet,
        time,
    )

    ecl_vec = (ICRS(icrs_pos)
        .transform_to(BarycentricMeanEcliptic())
        .cartesian
        .get_xyz()
        .value
    )
    ecl_vec /= np.linalg.norm(ecl_vec, axis=0, ord=2)
    ecl_vec = ecl_vec.transpose()

    return time, ecl_vec
"""

function get_start_angle(time_origin)
    starting_time, anti_sun_vec = py"planet_pos"("earth", time_origin, 0, 1)
    t_day = ap.time.Time(["2030-04-01T00:00:00", "2030-04-02T00:00:00"], scale="tdb")
    astro_day = t_day[2] - t_day[1]
    earth_theta, earth_phi = vec2ang(anti_sun_vec[1], anti_sun_vec[2], anti_sun_vec[3])
    return earth_phi
end

@. garactic2healpix(x) = 90 - x

function get_planet_posision(planet::String, start_observation_time::String, nday::String)
    path = "/group/cmb/litebird/usr/ytakase/planets_tod/$(planet)/$(start_observation_time)/$(nday).npz"
    data = np.load(path, allow_pickle=true)
    planet_vec = get(data, :planet_vec)
end

function get_observation_to_compact_sources(pointing_xyz, glat, glon, radius)
    theta_pccs = glat |> garactic2healpix .|> deg2rad
    phi_pccs = glon .|> deg2rad
    pccs_xyz = ang2vec(theta_pccs, phi_pccs)
    ang = diff_ang(pointing_xyz, pccs_xyz)
    idx = findall(x -> x .<= radius, ang)
    return idx
end

function get_planet_hit_angle(pointing_xyz, planet_vec, sampling_rate)
    ang = zeros(length(pointing_xyz[:,1]))
    j = 1
    #println(length(pla_pos[:,1,1]))
    @views @inbounds for i in eachindex(ang)
        cosk = pointing_xyz[i,:]⋅planet_vec[j,:,1]/(norm(pointing_xyz[i,:])*norm(planet_vec[j,:,1]))
        ang[i] = acos(cosk)
        if i%sampling_rate == 0
            j += 1
        end
    end
    return ang
end

function ecliptic2galactic_vec(pointing_xyz::Matrix)
    galactic = zeros(size(pointing_xyz))
    for i in eachindex(pointing_xyz[:,1])
        galactic[i,:] = ecliptic2galactic(pointing_xyz[i,:])
    end
    galactic
end

function get_observation_to_compact_sources(planet::String, pointing_xyz, start_observation_time, nday::String, radius)
    planet_vec = get_planet_posision(planet, start_observation_time, nday)
    planet_vec = ecliptic2galactic_vec(planet_vec)
    ang = get_planet_hit_angle(pointing_xyz, planet_vec, ss.sampling_rate)
    idx = findall(x -> x .<= radius, ang)
end

mutable struct planets
    mars
    jupiter
    saturn
    uranus
    neptune
end

function get_visit_time_and_attack_angle_to_compact_sources(ss, idxOfday, df, hit_angular_distance)
    planet_list = ["mars", "jupiter", "uranus", "saturn", "neptune"]
    day = 24*60*60
    t_day = ap.time.Time(["2030-01-01T00:00:00", "2030-01-02T00:00:00"], scale="tdb")
    astro_day = t_day[2] - t_day[1]

    visit_time_pccs = [Float32[] for i in eachindex(df.NAME)]
    visit_time_planets = [Float32[] for i in eachindex(planet_list)]
    attack_angle_pccs = [Float32[] for i in eachindex(df.NAME)]
    attack_angle_planets = [Float32[] for i in eachindex(planet_list)]

    for n in eachindex(idxOfday)
        nday = start_obs_astroday + n * astro_day
        #println("Days : ", nday.value)
        theta, phi, psi, time = get_pointings_tuple(ss, n * day, (n + 1) * day)
        pointing_xyz = ang2xyz(theta, phi)
        for j in eachindex(df.NAME)
            idx_pccs = get_observation_to_compact_sources(pointing_xyz, df.GLAT[j], df.GLON[j], hit_angular_distance);
            for i in eachindex(idx_pccs)
                push!(visit_time_pccs[j], time[idx_pccs[i]])
                push!(attack_angle_pccs[j], psi[idx_pccs[i]])
            end
        end
        for j in eachindex(planet_list)
            idx_planets = get_observation_to_compact_sources(planet_list[j], pointing_xyz, start_observation_time, nday.value, hit_angular_distance)
            for i in eachindex(idx_planets)
                push!(visit_time_planets[j], time[idx_planets[i]])
                push!(attack_angle_planets[j], psi[idx_planets[i]])
            end
        end
    end
    visit_planets = planets(visit_time_planets[1], visit_time_planets[2], visit_time_planets[3], visit_time_planets[4], visit_time_planets[5])
    attack_planets = planets(attack_angle_planets[1], attack_angle_planets[2], attack_angle_planets[3], attack_angle_planets[4], attack_angle_planets[5])
    return visit_time_pccs, visit_planets, attack_angle_pccs, attack_planets
end

function get_integration_time(ss::ScanningStrategy, visit_time::planets)
    integration_time_mars = length(visit_time.mars)/ss.sampling_rate
    integration_time_jupiter = length(visit_time.jupiter)/ss.sampling_rate
    integration_time_saturn = length(visit_time.saturn)/ss.sampling_rate
    integration_time_uranus = length(visit_time.uranus)/ss.sampling_rate
    integration_time_neptune = length(visit_time.neptune)/ss.sampling_rate
    return planets(integration_time_mars, integration_time_jupiter, integration_time_saturn, integration_time_uranus, integration_time_neptune)
end

function get_integration_time(ss::ScanningStrategy, visit_time::Vector)
    integration_times = zeros(length(visit_time))
    for i in eachindex(integration_times)
       integration_times[i] = length(visit_time[i])/ss.sampling_rate
    end
    return integration_times
end

function get_attack_angle_quantify(attack_angle)
    range_of_attack_angle = -180:180
    hist = fit(Histogram, rad2deg.(attack_angle), range_of_attack_angle)
    std_of_attack_angle = std(hist.weights)
    uniformity = 1. - length(hist.weights[hist.weights .== 0])/length(range_of_attack_angle)
    return std_of_attack_angle, uniformity
end

mutable struct quantify
    SD
    uniformity
end

function get_attack_angle_to_compact_sources_quantify(ss, attack_angle::planets)
    std_of_attack_angle_mars, uniformity_mars = get_attack_angle_quantify(attack_angle.mars)
    std_of_attack_angle_jupiter, uniformity_jupiter = get_attack_angle_quantify(attack_angle.jupiter)
    std_of_attack_angle_saturn, uniformity_saturn = get_attack_angle_quantify(attack_angle.saturn)
    std_of_attack_angle_uranus, uniformity_uranus = get_attack_angle_quantify(attack_angle.uranus)
    std_of_attack_angle_neptune, uniformity_neptune = get_attack_angle_quantify(attack_angle.neptune)
    std_of_attack_angles = planets(
        std_of_attack_angle_mars, 
        std_of_attack_angle_jupiter, 
        std_of_attack_angle_saturn, 
        std_of_attack_angle_uranus, 
        std_of_attack_angle_neptune)
    uniformities = planets(uniformity_mars, uniformity_jupiter, uniformity_saturn, uniformity_uranus, uniformity_neptune)
    return quantify(std_of_attack_angles, uniformities)
end

function get_attack_angle_to_compact_sources_quantify(ss, attack_angle::Vector)
    std_of_attack_angle = zeros(length(attack_angle))
    uniformities = zeros(length(attack_angle))
    for i in eachindex(attack_angle)
       std_of_attack_angle[i], uniformities[i] = get_attack_angle_quantify(attack_angle[i])
    end
    return quantify(std_of_attack_angle, uniformities)
end

@inline function vec2ang_ver2(x, y, z)
    norm = sqrt(x^2 + y^2 + z^2)
    theta = acos(z / norm)
    phi = atan(y, x)
    phi = ifelse(phi > π, π-phi, phi)
    return (theta, phi)
end

function vec2pix(res::Resolution, x,y,z)
    ang = vec2ang_ver2(x,y,z)
    pix = ang2pixRing(res, ang[1], ang[2])
end

mutable struct compact_sources
    Planets
    PCCS
end

mutable struct calibration_output
    integration_times
    attack_angles
    ss::ScanningStrategy
end

function get_integration_time_and_attack_angle(ss, idxOfday, df, hit_angular_distance)
    if ss.coord == "E"
        @error "ss.coord must be 'E'"
    end
    visit_time_pccs, visit_time_planets, attack_angle_pccs, attack_angle_planets = get_visit_time_and_attack_angle_to_compact_sources(ss, idxOfday, df, hit_angular_distance)
    
    integ_time_planets       = get_integration_time(ss, visit_time_planets)
    integ_time_pccs          = get_integration_time(ss, visit_time_pccs)
    attack_angle_info_planets = get_attack_angle_to_compact_sources_quantify(ss, attack_angle_planets)
    attack_angle_info_pccs    = get_attack_angle_to_compact_sources_quantify(ss, attack_angle_pccs)
    
    integration_time = compact_sources(integ_time_planets, integ_time_pccs)
    attack_angle = compact_sources(attack_angle_info_planets, attack_angle_info_pccs)
    return calibration_output(integration_time, attack_angle, ss)
end

function create_h5_file(dir::AbstractString, n::Int, name::AbstractString, result::calibration_output)
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
        write(file, "attack_angle/PCCS/std", result.attack_angles.PCCS.SD)
        write(file, "attack_angle/PCCS/uniformity", result.attack_angles.PCCS.uniformity)
        write(file, "integration_time/PCCS", result.integration_times.PCCS)
        
        for planet in propertynames(result.attack_angles.Planets.SD)
            write(file, "attack_angle/planets/std/$(planet)", getfield(result.attack_angles.Planets.SD, planet))
            write(file, "attack_angle/planets/uniformity/$(planet)", getfield(result.attack_angles.Planets.uniformity, planet))
            write(file, "integration_time/planets/$(planet)", getfield(result.integration_times.Planets, planet))
        end
        
        for f in propertynames(result.ss)
            write(file, "ss/$f", getfield(result.ss, f))
        end
    end 
end

mutable struct half_sky_coverage
    t
    coverage
    covertime
    hitmap
    ss::ScanningStrategy
end

#=
function get_time4half_sky_coverage(SS::ScanningStrategy,; division::Int)
    h = orientation_func_hwp
    resol = Resolution(SS.nside)
    npix = nside2npix(SS.nside)
    chunk = Int(SS.duration / division)
    ω_hwp = rpm2angfreq(SS.hwp_rpm)
    hitmap = zeros(Int64, npix)
    BEGIN = 0
    p = Progress(division)
    coverage_array = zeros(division)
    chunk_time = zeros(division)
    covertime = 0
    @views @inbounds for i = 1:division
        END = i * chunk
        coverage_chunk = sum(sign.(hitmap))/resol.numOfPixels
        coverage_array[i] = coverage_chunk
        chunk_time[i] = BEGIN
        if coverage_chunk > 0.5 
            covertime = chunk_time[findmax(coverage_array)[2]]
           return half_sky_coverage(chunk_time, coverage_array, covertime, hitmap, SS)
        end
        pix_tod, psi_tod, time_array = get_pointing_pixels(SS, BEGIN, END)
        @views @inbounds for j = eachindex(psi_tod[1,:])
            pix_tod_jth_det = pix_tod[:,j]
            psi_tod_jth_det = psi_tod[:,j]
            @views @inbounds for k = eachindex(psi_tod[:,1])
                t = time_array[k]
                coverage = sum(hitmap)/resol.numOfPixels
                Ωₖ = pix_tod_jth_det[k]
                ψₖ = psi_tod_jth_det[k]
                ϕ_hwp = mod2pi(ω_hwp*t)
                hitmap[Ωₖ] += 1
            end
        end
        BEGIN = END
        next!(p)
    end
    covertime = chunk_time[findmax(coverage_array)[2]]
    return half_sky_coverage(chunk_time, coverage_array, covertime, hitmap, SS)
end
=#

function get_time4half_sky_coverage(SS::ScanningStrategy,; division::Int)
    h = orientation_func_hwp
    resol = Resolution(SS.nside)
    npix = nside2npix(SS.nside)
    chunk = Int(SS.duration / division)
    ω_hwp = rpm2angfreq(SS.hwp_rpm)
    hitmap = zeros(Int64, npix)
    coverage_map = zeros(Int64, npix)
    BEGIN = 0
    p = Progress(division)
    coverage_array = zeros(division)
    chunk_time = zeros(division)
    covertime = 0
    @views @inbounds for i = 1:division
        END = i * chunk
        coverage_chunk = sum(coverage_map)/resol.numOfPixels
        coverage_array[i] = coverage_chunk
        chunk_time[i] = BEGIN
        pix_tod, psi_tod, time_array = get_pointing_pixels(SS, BEGIN, END)
        @views @inbounds for j = eachindex(psi_tod[1,:])
            pix_tod_jth_det = pix_tod[:,j]
            psi_tod_jth_det = psi_tod[:,j]
            @views @inbounds for k = eachindex(psi_tod[:,1])
                t = time_array[k]
                Ωₖ = pix_tod_jth_det[k]
                ψₖ = psi_tod_jth_det[k]
                ϕ_hwp = mod2pi(ω_hwp*t)
                hitmap[Ωₖ] += 1
                coverage_map[Ωₖ] = 1
                coverage = sum(coverage_map)/resol.numOfPixels
                
                if coverage > 0.5 
                    covertime = t #chunk_time[findmax(coverage_array)[2]]
                   return half_sky_coverage(chunk_time, coverage_array, covertime, hitmap, SS)
                end
            end
        end
        BEGIN = END
        next!(p)
    end
    covertime = chunk_time[findmax(coverage_array)[2]]
    return half_sky_coverage(chunk_time, coverage_array, covertime, hitmap, SS)
end

function create_h5_file(dir::AbstractString, n::Int, name::AbstractString, result::half_sky_coverage)
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
        write(file, "t", result.t)
        write(file, "coverage", result.coverage)
        write(file, "covertime", result.covertime)
        #write(file, "hitmap", result.hitmap)
        
        for f in propertynames(result.ss)
            write(file, "ss/$f", getfield(result.ss, f))
        end
    end 
end