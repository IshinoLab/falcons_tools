include("/home/cmb/yusuket/scan/functions/spin_characterisation_w_HWP.jl")
include("/home/cmb/yusuket/scan/functions/calibration.jl")

idx = parse(Int64, ARGS[1])
step = 3
alpha_plus_beta = 95
alpha = step:step:alpha_plus_beta-step
T_alpha = logspace(log10(19.71*60), log10(24*60*60*365), 50)

FWHM = 17.9/60
hwprpm = 61
f_hwp = hwprpm/60
alpha_grid, T_prec_grid, T_spin_grid = gen_scan_parameter_space(alpha, T_alpha, FWHM, f_hwp, alpha_plus_beta)

nside = 256
ss = gen_ScanningStrategy()
ss.nside = nside
ss.sampling_rate = 19.0
ss.FP_theta = [0]
ss.FP_phi = [0]
ss.hwp_rpm = hwprpm
ss.alpha = vec(alpha_grid)[idx]
ss.beta = alpha_plus_beta - ss.alpha
ss.spin_rpm = period2rpm(T_spin_grid[idx], unit="sec")
ss.prec_rpm = period2rpm(T_prec_grid[idx], unit="sec")
ss.coord = "G"
alpha_grid, T_prec_grid, T_spin_grid = 0, 0, 0

day_of_observation = 0:365*3
hit_angular_distance = deg2rad(0.5)
start_observation_time = "2030-04-01T00:00:00"
start_obs_astroday = ap.time.Time(start_observation_time, scale="tdb")
ss.start_angle = get_start_angle(start_observation_time)

df = CSV.read("/home/cmb/yusuket/scan/functions/PCCS_broadband_pol_source_list.csv", DataFrame);

@time result = get_integration_time_and_attack_angle(ss, day_of_observation, df, hit_angular_distance);

save_dir = "/group/cmb/litebird/usr/ytakase/scan_optimisation/calibration/comp_source_obs/output_temp"
create_h5_file(save_dir, idx, "230321", result)
