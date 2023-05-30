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
ss.coord = "E"
alpha_grid, T_prec_grid, T_spin_grid = 0, 0, 0

division = 365
phi = 90
@time data = get_scaninfo_per_latitude(ss, division=division, phi=deg2rad(phi));
result = get_revisit_info(data);

save_dir = "/group/cmb/litebird/usr/ytakase/scan_optimisation/calibration/revisit/output"
create_h5_file(save_dir, idx, "230325_phi=$(phi)", result)