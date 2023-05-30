include("/gpfs/home/cmb/yusuket/program/scan_strategy/optimisation2/functions/spin_characterisation_w_HWP.jl")

idx = parse(Int64, ARGS[1])
step = 3
alpha_plus_beta = 95
alpha = step:step:alpha_plus_beta-step
T_alpha = logspace(3.1, 7.2, 50)

FWHM = 17.9/60
hwprpm = parse(Int64, ARGS[2])
f_hwp = hwprpm/60

alpha_grid, T_prec_grid, T_spin_grid = gen_scan_parameter_space(alpha, T_alpha, FWHM, f_hwp, alpha_plus_beta)

nside = 256
ss = gen_ScanningStrategy()
ss.nside = nside
ss.sampling_rate = 19.0
ss.FP_theta = [0]
ss.FP_phi = [0]
ss.hwp_rpm = Float64(hwprpm)
ss.alpha = vec(alpha_grid)[idx]
ss.beta = alpha_plus_beta - ss.alpha
ss.spin_rpm = period2rpm(T_spin_grid[idx], unit="sec")
ss.prec_rpm = period2rpm(T_prec_grid[idx], unit="sec")
ss.coord = "G"
alpha_grid, T_prec_grid, T_spin_grid = 0, 0, 0

Args = Dict(
    "ss" => ss,
    "spin_n" => -5:5 |> collect ,
    "spin_m" => [-8,-4,0,4,8],
    "division" => 240*4,
    "fwhm" => deg2rad(FWHM),
    "rho" => deg2rad(1/60),
    "chi" => deg2rad(0.),
    "seed" => 1,
    "nreal" => 100,
    "sysfield" => pointing_offset_field,
    "lmax4likelihood" => 3nside-1,
    "sysmap_output" => false
);

result = run_systematics(Args)
save_path = "/home/cmb/yusuket/program/scan_strategy/optimisation2/systematics/output"
create_h5_file(save_path, idx, "230312_256_19Hz_1amin_$(hwprpm)rpm", result)
