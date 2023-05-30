#=
alpha = [45,47.5,50]
T_alpha = [1.0, 192.348/60, 10, 100] .* 3600 # sec
に固定して、T_betaをT_beta^lower < T_beta < T_alphaまでnoutput分割してxlink, delta_r を計算する。
=#
include("/gpfs/home/cmb/yusuket/program/scan_strategy/optimisation2/functions/spin_characterisation_w_HWP.jl")
T_beta_grid(a,b,c) = range(a,stop=b,length=c)

noutput = 100
idx = parse(Int64, ARGS[1])
alpha = parse(Float64, ARGS[2])
T_alpha = parse(Float64, ARGS[3]) #sec
hwprpm = parse(Float64, ARGS[4])
num_of_HWP_rotation = parse(Float64, ARGS[5])
alpha_plus_beta = 95
beta = alpha_plus_beta - alpha

FWHM = 17.9 / 60.
f_hwp = hwprpm/60.
dtheta = deg2rad(FWHM)
T_beta_lower = T_beta_lim(alpha, beta, T_alpha, dtheta, f_hwp, num_of_HWP_rotation)
T_beta = T_beta_grid(T_beta_lower, 4*T_beta_lower, noutput)[idx]
if T_beta > T_alpha
    T_beta = T_alpha
end

nside = 256
ss = gen_ScanningStrategy()
ss.nside = nside
ss.sampling_rate = 19.0
ss.FP_theta = [0]
ss.FP_phi = [0]
ss.hwp_rpm = Float64(hwprpm)
ss.alpha = alpha
ss.beta = beta
ss.spin_rpm = period2rpm(T_beta, unit="sec")
ss.prec_rpm = period2rpm(T_alpha, unit="sec")
ss.coord = "G"

Args = Dict(
    "ss" => ss,
    "spin_n" => -5:5 |> collect ,
    "spin_m" => [-8,-4,0,4,8],
    "division" => 240*4,
    "fwhm" => deg2rad(FWHM),
    "num_of_HWP_rotation" => num_of_HWP_rotation,
    "rho" => deg2rad(1/60),
    "chi" => deg2rad(0.),
    "seed" => 1,
    "nreal" => 100,
    "sysfield" => pointing_offset_field,
    "lmax4likelihood" => 3nside-1,
    "sysmap_output" => false
);

result = run_systematics(Args)
T_alpha_hrs = T_alpha/3600

save_path = "/group/cmb/litebird/usr/ytakase/scan_optimisation/systematics/lower_T_beta_space/output"
create_h5_file(save_path, idx, "230502_256_19Hz_1amin_alpha=$(ss.alpha)_T_alpha=$(T_alpha_hrs)hrs", result)
