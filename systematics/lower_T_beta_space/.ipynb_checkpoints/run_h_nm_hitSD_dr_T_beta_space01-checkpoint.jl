hwprpm = 61
HWPnrot = 1
alpha = [45, 47.5, 50]
T_alpha = [1.0, 192.348/60, 10, 100] .* 3600 # sec
for i in 1:100
    for j in eachindex(alpha)
        for k in eachindex(T_alpha)
            if i%10 == 0
                sleep(1)
            end
            run(`bsub -q l -J "Job.$i" -o "./log/stdout.$i" -e "./log/stderr.$i" julia h_nm_hitSD_dr_T_beta_space01.jl $i $(alpha[j]) $(T_alpha[k]) $hwprpm $HWPnrot`)
        end
    end
end

