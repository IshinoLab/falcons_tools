hwprpm=48
HWPnrot=2
for i in 1:1500
    if i%10 == 0
        sleep(1)
    end
    run(`bsub -q l -J "Job.$i" -o "./log/stdout.$i" -e "./log/stderr.$i" julia h_nm_hitSD_dr01.jl $i $hwprpm $HWPnrot`)
end