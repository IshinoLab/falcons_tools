hwprpm = [48, 61]
num_of_rot = [1, 2]

for i in hwprpm
    for j in num_of_rot
        for k in 1:1500
            if k%10 == 0
                sleep(1)
            end
            run(`bsub -q l -J "Job.$i" -o "./log/stdout.$i" -e "./log/stderr.$i" julia h_nm_hitSD_dr02.jl $k $i $j`)
        end
    end
end
