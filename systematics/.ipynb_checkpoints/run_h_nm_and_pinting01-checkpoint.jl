for i in 1:1500
    if i%10 == 0
        sleep(1)
    end
    run(`bsub -q s -J "Job.$i" -o "./log/stdout.%J.$i" -e "./log/stderr.%J.$i" julia h_nm_and_pointing01.jl $i`)
end