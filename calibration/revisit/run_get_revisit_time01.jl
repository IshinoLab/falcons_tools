for i in 1000:1500
    if i%10 == 0
        sleep(1)
    end
    run(`bsub -q s -J "Job.$i" -o "./log/stdout.%J.$i" -e "./log/stderr.%J.$i" julia get_revisit_time01.jl $i`)
end