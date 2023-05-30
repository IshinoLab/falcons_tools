for i in 1:1500
    if i%10 == 0
        sleep(1)
    end
    run(`bsub -q l -J "Job.$i" -o "./log/stdout.%J.$i" -e "./log/stderr.%J.$i" julia half_sky_coverage01.jl $i`)
end