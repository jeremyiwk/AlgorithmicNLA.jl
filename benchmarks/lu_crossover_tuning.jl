# Bounded size/block/cutoff comparison; uses the existing cooperative prototype.
include("lu_cooperative_bench.jl")

function crossover_main(args)
    output = get(args, 1, joinpath(@__DIR__, "results", "lu-crossover-tuning.csv"))
    BLAS.set_num_threads(4)
    Metal.allowscalar(false)
    return open(output, "w") do io
        println(io, "n,stage,blocksize,cutoff,sample,seconds,get01")
        for n in (2048, 3072, 4096)
            A = randn(MersenneTwister(2026), Float32, n, n)
            D = MtlArray(A)
            cases = [("LAPACK", 0, 0), ("MPS-LU", 0, 0)]
            append!(cases, [("prototype", b, c) for b in (128, 256), c in (4, 8)])
            for (stage, blocksize, cutoff) in cases
                f = stage == "LAPACK" ? () -> LinearAlgebra.lu(A) :
                    stage == "MPS-LU" ? () -> LinearAlgebra.lu(D) :
                    () -> CooperativeLU.factor(D; blocksize, cutoff)
                times = Float64[]
                F = nothing
                for sample in 0:3
                    F, seconds = elapsed_call(f, Metal.synchronize)
                    push!(times, seconds)
                    println("n=$n $stage block=$blocksize cutoff=$cutoff sample=$sample seconds=$seconds")
                    flush(stdout)
                end
                ratio = get01(A, F)
                @assert ratio < 30 && F.info == 0
                for (i, seconds) in enumerate(times)
                    println(io, join((n, stage, blocksize, cutoff, i - 1, seconds, ratio), ','))
                end
                flush(io)
                GC.gc()
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    crossover_main(ARGS)
end
