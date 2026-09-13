include("lu_cooperative_bench.jl")

function workgroup_main(args)
    output = get(args, 1, joinpath(@__DIR__, "results", "lu-workgroup-tuning.csv"))
    BLAS.set_num_threads(4)
    Metal.allowscalar(false)
    return open(output, "w") do io
        println(io, "n,workgroupsize,sample,seconds,get01")
        for n in (2048, 3072, 4096)
            A = randn(MersenneTwister(2026), Float32, n, n)
            D = MtlArray(A)
            reference = nothing
            for wg in (0, 256, 128, 64, 32)
                f = wg == 0 ? () -> LinearAlgebra.lu(A) :
                    () -> CooperativeLU.factor(D; blocksize = 128, cutoff = 4, workgroupsize = wg)
                times = Float64[]
                F = nothing
                for sample in 0:3
                    F, seconds = elapsed_call(f, Metal.synchronize)
                    push!(times, seconds)
                    println("n=$n wg=$wg sample=$sample seconds=$seconds")
                    flush(stdout)
                end
                ratio = get01(A, F)
                @assert ratio < 30 && F.info == 0
                if wg == 256
                    reference = (Array(F.factors), Array(F.ipiv))
                elseif wg != 0
                    @assert Array(F.factors) == reference[1]
                    @assert Array(F.ipiv) == reference[2]
                end
                for (i, seconds) in enumerate(times)
                    println(io, join((n, wg, i - 1, seconds, ratio), ','))
                end
                flush(io)
                GC.gc()
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    workgroup_main(ARGS)
end
