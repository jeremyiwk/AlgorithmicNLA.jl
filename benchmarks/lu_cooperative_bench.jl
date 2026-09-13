include("lu_feasibility.jl")
include("lu_cooperative.jl")

function cooperative_main(args)
    n = parse(Int, get(args, 1, "4096"))
    output = get(args, 2, joinpath(@__DIR__, "results", "lu-cooperative.csv"))
    BLAS.set_num_threads(4)
    Metal.allowscalar(false)
    A = randn(MersenneTwister(2026), Float32, n, n)
    D = MtlArray(A)
    cases = [
        ("LAPACK", () -> LinearAlgebra.lu(A)),
        ("MPS-LU", () -> LinearAlgebra.lu(D)),
        ("baseline-256", () -> AlgorithmicNLA.lu(D; blocksize = 256)),
        ("mps-views-only", () -> CooperativeLU.factor(D; cooperative = false)),
        ("cooperative-only-16", () -> CooperativeLU.factor(D; use_mps = false)),
        ("combined-8", () -> CooperativeLU.factor(D; cutoff = 8)),
        ("combined-16", () -> CooperativeLU.factor(D; cutoff = 16)),
        ("combined-32", () -> CooperativeLU.factor(D; cutoff = 32)),
    ]
    return open(output, "w") do io
        println(io, "stage,n,sample,seconds,get01")
        for (stage, f) in cases
            F = nothing
            times = Float64[]
            for sample in 0:3
                F, seconds = elapsed_call(f, Metal.synchronize)
                push!(times, seconds)
                println("$stage n=$n sample=$sample seconds=$seconds")
                flush(stdout)
            end
            ratio = get01(A, F)
            @assert ratio < 30 && F.info == 0
            for (i, seconds) in enumerate(times)
                println(io, join((stage, n, i - 1, seconds, ratio), ','))
            end
            flush(io)
            println("$stage GET01=$ratio")
            flush(stdout)
            GC.gc()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    cooperative_main(ARGS)
end
