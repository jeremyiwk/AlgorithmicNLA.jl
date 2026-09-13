include("lu_cooperative_bench.jl")

function confirm_main(args)
    output = get(args, 1, joinpath(@__DIR__, "results", "lu-cooperative-confirm.csv"))
    BLAS.set_num_threads(4)
    Metal.allowscalar(false)
    n = 4096
    return open(output, "w") do io
        println(io, "seed,stage,n,sample,seconds,get01,get02")
        for seed in (2026, 1, 2)
            A = randn(MersenneTwister(seed), Float32, n, n)
            B = randn(MersenneTwister(seed + 10), Float32, n, 4)
            D = MtlArray(A)
            cases = (
                ("LAPACK", () -> LinearAlgebra.lu(A)),
                ("combined-8", () -> CooperativeLU.factor(D; cutoff = 8)),
            )
            times = Dict(stage => Float64[] for (stage, _) in cases)
            factors = Dict{String, Any}()
            for sample in 0:5
                order = iseven(sample) ? cases : reverse(cases)
                for (stage, f) in order
                    factors[stage], seconds = elapsed_call(f, Metal.synchronize)
                    push!(times[stage], seconds)
                    println("seed=$seed $stage sample=$sample seconds=$seconds")
                    flush(stdout)
                end
            end
            for (stage, _) in cases
                F = factors[stage]
                r1 = get01(A, F)
                rhs = stage == "LAPACK" ? B : MtlArray(B)
                X = Float64.(Array(F \ rhs))
                original = Float64.(A)
                residual = Float64.(B) - original * X
                r2 = maximum(sum(abs, view(residual, :, j)) / (opnorm(original, 1) * sum(abs, view(X, :, j)) * (Float64(eps(Float32)) / 2)) for j in axes(X, 2))
                @assert r1 < 30 && r2 < 30 && F.info == 0
                println("seed=$seed $stage GET01=$r1 GET02=$r2")
                for (i, seconds) in enumerate(times[stage])
                    println(io, join((seed, stage, n, i - 1, seconds, r1, r2), ','))
                end
                flush(io)
            end
            GC.gc()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    confirm_main(ARGS)
end
