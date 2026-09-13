include("lu_cooperative_bench.jl")

function crossover_confirm_main(args)
    output = get(args, 1, joinpath(@__DIR__, "results", "lu-crossover-confirm.csv"))
    BLAS.set_num_threads(4)
    Metal.allowscalar(false)
    return open(output, "w") do io
        println(io, "n,stage,sample,seconds,get01,get02")
        for n in (3584, 3840, 4096)
            A = randn(MersenneTwister(2026), Float32, n, n)
            B = randn(MersenneTwister(2036), Float32, n, 4)
            D = MtlArray(A)
            cases = [
                ("LAPACK", () -> LinearAlgebra.lu(A)),
                ("block128-cutoff4", () -> CooperativeLU.factor(D; blocksize = 128, cutoff = 4)),
            ]
            if n == 4096
                push!(cases, ("block256-cutoff8", () -> CooperativeLU.factor(D; blocksize = 256, cutoff = 8)))
            end
            times = Dict(stage => Float64[] for (stage, _) in cases)
            factors = Dict{String, Any}()
            for sample in 0:5
                order = iseven(sample) ? cases : reverse(cases)
                for (stage, f) in order
                    factors[stage], seconds = elapsed_call(f, Metal.synchronize)
                    push!(times[stage], seconds)
                    println("n=$n $stage sample=$sample seconds=$seconds")
                    flush(stdout)
                end
            end
            for (stage, _) in cases
                F = factors[stage]
                r1 = get01(A, F)
                X = Float64.(Array(F \ (stage == "LAPACK" ? B : MtlArray(B))))
                original = Float64.(A)
                residual = Float64.(B) - original * X
                r2 = maximum(sum(abs, view(residual, :, j)) / (opnorm(original, 1) * sum(abs, view(X, :, j)) * (Float64(eps(Float32)) / 2)) for j in axes(X, 2))
                @assert r1 < 30 && r2 < 30 && F.info == 0
                for (i, seconds) in enumerate(times[stage])
                    println(io, join((n, stage, i - 1, seconds, r1, r2), ','))
                end
                flush(io)
            end
            GC.gc()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    crossover_confirm_main(ARGS)
end
