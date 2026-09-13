# Isolate the first trailing update, including its offsets and alpha/beta.
# Run with --project=test/metal. Output path is the optional first argument.
include("lu_feasibility.jl")

function update_main(args)
    output = get(args, 1, joinpath(@__DIR__, "results", "lu-update-feasibility.csv"))
    Metal.allowscalar(false)
    open(output, "w") do io
        println(io, "backend,n,blocksize,sample,phase,seconds,calls")
        for n in (2048, 4096), blocksize in (64, 256)
            data = Metal.ones(Float32, n, n)
            panel, last = 1:blocksize, (blocksize + 1):n
            A, B, C = view(data, last, panel), view(data, panel, last), view(data, last, last)
            a, b, c = copy(A), copy(B), copy(C)
            println("view dispatch: ", which(mul!, Tuple{typeof(C), Char, Char, typeof(A), typeof(B), Float32, Float32}))
            println("dense dispatch: ", which(mul!, Tuple{typeof(c), Char, Char, typeof(a), typeof(b), Float32, Float32}))
            for mode in ("view-auto", "dense-auto", "packed-auto")
                target = mode == "dense-auto" ? c : C
                for sample in 0:3
                    fill!(target, 0.0f0)
                    _, seconds = elapsed_call(Metal.synchronize) do
                        if mode == "view-auto"
                            mul!(C, A, B, -1.0f0, 1.0f0)
                        elseif mode == "dense-auto"
                            mul!(c, a, b, -1.0f0, 1.0f0)
                        else
                            packed_a, packed_b, packed_c = copy(A), copy(B), copy(C)
                            mul!(packed_c, packed_a, packed_b, -1.0f0, 1.0f0)
                            copyto!(C, packed_c)
                        end
                    end
                    record(io, mode, n, blocksize, sample, "first-update", seconds)
                    println("$mode n=$n nb=$blocksize sample=$sample seconds=$seconds")
                    flush(stdout)
                end
                @assert all(==(-Float32(blocksize)), Array(target))
            end
        end
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    update_main(ARGS)
end
