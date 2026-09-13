# Run from the repository root with --project=test/metal. Arguments: sizes, blocks,
# samples, output CSV. Comma-separated sizes/blocks default to 2048,4096 and 64,256.
# This diagnostic mirrors only LU's blocked driver; numerical primitives are shared.
using AlgorithmicNLA
using LinearAlgebra
using Metal
using Random

function elapsed_call(f, synchronize)
    synchronize()
    start = time_ns()
    result = f()
    synchronize()
    return result, (time_ns() - start) / 1.0e9
end

function phase_lu(A, blocksize, synchronize)
    n = size(A, 1)
    @assert size(A, 2) == n && 1 < blocksize < n
    times = zeros(6)
    counts = zeros(Int, 6)
    function phase(f, i)
        result, seconds = elapsed_call(f, synchronize)
        times[i] += seconds
        counts[i] += 1
        return result
    end
    data, ipiv, info = phase(1) do
        data = copy(A)
        @assert all(isfinite, data)
        ipiv = similar(data, LinearAlgebra.BlasInt, (n,))
        info = similar(data, LinearAlgebra.BlasInt, (1,))
        fill!(info, 0)
        (data, ipiv, info)
    end
    sfmin = AlgorithmicNLA.lu_safemin(Float32)
    for k in 1:blocksize:n
        width = min(blocksize, n - k + 1)
        panel = k:(k + width - 1)
        last = (k + width):n
        phase(2) do
            AlgorithmicNLA.getrf2!(data, ipiv, info, k, width, RowMaximum(), sfmin)
        end
        phase(3) do
            AlgorithmicNLA.lu_swap_rows!(data, ipiv, panel, 1:(k - 1))
            AlgorithmicNLA.lu_swap_rows!(data, ipiv, panel, last)
        end
        phase(4) do
            AlgorithmicNLA.lu_trsm!(view(data, panel, panel), view(data, panel, last), false, true)
        end
        if !isempty(last)
            phase(5) do
                mul!(view(data, last, last), view(data, last, panel), view(data, panel, last), -1.0f0, 1.0f0)
            end
        end
    end
    status = phase(6) do
        only(Array(info))
    end
    @assert status == 0
    return LinearAlgebra.LU(NLAArray(data), ipiv, status), times, counts
end

function get01(A, F)
    packed = Float64.(Array(F.factors))
    n = size(A, 1)
    p = collect(1:n)
    for (i, j) in enumerate(Array(F.ipiv))
        p[i], p[j] = p[j], p[i]
    end
    reference = Float64.(A)
    residual = Matrix(UnitLowerTriangular(packed)) * Matrix(UpperTriangular(packed)) - reference[p, :]
    return opnorm(residual, 1) / (n * opnorm(reference, 1) * (Float64(eps(Float32)) / 2))
end

function record(io, backend, n, blocksize, sample, phase, seconds, calls = 1)
    println(io, join((backend, n, blocksize, sample, phase, seconds, calls), ','))
    return flush(io)
end

function benchmark_case(io, f, synchronize, backend, A, n, blocksize, samples)
    F = nothing
    for sample in 0:samples
        F, seconds = elapsed_call(f, synchronize)
        record(io, backend, n, blocksize, sample, "total", seconds)
        println("$backend n=$n nb=$blocksize sample=$sample seconds=$seconds")
        flush(stdout)
    end
    ratio = get01(A, F)
    @assert ratio < 30
    println("GET01 $backend n=$n nb=$blocksize ratio=$ratio")
    flush(stdout)
    return F
end

function main(args)
    sizes = parse.(Int, split(get(args, 1, "2048,4096"), ','))
    blocks = parse.(Int, split(get(args, 2, "64,256"), ','))
    samples = parse(Int, get(args, 3, "3"))
    output = get(args, 4, joinpath(@__DIR__, "results", "lu-feasibility.csv"))
    @assert samples > 0 && all(n -> all(b -> 1 < b < n, blocks), sizes)
    BLAS.set_num_threads(4)
    Metal.allowscalar(false)
    println("Julia=$VERSION Metal=$(Base.pkgversion(Metal)) device=$(Metal.device())")
    println("BLAS=$(BLAS.get_config()) threads=$(BLAS.get_num_threads()) Float32 sizes=$sizes blocks=$blocks")
    println("Metal source=$(pathof(Metal)); matmul algorithm=$(Metal.matmul_alg[])")
    flush(stdout)
    open(output, "w") do io
        println(io, "backend,n,blocksize,sample,phase,seconds,calls")
        for n in sizes
            A = randn(MersenneTwister(2026), Float32, n, n)
            D = MtlArray(A)
            benchmark_case(io, () -> LinearAlgebra.lu(A), () -> nothing, "LAPACK", A, n, 0, samples)
            benchmark_case(io, () -> LinearAlgebra.lu(D), Metal.synchronize, "Metal-MPS", A, n, 0, samples)
            for blocksize in blocks
                GC.gc()
                benchmark_case(io, () -> AlgorithmicNLA.lu(A; blocksize), () -> nothing, "Package-CPU", A, n, blocksize, samples)
                baseline = benchmark_case(io, () -> AlgorithmicNLA.lu(D; blocksize), Metal.synchronize, "Package-Metal", A, n, blocksize, samples)
                for sample in 0:2
                    result, seconds = elapsed_call(() -> phase_lu(D, blocksize, Metal.synchronize), Metal.synchronize)
                    F, times, counts = result
                    record(io, "Phase-Metal", n, blocksize, sample, "total", seconds)
                    for (i, name) in enumerate(("setup", "panel", "swaps", "trsm", "gemm", "status"))
                        record(io, "Phase-Metal", n, blocksize, sample, name, times[i], counts[i])
                    end
                    @assert Array(F.factors) == Array(baseline.factors)
                    @assert Array(F.ipiv) == Array(baseline.ipiv)
                    println("Phase-Metal n=$n nb=$blocksize sample=$sample total=$seconds phases=$times (exact factors/pivots verified)")
                    flush(stdout)
                end
            end
        end
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
