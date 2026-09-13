# The benchmark suite. `runbenchmarks.jl` activates the environment, includes
# this file, and runs `SUITE`.
#
# Convention: one BenchmarkGroup per algorithm, keyed by backend ("Array",
# "MtlArray"), element type, and problem size, so that the Metal-vs-CPU
# crossover size stated in docs/src/roadmap.md can be read directly from the
# results. Benchmarks are direct measurements of the operation, synchronized
# before timing on asynchronous backends.

using BenchmarkTools
using AlgorithmicNLA
using LinearAlgebra
using Random

const SUITE = BenchmarkGroup()

function add_array_benchmarks!(
        suite, AT, backend;
        types = (Float16, Float32, Float64), sizes = (64, 512, 2048),
        synchronize = () -> nothing
    )
    haskey(suite, "array") || (suite["array"] = BenchmarkGroup())
    group = suite["array"]
    for T in types, n in sizes
        storage = AT(ones(T, n, n))
        for (label, A) in (("parent", storage), ("wrapped", NLAArray(storage)))
            dest = similar(A)
            group[backend, T, n, label, "broadcast!"] = @benchmarkable begin
                $synchronize()
                $dest .= $A .+ $A
                $synchronize()
            end evals = 1
            group[backend, T, n, label, "copy"] = @benchmarkable begin
                $synchronize()
                copy($A)
                $synchronize()
            end evals = 1
        end
    end
    return suite
end

add_array_benchmarks!(SUITE, Array, "Array")

function add_lu_benchmarks!(
        suite, AT, backend;
        types = (Float32, Float64), sizes = (32, 128, 512), synchronize = () -> nothing
    )
    haskey(suite, "lu") || (suite["lu"] = BenchmarkGroup())
    group = suite["lu"]
    for T in types, n in sizes
        rng = MersenneTwister(0)
        A = AT(T.(randn(rng, n, n)))
        B = AT(T.(randn(rng, n, 8)))
        F = AlgorithmicNLA.lu(A)
        group[backend, T, n, "lu"] = @benchmarkable begin
            $synchronize()
            AlgorithmicNLA.lu($A)
            $synchronize()
        end evals = 1
        group[backend, T, n, "solve-8-rhs"] = @benchmarkable begin
            $synchronize()
            $F \ $B
            $synchronize()
        end evals = 1
        if AT === Array
            group[backend, T, n, "LinearAlgebra.lu"] = @benchmarkable LinearAlgebra.lu($A)
        end
    end
    return suite
end

add_lu_benchmarks!(SUITE, Array, "Array")

# Opt-in: compiling and warming each configuration can be expensive on GPUs.
function add_lu_tuning_benchmarks!(
        suite, AT, backend;
        types = (Float32,), sizes = (128, 512), blocksizes = (64, 128, 256, 512),
        synchronize = () -> nothing
    )
    haskey(suite, "lu-tuning") || (suite["lu-tuning"] = BenchmarkGroup())
    group = suite["lu-tuning"]
    for T in types, n in sizes
        A = AT(T.(randn(MersenneTwister(0), n, n)))
        for blocksize in blocksizes
            group[backend, T, n, blocksize] = @benchmarkable begin
                $synchronize()
                AlgorithmicNLA.lu($A; blocksize = $blocksize)
                $synchronize()
            end evals = 1
        end
    end
    return suite
end

function add_triangular_solve_benchmarks!(
        suite, AT, backend;
        types = (Float32,), orders = (8, 16, 17, 32, 64), rhs_counts = (1, 8, 512),
        synchronize = () -> nothing
    )
    haskey(suite, "triangular-solve") || (suite["triangular-solve"] = BenchmarkGroup())
    group = suite["triangular-solve"]
    for T in types, n in orders, nrhs in rhs_counts
        rng = MersenneTwister(0)
        A = AT(Matrix(UnitLowerTriangular(T.(randn(rng, n, n)) ./ T(n))))
        B = AT(T.(randn(rng, n, nrhs)))
        work = similar(B)
        group[backend, T, n, nrhs] = @benchmarkable begin
            AlgorithmicNLA.lu_trsm!($A, $work, false, true)
            $synchronize()
        end setup = (copyto!($work, $B); $synchronize()) evals = 1
    end
    return suite
end

function add_row_swap_benchmarks!(
        suite, AT, backend;
        types = (Float32,), cases = ((128, 1, 1), (512, 64, 448), (1024, 1024, 8)),
        synchronize = () -> nothing
    )
    haskey(suite, "row-swaps") || (suite["row-swaps"] = BenchmarkGroup())
    group = suite["row-swaps"]
    for T in types, (n, npiv, ncols) in cases
        A = AT(reshape(T.(1:(n * ncols)), n, ncols))
        ipiv = AT(fill(n, npiv))
        work = similar(A)
        group[backend, T, n, npiv, ncols] = @benchmarkable begin
            AlgorithmicNLA.lu_swap_rows!($work, $ipiv, 1:$npiv, 1:$ncols)
            $synchronize()
        end setup = (copyto!($work, $A); $synchronize()) evals = 1
    end
    return suite
end
