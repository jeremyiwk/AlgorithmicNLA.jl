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

const SUITE = BenchmarkGroup()

# Example shape for when the first algorithm lands:
#
# SUITE["qr"] = BenchmarkGroup()
# for n in (256, 1024, 4096), T in (Float32,)
#     A = testmatrix(Array, T, n, n)
#     SUITE["qr"]["Array", T, n] = @benchmarkable householder_qr($A)
# end
