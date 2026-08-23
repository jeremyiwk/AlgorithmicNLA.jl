# Benchmarks

Direct measurements of performance for the algorithms in `src/`, comparing the
same algorithm across array backends.

- `benchmarks.jl` defines `SUITE`, a `BenchmarkTools.BenchmarkGroup` with one
  group per algorithm, keyed by backend, element type, and problem size.
- `runbenchmarks.jl` is the entry point: `julia benchmarks/runbenchmarks.jl`
  activates this environment, develops the package, and runs the suite.

CI runs the suite informationally on pull requests and on a weekly schedule
(`.github/workflows/Benchmarks.yml`); timing results never block a merge, but
regressions are reviewed at PR time and the roadmap performance goal is
enforced before tagging a release.

The goal these benchmarks exist to verify: for every algorithm, the Metal array
path must run faster than the CPU array path once the problem size is large
enough, and the crossover size should be measured and reported, not guessed.
