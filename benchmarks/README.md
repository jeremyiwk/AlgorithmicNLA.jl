# Benchmarks

Direct measurements of performance for the algorithms in `src/`, comparing the
same algorithm across array backends.

Each numerical feature follows the [GPU performance workflow](../docs/src/performance.md).
Its review record connects primitive measurements to complete-operation timings,
with separate package-CPU and LAPACK baselines. Retain reproducible drivers and raw
samples here; performance lessons and required experiments live in that document.

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

The `array` group compares wrapped and parent `copy` and broadcast operations.
These measure wrapper overhead; numerical algorithm crossover benchmarks will be
added with the algorithms. To add Metal measurements from an environment with
Metal and the benchmark dependencies loaded:

```julia
using Metal
include("benchmarks/benchmarks.jl")
add_array_benchmarks!(SUITE, MtlArray, "MtlArray";
    types=(Float16, Float32), synchronize=Metal.synchronize)
```

GPU entries synchronize before and after each measured operation and use one
evaluation per sample. Allocation and copies remain on the selected backend.

The `lu` group measures allocating factorization and an eight-right-hand-side
solve, alongside CPU `LinearAlgebra.lu` for reference. Add device entries with:

```julia
add_lu_benchmarks!(SUITE, MtlArray, "MtlArray";
    types=(Float16, Float32), synchronize=Metal.synchronize)
```

LU timings include input finiteness checking and the final device status transfer.
The array-based recursive implementation makes many small backend calls; use these
measurements to identify a concrete need before introducing custom kernel fusion.

An opt-in block-size sweep measures the complete allocating LU, including its
checks, under `SUITE["lu-tuning"]`:

```julia
add_lu_tuning_benchmarks!(SUITE, MtlArray, "MtlArray";
    sizes=(128, 512), blocksizes=(64, 128, 256, 512),
    synchronize=Metal.synchronize)
```

Warm each configuration before collecting steady-state timings, and report
compilation separately. Start with a bounded size range: larger blocks improve
the trailing GEMM's arithmetic intensity but increase panel work, so primitive
GEMM speedups do not predict the best LU block size.

`results/lu-pivot-tuning.csv` retains the 2026-09-13 Metal measurements for the
original implementation, mapped pivot reduction, bucketed reduction lengths, and
final runtime-dimension column scaling. Times are seconds. `first-call` means
the first call for that configuration, after any earlier configurations in the
same process; it is not a fresh-process startup measurement. `repeated` samples
may still contain compilation, so the file retains outliers. The final stage
uses twelve repeated calls, the earlier stages five. See `docs/src/lu.md` for
hardware, methodology, and the distinction between latency and throughput.

To isolate small unit-lower triangular solves, with reset excluded from timing:

```julia
add_triangular_solve_benchmarks!(SUITE, MtlArray, "MtlArray";
    orders=(8, 16, 17, 32, 64), rhs_counts=(1, 8, 512),
    synchronize=Metal.synchronize)
```

Entries live under `SUITE["triangular-solve"]`, keyed by backend, element type,
triangle order, and right-hand-side count. `results/lu-small-trsm.csv` records
the before/after full LU and eight-RHS solve comparison on Metal, plus CPU LU
measurements. Times are seconds; each configuration has one first call and seven
repeated calls. The initial calls can include compilation, and all outliers are
retained. See `docs/src/lu.md` for the measured scope and numerical checks.

Ordered pivot application can be measured separately with:

```julia
add_row_swap_benchmarks!(SUITE, MtlArray, "MtlArray";
    cases=((128, 1, 1), (512, 64, 448), (1024, 1024, 8)),
    synchronize=Metal.synchronize)
```

Each case is `(rows, pivot_count, columns)`. Entries under `SUITE["row-swaps"]`
use pivots targeting the final row, including overlapping swaps, and exclude
input reset from timing. `results/lu-ordered-swaps.csv` retains the subsequent
full LU/solve comparison, with small triangular solves already enabled in its
baseline. Its sampling and units follow `lu-small-trsm.csv`.

For a bounded feasibility comparison at 2048 and 4096:

```sh
julia --startup-file=no --project=test/metal benchmarks/lu_feasibility.jl
julia --startup-file=no --project=test/metal benchmarks/lu_update_feasibility.jl
```

The first driver compares allocating package CPU/Metal LU with CPU LAPACK and
Metal's MPS LU on the same Float32 inputs, using four BLAS threads. It then mirrors
the blocked driver with synchronized timers for setup, panels, swaps, triangular
solves, GEMMs, and final status. Shared numerical primitives are unchanged; each
instrumented factorization must produce exactly the ordinary implementation's
stored factors and pivots. GET01 checks on stored factors use the existing <30
gate. This diagnostic supports square Float32 matrices with `1 < blocksize < n`.
Arguments are comma-separated sizes, comma-separated block sizes, sample count,
and output CSV; defaults are `2048,4096 64,256 3 results/lu-feasibility.csv` (the
default output is resolved relative to this directory).

The second driver compares the first trailing update using views, prepacked dense
arrays, and packing/copy-back included in timing, under Metal's `:auto` GEMM
selection. It preserves LU's actual leading dimension, offsets, and `alpha=-1,
beta=1`. Inputs are ones and the destination is reset to zero outside timing;
each final output is checked exactly. Its optional argument is the output CSV.

`results/lu-feasibility.csv` and `results/lu-update-feasibility.csv` retain the
2026-09-13 samples. Seconds are the timing unit; sample zero is the first call for
that configuration, not necessarily fresh-process compilation. The phase driver
uses two subsequent samples and other measurements use three. Phase timings
perturb scheduling and include synchronization overhead; they are not a native
LAPACK profile or an exact decomposition of the uninstrumented GPU runtime. See
the LU documentation for results, source-dispatch observations, and feasibility
conclusions. MPS is a performance reference, not the package's implementation.

## Cooperative panel and MPS view prototype

`lu_cooperative.jl` defines the opt-in `CooperativeLU` module. It executes a small
GETRF2 dependency schedule inside one KernelAbstractions workgroup and provides
an MPS GEMM adapter for unit-row-stride views using offsets and leading dimensions.
It does not overwrite production methods. Its entry point accepts square
`MtlMatrix` inputs with `1 < blocksize < n`, uses partial pivoting, checks input
finiteness, and returns a standard LU with backend factors/pivots and host status.
It returns singular status rather than reproducing the full public error API.
MPS GEMM is restricted to Float32; `use_mps=false` exercises generic panel
arithmetic on other supported Metal floating types. CUDA and CPU kernel execution
are untested. The explicit KA dependency lives in `test/metal`, not the package.

```julia
using Metal
include("benchmarks/lu_cooperative.jl")
F = CooperativeLU.factor(A; blocksize=256, cutoff=8)
```

Run these from the repository root in the Metal environment:

```sh
julia --startup-file=no --project=test/metal benchmarks/lu_cooperative_bench.jl
julia --startup-file=no --project=test/metal benchmarks/lu_cooperative_confirm.jl
julia --startup-file=no --project=test/metal benchmarks/test_lu_cooperative.jl
```

The first compares the existing implementation, MPS-view-only and
cooperative-only variants, and combined cutoffs 8/16/32. Its optional arguments
are matrix size and output CSV. The second alternates LAPACK and the combined
cutoff-8 prototype on three 4096 inputs, recording five warmed samples per input
and checking GET01 and GET02. Its optional argument is the output CSV.
`results/lu-cooperative.csv` and `results/lu-cooperative-confirm.csv` preserve every
sample, including first calls and outliers. Ratios in each case are computed from
the final factorization, outside timing. Both drivers use four OpenBLAS threads,
Float32 storage, and include allocating factorization, schedule creation/transfers,
finite-input checking, and final status synchronization. Input upload is excluded.

The validation driver reuses the existing LAPACK gate/generator definitions and
covers six Metal element types, singularity, pivot ties, safe-minimum scaling,
partial panels/workgroups, and MPS view offsets/padding. It intentionally does
not claim full production-interface coverage. The source adaptation for MPS
encoding is credited under `licenses/Metal.txt`.

The bounded follow-up below compares smaller sizes and tuning parameters:

```sh
julia --startup-file=no --project=test/metal benchmarks/lu_crossover_tuning.jl
julia --startup-file=no --project=test/metal benchmarks/lu_workgroup_tuning.jl
julia --startup-file=no --project=test/metal benchmarks/lu_crossover_confirm.jl
```

Each accepts an optional output CSV path. The size/block/cutoff sweep covers
2048/3072/4096, block widths 128/256, and cutoffs 4/8. The workgroup sweep holds
block width 128 and cutoff 4 fixed and compares 32/64/128/256 threads; CSV
workgroup zero denotes CPU LAPACK. Nonzero group sizes must produce exactly the
256-thread factors and pivots. Both sweeps record one first call and three warmed
samples, plus a GET01 check on the final factorization in each case.

Confirmation alternates execution order with LAPACK for five warmed samples at
3584/3840/4096, including the previous block-256/cutoff-8 candidate at 4096.
GET01 and four-RHS GET02 checks run outside timing. All three drivers use seed
2026, Float32, four BLAS threads, and the same allocating/timing boundaries as
the earlier prototype measurements. Every sample is retained in the respective
`results/lu-crossover-tuning.csv`, `lu-workgroup-tuning.csv`, and
`lu-crossover-confirm.csv` files.

The best tested follow-up configuration is
`CooperativeLU.factor(A; blocksize=128, cutoff=4, workgroupsize=256)`.
Smaller workgroups were slower, but the option remains available to reproduce
the experiment. Defaults are preserved for reproducing prior comparisons. This
configuration gives a modest 4096 improvement and tentative near-parity at 3840;
it does not establish a broad crossover across inputs or hardware.
