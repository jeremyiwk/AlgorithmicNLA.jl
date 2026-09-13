# LU factorization

Wrap the input to select AlgorithmicNLA through `LinearAlgebra`, or use the
qualified package function on ordinary backend arrays:

```jldoctest
julia> using AlgorithmicNLA, LinearAlgebra

julia> A = [0.0 2.0; 1.0 3.0]; F = lu(NLAArray(A));

julia> Array(F.L) * Array(F.U) == A[F.p, :]
true

julia> F \ [1.0, 2.0]
2-element Vector{Float64}:
 0.5
 0.5

julia> AlgorithmicNLA.lu(A).p == F.p
true
```

`lu!` overwrites its input. Both entry points return `LinearAlgebra.LU` with
`NLAArray` factors and backend pivot storage. Floating-point input retains its
element type. Allocating LU of integer input follows Julia's `lutype` conversion;
callers choose conversions explicitly for floating-point input.

The interface supports rectangular factorization, `RowMaximum`, `NoPivot`,
`RowNonZero`, `check`, and `allowsingular`. The `allowsingular` behavior is also
provided on Julia 1.10. `check=true` rejects nonfinite input before mutation;
`check=false` leaves responsibility for checking `F.info` or `issuccess(F)` to
the caller. A negative `info` denotes a failed `NoPivot` factorization, and a
positive `info` records the first exact zero pivot. Zero pivots in `NoPivot`
still throw when `check=true, allowsingular=true`.

`F.L`, `F.U`, and `F.P` allocate on the factor backend. `F.p` explicitly copies
pivots to the host and constructs Julia's permutation vector. Square factors
support vector/matrix solves, `ldiv!`, transpose and adjoint solves, right division,
`inv`, `det`, and `logabsdet`. Right-hand sides should reside on the same backend
as the factors. Allocating solves preserve the right-hand side's backend and
use Julia's result element-type rule without converting the factors.

## Algorithm and device execution

The driver follows LAPACK 3.12.1
[`GETRF`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dgetrf.f),
including recursive
[`GETRF2`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dgetrf2.f)
panels, pivot application to both sides of each panel, triangular substitution,
and `mul!` trailing updates. The default block size is 64, as in reference
`ILAENV`; `blocksize` can be changed for measurements and boundary testing.

Pivot selection uses a dimension-wise mapped reduction over metric/index pairs,
retaining the result on the device and writing its index directly into pivot
storage. The metric vector and separate value/index outputs are not materialized.
Device reductions use suffix lengths rounded up to powers of two (capped by the
row count), with completed rows contributing neutral metric/index pairs. This
limits reduction-length specializations to logarithmically many per matrix;
ordinary CPU arrays scan only the active suffix. The extra device scan is less
than twice the active length and does not change the eligible pivots.
LAPACK's complex pivot metric is `abs(real(z)) + abs(imag(z))`; this intentionally
differs from Julia's generic non-BLAS complex path, which uses `abs(z)`.
Ties choose the first row. There is no host read of each pivot.

Pivot application follows
[`LASWP`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dlaswp.f):
one `map!` worker per column applies its swaps sequentially in the requested
order. Different columns are independent, so overlapping pivots and reverse
application need no inter-worker synchronization or row workspace. Pivot indices
are read inside the device map. Transpose/adjoint map destinations are normalized
to their underlying storage, preserving conjugation and device dispatch.

Triangular substitution recursively partitions triangles larger than order 16
and updates with `mul!`. Small blocks use reference BLAS
[`TRSM`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/BLAS/SRC/dtrsm.f)
forward/back substitution inside `map!`, with one independent worker per
right-hand-side column. The scalar accesses run inside the device map; each
worker modifies only its own column. Unit diagonals and the unused triangle are
not read. Arithmetic is rounded to right-hand-side storage at each update, with
the existing robust complex division. It does not use Metal's native MPS
triangular solver, whose Float16 dispatch currently aborts. Existing backend
operations supply all device kernels; this package has no direct
KernelAbstractions dependency.

Column scaling uses `map!`, reading the one-element diagonal view inside the map.
GPUArrays supplies its kernel with runtime dimensions, avoiding Metal's repeated
broadcast specialization as the column shrinks. The diagonal and destination
views are disjoint; no host pivot or diagonal read is required.

The reciprocal-versus-division branch uses LAPACK's safe minimum. Complex scalar
division follows
[`LADIV`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dladiv.f)
scaling, with Float32 arithmetic for Float16/BFloat16 components and the original
precision for wider components. Results are rounded back to the stored element
type. Julia's `inv(::ComplexF32)` returns `ComplexF32` but
[explicitly widens its intermediates](https://github.com/JuliaLang/julia/blob/v1.12.6/base/complex.jl#L473-L480)
to Float64, which Metal cannot execute. Metal supplies a Float32 device override
for complex division; the common LADIV implementation here also covers generic
low-precision types and LAPACK's overflow/underflow scaling. GEMM accumulation
remains the backend's choice; factor storage is never implicitly widened.

LU inherits the backend's floating-point arithmetic, including flushing
subnormals to zero. On Apple GPUs this affects Float32 and arithmetic using
Float32 intermediates, including BFloat16. A stored subnormal pivot can therefore
compare equal to zero or produce an invalid multiplier. Such inputs do not have
CPU LAPACK's gradual-underflow guarantees. This follows the
[Metal arithmetic specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
(section 8.1); no software emulation or whole-array promotion is performed.

Input finiteness validation and the final status transfer synchronize.
`check=false` suppresses validation/errors, but constructing a standard `LU`
still needs its host `info` field. Benchmark timings include that transfer.

## Numerical validation

The LU tests use LAPACK 3.12.1
[`GET01`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/TESTING/LIN/dget01.f)
and [`GET02`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/TESTING/LIN/dget02.f)
ratios, with the threshold **strictly below 30** from all four precision-specific
test input files, including
[`dtest.in`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/TESTING/dtest.in).
No universal baseline multiplier is used.

- Reconstruction: `norm(L*U - A[p,:], 1) / (n * norm(A, 1) * u)`.
- Solve: the maximum over right-hand sides of
  `norm(b - op(A)*x, 1) / (norm(op(A), 1) * norm(x, 1) * u)`.

Here matrix norms are induced 1-norms and `u = eps(realtype(T))/2`, matching
`LAMCH('E')`. Complex solve vector norms use sums of `abs(real)+abs(imag)`, as
in `ZGET02`'s `DZASUM`; its matrix norm uses complex absolute values.

The corpus adapts the eleven general-matrix families from `CHKGE`/`LATB4`:
diagonal, upper/lower triangular, conditioned dense, three zero-column patterns,
two ill-conditioned cases, and small/large scales. Generic high-precision QR
constructs their spectra; it does not reproduce `LATMS`'s exact random matrices.
Generation uses `LAMCH('P') = eps(T)` for the prescribed condition numbers and
scales, distinct from the residual normalization. Additional tests cover
rectangles, block boundaries, pivot ties, safe-minimum division, and API behavior.
The tiny-pivot regression probes subnormal arithmetic separately: gradual-underflow
backends must pass, while flushing backends record the unavailable IEEE behavior
as an expected broken test. The GE-family gates retain the same threshold.

Residuals are evaluated in widened CPU arithmetic from the represented input and
stored factors, using the input precision's `u`. This prevents residual evaluation
overflow in low precision. It is an explicit adaptation of LAPACK's evaluation
arithmetic, with the same ratios and threshold. Supported CPU types also run
LAPACK reference factorizations through the same reconstruction gate.

The shared corpus runs on CPU and Metal; CUDA hardware validation remains
outstanding. Performance measurements are informational until the release
crossover target is demonstrated on hardware.

## Initial performance measurements

Float32 square matrices on Apple M5, Julia 1.12.6, Metal 1.11.0, and four
OpenBLAS threads (2026-09-13). Times are BenchmarkTools minima in milliseconds,
with one evaluation per sample and synchronization before and after GPU work.
Factorization includes allocation, finiteness checking, and the final status read.
Solves use existing factors and eight right-hand sides.

| Size | LU Array | LU LAPACK | LU Metal | Solve Array | Solve Metal |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.0288 | 0.00354 | 30.9 | 0.00658 | 1.92 |
| 128 | 0.224 | 0.0489 | 128 | 0.0298 | 3.72 |
| 512 | 5.98 | 0.718 | 470 | 0.160 | 12.9 |
| 1024 | 18.0 | 4.06 | 948 | 0.373 | 25.2 |

Five samples were requested with a ten-second sampling budget; Metal LU used
two, three, and four samples at sizes 128, 512, and 1024 respectively. All other
entries used five. These are an initial performance snapshot, not a crossover
study or a statistical regression threshold.

No crossover was observed in this range. A crossover is expected only once GEMM
work makes the factorization compute-bound, and need not occur immediately then.
These timings do not establish whether that condition holds at the largest size.
Small-size efficiency still needs profiling: separate pivot reductions, panel
work, row swaps, allocations, synchronization, and GEMMs. In particular, the
GPUArrays `findmax` used for this baseline materializes a metric vector and both
value/index outputs although LU only uses the index. The tuning follow-up below
replaces it with a mapped argmax reduction using existing array primitives.

## Bounded crossover probe

A follow-up on the same Apple M5, Julia 1.12.6, Metal 1.11.0, and four-thread
OpenBLAS setup found a **GEMM crossover, but not an end-to-end LU crossover**.
For `C += A*B`, with Float32 storage and a 4096-by-4096 output:

| Inner dimension `k` | CPU minimum (ms) | Metal minimum (ms) | CPU / Metal |
| ---: | ---: | ---: | ---: |
| 64 | 5.11 | 5.56 | 0.92 |
| 512 | 40.95 | 5.16 | 7.93 |
| 4096 | 307.37 | 40.95 | 7.51 |

Each case had one full-size warmup and two timed repetitions. Metal synchronized
before and after timing; input transfer and output validation were outside the
timed region. Dense inputs were filled with ones, with `C` reset to zero before
each repetition; every output entry was checked against the exact value `k`.
The two Metal measurements at `k=512` were 5.16 and 5.62 ms, versus 40.95 and
41.81 ms on CPU, so the advantage held in both repetitions. These are primitive
measurements, not LU timings or a precise search for the first crossover point.

The `k=64` case is representative of the inner dimension in the default blocked
LU trailing update. Its lack of a measured GPU advantage, alongside the roughly
8-fold advantage at `k=512`, makes GPU block-size tuning a concrete next experiment
within the existing LAPACK algorithm. GEMM dominance alone does not establish
that those GEMMs have enough arithmetic intensity to exploit the GPU.

The larger LU probe was stopped at 4096-by-4096 Float32: CPU timings were
0.65–0.71 s for this package and about 0.13 s for LAPACK, but the first Metal call
was interrupted before completion. A one-second process sample showed
heavy Julia compilation/type-inference activity and a 9.8 GiB physical footprint.
It did not yield a steady-state GPU timing. A separate Float16 probe at 1024
measured 25.2–25.8 ms on CPU, 159 s on the first Metal call, and 1.04 s on its
second call; the follow-up was stopped after a three-minute limit during its first
2048 Metal call. Thus the full LU crossover remains unmeasured. The GEMM results demonstrate
useful backend throughput, while the LU compilation growth and blocking overhead
need investigation before extending the size sweep.

## Pivot and compilation tuning

A bounded follow-up on the same machine replaced `findmax` with a mapped pair
reduction, reused reduction lengths, and changed column scaling to `map!`.
The default block size remains 64. No custom kernels or dependencies were added.

The mapped reduction alone reduced the best 512-by-512 LU time from 488 to
452 ms (about 7.5%). The final `map!` scaling path adds some execution overhead,
but avoids the expensive broadcast specializations encountered during repeated
calls. The final measurements show **a large latency improvement, with little
net improvement in best-case throughput**:

| Float32 size | Original minimum (ms) | Final minimum (ms) | Final repeated-call range (ms) |
| ---: | ---: | ---: | ---: |
| 128 | 116.4 | 118.8 | 118.8–121.8 |
| 512 | 488.3 | 480.3 | 480.3–514.5 |
| 1024 | — | 968.3 | 968.3–991.4 |

Inputs were random normal matrices with seed 0, already on the device. Timings
include allocating LU, finiteness checking, final status transfer, and explicit
synchronization. Original minima use five calls after a first call; final minima
and ranges use twelve calls after a first call. These were separate runs on
Julia 1.12.6 / Metal 1.11.0 with four OpenBLAS threads. Small differences should
not be treated as statistically established speedups. The earlier 1024 baseline
is a separate historical measurement, so it is not repeated in this comparison.

After running size 128, the first size-512 call took 61.3 seconds in the original
probe and 1.23 seconds in the final probe. The original probe also had a 122.6-second
outlier among its repeated size-512 calls; all twelve final repeated calls took
0.48–0.52 seconds. Source inspection and interruption stacks identified two
causes: Metal passes reduction extents as `Val` parameters, and specializes
frequently used broadcasts on their shape. Shrinking columns therefore generated
many kernel variants. Reusing reduction lengths and using runtime dimensions for
scaling address those two paths; these measurements do not guarantee that all
compilation spikes are eliminated for other sizes or backends.

The final block-size sweep gave the following minima in milliseconds:

| Size | Block 64 | Block 128 | Block 256 | Block 512 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 118.8 | 118.4 | 118.9 | 119.1 |
| 512 | 480.3 | 486.5 | 491.5 | 491.8 |
| 1024 | 968.3 | 990.8 | 996.9 | 1012.8 |

Larger blocks did not provide a useful improvement in this range. Full LU still
has no demonstrated CPU crossover. That pass left row-swap launches and small
recursive triangular operations as throughput targets.
The opt-in `add_lu_tuning_benchmarks!` helper reproduces block-size comparisons;
individual samples, including compilation outliers and intermediate variants,
are retained in `benchmarks/results/lu-pivot-tuning.csv`.

## Small triangular solves

Replacing recursion below order 16 with mapped column substitution improved
solves using existing LU factors. Larger triangles retain recursive GEMM updates.
A preliminary unit-lower-triangle probe covered orders 4–64 with 1, 8, and 512
right-hand sides. The mapped path helped at orders 8–32, but was slower at order
64 with 512 right-hand sides; 16 is a conservative cutoff, not a claimed optimum.

The following complete-operation comparison used Float32 on the same Apple M5,
Julia 1.12.6, Metal 1.11.0, and four-thread OpenBLAS setup. Each entry is the
minimum of seven synchronized calls after a warmup, in milliseconds. The previous
and new implementations ran in the same process on identical resident inputs.

| Size | LU before | LU after | Eight-RHS solve before | Eight-RHS solve after |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 111.84 | 110.13 | 3.35 | 3.59 |
| 512 | 451.90 | 444.73 | 12.00 | 6.88 |
| 1024 | 914.02 | 896.92 | 24.54 | 13.47 |

The eight-RHS solve used existing factors, including pivot application and output
allocation. Its best time fell about 43% at size 512 and 45% at size 1024.
The small size-128 solve did not improve in the minimum-time comparison; timings
there were noisy. The roughly 2% LU changes are modest and should not be treated
as established speedups without further sampling. Pivoting and row swaps still
limit factorization throughput; no CPU crossover is claimed.

Every measured factorization and solve passed the existing GET01/GET02 threshold
of 30, evaluated in Float64 from Float32 storage. Factorization ratios were below
0.08 and solve ratios below 6.1. Regression tests additionally exercise both
triangles, unit and non-unit diagonals, poisoned unused storage, strided views,
and orders around the cutoff across all discovered element types.

Individual samples, including outliers, are retained in
`benchmarks/results/lu-small-trsm.csv`. The opt-in
`add_triangular_solve_benchmarks!` helper measures unit-lower triangular solves
separately, with right-hand-side reset outside the timed region. Existing LU
benchmarks measure the complete factorization and solve effects. No custom
kernels or dependencies were added.

## Ordered pivot application

A subsequent pass replaced the three-copy implementation of each row swap with
mapped, ordered swaps. This retains the small-triangle optimization above and
removes the row workspace from LU, solves, and permutation-matrix construction.
Pivot selection, factor storage types, and numerical acceptance thresholds are
unchanged. An isolated probe applying 1024 pivots to eight columns dropped from
about 11.3 ms to 0.33 ms; the pivots repeatedly targeted the final row, exercising
overlapping swaps.

The full-operation comparison below used the same Apple M5 / Julia 1.12.6 /
Metal 1.11.0 setup, Float32 storage, and four OpenBLAS threads. Each entry is the
minimum of seven synchronized calls after a warmup, in milliseconds. Before/after
implementations ran in the same process with identical resident inputs. The
baseline already includes small triangular solves; compare within this table
rather than treating timings from separate probes as interchangeable.

| Size | LU before | LU after | Eight-RHS solve before | Eight-RHS solve after |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 111.39 | 6.62 | 2.24 | 0.70 |
| 512 | 440.18 | 24.10 | 7.09 | 1.42 |
| 1024 | 881.93 | 53.12 | 16.77 | 2.45 |

At size 1024, this is about **16.6 times faster for LU** and **6.8 times faster
for the eight-RHS solve** than the preceding implementation. GPU LU is still
slower than CPU: this package's updated CPU LU minimum was 7.27 ms at size 1024.
No crossover or optimal block size is established by this probe.

LU timings include allocation, finiteness checking, and final status transfer.
Solve timings include output allocation and pivot application with existing
factors. GET01 and GET02 checks, evaluated from Float32 storage in Float64, passed
the threshold of 30; their computed ratios were identical before and after the
swap change on these matrices. First-use compilation and timing outliers remain
visible in `benchmarks/results/lu-ordered-swaps.csv`; for example, one final
size-1024 LU sample took 177 ms while the other six took 53–56 ms.

Regression coverage includes overlapping and self-swaps, reverse order, empty
ranges, selected columns, strided storage, and real/complex transpose and adjoint
destinations. `add_row_swap_benchmarks!` isolates pivot application with input
reset outside timing. Existing backend array primitives supply the kernels;
there is no new dependency or host scalar indexing of GPU data.

## Remaining performance options

The reported roughly fivefold gap to LAPACK at size 8192 motivates a new phase
profile after the swap changes. That user-reported result has not been reproduced
in the checked-in measurements; its timing configuration remains to be recorded.
The following are hypotheses, not measured speedups:

1. **Retune blocking and inspect delegated GEMM.** Compare complete LU with block
   widths 64, 128, 256, and 512, alongside its actual trailing-update views and
   `alpha=-1, beta=1` products. Panel optimizations can change the best width.
   Compare supported backend GEMM implementations with the same update contract;
   an accelerated path for plain `A*B` need not support LU's accumulating update.
2. **Reduce panel launches and temporary reductions.** Each current column leaf
   separately selects a pivot, writes its index, swaps its diagonal entry, updates
   status, and scales below the pivot. Preallocated reduction scratch and fusing
   compatible steps are candidates. The recursive panel also invokes many tiny
   solves and products. A cooperative small-panel/base-case kernel could retain
   the reference dependencies with fewer launches. It needs deterministic pivot
   ties, safe-minimum handling, and a valid reduction across all candidate rows.
3. **Investigate scheduling only where the profile justifies it.** Submission
   batching can reduce host overhead, but it does not fuse device work. Overlapping
   preparation of the next panel with a disjoint trailing update (look-ahead) is
   a larger option requiring explicit dependencies, stream support, and an audit
   of pivot effects. It is not the first implementation change to make.

These can be implemented in Julia. KernelAbstractions is the portable candidate
when cooperative execution exceeds what the existing array primitives express.
The inspected [rocSOLVER GETF2 source](https://github.com/ROCm/rocSOLVER/blob/develop/library/src/lapack/roclapack_getf2.hpp)
contains fused pivot/status/swaps and optional specialized small/skinny-matrix
kernels. This supports investigating fusion; its dispatch conditions and numerical
details are not a substitute for our LAPACK contract or measurements on Metal.

Start with an uninstrumented warmed baseline and a phase profile at representative
small and large sizes, then the bounded block/GEMM comparison. Use the
[performance workflow](performance.md) to estimate how much of the total gap the
measured dominant phase can explain before committing to a panel-kernel rewrite.

## Feasibility at 2048 and 4096

A bounded follow-up on 2026-09-13 compared the current implementation with CPU
LAPACK and Metal's existing MPS LU. Hardware: Apple M5; Julia 1.12.6; Metal 1.11.0
loaded from the installed `pSzWS` package tree; ILP64 OpenBLAS with four threads;
Float32 random square matrices generated with `MersenneTwister(2026)`.
Metal used its default `:auto` GEMM choice, with host scalar indexing disabled.
Inputs were already resident on each backend. These are allocating factorizations;
package timings include the finite-input check and final status read. MPS is a
performance reference with its own checking contract, not a drop-in validation
of our full API or precision coverage.

Each configuration had one first call and three subsequent synchronized samples.
The table gives their warmed minimum in milliseconds; all samples are retained.

| Size | CPU LAPACK | Metal MPS LU | Package Metal, NB=64 | Package Metal, NB=256 |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 21.81 | 23.91 | 133.39 | 128.14 |
| 4096 | 129.13 | 67.43 | 447.17 | 384.61 |

LAPACK's three samples were 21.81–21.87 ms and 129.13–131.51 ms; MPS's were
23.91–24.85 ms and 67.43–69.71 ms. Package Metal showed more variability:
133–254 / 128–258 ms at 2048 and 447–529 / 385–464 ms at 4096 for NB=64 / 256.
Package CPU minima were 38.05 / 36.46 ms and 241.10 / 221.98 ms respectively;
those are separate from LAPACK. First Metal package calls took 4.69 s at 2048/64
and 1.27 s at 4096/64; warmed timings do not include that compilation cost.

The MPS result demonstrates that GPU LU can be competitive in the requested size
range on this machine: it nearly matches CPU at 2048 and beats it at 4096. It does
not demonstrate that our portable implementation can achieve MPS performance.

### Where the current implementation spends time

A diagnostic driver times each outer blocked-LU phase with synchronization at
its boundaries. Panels include recursive inner swaps, solves, and GEMMs; the
other categories below count only operations outside those panels. The following
rows are complete individual warmed diagnostic samples, in milliseconds:

| Size / block width | Panel | Outer swaps | Outer TRSM | Trailing GEMM | Setup + status | Total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 / 64 | 107.11 | 14.03 | 9.96 | 35.35 | 2.30 | 168.90 |
| 2048 / 256 | 106.70 | 7.50 | 6.52 | 13.75 | 1.74 | 136.27 |
| 4096 / 64 | 264.94 | 34.62 | 22.64 | 208.07 | 4.64 | 535.41 |
| 4096 / 256 | 238.41 | 22.44 | 19.03 | 114.79 | 4.01 | 398.82 |

Totals also include host work between phases. These timers perturb scheduling:
compare with the uninstrumented timings above, and do not interpret their sum as
an exact device-only profile or a strict performance lower bound. Instrumented
results matched ordinary LU's packed factors and pivots exactly. Every backend
and block configuration passed GET01 <30, evaluated from Float32 factors in
Float64; the largest measured ratio was below 0.061. No numerical code changed.

### Views select a different GEMM implementation

Installed-source inspection and method lookup identify a material difference:
Metal's `mul!` for range views (`src/linalg.jl:114`) selects its native GEMM path
under `:auto`; contiguous `MtlMatrix` operands (`src/linalg.jl:70`) can select
MPSGraph. A standalone dense matrix multiplication therefore does not measure
the implementation used by our LU trailing updates.

A separate probe measures the actual first update at size 4096 with
`alpha=-1, beta=1`, preserving its parent leading dimension and offsets. The
contiguous case has the same mathematical dimensions but is packed before timing;
the last column includes fresh packing of all three operands and output copy-back.

| Block width | Output / inner dimensions | Views (ms) | Contiguous (ms) | Packing + GEMM + copy-back (ms) |
| ---: | :--- | ---: | ---: | ---: |
| 64 | 4032×4032 / 64 | 9.05 | 1.43 | 11.68 |
| 256 | 3840×3840 / 256 | 20.05 | 2.27 | 12.88 |

These are minima of three samples after a first call. All three size-4096 samples
were within 9.05–9.06 / 20.05–21.07 ms for views, 1.43–1.54 / 2.27–2.56 ms for
contiguous matrices, and 11.68–12.04 / 12.88–13.02 ms with packing. The inputs were
ones, destination reset to zero outside timing, and every output entry checked
against the exact result. This is one update, not a whole-factorization speedup.
Packing is slower at width 64 and helps at 256; it is not a blanket solution.

### Decision

Competitive LU at 4096 is a credible target, while 2048 is more demanding. The
current evidence favors a substantial change in execution strategy over further
small block-size adjustments alone:

- Prototype a cooperative panel/base-case implementation preserving the LAPACK
  dependencies, pivot ties, safe-minimum handling, and numerical gates. The
  existing independent maps do not address all panel launches and reductions.
- Improve GEMM execution for the actual views, through backend support or measured
  packing/workspace strategies. At 4096/256, deleting even the entire measured
  panel phase would leave about 160 ms in the instrumented run, so a panel-only
  optimization is not a sufficient demonstrated route to the 129 ms CPU target.
- Test a complete 4096 factorization after these bounded prototypes before
  committing to a wider kernel rewrite. For planning only, panel <=50 ms and
  trailing updates <=30 ms, with other costs near 45 ms, would fit the current CPU
  timing budget. These are unachieved performance targets, not numerical gates.

Explicit GPU kernels can remain in Julia: Metal.jl compiles Julia kernels to
Metal functions, and KernelAbstractions provides portable workgroup programming.
See [Metal's compiler API](https://metal.juliagpu.org/stable/api/compiler/) and the
[KernelAbstractions quickstart](https://juliagpu.github.io/KernelAbstractions.jl/stable/quickstart/).
The measurements establish neither a need to switch source language to Metal
Shading Language nor a guarantee that portable kernels will match MPS.

Reproduce with `benchmarks/lu_feasibility.jl` and
`benchmarks/lu_update_feasibility.jl` in the `test/metal` environment. Their raw
samples live in `benchmarks/results/lu-feasibility.csv` and
`benchmarks/results/lu-update-feasibility.csv`; see `benchmarks/README.md` for
arguments and timing boundaries. CUDA and other precisions were not measured.

## Cooperative panels and MPS views at 4096

The bounded prototype requested after the feasibility study reached roughly CPU
LAPACK timing on 4096-by-4096 Float32 matrices. It lives in
`benchmarks/lu_cooperative.jl` as `CooperativeLU.factor`; the public package LU
implementation remains unchanged. This is evidence to continue development, not
a release-level performance claim or a complete replacement of the public API.

The cooperative kernel executes a small GETRF2 dependency schedule inside one
256-thread workgroup. That group scans every candidate pivot row, reduces with
first-index tie breaking, and performs the required swaps, scaling, triangular
substitution, and updates in dependency order. It uses workgroup-local reduction
storage and barriers covering global and local memory. Larger panels retain the
recursive driver and backend matrix products. This avoids a device-wide barrier
or an assumption that multiple workgroups can synchronize inside the kernel.
GEMM accumulators use at least Float32, following the backend convention, with
results rounded into the original factor storage. Safe-minimum scaling reuses
the existing implementation.

The MPS adapter constructs matrix descriptors with each view's byte offset and
leading dimension, encoding the product with reversed operands for column-major
storage. Inputs and outputs remain views of the original buffer; no packing or
copy-back is required. This adapter is specific to Metal and currently restricted
to Float32, while the panel arithmetic is written with KernelAbstractions. Its
MPS encoding adaptation is credited in `licenses/Metal.txt`.

### Initial comparison

Apple M5, Julia 1.12.6, Metal 1.11.0, KernelAbstractions 0.9.42, four-thread ILP64
OpenBLAS, Float32, seed 2026. Times below are warmed minima in milliseconds from
three repetitions after a first call. Each case includes allocating factorization,
the finite-input check where performed by that implementation, final status
synchronization, and prototype schedule allocation/transfers. Initial input upload
is excluded. All final factorizations passed GET01 <30.

| Implementation, block width 256 | Minimum (ms) |
| :--- | ---: |
| CPU LAPACK | 128.27 |
| Existing Metal MPS LU | 67.62 |
| Package Metal | 370.62 |
| MPS views, original panels | 275.88 |
| Cooperative cutoff 16, original GEMMs | 240.89 |
| Cooperative cutoff 8 + MPS views | 123.49 |
| Cooperative cutoff 16 + MPS views | 139.15 |
| Cooperative cutoff 32 + MPS views | 177.21 |

The combined cutoff-8 result is about three times faster than the package
baseline. Larger cooperative panels lost performance, so eight columns was the
candidate carried into confirmation. No Metal Shading Language kernel was needed.

### Interleaved confirmation

Five warmed samples per input alternated the order of LAPACK and the cutoff-8
prototype, after a first call for each. The three matrices used seeds 2026, 1,
and 2. These are medians, with full warmed ranges in parentheses, in milliseconds:

| Seed | LAPACK | Cooperative + MPS views |
| ---: | :--- | :--- |
| 2026 | 134.16 (129.27–146.07) | 124.19 (123.30–126.33) |
| 1 | 129.20 (127.69–133.06) | 123.89 (123.03–126.37) |
| 2 | 131.13 (125.55–135.75) | 129.74 (126.99–159.16) |

This supports rough parity, not a decisive or universal GPU advantage. In
particular, LAPACK had the better minimum on seed 2, and the prototype had a
159 ms outlier. First-use compilation also remains: the first combined call in
the confirmation process took 2.26 seconds.

Reconstruction and four-RHS solve checks on all three matrices passed GET01 and
GET02 <30, evaluated in Float64 from represented Float32 input/factors. The
prototype's maximum GET01 was below 0.060 and maximum GET02 below 18.90. These
checks are outside timing. The focused validation suite additionally covers the
eleven LAPACK GE families across all six discovered Metal element types, partial
panels/workgroups, pivot ties, singularity, safe-minimum scaling, and MPS descriptor
offsets/padding. Other element types use `use_mps=false`; the combined performance
result applies only to Float32. No CUDA execution was performed.

### Scope and next decision

The feasibility experiment succeeded at the agreed 4096 target, so its result
does not trigger the proposed performance pause. The next implementation work is
integration: preserve the complete LU interface and fallback coverage, place the
Metal-specific descriptor adapter behind backend dispatch, and validate kernel
execution on other backends. Production integration must also cover supported
pivot strategies and rectangular/wrapped inputs. The prototype currently accepts
square `MtlMatrix` inputs with partial pivoting and returns singular status without
implementing the full public exception policy.

The standard `AlgorithmicNLA.lu` has not acquired this acceleration or a new
dependency. KernelAbstractions is an explicit dependency of the Metal experiment
environment. Run `benchmarks/lu_cooperative_bench.jl`,
`benchmarks/lu_cooperative_confirm.jl`, and `benchmarks/test_lu_cooperative.jl`
with `--project=test/metal`; instructions are in `benchmarks/README.md`.
Raw samples and final-case residual ratios are preserved in
`benchmarks/results/lu-cooperative.csv` and
`benchmarks/results/lu-cooperative-confirm.csv`.

## Smaller crossover and configuration follow-up

A bounded sweep on the same M5 / Julia 1.12.6 / Metal 1.11.0 / KA 0.9.42 setup
tested whether the cooperative prototype could become competitive below 4096.
Float32, four-thread OpenBLAS, and seed 2026 were retained. Block width 128 with
cutoff 4 improved on the previous block-256/cutoff-8 candidate. The following
medians are from three warmed calls after a first call, in milliseconds:

| Size | LAPACK | Previous prototype | Block 128 / cutoff 4 | Metal MPS LU |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 21.69 | 42.27 | 40.25 | 23.83 |
| 3072 | 59.55 | 77.27 | 72.21 | 42.21 |
| 4096 | 129.22 | 125.54 | 116.61 | 68.27 |

This improves the prototype at 4096, but does not close the gap at 2048 or 3072.
An additional workgroup sweep compared 32/64/128/256 threads at block 128 and
cutoff 4. Smaller groups were consistently slower. At 3072 the corresponding
medians were 148.16/104.32/82.22/73.17 ms; 256 remains the selected group size.
Every tested group size produced exactly the same factors and pivots as 256.

### Confirmation just below 4096

Five warmed repetitions alternated execution order with LAPACK at 3584, 3840,
and 4096. The previous prototype was also timed at 4096 in that same interleaved
run. Each cell gives minimum / median / maximum in milliseconds:

| Size | LAPACK | Block 128 / cutoff 4 |
| ---: | :--- | :--- |
| 3584 | 86.65 / 89.06 / 111.47 | 92.07 / 94.35 / 200.97 |
| 3840 | 107.24 / 116.22 / 204.40 | 103.90 / 104.78 / 105.27 |
| 4096 | 128.45 / 152.64 / 226.38 | 116.27 / 117.93 / 121.10 |

The previous prototype at 4096 measured 123.19 / 123.56 / 125.20 ms, so the
updated configuration reduced its median by about 4.6% in this confirmation.
CPU timing outliers were larger here than in the initial sweep; they must not
be interpreted as a large new GPU speedup. In particular, the minimum advantage
at 3840 is only about 3%. This is tentative near-parity at 3840 for the measured
input, not a precise or broadly confirmed crossover. At 3584 the CPU still wins.
Metal's native MPS LU remains faster at 3072, demonstrating room for improvement
in the prototype but not how much of that improvement another tuning pass can
recover. No further kernel redesign was attempted in this follow-up.

All sweep factorizations passed GET01 <30 (maximum below 0.063), and all final
confirmation factorizations/solves passed GET01 and GET02 <30 (maximum GET02
below 18.42). Residuals use Float64 evaluation of represented Float32 input and
factors. Input upload and validation are excluded from timings; factor allocation,
schedule construction/transfers, finite-input checks, and final status are included.
First calls and all outliers are preserved in the raw CSV files. The prototype's
workgroup-size parameter is now explicit, with 256 retained as the default; neither
the reference algorithm nor the production LU implementation changed.

Run `benchmarks/lu_crossover_tuning.jl`, `benchmarks/lu_workgroup_tuning.jl`, and
`benchmarks/lu_crossover_confirm.jl` in the Metal environment. The matching CSV
files under `benchmarks/results/` retain all samples and final-case residuals.
The tested candidate is
`CooperativeLU.factor(A; blocksize=128, cutoff=4, workgroupsize=256)`.
