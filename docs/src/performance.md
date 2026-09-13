# GPU performance workflow

Apply this workflow when planning each numerical feature and reviewing a
performance change. Record the evidence in the feature documentation and retain
the benchmark driver and raw samples under `benchmarks/`. Timing is informational
on shared CI; numerical gates remain blocking and follow LAPACK.

## Lessons from LU

- **Count device operations, not Julia statements.** Views avoid copies but do
  not eliminate launches, reductions, allocations, or synchronization. Inspect
  the delegated implementation and the device trace of dependent inner loops.
  Keep pivots and intermediate status on device until the public API needs them.
- **Move suitable sequential work into a device worker.** Ordered LASWP swaps
  have independent columns; one worker can apply a column's whole pivot sequence.
  Small TRSM has the same independent-RHS opportunity. This removed many launches
  without changing reference dependencies. Preserve overlap order and aliasing;
  one worker per column can still have strided accesses or too little parallelism.
- **Measure the limit of fusion.** LU uses substitution through order 16 and
  recursive GEMM updates above it. That cutoff is measured for LU, not a universal
  constant for Cholesky or QR. Larger fused work can reduce occupancy or serialize
  too much computation. Test the dimensions immediately around each cutoff.
  The later cooperative LU prototype was fastest at eight columns among tested
  cutoffs 8/16/32; the ordinary TRSM cutoff of 16 did not predict that choice.
  A subsequent 4/8 comparison favored four columns with a smaller outer block.
  Record the tested search range; a winning cutoff is not an established optimum.
- **Separate compilation from execution.** Shrinking views caused repeated Metal
  specialization. Runtime dimensions in mapped column scaling and bucketed
  reduction lengths reduced that cost. Warm a sequence of changing shapes as well
  as one repeated shape; retain first calls and outliers with their interpretation.
- **Benchmark the actual BLAS shapes.** A square GEMM speedup does not predict
  a trailing update with small inner dimension, strided views, and nonzero beta.
  LU's earlier 4096-output probe had little GPU advantage at inner dimension 64
  and about an eightfold advantage at 512. Retune the complete factorization after
  making its panels faster; a former block-size choice may cease to be best.
  Check view versus dense dispatch as well: the installed Metal backend's LU
  views bypass the MPSGraph path used by contiguous matrices. Packing costs must
  be included when evaluating that alternative; avoiding copies does not by
  itself ensure the fastest complete operation.
  First check whether the library exposes offset/leading-dimension descriptors:
  the cooperative LU prototype used those to reach MPS GEMM without packing.
- **Wrappers and precision belong in the optimization tests.** Exercise transpose,
  complex adjoint, strided views, padding, and implicit unit diagonals with host
  scalar indexing disabled. A wrapped `map!` destination can select a host fallback.
  Copy underlying device storage before constructing a CPU reference wrapper.
  Preserve factor storage precision, robust division, pivot ties, singularity
  reporting, and the accepted backend arithmetic limits.

The [LU measurements](lu.md) and `benchmarks/results/lu-*.csv` record the scope of
these observations. They establish useful experiments for later features, not
hardware-independent speedups or cutoffs.

## Required bounded performance review

1. Identify the reference algorithm and its panel, update, reduction, solve,
   permutation, allocation, and status operations. State which independent work
   can share a device launch and which dependencies require synchronization.
2. Measure the same represented inputs using both this package's CPU path and
   `LinearAlgebra`/LAPACK where supported. Label these two CPU baselines separately.
   Record hardware, Julia/backend versions, BLAS provider and thread count,
   element type, dimensions, block sizes, and relevant backend preferences.
3. Include a small case, a block/cutoff boundary, and a bounded larger case.
   Separate first-use compilation from warmed samples. Synchronize GPU timing
   boundaries; use one evaluation per mutating sample with an appropriate reset.
   Declare whether allocation, transfers, input checks, and final status are timed.
4. Attribute costs before choosing the next optimization: host submission and
   compilation, panels/reductions, swaps/solves, actual trailing updates, and data
   movement. Measure allocations on host and device separately. Synchronizing
   every primitive changes scheduling; use that only as an isolated diagnostic,
   and confirm conclusions against ordinary complete-operation timings. A device
   profiler can also alter batching, so record instrumentation effects.
5. Test at least the relevant small-operation fusion and block-size hypotheses,
   or record why each is inapplicable. Existing broadcasts, reductions, `map!`,
   views, and `mul!` come first. Introduce KernelAbstractions when a measured cost
   or missing capability needs cooperative reduction, shared storage, or explicit
   work distribution. Scalar indexing inside a device worker is allowed.
6. Validate the changed operation against the reference and the full numerical
   gates. Retain focused regressions for ordering, aliasing, wrappers, precision,
   and boundary cases. Confirm the whole feature benefits; avoid timing assertions
   in unit tests. Document backend coverage and any unmeasured hypothesis.

For a phase consuming fraction `f` of elapsed time, accelerating that phase by
`s` gives an ideal total speedup `1 / ((1 - f) + f / s)` if other costs stay fixed.
Use this budget to choose work: closing a fivefold gap through one phase alone
requires it to account for at least 80% of the original time, even with that phase
eliminated. A phase breakdown must use non-overlapping costs for this estimate.

## Carry forward to the next features

| Feature | Experiments to include |
| :--- | :--- |
| Cholesky | Small diagonal-block factorization and triangular solves; panel versus trailing SYRK/HERK or GEMM time; triangle/adjoint dispatch; block-size sweep. |
| QR | Reflector norm/scaling reductions and robust division; small panel and reflector-application fusion; compact WY update shapes; block-size sweep. |
| Both | Existing helper reuse after checking its contracts; changing-shape compilation; all supported element types; CPU LAPACK and package baselines. |

Use this compact record in each feature's plan and completed documentation:

```text
Reference routines / numerical gates:
Backend, versions, precision, shapes, BLAS settings:
Timed boundary and warmup / changing-shape procedure:
Dominant measured costs and plausible total speedup:
Fusion / blocking experiments (or why inapplicable):
Before / after complete-operation samples and benchmark driver:
Correctness regressions and backend coverage:
Remaining uncertainty and next experiment:
```

[Metal's profiling guide](https://metal.juliagpu.org/stable/profiling/) describes
synchronized measurements and tracing. The
[KernelAbstractions API](https://juliagpu.github.io/KernelAbstractions.jl/stable/api/)
provides workgroup-local storage and barriers for cooperative kernels; a workgroup
barrier is not a device-wide barrier. A tall panel spanning workgroups needs a
correct staged reduction and dependency scheme, not an assumed global barrier.
