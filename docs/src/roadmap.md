# Roadmap

A coarse ordering of the work. Phases are sequenced so that each one supplies the
building blocks the next one needs; within a phase, features are planned and
implemented one at a time via the `plan-feature` workflow.

Each phase lists its **deliverables** — the concrete types and functions it adds —
and a **phase gate**: the test criteria that define completion. The gates are a
developer contract: a phase is complete only when its gate passes in CI, a gate may
be strengthened but never weakened once written, and a feature that cannot meet its
gate is redesigned, not exempted. Throughout, `u` denotes the unit roundoff of the
element type and `c` a modest constant independent of the matrix; "reference" means
LAPACK via `LinearAlgebra` on the `Array` path.

## Cross-cutting requirements

These apply to every phase rather than being phases themselves:

- **Backend validation is continuous, not deferred.** Every algorithm is written
  against the `AbstractArray` interface, validated on the `Array` path against
  LAPACK first, and then run against at least one GPU backend (Metal) before it is
  considered done. Every gate below is evaluated for every array type in
  `ARRAY_TYPES` and every element type the backend supports, with
  `allowscalar(false)` in force on device arrays.
- **Two testing layers, always both.** Every feature test set asserts the
  mathematical bounds of its phase gate *and* baseline agreement with the
  reference implementation on the same input: the measured error must satisfy
  `err ≤ 4 · max(err_ref, n·u·scale)` (the harness's `within_baseline`) — floored
  at one expected unit so an anomalously exact reference does not fail correct
  rounding. The factor `4` reflects that we implement the *same algorithms* as the
  reference: evaluation order and blocking move the error constant by factors
  near `√2`–`2`, essentially never beyond `3`, while a genuine defect grows with
  `n`, `κ`, or `1/u` and exceeds any constant immediately. A deliberately
  different algorithm may pass a larger factor explicitly, with a stated reason. The reference is
  LAPACK via `LinearAlgebra` where the element type has a LAPACK path (`Float32`,
  `Float64`, their complex types) and the same computation in `referencetype(T)`
  otherwise. Baseline comparisons respect the invariances of the factorization:
  spectra compared after sorting (`compare_spectra`), vectors up to phase
  (`matches_up_to_phase`), clustered eigenvectors as subspaces
  (`subspace_distance`) — never raw elementwise equality of non-unique objects.
- **Identities are asserted over the corpus, not a draw.** Algebraic invariances
  (shift `eig(A + σI) = eig(A) + σ`, scaling equivariance, orthogonal similarity
  invariance, transpose relations) run over all of `TEST_SEEDS`, the
  `condition_sweep(T)` matrices, and the phase's named adversarial matrices — a
  single random instance is almost always well conditioned and demonstrates
  nothing.
- **Conditioning is swept, not sampled.** Gates are evaluated on
  `conditioned_testmatrix` inputs across `condition_sweep(T)` — well conditioned,
  `u^{-1/2}`, and near numerically singular — in addition to uniform random
  matrices.
- **Nonfinite inputs follow LAPACK:** no input validation, `NaN`/`Inf` propagate
  to the output. Tests assert the LAPACK behavior — completion without a crash or
  hang, nonfinite in implies nonfinite (or a reported failure flag) out — and
  never a silent finite wrong answer.
- **Benchmarks land with the feature.** Each algorithm adds entries to the
  `benchmarks/` suite keyed by backend, element type, and size.
- **Docs land with the feature.** Each algorithm's docstring appears in the API
  reference when it merges.
- **Shape edge cases are part of every gate:** `0×0`, `1×1`, a single row or
  column, tall, wide, and dimensions that are not multiples of any block size.
- **Regression corpus** (a convention, not part of the gates): every bug that
  escapes to a reported failure gets its minimal reproducer pinned as a
  permanent named test, the way the `Metal` `Float16` `lu` segfault already is.

## Phase 0 — Foundations *(complete)*

Package scaffolding, the backend-parametrized test harness, and the conventions for
accuracy testing (residual, orthogonality, and backward error bounds measured in
units of the unit roundoff). Element types are discovered by probing the backend.

## Phase 1 — Primitives

The building blocks everything later reuses. Triangular solves sit here, not after
the eigensolvers, because least squares in Phase 2 already requires them and because
they are among the operations GPU backends most commonly lack.

**Deliverables**

- `HouseholderReflector{T}` — the reflector `H = I − τ v vᴴ` with `v₁ = 1` stored
  implicitly, following LAPACK `xLARFG`; constructor `householder!(x)` and
  application `lmul!` / `rmul!` operating on matrix panels without forming `H`.
- Givens rotations: construction and application compatible with
  `LinearAlgebra.givens`, expressed without scalar indexing of device arrays.
- Triangular solves: in-place forward and back substitution for upper and lower,
  unit and non-unit diagonal, unblocked and blocked. Internal, non-pirating names;
  wiring into `LinearAlgebra.ldiv!` is Phase 7's job.

**Phase gate**

- Reflector correctness: applying `H` to its defining vector annihilates all
  entries below the first, with the residual below `c·n·u·‖x‖`; `‖HᴴH − I‖ ≤ c·u`;
  sign and `τ` conventions match `xLARFG`, verified against reference output.
  Degenerate inputs covered: `x = α e₁` (τ = 0 path), `x = 0`, complex `x` with
  non-real leading entry.
- Givens: the rotation zeroes its target entry to `≤ c·u·‖(a,b)‖` and agrees with
  `LinearAlgebra.givens` on the `Array` path.
- Triangular solves: componentwise backward error `≤ c·n·u` on well conditioned
  triangles; on adversarially graded triangles (diagonals spanning
  `u … 1/u`, the `xLATRS`-style scaling tests) the solve either meets the
  normwise bound `‖Ax̂ − b‖ ≤ c·n·u·‖A‖·‖x̂‖` or overflows in the same cases the
  reference does — silent wrong answers are the failure mode being excluded.

## Phase 2 — One-sided factorizations

The highest-demand gap-fillers for `Metal.jl`, built directly on Phase 1.

**Deliverables**

- `HouseholderQR{T}` holding the packed factors and `τ`; `qr!` unblocked, then
  blocked via `CompactWY{T}` (the `T` factor of `xLARFT`); column-pivoted variant
  returning a permutation; least squares solve for tall systems on top.
- `cholesky!` returning a `Cholesky`-style factor plus a success flag, failing at
  the correct pivot on indefinite input.
- `lu!` with partial pivoting, returning packed `L`/`U` and a pivot vector, with
  singularity reported by index, and the associated solves.

**Phase gate**

- QR: `‖A − QR‖ ≤ c·n·u·‖A‖` and `‖QᴴQ − I‖ ≤ c·n·u`, with the blocked and
  unblocked paths agreeing to the same bound. Pivoted QR additionally: `|R₁₁| ≥
  |R₂₂| ≥ …`, and on rank-deficient inputs (including the Kahan matrix, which
  defeats naive column norms) the numerical rank matches reference.
- Least squares: residual and solution match reference `xGELS` to `c·κ(A)·n·u`
  on full-rank problems, including nearly rank-deficient ones.
- Cholesky: `‖A − LLᴴ‖ ≤ c·n·u·‖A‖` on SPD test matrices across the condition
  range `1 … 1/(n·u)`; indefinite input fails at the same pivot index as
  reference, and never returns a factor silently.
- LU: `‖PA − LU‖ ≤ c·g·n·u·‖A‖` with `g` the growth factor; the Wilkinson
  worst-case growth matrix (`g = 2ⁿ⁻¹`) is included so the bound is honest, not
  hidden; pivot sequence matches reference on the `Array` path; exact singularity
  is reported at the correct column. The `Metal` `Float16` case that segfaults
  through the stdlib fallback completes correctly — this single test is the
  package's reason to exist and is pinned permanently.

## Phase 3 — Symmetric and Hermitian eigenvalue problem

**Deliverables**

- `tridiagonalize!` reducing Hermitian `A` to real `SymTridiagonal` form by
  Householder similarity, retaining the transformation in packed form.
- Implicit QL/QR iteration with Wilkinson shifts on the tridiagonal matrix
  (divide and conquer optional, later), eigenvalues only and
  eigenvalue/eigenvector variants, and back transformation.
- A user-facing `eigen`-shaped entry point for Hermitian matrices.

**Phase gate**

- Per pair: `‖Ax − λx‖ ≤ c·n·u·‖A‖`; eigenvector matrix orthonormal to
  `c·n·u`; eigenvalues sorted and real; `tr(A)` and `‖A‖_F` recovered from the
  spectrum to `c·n·u` relative error.
- Against reference `xSYEVR`/`xHEEV` eigenvalues: agreement to `c·n·u·‖A‖`
  absolute error per eigenvalue.
- Adversarial structure: the Wilkinson `W₂₁⁺` matrix (pathologically close
  eigenvalue pairs), glued Wilkinson matrices (clusters of near-multiplicity),
  matrices with eigenvalue gaps below `u·‖A‖` (orthogonality of the computed
  basis is the criterion there, not vector-wise agreement), and diagonal
  matrices with entries spanning `u … 1/u`.

## Phase 4 — Singular value decomposition

**Deliverables**

- `bidiagonalize!` (Golub–Kahan) to real `Bidiagonal` form with packed `U`/`V`
  transformations.
- Implicit-shift bidiagonal QR SVD with the zero-shift (Demmel–Kahan) path for
  high relative accuracy of small singular values; one-sided Jacobi as an
  alternative where relative accuracy or backend simplicity favors it.
- A user-facing `svd`-shaped entry point; singular values only and full variants.

**Phase gate**

- `‖A − UΣVᴴ‖ ≤ c·n·u·‖A‖`; `U`, `V` orthonormal to `c·n·u`; singular values
  non-negative and non-increasing.
- Against reference `xGESVD`: absolute agreement `≤ c·u·σ₁` per singular value;
  where the Demmel–Kahan or Jacobi path claims high relative accuracy, relative
  agreement `≤ c·n·u` per singular value on graded matrices whose singular
  values span `u² … 1`.
- Adversarial structure: the Kahan matrix, graded matrices (both row- and
  column-scaled), matrices with singular values below `u·σ₁`, and tall/wide
  extremes (`m ≫ n` and `n ≫ m`).

## Phase 5 — Nonsymmetric eigenvalue problem

**Deliverables**

- `balance!` (Osborne scaling and permutation to isolate eigenvalues), and its
  inverse applied to eigenvectors.
- `hessenberg!` by Householder similarity.
- Francis implicit multishift QR with exceptional shifts and aggressive early
  deflation kept as a later refinement; real and complex Schur forms.
- Eigenvalues from the Schur form; eigenvectors by back substitution and back
  transformation.

**Phase gate**

- `‖A − QTQᴴ‖ ≤ c·n·u·‖A‖` with `Q` orthogonal to `c·n·u`; `T` genuinely
  (quasi-)triangular — subdiagonal structure verified, not assumed.
- Eigenvalues against reference `xGEES`/`xGEEV`: matching to `c·n·u·‖A‖` after
  pairing; complex conjugate pairs exactly conjugate in the real Schur form.
- Right eigenvectors: `‖Av − λv‖ ≤ c·n·u·‖A‖` per computed pair.
- Adversarial structure: badly balanced matrices (entries spanning many orders
  of magnitude — the gate for `balance!` is eigenvalue accuracy improving to the
  balanced bound), the Grcar matrix (high eigenvalue sensitivity), Jordan blocks
  and their perturbations (defective and near-defective — the criterion is
  backward error, not forward eigenvalue error), and the classical QR-stagnation
  examples that require exceptional shifts (cyclic permutation matrices).
  Convergence failure must raise an error identifying the unconverged block,
  never return silently.

## Phase 6 — Performance

With correctness continuously validated by the earlier phases, this phase is about
speed.

**Deliverables**

- Blocked variants wherever Phases 1–5 shipped unblocked code, with block sizes
  chosen per backend; elimination of remaining host-side scalar work.
- Mixed-precision iterative refinement (factor in low precision, refine the solve
  in a wider accumulator) for the Phase 2 factorizations — where `Float16` and
  `BFloat16` stop being merely supported and become the fast path.
- The measured crossover table published in the documentation.

**Phase gate**

- The performance goal below holds for every shipped algorithm, evidenced by the
  benchmark suite's crossover table checked into the docs.
- The full test suite passes with `allowscalar(false)` — no host-scalar fallback
  anywhere.
- Refined solves in `Float16`/`BFloat16` reach `Float32`-level backward error on
  systems with `κ(A) ≤ 1/(n·u_half)`, matching the standard iterative-refinement
  guarantee.
- A second GPU backend (`CuArray` or `ROCArray`) passes the full test suite
  unmodified, confirming the abstraction is real.

## Phase 7 — Interface and release

**Deliverables**

- Factorization objects conforming to the `LinearAlgebra.Factorization`
  interface: `\`, `ldiv!`, `size`, `getproperty` accessors (`Q`, `R`, `L`, `U`,
  `S`, `V`, `values`, `vectors`), and in-place variants — wired to the stdlib
  generic names only where doing so pirates nothing (own types, own methods).
- Condition number estimation for the shipped factorizations.
- Complete documentation with doctested examples; registration in General.

**Phase gate**

- An API parity checklist against `LinearAlgebra` for every shipped feature —
  name, argument order, keyword, and return-shape agreement — with every
  deliberate deviation documented in the docstring.
- Aqua piracy check remains clean; every public name docstringed and doctested;
  the registration PR passes General's automerge checks.

## Phase 8 — Autodiff compatibility

Differentiation support, added last because the rules must differentiate the
*mathematical* factorizations, which need to be stable before rules are written
against them.

**Deliverables**

- `ChainRulesCore` rules (`rrule`/`frule`) for the shipped factorizations — QR,
  Cholesky, LU, Hermitian eigen, SVD — following the established matrix-calculus
  results (Giles-style factorization derivatives), written backend-generically:
  no scalar indexing inside a rule, so the same rule serves CPU and GPU.
- Where a factorization is used only as a solver, rules for the solve
  (`A \ b` through the factorization) so gradients avoid differentiating the
  factorization itself.

**Phase gate**

- `ChainRulesTestUtils.test_rrule`/`test_frule` pass for every rule on the
  `Array` path, with finite-difference agreement to the tolerance implied by the
  element type.
- The same rules execute on at least one GPU backend with gradients matching the
  `Array`-path gradients to `c·n·u`.
- Degenerate-direction behavior is defined and tested: repeated eigenvalues and
  singular values (where the derivative is not unique) either follow the
  documented convention or raise a precise error — never return silently wrong
  gradients.

## Later, deliberately unscheduled

Worth doing, not worth sequencing yet: randomized methods (randomized range
finders and randomized SVD are unusually GPU-friendly and may become a phase of
their own), and the polar decomposition.

## Performance goal

Benchmarks live in `benchmarks/` and are direct measurements of performance, not
proxies. The standing target: for every algorithm, the Metal array path runs faster
than the CPU array path once the problem size is large enough, and the crossover
size is measured and reported per algorithm and element type. An algorithm whose
GPU path never overtakes the CPU at any size has a portability or blocking problem
to fix, not a benchmark to excuse.
