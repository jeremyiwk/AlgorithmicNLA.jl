# Roadmap

A coarse ordering of the work. Phases are sequenced so that each one supplies the
building blocks the next one needs; within a phase, features are planned and
implemented one at a time via the `plan-feature` workflow.

## Cross-cutting requirements

These apply to every phase rather than being phases themselves:

- **Backend validation is continuous, not deferred.** Every algorithm is written
  against the `AbstractArray` interface, validated on the `Array` path against
  LAPACK first, and then run against at least one GPU backend (Metal) before it is
  considered done. Discovering that the abstraction fails is cheap per feature and
  expensive after six phases.
- **Benchmarks land with the feature.** Each algorithm adds entries to the
  `benchmarks/` suite keyed by backend, element type, and size, so the crossover
  measurement exists from day one rather than being reconstructed at release time.
- **Docs land with the feature.** Each algorithm's docstring appears in the API
  reference when it merges.

## Phase 0 — Foundations *(complete)*

Package scaffolding, the backend-parametrized test harness, and the conventions for
accuracy testing (residual, orthogonality, and backward error bounds measured in
units of the unit roundoff). Element types are discovered by probing the backend, so
every algorithm is tested against everything the backend supports and a new element
type costs no work.

## Phase 1 — Primitives

The building blocks everything later reuses: Householder reflectors and their
application, Givens rotations, and triangular solves (`ldiv!`/`rdiv!` for upper and
lower triangular systems, unblocked then blocked). Triangular solves sit here, not
after the eigensolvers, because least squares in Phase 2 already requires them and
because they are among the operations GPU backends most commonly lack.

## Phase 2 — One-sided factorizations

The highest-demand gap-fillers for `Metal.jl`, built directly on Phase 1:

- QR — unblocked, then blocked (compact WY), with and without column pivoting,
  and least squares solves on top.
- Cholesky — the simplest full factorization and the entry point for `\` on
  symmetric positive definite systems.
- LU with partial pivoting and the associated solves. This is the factorization
  whose generic fallback segfaults today for `Metal` `Float16` arrays, so it is
  the clearest single measure of the package's reason to exist.

## Phase 3 — Symmetric and Hermitian eigenvalue problem

Reduction to tridiagonal form, followed by an eigenvalue iteration (implicit QL/QR
with Wilkinson shifts, and/or divide and conquer) and back transformation of
eigenvectors.

## Phase 4 — Singular value decomposition

Bidiagonalization and the implicit-shift bidiagonal SVD, plus a one-sided Jacobi
path where high relative accuracy or backend simplicity favors it.

## Phase 5 — Nonsymmetric eigenvalue problem

Balancing, reduction to Hessenberg form, the Francis multishift QR iteration, real
and complex Schur factorizations, and eigenvector computation by back substitution.

## Phase 6 — Performance

With correctness continuously validated by the earlier phases, this phase is about
speed: blocking for the memory hierarchy, replacing any remaining host-side scalar
work with vectorized or kernel-based equivalents, and mixed-precision iterative
refinement, which is where the half-precision element types stop being merely
supported and start being the fast path. Exit criterion: the performance goal below
holds for every shipped algorithm, and a second GPU backend (`CuArray` or
`ROCArray`) passes the full test suite to confirm the abstraction is real.

## Phase 7 — Interface and release

Full coverage of the relevant `LinearAlgebra` API (factorization objects, `\`,
in-place variants), condition number estimation for the shipped factorizations,
documentation, and registration in General.

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
