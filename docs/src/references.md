# References

The bounded set of sources this project works from. The survey step of every
feature consults these and only these; adding a source is a deliberate decision
recorded here with a reason, not an ambient accumulation. The collection below is
sufficient for every phase of the roadmap.

## Literature

- **Golub & Van Loan, *Matrix Computations* (4th ed.)** — the primary algorithmic
  reference: algorithm statements, blocked formulations, and operation counts.
- **Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed.)** — the
  error-analysis reference: every tolerance and backward-error bound asserted in
  the tests should be traceable to a result here (or to the LAPACK
  documentation).
- **Stewart, *Matrix Algorithms* I (Basic Decompositions) and II
  (Eigensystems)** — pseudocode-level algorithm detail; often the clearest
  bridge between the textbook statement and an implementation.
- **Wilkinson, *The Algebraic Eigenvalue Problem*** — classical convergence
  arguments and the origin of many of the adversarial test matrices the phase
  gates name.
- **Stewart & Sun, *Matrix Perturbation Theory*** — perturbation bounds; used
  when a gate needs a statement about clustered eigenvalues or subspace
  sensitivity, not for day-to-day implementation.
- **Demmel & Kahan, “Accurate Singular Values of Bidiagonal Matrices” (1990)** —
  the zero-shift bidiagonal QR algorithm of Phase 4; short and readable.

## Code and documentation

- **LAPACK** — [source](https://github.com/Reference-LAPACK/lapack), the
  [Users' Guide](https://www.netlib.org/lapack/lug/), and the
  [LAPACK Working Notes](https://www.netlib.org/lapack/lawns/) — the
  implementation reference: storage conventions, scaling strategies
  (`xLARFG`, `xLASSQ`, `xLATRS`), edge-case handling, and the test-ratio
  methodology the harness's `within_baseline` imitates.
- **`LinearAlgebra` (stdlib)** — the API surface to match: names, argument
  order, factorization types, accessors.
- **`GenericLinearAlgebra.jl`** — the closest sibling: module layout, naming,
  and the reference for element-type-generic algorithm structure; also the
  same-precision reference implementation for element types LAPACK cannot run.
- **`GPUArrays.jl` / `Metal.jl` / `CUDA.jl`** — how backend-agnostic array code
  avoids scalar indexing and what primitives backends actually provide.

## Deliberately not in the set

*Applied Numerical Linear Algebra* (Demmel) and other general texts overlap the
books above without adding coverage this roadmap needs; direct sparse methods
(Davis) belong to `MetalSparseArrays.jl`'s domain and are a non-goal even there.
Grow this page only when a phase demonstrably needs a source the set lacks.
