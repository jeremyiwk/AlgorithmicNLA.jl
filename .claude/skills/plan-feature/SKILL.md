---
name: plan-feature
description: Plan, specify, implement, and integrate one AlgorithmicNLA feature (a factorization, decomposition, solver, or supporting kernel). Use when starting new numerical linear algebra work, when the user asks to add or plan an algorithm, or when picking up the next item from docs/src/roadmap.md.
---

# Planning a feature

One feature at a time. A feature is one algorithm plus the interface that exposes
it: for example Householder reflectors, QR with column pivoting, or the bidiagonal
SVD. Do not start a second feature until the current one meets its completion
criteria.

The output of the planning stage is `.plan/current.md` (gitignored working memory).
The output of the implementation stage is source, tests, docstrings, and an updated
roadmap.

## 1. Scope the feature

Read `.plan/current.md` and `docs/src/roadmap.md`. Confirm which feature is next and
what it depends on. If a dependency is missing, that dependency is the feature.

State in one sentence what the feature computes, in the language of the literature.

## 2. Survey reference implementations

This is not optional and it comes before any design decision. The goal is to reuse
established interfaces and established correctness measures rather than invent them.

Consult, in roughly this order:

- `LinearAlgebra` (stdlib) — the API this package must be compatible with:
  function names, argument order, keyword arguments, factorization types, and the
  `Factorization` accessors (`Base.getproperty`, `size`, `\`, `adjoint`).
- `GenericLinearAlgebra.jl` — the closest sibling project. Follow its module layout,
  naming, algorithm structure, and test organization wherever it applies.
- LAPACK — the algorithmic reference. Identify the corresponding driver and
  computational routines (for example `geqrf`, `orgqr`, `steqr`, `bdsqr`) and their
  documented conventions for storage, scaling, and edge cases.
- The GPU array ecosystem (`GPUArrays.jl`, `CUDA.jl`, `Metal.jl`) — how comparable
  routines avoid scalar indexing and how they structure backend-agnostic code.
- The literature — Golub & Van Loan, *Matrix Computations*; Higham, *Accuracy and
  Stability of Numerical Algorithms*; Demmel, *Applied Numerical Linear Algebra* —
  for the algorithm statement, its error analysis, and the bounds the tests assert.

Record in `.plan/current.md` what was consulted and which specific conventions,
guarantees, and test measures are being adopted from each.

## 3. Define the interface

Write the exact signatures before implementing:

- The exported function and its in-place variant (`f` and `f!`), matching stdlib
  naming and argument order.
- The returned type. Reuse a `LinearAlgebra` factorization type if it is generic
  enough; define a new one only with a stated reason.
- The set of methods the array backend must support (which broadcasts, reductions,
  `mul!` forms, views). Keep this set as small as possible; every entry is a
  portability constraint. No scalar indexing of backend arrays in inner loops.
- Nothing about the element type. Signatures are generic in `T<:Number`; a concrete
  floating point type must not appear in a method signature, and every constant,
  threshold, and tolerance is derived from `T` through `realtype` and
  `unit_roundoff`. If the algorithm genuinely cannot be written for some supported
  element type, that restriction is a documented design decision with a stated
  reason, not an untested assumption.

## 4. Define completion criteria

Criteria must be checkable, and stated as numerical bounds rather than as "works".
Adopt the measures used by the reference implementations. Typically:

- **Backward error.** `norm(A - reconstruct(F)) / norm(A)` is a modest multiple of
  `unit_roundoff(eltype(A))`, with the multiple growing at most linearly in the
  problem dimension.
- **Orthogonality.** `norm(Q'Q - I)` for every computed orthogonal or unitary
  factor, to the same standard.
- **Agreement with a trusted reference.** On `Array` inputs, results agree with
  LAPACK via `LinearAlgebra` up to the accuracy of the problem and the appropriate
  invariances (sign, phase, ordering, and the choice of basis within an eigenspace).
- **Backend parity.** The same test set passes for every type in `ARRAY_TYPES`.
- **Element type coverage.** The same test set passes for every type in
  `element_types(AT)` for each backend, real and complex, including the low
  precision types. Tolerances scale with `unit_roundoff(T)`, so the same assertion
  holds in `Float16` and in `Float64` without a separate case. A new element type
  must require no change to `src`.
- **Edge cases.** Empty and 1-by-1 inputs, tall and wide rectangles, exact rank
  deficiency, repeated and clustered eigenvalues or singular values, entries scaled
  near the overflow and underflow thresholds, and matrices that are already in the
  target form.
- **Type stability and no unintended allocation** in the in-place variant.

## 5. Write the plan

Overwrite `.plan/current.md` with: the feature, its status, references consulted,
the interface, the completion criteria as a checklist, the implementation steps as a
checklist, and open questions. Keep it terse; it is a memory aid, not a document.

## 6. Implement

Follow the steps in the plan, checking them off as they land. One algorithm per file
in `src/`, named after the algorithm. Write the reference (`Array`) path first and
confirm it against LAPACK before generalizing to other backends.

Observe the conventions in `CLAUDE.md`: mathematical names, precise and concise
docstrings, minimal comments.

## 7. Integrate

A feature is not done until all of the following hold:

- The algorithm file is `include`d in `src/AlgorithmicNLA.jl` and its public names
  are exported.
- Every exported name has a docstring stating what is computed, the conditions on
  the input, and the properties of the output.
- A matching `test/test_<feature>.jl` is included from `test/runtests.jl` and loops
  over `ARRAY_TYPES` and, for each, over `element_types(AT)`.
- `Pkg.test()` passes.
- `docs/src/roadmap.md` reflects the new state.
- `.plan/current.md` is updated: criteria checked off, and the next feature named.
