# Roadmap

A coarse ordering of the work. Phases are sequenced so that each one supplies the
building blocks the next one needs; within a phase, features are planned and
implemented one at a time via the `plan-feature` workflow.

## Phase 0 — Foundations

Package scaffolding, the backend-parametrized test harness, and the conventions for
accuracy testing (residual, orthogonality, and backward error bounds measured in
units of the unit roundoff).

## Phase 1 — Orthogonal transformations

Householder reflectors and Givens rotations, their blocked (compact WY)
representations, and QR factorization with and without column pivoting. Least
squares solves built on top.

## Phase 2 — Symmetric and Hermitian eigenvalue problem

Reduction to tridiagonal form, followed by an eigenvalue iteration (implicit QL/QR
with Wilkinson shifts, and/or divide and conquer) and back transformation of
eigenvectors.

## Phase 3 — Singular value decomposition

Bidiagonalization and the implicit-shift bidiagonal SVD, plus a one-sided Jacobi
path where high relative accuracy or backend simplicity favors it.

## Phase 4 — Nonsymmetric eigenvalue problem

Reduction to Hessenberg form, the Francis multishift QR iteration, real and complex
Schur factorizations, and eigenvector computation by back substitution.

## Phase 5 — Triangular factorizations and solves

LU with partial pivoting, Cholesky, and the associated triangular solves, provided
for backends that do not supply them.

## Phase 6 — Backend portability

Replacing any remaining host-side scalar work with vectorized or kernel-based
equivalents, blocking for memory hierarchy, and validating the full test suite on at
least one GPU backend (Metal first, then a second backend to confirm the abstraction
holds).

## Phase 7 — Interface and release

Full coverage of the relevant `LinearAlgebra` API (factorization objects, `\`,
in-place variants), documentation, and registration.
