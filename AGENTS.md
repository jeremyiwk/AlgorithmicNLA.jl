# Repository guidance

Read `CLAUDE.md` and `.plan/current.md` at the start of a session when present.
They hold the detailed development conventions and current feature decisions.
Read `docs/src/roadmap.md` and `docs/src/references.md` when planning numerical
work. Apply `docs/src/performance.md` when planning each numerical feature and
reviewing performance changes; record the bounded experiments or why they do
not apply. Local workflows live in `.claude/skills/`; these files and `CLAUDE.md` are
gitignored and may be absent in a fresh checkout.

- Match `LinearAlgebra` in API and algorithm as closely as possible; where
  backend portability prevents that, match LAPACK. Use a suboptimal variant only
  where LAPACK also uses it for the corresponding operation and problem regime.
- Match LAPACK's numerical tests and acceptance thresholds. Cite the routines
  and document adaptations for unsupported element types.
- Write backend-generic array code using views, broadcasts, reductions, and
  `mul!`. Validate device execution with scalar indexing disabled. Avoid type
  piracy; reuse standard wrappers and factorization types when their methods
  permit the required behavior.
- Develop LU first, then Cholesky, then QR, adding prerequisites as needed.
- Ask unresolved design questions one at a time, retain the answers in
  `.plan/current.md`, and do not ask again about decisions already made.

Run `julia --project=. -e 'using Pkg; Pkg.test()'` for implementation changes.
Feature work also includes numerical reference tests, GPU validation, benchmarks,
and API documentation; see `CLAUDE.md` for the full checks when available.
