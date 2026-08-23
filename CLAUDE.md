# AlgorithmicNLA.jl

## What this project is

A Julia package of numerical linear algebra algorithms that abstract over the
**array backend**. The relationship to `GenericLinearAlgebra.jl` is deliberate:
where that package abstracts over the *element type* so that algorithms run for
`BigFloat`, `Double64`, and friends, this package abstracts over the *array type* so
that the same algorithms run on CPU arrays, `MtlArray`, `CuArray`, and whatever
comes next.

- **Ultimate goal:** portable, high quality linear algebra across GPU backends.
- **Proximate goal:** a library abstract enough to be backend-agnostic and complete
  enough to fill the gaps in `Metal.jl`'s coverage of the `LinearAlgebra` API.

Quality is the point. Prefer a small number of algorithms that are correct,
validated, and well documented over broad coverage that is not.

## Layout

- `src/` — one algorithm per file, included from `src/AlgorithmicNLA.jl`.
  `src/common.jl` holds shared utilities (`realtype`, `unit_roundoff`).
- `test/` — `runtests.jl` includes one `test_<feature>.jl` per feature.
  `test/testsuite.jl` defines `ARRAY_TYPES`, the element type probe `element_types`,
  and the test problem generator `testmatrix`; every test set loops over the array
  types and over the element types each one supports, so a backend and an element
  type are validated by the same tests as the reference path.
- `docs/src/roadmap.md` — the coarse phase ordering of the work and the standing
  performance goal. `docs/` is a Documenter.jl project (`make.jl`, `src/`)
  deployed by CI.
- `benchmarks/` — the BenchmarkTools suite (`benchmarks.jl` defines `SUITE`,
  `runbenchmarks.jl` runs it).
- `.plan/current.md` — gitignored working memory for the feature in progress.
- `.claude/skills/plan-feature/` — the workflow for adding a feature.

## Workflow

Adding or planning any algorithm goes through the `plan-feature` skill. It defines
the sequence: scope the feature, survey reference implementations, fix the
interface, define numerical completion criteria, write `.plan/current.md`, implement,
integrate. Read `.plan/current.md` at the start of a session to recover context.

Run the tests with:

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Development cycle standards

Two different questions are answered at two different times, and they are not
conflated: pull request CI answers "is this change safe to merge"; release-time
checks answer "is this state good enough to publish."

### Pull request gates (blocking — main stays green by construction)

All of these run in CI on every PR and must pass before merging into `main`:

1. **Tests** — `Pkg.test()` on the oldest supported Julia (see `[compat]`) and
   the latest stable release, on Linux and macOS. Coverage is uploaded to
   Codecov; a change that lowers coverage needs a reason.
2. **QA** — `Aqua.test_all(AlgorithmicNLA)` (method ambiguities, unbound type
   parameters, undefined exports, stale dependencies, missing compat bounds,
   type piracy) plus ExplicitImports (`check_no_implicit_imports`,
   `check_no_stale_explicit_imports` — every `using` names what it imports).
   Reproduce locally with:

   ```
   julia -e 'using Pkg; Pkg.activate(temp = true); Pkg.develop(path = "."); Pkg.add(["Aqua", "ExplicitImports"]);
             using Aqua, ExplicitImports, AlgorithmicNLA; Aqua.test_all(AlgorithmicNLA);
             check_no_implicit_imports(AlgorithmicNLA); check_no_stale_explicit_imports(AlgorithmicNLA)'
   ```

3. **Formatting** — all Julia source must be formatted with
   [Runic](https://github.com/fredrikekre/Runic.jl). Install it once into a shared
   environment (`julia -e 'using Pkg; Pkg.activate("runic", shared = true); Pkg.add("Runic")'`),
   then check with `julia --project=@runic -m Runic --check --diff .` and fix with
   `julia --project=@runic -m Runic --inplace .`.
4. **Docs build** — `julia --project=docs docs/make.jl` must build without
   errors (Documenter is strict: broken doctests, broken cross-references, and
   docstrings missing from the pages all fail the build). Every exported name
   has a docstring that appears in `docs/src/api.md`.

### Benchmarks (informational at PR time, enforced at release time)

The benchmark suite in `benchmarks/` runs on PRs and weekly
(`.github/workflows/Benchmarks.yml`). Timing on shared runners is noisy, so the
job never blocks a merge — but a regression flagged at PR time is reviewed by a
human before merging, and the roadmap performance goal (Metal beats CPU beyond a
measured crossover size) is verified on real hardware before any release. A new
algorithm lands with benchmark entries in `SUITE`, keyed by backend, element
type, and size.

### Publishing (not gating)

- Pushes to `main` deploy the `/dev` docs; tags deploy versioned docs. Nothing
  is decided at these stages that PR CI did not already decide.
- **Releasing a version:** update `CHANGELOG.md`, bump `version` in
  `Project.toml` following SemVer (any export removed or changed is a breaking
  bump), verify the benchmark goals above, then register (Registrator) —
  TagBot tags and triggers the versioned docs deploy. CompatHelper runs on a
  schedule and keeps `[compat]` current; new dependencies always get a bounded
  compat entry (Aqua enforces this).

## Development best practices

### 1. Follow existing high quality libraries

Before designing anything, look at how it has already been done well, and reuse it.
`GenericLinearAlgebra.jl` is the primary model for module layout, naming, algorithm
structure, and test organization. `LinearAlgebra` (stdlib) defines the API surface to
match: function names, argument order, keyword arguments, factorization types, and
`Factorization` accessors. LAPACK is the algorithmic reference for storage
conventions, scaling, and edge case handling. Reuse their correctness, validation,
and robustness measures — the error bounds their tests assert, the invariances they
account for, the degenerate inputs they cover — rather than inventing our own. A
deviation from an established interface needs a stated reason.

### 2. Mathematical clarity in names

Function and variable names match the nomenclature of the literature and of LAPACK
as closely as the language allows: `A`, `Q`, `R`, `tau`, `householder`, `bidiagonalize`,
`wilkinson_shift`. A reader who knows Golub & Van Loan should recognize the code.

Avoid special symbols and non-ASCII characters unless they are already used for the
same purpose in the standard library (`I`, `∘`, `⋅`, `'` are fine because stdlib uses
them; `σ`, `λ`, `α` as variable names are not — write `sigma`, `lambda`, `alpha`).
Do not abbreviate past the point of recognition, and do not expand standard
abbreviations into prose.

### 3. Docstrings and comments

Docstrings are mathematically precise and concise. State what is computed, the
conditions on the input, and the properties of the output. Use a term only in its
precise mathematical sense: *orthogonal*, *unitary*, *nonsingular*, *positive
definite*, *backward stable*, *rank* mean exactly what they mean in the literature.
Do not use a vague word where a precise counterpart exists — not "roughly equal" but
a stated bound; not "fast" but a stated complexity; not "stable" unless the stability
is the documented kind.

No marketing, no restating the signature in words, no examples unless the usage is
genuinely non-obvious.

Code comments are the exception, not the rule. Add one only where something truly
needs explaining — a non-obvious scaling to avoid overflow, a deviation from the
textbook algorithm, a backend-specific workaround — and write it to the same
precision standard as a docstring.

### 4. Element type genericity

Every algorithm must work for every element type the backend supports, and adding a
new element type must cost nothing beyond one line in `CANDIDATE_ELEMENT_TYPES` in
the test harness.

- Never dispatch on, or annotate an argument with, a concrete floating point type.
  Write `T<:Number` and derive everything else from it.
- Derive constants and tolerances from the element type: `zero(T)`, `one(T)`,
  `T(2)`, `realtype(T)`, `unit_roundoff(T)`. No floating point literals in algorithm
  code, no fixed tolerances such as `1e-8`, and no `eps()` without an argument.
- Do not assume the element type is a double precision hardware float. The supported
  set includes `Float16` and `BFloat16`, whose unit roundoff is about `5e-4` and
  `4e-3`; a threshold or convergence test tuned to double precision silently fails
  for them. It also includes `BigFloat`, which has no fixed precision at all.
- Where accumulation in the element type loses too much accuracy to be useful — a
  long inner product in `Float16`, say — widen the accumulator deliberately and say
  so in the docstring. Do not widen silently and do not refuse the type.
- Which element types a backend supports is discovered by probing (`element_types`),
  not hard coded, because it varies by backend, device, and driver version.

The failure this guards against is real: with `Metal` and `LinearAlgebra` loaded,
`lu(Metal.ones(Float16, 10, 10))` segfaults, even though the array is valid and
arithmetic on it works. Generic fallbacks that reach code assuming BLAS element
types crash rather than erroring. Testing every algorithm on every supported element
type is what catches this class of failure here instead of in a user's process.

### 5. Backend portability

Algorithms are written against the `AbstractArray` interface. No scalar indexing of
backend arrays in inner loops; express work as broadcasts, reductions, views, and
`mul!`. Keep the set of operations a backend must support small and record it when
planning a feature — every entry is a portability constraint. Write and validate the
`Array` path against LAPACK first, then generalize.
