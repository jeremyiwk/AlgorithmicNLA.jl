# Shared helpers for running every test set against multiple array backends.

using AlgorithmicNLA
using BFloat16s
using LinearAlgebra
using Random
using StableRNGs
using Test

using AlgorithmicNLA: realtype, unit_roundoff

"""
    ARRAY_TYPES

Array types the test suite runs against. `Array` is always present; to test a
backend, load its package here and append its array type (for example `MtlArray`).
Every test set loops over this list, so a backend is validated by the same tests as
the reference implementation.
"""
const ARRAY_TYPES = Any[Array]

"""
    CANDIDATE_ELEMENT_TYPES

Every element type the suite knows how to build a test problem for. Which of them a
given backend actually supports is decided at run time by `element_types`;
supporting a new element type means adding it to this tuple and nothing else.
"""
const CANDIDATE_ELEMENT_TYPES = (
    Float16, Float32, Float64, BFloat16, BigFloat,
    ComplexF16, ComplexF32, ComplexF64,
    Complex{BFloat16}, Complex{BigFloat},
)

"""
    element_types(ArrayType)

Those of `CANDIDATE_ELEMENT_TYPES` that `ArrayType` can store, broadcast over, and
reduce, determined by trying each one. Backends differ in what they support and the
set can change with a driver or a device, so it is probed rather than fixed: Apple
GPUs, for instance, have no double precision unit but do support `Float16` and
`BFloat16`.

An algorithm is expected to work for every type this returns. Nothing in `src` may
dispatch on a concrete floating point type, so a new element type costs no work
beyond a line in `CANDIDATE_ELEMENT_TYPES`.
"""
function element_types(::Type{AT}) where {AT}
    return filter(collect(CANDIDATE_ELEMENT_TYPES)) do T
        try
            a = AT(ones(T, 4))
            sum(a .+ a)
            true
        catch
            false
        end
    end
end

"""
    referencetype(T)

The double precision element type corresponding to `T`: `Float64` for real `T` and
`ComplexF64` for complex `T`. A result computed in a low precision element type is
validated against a reference computed in this type, with the tolerance scaled by
`unit_roundoff(T)` rather than by a constant.
"""
referencetype(::Type{<:Real}) = Float64
referencetype(::Type{<:Complex}) = ComplexF64

"""
    uniform(rng, T, dims...)

An array of independent entries uniform on `[-1, 1]` for real `T`, and with real and
imaginary parts independent and uniform on `[-1, 1]` for complex `T`.
"""
function uniform(rng::AbstractRNG, ::Type{R}, dims::Integer...) where {R <: Real}
    return 2 .* rand(rng, R, dims...) .- one(R)
end

function uniform(rng::AbstractRNG, ::Type{Complex{R}}, dims::Integer...) where {R <: Real}
    return complex.(uniform(rng, R, dims...), uniform(rng, R, dims...))
end

"""
    testmatrix(ArrayType, T, m, n; seed)

An `m`-by-`n` matrix of type `ArrayType` with element type `T` and independent
entries drawn uniformly from `[-1, 1]` (or its complex analogue). The entries are
generated in double precision and then rounded to `T`, so the same `seed` gives the
same matrix in every element type and results across precisions are directly
comparable.
"""
function testmatrix(::Type{AT}, ::Type{T}, m::Integer, n::Integer; seed = 0) where {AT, T}
    rng = StableRNG(seed)
    A = uniform(rng, referencetype(T), m, n)
    return convert(AT{T}, T.(A))
end

"""
    TEST_SEEDS

Seeds every property and identity test loops over. Random matrices are almost
always well conditioned, so a single instance demonstrates nothing; identities and
invariances are asserted over this whole family (and, where meaningful, over the
`condition_sweep` and the named adversarial matrices) rather than on one draw.
"""
const TEST_SEEDS = 0:7

"""
    condition_sweep(T)

Condition numbers for stress testing in element type `T`: well conditioned (`1`),
moderately ill conditioned (`u^-1/2`), and close to numerically singular
(`u^-1 / 10`), where `u` is the unit roundoff of `T`. Derived from the element type
so the sweep is meaningful in every precision: `u^-1/2` is about `1e8` for
`Float64` but only about `45` for `Float16`.
"""
function condition_sweep(::Type{T}) where {T}
    u = Float64(unit_roundoff(T))
    return (1.0, inv(sqrt(u)), inv(u) / 10)
end

"""
    conditioned_testmatrix(ArrayType, T, m, n; kappa, seed)

An `m`-by-`n` matrix with prescribed 2-norm condition number `kappa`: random
orthogonal factors around a geometrically graded diagonal (singular values from `1`
down to `1 / kappa`), in the manner of LAPACK's `xLATMS`. Generated in double
precision and rounded to `T`, so the same `seed` gives the same matrix in every
element type. `kappa` should not exceed the largest value in `condition_sweep(T)`,
beyond which the grading is unrepresentable in `T`. When `min(m, n) == 1` there is
a single singular value and the condition number is `1` regardless of `kappa`.
"""
function conditioned_testmatrix(
        ::Type{AT}, ::Type{T}, m::Integer, n::Integer;
        kappa, seed = 0
    ) where {AT, T}
    rng = StableRNG(seed)
    R = referencetype(T)
    k = min(m, n)
    sigma = kappa .^ .-range(0.0, 1.0, length = max(k, 2))[1:k]
    U = Matrix(qr(uniform(rng, R, m, k)).Q)
    V = Matrix(qr(uniform(rng, R, n, k)).Q)
    A = U * Diagonal(R.(sigma)) * V'
    return convert(AT{T}, T.(A))
end

"""
    within_baseline(err, referr, n, T, scale; factor = 4)

Whether an error is acceptable relative to a reference implementation's error on
the same problem. The test is

    err ≤ factor * max(referr, n * u * scale)

with `u = unit_roundoff(T)`, floored at one expected unit `n * u * scale` so that
a reference which is anomalously exact on a structured input (a triangular or
orthogonal matrix, say) does not turn an acceptable rounding error into a failure.
`scale` is the natural size of the quantity being measured — `norm(A)` for a
factorization residual, `1` for an orthogonality defect.

The default `factor = 4` is chosen for implementations of the *same algorithm* as
the reference: backward error then differs only through evaluation order and
blocking, which move the constant by factors near `√2`–`2` and essentially never
beyond `3`, while a genuine defect grows with `n`, `κ`, or `1/u` and exceeds any
constant factor immediately. A deliberately different algorithm with a genuinely
different error constant may pass a larger `factor` explicitly, with a comment
saying why.

This is the baseline layer of testing: it bounds how much worse than the reference
we are, on the same input, independent of the mathematical bound the same test set
asserts. Reference errors come from LAPACK via `LinearAlgebra` where the element
type has a LAPACK path (`Float32`, `Float64`, and their complex types) and from the
same computation carried out in `referencetype(T)` otherwise.
"""
function within_baseline(err, referr, n::Integer, ::Type{T}, scale; factor = 4) where {T}
    u = Float64(unit_roundoff(T))
    return Float64(err) <= factor * max(Float64(referr), n * u * Float64(scale))
end

"""
    compare_spectra(computed, reference; rtol)

Whether two collections of eigenvalues or singular values agree to `rtol` relative
to the largest reference magnitude, after sorting both by `(real, imag)` so the
comparison is invariant to output ordering. Complex conjugate pairs must agree
pairwise. This is the only meaningful direct comparison for spectra; individual
eigenvector comparison is not (see `subspace_distance`).
"""
function compare_spectra(computed, reference; rtol)
    length(computed) == length(reference) || return false
    by = x -> (real(x), imag(x))
    c, r = sort(collect(computed); by), sort(collect(reference); by)
    scale = maximum(abs, r; init = 0.0)
    return all(abs.(c .- r) .<= rtol * max(scale, one(scale)))
end

"""
    matches_up_to_phase(u, v; atol)

Whether two unit vectors agree up to a unimodular phase (a sign, in the real
case): `min over phase of norm(u - phase * v) ≤ atol`, computed as
`norm(u - sign(dot(v, u)) * v)`. Eigenvectors and singular vectors are defined
only up to phase, so direct comparison without this alignment is meaningless.
Only valid for well separated eigenvalues; for clustered ones compare the spanned
subspaces with `subspace_distance` instead.
"""
function matches_up_to_phase(u, v; atol)
    s = dot(v, u)
    phase = iszero(s) ? one(s) : s / abs(s)
    return norm(u .- phase .* v) <= atol
end

"""
    subspace_distance(U, V)

The sine of the largest principal angle between the column spans of the orthonormal
`U` and `V`. This is the correct way to compare computed eigenvector or singular
vector bases when eigenvalues are clustered: the individual vectors within a
cluster are not well determined, but the spanned subspace is, and this distance is
small exactly when the subspaces agree. Near-zero angles are resolved only to
about `sqrt(u)` of the working precision (the `sqrt(1 - s^2)` form loses half the
digits there), so tolerances on this quantity must not be tighter than that.
"""
function subspace_distance(U, V)
    s = svdvals(U' * V)
    smin = isempty(s) ? one(real(eltype(U))) : minimum(s)
    return sqrt(max(zero(smin), one(smin) - smin^2))
end

# Make the exercised configuration visible in every test log, so a coverage
# regression (a backend or element type silently dropping out) is observable.
@info "AlgorithmicNLA test configuration" [AT => element_types(AT) for AT in ARRAY_TYPES]
