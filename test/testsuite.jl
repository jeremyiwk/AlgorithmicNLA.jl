# Shared helpers for running every test set against multiple array backends.

using AlgorithmicNLA
using BFloat16s
using LinearAlgebra
using Random
using Test

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
    rng = MersenneTwister(seed)
    A = uniform(rng, referencetype(T), m, n)
    return convert(AT{T}, T.(A))
end
