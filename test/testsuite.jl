# Shared helpers for running every test set against multiple array backends.

using AlgorithmicNLA
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
    ELEMENT_TYPES

Element types every backend is expected to support.
"""
const ELEMENT_TYPES = (Float32, Float64, ComplexF32, ComplexF64)

"""
    testmatrix(ArrayType, T, m, n; seed)

An `m`-by-`n` matrix of type `ArrayType` with element type `T` and independent
entries drawn uniformly from `[-1, 1]` (or its complex analogue).
"""
function testmatrix(::Type{AT}, ::Type{T}, m::Integer, n::Integer; seed = 0) where {AT,T}
    rng = MersenneTwister(seed)
    A = 2 .* rand(rng, T, m, n) .- one(T)
    return convert(AT{T}, A)
end
