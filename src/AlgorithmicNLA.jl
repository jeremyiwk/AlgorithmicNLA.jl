"""
    AlgorithmicNLA

Numerical linear algebra algorithms written against the `AbstractArray` interface so
that they run on any array backend supporting a small set of vectorized primitives.

The package is organized as a set of algorithm files, each implementing one
factorization or decomposition, included below.
"""
module AlgorithmicNLA

using LinearAlgebra

include("common.jl")

end # module
