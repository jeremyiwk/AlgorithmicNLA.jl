# Array interface

`NLAArray(A)` wraps existing storage without copying or converting its elements.
Use `parent(A)` to recover the underlying array. The wrapper is the dispatch
boundary for AlgorithmicNLA implementations of standard `LinearAlgebra` functions.

```jldoctest
julia> using AlgorithmicNLA, LinearAlgebra

julia> storage = [0.0 2.0; 1.0 3.0]; A = NLAArray(storage);

julia> parent(A) === storage
true

julia> B = copy(A); view(B, :, 1) .= 0;

julia> Array(A)
2×2 Matrix{Float64}:
 0.0  2.0
 1.0  3.0

julia> lu(A).factors == lu(storage).factors
true
```

Copies, allocation through `similar`, slices, views, reshapes, transpose/adjoint,
broadcasting, `map`, and array-valued `mapreduce` results preserve the wrapper.
Views and reshapes share storage and preserve its identity for alias detection,
even when the backend's derived arrays do not. Broadcast assignments copy aliased
inputs when needed before executing through the parent's broadcast implementation.
Wrap storage before creating related views: independently wrapping aliases created
by the backend still depends on that backend's storage identity reporting.
Products and `mul!` with wrapped
matrix/vector operands delegate to the corresponding parent operation.

Explicit backend conversion uses `Adapt.adapt`, for example
`adapt(MtlArray, NLAArray(A))` or `adapt(Array, wrapped_device_array)`. Adaptors can
also request an explicit element-type conversion. Scalar indexing delegates to
the parent, so a device array still rejects it when scalar indexing is disabled.

## Current factorization coverage

LU and its in-place variant select the portable numerical implementation described
in [LU factorization](lu.md), also available as `AlgorithmicNLA.lu` and
`AlgorithmicNLA.lu!` on ordinary backend arrays.

The wrapper forwards `qr` and `qr!` to the parent backend, including supported
pivot strategies and keywords. These return the backend's native factorization
objects. Returned floating-point factor storage must have the input element type;
a change raises `ArgumentError` after the delegated call, without rolling back
in-place mutations. Backend exceptions propagate unchanged.

Cholesky, other decompositions, mixed wrapped/unwrapped matrix products, and
arbitrary nesting of standard structural wrappers are not yet covered by the
wrapper's dispatch contract. They need validation as features are added.

## Metal validation

The separate test environment runs the same array tests on CPU and Metal with
scalar indexing disabled. It requires a working Metal device and fails if one is
unavailable:

```sh
julia --project=test/metal -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test/metal test/metal/runtests.jl
```

CPU-only CI continues to use `Pkg.test()`. The Metal environment validates array operations, backend adaptation, and the
LU numerical and interface tests.
