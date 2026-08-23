# AlgorithmicNLA.jl

Numerical linear algebra algorithms that abstract over the **array backend**.
Where `GenericLinearAlgebra.jl` abstracts over the *element type* so that
algorithms run for `BigFloat`, `Double64`, and friends, this package abstracts
over the *array type* so that the same algorithms run on CPU arrays, `MtlArray`,
`CuArray`, and whatever comes next.

- **Ultimate goal:** portable, high quality linear algebra across GPU backends.
- **Proximate goal:** a library abstract enough to be backend-agnostic and
  complete enough to fill the gaps in `Metal.jl`'s coverage of the
  `LinearAlgebra` API.

Quality is the point: a small number of algorithms that are correct, validated,
and well documented over broad coverage that is not.

## Getting started

The package is registered nowhere yet; install it by developing the repository:

```julia
using Pkg
Pkg.develop(url = "https://github.com/jeremyiwk/AlgorithmicNLA.jl")
using AlgorithmicNLA
```

Algorithms are written against the `AbstractArray` interface and follow the
naming and calling conventions of the `LinearAlgebra` standard library. Every
algorithm works for every element type the backend supports, including
`Float16`, `BFloat16`, and `BigFloat`; constants and tolerances are derived
from the element type, never hard coded.

## Design

- One algorithm per file under `src/`, validated first against LAPACK on the
  `Array` path and then generalized.
- No scalar indexing of backend arrays in inner loops; work is expressed as
  broadcasts, reductions, views, and `mul!`.
- Element type support is discovered by probing the backend, not hard coded,
  because it varies by backend, device, and driver version.

See the [Roadmap](@ref) for the phase ordering of the work and the
[API reference](@ref) for what is implemented today.
