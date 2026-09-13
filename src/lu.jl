lu_abs1(x) = abs(real(x)) + abs(imag(x))
lu_nonzero(x) = !iszero(x)

function lu_safemin(::Type{T}) where {T}
    small = inv(floatmax(T))
    tiny = floatmin(T)
    return small >= tiny ? small * (one(T) + eps(T)) : tiny
end

# LAPACK 3.12.1 LADIV/LADIV1/LADIV2 scaling, evaluated in at least Float32.
# The LAPACK copyright and BSD license are reproduced in licenses/LAPACK.txt.
# Julia's ComplexF32 inv and division use Float64 intermediates, unavailable on Metal;
# its generic complex inv squares the denominator and can overflow/underflow.
lu_divide(x, y) = x / y
function lu_divide(x::Number, y::Complex)
    T = promote_type(typeof(x), typeof(y))
    R = promote_type(realtype(T), Float32)
    a, b = R(real(x)), R(imag(x))
    c, d = R(real(y)), R(imag(y))
    ab, cd = max(abs(a), abs(b)), max(abs(c), abs(d))
    scale = one(R)
    two = R(2)
    half = inv(two)
    overflow = floatmax(R)
    un = lu_safemin(R)
    e = eps(R) / two
    be = two / (e * e)
    if ab >= half * overflow
        a *= half
        b *= half
        scale *= two
    end
    if cd >= half * overflow
        c *= half
        d *= half
        scale *= half
    end
    if ab <= un * two / e
        a *= be
        b *= be
        scale /= be
    end
    if cd <= un * two / e
        c *= be
        d *= be
        scale *= be
    end
    if abs(d) <= abs(c)
        p, q = lu_divide1(a, b, c, d)
    else
        p, q = lu_divide1(b, a, d, c)
        q = -q
    end
    return T(complex(p * scale, q * scale))
end

function lu_divide1(a, b, c, d)
    r = d / c
    t = inv(c + d * r)
    return (lu_divide2(a, b, c, d, r, t), lu_divide2(b, -a, c, d, r, t))
end

function lu_divide2(a, b, c, d, r, t)
    if !iszero(r)
        br = b * r
        return !iszero(br) ? (a + br) * t : a * t + (b * t) * r
    end
    return (a + d * (b / c)) * t
end

function lu_swap_column!(A, ipiv, order, j)
    for k in order
        p = @inbounds ipiv[k]
        if p != k
            @inbounds begin
                x = A[k, j]
                A[k, j] = A[p, j]
                A[p, j] = x
            end
        end
    end
    return @inbounds A[first(order), j]
end

function lu_swap_rows!(A, ipiv, order, columns)
    (isempty(columns) || isempty(order)) && return A
    # Each map worker owns a column and applies LASWP's swaps in order, so even
    # overlapping pivots need neither inter-worker synchronization nor workspace.
    lu_map!(j -> lu_swap_column!(A, ipiv, order, j), lu_view(A, first(order), columns), columns)
    return A
end

function lu_scale(x, pivot, sfmin)
    iszero(pivot) && return x
    return abs(pivot) >= sfmin ? x * lu_divide(one(pivot), pivot) : lu_divide(x, pivot)
end

lu_info(info, pivot, k) = iszero(info) && iszero(pivot) ? k : info

function lu_pivot_max(a, b)
    x, i = a
    y, j = b
    isless(x, y) && return b
    return isequal(x, y) ? (x, min(i, j)) : a
end

function lu_pivot!(indices, column, rows, k, metric::F) where {F}
    # Map the metric lazily and keep the dimension-wise result on the device.
    # Both pivot metrics are nonnegative; the first row wins an all-zero tie.
    neutral = (zero(metric(zero(eltype(column)))), k)
    selected = mapreduce(
        (x, i) -> i < k ? neutral : (metric(x), i), lu_pivot_max, column, rows;
        dims = 1, init = neutral
    )
    indices .= last.(selected)
    return indices
end

function lu_pivot_rows(A, k)
    # Metal specializes reductions on their lengths. Bucket the suffix length
    # to reuse O(log(m)) kernels; completed rows contribute neutral pairs.
    m = size(A, 1)
    count = min(m, nextpow(2, m - k + 1))
    return (m - count + 1):m
end
lu_pivot_rows(A::Array, k) = k:size(A, 1)

function getrf_column!(A, ipiv, info, k, pivot, sfmin)
    indices = view(ipiv, k:k)
    if pivot isa LinearAlgebra.NoPivot
        fill!(indices, k)
    else
        metric = pivot isa LinearAlgebra.RowNonZero ? lu_nonzero : lu_abs1
        rows = lu_pivot_rows(A, k)
        lu_pivot!(indices, view(A, rows, k), rows, k, metric)
    end
    lu_swap_rows!(A, ipiv, k:k, k:k)
    diagonal = view(A, k:k, k)
    info .= lu_info.(info, diagonal, k)
    below = view(A, (k + 1):size(A, 1), k)
    # GPUArrays map! uses runtime dimensions, avoiding Metal's repeated shape
    # specialization of this shrinking column. only(diagonal) executes in the map.
    map!(x -> lu_scale(x, only(diagonal), sfmin), below, below)
    return A
end

# Recursive triangular substitution retains BLAS-3 updates without depending on
# a backend's triangular solve coverage (notably Metal's Float32-only MPS path).
lu_view(A, rows, columns) = @inbounds view(A, rows, columns)
lu_view(A::LinearAlgebra.Transpose, rows, columns) = transpose(lu_view(parent(A), columns, rows))
lu_view(A::LinearAlgebra.Adjoint, rows, columns) = adjoint(lu_view(parent(A), columns, rows))

# Base collapses nested views by indexing the parent ranges on the host.
# Broadcast that reindexing when a selector contains device pivot indices.
lu_reindex(indices, selection) = indices[selection]
lu_reindex(indices::AbstractRange, selection::AbstractArray) = getindex.(Ref(indices), selection)
lu_reindex(indices::AbstractRange, selection::AbstractRange) = indices[selection]
function lu_view(
        A::SubArray{T, 2, P, I}, rows, columns
    ) where {T, P <: AbstractMatrix, I <: Tuple{AbstractRange, AbstractRange}}
    i, j = parentindices(A)
    return lu_view(parent(A), lu_reindex(i, rows), lu_reindex(j, columns))
end

lu_mul!(C, A, B, alpha, beta) = LinearAlgebra.mul!(C, A, B, alpha, beta)
lu_mul!(C::LinearAlgebra.Transpose, A, B, alpha, beta) =
    lu_mul!(parent(C), transpose(B), transpose(A), alpha, beta)
lu_mul!(C::LinearAlgebra.Adjoint, A, B, alpha, beta) =
    lu_mul!(parent(C), adjoint(B), adjoint(A), conj(alpha), conj(beta))

function lu_copyto!(dest, src)
    # Metal's native ComplexF32 transpose-copy path loses complex components.
    if src isa Union{LinearAlgebra.Transpose, LinearAlgebra.Adjoint}
        dest .= src
    else
        copyto!(dest, src)
    end
    return dest
end
lu_copyto!(dest::LinearAlgebra.Transpose, src) = lu_copyto!(parent(dest), transpose(src))
lu_copyto!(dest::LinearAlgebra.Adjoint, src) = lu_copyto!(parent(dest), adjoint(src))

# Normalize wrapped destinations so GPUArrays map! dispatch stays on the device.
lu_map!(f::F, dest, indices) where {F} = map!(f, dest, indices)
lu_map!(f::F, dest::LinearAlgebra.Transpose, indices) where {F} = map!(f, parent(dest), indices)
lu_map!(f::F, dest::LinearAlgebra.Adjoint, indices) where {F} =
    map!(j -> conj(f(j)), parent(dest), indices)

function lu_trsm_column!(A, B, j, upper, unit)
    n = size(A, 1)
    for k in (upper ? (n:-1:1) : (1:n))
        x = @inbounds B[k, j]
        if !iszero(x)
            if !unit
                @inbounds B[k, j] = lu_divide(x, A[k, k])
                x = @inbounds B[k, j]
            end
            for i in (upper ? (1:(k - 1)) : ((k + 1):n))
                @inbounds B[i, j] = B[i, j] - x * A[i, k]
            end
        end
    end
    return @inbounds B[1, j]
end

function lu_trsm!(A, B, upper::Bool, unit::Bool)
    n = size(A, 1)
    (iszero(n) || isempty(B)) && return B
    if n == 1
        unit || (B .= lu_divide.(B, lu_view(A, 1:1, 1:1)))
        return B
    end
    if n <= 16
        # Reference BLAS TRSM substitution, one independent column per map worker.
        # Scalar accesses execute inside the device map; A and B must be disjoint.
        lu_map!(j -> lu_trsm_column!(A, B, j, upper, unit), lu_view(B, 1, axes(B, 2)), axes(B, 2))
        return B
    end
    split = n ÷ 2
    first = 1:split
    last = (split + 1):n
    solved, remaining = upper ? (last, first) : (first, last)
    lu_trsm!(lu_view(A, solved, solved), lu_view(B, solved, :), upper, unit)
    T = eltype(B)
    lu_mul!(
        lu_view(B, remaining, :), lu_view(A, remaining, solved),
        lu_view(B, solved, :), -one(T), one(T)
    )
    lu_trsm!(lu_view(A, remaining, remaining), lu_view(B, remaining, :), upper, unit)
    return B
end

function getrf2!(A, ipiv, info, k, width, pivot, sfmin)
    rows = size(A, 1) - k + 1
    (iszero(rows) || iszero(width)) && return A
    if min(rows, width) == 1
        return getrf_column!(A, ipiv, info, k, pivot, sfmin)
    end
    split = min(rows, width) ÷ 2
    first = k:(k + split - 1)
    last = (k + split):(k + width - 1)
    below = (k + split):size(A, 1)
    getrf2!(A, ipiv, info, k, split, pivot, sfmin)
    lu_swap_rows!(A, ipiv, first, last)
    lu_trsm!(view(A, first, first), view(A, first, last), false, true)
    T = eltype(A)
    LinearAlgebra.mul!(
        view(A, below, last), view(A, below, first),
        view(A, first, last), -one(T), one(T)
    )
    getrf2!(A, ipiv, info, k + split, width - split, pivot, sfmin)
    lu_swap_rows!(A, ipiv, (k + split):min(size(A, 1), k + width - 1), first)
    return A
end

"""
    AlgorithmicNLA.lu(A, pivot=RowMaximum(); check=true, allowsingular=false, blocksize=64)

Compute a dense LU factorization with `F.L * F.U == A[F.p, :]`, using a
copy of `A`. Return a standard `LinearAlgebra.LU` with `NLAArray` factors and
pivots stored on the input backend. Floating-point factor storage retains the
input element type; integer input uses Julia's allocating LU element type.

The blocked driver and recursive panels follow LAPACK 3.12.1 GETRF/GETRF2.
`RowMaximum` uses absolute value for real entries and `abs(real)+abs(imag)` for
complex entries, with first-index tie breaking. `NoPivot` and `RowNonZero` are
also supported. `check=true` rejects nonfinite inputs and throws for singular
factors unless `allowsingular=true`; a zero pivot with `NoPivot` still throws.

Array primitives keep pivots on device; checking input finiteness and reading
the final status synchronize. Backend matrix products determine their internal
compute precision. No custom GPU kernels or implicit factor promotion are used.
Complex division follows LAPACK LADIV in at least Float32 arithmetic, rounding
back to the storage type. This avoids Float64 intermediates for ComplexF32 on
Metal and overflow/underflow in a naive reciprocal.
"""
function lu(A::AbstractMatrix{T}, pivot = LinearAlgebra.RowMaximum(); kwargs...) where {T}
    S = T <: Union{AbstractFloat, Complex{<:AbstractFloat}} ? T : LinearAlgebra.lutype(T)
    B = similar(unwrap(A), S, size(A))
    lu_copyto!(B, unwrap(A))
    return lu!(B, pivot; kwargs...)
end

"""
    AlgorithmicNLA.lu!(A, pivot=RowMaximum(); check=true, allowsingular=false, blocksize=64)

Compute [`AlgorithmicNLA.lu`](@ref) in place, overwriting `A` with packed factors.
The returned factor storage aliases `A`. Pivot and status arrays
are allocated on the same backend. `blocksize` must be a positive integer.
"""
function lu!(
        A::AbstractMatrix{T}, pivot = LinearAlgebra.RowMaximum();
        check::Bool = true, allowsingular::Bool = false, blocksize::Integer = 64
    ) where {T}
    Base.require_one_based_indexing(A)
    pivot isa Union{LinearAlgebra.RowMaximum, LinearAlgebra.NoPivot, LinearAlgebra.RowNonZero} ||
        throw(ArgumentError("unsupported LU pivot strategy: $pivot"))
    blocksize > 0 || throw(ArgumentError("blocksize must be positive"))
    data = unwrap(A)
    check && !all(isfinite, data) && throw(ArgumentError("matrix contains Infs or NaNs"))
    m, n = size(data)
    count = min(m, n)
    ipiv = similar(data, LinearAlgebra.BlasInt, (count,))
    info = similar(data, LinearAlgebra.BlasInt, (1,))
    fill!(info, 0)
    sfmin = lu_safemin(realtype(T))
    if blocksize >= count || blocksize == 1
        getrf2!(data, ipiv, info, 1, n, pivot, sfmin)
    else
        for k in 1:blocksize:count
            width = min(blocksize, count - k + 1)
            panel = k:(k + width - 1)
            last = (k + width):n
            below = (k + width):m
            getrf2!(data, ipiv, info, k, width, pivot, sfmin)
            lu_swap_rows!(data, ipiv, panel, 1:(k - 1))
            lu_swap_rows!(data, ipiv, panel, last)
            lu_trsm!(view(data, panel, panel), view(data, panel, last), false, true)
            if !isempty(below) && !isempty(last)
                LinearAlgebra.mul!(
                    view(data, below, last), view(data, below, panel),
                    view(data, panel, last), -one(T), one(T)
                )
            end
        end
    end
    status = only(Array(info))
    pivot isa LinearAlgebra.NoPivot && (status = -status)
    if check
        status < 0 && throw(LinearAlgebra.ZeroPivotException(-status))
        !allowsingular && status > 0 && throw(LinearAlgebra.SingularException(status))
    end
    return LinearAlgebra.LU(NLAArray(A), ipiv, status)
end

for f in (:lu, :lu!)
    @eval LinearAlgebra.$f(A::NLAArray{T, 2}; kwargs...) where {T} = $f(A; kwargs...)
    for strategy in (:RowMaximum, :NoPivot, :RowNonZero)
        @eval LinearAlgebra.$f(
            A::NLAArray{T, 2}, pivot::LinearAlgebra.$strategy;
            kwargs...
        ) where {T} = $f(A, pivot; kwargs...)
    end
end

function Base.getproperty(F::LinearAlgebra.LU{T, <:NLAArray}, name::Symbol) where {T}
    A = getfield(F, :factors)
    m, n = size(A)
    if name === :L || name === :U
        rows, columns = name === :L ? (m, min(m, n)) : (min(m, n), n)
        B = similar(A, T, (rows, columns))
        i = 1:rows
        j = transpose(1:columns)
        source = view(parent(A), 1:rows, 1:columns)
        if name === :L
            parent(B) .= ifelse.(i .> j, source, ifelse.(i .== j, one(T), zero(T)))
        else
            parent(B) .= ifelse.(i .<= j, source, zero(T))
        end
        return B
    elseif name === :p
        return LinearAlgebra.ipiv2perm(Array(getfield(F, :ipiv)), m)
    elseif name === :P
        B = similar(A, T, (m, m))
        parent(B) .= ifelse.((1:m) .== transpose(1:m), one(T), zero(T))
        lu_swap_rows!(parent(B), getfield(F, :ipiv), 1:min(m, n), 1:m)
        return B
    end
    return getfield(F, name)
end

Base.copy(F::LinearAlgebra.LU{T, <:NLAArray}) where {T} =
    LinearAlgebra.LU(copy(F.factors), copy(F.ipiv), F.info)

function lu_ldiv!(F, B, operation)
    n = LinearAlgebra.checksquare(F)
    size(B, 1) == n || throw(DimensionMismatch("LU has $n rows but right-hand side has $(size(B, 1))"))
    Base.require_one_based_indexing(B)
    F.info < 0 && throw(LinearAlgebra.ZeroPivotException(-F.info))
    F.info > 0 && throw(LinearAlgebra.SingularException(F.info))
    A = parent(F.factors)
    data = ndims(B) == 1 ? reshape(unwrap(B), n, 1) : unwrap(B)
    if operation === identity
        lu_swap_rows!(data, F.ipiv, 1:n, axes(data, 2))
        lu_trsm!(A, data, false, true)
        lu_trsm!(A, data, true, false)
    else
        transformed = operation(A)
        lu_trsm!(transformed, data, false, false)
        lu_trsm!(transformed, data, true, true)
        lu_swap_rows!(data, F.ipiv, n:-1:1, axes(data, 2))
    end
    return B
end

LinearAlgebra.ldiv!(F::LinearAlgebra.LU{T, <:NLAArray}, B::AbstractVecOrMat) where {T} =
    lu_ldiv!(F, B, identity)
LinearAlgebra.ldiv!(
    F::LinearAlgebra.TransposeFactorization{T, <:LinearAlgebra.LU{T, <:NLAArray}},
    B::AbstractVecOrMat
) where {T} = lu_ldiv!(parent(F), B, transpose)
LinearAlgebra.ldiv!(
    F::LinearAlgebra.AdjointFactorization{T, <:LinearAlgebra.LU{T, <:NLAArray}},
    B::AbstractVecOrMat
) where {T} = lu_ldiv!(parent(F), B, adjoint)

function lu_ldiv(F, B)
    T = typeof(oneunit(eltype(B)) / oneunit(eltype(F)))
    X = similar(B, T, size(B))
    lu_copyto!(unwrap(X), unwrap(B))
    return LinearAlgebra.ldiv!(F, X)
end

LinearAlgebra.ldiv(F::LinearAlgebra.LU{T, <:NLAArray}, B::AbstractVecOrMat) where {T} =
    lu_ldiv(F, B)
LinearAlgebra.ldiv(
    F::LinearAlgebra.TransposeFactorization{T, <:LinearAlgebra.LU{T, <:NLAArray}},
    B::AbstractVecOrMat
) where {T} = lu_ldiv(F, B)
LinearAlgebra.ldiv(
    F::LinearAlgebra.AdjointFactorization{T, <:LinearAlgebra.LU{T, <:NLAArray}},
    B::AbstractVecOrMat
) where {T} = lu_ldiv(F, B)

function LinearAlgebra.rdiv!(B::AbstractMatrix, F::LinearAlgebra.LU{T, <:NLAArray}) where {T}
    lu_ldiv!(F, transpose(B), transpose)
    return B
end

function Base.inv(F::LinearAlgebra.LU{T, <:NLAArray}) where {T}
    n = LinearAlgebra.checksquare(F)
    B = similar(F.factors, T, (n, n))
    parent(B) .= ifelse.((1:n) .== transpose(1:n), one(T), zero(T))
    return LinearAlgebra.ldiv!(F, B)
end

function LinearAlgebra.det(F::LinearAlgebra.LU{T, <:NLAArray}) where {T}
    n = LinearAlgebra.checksquare(F)
    iszero(F.info) || return zero(T)
    diagonal = view(parent(F.factors), LinearAlgebra.diagind(parent(F.factors)))
    return prod(ifelse.(F.ipiv .== (1:n), diagonal, .-diagonal))
end

function LinearAlgebra.logabsdet(F::LinearAlgebra.LU{T, <:NLAArray}) where {T}
    n = LinearAlgebra.checksquare(F)
    iszero(F.info) || return (log(zero(realtype(T))), zero(T))
    diagonal = view(parent(F.factors), LinearAlgebra.diagind(parent(F.factors)))
    signs = ifelse.(F.ipiv .== (1:n), sign.(diagonal), .-sign.(diagonal))
    return (sum(log.(abs.(diagonal))), prod(signs))
end
