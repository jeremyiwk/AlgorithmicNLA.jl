"""
    NLAArray(A::AbstractArray)

Wrap `A` without copying or changing its element type to select AlgorithmicNLA
methods through the standard array and `LinearAlgebra` interfaces. `parent(A)`
on the wrapper returns the underlying storage.

Copies, views, reshapes, broadcasts, and array-valued reductions preserve the
wrapper. Scalar indexing obeys the parent's indexing policy. Backend adaptation
uses `Adapt.adapt` and preserves the wrapper.

LU selects this package's portable implementation; QR currently delegates to the
parent backend. QR delegation preserves backend errors and rejects a change to
floating-point factor storage precision.
"""
struct NLAArray{T, N, A <: AbstractArray{T, N}, I <: Tuple{Vararg{UInt}}} <: AbstractArray{T, N}
    data::A
    storageids::I
end

NLAArray(A::AbstractArray) = NLAArray(A, Base.dataids(A))
NLAArray(A::NLAArray) = A

Base.parent(A::NLAArray) = A.data
Base.size(A::NLAArray) = size(parent(A))
Base.axes(A::NLAArray) = axes(parent(A))
Base.IndexStyle(::Type{<:NLAArray{T, N, A}}) where {T, N, A} = IndexStyle(A)
Base.strides(A::NLAArray) = strides(parent(A))
Base.dataids(A::NLAArray) = A.storageids

unwrap(A::NLAArray) = parent(A)
unwrap(x) = x
wrap(A::AbstractArray) = NLAArray(A)
wrap(x) = x

Base.getindex(A::NLAArray, I...) = wrap(getindex(parent(A), map(unwrap, I)...))
Base.getindex(A::NLAArray, I::Int...) = getindex(parent(A), I...)
Base.getindex(A::NLAArray, I::CartesianIndex) = getindex(parent(A), I)
function Base.setindex!(A::NLAArray, value, I...)
    setindex!(parent(A), unwrap(value), map(unwrap, I)...)
    return A
end

Base.copy(A::NLAArray) = NLAArray(copy(parent(A)))
Base.unaliascopy(A::NLAArray) = NLAArray(Base.unaliascopy(parent(A)))
Base.similar(A::NLAArray, ::Type{T}, dims::Dims) where {T} =
    NLAArray(similar(parent(A), T, dims))
Base.similar(A::NLAArray, ::Type{T}) where {T} = NLAArray(similar(parent(A), T))
# Some backends return a new array object for an alias, with unrelated dataids.
Base.view(A::NLAArray, I...) = NLAArray(view(parent(A), map(unwrap, I)...), Base.dataids(A))
Base.reshape(A::NLAArray, dims::Dims) = NLAArray(reshape(parent(A), dims), Base.dataids(A))
Base.reshape(A::NLAArray, dims::Int...) = reshape(A, dims)
Base.vec(A::NLAArray) = NLAArray(vec(parent(A)), Base.dataids(A))
Base.transpose(A::NLAArray) = NLAArray(transpose(parent(A)), Base.dataids(A))
Base.adjoint(A::NLAArray) = NLAArray(adjoint(parent(A)), Base.dataids(A))
Base.Array(A::NLAArray) = Array(parent(A))

function Base.fill!(A::NLAArray, value)
    fill!(parent(A), value)
    return A
end

function Base.copyto!(dest::NLAArray, src::AbstractArray)
    copyto!(parent(dest), unwrap(src))
    return dest
end
function Base.copyto!(dest::Array, src::NLAArray)
    copyto!(dest, parent(src))
    return dest
end
function Base.copyto!(dest::NLAArray, src::NLAArray)
    copyto!(parent(dest), parent(src))
    return dest
end

function Adapt.adapt_structure(to, A::NLAArray)
    data = Adapt.adapt(to, parent(A))
    return data === parent(A) ? NLAArray(data, Base.dataids(A)) : NLAArray(data)
end

struct NLAArrayStyle{S <: Broadcast.BroadcastStyle} <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:NLAArray{T, N, A}}) where {T, N, A} =
    NLAArrayStyle{typeof(Base.BroadcastStyle(A))}()
Base.BroadcastStyle(::NLAArrayStyle{S}, ::NLAArrayStyle{R}) where {S, R} =
    NLAArrayStyle{typeof(Broadcast.result_style(S(), R()))}()
Base.BroadcastStyle(::NLAArrayStyle{S}, other::Broadcast.BroadcastStyle) where {S} =
    NLAArrayStyle{typeof(Broadcast.result_style(S(), other))}()
Base.BroadcastStyle(style::NLAArrayStyle{S}, ::Broadcast.Unknown) where {S} = style
Base.BroadcastStyle(other::Broadcast.BroadcastStyle, ::NLAArrayStyle{S}) where {S} =
    NLAArrayStyle{typeof(Broadcast.result_style(other, S()))}()

function unwrap(bc::Broadcast.Broadcasted)
    args = map(unwrap, bc.args)
    return Broadcast.Broadcasted(bc.f, args, bc.axes)
end

Base.copy(bc::Broadcast.Broadcasted{<:NLAArrayStyle}) = wrap(copy(unwrap(bc)))

unalias_broadcast(dest, x) = unwrap(Broadcast.broadcast_unalias(dest, x))
function unalias_broadcast(dest, bc::Broadcast.Broadcasted)
    args = map(x -> unalias_broadcast(dest, x), bc.args)
    return Broadcast.Broadcasted(bc.f, args, bc.axes)
end

function Broadcast.materialize!(::NLAArrayStyle, dest::AbstractArray, bc::Broadcast.Broadcasted)
    Broadcast.materialize!(unwrap(dest), unalias_broadcast(dest, bc))
    return dest
end

Base.mapreduce(f, op, A::NLAArray; kwargs...) = wrap(mapreduce(f, op, parent(A); kwargs...))
Base.map(f, A::NLAArray) = wrap(map(f, parent(A)))
LinearAlgebra.norm(A::NLAArray, p::Real = 2) = LinearAlgebra.norm(parent(A), p)

function Base.:*(A::NLAArray{T, 2}, B::NLAArray{S, N}) where {T, S, N}
    return wrap(parent(A) * parent(B))
end

function LinearAlgebra.mul!(
        C::NLAArray, A::NLAArray{T, 2}, B::NLAArray,
        alpha::Number, beta::Number
    ) where {T}
    LinearAlgebra.mul!(parent(C), parent(A), parent(B), alpha, beta)
    return C
end

function checked_factorization(f, A, args...; kwargs...)
    F = f(A, args...; kwargs...)
    T = eltype(A)
    if T <: Union{AbstractFloat, Complex{<:AbstractFloat}} && eltype(F) !== T
        throw(ArgumentError("backend $f changed factor storage from $T to $(eltype(F)); convert the input explicitly to request this precision"))
    end
    return F
end

for f in (:qr, :qr!)
    @eval function LinearAlgebra.$f(A::NLAArray{T, 2}; kwargs...) where {T}
        return checked_factorization(LinearAlgebra.$f, parent(A); kwargs...)
    end
end

for (f, strategies) in (
        (:qr, (:NoPivot, :ColumnNorm)),
        (:qr!, (:NoPivot, :ColumnNorm)),
    )
    for strategy in strategies
        @eval function LinearAlgebra.$f(
                A::NLAArray{T, 2}, pivot::LinearAlgebra.$strategy; kwargs...
            ) where {T}
            return checked_factorization(LinearAlgebra.$f, parent(A), pivot; kwargs...)
        end
    end
end
