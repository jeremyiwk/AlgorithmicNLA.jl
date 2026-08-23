"""
    realtype(T)

The real floating point type used for magnitudes of elements of type `T`, i.e.
`real(float(T))`. Used for norms, tolerances, and singular values.
"""
realtype(::Type{T}) where {T<:Number} = real(float(T))
realtype(x) = realtype(typeof(x))

"""
    unit_roundoff(T)

The unit roundoff `u = eps(realtype(T)) / 2` of the floating point type associated
with `T`.
"""
unit_roundoff(::Type{T}) where {T<:Number} = eps(realtype(T)) / 2
unit_roundoff(x) = unit_roundoff(typeof(x))
