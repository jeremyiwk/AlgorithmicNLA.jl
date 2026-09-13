using Adapt: adapt

struct FactorizationBackend{T} <: AbstractMatrix{T}
    data::Matrix{T}
    returned_type::DataType
end
Base.size(A::FactorizationBackend) = size(A.data)
function LinearAlgebra.qr(A::FactorizationBackend, pivot::NoPivot = NoPivot())
    return qr(convert(Matrix{A.returned_type}, A.data), pivot)
end

@testset "NLAArray" begin
    @testset "array operations" for AT in ARRAY_TYPES, T in element_types(AT)
        entries = reshape(T.(1:12), 4, 3)
        storage = AT(entries)
        A = NLAArray(storage)
        @test parent(A) === storage
        @test NLAArray(A) === A
        @test eltype(A) === T
        @test axes(A) == axes(storage)
        @test IndexStyle(typeof(A)) == IndexStyle(typeof(storage))

        B = copy(A)
        @test B isa NLAArray
        @test parent(B) !== storage
        @test !Base.mightalias(A, B)
        fill!(B, zero(T))
        @test Array(A) == entries
        @test Array(B) == zeros(T, 4, 3)
        @test copyto!(B, A) === B
        @test Array(B) == entries

        V = view(A, 2:3, 1:2)
        @test V isa NLAArray
        @test Base.mightalias(A, V)
        V .= -one(T)
        entries[2:3, 1:2] .= -one(T)
        @test Array(A) == entries
        S = A[1:2, :]
        @test S isa NLAArray
        fill!(S, zero(T))
        @test Array(A) == entries

        R = reshape(A, 3, 4)
        @test R isa NLAArray
        @test Base.mightalias(A, R)
        @test Array(R) == reshape(entries, 3, 4)
        @test Base.mightalias(A, adapt(nothing, R))
        @test Array(vec(A)) == vec(entries)
        @test Array(transpose(A)) == transpose(entries)
        @test Array(adjoint(A)) == adjoint(entries)

        C = similar(A, T, (2, 5))
        @test C isa NLAArray
        @test typeof(parent(C)) === typeof(storage)
        @test eltype(C) === T
        @test size(C) == (2, 5)
        D = similar(A)
        @test typeof(D) === typeof(A)
        @test size(D) == size(A)

        C = A .+ T(2) .* B
        @test C isa NLAArray
        @test typeof(parent(C)) === typeof(storage)
        @test Array(C) == entries .+ T(2) .* Array(B)
        @test eltype(C) === T
        B .= A .+ one(T)
        @test Array(B) == entries .+ one(T)
        raw = similar(storage)
        raw .= A .+ one(T)
        @test Array(raw) == Array(B)
        B .= raw
        @test Array(B) == Array(raw)
        @test Array(A .+ storage) == entries .+ entries

        # Both contiguous GPU views and nested broadcasts need alias handling.
        x = NLAArray(AT(T.(1:5)))
        view(x, 2:5) .= view(x, 1:4)
        @test Array(x) == T[1, 1, 2, 3, 4]
        x .= AT(T.(1:5))
        view(x, 1:4) .= one(T) .+ T(2) .* view(x, 2:5)
        @test Array(x) == T[5, 7, 9, 11, 5]
        x .= AT(T.(1:5))
        x .= vec(reshape(x, 1, 5)) .+ one(T)
        @test Array(x) == T[2, 3, 4, 5, 6]
        Vcopy = Base.unaliascopy(V)
        @test typeof(Vcopy) === typeof(V)
        @test !Base.mightalias(V, Vcopy)
        @test Array(Vcopy) == Array(V)

        @test sum(A) == sum(entries)
        @test maximum(abs, A) == maximum(abs, entries)
        totals = sum(A; dims = 1)
        @test totals isa NLAArray
        @test Array(totals) == sum(entries; dims = 1)
        @test Array(map(-, A)) == -entries
        @test adapt(Array, A) isa NLAArray
        @test parent(adapt(Array, A)) == entries

        empty = NLAArray(AT(zeros(T, 0, 3)))
        @test size(empty .+ one(T)) == (0, 3)
        @test Array(sum(empty; dims = 1)) == zeros(T, 1, 3)
        @test_throws DimensionMismatch B .= NLAArray(AT(ones(T, 2, 2)))
    end

    @testset "scalar indexing and array elements" begin
        A = NLAArray(reshape(1:6, 2, 3))
        @test A[2, 3] == 6
        @test A[CartesianIndex(2, 3)] == 6
        @test_throws BoundsError A[3, 1]
        entries = [[1, 2], [3, 4]]
        @test NLAArray(entries)[1] === entries[1]
        @test NLAArray(fill(7))[] == 7
    end

    @testset "matrix products" for AT in ARRAY_TYPES, T in element_types(AT)
        # Exactly representable products isolate wrapper dispatch from rounding.
        A = NLAArray(AT(T[1 2; 0 1]))
        B = NLAArray(AT(T[1 0; 2 1]))
        C = similar(A)
        fill!(C, one(T))
        @test Array(A * B) == T[5 2; 2 1]
        @test mul!(C, A, B, one(T), one(T)) === C
        @test Array(C) == T[6 3; 3 2]
        @test mul!(C, A, B) === C
        @test Array(C) == T[5 2; 2 1]
        b = NLAArray(AT(T[1, 2]))
        @test Array(A * b) == T[5, 2]
        @test Array(A' * A) == T[1 2; 2 5]
    end

    @testset "QR delegation" begin
        A = NLAArray([1.0 2.0; 0.0 1.0; 1.0 0.0])
        for pivot in (NoPivot(), ColumnNorm())
            F = qr(A, pivot)
            reference = qr(parent(A), pivot)
            @test typeof(F) === typeof(reference)
            @test F.factors == reference.factors
            B = copy(A)
            @test qr!(B, pivot).factors === parent(B)
        end
    end

    @testset "backend dispatch and precision contract" begin
        entries = Float32[0 2; 1 3]
        A = NLAArray(FactorizationBackend(entries, Float32))
        @test qr(A, NoPivot()).factors == qr(entries).factors
        promoted = NLAArray(FactorizationBackend(entries, Float64))
        @test_throws ArgumentError qr(promoted)
    end
end
