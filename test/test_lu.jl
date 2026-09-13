# LAPACK 3.12.1 GET01/GET02 ratios, with represented inputs and factors evaluated
# in widened arithmetic. LAMCH('E') = eps/2; all four test.in drivers use 30.
lu_reference_type(::Type{T}) where {T} = T <: Complex ? Complex{BigFloat} : BigFloat
lu_test_abs1(x) = abs(real(x)) + abs(imag(x))

function get01_ratio(A, F)
    u = unit_roundoff(eltype(A))
    return setprecision(BigFloat, max(128, 2precision(realtype(eltype(A))))) do
        get01_ratio(A, F, BigFloat(u))
    end
end

function get01_ratio(A, F, u)
    m, n = size(A)
    min(m, n) == 0 && return BigFloat(0)
    R = lu_reference_type(eltype(A))
    original = R.(A)
    residual = R.(Array(F.L)) * R.(Array(F.U)) - original[F.p, :]
    anorm = opnorm(original, 1)
    rnorm = opnorm(residual, 1)
    iszero(anorm) && return iszero(rnorm) ? BigFloat(0) : inv(u)
    return ((rnorm / n) / anorm) / u
end

function get02_ratio(A, X, B, operation = identity)
    u = unit_roundoff(eltype(A))
    return setprecision(BigFloat, max(128, 2precision(realtype(eltype(A))))) do
        get02_ratio(A, X, B, operation, BigFloat(u))
    end
end

function get02_ratio(A, X, B, operation, u)
    isempty(X) && return BigFloat(0)
    R = lu_reference_type(eltype(A))
    original = operation(R.(A))
    x = reshape(R.(Array(X)), size(original, 2), :)
    b = reshape(R.(Array(B)), size(original, 1), :)
    residual = b - original * x
    anorm = opnorm(original, 1)
    return maximum(axes(x, 2)) do j
        xnorm = sum(lu_test_abs1, view(x, :, j))
        iszero(xnorm) || iszero(anorm) ? inv(u) :
            ((sum(lu_test_abs1, view(residual, :, j)) / anorm) / xnorm) / u
    end
end

function lu_ge_matrix(::Type{T}, kind, n, seed) where {T}
    R = lu_reference_type(T)
    # LATB4 uses LAMCH('P'), unlike GET01/GET02. Extra precision is essential
    # when constructing the prescribed spectra for BigFloat inputs.
    p = precision(realtype(T))
    e = BigFloat(eps(realtype(T)))
    wide = setprecision(BigFloat, max(128, 2p)) do
        kappa = kind == 8 ? sqrt(inv(10e)) : kind == 9 ? inv(10e) : BigFloat(2)
        sigma = exp.(range(BigFloat(0), -log(kappa); length = n))
        rng = StableRNG(seed)
        if kind == 1
            A = Matrix(Diagonal(R.(sigma)))
        else
            U = Matrix(qr(R.(uniform(rng, referencetype(T), n, n))).Q)
            V = Matrix(qr(R.(uniform(rng, referencetype(T), n, n))).Q)
            A = U * Diagonal(sigma) * V'
            if kind == 2
                A = Matrix(qr(A).R)
            elseif kind == 3
                A = Matrix(qr(A).R')
            end
        end
        if kind == 5
            A[:, 1] .= 0
        elseif kind == 6
            A[:, n] .= 0
        elseif kind == 7
            A[:, (n ÷ 2 + 1):n] .= 0
        end
        small = BigFloat(floatmin(realtype(T))) / (4e)
        scale = kind == 10 ? small : kind == 11 ? inv(small) : BigFloat(1)
        A .*= scale
        return A
    end
    return T.(wide)
end

@testset "LU" begin
    @testset "ordered row swaps $AT $T" for AT in ARRAY_TYPES, T in element_types(AT)
        original = reshape(T.(1:48), 8, 6)
        T <: Complex && (original .+= T(im))
        pivots = [8, 8, 7, 5, 8, 6, 8, 8]
        for operation in (identity, transpose, adjoint)
            host = copy(operation(original))
            padded = fill(T(-1), 2size(host, 1), 2size(host, 2))
            padded[1:2:end, 1:2:end] .= host
            storage = AT(padded)
            data = operation(view(storage, 1:2:size(storage, 1), 1:2:size(storage, 2)))
            ipiv = AT(pivots)
            expected = copy(original)
            for order in (1:8, 8:-1:1)
                for k in order
                    row = copy(expected[k, 2:2:6])
                    expected[k, 2:2:6] = expected[pivots[k], 2:2:6]
                    expected[pivots[k], 2:2:6] = row
                end
                @test AlgorithmicNLA.lu_swap_rows!(data, ipiv, order, 2:2:6) === data
                @test operation(Array(storage)[1:2:end, 1:2:end]) == expected
            end
            @test operation(Array(storage)[1:2:end, 1:2:end]) == original
            @test all(==(-one(T)), Array(storage)[2:2:end, :])
            @test all(==(-one(T)), Array(storage)[:, 2:2:end])
            @test AlgorithmicNLA.lu_swap_rows!(data, ipiv, 1:0, 1:6) === data
            @test AlgorithmicNLA.lu_swap_rows!(data, ipiv, 1:8, 1:0) === data
            @test operation(Array(storage)[1:2:end, 1:2:end]) == original
        end
    end

    @testset "small triangular solves $AT $T" for AT in ARRAY_TYPES, T in element_types(AT)
        for n in (2, 15, 16, 17, 33), upper in (false, true), unit in (false, true)
            A = testmatrix(Array, T, n, n) ./ T(n)
            for i in 1:n
                A[i, i] = T(2)
            end
            triangle = upper ? (unit ? UnitUpperTriangular(A) : UpperTriangular(A)) :
                (unit ? UnitLowerTriangular(A) : LowerTriangular(A))
            C = Matrix(triangle)
            # Poison storage the solve must not read, including the unit diagonal.
            for j in 1:n, i in 1:n
                if (upper ? i > j : i < j) || (unit && i == j)
                    A[i, j] = T(NaN)
                end
            end
            B = testmatrix(Array, T, n, 5)
            storage = AT(zeros(T, 2n, 10))
            X = view(storage, 1:2:(2n), 1:2:10)
            X .= AT(B)
            source = AT(zeros(T, 2n, 2n))
            data = view(source, 1:2:(2n), 1:2:(2n))
            data .= AT(A)
            @test AlgorithmicNLA.lu_trsm!(data, X, upper, unit) === X
            @test get02_ratio(C, X, B) < 30
            @test eltype(X) === T
            @test all(iszero, Array(storage)[2:2:end, :])
            @test all(iszero, Array(storage)[:, 2:2:end])
        end
    end

    @testset "GE families $AT $T seed=$seed" for AT in ARRAY_TYPES, T in element_types(AT), seed in TEST_SEEDS
        @testset "kind=$kind" for kind in 1:11
            A = lu_ge_matrix(T, kind, 5, seed)
            wrapped = NLAArray(AT(A))
            F = lu(wrapped; check = false, blocksize = 3)
            expected_info = kind == 5 ? 1 : kind == 6 ? 5 : kind == 7 ? 3 : 0
            @test F.info == expected_info
            @test get01_ratio(A, F) < 30
            @test Array(wrapped) == A
            @test eltype(F) === T
            @test parent(F.factors) isa AT
            @test F.ipiv isa AT
            if AT === Array && T <: LinearAlgebra.BlasFloat
                reference = LinearAlgebra.lu(A; check = false)
                @test F.info == reference.info
                @test get01_ratio(A, reference) < 30
            end
            if iszero(expected_info)
                exact = T.(reshape(1:10, 5, 2) .% 3 .- 1)
                B = AT(A * exact)
                X = F \ B
                @test get02_ratio(A, X, B) < 30
            end
        end
    end

    @testset "shapes and blocking" for AT in ARRAY_TYPES, T in element_types(AT)
        for (m, n) in (
                (0, 0), (0, 5), (5, 0), (1, 1), (1, 5), (5, 1),
                (3, 5), (5, 3), (7, 7), (65, 67), (67, 65), (65, 65),
            )
            A = testmatrix(Array, T, m, n)
            data = AT(A)
            F = AlgorithmicNLA.lu!(data; check = false)
            @test parent(F.factors) === data
            @test get01_ratio(A, F) < 30
            @test size(F.L) == (m, min(m, n))
            @test size(F.U) == (min(m, n), n)
            @test length(F.ipiv) == min(m, n)
        end
        A = testmatrix(Array, T, 7, 7)
        for blocksize in (1, 2, 3, 20)
            F = AlgorithmicNLA.lu(AT(A); blocksize)
            @test get01_ratio(A, F) < 30
        end
    end

    @testset "solves and factorization interface" for AT in ARRAY_TYPES, T in element_types(AT)
        A = T[0 2 1; 1 3 0; 2 0 1]
        T <: Complex && (A[2, 3] = complex(one(realtype(T)), one(realtype(T))))
        F = lu(NLAArray(AT(A)))
        @test F isa LU
        @test issuccess(F)
        @test Array(F.P) * A == A[F.p, :]
        @test Array(copy(F).factors) == Array(F.factors)
        @test get01_ratio(A, F) < 30
        for operation in (identity, transpose, adjoint), nrhs in (1, 2, 15)
            B = AT(T.(reshape(1:(3nrhs), 3, nrhs)))
            G = operation(F)
            X = G \ B
            @test get02_ratio(A, X, B, operation) < 30
            wrapped = NLAArray(copy(B))
            @test ldiv!(G, wrapped) === wrapped
            @test get02_ratio(A, wrapped, B, operation) < 30
            destination = similar(B)
            @test ldiv!(destination, G, B) === destination
            @test get02_ratio(A, destination, B, operation) < 30
        end
        b = AT(T[1, 2, 3])
        @test get02_ratio(A, F \ b, b) < 30
        Iref = Matrix{T}(I, 3, 3)
        @test get02_ratio(A, inv(F), Iref) < 30
        B = AT(T[1 2 3; 4 5 6])
        X = B / F
        @test get02_ratio(A, transpose(Array(X)), transpose(Array(B)), transpose) < 30
        @test det(F) ≈ det(A) rtol = 30unit_roundoff(T)
        value, phase = logabsdet(F)
        reference_value, reference_phase = logabsdet(A)
        @test value ≈ reference_value rtol = 30unit_roundoff(T)
        @test phase ≈ reference_phase rtol = 30unit_roundoff(T)
        @test_throws DimensionMismatch F \ AT(ones(T, 2))
    end

    @testset "pivoting, scaling and failures" for AT in ARRAY_TYPES, T in element_types(AT)
        # Ties spanning reduction groups, including a pivot after the first row.
        tall = zeros(T, 513, 3)
        tall[513, 1] = T(4)
        tall[513, 2] = T(8)
        tall[513, 3] = T(16)
        tall[257, 2] = T(2)
        tall[400, 2] = T(-2)
        tall[257, 3] = T(8)
        tall[400, 3] = T(-8)
        tall[300, 3] = T(1)
        tall[450, 3] = T(-1)
        for pivot in (RowMaximum(), RowNonZero())
            F = AlgorithmicNLA.lu(AT(tall), pivot)
            # The third reduction must ignore the larger U[2, 3] in its bucket.
            @test Array(F.ipiv) == [513, 257, 300]
            @test get01_ratio(tall, F) < 30
        end
        A = T[1 2; -1 3; 0 1]
        @test Array(AlgorithmicNLA.lu(AT(A)).ipiv)[1] == 1
        # Wilkinson's partial-pivoting growth example has exact binary factors.
        W = Matrix{T}(I, 6, 6) - tril!(ones(T, 6, 6), -1)
        W[:, end] .= one(T)
        growth = AlgorithmicNLA.lu(AT(W))
        @test Array(growth.ipiv) == collect(1:6)
        @test Array(growth.U)[end, end] == T(32)
        @test get01_ratio(W, growth) < 30
        if T <: Complex
            # abs1 prefers 3+3im over 5, unlike the generic Julia abs metric.
            A = T[5 1; 3 + 3im 2]
            @test Array(AlgorithmicNLA.lu(AT(A)).ipiv)[1] == 2
        end
        s = max(floatmin(realtype(T)) / 8, 2nextfloat(zero(realtype(T))))
        A = reshape(T[s, s / 2], 2, 1)
        arithmetic = only(Array(AT([s]) .+ zero(s)))
        @test arithmetic == s || iszero(arithmetic)
        if arithmetic == s
            F = AlgorithmicNLA.lu(AT(A))
            @test Array(F.factors)[2, 1] == T(1) / T(2)
        else
            # Metal flushes subnormal operands to zero. Pin the lost IEEE
            # behavior explicitly; the package inherits backend arithmetic.
            @test_broken begin
                F = AlgorithmicNLA.lu(AT(A); check = false)
                iszero(F.info) && Array(F.factors)[2, 1] == T(1) / T(2)
            end
        end
        zero_pivot = NLAArray(AT(T[0 1; 1 1]))
        @test_throws ZeroPivotException lu(zero_pivot, NoPivot())
        @test lu(zero_pivot, NoPivot(); check = false).info == -1
        @test_throws ZeroPivotException lu(zero_pivot, NoPivot(); allowsingular = true)
        @test Array(lu(zero_pivot, RowNonZero()).ipiv) == [2, 2]
        singular = NLAArray(AT(zeros(T, 2, 2)))
        @test_throws SingularException lu(singular)
        F = lu(singular; allowsingular = true)
        @test F.info == 1
        @test_throws SingularException F \ AT(ones(T, 2))
        @test det(F) == zero(T)
        @test logabsdet(F) == (log(zero(realtype(T))), zero(T))
        for value in (T(NaN), T(Inf))
            data = AT(fill(value, 2, 2))
            @test_throws ArgumentError AlgorithmicNLA.lu!(data)
            @test all(isequal.(Array(data), value))
            @test AlgorithmicNLA.lu(data; check = false) isa LU
        end
        @test_throws ArgumentError lu(zero_pivot; blocksize = 0)
    end

    @testset "strided factors and right-hand sides" for AT in ARRAY_TYPES, T in element_types(AT)
        A = T[0 2 1; 1 3 0; 2 0 1]
        T <: Complex && (A[2, 3] = complex(one(realtype(T)), one(realtype(T))))
        storage = AT(zeros(T, 7, 7))
        V = view(NLAArray(storage), 2:2:6, 2:2:6)
        copyto!(V, AT(A))
        F = lu!(V)
        @test F.factors === V
        @test get01_ratio(A, F) < 30
        @test Array(copy(F).factors) == Array(F.factors)
        untouched = Array(storage)
        untouched[2:2:6, 2:2:6] .= 0
        @test iszero(untouched)
        for operation in (identity, transpose, adjoint)
            B = T[1 2; 3 4; 5 6]
            rhs_storage = AT(zeros(T, 7, 5))
            rhs = view(NLAArray(rhs_storage), 2:2:6, 2:2:4)
            copyto!(rhs, AT(B))
            @test ldiv!(operation(F), rhs) === rhs
            @test get02_ratio(A, rhs, B, operation) < 30
        end
    end

    @testset "transposed inputs" for AT in ARRAY_TYPES, T in element_types(AT)
        A = T[0 2 1; 1 3 0; 2 0 1]
        T <: Complex && (A[2, 3] = complex(one(realtype(T)), one(realtype(T))))
        B = T[1 2 3; 4 5 6]
        for operation in (transpose, adjoint)
            F = AlgorithmicNLA.lu(operation(AT(A)))
            original = Matrix(operation(A))
            @test get01_ratio(original, F) < 30
            rhs = NLAArray(operation(AT(B)))
            @test get02_ratio(original, F \ rhs, operation(B)) < 30
            @test ldiv!(F, rhs) === rhs
            @test get02_ratio(original, rhs, operation(B)) < 30
        end
    end
end
