@testset "common" begin
    @testset "realtype" begin
        @test AlgorithmicNLA.realtype(Float32) === Float32
        @test AlgorithmicNLA.realtype(ComplexF64) === Float64
        @test AlgorithmicNLA.realtype(Complex{BFloat16}) === BFloat16
        @test AlgorithmicNLA.realtype(Int) === Float64
        @test AlgorithmicNLA.realtype(1.0f0) === Float32
    end

    @testset "unit_roundoff" for AT in ARRAY_TYPES, T in element_types(AT)
        u = AlgorithmicNLA.unit_roundoff(T)
        @test u == eps(AlgorithmicNLA.realtype(T)) / 2
        @test AlgorithmicNLA.realtype(T)(1) + 2u > 1
    end
end

@testset "element types" begin
    @test referencetype(Float16) === Float64
    @test referencetype(Complex{BFloat16}) === ComplexF64

    # The reference backend supports every candidate; a device backend supports a
    # subset, and the suite must adapt to it rather than assume Float64.
    @test element_types(Array) == collect(CANDIDATE_ELEMENT_TYPES)
    for AT in ARRAY_TYPES
        @test !isempty(element_types(AT))
    end
end

@testset "test problems" for AT in ARRAY_TYPES, T in element_types(AT)
    A = testmatrix(AT, T, 6, 4)
    @test size(A) == (6, 4)
    @test eltype(A) === T
    @test A == testmatrix(AT, T, 6, 4)
    entries = Array(A)
    one_T = one(AlgorithmicNLA.realtype(T))
    @test all(<=(one_T), abs.(real.(entries)))
    @test all(<=(one_T), abs.(imag.(entries)))
    @test entries == T.(Array(testmatrix(Array, referencetype(T), 6, 4)))
end
