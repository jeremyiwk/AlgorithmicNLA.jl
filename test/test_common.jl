@testset "common" begin
    @testset "realtype" begin
        @test AlgorithmicNLA.realtype(Float32) === Float32
        @test AlgorithmicNLA.realtype(ComplexF64) === Float64
        @test AlgorithmicNLA.realtype(Int) === Float64
        @test AlgorithmicNLA.realtype(1.0f0) === Float32
    end

    @testset "unit_roundoff" begin
        for T in ELEMENT_TYPES
            u = AlgorithmicNLA.unit_roundoff(T)
            @test u == eps(AlgorithmicNLA.realtype(T)) / 2
            @test AlgorithmicNLA.realtype(T)(1) + 2u > 1
        end
    end
end
