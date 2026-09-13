include("lu_cooperative.jl")
include("../test/testsuite.jl")
using Metal: Metal, MtlArray

# Reuse the established GE generators and exact gate definitions without running
# unrelated production-interface tests against this deliberately opt-in prototype.
source = joinpath(@__DIR__, "..", "test", "test_lu.jl")
marker = "@testset \"LU\" begin"
@assert occursin(marker, read(source, String))
include_string(Main, first(split(read(source, String), marker; limit = 2)), source)

Metal.allowscalar(false)
@testset "Cooperative LU prototype" begin
    @testset "GE families $T" for T in element_types(MtlArray)
        for kind in 1:11
            A = lu_ge_matrix(T, kind, 33, 0)
            F = CooperativeLU.factor(MtlArray(A); blocksize = 16, cutoff = 8, use_mps = T == Float32)
            @test get01_ratio(A, F) < 30
            @test eltype(F.factors) == T
            reference = AlgorithmicNLA.lu(A; check = false)
            @test F.info == reference.info
            if F.info == 0
                B = testmatrix(Array, T, 33, 3; seed = kind)
                @test get02_ratio(A, F \ MtlArray(B), B) < 30
            end
        end
    end
    @testset "partial groups and panels" for n in (17, 31, 63, 65, 129, 257), cutoff in (4, 8, 16)
        A = testmatrix(Array, Float32, n, n)
        F = CooperativeLU.factor(MtlArray(A); blocksize = 16, cutoff)
        @test get01_ratio(A, F) < 30
        @test F.info == 0
    end
    @testset "ties, singularity, and safe minimum" begin
        for A in (
                Matrix{Float32}(I, 33, 33),
                zeros(Float32, 33, 33),
                diagm(0 => vcat(ones(Float32, 20), zeros(Float32, 13))),
                Matrix{Float32}(I, 33, 33) * (2floatmin(Float32)),
            )
            A[end, 1] = A[1, 1]
            F = CooperativeLU.factor(MtlArray(A); blocksize = 16, cutoff = 8)
            reference = AlgorithmicNLA.lu(A; check = false)
            @test F.info == reference.info
            @test Array(F.ipiv) == reference.ipiv
            @test get01_ratio(A, F) < 30
        end
    end
    @testset "MPS descriptor offsets and padding" begin
        host = fill(-9.0f0, 71, 57)
        host[3:35, 2:8] .= 1
        host[3:9, 10:28] .= 2
        host[37:69, 30:48] .= 3
        data = MtlArray(host)
        A, B, C = view(data, 3:35, 2:8), view(data, 3:9, 10:28), view(data, 37:69, 30:48)
        CooperativeLU.gemm!(C, A, B, true)
        host[37:69, 30:48] .= -11
        @test Array(data) == host
    end
    @testset "workgroup partition $T" for T in element_types(MtlArray)
        A = MtlArray(testmatrix(Array, T, 257, 257; seed = 7))
        reference = CooperativeLU.factor(A; blocksize = 128, cutoff = 4, use_mps = T == Float32)
        for wg in (32, 64, 128)
            F = CooperativeLU.factor(A; blocksize = 128, cutoff = 4, use_mps = T == Float32, workgroupsize = wg)
            @test Array(F.factors) == Array(reference.factors)
            @test Array(F.ipiv) == Array(reference.ipiv)
            @test F.info == reference.info
        end
    end
end
