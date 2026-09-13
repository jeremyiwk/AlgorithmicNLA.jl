using Metal: Metal, MtlArray

Metal.functional() || error("Metal tests require a functional GPU")
Metal.allowscalar(false)

include("../testsuite.jl")
push!(ARRAY_TYPES, MtlArray)
@info "Metal test element types" element_types(MtlArray)

@testset "AlgorithmicNLA with Metal" begin
    include("../test_common.jl")
    include("../test_array.jl")
    include("../test_lu.jl")

    @testset "device indexing policy and adaptation" begin
        A = NLAArray(reshape(Float32.(1:6), 2, 3))
        device = adapt(MtlArray, A)
        @test device isa NLAArray
        @test parent(device) isa MtlArray
        @test Array(device) == Array(A)
        @test_throws ErrorException device[1, 1]
        @test_throws ErrorException device[1, 1] = 0
        @test_throws ErrorException view(device, 1:2, 1:2)[1, 1]
    end
end
