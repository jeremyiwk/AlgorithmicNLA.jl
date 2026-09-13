include("testsuite.jl")

@testset "AlgorithmicNLA" begin
    include("test_common.jl")
    include("test_array.jl")
    include("test_lu.jl")
end
