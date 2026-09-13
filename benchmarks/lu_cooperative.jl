# Opt-in prototype: GETRF2 dependency schedule executed by one cooperative group
# per small panel, plus MPS GEMM descriptors for column-major range views.
# Numerical ordering follows src/lu.jl; no production methods are replaced.
# MPS descriptor/encoding adaptation follows Metal.jl's lib/mps/matrix.jl;
# its copyright and MIT license are reproduced in licenses/Metal.txt.
module CooperativeLU

using AlgorithmicNLA
using LinearAlgebra
using Metal
using KernelAbstractions: @kernel, @index, @localmem, @synchronize, @uniform, get_backend

const AN = AlgorithmicNLA

function schedule(width)
    ops = NTuple{4, Int32}[]
    function visit(k, w)
        return if w == 1
            push!(ops, (1, k, 0, 0))
        else
            s = w ÷ 2
            visit(k, s)
            push!(ops, (2, k, s, w))
            push!(ops, (3, k, s, w))
            push!(ops, (4, k, s, w))
            visit(k + s, w - s)
            push!(ops, (5, k, s, w))
        end
    end
    visit(0, width)
    return reduce(hcat, collect.(ops))
end

@kernel function panel_kernel!(A, ipiv, info, base, ops, sfmin, ::Val{WG}) where {WG}
    tid = @index(Local, Linear)
    @uniform m = Int32(size(A, 1))
    values = @localmem typeof(AN.lu_abs1(zero(eltype(A)))) (WG,)
    indices = @localmem Int32 (WG,)
    for step in 1:size(ops, 2)
        @uniform op = ops[1, step]
        @uniform k = base + ops[2, step]
        @uniform split = ops[3, step]
        @uniform width = ops[4, step]
        if op == 1
            v, p = zero(eltype(values)), k
            for i in (k + tid - 1):WG:m
                pair = AN.lu_pivot_max((v, p), (AN.lu_abs1(@inbounds A[i, k]), Int32(i)))
                v, p = pair
            end
            @inbounds values[tid] = v
            @inbounds indices[tid] = p
            @synchronize
            for stride in ntuple(i -> WG >> i, Val(trailing_zeros(WG)))
                if tid <= stride
                    @inbounds pair = AN.lu_pivot_max((values[tid], indices[tid]), (values[tid + stride], indices[tid + stride]))
                    @inbounds values[tid], indices[tid] = pair
                end
                @synchronize
            end
            if tid == 1
                @inbounds begin
                    p = indices[1]
                    ipiv[k] = p
                    x = A[k, k]
                    A[k, k] = A[p, k]
                    A[p, k] = x
                    info[1] = AN.lu_info(info[1], A[k, k], k)
                end
            end
            @synchronize
            for i in (k + tid):WG:m
                @inbounds A[i, k] = AN.lu_scale(A[i, k], A[k, k], sfmin)
            end
        elseif op == 2 || op == 5
            @uniform jfirst = op == 2 ? k + split : k
            @uniform jlast = op == 2 ? k + width - 1 : k + split - 1
            @uniform pfirst = op == 2 ? k : k + split
            @uniform plast = op == 2 ? k + split - 1 : k + width - 1
            for j in (jfirst + tid - 1):WG:jlast
                for i in pfirst:plast
                    @inbounds begin
                        p = ipiv[i]
                        x = A[i, j]
                        A[i, j] = A[p, j]
                        A[p, j] = x
                    end
                end
            end
        elseif op == 3
            for j in (k + split + tid - 1):WG:(k + width - 1)
                for p in k:(k + split - 1)
                    @inbounds x = A[p, j]
                    if !iszero(x)
                        for i in (p + 1):(k + split - 1)
                            @inbounds A[i, j] = A[i, j] - x * A[i, p]
                        end
                    end
                end
            end
        elseif op == 4
            @uniform rows = m - k - split + 1
            # Match the backend GEMM convention of at least Float32 accumulation;
            # each update is rounded back into the original factor storage.
            @uniform R = promote_type(eltype(A), Float32)
            for index in tid:WG:(rows * (width - split))
                i = k + split + (index - 1) % rows
                j = k + split + (index - 1) ÷ rows
                total = zero(R)
                for p in k:(k + split - 1)
                    @inbounds total += R(A[i, p]) * R(A[p, j])
                end
                @inbounds A[i, j] = R(A[i, j]) - total
            end
        end
        @synchronize
    end
end

function mps_matrix(A::SubArray)
    root = parent(A)
    @assert root isa MtlMatrix && stride(A, 1) == 1
    i, j = parentindices(A)
    offset = root.offset + sizeof(eltype(A)) * (first(i) - 1 + (first(j) - 1) * stride(root, 2))
    desc = Metal.MPS.MPSMatrixDescriptor(size(A, 2), size(A, 1), sizeof(eltype(A)) * stride(A, 2), eltype(A))
    return Metal.MPS.MPSMatrix(root, desc, offset)
end
mps_matrix(A::MtlMatrix) = Metal.MPS.MPSMatrix(A)

function gemm!(C, A, B, use_mps)
    isempty(C) && return C
    if !use_mps
        return mul!(C, A, B, -one(eltype(C)), one(eltype(C)))
    end
    ma, mb, mc = mps_matrix(A), mps_matrix(B), mps_matrix(C)
    kernel = Metal.MPS.MPSMatrixMultiplication(Metal.device(), false, false, size(C, 2), size(C, 1), size(A, 2), -1, 1)
    command = Metal.MTL.MTLCommandBuffer(Metal.global_queue(Metal.device()))
    Metal.MPS.encode!(command, kernel, mb, ma, mc)
    Metal.MTL.commit!(command)
    return C
end

function panel!(A, ipiv, info, k, width, sfmin, schedules, cutoff, use_mps, workgroupsize)
    if width <= cutoff
        panel_kernel!(get_backend(A), workgroupsize)(A, ipiv, info, Int32(k), schedules[width], sfmin, Val(workgroupsize); ndrange = workgroupsize)
        return A
    end
    split = width ÷ 2
    first, last = k:(k + split - 1), (k + split):(k + width - 1)
    below = (k + split):size(A, 1)
    panel!(A, ipiv, info, k, split, sfmin, schedules, cutoff, use_mps, workgroupsize)
    AN.lu_swap_rows!(A, ipiv, first, last)
    AN.lu_trsm!(view(A, first, first), view(A, first, last), false, true)
    gemm!(view(A, below, last), view(A, below, first), view(A, first, last), use_mps && split >= 16)
    panel!(A, ipiv, info, k + split, width - split, sfmin, schedules, cutoff, use_mps, workgroupsize)
    AN.lu_swap_rows!(A, ipiv, last, first)
    return A
end

function factor(A::MtlMatrix; blocksize = 256, cutoff = 16, use_mps = true, cooperative = true, workgroupsize = 256)
    n = size(A, 1)
    @assert size(A, 2) == n && 1 < blocksize < n
    @assert cutoff > 0
    @assert workgroupsize in (32, 64, 128, 256)
    @assert !use_mps || eltype(A) == Float32
    data = copy(A)
    @assert all(isfinite, data)
    ipiv = similar(data, LinearAlgebra.BlasInt, n)
    info = similar(ipiv, 1)
    fill!(info, 0)
    schedules = Dict(w => MtlArray(schedule(w)) for w in 1:cutoff)
    sfmin = AN.lu_safemin(AN.realtype(eltype(A)))
    for k in 1:blocksize:n
        width = min(blocksize, n - k + 1)
        panel, last = k:(k + width - 1), (k + width):n
        if cooperative
            panel!(data, ipiv, info, k, width, sfmin, schedules, cutoff, use_mps, workgroupsize)
        else
            AN.getrf2!(data, ipiv, info, k, width, RowMaximum(), sfmin)
        end
        AN.lu_swap_rows!(data, ipiv, panel, 1:(k - 1))
        AN.lu_swap_rows!(data, ipiv, panel, last)
        AN.lu_trsm!(view(data, panel, panel), view(data, panel, last), false, true)
        gemm!(view(data, last, last), view(data, last, panel), view(data, panel, last), use_mps)
    end
    status = only(Array(info))
    return LinearAlgebra.LU(NLAArray(data), ipiv, status)
end

end
