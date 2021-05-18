# based on https://github.com/chengchingwen/Transformers.jl/blob/master/src/fix/batched_gemm.jl

import NNlib: batched_gemm!
import Zygote: @adjoint
import CUDA: CuArray
import LinearAlgebra: mul!

const _GemmFloat = Union{Float64, Float32, ComplexF64, ComplexF32}

function batched_mul(A::AbstractArray{T1, 3}, B::AbstractArray{T2, 3}; transA=false, transB=false) where {T1, T2}
    axes(A, 3) == axes(B, 3) || throw(DimensionMismatch("batch size mismatch"))
    T = promote_type(T1, T2)
    C = similar(A, T, (axes(A, transA ? 2 : 1), axes(B, transB ? 1 : 2), axes(A, 3)))
    out  = batched_mul!(C, A, B, transA, transB)
    out
end

function batched_mul!(C::StridedArray{T, 3}, A::StridedArray{T, 3}, B::StridedArray{T, 3}, transA, transB) where {T<:_GemmFloat}
    transA = transA ? 'T' : 'N'
    transB = transB ? 'T' : 'N'
    batched_gemm!(transA, transB, one(T), A, B, zero(T), C)
end

function batched_mul!(C::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3}, transA, transB)
    axes(A, 3) == axes(B, 3) == axes(C, 3) || throw(DimensionMismatch("batch size mismatch"))
    if transA; fA = transpose
    else fA = identity; end

    if transB; fB = transpose
    else fB = identity; end
    
    @debug "calling fallback method for batched_mul!" typeof(A) typeof(B) typeof(C)
    @inbounds for k in axes(C, 3)
        @views mul!(C[:,:,k], fA(A[:,:,k]), fB(B[:,:,k]))
    end
    C
end

# batched_mul(rand(10, 20, 3), rand(20, 100, 3))

function batched_mul!(C::CuArray{T, 3}, A::CuArray{T, 3}, B::CuArray{T, 3}, transA, transB) where T
    transA = transA ? 'T' : 'N'
    transB = transB ? 'T' : 'N'
    CUBLAS.gemm_strided_batched!(transA, transB, one(T), A, B, zero(T), C)
end

# CUDA.@time batched_mul(cu(rand(10, 20, 3)), cu(rand(20, 100, 3)))

@adjoint function batched_mul(A, B; transA=false, transB=false)
    C = batched_mul(A, B; transA, transB)
    pullback(Δ) = begin
        if transA
            if transB # A'B'
                Ā, B̄ = batched_mul(B, Δ; transA=true, transB=true), batched_mul(Δ, A; transA=true, transB=true)
            else # A'B
                Ā, B̄ = batched_mul(B, Δ; transB=true), batched_mul(A, Δ)
            end
        else
            if transB # A*B'
                Ā, B̄ = batched_mul(Δ, B), batched_mul(Δ, A, transA=true)
            else # A*B̄
                Ā, B̄ = batched_mul(Δ, B, transB=true), batched_mul(A, Δ, transA=true)
            end
        end
        (Ā, B̄)
    end
    (C, pullback)
end

# a = rand(2, 3, 4)

# Zygote.gradient(a) do a
#     sum(batched_mul(a, a, transB=true))
# end
