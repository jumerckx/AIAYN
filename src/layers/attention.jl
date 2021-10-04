# V*softmax((K'Q)/eltype(K)(√(size(K, 1))))

function attention(Q, K, V)
    score = batched_mul(K, Q, transA=true)
    score = softmax(score ./ eltype(K)(sqrt(size(K, 1))))
    batched_mul(V, score)
end

function attention(Q, K, V, mask)
    if isnothing(mask); return attention(Q, K, V); end

    score = batched_mul(K, Q, transA=true)
    score = score ./ eltype(K)(sqrt(size(K, 1)))
    mask = Flux.Zygote.ignore() do 
        ((1 .- mask).*eltype(score)(-1e9))
    end
    score = softmax(score .+ mask)
    batched_mul(V, score)
end

mutable struct Affine{F, M<:AbstractMatrix, B}
    W::M
    b::B
    σ::F
  end
  
function Affine(in::Integer, out::Integer, σ = identity; initW = Flux.glorot_uniform, initb = zeros)
return Affine(initW(out, in), initb(out), σ)
end
function (a::Affine)(x::AbstractArray{T, 3}) where T
    W, b, σ = a.W, a.b, a.σ
    orig_size = size(x)

    x = reshape(x, size(x, 1), :)
    out = σ.(W*x) .+ b

    reshape(out, :, orig_size[2:3]...)
end
Flux.@functor Affine

struct MultiHeadedAttention{P, O}
    feature_size::Int
    nheads::Int

    project_Q::P
    project_K::P
    project_V::P

    out::O
end
function MultiHeadedAttention(feature_size::Integer, nheads::Integer)
    @assert feature_size % nheads == 0
    projections = [Affine(feature_size, feature_size) for _ in 1:3]
    out = Affine(feature_size, feature_size)
    MultiHeadedAttention(feature_size, nheads, projections..., out)
end
function (a::MultiHeadedAttention)(Q, K, V, mask=nothing)
    head_size = a.feature_size ÷ a.nheads
    
    Q = a.project_Q(Q)
    K = a.project_K(K)
    V = a.project_V(V)
    
    attended = reduce(vcat,
        [attention(
            Q[i*head_size+1:(i+1)*head_size, :, :],
            K[i*head_size+1:(i+1)*head_size, :, :],
            V[i*head_size+1:(i+1)*head_size, :, :],
            mask) for i in 0:(a.nheads-1)
        ]
    )
    
    a.out(attended)
    
end
Flux.@functor MultiHeadedAttention
