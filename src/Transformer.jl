using CUDA, KernelAbstractions, Flux, LinearAlgebra, Statistics
include("batched_mul.jl")

CUDA.allowscalar(false)

struct Embedding
    table
end
Embedding(voc_size, feature_size) = Embedding(Flux.glorot_normal(feature_size, voc_size))
(e::Embedding)(x) = e.table[:, x]
Flux.@functor Embedding

struct PosEnc{F, W<:AbstractArray{F}}
    model_size::Int
    max_seq::Int
    lookup::W
end
function PosEnc(max_seq, feature_size)
    lookup = zeros(feature_size, max_seq)
    for index in CartesianIndices(lookup)
        if iseven(index[1])
            lookup[index] = sin(index[2]/10000^((2*index[1])/feature_size))
        else
            lookup[index] = cos(index[2]/10000^((2*index[1])/feature_size))
        end
    end
    return PosEnc(feature_size, max_seq, lookup)
end
(p::PosEnc)(x) = x .+ p.lookup[:, 1:size(x, 2)]
Flux.@functor PosEnc
Flux.trainable(::PosEnc) = ()

#V*softmax((K'Q)/eltype(K)(√(size(K, 1))))
function attention(Q, K, V)
    score = batched_mul(K, Q, transA=true)
    score = softmax(score ./ eltype(K)(sqrt(size(K, 1))))
    batched_mul(V, score)
end

# function _attention(Q, K, V, mask)
function attention(Q, K, V, mask)
    score = batched_mul(K, Q, transA=true)
    score = score ./ eltype(K)(sqrt(size(K, 1)))
    score = softmax(score .+ ((1 .- mask).*eltype(score)(-1e9)))
    batched_mul(V, score)
end

# function attention(Q::C, K::C, V::C, mask) where C<:CuArray
#     _attention(Q, K, V, cu(mask))
# end
# attention(Q, K, V, mask) = _attention(Q, K, V, mask)

mutable struct Affine{F, S<:AbstractArray, T<:AbstractArray}
  W::S
  b::T
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

struct Multihead_attention
    heads
    out
end
function Multihead_attention(feature_size::Integer, nheads::Integer)
    heads = [Affine(feature_size, feature_size÷3) for _ in 1:nheads]
    out = Affine(nheads*(feature_size÷3), feature_size)
    Multihead_attention(heads, out)
end
function (a::Multihead_attention)(Q, K, V)
    a.out(cat([attention(head.([Q, K, V])...) for head in a.heads]..., dims=1))
end
function (a::Multihead_attention)(Q, K, V, mask)
    # if Q isa CuArray; mask = CuArray(mask) end

    a.out(cat([attention(head.([Q, K, V])..., mask) for head in a.heads]..., dims=1))
end
Flux.@functor Multihead_attention

struct Encoder
    embedding
    posenc
    attention
    feedforward
    norm
end
Encoder(nlayers::I, voc_size::I, max_seq::I, feature_size::I, nheads::I) where I<:Integer = Encoder(
    Embedding(voc_size, feature_size),
    PosEnc(max_seq, feature_size),
    [Multihead_attention(feature_size, nheads) for _ in 1:nlayers],
    [Chain(Affine(feature_size, feature_size, relu), Affine(feature_size, feature_size)) for _ in 1:nlayers],
    [[Flux.LayerNorm(feature_size) for _ in 1:2] for _ in 1:nlayers])
function (e::Encoder)(x, src_mask)
    x = Flux.Dropout(0.1)(x |> e.embedding |> e.posenc)
    for (attention, feedforward, norm) in zip(e.attention, e.feedforward, e.norm)
        x = norm[1](Dropout(0.1)(attention(x, x, x, src_mask)) .+ x)
        x = norm[2](Dropout(0.1)(feedforward(x)) .+ x)
    end
    return x
end
Flux.@functor Encoder

struct Decoder
    embedding
    posenc
    src_attention
    tgt_attention
    feedforward
    norm
end
Decoder(nlayers::I, voc_size::I, max_seq::I, feature_size::I, nheads::I) where I<:Integer = Decoder(
    Embedding(voc_size, feature_size),
    PosEnc(max_seq, feature_size),
    [Multihead_attention(feature_size, nheads) for _ in 1:nlayers],
    [Multihead_attention(feature_size, nheads) for _ in 1:nlayers],
    [Chain(Affine(feature_size, feature_size, relu), Affine(feature_size, feature_size)) for _ in 1:nlayers],
    [[Flux.LayerNorm(feature_size) for _ in 1:3] for _ in 1:nlayers])
function (d::Decoder)(enc_out, src_mask, y, tgt_mask)
    # tgt_mask = gpu(triu(ones(Bool, size(y, 1), size(y, 1)))) .* tgt_mask
    y = Flux.Dropout(0.1)(y |> d.embedding |> d.posenc)
    for (src_attention, tgt_attention, feedforward, norm) in zip(d.src_attention, d.tgt_attention, d.feedforward, d.norm)
        y = norm[1](Flux.Dropout(0.1)(tgt_attention(y, y, y, tgt_mask)) .+ y)
        y = norm[2](Flux.Dropout(0.1)(src_attention(y, enc_out, enc_out, src_mask)) .+ y)
        y = norm[3](Flux.Dropout(0.1)(feedforward(y)) .+ y)
    end
    return y
end

Flux.@functor Decoder

struct Transformer
    encoder
    decoder
    out
end
Transformer(nlayers, voc_size, max_seq, feature_size, nheads) = Transformer(
    Encoder(nlayers, voc_size, max_seq, feature_size, nheads),
    Decoder(nlayers, voc_size, max_seq, feature_size, nheads),
    Affine(feature_size, voc_size))
function (t::Transformer)(x, src_mask, y, tgt_mask)
    tgt_mask = gpu(triu(ones(Bool, size(y, 1), size(y, 1)))) .* tgt_mask

    enc_out = t.encoder(x, src_mask)
    dec_out = t.decoder(enc_out, src_mask, y, tgt_mask)
    softmax(t.out(dec_out))
end
Flux.@functor Transformer


loss(model, x, src_mask, y, tgt_mask, PAD=PAD) = begin
    ŷ = model(x, src_mask, y[1:end-1, :], tgt_mask[1:end-1, :, :])

    y = y[2:end, :]
    I = map(CartesianIndices(y)) do i
        CartesianIndex(y[i], Tuple(i)...)
    end
    # make this nicer when tgt_mask is on cpu:
    I = filter(I[:]) do i
        !(Tuple(i)[1] == PAD)
    end

    # I = map((x)->CartesianIndex(y[2:end, :][x], Tuple(x)...), CartesianIndices(y[2:end, :]))
    return mean(-log.(ŷ[I]))
end
