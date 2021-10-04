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
    [MultiHeadedAttention(feature_size, nheads) for _ in 1:nlayers],
    [Chain(Affine(feature_size, feature_size, relu), Affine(feature_size, feature_size)) for _ in 1:nlayers],
    [[[Flux.LayerNorm(feature_size) for _ in 1:2] for _ in 1:nlayers]..., Flux.LayerNorm(feature_size)])

function (e::Encoder)(x, src_mask; p_drop=0.1)
    x = Flux.Dropout(p_drop)(x |> e.embedding |> e.posenc)
    for (attention, feedforward, norm) in zip(e.attention, e.feedforward, e.norm[1:end-1])
        x = norm[1](Dropout(p_drop)(attention(x, x, x, src_mask)) .+ x)
        x = norm[2](Dropout(p_drop)(feedforward(x)) .+ x)
    end
    return e.norm[end](x)
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
    [MultiHeadedAttention(feature_size, nheads) for _ in 1:nlayers],
    [MultiHeadedAttention(feature_size, nheads) for _ in 1:nlayers],
    [Chain(Affine(feature_size, feature_size, relu), Affine(feature_size, feature_size)) for _ in 1:nlayers],
    [[[Flux.LayerNorm(feature_size) for _ in 1:3] for _ in 1:nlayers]..., Flux.LayerNorm(feature_size)])

function (d::Decoder)(enc_out, src_mask, y, tgt_mask; p_drop=0.1)
    y = Flux.Dropout(p_drop)(y |> d.embedding |> d.posenc)
    for (src_attention, tgt_attention, feedforward, norm) in zip(d.src_attention, d.tgt_attention, d.feedforward, d.norm[1:end-1])
        y = norm[1](Flux.Dropout(p_drop)(tgt_attention(y, y, y, tgt_mask)) .+ y)
        y = norm[2](Flux.Dropout(p_drop)(src_attention(y, enc_out, enc_out, src_mask)) .+ y)
        y = norm[3](Flux.Dropout(p_drop)(feedforward(y)) .+ y)
    end
    return d.norm[end](y)
end

Flux.@functor Decoder
