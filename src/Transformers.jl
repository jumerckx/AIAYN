module Transformers

export Transformer, loss

using CUDA, Flux, LinearAlgebra, Statistics

include("fix/batched_mul.jl")
include("layers/embedding.jl")
include("layers/attention.jl")
include("layers/encoder-decoder.jl")

struct Transformer
    encoder
    decoder
    out
end
Transformer(nlayers, voc_size, max_seq, feature_size, nheads) = Transformer(
    Encoder(nlayers, voc_size, max_seq, feature_size, nheads),
    Decoder(nlayers, voc_size, max_seq, feature_size, nheads),
    Affine(feature_size, voc_size))
function (t::Transformer)(x, src_mask, y, tgt_mask; p_drop=0.1)
    tgt_mask = Flux.Zygote.ignore() do
        gpu(triu(ones(Bool, size(y, 1), size(y, 1)))) .* tgt_mask # no peaking ahead!
    end

    enc_out = t.encoder(x, src_mask; p_drop)
    dec_out = t.decoder(enc_out, src_mask, y, tgt_mask; p_drop)
    softmax(t.out(dec_out))
end
Flux.@functor Transformer


function loss(model, x, src_mask, y, tgt_mask, PAD=PAD)
    ŷ = model(x, src_mask, y[1:end-1, :], tgt_mask[1:end-1, :, :])

    y = y[2:end, :]
    I = Flux.Zygote.ignore() do
        I = map(CartesianIndices(y)) do i
            CartesianIndex(y[i], Tuple(i)...)
        end
        return filter(I[:]) do i
            !(Tuple(i)[1] == PAD)
        end
    end
    
    return mean(-log.(ŷ[I]))
end

end
