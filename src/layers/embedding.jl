struct Embedding{M<:AbstractMatrix}
    table::M
end
Embedding(voc_size, feature_size) = Embedding(randn(feature_size, voc_size))
(e::Embedding)(x) = e.table[:, x]
Flux.@functor Embedding

struct PosEnc{M<:AbstractMatrix}
    feature_size::Int
    max_seq::Int
    lookup::M
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
(p::PosEnc)(x) = x .* eltype(x)(√p.feature_size) .+ p.lookup[:, 1:size(x, 2)]
Flux.@functor PosEnc
Flux.trainable(::PosEnc) = ()
