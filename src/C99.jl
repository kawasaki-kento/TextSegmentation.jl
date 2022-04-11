module C99
include("Utls.jl")
using Statistics
using .Utls

"""
    C99.SegmentObject(window_size, similarity_matrix, rank_matrix, sum_matrix, std_coeff, tokenizer)
C99 is a method for determining segment boundaries through divisive clustering.
# Arguments
- `window_size`: window_size is used to create a rank matrix and specifies the range of adjacent sentences to be referenced.
- `similarity_matrix`: Matrix of calculated cosine similarity between sentences.
- `rank_matrix`: Each value in the similarity matrix is replaced by a rank in the local domain. A rank is the number of neighboring elements with lower similarity score.
- `sum_matrix`: Sum of rank matrix in segment regions i to j.
- `std_coeff`: std_coeff is used for the threshold that determines the segment boundary. μ and v are the mean and variance of the gradient δD(n) of the internal density.
- `tokenizer`: Tokenizer for word segmentation.
"""
mutable struct SegmentObject
    window_size::Int
    similarity_matrix::Array{Float64,2}
    rank_matrix::Array{Float64,2}
    sum_matrix::Array{Float64,2}
    std_coeff::Float64
    tokenizer::Any
end

function preprocessing(seg::SegmentObject, document)
    n = length(document)
    @assert n > 0 && length([d for d in document if typeof(d) != String]) == 0
    if n < 3
        return [1] + [0 for _ = 1:n-1]
    end

    seg.window_size = minimum([seg.window_size, n])
    return [Utls.count_elements(seg.tokenizer(document[i])) for i = 1:n]
end

function create_similarity_matrix(seg::SegmentObject, n, preprocessed_document)
    seg.similarity_matrix = zeros(n, n)
    for i = 1:n
        for j = i:n
            seg.similarity_matrix[i, j] = Utls.calculate_cosin_similarity(
                preprocessed_document[i],
                preprocessed_document[j],
            )
            seg.similarity_matrix[j, i] = seg.similarity_matrix[i, j]
        end
    end
end

function create_rank_matrix(seg::SegmentObject, n)
    seg.rank_matrix = zeros(n, n)
    for i = 1:n
        for j = i:n
            r1 = maximum([1, i - seg.window_size + 1])
            r2 = minimum([n, i + seg.window_size - 1])
            c1 = maximum([1, j - seg.window_size + 1])
            c2 = minimum([n, j + seg.window_size - 1])

            s = seg.similarity_matrix[r1:(r2), c1:(c2)]
            k = size(s)
            l = k[1] * k[2]
            sublist = reshape(transpose(s), 1, l)
            lowlist = [x for x in sublist if x < seg.similarity_matrix[i, j]]

            seg.rank_matrix[i, j] = 1.0 * length(lowlist) / ((r2 - r1 + 1) * (c2 - c1 + 1))
            seg.rank_matrix[j, i] = seg.rank_matrix[i, j]
        end
    end
end

function create_sum_matrix(seg::SegmentObject, n)
    seg.sum_matrix = zeros(n, n)
    prefix_sm = zeros(n, n)
    for i = 1:n
        for j = 1:n
            prefix_sm[i, j] = seg.rank_matrix[i, j]
            if i - 1 >= 1
                prefix_sm[i, j] += prefix_sm[i-1, j]
            end
            if j - 1 >= 1
                prefix_sm[i, j] += prefix_sm[i, j-1]
            end
            if i - 1 >= 1 && j - 1 >= 1
                prefix_sm[i, j] -= prefix_sm[i-1, j-1]
            end
        end
    end
    for i = 1:n
        for j = i:n
            if i == 1
                seg.sum_matrix[i, j] = prefix_sm[j, j]
            else
                seg.sum_matrix[i, j] =
                    prefix_sm[j, j] - prefix_sm[i-1, j] - prefix_sm[j, i-1] +
                    prefix_sm[i-1, i-1]
            end
            seg.sum_matrix[j, i] = seg.sum_matrix[i, j]
        end
    end
end

function determine_boundaries(seg::SegmentObject, n)
    D = 1.0 * seg.sum_matrix[1, n] / (n * n)
    g = init_region(1, n, seg)
    darr, region_arr, idx = [D], [g], []
    sum_region, sum_area = float(seg.sum_matrix[1, n]), float(n * n)
    for i = 1:n-1
        mx, pos = -1e9, -1

        for (j, region) in enumerate(region_arr)
            if region.l == region.r
                continue
            end

            split_region(region, seg)
            den = sum_area - region.area + region.lch.area + region.rch.area
            cur = (sum_region - region.tot + region.lch.tot + region.rch.tot) / den
            if cur > mx
                mx, pos = cur, j
            end
        end

        @assert pos >= 1
        tmp = region_arr[pos]
        region_arr[pos] = tmp.rch
        insert!(region_arr, pos, tmp.lch)
        sum_region += tmp.lch.tot + tmp.rch.tot - tmp.tot
        sum_area += tmp.lch.area + tmp.rch.area - tmp.area
        append!(darr, sum_region / sum_area)
        append!(idx, tmp.best_pos)
    end

    dgrad = [(darr[i+1] - darr[i]) for i = 1:length(darr)-1]
    cutoff_threshold = mean(dgrad) + seg.std_coeff * std(dgrad)

    @assert length(idx) == length(dgrad)
    above_cutoff_idx = [i for i in 1:length(dgrad) if dgrad[i] >= cutoff_threshold]

    boundaries = ["0" for _ = 1:n]
    if length(above_cutoff_idx) != 0
        for i in idx[1:maximum(above_cutoff_idx)]
            boundaries[i] = "1"
        end
    end
    return join(boundaries[1:end-1])
end

mutable struct Region
    tot::Float64
    l::Int
    r::Int
    area::Float64
    lch::Any
    rch::Any
    best_pos::Int
end

function init_region(l, r, seg::SegmentObject)
    @assert r >= l
    g = Region(seg.sum_matrix[l, r], l, r, 0, nothing, nothing, -1)
    g.area = (r - l + 1)^2
    return g
end

function split_region(g::Region, seg::SegmentObject)
    if g.best_pos >= 1
        return
    end
    if g.l == g.r
        g.best_pos = g.l
        return
    end

    @assert g.r > g.l
    mx, pos = -1e9, -1
    for i = g.l:g.r-1
        carea = (i - g.l + 1)^2 + (g.r - i)^2
        cur = (seg.sum_matrix[g.l, i] + seg.sum_matrix[i+1, g.r]) / carea
        if cur > mx
            mx, pos = cur, i
        end
    end
    @assert pos >= g.l && pos < g.r
    g.lch = init_region(g.l, pos, seg)
    g.rch = init_region(pos + 1, g.r, seg)
    g.best_pos = pos

end

"""
    C99.segment(seg, document, n) -> String
Performs the splitting of the document entered in the `document` argument.
# Arguments
- `seg`: Segment object.
- `document`: The document to be text segmented.
- `n`: Document Length.

# Examples
```jldoctest
using TextSegmentation

n = length(document)
init_matrix = zeros(n, n)
window_size = 2
std_coeff = 1.2

c99 = C99.SegmentObject(window_size, init_matrix, init_matrix, init_matrix, std_coeff, Utls.tokenize)
result = C99.segment(c99, document, n)
println(result)
000100010000
```
"""
function segment(seg, document, n)
    preprocessed_document = preprocessing(seg, document)
    create_similarity_matrix(seg, n, preprocessed_document)
    create_rank_matrix(seg, n)
    create_sum_matrix(seg, n)
    return determine_boundaries(seg, n)
end
end

