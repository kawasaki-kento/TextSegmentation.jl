module TextTiling
include("Utils.jl")
using Statistics
using .Utils

"""
    TextTiling.SegmentObject(window_size, do_smooth, smooth_window_size, tokenizer)
TextTiling is a method for finding segment boundaries based on lexical cohesion and similarity between adjacent blocks.
# Arguments
- `window_size`: Sliding window size.
- `do_smooth`: If true, smoothing depth scores.
- `smooth_window_size`: Window size for smoothing depth scores.
- `tokenizer`: Tokenizer for word segmentation.
"""
mutable struct SegmentObject
    window_size::Int
    do_smooth::Bool
    smooth_window_size::Int
    tokenizer::Any
end

function preprocessing(seg::SegmentObject, document)
    n = length(document)
    @assert n > 0 && length([d for d in document if typeof(d) != String]) == 0
    seg.window_size = maximum([minimum([seg.window_size, n / 3]), 1])
    return [Utils.count_elements(seg.tokenizer(document[i])) for i = 1:n]
end

function calculate_gap_score(seg::SegmentObject, preprocessed_document)
    n = length(preprocessed_document)
    gap_score = [0.0 for _ = 1:n]

    for i = 1:n
        sz = minimum([minimum([i, n - i]), seg.window_size])
        left_side, right_side = Utils.SentenceElements(Dict{String,Int}()),
        Utils.SentenceElements(Dict{String,Int}())

        for j = Int(i - sz + 1):Int(i)
            Utils.merge_elements(left_side, preprocessed_document[j])
        end

        for j = Int(i + 1):Int(i + sz)
            Utils.merge_elements(right_side, preprocessed_document[j])
        end

        gap_score[i] = Utils.calculate_cosin_similarity(
            left_side.elements_dct,
            right_side.elements_dct,
        )
    end
    return gap_score
end

function calculate_depth_score(seg::SegmentObject, gap_score)
    n = length(gap_score)
    depth_score = [0.0 for _ = 1:n]

    for i = 1:n
        if i <= seg.window_size || i + seg.window_size > n
            continue
        end

        ptr = i - 1
        while ptr >= 1 && gap_score[ptr] >= gap_score[ptr+1]
            ptr -= 1
        end
        lval = gap_score[ptr+1]

        ptr = i + 1
        while ptr < n && gap_score[ptr] >= gap_score[ptr-1]
            ptr += 1
        end
        rval = gap_score[ptr-1]

        depth_score[i] = 1 / 2 * (lval - gap_score[i] + rval - gap_score[i])
    end
    return depth_score
end

function smoothing(seg::SegmentObject, depth_score)
    n = length(depth_score)
    smooth_depth_score = [0.0 for _ = 1:n]

    for i = 1:n
        if i - seg.smooth_window_size < 1 || i + seg.smooth_window_size >= n
            smooth_depth_score[i] = depth_score[i]
        else
            smooth_depth_score[i] =
                mean(depth_score[(i-seg.smooth_window_size):(i+seg.smooth_window_size)])
        end
    end
    return smooth_depth_score
end

function determine_boundaries(seg::SegmentObject, smooth_depth_score, num_topics)
    n = length(smooth_depth_score)
    boundaries = ["0" for _ = 1:n]
    cutoff_threshold = mean(smooth_depth_score) - std(smooth_depth_score) / 2.0
    depth_tuples = [(x, y) for (x, y) in zip(smooth_depth_score, [i for i = 1:n])]
    depth_tuples = reverse(sort(depth_tuples))

    if num_topics == Nothing
        for x in depth_tuples
            if x[1] > cutoff_threshold
                boundaries[x[2]] = "1"
            end
        end
    else
        for (i, x) in enumerate(depth_tuples)
            if x[1] > cutoff_threshold && i <= num_topics - 1
                boundaries[x[2]] = "1"
            end
        end
    end
    return join(boundaries[1:end-1])
end

"""
    TextTiling.segment(seg, document, [num_topics]) -> String
Performs the splitting of the document entered in the `document` argument.
# Arguments
- `seg`: Segment object.
- `document`: The document to be text segmented.
- `num_topics`: num_topics is the number of topics in the document. If this value is specified, segment boundaries are determined by the number of num_topics, starting with the highest depth score.

# Examples
```jldoctest
using TextSegmentation

window_size = 2
do_smooth = false
smooth_window_size = 1
num_topics = 3
tt = TextTiling.SegmentObject(window_size, do_smooth, smooth_window_size, Utils.tokenize)
result = TextTiling.segment(tt, document, num_topics)
println(result)
00010001000
```
"""
function segment(seg, document, num_topics = Nothing)
    preprocessed_document = preprocessing(seg, document)
    gap_score = calculate_gap_score(seg, preprocessed_document)
    depth_score = calculate_depth_score(seg, gap_score)
    if seg.do_smooth
        depth_score = smoothing(seg, depth_score)
    end
    return determine_boundaries(seg, depth_score, num_topics)
end
end





