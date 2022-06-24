module TopicTiling
include("Utls.jl")
using Statistics
using .Utls

"""
    TopicTiling.SegmentObject(window_size, smooth_window_size, lda_model, dictionary)
TopicTiling is an extension of TextTiling that uses the topic IDs of words in a sentence to calculate the similarity between blocks.
# Arguments
- `window_size`: Sliding window size.
- `do_smooth`: If true, smoothing depth scores.
- `smooth_window_size`: Window size for smoothing depth scores.
- `lda_model`: Trained LDA topic model.
- `dictionary`: A dictionary showing word-id mappings.
"""
mutable struct SegmentObject
    window_size::Int
    do_smooth::Bool
    smooth_window_size::Int
    lda_model::Any
    dictionary::Any
end

function preprocessing(seg::SegmentObject, tokenized_document)
    n = length(tokenized_document)
    seg.window_size = maximum([minimum([seg.window_size, n / 3]), 1])
    preprocessed_document = []
    for i = 1:n
        sentence_ids = []
        for w in seg.dictionary.doc2bow(tokenized_document[i])
            topic_id = seg.lda_model.get_term_topics(w[1])
            if topic_id != []
                append!(sentence_ids, [string.(sort(topic_id, by=last)[end][1])])
            end
        end
        append!(preprocessed_document, [Utls.count_elements(sentence_ids)])
    end

    return preprocessed_document
end

function calculate_gap_score(seg::SegmentObject, preprocessed_document)
    n = length(preprocessed_document)
    gap_score = [0.0 for _ = 1:n]

    for i = 1:n
        sz = minimum([minimum([i, n - i]), seg.window_size])
        left_side, right_side = Utls.SentenceElements(Dict{String,Int}()),
        Utls.SentenceElements(Dict{String,Int}())

        for j = Int(i - sz + 1):Int(i)
            Utls.merge_elements(left_side, preprocessed_document[j])
        end

        for j = Int(i + 1):Int(i + sz)
            Utls.merge_elements(right_side, preprocessed_document[j])
        end

        gap_score[i] =
            Utls.calculate_cosin_similarity(left_side.elements_dct, right_side.elements_dct)
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
    TopicTiling.segment(seg, document, [num_topics]) -> String
Performs the splitting of the document entered in the `document` argument.
# Arguments
- `seg`: Segment object.
- `document`: The document to be text segmented.
- `num_topics`: num_topics is the number of topics in the document. If this value is specified, segment boundaries are determined by the number of num_topics, starting with the highest depth score.

# Examples
```jldoctest
using TextSegmentation

# LDA Topic Model
pygensim = pyimport("gensim")
# train_document
# Data to be used when training the LDA topic model.
# Data from the same domain as the text to be segmented is preferred.

function read_file(file_path)
    f = open(file_path, "r")
    return filter((i) -> length(i) > 5, split.(lowercase(replace.(read(f, String), "\r"=>"")), "\n"))
    close(f)
end

file_path = [
    "/data/Relativity the Special and General Theory.txt",
    "/data/On Liberty.txt",
    "/data/Dream Psychology Psychoanalysis for Beginners.txt",
]

train_document = []
for i in file_path
    append!(train_document, read_file(i))
end

tokenized_train_document = [Utls.tokenize(i) for i in train_document]
dictionary = pygensim.corpora.Dictionary(tokenized_train_document)
corpus = [dictionary.doc2bow(text) for text in tokenized_train_document]
lda_model = pygensim.models.ldamodel.LdaModel(
    corpus = corpus,
    id2word = dictionary,
    minimum_probability = 0.0001,
    num_topics = 3,
    random_state=1234,
)

# TopicTiling
window_size = 2
do_smooth = false
smooth_window_size = 1
num_topics = 3
to = TopicTiling.SegmentObject(window_size, do_smooth, smooth_window_size, lda_model, dictionary)
result = TopicTiling.segment(to, document, num_topics)
println(result)
00010010000
```
"""
function segment(seg, test_document, num_topics=Nothing)
    tokenized_test_document = [Utls.tokenize(i) for i in test_document]
    preprocessed_document = preprocessing(seg, tokenized_test_document)
    gap_score = calculate_gap_score(seg, preprocessed_document)
    depth_score = calculate_depth_score(seg, gap_score)
    if seg.do_smooth
        depth_score = smoothing(seg, depth_score)
    end
    return determine_boundaries(seg, depth_score, num_topics)
end
end
