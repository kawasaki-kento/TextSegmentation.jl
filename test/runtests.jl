using TextSegmentation
using Test
using PyCall

document = [
    # Derek J. de Solla Price. (2009). On the Origin of Clockwork, Perpetual Motion Devices, and the Compass. Urbana, Illinois: Project Gutenberg. Retrieved February 25, 2022, from www.gutenberg.org/ebooks/30001.
    "Once the floodgates of Arabic learning were opened, a stream of mechanized astronomical models poured into Europe.",
    "Astrolabes and equatoria rapidly became very popular, mainly through the reason for which they had been first devised, the avoidance of tedious written computation.",
    "Many medieval astrolabes have survived, and at least three medieval equatoria are known.",
    "Chaucer is well known for his treatise on the astrolabe; a manuscript in Cambridge, containing a companion treatise on the equatorium, has been tentatively suggested by the present author as also being the work of Chaucer and the only piece written in his own hand.",
    # Carroll, Lewis. (2006). Alice in Wonderland. Urbana, Illinois: Project Gutenberg. Retrieved February 25, 2022, from www.gutenberg.org/ebooks/19033.
    "\"You are not attending!\" said the Mouse to Alice, severely.",
    "\"What are you thinking of?\"",
    "\"I beg your pardon,\" said Alice very humbly, \"you had got to the fifth bend, I think?\"",
    "\"You insult me by talking such nonsense!\" said the Mouse, getting up and walking away.",
    "\"Please come back and finish your story!\" Alice called after it.",
    # John Stuart Mill. (2011). On Liberty. Urbana, Illinois: Project Gutenberg. Retrieved February 25, 2022, from www.gutenberg.org/ebooks/34901.
    "The struggle between Liberty and Authority is the most conspicuous feature in the portions of history with which we are earliest familiar, particularly in that of Greece, Rome, and England.",
    "But in old times this contest was between subjects, or some classes of subjects, and the government.",
    "By liberty, was meant protection against the tyranny of the political rulers.",
    "The rulers were conceived (except in some of the popular governments of Greece) as in a necessarily antagonistic position to the people whom they ruled.",
    ]

@testset "TextTiling.jl" begin
    window_size = 2
    smooth_window_size = 1
    num_topics = 3
    tt = TextTiling.SegmentObject(window_size, smooth_window_size, Utls.tokenize)
    result = TextTiling.segment(tt, document, num_topics)
    @test result == "000100010000"
end

@testset "C99.jl" begin
    n = length(document)
    init_matrix = zeros(n, n)
    window_size = 2
    std_coeff = 1.2
    c99 = C99.SegmentObject(window_size, init_matrix, init_matrix, init_matrix, std_coeff, Utls.tokenize)
    result = C99.segment(c99, document, n)
    @test result == "000100001000"
end

@testset "TopicTiling.jl" begin
    # LDA Topic Model
    pygensim = pyimport("gensim")
    tokenized_train_document = [Utls.tokenize(i) for i in document]
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
    smooth_window_size = 1
    num_topics = 3
    to = TopicTiling.SegmentObject(window_size, smooth_window_size, lda_model, dictionary)
    result = TopicTiling.segment(to, document, num_topics)
    @test result == "000110000000"
end