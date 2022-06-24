using TextSegmentation
using Test
using PyCall

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

document = [
    # Albert Einstein. (2004). Relativity : the Special and General Theory. Urbana, Illinois: Project Gutenberg. Retrieved June 23, 2022, from https://www.gutenberg.org/ebooks/5001.
    "Now in virtue of its motion in an orbit round the sun, our earth is comparable with a railway carriage travelling with a velocity of about 30 kilometres per second. ",
    "If the principle of relativity were not valid we should therefore expect that the direction of motion of the earth at any moment would enter into the laws of nature, and also that physical systems in their behaviour would be dependent on the orientation in space with respect to the earth. For owing to the alteration in direction of the velocity of revolution of the earth in the course of a year, the earth cannot be at rest relative to the hypothetical system K[0] throughout the whole year. ",
    "However, the most careful observations have never revealed such anisotropic properties in terrestrial physical space, i.e. a physical non-equivalence of different directions. ",
    "This is very powerful argument in favour of the principle of relativity.",
    # John Stuart Mill. (2011). On Liberty. Urbana, Illinois: Project Gutenberg. Retrieved June 23, 2022, from https://www.gutenberg.org/ebooks/34901.
    "The struggle between Liberty and Authority is the most conspicuous feature in the portions of history with which we are earliest familiar, particularly in that of Greece, Rome, and England.",
    "But in old times this contest was between subjects, or some classes of subjects, and the government.",
    "By liberty, was meant protection against the tyranny of the political rulers.",
    "The rulers were conceived (except in some of the popular governments of Greece) as in a necessarily antagonistic position to the people whom they ruled.",
    # Sigmund Freud. (2005). Dream Psychology: Psychoanalysis for Beginners. Urbana, Illinois: Project Gutenberg. Retrieved June 23, 2022, from https://www.gutenberg.org/ebooks/15489.
    "Through condensation of the dream certain constituent parts of its content are explicable which are peculiar to the dream life alone, and which are not found in the waking state.",
    "Such are the composite and mixed persons, the extraordinary mixed figures, creations comparable with the fantastic animal compositions of Orientals; a moment's thought and these are reduced to unity, whilst the fancies of the dream are ever formed anew in an inexhaustible profusion.",
    "Every one knows such images in his own dreams; manifold are their origins.",
    "I can build up a person by borrowing one feature from one person and one from another, or by giving to the form of one the name of another in my dream.",
    ]

@testset "TextTiling.jl" begin
    window_size = 2
    do_smooth = false
    smooth_window_size = 1
    num_topics = 3
    tt = TextTiling.SegmentObject(window_size, do_smooth, smooth_window_size, Utls.tokenize)
    result = TextTiling.segment(tt, document, num_topics)
    @test result == "00010001000"
end

@testset "C99.jl" begin
    n = length(document)
    init_matrix = zeros(n, n)
    window_size = 2
    std_coeff = 1.2
    c99 = C99.SegmentObject(window_size, init_matrix, init_matrix, init_matrix, std_coeff, Utls.tokenize)
    result = C99.segment(c99, document, n)
    @test result == "00010001000"
end

@testset "TopicTiling.jl" begin
    # LDA Topic Model
    pygensim = pyimport("gensim")
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
    @test result == "00010010000"
end