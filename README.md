# TextSegmentation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kawasaki-kento.github.io/TextSegmentation.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kawasaki-kento.github.io/TextSegmentation.jl/dev)
[![Build Status](https://travis-ci.com/kawasaki-kento/TextSegmentation.jl.svg?branch=main)](https://travis-ci.com/kawasaki-kento/TextSegmentation.jl)

# Introduction
TextSegmentation.jl provides a julia implementation of the unsupervised text segmentation method.  
The main methods are as follows.
 - TextTiling
 - C99
 - TopicTiling

# Documentation
 - [TextSegmentation.jl](https://kawasaki-kento.github.io/TextSegmentation.jl/build/)

# Requirements
```
Julia Version 1.7.2
Languages v0.4.3
PyCall v1.93.0
```

# Usage
## Import Package
```julia
using TextSegmentation
```
## Sample Data
```julia
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
```
 - The input data is for documents that are divided line by line and are in list form.
 - The document consists of 13 sentences, taken from [Project Gutenberg](https://www.gutenberg.org/), consisting of three topics.
 - For each of the 12 candidate segment boundaries in the document, we determine 0 (not a segment boundary) or 1 (a segment boundary).
 - Demonstrations of each method are shown below.

## TextTiling
```julia
window_size = 2
do_smooth = false
smooth_window_size = 1
num_topics = 3
tt = TextTiling.SegmentObject(window_size, do_smooth, smooth_window_size, Utils.tokenize)
result = TextTiling.segment(tt, document, num_topics)
println(result)
>>> 00010001000
```
 - TextTiling is a method for finding segment boundaries based on lexical cohesion and similarity between adjacent blocks.
 - window_size is the size of the block and smooth_window_size is set to smooth the depth score.
 - num_topics is the number of topics in the document. If this value is specified, segment boundaries are determined by the number of num_topics, starting with the highest depth score.
 - Utils.tokenize is a tokenizer for word segmentation of sentences; other languages can be supported by changing the tokenizer.

## C99
```julia
n = length(document)
init_matrix = zeros(n, n)
window_size = 2
std_coeff = 1.2
c99 = C99.SegmentObject(window_size, init_matrix, init_matrix, init_matrix, std_coeff, Utils.tokenize)
result = C99.segment(c99, document, n)
println(result)
>>> 00010001000
```
 - C99 is a method for determining segment boundaries through segmented clustering.
 - window_size is used to create a rank matrix and specifies the range of adjacent sentences to be referenced.
 - std_coeff is used for the threshold that determines the segment boundary. μ and v are the mean and variance of the gradient δD(n) of the internal density. The threshold value is calculated by the following equation.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mu&space;&plus;&space;c\sqrt{v}" title="\mu + c\sqrt{v}" />
</p>

 - It is known that c in the equation is std_coeff and that c=1.2 works well.
 - init_matrix is set to initialize the similarity matrix and rank matrix.

## TopicTiling
```julia
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

tokenized_train_document = [Utils.tokenize(i) for i in train_document]
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
>>> 00010010000
```
 - TopicTiling is an extension of TextTiling that uses the topic IDs of words in a sentence to calculate the similarity between blocks.
 - lda_model uses gensim, a Python library.
 - TopicTiling requires a training document to build the lda_model in addition to the documents to be segmented. (The training document should be close in domain to the document to be segmented.)
 - See [models.ldamodel](https://radimrehurek.com/gensim/models/ldamodel.html) for library details and parameter settings.
 - window_size is the size of the block and smooth_window_size is set to smooth the depth score.
 - num_topics is the number of topics in the document. If this value is specified, segment boundaries are determined by the number of num_topics, starting with the highest depth score.

# Reference
 - [TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages](https://aclanthology.org/J97-1003.pdf)
 - [Advances in domain independent linear text segmentation](https://arxiv.org/pdf/cs/0003083.pdf)
 - [TopicTiling: A Text Segmentation Algorithm based on LDA](https://aclanthology.org/W12-3307.pdf)
 - [uts](https://github.com/intfloat/uts)