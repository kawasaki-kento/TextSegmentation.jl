# Introduction
TextSegmentation.jlは、教師なしテキストセグメンテーション手法のjulia実装を提供します。  
主な手法は以下の通りです。
 - TextTiling
 - C99
 - TopicTiling

# Requirements
```
Julia Version 1.7.2
Languages v0.4.3
PyCall v1.93.0
Statistics
```

# Usage
## Import Package
```julia
include("TextTiling.jl")
include("C99.jl")
include("TopicTiling.jl")

using PyCall
using .Utls
using .TextTiling
using .C99
using .TopicTiling
```
## Sample Data
```julia
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
```
 - 入力データはドキュメントが一行ずつ分割され、リスト形式のデータになっているものを対象とします。
 - documentは[Project Gutenberg](https://www.gutenberg.org/)から引用した、3つのトピックからなる13文で構成されています。
 - このドキュメントに含まれる12個のセグメント境界候補に対して、0（セグメント境界でない）or 1（セグメント境界である）を決定します。
 - 以下に各手法のデモを示します。

## TextTiling
```julia
window_size = 2
smooth_window_size = 1
num_topics = 3
tt = TextTiling.Segmentation(window_size, smooth_window_size, Utls.tokenize)
result = TextTiling.segment(tt, document, num_topics)
println(result)
>>> 000100010000
```
 - TextTilingは語彙のまとまりに基づいて，隣接するブロック間の類似度からセグメント境界を探索する手法です。
 - window_sizeはブロックの大きさ、smooth_window_sizeは深度スコアを平滑化するために設定します。
 - num_topicsはドキュメントに含まれるトピックの数です。この値が指定された場合、深度スコアが大きいものから順にnum_topicsの数だけセグメント境界を決定します。
 - Utls.tokenizeは、文を単語分割するためのtokenizerです。tokenizerを変更することで他言語にも対応可能です。

## C99
```julia
n = length(document)
init_matrix = zeros(n, n)
window_size = 2
std_coeff = 1.2
c99 = C99.Segmentation(window_size, init_matrix, init_matrix, init_matrix, std_coeff, Utls.tokenize)
result = C99.segment(c99, document, n)
println(result)
>>> 000100001000
```
 - C99は分割型クラスタリングによってセグメント境界を決定する手法です。
 - window_sizeはrank matrixを作成する際に使用し、隣接する文の範囲を指定します。
 - std_coeffはセグメント境界を決定する閾値に使われます。μとvは、内部密度の勾配δD(n)の平均と分散であり、閾値は下記の式で求められます。

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mu&space;&plus;&space;c\sqrt{v}" title="\mu + c\sqrt{v}" />
</p>

 - 式中のcがstd_coeffであり、c=1.2がよく機能することが知られています。
 - init_matrixは、similarity matrixやrank matrixを初期化するために設定しています。

## TopicTiling
```julia
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
smooth_window_size = 1
num_topics = 3
to = TopicTiling.Segmentation(window_size, smooth_window_size, lda_model, dictionary)
result = TopicTiling.segment(to, document, num_topics)
println(result)
>>> 000110000000
```
 - TopicTilingはTextTilingを拡張した手法であり、文に含まれる単語のトピックIDを用いてブロック間の類似度を算出します。
 - lda_modelは、Pythonライブラリであるgensimを使用しています。
 - TopicTilingでは、セグメント対象のドキュメントの他にlda_modelを構築するための学習ドキュメントが必要です。（学習ドキュメントは、セグメント対象のドキュメントとドメインが近いものが望ましいです。）
 - ライブラリの詳細やパラメータの設定については[models.ldamodel](https://radimrehurek.com/gensim/models/ldamodel.html)を参照してください。
 - window_sizeはブロックの大きさ、smooth_window_sizeは深度スコアを平滑化するために設定します。
 - num_topicsはドキュメントに含まれるトピックの数です。この値が指定された場合、深度スコアが大きいものから順にnum_topicsの数だけセグメント境界を決定します。

# Reference
 - [TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages](https://aclanthology.org/J97-1003.pdf)
 - [Advances in domain independent linear text segmentation](https://arxiv.org/pdf/cs/0003083.pdf)
 - [TopicTiling: A Text Segmentation Algorithm based on LDA](https://aclanthology.org/W12-3307.pdf)
 - [uts](https://github.com/intfloat/uts)