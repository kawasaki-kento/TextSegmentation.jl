module TextSegmentation

include("Utils.jl")
include("TextTiling.jl")
include("C99.jl")
include("TopicTiling.jl")

export Utils, TextTiling, C99, TopicTiling

end
