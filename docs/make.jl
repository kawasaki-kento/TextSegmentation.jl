using TextSegmentation
using Documenter

DocMeta.setdocmeta!(TextSegmentation, :DocTestSetup, :(using TextSegmentation); recursive=true)

makedocs(;
    modules=[TextSegmentation],
    authors="Kento Kawasaki",
    repo="https://github.com/kawasaki-kento/TextSegmentation.jl/blob/{commit}{path}#{line}",
    sitename="TextSegmentation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kawasaki-kento.github.io/TextSegmentation.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    doctest = false,
)

deploydocs(;
    repo="github.com/kawasaki-kento/TextSegmentation.jl",
    devbranch="main",
)
