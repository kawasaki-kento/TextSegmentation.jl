using Documenter
using TextSegmentation

makedocs(
    sitename = "TextSegmentation",
    format = Documenter.HTML(),
    modules = [TextSegmentation]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
