using Documenter
using AlgorithmicNLA

makedocs(
    sitename = "AlgorithmicNLA.jl",
    modules = [AlgorithmicNLA],
    format = Documenter.HTML(
        canonical = "https://jeremyiwk.github.io/AlgorithmicNLA.jl",
        edit_link = "master",
    ),
    pages = [
        "Home" => "index.md",
        "Roadmap" => "roadmap.md",
        "API reference" => "api.md",
        "References" => "references.md",
        "Changelog" => "changelog.md",
    ],
)

deploydocs(
    repo = "github.com/jeremyiwk/AlgorithmicNLA.jl.git",
    devbranch = "master",
    push_preview = false,
)
