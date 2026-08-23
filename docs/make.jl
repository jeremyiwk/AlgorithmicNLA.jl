using Documenter
using AlgorithmicNLA

makedocs(
    sitename = "AlgorithmicNLA.jl",
    modules = [AlgorithmicNLA],
    format = Documenter.HTML(
        canonical = "https://jeremyiwk.github.io/AlgorithmicNLA.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Roadmap" => "roadmap.md",
        "API reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/jeremyiwk/AlgorithmicNLA.jl.git",
    devbranch = "main",
    push_preview = false,
)
