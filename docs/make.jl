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
        "Array interface" => "arrays.md",
        "LU factorization" => "lu.md",
        "GPU performance workflow" => "performance.md",
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
