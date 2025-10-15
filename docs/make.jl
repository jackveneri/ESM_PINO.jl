using Documenter
using ESM_PINO

DocMeta.setdocmeta!(ESM_PINO, :DocTestSetup, :(using ESM_PINO); recursive=true)

makedocs(;
    modules=[ESM_PINO],
    authors="Giacomo Veneri <giacomo.veneri@tum.de>",
    sitename="ESM_PINO.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jackveneri.github.io/ESM_PINO.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Reference" => "ref.md",
    ],
)

deploydocs(;
    repo="github.com/jackveneri/ESM_PINO.jl.git",
    devbranch="main",
    push_preview=true,
)
