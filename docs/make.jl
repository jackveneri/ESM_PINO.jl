using Documenter, DocumenterVitepress
using ESM_PINO
include("../ext/ESM_PINOQG3Ext.jl")
using .ESM_PINOQG3Ext

DocMeta.setdocmeta!(ESM_PINO, :DocTestSetup, :(using ESM_PINO); recursive=true)

makedocs(;
    modules=[ESM_PINO, ESM_PINOQG3Ext],
    authors="Giacomo Veneri <giacomo.veneri@tum.de>",
    sitename="ESM_PINO.jl",
    format=DocumenterVitepress.MarkdownVitepress(repo = "github.com/jackveneri/ESM_PINO.jl.git", devbranch = "main", devurl = "https://jackveneri.github.io/ESM_PINO.jl/dev"),
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
