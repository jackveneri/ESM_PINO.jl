using Documenter, DocumenterVitepress
using ESM_PINO
include("../ext/ESM_PINOQG3Ext.jl")
using .ESM_PINOQG3Ext

DocMeta.setdocmeta!(ESM_PINO, :DocTestSetup, :(using ESM_PINO); recursive=true)

makedocs(;
    modules=[ESM_PINO, ESM_PINOQG3Ext],
    authors="Giacomo Veneri <giacomo.veneri@tum.de>",
    sitename="ESM_PINO.jl",
    repo="https://github.com/jackveneri/ESM_PINO.jl.git",
    format=DocumenterVitepress.MarkdownVitepress(repo = "github.com/jackveneri/ESM_PINO.jl.git", devbranch = "main",
    devurl = "dev"),
    pages=[
        "Home" => "index.md",
        "Reference" => "ref.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo="github.com/jackveneri/ESM_PINO.jl.git",
    target="build",
    devbranch="main",
    push_preview=true,
)
