using ESM_PINO
include("../ext/ESM_PINOQG3Ext.jl")
using .ESM_PINOQG3Ext
using Documenter
using DocumenterVitepress


DocMeta.setdocmeta!(ESM_PINO, :DocTestSetup, :(using ESM_PINO); recursive=true)

makedocs(;
    modules=[ESM_PINO, ESM_PINOQG3Ext],
    warnonly = true,
    authors="Giacomo Veneri <giacomo.veneri@tum.de>",
    sitename="ESM_PINO.jl",
    repo="https://github.com/jackveneri/ESM_PINO.jl",
    format=DocumenterVitepress.MarkdownVitepress(repo = "https://github.com/jackveneri/ESM_PINO.jl", devurl="dev", deploy_url="jackveneri.github.io/ESM_PINO.jl"),
    pages=[
        "Home" => "index.md",
        "Reference" => "ref.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo="github.com/jackveneri/ESM_PINO.jl",
    push_preview=true,
)
