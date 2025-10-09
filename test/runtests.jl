using ESM_PINO
using Test


#include("SpectralConv_test.jl")
#include("SpectralKernel_test.jl")
#include("ChannelMLP_test.jl")
#include("FNO_Block_test.jl")
#include("FourierNeuralOperator_test.jl")

#maybe put a flag here to skip if you don't want to test extension
include("SphericalConv_test.jl")