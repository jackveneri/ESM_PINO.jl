using Lux, Random, Test
using LuxTestUtils  # For gradient testing

@testset "SpectralKernel Tests" begin
    
    rng = Random.default_rng()
    Random.seed!(rng, 42)  
    
    # Test configuration constants
    in_channels = 3
    out_channels = 3
    modes = (16,16)
    batch_size = 10
    input_dims = (64, 64, in_channels, batch_size)

    @testset "Initialization" begin
        # Test that layer constructs properly
        layer = ESM_PINO.SpectralKernel(in_channels, out_channels, modes)
        
        # Test parameter initialization
        ps, st = Lux.setup(rng, layer)
        
        # Verify parameters exist and have correct shapes
        @test haskey(ps, :spatial)
        @test haskey(ps, :spectral)
        @test size(ps.spectral.weight) == (modes..., out_channels, in_channels)
        @test size(ps.spatial.weight) == (1, 1, in_channels, out_channels)
        # Verify state is properly initialized 
        @test st != nothing
    end
    
    @testset "Forward Pass" begin
        # Setup layer and input data
        layer = ESM_PINO.SpectralKernel(in_channels, out_channels, modes)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, Float32, input_dims...)
        
        # Test basic forward pass
        y, st_update = Lux.apply(layer, x, ps, st)
        
        # Test output shape
        @test size(y) == (input_dims[1], input_dims[2], out_channels, batch_size)
        
        # Test output values are valid
        @test all(isfinite, y)
        
    end
    
    @testset "Backward Pass - Gradient Correctness" begin
        layer = ESM_PINO.SpectralKernel(in_channels, out_channels, modes)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, Float32, input_dims...)
        
        function loss_fn(x, ps)
            y, _ = Lux.apply(layer, x, ps, st)
            return sum(abs2, y)  # Simple L2 loss
        end
        
        # Test gradients using LuxTestUtils :
        @testset "Gradient Tests" begin
            @test_gradients(loss_fn, x, ps; skip_backends=[:Enzyme, :AutoEnzyme])
        end
    end    
end