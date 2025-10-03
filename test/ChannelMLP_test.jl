using Lux, Random, Test
using LuxTestUtils  # For gradient testing

@testset "ChannelMLP Tests" begin
    
    rng = Random.default_rng()
    Random.seed!(rng, 42)  
    
    # Test configuration constants
    channels = 16
    expansion_factor = 2
    batch_size = 10
    input_dims = (64, 64, channels, batch_size)
    hidden_channels = Int(channels * expansion_factor)
    input_dims = (64, 64, channels, batch_size)

    @testset "Initialization" begin
        # Test that layer constructs properly
        layer = ESM_PINO.ChannelMLP(channels, expansion_factor=expansion_factor)
        
        # Test parameter initialization
        ps, st = Lux.setup(rng, layer)
        
        # Verify parameters exist and have correct shapes
        @test haskey(ps, :mlp)
        @test haskey(ps.mlp, :layer_1)
        @test haskey(ps.mlp, :layer_2)
        @test haskey(ps, :skip)
        @test size(ps.mlp.layer_1.weight) == (1,1, channels, hidden_channels)
        @test size(ps.mlp.layer_2.weight) == (1,1, hidden_channels, channels)
        @test size(ps.skip.weight) == (1,1, channels, 1)
        
        # Verify state is properly initialized 
        @test st != nothing
    end
    
    @testset "Forward Pass" begin
        # Setup layer and input data
        layer = ESM_PINO.ChannelMLP(channels, expansion_factor=expansion_factor )
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, Float32, (input_dims)...)
        
        # Test basic forward pass
        y, st_update = Lux.apply(layer, x, ps, st)
        
        # Test output shape
        @test size(y) == (input_dims[1], input_dims[2], channels, batch_size)
        
        # Test output values are valid
        @test all(isfinite, y)
        
    end
    
    @testset "Backward Pass - Gradient Correctness" begin
        layer = ESM_PINO.ChannelMLP(channels, expansion_factor=expansion_factor)
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