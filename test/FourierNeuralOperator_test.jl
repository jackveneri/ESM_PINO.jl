using Lux, Random, Test
using LuxTestUtils  # For gradient testing

@testset "FourierNeuralOperator Tests" begin
    
    rng = Random.default_rng()
    Random.seed!(rng, 42)  
    
    # Test configuration constants
    in_channels = 3
    out_channels = 3
    hidden_channels = 32
    modes = (16,16)
    n_layers = 4
    positional_embedding = "grid"
    batch_size = 10
    input_dims = (64, 64, in_channels, batch_size)

    @testset "Initialization" begin
        # Test that layer constructs properly
        layer = FourierNeuralOperator(
            in_channels=in_channels, 
            out_channels=out_channels, 
            hidden_channels=hidden_channels, 
            n_layers=n_layers,
            n_modes=modes,
            positional_embedding=positional_embedding)
        
        # Test parameter initialization
        ps, st = Lux.setup(rng, layer)
        
        # Verify parameters exist (shapes are checked in sub-component tests)
        @test haskey(ps, :embedding)
        @test haskey(ps, :lifting)
        @test haskey(ps, :fno_blocks)
        @test haskey(ps, :projection)
       
        # Verify state is properly initialized 
        @test st != nothing
    end
    
    @testset "Forward Pass" begin
        # Setup layer and input data
        layer = FourierNeuralOperator(
            in_channels=in_channels, 
            out_channels=out_channels, 
            hidden_channels=hidden_channels, 
            n_layers=n_layers,
            n_modes=modes,
            positional_embedding=positional_embedding)
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
        layer = FourierNeuralOperator(
            in_channels=in_channels, 
            out_channels=out_channels, 
            hidden_channels=hidden_channels, 
            n_layers=n_layers,
            n_modes=modes,
            positional_embedding=positional_embedding)
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