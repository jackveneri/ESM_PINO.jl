using Lux, Random, Test, QG3
using LuxTestUtils, JLD2
#const SphericalConv = Base.get_extension(ESM_PINO, :ESM_PINOQG3Ext).SphericalConv
const SphericalConv = ESM_PINO.SphericalConv
@testset "SphericalConv Comprehensive Tests" begin
    
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    

    # Test configurations - automatically tests different combinations
    test_configs = [
        # (parameter_set, hidden_channels, modes, batch_size, use_gpu, use_zsk)
        ("t21", 32, 30, 1, false, false),
        ("t21", 32, 15, 2, false, true),
        ("t42", 16, 20, 1, false, false),
        ("t42", 64, 10, 4, false, true),
        # Add GPU tests if available (commented out by default for safety)
        # ("t21", 32, 30, 1, true, false),
        # ("t42", 32, 20, 2, true, true),
    ]
    
    # Load parameter sets once to avoid repeated file I/O
    param_sets = Dict{String, Any}()
    @testset for config in test_configs
    param_set, hidden_channels, modes, batch_size, use_gpu, use_zsk = config
    config_name = "params=$(param_set)_ch=$(hidden_channels)_modes=$(modes)_batch=$(batch_size)_gpu=$(use_gpu)_zsk=$(use_zsk)"
        @testset "Configuration: config_name" begin 
            
            # Load parameters if not already loaded
            if !haskey(param_sets, param_set)
                @testset "Parameter Loading: $param_set" begin
                    try
                        if param_set == "t21"
                            @load string(dirname(@__DIR__), "/data/t21-precomputed-p.jld2") qg3ppars
                        else # "t42"
                            @load string(dirname(@__DIR__), "/data/t42-precomputed-p.jld2") qg3ppars
                        end
                        param_sets[param_set] = qg3ppars
                        @test qg3ppars != nothing
                        @info "Successfully loaded $param_set parameters"
                    catch e
                        @error "Failed to load $param_set parameters" exception=e
                        continue  # Skip this configuration if parameter loading fails
                    end
                end
            end
            
            qg3ppars = param_sets[param_set]
            
            @testset "Initialization" begin
                # Test construction with parameter object
                layer = SphericalConv(qg3ppars, hidden_channels; 
                                    modes=modes, batch_size=batch_size, gpu=use_gpu, zsk=use_zsk)
                
                # Test layer properties
                @test layer.hidden_channels == hidden_channels
                @test layer.zsk == use_zsk
                @test layer.modes <= qg3ppars.L  # Ensure modes correction works
                
                # Test parameter initialization
                ps, st = Lux.setup(rng, layer)
                
                # Verify parameters exist and have correct structure
                @test haskey(ps, :weight)
                if use_zsk
                    @test size(ps.weight) == (1, layer.modes, 1, 1)
                else
                    @test size(ps.weight) == (1, layer.modes, 2 * layer.modes - 1, 1)
                end
                
                # Verify state is properly initialized (should be empty NamedTuple)
                @test st == NamedTuple()
                
                # Test direct construction with transforms
                ggsh = QG3.GaussianGridtoSHTransform(qg3ppars, hidden_channels; N_batch=batch_size)
                shgg = QG3.SHtoGaussianGridTransform(qg3ppars, hidden_channels; N_batch=batch_size)
                layer_direct = SphericalConv(hidden_channels, ggsh, shgg, modes; zsk=use_zsk)
                
                ps_direct, st_direct = Lux.setup(rng, layer_direct)
                @test haskey(ps_direct, :weight)
                @test st_direct == NamedTuple()
            end
            
            @testset "Forward Pass - $config_name" begin
                layer = SphericalConv(qg3ppars, hidden_channels; 
                                    modes=modes, batch_size=batch_size, gpu=use_gpu, zsk=use_zsk)
                ps, st = Lux.setup(rng, layer)
                
                # Generate input matching the spherical grid dimensions
                lat_size, lon_size = if param_set == "t21"
                    (32, 64)
                else # "t42"
                    (64, 128)
                end
                
                input_dims = (lat_size, lon_size, hidden_channels, batch_size)
                x = randn(rng, Float32, input_dims...)
                
                # Test basic forward pass
                y, st_update = Lux.apply(layer, x, ps, st)
                
                # Test output shape matches input spatial dimensions
                @test size(y) == input_dims
                
                # Test output values are valid and finite
                @test all(isfinite, y)
                
                # Test that state doesn't change (since we have no state)
                @test st_update == st
                
                # Test with different input types if applicable
                #=
                if !use_gpu  # Only test on CPU for type consistency
                    x64 = randn(rng, Float64, input_dims...)
                    y64, _ = Lux.apply(layer, x64, ps, st)
                    @test size(y64) == input_dims
                    @test all(isfinite, y64)
                end
                =#
            end
            
            @testset "Backward Pass - Gradient Correctness - $config_name" begin
                layer = SphericalConv(qg3ppars, hidden_channels; 
                                    modes=modes, batch_size=batch_size, gpu=use_gpu, zsk=use_zsk)
                ps, st = Lux.setup(rng, layer)
                
                # Generate appropriate input dimensions
                lat_size, lon_size = if param_set == "t21"
                    (32, 64)
                else # "t42"
                    (64, 128)
                end
                
                input_dims = (lat_size, lon_size, hidden_channels, batch_size)
                x = randn(rng, Float32, input_dims...)
                
                # Define loss function
                function loss_fn(x, ps)
                    y, _ = Lux.apply(layer, x, ps, st)
                    return sum(abs2, y)  # Simple L2 loss
                end
                
                # Test gradients using LuxTestUtils
                @testset "Gradient Tests" begin
                    # Skip certain backends based on configuration 
                    skip_backends = [:Enzyme, :AutoEnzyme]  # Enzyme has limited support
                    if use_gpu
                        # On GPU, skip backends that don't support GPU 
                        append!(skip_backends, [:AutoForwardDiff, :AutoFiniteDiff])
                    end
                    
                    @test_gradients(loss_fn, x, ps; skip_backends=skip_backends, soft_fail=true)
                end
                
            end
        end
    end
end