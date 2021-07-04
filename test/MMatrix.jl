@testset "MMatrix" begin
    @testset "Inner Constructors" begin
        @test MMatrix{1,1,Int,1}((1,)).data === (1,)
        @test MMatrix{1,1,Float64,1}((1,)).data === (1.0,)
        @test MMatrix{2,2,Float64,4}((1, 1.0, 1, 1)).data === (1.0, 1.0, 1.0, 1.0)
        @test isa(MMatrix{1,1,Int,1}(undef), MMatrix{1,1,Int,1})
        @test isa(MMatrix{1,1,Int}(undef), MMatrix{1,1,Int,1})

        # Bad input
        @test_throws Exception MMatrix{2,1,Int,2}((1,))
        @test_throws Exception MMatrix{1,1,Int,1}(())

        # Bad parameters
        @test_throws Exception MMatrix{1,1,Int,2}((1,))
        @test_throws Exception MMatrix{1,1,1,1}((1,))
        @test_throws ArgumentError MMatrix{1,2,Int,1}((1,))
        @test_throws ArgumentError MMatrix{2,1,Int,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        @test MMatrix{1,1,Int}((1,)).data === (1,)
        @test MMatrix{1,1}((1,)).data === (1,)
        @test MMatrix{1}((1,)).data === (1,)

        @test MMatrix{2,2,Int}((1,2,3,4)).data === (1,2,3,4)
        @test MMatrix{2,2}((1,2,3,4)).data === (1,2,3,4)
        @test MMatrix{2}((1,2,3,4)).data === (1,2,3,4)

        # test for #557-like issues
        @test (@inferred MMatrix(SMatrix{0,0,Float64}()))::MMatrix{0,0,Float64} == MMatrix{0,0,Float64}()

        @test ((@MMatrix [1.0])::MMatrix{1,1}).data === (1.0,)
        @test ((@MMatrix [1 2])::MMatrix{1,2}).data === (1, 2)
        @test ((@MMatrix [1 ; 2])::MMatrix{2,1}).data === (1, 2)
        @test ((@MMatrix [1 2 ; 3 4])::MMatrix{2,2}).data === (1, 3, 2, 4)

        @test ((@MMatrix Int[1.0])::MMatrix{1,1}).data === (1,)
        @test ((@MMatrix Float64[1 2])::MMatrix{1,2}).data === (1.0, 2.0)
        @test ((@MMatrix Float64[1 ; 2])::MMatrix{2,1}).data === (1.0, 2.0)
        @test ((@MMatrix Float64[1 2 ; 3 4])::MMatrix{2,2}).data === (1.0, 3.0, 2.0, 4.0)

        @test ((@MMatrix [i*j for i = 1:2, j=2:3])::MMatrix{2,2}).data === (2, 4, 3, 6)
        @test ((@MMatrix Float64[i*j for i = 1:2, j=2:3])::MMatrix{2,2}).data === (2.0, 4.0, 3.0, 6.0)

        test_expand_error(:(@MMatrix [1 2; 3]))
        test_expand_error(:(@MMatrix Float32[1 2; 3]))
        test_expand_error(:(@MMatrix [i*j*k for i = 1:2, j=2:3, k=3:4]))
        test_expand_error(:(@MMatrix Float32[i*j*k for i = 1:2, j=2:3, k=3:4]))
        test_expand_error(:(@MMatrix fill(2.3, 4, 5, 6)))
        test_expand_error(:(@MMatrix ones(4, 5, 6, 7)))
        test_expand_error(:(@MMatrix ones))
        test_expand_error(:(@MMatrix sin(1:5)))
        test_expand_error(:(@MMatrix [1; 2; 3; 4]...))

        @test ((@MMatrix zeros(2,2))::MMatrix{2, 2, Float64}).data === (0.0, 0.0, 0.0, 0.0)
        @test ((@MMatrix fill(3.4, 2,2))::MMatrix{2, 2, Float64}).data === (3.4, 3.4, 3.4, 3.4)
        @test ((@MMatrix ones(2,2))::MMatrix{2, 2, Float64}).data === (1.0, 1.0, 1.0, 1.0)
        @test isa(@MMatrix(rand(2,2)), MMatrix{2, 2, Float64})
        @test isa(@MMatrix(randn(2,2)), MMatrix{2, 2, Float64})
        @test isa(@MMatrix(randexp(2,2)), MMatrix{2, 2, Float64})

        @test ((@MMatrix zeros(Float32, 2, 2))::MMatrix{2,2,Float32}).data === (0.0f0, 0.0f0, 0.0f0, 0.0f0)
        @test ((@MMatrix ones(Float32, 2, 2))::MMatrix{2,2,Float32}).data === (1.0f0, 1.0f0, 1.0f0, 1.0f0)
        @test isa(@MMatrix(rand(Float32, 2, 2)), MMatrix{2, 2, Float32})
        @test isa(@MMatrix(randn(Float32, 2, 2)), MMatrix{2, 2, Float32})
        @test isa(@MMatrix(randexp(Float32, 2, 2)), MMatrix{2, 2, Float32})

        @test MMatrix(SMatrix{1,1,Int,1}((1,))).data == (1,)
        @test_throws DimensionMismatch MMatrix{3}((1,2,3,4))
    end

    @testset "Methods" begin
        m = @MMatrix [11 13; 12 14]

        @test isimmutable(m) == false

        @test m[1] === 11
        @test m[2] === 12
        @test m[3] === 13
        @test m[4] === 14

        @testinf Tuple(m) === (11, 12, 13, 14)

        @test size(m) === (2, 2)
        @test size(typeof(m)) === (2, 2)
        @test size(MMatrix{2,2}) === (2, 2)

        @test size(m, 1) === 2
        @test size(m, 2) === 2
        @test size(typeof(m), 1) === 2
        @test size(typeof(m), 2) === 2

        @test length(m) === 4

        @test reverse(m) == reverse(reverse(collect(m), dims = 2), dims = 1)
    end
end
