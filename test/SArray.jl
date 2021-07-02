@testset "SArray" begin
    @testset "Inner Constructors" begin
        @test SArray{Tuple{1},Int,1,1}((1,)).data === (1,)
        @test SArray{Tuple{1},Float64,1,1}((1,)).data === (1.0,)
        @test SArray{Tuple{2,2},Float64,2,4}((1, 1.0, 1, 1)).data === (1.0, 1.0, 1.0, 1.0)

        # Bad input
        @test_throws Exception SArray{Tuple{1},Int,1,1}()
        @test_throws Exception SArray{Tuple{2},Int,1,2}((1,))
        @test_throws Exception SArray{Tuple{1},Int,1,1}(())

        # Bad parameters
        @test_throws Exception SArray{Tuple{1},Int,1,2}((1,))
        @test_throws Exception SArray{Tuple{1},Int,2,1}((1,))
        @test_throws Exception SArray{Tuple{1},1,1,1}((1,))
        @test_throws Exception SArray{Tuple{2},Int,1,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        @test SArray{Tuple{1},Int,1}((1,)).data === (1,)
        @test SArray{Tuple{1},Int}((1,)).data === (1,)
        @test SArray{Tuple{1}}((1,)).data === (1,)

        @test SArray{Tuple{2,2},Int,2}((1,2,3,4)).data === (1,2,3,4)
        @test SArray{Tuple{2,2},Int}((1,2,3,4)).data === (1,2,3,4)
        @test SArray{Tuple{2,2}}((1,2,3,4)).data === (1,2,3,4)

        @test SArray(SArray{Tuple{2}}(1,2)) === SArray{Tuple{2}}(1,2)

        @test (@inferred SArray(MMatrix{0,0,Float64}()))::SMatrix{0,0,Float64} == SMatrix{0,0,Float64}()

        @test SArray{Tuple{}}(i for i in 1:1).data === (1,)
        @test SArray{Tuple{3}}(i for i in 1:3).data === (1,2,3)
        @test SArray{Tuple{3}}(float(i) for i in 1:3).data === (1.0,2.0,3.0)
        @test SArray{Tuple{2,3}}(i+10j for i in 1:2, j in 1:3).data === (11,12,21,22,31,32)
        @test SArray{Tuple{1,2,3}}(i+10j+100k for i in 1:1, j in 1:2, k in 1:3).data === (111,121,211,221,311,321)
        @test_throws Exception SArray{Tuple{}}(i for i in 1:0)
        @test_throws Exception SArray{Tuple{}}(i for i in 1:2)
        @test_throws Exception SArray{Tuple{3}}(i for i in 1:2)
        @test_throws Exception SArray{Tuple{3}}(i for i in 1:4)
        @test_throws Exception SArray{Tuple{2,3}}(10i+j for i in 1:1, j in 1:3)
        @test_throws Exception SArray{Tuple{2,3}}(10i+j for i in 1:3, j in 1:3)

        @test StaticArrays.sacollect(SVector{6}, Iterators.product(1:2, 1:3)) ==
            SVector{6}(collect(Iterators.product(1:2, 1:3)))
        @test StaticArrays.sacollect(SVector{2}, Iterators.zip(1:2, 2:3)) ==
            SVector{2}(collect(Iterators.zip(1:2, 2:3)))
        @test StaticArrays.sacollect(SVector{3}, Iterators.take(1:10, 3)) ==
            SVector{3}(collect(Iterators.take(1:10, 3)))
        @test StaticArrays.sacollect(SMatrix{2,3}, Iterators.product(1:2, 1:3)) ==
            SMatrix{2,3}(collect(Iterators.product(1:2, 1:3)))
        @test StaticArrays.sacollect(SArray{Tuple{2,3,4}}, 1:24) ==
            SArray{Tuple{2,3,4}}(collect(1:24))

        @test ((@SArray [1])::SArray{Tuple{1}}).data === (1,)
        @test ((@SArray [1,2])::SArray{Tuple{2}}).data === (1,2)
        @test ((@SArray Float64[1,2,3])::SArray{Tuple{3}}).data === (1.0, 2.0, 3.0)
        @test ((@SArray [1 2])::SArray{Tuple{1,2}}).data === (1, 2)
        @test ((@SArray Float64[1 2])::SArray{Tuple{1,2}}).data === (1.0, 2.0)
        @test ((@SArray [1 ; 2])::SArray{Tuple{2,1}}).data === (1, 2)
        @test ((@SArray Float64[1 ; 2])::SArray{Tuple{2,1}}).data === (1.0, 2.0)
        @test ((@SArray [1 2 ; 3 4])::SArray{Tuple{2,2}}).data === (1, 3, 2, 4)
        @test ((@SArray Float64[1 2 ; 3 4])::SArray{Tuple{2,2}}).data === (1.0, 3.0, 2.0, 4.0)

        @test ((@SArray [i for i = 1:2])::SArray{Tuple{2}}).data === (1, 2)
        @test ((@SArray [i*j for i = 1:2, j = 2:3])::SArray{Tuple{2,2}}).data === (2, 4, 3, 6)
        @test ((@SArray [i*j*k for i = 1:2, j = 2:3, k = 3:4])::SArray{Tuple{2,2,2}}).data === (6, 12, 9, 18, 8, 16, 12, 24)
        @test ((@SArray [i*j*k*l for i = 1:2, j = 2:3, k = 3:4, l = 1:2])::SArray{Tuple{2,2,2,2}}).data === (6, 12, 9, 18, 8, 16, 12, 24, 12, 24, 18, 36, 16, 32, 24, 48)
        @test ((@SArray [i*j*k*l*m for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2])::SArray{Tuple{2,2,2,2,2}}).data === (6, 12, 9, 18, 8, 16, 12, 24, 12, 24, 18, 36, 16, 32, 24, 48, 2*6, 2*12, 2*9, 2*18, 2*8, 2*16, 2*12, 2*24, 2*12, 2*24, 2*18, 2*36, 2*16, 2*32, 2*24, 2*48)
        @test ((@SArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2])::SArray{Tuple{2,2,2,2,2,2}}).data === ntuple(i->1, 64)
        @test ((@SArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2])::SArray{Tuple{2,2,2,2,2,2,2}}).data === ntuple(i->1, 128)
        @test ((@SArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2])::SArray{Tuple{2,2,2,2,2,2,2,2}}).data === ntuple(i->1, 256)
        test_expand_error(:(@SArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2, q = 1:2]))
        @test ((@SArray Float64[i for i = 1:2])::SArray{Tuple{2}}).data === (1.0, 2.0)
        @test ((@SArray Float64[i*j for i = 1:2, j = 2:3])::SArray{Tuple{2,2}}).data === (2.0, 4.0, 3.0, 6.0)
        @test ((@SArray Float64[i*j*k for i = 1:2, j = 2:3, k =3:4])::SArray{Tuple{2,2,2}}).data === (6.0, 12.0, 9.0, 18.0, 8.0, 16.0, 12.0, 24.0)
        @test ((@SArray Float64[i*j*k*l for i = 1:2, j = 2:3, k = 3:4, l = 1:2])::SArray{Tuple{2,2,2,2}}).data === (6.0, 12.0, 9.0, 18.0, 8.0, 16.0, 12.0, 24.0, 12.0, 24.0, 18.0, 36.0, 16.0, 32.0, 24.0, 48.0)
        @test ((@SArray Float64[i*j*k*l*m for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2])::SArray{Tuple{2,2,2,2,2}}).data === (6.0, 12.0, 9.0, 18.0, 8.0, 16.0, 12.0, 24.0, 12.0, 24.0, 18.0, 36.0, 16.0, 32.0, 24.0, 48.0, 2*6.0, 2*12.0, 2*9.0, 2*18.0, 2*8.0, 2*16.0, 2*12.0, 2*24.0, 2*12.0, 2*24.0, 2*18.0, 2*36.0, 2*16.0, 2*32.0, 2*24.0, 2*48.0)
        @test ((@SArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2])::SArray{Tuple{2,2,2,2,2,2}}).data === ntuple(i->1.0, 64)
        @test ((@SArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2])::SArray{Tuple{2,2,2,2,2,2,2}}).data === ntuple(i->1.0, 128)
        @test ((@SArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2])::SArray{Tuple{2,2,2,2,2,2,2,2}}).data === ntuple(i->1.0, 256)
        test_expand_error(:(@SArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2, q = 1:2]))

        test_expand_error(:(@SArray [1 2; 3]))
        test_expand_error(:(@SArray Float64[1 2; 3]))
        test_expand_error(:(@SArray ones))
        test_expand_error(:(@SArray fill))
        test_expand_error(:(@SArray ones()))
        test_expand_error(:(@SArray sin(1:5)))
        test_expand_error(:(@SArray fill()))
        test_expand_error(:(@SArray fill(1)))
        test_expand_error(:(@SArray [1; 2; 3; 4]...))

        @test ((@SArray fill(3.,2,2,1))::SArray{Tuple{2,2,1}, Float64}).data === (3.0, 3.0, 3.0, 3.0)
        @test ((@SArray zeros(2,2,1))::SArray{Tuple{2,2,1}, Float64}).data === (0.0, 0.0, 0.0, 0.0)
        @test ((@SArray ones(2,2,1))::SArray{Tuple{2,2,1}, Float64}).data === (1.0, 1.0, 1.0, 1.0)
        @test isa(@SArray(rand(2,2,1)), SArray{Tuple{2,2,1}, Float64})
        @test isa(@SArray(randn(2,2,1)), SArray{Tuple{2,2,1}, Float64})
        @test isa(@SArray(randexp(2,2,1)), SArray{Tuple{2,2,1}, Float64})

        @test ((@SArray zeros(Float32, 2, 2, 1))::SArray{Tuple{2,2,1},Float32}).data === (0.0f0, 0.0f0, 0.0f0, 0.0f0)
        @test ((@SArray ones(Float32, 2, 2, 1))::SArray{Tuple{2,2,1},Float32}).data === (1.0f0, 1.0f0, 1.0f0, 1.0f0)
        @test isa(@SArray(rand(Float32, 2, 2, 1)), SArray{Tuple{2,2,1}, Float32})
        @test isa(@SArray(randn(Float32, 2, 2, 1)), SArray{Tuple{2,2,1}, Float32})
        @test isa(@SArray(randexp(Float32, 2, 2, 1)), SArray{Tuple{2,2,1}, Float32})

        m = [1 2; 3 4]
        @test SArray{Tuple{2,2}}(m) === @SArray [1 2; 3 4]

        # Non-square comprehensions built from SVectors - see #76
        @test @SArray([1 for x = SVector(1,2), y = SVector(1,2,3)]) == ones(2,3)
    end

    @testset "Methods" begin
        m = @SArray [11 13; 12 14]

        @test isimmutable(m) == true

        @test m[1] === 11
        @test m[2] === 12
        @test m[3] === 13
        @test m[4] === 14

        @testinf Tuple(m) === (11, 12, 13, 14)

        @test size(m) === (2, 2)
        @test size(typeof(m)) === (2, 2)
        @test size(SArray{Tuple{2,2},Int,2}) === (2, 2)
        @test size(SArray{Tuple{2,2},Int}) === (2, 2)
        @test size(SArray{Tuple{2,2}}) === (2, 2)

        @test size(m, 1) === 2
        @test size(m, 2) === 2
        @test size(typeof(m), 1) === 2
        @test size(typeof(m), 2) === 2

        @test length(m) === 4

        @test_throws Exception m[1] = 1

        @test Base.dataids(m) === ()

        @test (@inferred view(m, :, :)) === m
        @test (@inferred view(m, :, 1)) === @SArray [11, 12]
        @test (@inferred view(m, SVector{2,Int}(1,2), 1)) === @SArray [11, 12]
        @test (@inferred view(m, SMatrix{2,2,Int}(1,2,3,4))) === m
        @test (@inferred view(m, SOneTo(2), 1)) === @SArray [11, 12]
        @test (@inferred view(m, 1, 1)) === Scalar(m[1, 1])
        @test (@inferred view(m, CartesianIndex(1, 1))) === Scalar(m[1, 1])
        @test (@inferred view(m, CartesianIndex(1, 1, 1))) === Scalar(m[1, 1])
        @test (@inferred view(m, 1, 1, CartesianIndex(1))) === Scalar(m[1, 1])

        @test reverse(m) == reverse(reverse(collect(m), dims = 2), dims = 1)

        m1 = reshape(m, Val(1))

        m1 = @inferred reshape(m, Val(1))

        @test m1 isa SVector
        @test all(((x, y),) -> isequal(x,y), zip(m, m1))

        m2 = @inferred reshape(m, Val(2))
        @test m2 === m

        m3 = @inferred reshape(m, Val(3))
        @test eltype(m3) == eltype(m)
        @test ndims(m3) == 3
        @test size(m3) == (size(m)..., 1)
        @test all(((x, y),) -> isequal(x,y), zip(m, m3))
    end

    @testset "promotion" begin
        @test @inferred(promote_type(SVector{1,Float64}, SVector{1,BigFloat})) == SVector{1,BigFloat}
        @test @inferred(promote_type(SVector{2,Int}, SVector{2,Float64})) === SVector{2,Float64}
        @test @inferred(promote_type(SMatrix{2,3,Float32,6}, SMatrix{2,3,Complex{Float64},6})) === SMatrix{2,3,Complex{Float64},6}
    end

    @testset "show" begin
        io = IOBuffer()
        S = SVector{2, Float64}(1, 2)
        show(io, S)
        S_str = String(take!(io))
        @test S_str == "SVector{2, Float64}((1.0, 2.0))"

        S = SMatrix{1, 2, Int, 2}(1, 2)
        show(io, S)
        S_str = String(take!(io))
        @test S_str == "SMatrix{1, 2, $Int, 2}((1, 2))"
    end
end
