using StaticArrays, Test

@testset "accumulate" begin
    @testset "cumsum(::$label)" for (label, T) in [
        # label, T
        ("SVector", SVector),
        ("MVector", MVector),
        ("SizedVector", SizedVector{3}),
    ]
        a = T(SA[1, 2, 3])
        @test cumsum(a) == cumsum(collect(a))
        @test cumsum(a) isa similar_type(a)
        @inferred cumsum(a)
    end

    @testset "cumsum(::$label; dims=2)" for (label, T) in [
        # label, T
        ("SMatrix", SMatrix),
        ("MMatrix", MMatrix),
        ("SizedMatrix", SizedMatrix{3,2}),
    ]
        a = T(SA[1 2; 3 4; 5 6])
        @test cumsum(a; dims = 2) == cumsum(collect(a); dims = 2)
        @test cumsum(a; dims = 2) isa similar_type(a)
        @inferred cumsum(a; dims = Val(2))
    end

    @testset "cumsum(a::SArray; dims=$i); ndims(a) = $d" for d in 1:4, i in 1:d
        shape = Tuple(1:d)
        a = similar_type(SArray, Int, Size(shape))(1:prod(shape))
        @test cumsum(a; dims = i) == cumsum(collect(a); dims = i)
        @test cumsum(a; dims = i) isa SArray
        @inferred cumsum(a; dims = Val(i))
    end

    @testset "cumprod" begin
        a = SA[1, 2, 3]
        @test cumprod(a)::SArray == cumprod(collect(a))
        @inferred cumprod(a)
    end
end
