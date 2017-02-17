@testset "AbstractArray interface" begin
    @testset "size and length" begin
        m = @SMatrix [1 2 3; 4 5 6; 7 8 9; 10 11 12]

        @test length(m) == 12
        @test @compat(Base.IndexStyle(m)) == @compat(Base.IndexLinear())
        @test Base.isassigned(m, 2, 2) == true
    end

    @testset "similar_type" begin
        @test @inferred(similar_type(SVector{3,Int})) == SVector{3,Int}
        @test @inferred(similar_type(@SVector [1,2,3])) == SVector{3,Int}

        @test @inferred(similar_type(SVector{3,Int}, Float64)) == SVector{3,Float64}
        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Size(2))) == SVector{2, Int}
        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Float64, Size(2))) == SVector{2, Float64}
        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Float64, Size(2))) == SVector{2, Float64}

        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Float64)) == SMatrix{3, 3, Float64, 9}
        @test @inferred(similar_type(SVector{2,Int}, Size(3,3))) == SMatrix{3, 3, Int, 9}
        @test @inferred(similar_type(SVector{2,Int}, Float64, Size(3,3))) == SMatrix{3, 3, Float64, 9}

        @test @inferred(similar_type(SArray{(4,4,4),Int,3,64}, Float64)) == SArray{(4,4,4), Float64, 3, 64}
        @test @inferred(similar_type(SVector{2,Int}, Size(3,3,3))) == SArray{(3,3,3), Int, 3, 27}
        @test @inferred(similar_type(SVector{2,Int}, Float64, Size(3,3,3))) == SArray{(3,3,3), Float64, 3, 27}

        # Some specializations for the mutable case
        @test @inferred(similar_type(MVector{3,Int}, Float64)) == MVector{3,Float64}
        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Size(2))) == MVector{2, Int}
        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Float64, Size(2))) == MVector{2, Float64}
        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Float64, Size(2))) == MVector{2, Float64}

        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Float64)) == MMatrix{3, 3, Float64, 9}
        @test @inferred(similar_type(MVector{2,Int}, Size(3,3))) == MMatrix{3, 3, Int, 9}
        @test @inferred(similar_type(MVector{2,Int}, Float64, Size(3,3))) == MMatrix{3, 3, Float64, 9}

        @test @inferred(similar_type(MArray{(4,4,4),Int,3,64}, Float64)) == MArray{(4,4,4), Float64, 3, 64}
        @test @inferred(similar_type(MVector{2,Int}, Size(3,3,3))) == MArray{(3,3,3), Int, 3, 27}
        @test @inferred(similar_type(MVector{2,Int}, Float64, Size(3,3,3))) == MArray{(3,3,3), Float64, 3, 27}
    end

    @testset "similar" begin
        sv = @SVector [1,2,3]
        sm = @SMatrix [1 2; 3 4]
        sa = SArray{(1,1,1),Int,3,1}((1,))

        @test isa(@inferred(similar(sv)), MVector{3,Int})
        @test isa(@inferred(similar(sv, Float64)), MVector{3,Float64})
        @test isa(@inferred(similar(sv, Size(4))), MVector{4,Int})
        @test isa(@inferred(similar(sv, Float64, Size(4))), MVector{4,Float64})

        @test isa(@inferred(similar(sm)), MMatrix{2,2,Int,4})
        @test isa(@inferred(similar(sm, Float64)), MMatrix{2,2,Float64,4})
        @test isa(@inferred(similar(sv, Size(3,3))), MMatrix{3,3,Int,9})
        @test isa(@inferred(similar(sv, Float64, Size(3,3))), MMatrix{3,3,Float64,9})

        @test isa(@inferred(similar(sa)), MArray{(1,1,1),Int,3,1})
        @test isa(@inferred(similar(sa, Float64)), MArray{(1,1,1),Float64,3,1})
        @test isa(@inferred(similar(sv, Size(3,3,3))), MArray{(3,3,3),Int,3,27})
        @test isa(@inferred(similar(sv, Float64, Size(3,3,3))), MArray{(3,3,3),Float64,3,27})
    end

    @testset "reshape" begin
        @test @inferred(reshape(SVector(1,2,3,4), Size(2,2))) === SMatrix{2,2}(1,2,3,4)
        @test @inferred(reshape([1,2,3,4], Size(2,2)))::SizedArray{(2,2),Int,2,1} == [1 3; 2 4]
    end
end
