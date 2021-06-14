using StaticArrays, Test

@testset "Indexing" begin
    @testset "Linear getindex() on SVector" begin
        sv = SVector{4}(4,5,6,7)

        # SVector
        @test (@inferred getindex(sv, SVector(4,3,2,1))) === SVector((7,6,5,4))

        # Colon
        @test (@inferred getindex(sv,:)) === sv

        # SArray
        @test (@inferred getindex(sv, SMatrix{2,2}(1,4,2,3))) === SMatrix{2,2}(4,7,5,6)
    end

    @testset "Linear getindex() on SMatrix" begin
        sv = SVector{4}(4,5,6,7)
        sm = SMatrix{2,2}(4,5,6,7)

        # SVector
        @test (@inferred getindex(sm, SVector(4,3,2,1))) === SVector((7,6,5,4))

        # Colon
        @test (@inferred getindex(sm,:)) === sv

        # SArray
        @test (@inferred getindex(sm, SMatrix{2,2}(1,4,2,3))) === SMatrix{2,2}(4,7,5,6)
    end

    @testset "Linear getindex()/setindex!() on MVector" begin
        vec = @SVector [4,5,6,7]

        # SVector
        mv = MVector{4,Int}(undef)
        @test (mv[SVector(1,2,3,4)] = vec; (@inferred getindex(mv, SVector(4,3,2,1)))::MVector{4,Int} == MVector((7,6,5,4)))
        @test setindex!(mv, vec, SVector(1,2,3,4)) === mv

        mv = MVector{4,Int}(undef)
        @test (mv[SVector(1,2,3,4)] = [4, 5, 6, 7]; (@inferred getindex(mv, SVector(4,3,2,1)))::MVector{4,Int} == MVector((7,6,5,4)))
        @test (mv[SVector(1,2,3,4)] = 2; (@inferred getindex(mv, SVector(4,3,2,1)))::MVector{4,Int} == MVector((2,2,2,2)))

        mv = MVector(0,0,0)
        @test (mv[SVector(1,3)] = [4, 5]; (@inferred mv == MVector(4,0,5)))

        mv = MVector(0,0,0)
        @test (mv[SVector(1,3)] = SVector(4, 5); (@inferred mv == MVector(4,0,5)))

        mv = MVector(0,0,0)
        @test (mv[SMatrix{2,1}(1,3)] = SMatrix{2,1}(4, 5); (@inferred mv == MVector(4,0,5)))

        # Colon
        mv = MVector{4,Int}(undef)
        @test (mv[:] = vec; (@inferred getindex(mv, :))::MVector{4,Int} == MVector((4,5,6,7)))
        @test (mv[:] = [4, 5, 6, 7]; (@inferred getindex(mv, :))::MVector{4,Int} == MVector((4,5,6,7)))
        @test (mv[:] = 2; (@inferred getindex(mv, :))::MVector{4,Int} == MVector((2,2,2,2)))
        @test setindex!(mv, 2, :) === mv

        @test_throws DimensionMismatch setindex!(mv, SVector(1,2,3), SVector(1,2,3,4))
        @test_throws DimensionMismatch setindex!(mv, SVector(1,2,3), :)
        @test_throws DimensionMismatch setindex!(mv, view(ones(8), 1:5), :)
        @test_throws DimensionMismatch setindex!(mv, [1,2,3], SVector(1,2,3,4))
    end

    @testset "Linear getindex()/setindex!() on MMatrix" begin
        vec = @SVector [4,5,6,7]

        # SVector
        mm = MMatrix{2,2,Int}(undef)
        @test (mm[SVector(1,2,3,4)] = vec; (@inferred getindex(mm, SVector(4,3,2,1)))::MVector{4,Int} == MVector((7,6,5,4)))

        # Colon
        mm = MMatrix{2,2,Int}(undef)
        @test (mm[:] = vec; (@inferred getindex(mm, :))::MVector{4,Int} == MVector((4,5,6,7)))

        # SMatrix
        mm = MMatrix{2,2,Int}(undef)
        mi = MMatrix{2,2}(4,2,1,3)
        data = @SMatrix [4 5; 6 7]
        @test (mm[mi] = data; (@inferred getindex(mm, :))::MVector{4,Int} == MVector((5,6,7,4)))
    end

    @testset "Linear getindex()/setindex!() with a SVector on an Array" begin
        v = [11,12,13]

        @test (v[SVector(2,3)] = [22,23]; (v[2] == 22) & (v[3] == 23))
    end

    @testset "Fancy APL indexing" begin
        @test @SVector([1,2,3,4])[@SMatrix([1 2; 3 4])] === @SMatrix([1 2; 3 4])
    end

    @testset "2D getindex() on SVector" begin
        v = @SVector [1,2]
        @test v[1,1] == 1
        @test v[2,1] == 2
        @test_throws BoundsError v[1,2]
        @test_throws BoundsError v[3,1]
    end

    @testset "2D getindex() on SMatrix" begin
        sm = @SMatrix [1 3; 2 4]

        # Scalar
        @test sm[1,1] === 1
        @test sm[2,1] === 2
        @test sm[1,2] === 3
        @test sm[2,2] === 4

        # SVector, scalar
        @test (@inferred getindex(sm, SVector(2,1), SVector(2,1))) === @SMatrix [4 2; 3 1]
        @test (@inferred getindex(sm, 1, SVector(1,2))) === @SVector [1,3]
        @test (@inferred getindex(sm, SVector(1,2), 1)) === @SVector [1,2]

        # Colon
        @test (@inferred getindex(sm, :, :)) === @SMatrix [1 3; 2 4]
        @test (@inferred getindex(sm, SVector(2,1), :)) === @SMatrix [2 4; 1 3]
        @test (@inferred getindex(sm, :, SVector(2,1))) === @SMatrix [3 1; 4 2]
        @test (@inferred getindex(sm, 1, :)) === @SVector [1,3]
        @test (@inferred getindex(sm, :, 1)) === @SVector [1,2]

        # SOneTo
        @testinf sm[SOneTo(1),:] === @SMatrix [1 3]
        @testinf sm[:,SOneTo(1)] === @SMatrix [1;2]
    end

    @testset "2D getindex()/setindex! on MMatrix" begin
        sm = @MMatrix [1 3; 2 4]

        # Tuple, scalar
        @test (mm = MMatrix{2,2,Int}(undef); mm[SVector(2,1),SVector(2,1)] = sm[SVector(2,1),SVector(2,1)]; (@inferred getindex(mm, SVector(2,1), SVector(2,1)))::MMatrix == @MMatrix [4 2; 3 1])
        @test (mm = MMatrix{2,2,Int}(undef); mm[1,SVector(1,2)] = sm[1,SVector(1,2)]; (@inferred getindex(mm, 1, SVector(1,2)))::MVector == @MVector [1,3])
        @test (mm = MMatrix{2,2,Int}(undef); mm[SVector(1,2),1] = sm[SVector(1,2),1]; (@inferred getindex(mm, SVector(1,2), 1))::MVector == @MVector [1,2])

        # Colon
        @test (mm = MMatrix{2,2,Int}(undef); mm[:,:] = sm[:,:]; (@inferred getindex(mm, :, :))::MMatrix == @MMatrix [1 3; 2 4])
        @test (mm = MMatrix{2,2,Int}(undef); mm[SVector(2,1),:] = sm[SVector(2,1),:]; (@inferred getindex(mm, SVector(2,1), :))::MMatrix == @MMatrix [2 4; 1 3])
        @test (mm = MMatrix{2,2,Int}(undef); mm[:,SVector(2,1)] = sm[:,SVector(2,1)]; (@inferred getindex(mm, :, SVector(2,1)))::MMatrix == @MMatrix [3 1; 4 2])
        @test (mm = MMatrix{2,2,Int}(undef); mm[1,:] = sm[1,:]; (@inferred getindex(mm, 1, :))::MVector == @MVector [1,3])
        @test (mm = MMatrix{2,2,Int}(undef); mm[:,1] = sm[:,1]; (@inferred getindex(mm, :, 1))::MVector == @MVector [1,2])

        # SOneTo
        @test (mm = MMatrix{2,2,Int}(undef); mm[SOneTo(1),:] = sm[SOneTo(1),:]; (@inferred getindex(mm, SOneTo(1), :))::MMatrix == @MMatrix [1 3])
        @test (mm = MMatrix{2,2,Int}(undef); mm[:,SOneTo(1)] = sm[:,SOneTo(1)]; (@inferred getindex(mm, :, SOneTo(1)))::MMatrix == @MMatrix [1;2])

        # #866
        @test_throws DimensionMismatch setindex!(MMatrix(SA[1 2; 3 4]), SA[3,4], 1, SA[1,2,3])
        @test_throws DimensionMismatch setindex!(MMatrix(SA[1 2; 3 4]), [3,4], 1, SA[1,2,3])
    end

    @testset "3D scalar indexing" begin
        sa = SArray{Tuple{2,2,2}, Int}([i*j*k for i = 1:2, j = 2:3, k=3:4])

        @test sa[1,1,2] === 8
        @test sa[1,2,1] === 9
        @test sa[2,1,1] === 12

        ma = MArray{Tuple{2,2,2}, Int}(undef)
        @test (ma[1,1,2] = 8; ma[1,1,2] === 8)
        @test (ma[1,2,1] = 9; ma[1,2,1] === 9)
        @test (ma[2,1,1] = 12; ma[2,1,1] === 12)
    end

    @testset "4D scalar indexing" begin
        sa = SArray{Tuple{2,2,2,2}, Int}([i*j*k*l for i = 1:2, j = 2:3, k=3:4, l=4:5])

        @test sa[1,1,1,2] === 30
        @test sa[1,1,2,1] === 32
        @test sa[1,2,1,1] === 36
        @test sa[2,1,1,1] === 48

        ma = MArray{Tuple{2,2,2,2}, Int}(undef)
        @test (ma[1,1,1,2] = 30; ma[1,1,1,2] === 30)
        @test (ma[1,1,2,1] = 32; ma[1,1,2,1] === 32)
        @test (ma[1,2,1,1] = 36; ma[1,2,1,1] === 36)
        @test (ma[2,1,1,1] = 48; ma[2,1,1,1] === 48)
    end

    @testset "4D StaticArray indexing" begin
        sa = SArray{Tuple{2,2,2,2}, Int}([i*j*k*l for i = 1:2, j = 2:3, k=3:4, l=4:5])
        @test (@inferred getindex(sa, 1, 1, 1, SVector(1,2))) === @SVector [24,30]
        @test (@inferred getindex(sa, 1, 1, SVector(1,2), 1)) === @SVector [24,32]
        @test (@inferred getindex(sa, 1, SVector(1,2), 1, 1)) === @SVector [24,36]
        @test (@inferred getindex(sa, SVector(1,2), 1, 1, 1)) === @SVector [24,48]
        a = [i*j*k*l for i = 1:2, j = 2:3, k=3:4, l=4:5]
        @test (@inferred getindex(a, 1, 1, 1, SVector(1,2))) == [24,30]
        @test (@inferred getindex(a, 1, 1, SVector(1,2), 1)) == [24,32]
        @test (@inferred getindex(a, 1, SVector(1,2), 1, 1)) == [24,36]
        @test (@inferred getindex(a, SVector(1,2), 1, 1, 1)) == [24,48]
    end

    @testset "Indexing with empty vectors" begin
        a = [1.0 2.0; 3.0 4.0]
        @test a[SVector{0,Int}()] == SVector{0,Float64}(())
        @test a[SVector{0,Int}(),SVector{0,Int}()] == SMatrix{0,0,Float64,0}(())
        b = copy(a)
        a[SVector{0,Int}()] = 5.0
        @test b == a
    end

    @testset "Indexing empty arrays" begin
        @test size(SVector{0,Float64}()[:]) == (0,)
        @test size(SMatrix{0,0,Float64}()[:,:]) == (0,0)
        @test size(SMatrix{5,0,Float64}()[1,:]) == (0,)
        @test size(SMatrix{5,0,Float64}()[:,:]) == (5,0)
        @test size(SMatrix{0,5,Float64}()[:,1]) == (0,)
        @test size(SMatrix{0,5,Float64}()[:,:]) == (0,5)

        @test (zeros(0)[SVector{0,Int}()] = 0) == 0
        @test (zeros(0,2)[SVector{0,Int}(),SVector(1)] = 0) == 0
        @test (zeros(2,0)[SVector(1),SVector{0,Int}()] = 0) == 0
    end

    @testset "Viewing zero-dimensional arrays" begin
        # issue #705
        A = zeros()
        B = MArray{Tuple{},Float64,0,1}(0.0)
        @test @inferred(view(A))[] == 0.0
        @test @inferred(view(B))[] == 0.0
    end

    @testset "Using SArray as index for view" begin
        a = collect(11:20)
        @test view(a, SVector(1,2,3)) == [11,12,13]
        @test_throws BoundsError view(a, SVector(1,11,3))
        B = rand(Int,3,4,5,6)
        Bv = view(B, 1, (@SVector [2, 1]), [2, 3], (@SVector [4]))
        @test Bv == B[1, [2,1], 2:3, [4]]
        @test axes(Bv, 1) === SOneTo(2)
        @test axes(Bv, 3) === SOneTo(1)
        Bvv = view(Bv, (@SVector [1, 2]), 2, 1)
        @test axes(Bvv) === (SOneTo(2),)
        @test Bvv[1] == B[1, 2, 3, 4]
        Bvv[1] = 100
        @test Bvv[1] == 100
        @test B[1,2,3,4] == 100
        @test eltype(Bvv) == Int
        @test Bvv[:] == [B[1,2,3,4], B[1,1,3,4]]
    end

    @testset "Indexing with constants" begin
        function SVector_UnitRange()
            x = SA[1, 2, 3]
            x[2:end]
        end
        @test SVector_UnitRange() === SA[2, 3]
        VERSION ≥ v"1.2" && @test_const_fold SVector_UnitRange()

        function SVector_StepRange()
            x = SA[1, 2, 3, 4]
            x[1:2:end]
        end
        @test SVector_StepRange() === SA[1, 3]
        VERSION ≥ v"1.2" && @test_const_fold SVector_StepRange()

        function SMatrix_UnitRange_UnitRange()
            x = SA[1 2 3; 4 5 6]
            x[1:2, 2:end]
        end
        @test SMatrix_UnitRange_UnitRange() === SA[2 3; 5 6]
        VERSION ≥ v"1.2" && @test_const_fold SMatrix_UnitRange_UnitRange()

        function SMatrix_StepRange_StepRange()
            x = SA[1 2 3; 4 5 6]
            x[1:1:2, 1:2:end]
        end
        @test SMatrix_StepRange_StepRange() === SA[1 3; 4 6]
        VERSION ≥ v"1.2" && @test_const_fold SMatrix_StepRange_StepRange()
    end
end
