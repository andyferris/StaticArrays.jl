using StaticArrays, Test, LinearAlgebra

mul_wrappers = [
    m -> m,
    m -> Symmetric(m, :U),
    m -> Symmetric(m, :L),
    m -> Hermitian(m, :U),
    m -> Hermitian(m, :L),
    m -> UpperTriangular(m),
    m -> LowerTriangular(m),
    m -> UnitUpperTriangular(m),
    m -> UnitLowerTriangular(m),
    m -> Adjoint(m),
    m -> Transpose(m),
    m -> Diagonal(m)]

@testset "Matrix multiplication" begin
    @testset "Matrix-vector" begin
        m = @SMatrix [1 2; 3 4]
        v = @SVector [1, 2]
        v_bad = @SVector [1, 2, 3]
        @test m*v === @SVector [5, 11]
        for wrapper in mul_wrappers
            @test (@inferred wrapper(m)*v)::SVector{2} == wrapper(Array(m))*Array(v)
        end
        @test_throws DimensionMismatch m*v_bad
        # More complicated eltype inference
        v2 = @SVector [CartesianIndex((1,3)), CartesianIndex((3,1))]
        x = @inferred(m*v2)
        @test isa(x, SVector{2,CartesianIndex{2}})
        @test x == @SVector [CartesianIndex((7,5)), CartesianIndex((15,13))]

        # block matrices
        bm = @SMatrix [m m; m m]
        bv = @SVector [v,v]
        @test (bm*bv)::SVector{2,SVector{2,Int}} == @SVector [[10,22],[10,22]]
        for wrapper in mul_wrappers
            # there may be some problems with inferring the result type of symmetric block matrices
            # and for some reason setindex! in Julia LinearAlgebra has == 0 and == 1 tests
            if !any(x->isa(wrapper(bm), x), [UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular])
                @test wrapper(bm)*bv == wrapper(Array(bm))*Array(bv)
            end
        end

        # inner product
        @test @inferred(v'*v) === 5
        @test @inferred(transpose(v)*v) === 5

        # outer product
        @test @inferred(v*v') === @SMatrix [1 2; 2 4]
        @test @inferred(v*transpose(v)) === @SMatrix [1 2; 2 4]

        v3 = [1, 2]
        @test m*v3 == @MVector [5, 11]
        v3_bad = [1, 2, 3]
        @test_throws DimensionMismatch m*v3_bad

        m2 = @MMatrix [1 2; 3 4]
        v4 = @MVector [1, 2]
        @test (m2*v4)::MVector == @MVector [5, 11]

        m3 = @SArray [1 2; 3 4]
        v5 = @SArray [1, 2]
        @test m3*v5 === @SVector [5, 11]

        m4 = @MArray [1 2; 3 4]
        v6 = @MArray [1, 2]
        @test (m4*v6) == @MVector [5, 11]

        m5 = @SMatrix [1.0 2.0; 3.0 4.0]
        v7 = [1.0, 2.0]
        @test (m5*v7)::SVector ≈ @SVector [5.0, 11.0]

        m6 = @SMatrix Float32[1.0 2.0; 3.0 4.0]
        v8 = Float64[1.0, 2.0]
        @test (m6*v8)::SVector{2,Float64} ≈ @SVector [5.0, 11.0]

    end

    @testset "Vector-matrix" begin
        m = @SMatrix [1 2 3 4]
        v = @SVector [1, 2]
        @test @inferred(v*m) === @SMatrix [1 2 3 4; 2 4 6 8]

        # block matrices
        m = @SMatrix [1 2; 3 4]
        bm = @SMatrix [m m; m m]
        bv = @SVector [v,v]

        @test (bv'*bm)'::SVector{2,SVector{2,Int}} == @SVector [[14,20],[14,20]]

        # Outer product
        v2 = SVector(1, 2)
        v3 = SVector(3, 4)
        @test v2 * v3' === @SMatrix [3 4; 6 8]
        @test v2 * transpose(v3) === @SMatrix [3 4; 6 8]

        v4 = SVector(1+0im, 2+0im)
        v5 = SVector(3+0im, 4+0im)
        @test v4 * v5' === @SMatrix [3+0im 4+0im; 6+0im 8+0im]
        @test v4 * transpose(v5) === @SMatrix [3+0im 4+0im; 6+0im 8+0im]
    end

    @testset "Column vector-vector" begin
        cv_array = rand(4,1)
        rv_array = rand(4)
        a_array = cv_array * rv_array'

        cv = SMatrix{4,1}(cv_array)
        rv = SVector{4}(rv_array)
        @test (cv*adjoint(rv))::SMatrix ≈ a_array

        cv = MMatrix{4,1}(cv_array)
        rv = MVector{4}(rv_array)
        @test (cv*adjoint(rv))::MMatrix ≈ a_array

        cv = SMatrix{4,1}(cv_array)
        rv = SVector{4}(rv_array)
        @test (cv*transpose(rv))::SMatrix ≈ a_array

        cv = MMatrix{4,1}(cv_array)
        rv = MVector{4}(rv_array)
        @test (cv*transpose(rv))::MMatrix ≈ a_array

        cv_bad = @SMatrix rand(4,2)
        rv = @SVector rand(4)
        @test_throws DimensionMismatch cv_bad*transpose(rv)
        @test_throws DimensionMismatch cv_bad*adjoint(rv)
        @test_throws DimensionMismatch cv_bad*rv

        cv_bad = @MMatrix rand(4,2)
        rv = @MVector rand(4)

        @test_throws DimensionMismatch cv_bad*transpose(rv)
        @test_throws DimensionMismatch cv_bad*adjoint(rv)
        @test_throws DimensionMismatch cv_bad*rv
    end

    @testset "vector-column vector" begin
        for T1 in [Float64, ComplexF64, SMatrix{2,2,Float64,4}, SMatrix{2,2,ComplexF64,4}],
            T2 in [Float64, ComplexF64, SMatrix{2,2,Float64,4}, SMatrix{2,2,ComplexF64,4}]

            v1 = @SVector randn(T1, 4)
            v2 = @SVector randn(T2, 4)
            if T1 <: Number
                v1a = Array(v1)
            else
                v1a = Array(Array.(v1))
            end
            if T2 <: Number
                v2a = Array(v2)
            else
                v2a = Array(Array.(v2))
            end
            @test all(isapprox.(v1 * adjoint(v2), v1a * adjoint(v2a)))
            @test all(isapprox.(v1 * transpose(v2), v1a * transpose(v2a)))
        end
    end

    @testset "Matrix-matrix" begin
        m = @SMatrix [1 2; 3 4]
        n = @SMatrix [2 3; 4 5]
        nc = @SMatrix [CartesianIndex((2,2)) CartesianIndex((3,3));
                       CartesianIndex((4,4)) CartesianIndex((5,5))]
        @test @inferred(m*n)  === @SMatrix [10 13; 22 29]
        @test m*nc === @SMatrix [CartesianIndex((10,10)) CartesianIndex((13,13));
                                 CartesianIndex((22,22)) CartesianIndex((29,29))]

        @test m'*n === @SMatrix [14 18; 20 26]
        @test m*n' === @SMatrix [8 14; 18 32]
        @test m'*n' === @SMatrix [11 19; 16 28]
        @test transpose(m)*n === @SMatrix [14 18; 20 26]
        @test m*transpose(n) === @SMatrix [8 14; 18 32]
        @test transpose(m)*transpose(n) === @SMatrix [11 19; 16 28]

        @test @inferred(m*n*m)  === @SMatrix [49 72; 109 160]
        @test @inferred(m*n*m*n)  === @SMatrix [386 507; 858 1127]

        # check different sizes because there are multiple implementations for matrices of different sizes
        for (mm, nn) in [
            (m, n),
            (SMatrix{10, 10}(collect(1:100)), SMatrix{10, 10}(collect(1:100))),
            (SMatrix{15, 15}(collect(1:225)), SMatrix{15, 15}(collect(1:225)))
            ]
            for wrapper_m in mul_wrappers, wrapper_n in mul_wrappers
                wm = wrapper_m(mm)
                wn = wrapper_n(nn)
                if length(mm) >= 255 && (!isa(wm, StaticArray) || !isa(wn, StaticArray))
                    continue
                end
                res_structure = StaticArrays.mul_result_structure(wm, wn)
                expected_type = if length(m) >= 100
                    Matrix{Int}
                elseif res_structure == identity || length(m) >= 100
                    typeof(mm)
                elseif res_structure == LowerTriangular
                    LowerTriangular{Int,typeof(mm)}
                elseif res_structure == UpperTriangular
                    UpperTriangular{Int,typeof(mm)}
                elseif res_structure == Diagonal
                    Diagonal{Int,<:SVector}
                else
                    error("Unknown structure: ", res_structure)
                end
                @test (@inferred wm * wn)::expected_type == wrapper_m(Array(mm)) * wrapper_n(Array(nn))
            end
        end

        m = @MMatrix [1 2; 3 4]
        n = @MMatrix [2 3; 4 5]
        @test (m*n) == @MMatrix [10 13; 22 29]

        m = @SArray [1 2; 3 4]
        n = @SArray [2 3; 4 5]
        @test m*n === @SMatrix [10 13; 22 29]

        m = @MArray [1 2; 3 4]
        n = @MArray [2 3; 4 5]
        @test (m*n) == @SMatrix [10 13; 22 29]

        # block matrices
        bm = @SMatrix [m m; m m]
        bm2 = @MMatrix [14 20; 30 44]
        @test (bm*bm)::SMatrix{2,2,MMatrix{2,2,Int,4}} == @SMatrix [bm2 bm2; bm2 bm2]

        # Alternative methods used between 8 < n <= 14 and n > 14
        m_array = rand(1:10, 10, 10)
        n_array = rand(1:10, 10, 10)
        a_array = m_array*n_array

        m = SMatrix{10,10}(m_array)
        n = SMatrix{10,10}(n_array)
        @test m*n === SMatrix{10,10}(a_array)

        m_array2 = rand(1:10, 16, 16) # see JuliaLang/julia#18794 for reason for variable name changes
        n_array2 = rand(1:10, 16, 16)
        a_array2 = m_array2*n_array2

        m2 = SMatrix{16,16}(m_array2)
        n2 = SMatrix{16,16}(n_array2)
        @test m2*n2 === SMatrix{16,16}(a_array2)

        # Non-square version
        m_array3 = rand(1:10, 9, 10)
        n_array3 = rand(1:10, 10, 11)
        a_array3 = m_array3*n_array3
        m3 = SMatrix{9,10}(m_array3)
        n3 = SMatrix{10,11}(n_array3)
        @test m3*n3 === SMatrix{9,11}(a_array3)

        # Mutating types follow different behaviour
        m_array = rand(1:10, 10, 10)
        n_array = rand(1:10, 10, 10)
        a_array = m_array*n_array

        m = MMatrix{10,10}(m_array)
        n = MMatrix{10,10}(n_array)
        @test (m*n)::MMatrix == a_array

        m_array = rand(1:10, 16, 16)
        n_array = rand(1:10, 16, 16)
        a_array = m_array*n_array

        m = MMatrix{16,16}(m_array)
        n = MMatrix{16,16}(n_array)
        @test (m*n)::MMatrix == a_array

        # Mutating BLAS types follow yet different behaviour
        m_array = randn(4, 4)
        n_array = randn(4, 4)
        a_array = m_array*n_array

        m = MMatrix{4,4}(m_array)
        n = MMatrix{4,4}(n_array)
        @test m*n::MMatrix ≈ a_array

        m_array = randn(10, 10)
        n_array = randn(10, 10)
        a_array = m_array*n_array

        m = MMatrix{10,10}(m_array)
        n = MMatrix{10,10}(n_array)
        @test m*n::MMatrix ≈ a_array

        m_array = randn(16, 16)
        n_array = randn(16, 16)
        a_array = m_array*n_array

        m = MMatrix{16,16}(m_array)
        n = MMatrix{16,16}(n_array)
        @test m*n::MMatrix ≈ a_array

        # Complex numbers
        m_array = randn(4, 4)
        n_array = im*randn(4, 4)
        a_array = m_array*n_array

        m = MMatrix{4,4}(m_array)
        n = MMatrix{4,4}(n_array)
        @test m*n::MMatrix ≈ a_array

        m_array = randn(10, 10)
        n_array = im*randn(10, 10)
        a_array = m_array*n_array

        m = MMatrix{10,10}(m_array)
        n = MMatrix{10,10}(n_array)
        @test m*n::MMatrix ≈ a_array

        m_array = randn(16, 16)
        n_array = im*randn(16, 16)
        a_array = m_array*n_array

        m = MMatrix{16,16}(m_array)
        n = MMatrix{16,16}(n_array)
        @test m*n::MMatrix ≈ a_array

        # Bad dimensions
        m_array = randn(16, 16)
        n_array = randn(17, 17)
        m = MMatrix{16,16}(m_array)
        n = MMatrix{17,17}(n_array)
        @test_throws DimensionMismatch m*n
        m_array = randn(10, 10)
        n_array = randn(11, 11)
        m = MMatrix{10,10}(m_array)
        n = MMatrix{11,11}(n_array)
        @test_throws DimensionMismatch m*n
        m_array = randn(4, 4)
        n_array = randn(3, 3)
        m = MMatrix{4,4}(m_array)
        n = MMatrix{3,3}(n_array)
        @test_throws DimensionMismatch m*n
    end

    @testset "mul!" begin
        v = @SVector [2, 4]
        v2 = [2, 4]
        m = @SMatrix [1 2; 3 4]
        n = @SMatrix [2 3; 4 5]

        outvec = MVector{2,Int}(undef)
        mul!(outvec, m, v)
        @test outvec == @MVector [10,22]
        outvec2 = Vector{Int}(undef, 2)
        mul!(outvec2, m, v2)
        @test outvec2 == [10,22]

        # Bad dimensions
        outvec_bad = MVector{3,Int}(undef)
        @test_throws DimensionMismatch mul!(outvec_bad, m, v)

        a = MMatrix{2,2,Int,4}(undef)
        mul!(a, m, n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [10 13; 22 29]

        a = MMatrix{2,2,Int,4}(undef)
        mul!(a, m', n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [14 18; 20 26]
        mul!(a, m, n')
        @test a::MMatrix{2,2,Int,4} == @MMatrix [8 14; 18 32]
        mul!(a, m', n')
        @test a::MMatrix{2,2,Int,4} == @MMatrix [11 19; 16 28]
        mul!(a, transpose(m), n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [14 18; 20 26]
        mul!(a, m, transpose(n))
        @test a::MMatrix{2,2,Int,4} == @MMatrix [8 14; 18 32]
        mul!(a, transpose(m), transpose(n))
        @test a::MMatrix{2,2,Int,4} == @MMatrix [11 19; 16 28]
        for wrapper_m in mul_wrappers, wrapper_n in mul_wrappers
            mul!(a, wrapper_m(m), wrapper_n(n))
            @test a::MMatrix{2,2,Int,4} == wrapper_m(Array(m))*wrapper_n(Array(n))
        end

        a2 = MArray{Tuple{2,2},Int,2,4}(undef)
        mul!(a2, m, n)
        @test a2::MArray{Tuple{2,2},Int,2,4} == @MArray [10 13; 22 29]

        # Alternative builtin method used for n > 8
        m_array_2 = rand(1:10, 10, 10)
        n_array_2 = rand(1:10, 10, 10)
        a_array_2 = m_array_2*n_array_2

        m_2 = MMatrix{10,10}(m_array_2)
        n_2 = MMatrix{10,10}(n_array_2)
        a_2 = MMatrix{10,10,Int}(undef)
        mul!(a_2, m_2, n_2)
        @test a_2 == a_array_2

        # BLAS used for n > 14
        m_array_3 = randn(4, 4)
        n_array_3 = randn(4, 4)
        a_array_3 = m_array_3*n_array_3

        m_3 = MMatrix{4,4}(m_array_3)
        n_3 = MMatrix{4,4}(n_array_3)
        a_3 = MMatrix{4,4,Float64}(undef)
        mul!(a_3, m_3, n_3)
        @test a_3 ≈ a_array_3

        m_array_4 = randn(10, 10)
        n_array_4 = randn(10, 10)
        a_array_4 = m_array_4*n_array_4

        m_4 = MMatrix{10,10}(m_array_4)
        n_4 = MMatrix{10,10}(n_array_4)
        a_4 = MMatrix{10,10,Float64}(undef)
        mul!(a_4, m_4, n_4)
        @test a_4 ≈ a_array_4

        m_array_5 = rand(1:10, 16, 16)
        n_array_5 = rand(1:10, 16, 16)
        a_array_5 = m_array_5*n_array_5

        m_5 = MMatrix{16,16}(m_array_5)
        n_5 = MMatrix{16,16}(n_array_5)
        a_5 = MMatrix{16,16,Int}(undef)
        mul!(a_5, m_5, n_5)
        @test a_5 ≈ a_array_5

        m_array_6 = rand(1:10, 8, 10)
        n_array_6 = rand(1:10, 10, 8)
        a_array_6 = m_array_6*n_array_6

        m_6 = MMatrix{8,10}(m_array_6)
        n_6 = MMatrix{10,8}(n_array_6)
        a_6 = MMatrix{8,8,Int}(undef)
        mul!(a_6, m_6, n_6)
        @test a_6 == a_array_6

        # Float64
        vf = @SVector [2.0, 4.0]
        vf2 = [2.0, 4.0]
        mf = @SMatrix [1.0 2.0; 3.0 4.0]

        outvecf = MVector{2,Float64}(undef)
        mul!(outvecf, mf, vf)
        @test outvecf ≈ @MVector [10.0, 22.0]
        outvec2f = Vector{Float64}(undef, 2)
        mul!(outvec2f, mf, vf2)
        @test outvec2f ≈ [10.0, 22.0]
    end
end
