using StaticArrays, Test, LinearAlgebra

@testset "SDiagonal" begin
    @testset "Constructors" begin
        @test SDiagonal{1,Int64}((1,)).diag === SVector{1,Int64}((1,))
        @test SDiagonal{1,Float64}((1,)).diag === SVector{1,Float64}((1,))

        @test SDiagonal{4,Float64}((1, 1.0, 1, 1)).diag.data === (1.0, 1.0, 1.0, 1.0)
        @test SDiagonal{4}((1, 1.0, 1, 1)).diag.data === (1.0, 1.0, 1.0, 1.0)
        @test SDiagonal((1, 1.0, 1, 1)).diag.data === (1.0, 1.0, 1.0, 1.0)

        # Bad input
        @test_throws Exception SMatrix{1,Int}()
        @test_throws Exception SMatrix{2,Int}((1,))

        # From SMatrix
        @test SDiagonal(SMatrix{2,2,Int}((1,2,3,4))).diag.data === (1,4)

        @test SDiagonal{1,Int}(SDiagonal{1,Float64}((1,))).diag[1] === 1

    end

    @testset "Methods" begin

        m = SDiagonal(@SVector [11, 12, 13, 14])

        @test diag(m) === m.diag

        m2 = diagm(0 => [11, 12, 13, 14])

        @test logdet(m) == logdet(m2)
        @test logdet(im*m) ≈ logdet(im*m2)
        @test det(m) == det(m2)
        @test tr(m) == tr(m2)
        @test log(m) == log(m2)
        @test exp(m) == exp(m2)
        @test sqrt(m) == sqrt(m2)
        @test cholesky(m).U == cholesky(m2).U

        # Aparently recursive chol never really worked
        #@test_broken chol(reshape([1.0*m, 0.0*m, 0.0*m, 1.0*m], 2, 2)) ==
        #    reshape([chol(1.0*m), 0.0*m, 0.0*m, chol(1.0*m)], 2, 2)

        @test isimmutable(m) == true

        @test m[1,1] === 11
        @test m[2,2] === 12
        @test m[3,3] === 13
        @test m[4,4] === 14

        for i in 1:4
            for j in 1:4
                i == j || @test m[i,j] === 0
            end
        end

        @test_throws BoundsError m[5,5]

        @test_throws BoundsError m[1,5]


        @test size(m) === (4, 4)
        @test size(typeof(m)) === (4, 4)
        @test size(SDiagonal{4}) === (4, 4)

        @test size(m, 1) === 4
        @test size(m, 2) === 4
        @test size(typeof(m), 1) === 4
        @test size(typeof(m), 2) === 4

        @test length(m) === 4*4

        m2 = SMatrix{4,4}(m)
        @test axes(m) === axes(m2)
        @test axes(m, 1) === axes(m2, 1)
        @test axes(m, 3) == SOneTo(1)

        @test m[:, 1] === SVector{4}(m[1,1], 0, 0, 0)
        @test m[:, :] === m2
        @test m[2, 2, 1] === m[2, 2]

        @test_throws Exception m[1] = 1

        b = @SVector [2,-1,2,1]
        b2 = Vector(b)


        @test m*b ==  @SVector [22,-12,26,14]
        @test (b'*m)' ==  @SVector [22,-12,26,14]
        @test transpose(transpose(b)*m) ==  @SVector [22,-12,26,14]

        @test m\b == m2\b

        @test b'/m == b'/m2
        @test transpose(b)/m == transpose(b)/m2
        # @test_throws Exception b/m # Apparently this is now some kind of minimization problem
        @test m*m == m2*m

        @test ishermitian(m) == ishermitian(m2)
        @test ishermitian(m/2)
        m_ireal = SDiagonal(@SVector [11+0im, 12+0im, 13+0im, 14+0im])
        @test ishermitian(m_ireal)

        @test isposdef(m) == isposdef(m2)
        @test issymmetric(m) == issymmetric(m2)

        @test (2*m/2)' == m
        @test transpose(2*m/2) == m
        @test 2m == m + m
        @test -(-m) == m
        @test m - SMatrix{4,4}(zeros(4,4)) == m
        @test m*0 == m - m

        @test m*inv(m) == m/m == m\m == one(SDiagonal{4,Float64})

        @test factorize(m) == m
        @test m*[1; 1; 1; 1] == [11; 12; 13; 14]
        @test m\[1; 1; 1; 1] == [11; 12; 13; 14].\[1; 1; 1; 1]
        @test SMatrix{4,4}(Matrix{Float64}(I, 4, 4))*m == m
        @test m*SMatrix{4,4}(Matrix{Float64}(I, 4, 4)) == m
        @test SMatrix{4,4}(Matrix{Float64}(I, 4, 4))/m == diagm(0 => [11; 12; 13; 14].\[1; 1; 1; 1])
        @test m\SMatrix{4,4}(Matrix{Float64}(I, 4, 4)) == diagm(0 => [11; 12; 13; 14].\[1; 1; 1; 1])

        @test m + zero(m) == m
        @test m + zero(typeof(m)) == m

        @test copy(m) === m
    end
end
