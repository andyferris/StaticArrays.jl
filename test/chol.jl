using StaticArrays, Test, LinearAlgebra
using LinearAlgebra: PosDefException

@testset "Cholesky decomposition" begin
    for elty in [Float32, Float64, ComplexF64]
        @testset "1×1" begin
            m = @SMatrix [4.0]
            (c,) = cholesky(m).U
            @test c === 2.0
        end

        @testset "2×2" for i = 1:100
            m_a = randn(elty, 2,2)
            #non hermitian
            @test_throws PosDefException cholesky(SMatrix{2,2}(m_a))
            m_a = m_a*m_a'
            m = SMatrix{2,2}(m_a)
            @test cholesky(Hermitian(m)).U ≈ cholesky(m_a).U
            @test cholesky(Hermitian(m)).L ≈ cholesky(m_a).L
        end

        @testset "3×3" for i = 1:100
            m_a = randn(elty, 3,3)
            #non hermitian
            @test_throws PosDefException cholesky(SMatrix{3,3}(m_a))
            m_a = m_a*m_a'
            m = SMatrix{3,3}(m_a)
            @test cholesky(m).U ≈ cholesky(m_a).U
            @test cholesky(m).L ≈ cholesky(m_a).L
            @test cholesky(Hermitian(m)).U ≈ cholesky(m_a).U
            @test cholesky(Hermitian(m)).L ≈ cholesky(m_a).L
        end
        @testset "4×4" for i = 1:100
            m_a = randn(elty, 4,4)
            #non hermitian
            @test_throws PosDefException cholesky(SMatrix{4,4}(m_a))
            m_a = m_a*m_a'
            m = SMatrix{4,4}(m_a)
            @test cholesky(m).L ≈ cholesky(m_a).L
            @test cholesky(m).U ≈ cholesky(m_a).U
            @test cholesky(Hermitian(m)).L ≈ cholesky(m_a).L
            @test cholesky(Hermitian(m)).U ≈ cholesky(m_a).U
        end

        @testset "large (25x25)" begin
            m_a = randn(elty, 25, 25)
            m_a = m_a*m_a'
            m = SMatrix{25,25}(m_a)
            @test cholesky(m).L ≈ cholesky(m_a).L
        end

        @testset "Inverse" begin
            m_a = randn(elty, 3, 3)
            m_a = m_a*m_a'
            m = SMatrix{3,3}(m_a)
            c = cholesky(m)
            @test (@inferred inv(c)) isa SMatrix{3,3,elty}
            @test inv(c) ≈ SMatrix{3,3}(inv(m_a))
        end

        @testset "Division" begin
            m_a = randn(elty, 3, 3)
            m_a = m_a*m_a'
            m = SMatrix{3,3}(m_a)
            c = cholesky(m)
            c_a = cholesky(m_a)
            
            d_a = randn(elty, 3, 3)
            d = SMatrix{3,3}(d_a)

            @test (@inferred c \ d) isa SMatrix{3,3,elty}
            @test c \ d ≈ c_a \ d_a
            @test (@inferred c \ Symmetric(d)) isa SMatrix{3,3,elty}
            @test c \ Symmetric(d) ≈ c_a \ Symmetric(d_a)

            if VERSION >= v"1.3"
                # on earlier versions of Julia d_a / c_a fails
                @test (@inferred d / c) isa SMatrix{3,3,elty}
                @test d / c ≈ d_a / c_a
                @test (@inferred Symmetric(d) / c) isa SMatrix{3,3,elty}
                @test Symmetric(d) / c ≈ Symmetric(d_a) / c_a
            end

            v_a = randn(elty, 3)
            v = SVector{3}(v_a)
            @test (@inferred c \ v) isa SVector{3,elty}
            @test c \ v ≈ c_a \ v_a
        end

        @testset "indefinite matrices" for i = 1:10
            m_a = randn(elty, 5,5)
            @test cholesky(SMatrix{5,5}(m_a), check=false).info == -1
            @test !isposdef(SMatrix{5,5}(m_a))
            m_a = m_a*m_a'

            # Make the matrix indefinite
            D,U = eigen(m_a)
            D[1] = -10*eps(real(elty))
            m_a = Hermitian(U*Diagonal(D)*U')
            m = SMatrix{5,5}(m_a)

            @test_throws PosDefException cholesky(m_a)
            @test_throws PosDefException cholesky(m)
            @test_throws PosDefException cholesky(Hermitian(m))
            @test cholesky(m_a, check=false).info != 0 
            @test cholesky(m, check=false).info != 0 
            @test cholesky(Hermitian(m), check=false).info != 0 
            @test !isposdef(m_a)
            @test !isposdef(m)
        end
    end

    @testset "static blockmatrix" for i = 1:10
        m_a = randn(3,3)
        m_a = m_a*m_a'
        m = SMatrix{3,3}(m_a)
        @test_broken cholesky(reshape([m, 0m, 0m, m], 2, 2)) ==
            reshape([chol(m), 0m, 0m, chol(m)], 2, 2)
    end

end
