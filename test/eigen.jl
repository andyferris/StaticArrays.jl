using StaticArrays, Test, LinearAlgebra

@testset "Eigenvalue decomposition" begin
    @testset "1×1" begin
        m = @SMatrix [2.0]
        (vals, vecs) = eigen(m)
        @test vals === SVector(2.0)
        @test eigvals(m) === vals
        @test vecs === SMatrix{1,1}(1.0)
        ef = eigen(m)
        @test ef.values === SVector(2.0)
        @test ef.vectors === SMatrix{1,1}(1.0)

        (vals, vecs) = eigen(Symmetric(m))
        @test vals === SVector(2.0)
        @test vecs === SMatrix{1,1}(1.0)
        ef = eigen(Symmetric(m))
        @test ef.values === SVector(2.0)
        @test ef.vectors === SMatrix{1,1}(1.0)
        # handle non-Hermitian case
        m = @SMatrix [2.0+im]
        @test eigvals(m) === SVector(2.0+im)
    end

    @testset "2×2" for i = 1:100
        m_a = randn(2,2)
        @test_throws ErrorException eigvals(SMatrix{2,2}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{2,2}(m_a)

        (vals_a, vecs_a) = eigen(m_a)
        (vals, vecs) = eigen(m)
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m
        ef = eigen(m)
        @test ef.values::SVector ≈ vals_a
        @test (ef.vectors*diagm(Val(0) => vals)*ef.vectors')::SMatrix ≈ m

        (vals, vecs) = eigen(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m
        ef = eigen(Symmetric(m))
        @test ef.values::SVector ≈ vals_a
        @test (ef.vectors*diagm(Val(0) => vals)*ef.vectors')::SMatrix ≈ m
        ef = eigen(Symmetric(m, :L))
        @test ef.values::SVector ≈ vals_a
        @test (ef.vectors*diagm(Val(0) => vals)*ef.vectors')::SMatrix ≈ m

        (vals, vecs) = eigen(Hermitian(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(Hermitian(m)) ≈ vals
        @test eigvals(Hermitian(m, :L)) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m
        ef = eigen(Hermitian(m))
        @test ef.values::SVector ≈ vals_a
        @test (ef.vectors*diagm(Val(0) => vals)*ef.vectors')::SMatrix ≈ m
        ef = eigen(Hermitian(m, :L))
        @test ef.values::SVector ≈ vals_a
        @test (ef.vectors*diagm(Val(0) => vals)*ef.vectors')::SMatrix ≈ m

        m_d = randn(SVector{2}); m = diagm(Val(0) => m_d)
        (vals, vecs) = eigen(Hermitian(m))
        @test vals::SVector ≈ sort(m_d)
        (vals, vecs) = eigen(Hermitian(m, :L))
        @test vals::SVector ≈ sort(m_d)
        @test eigvals(m) ≈ sort(m_d)
        @test eigvals(Hermitian(m)) ≈ sort(m_d)

        m_c = complex(m) # issue 614 (diagonal complex Hermitian)
        (vals, vecs) = eigen(Hermitian(m_c))
        @test vals::SVector ≈ sort(m_d)
        (vals, vecs) = eigen(Hermitian(m_c, :L))
        @test vals::SVector ≈ sort(m_d)
        @test eigvals(m_c) ≈ sort(m_d)
        @test eigvals(Hermitian(m_c)) ≈ sort(m_d)
    end

    # issue #523, #694
    zero = 0.0
    smallest_non_zero = nextfloat(zero)
    smallest_normal = floatmin(zero)
    largest_subnormal = prevfloat(smallest_normal)
    epsilon = eps(1.0)
    one_p_epsilon = nextfloat(1.0)
    degenerate = (zero, -1, 1, smallest_non_zero, smallest_normal, largest_subnormal, epsilon, one_p_epsilon, -one_p_epsilon)
    @testset "2×2 degenerate cases" for (i, j, k) in Iterators.product(degenerate,degenerate,degenerate), uplo in (:U, :L)
        A = SMatrix{2,2,Float64}((i, k, k, j))
        E = eigen(Symmetric(A, uplo))
        @test eigvecs(E) * SDiagonal(eigvals(E)) * eigvecs(E)' ≈ A
    end

    m1_a = randn(2,2)
    m1_a = m1_a*m1_a'
    m1 = SMatrix{2,2}(m1_a)
    m2_a = randn(2,2)
    m2_a = m2_a*m2_a'
    m2 = SMatrix{2,2}(m2_a)
    @test (@inferred_maybe_allow SVector{2,ComplexF64} eigvals(m1, m2)) ≈ eigvals(m1_a, m2_a)
    @test (@inferred_maybe_allow SVector{2,ComplexF64} eigvals(Symmetric(m1), Symmetric(m2))) ≈ eigvals(Symmetric(m1_a), Symmetric(m2_a))

    @test_throws DimensionMismatch eigvals(SA[1 2 3; 4 5 6], SA[1 2 3; 4 5 5])
    @test_throws DimensionMismatch eigvals(SA[1 2; 4 5], SA[1 2 3; 4 5 5; 3 4 5])

    @testset "3×3" for i = 1:100
        m_a = randn(3,3)
        m_a = m_a*m_a'
        m = SMatrix{3,3}(m_a)

        (vals_a, vecs_a) = eigen(m)
        (vals, vecs) = eigen(m)
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eigen(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test eigvals(Hermitian(m)) ≈ vals
        @test eigvals(Hermitian(m, :L)) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eigen(Symmetric(m, :L))
        @test vals::SVector ≈ vals_a

        m_d = randn(SVector{3}); m = diagm(Val(0) => m_d)
        (vals, vecs) = eigen(Hermitian(m))
        @test vals::SVector ≈ sort(m_d)
        (vals, vecs) = eigen(Hermitian(m, :L))
        @test vals::SVector ≈ sort(m_d)
        @test eigvals(m) ≈ sort(m_d)
        @test eigvals(Hermitian(m)) ≈ sort(m_d)

        m1_a = randn(3,3)
        m1_a = m1_a*m1_a'
        m1 = SMatrix{3,3}(m1_a)
        m2_a = randn(3,3)
        m2_a = m2_a*m2_a'
        m2 = SMatrix{3,3}(m2_a)
        @test (@inferred_maybe_allow SVector{3,ComplexF64} eigvals(m1, m2)) ≈ eigvals(m1_a, m2_a)
    end

    @testset "3x3 degenerate cases" begin
        # Rank 1
        v = randn(SVector{3,Float64})
        m = v*v'
        vv = sum(abs2, v)
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vecs'*vecs ≈ one(SMatrix{3,3,Float64})
        @test vals ≈ SVector(0.0, 0.0, vv)
        @test eigvals(m) ≈ vals

        # Rank 2
        v2 = randn(SVector{3,Float64})
        v2 -= dot(v,v2)*v/(vv)
        v2v2 = sum(abs2, v2)
        m += v2*v2'
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vecs'*vecs ≈ one(SMatrix{3,3,Float64})
        if vv < v2v2
            @test vals ≈ SVector(0.0, vv, v2v2)
        else
            @test vals ≈ SVector(0.0, v2v2, vv)
        end
        @test eigvals(m) ≈ vals

        # Degeneracy (2 large)
        m = -99*(v*v')/vv + 100*one(SMatrix{3,3,Float64})
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vecs'*vecs ≈ one(SMatrix{3,3,Float64})
        @test vals ≈ SVector(1.0, 100.0, 100.0)
        @test eigvals(m) ≈ vals

        # Degeneracy (2 small)
        m = (v*v')/vv + 1e-2*one(SMatrix{3,3,Float64})
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vecs'*vecs ≈ one(SMatrix{3,3,Float64})
        @test vals ≈ SVector(1e-2, 1e-2, 1.01)
        @test eigvals(m) ≈ vals

        # Block diagonal
        m = @SMatrix [1.0 0.0 0.0;
                      0.0 1.0 1.0;
                      0.0 1.0 1.0]
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs*diagm(Val(0) => vals)*vecs' ≈ m
        @test eigvals(m) ≈ vals

        m = @SMatrix [1.0 0.0 1.0;
                      0.0 1.0 0.0;
                      1.0 0.0 1.0]
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs*diagm(Val(0) => vals)*vecs' ≈ m
        @test eigvals(m) ≈ vals

        m = @SMatrix [1.0 1.0 0.0;
                      1.0 1.0 0.0;
                      0.0 0.0 1.0]
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs*diagm(Val(0) => vals)*vecs' ≈ m
        @test eigvals(m) ≈ vals
    end

    @testset "4×4" for i = 1:100
        m_a = randn(4,4)
        m_a = m_a*m_a'
        m = SMatrix{4,4}(m_a)

        (vals_a, vecs_a) = eigen(m_a)
        (vals, vecs) = eigen(m)
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eigen(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test eigvals(Hermitian(m)) ≈ vals
        @test eigvals(Hermitian(m, :L)) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eigen(Symmetric(m, :L))
        @test vals::SVector ≈ vals_a
        m_d = randn(SVector{4}); m = diagm(Val(0) => m_d)
        (vals, vecs) = eigen(Hermitian(m))
        @test vals::SVector ≈ sort(m_d)
        (vals, vecs) = eigen(Hermitian(m, :L))
        @test vals::SVector ≈ sort(m_d)
        @test eigvals(m) ≈ sort(m_d)
        @test eigvals(Hermitian(m)) ≈ sort(m_d)
    end

    @testset "complex" begin
        for n=1:5
            a = randn(n,n)+im*randn(n,n)
            a = a+a'
            A = Hermitian(SMatrix{n,n}(a))
            D,V = eigen(A)
            @test V'V ≈ Matrix(I, n, n)
            @test V*diagm(Val(0) => D)*V' ≈ A
            @test V'*A*V ≈ diagm(Val(0) => D)
        end
    end

    @testset "hermitian type stability" begin
        for n=1:4
            m = @SMatrix randn(n,n)
            m += m'

            @inferred eigen(Hermitian(m))
            @inferred eigen(Symmetric(m))

            # Test that general eigen() gives a small union of concrete types
            SEigen{T} = Eigen{T, T, SArray{Tuple{n,n},T,2,n*n}, SArray{Tuple{n},T,1,n}}
            @inferred_maybe_allow Union{SEigen{ComplexF64},SEigen{Float64}} eigen(m)

            mc = @SMatrix randn(ComplexF64, n, n)
            @inferred eigen(Hermitian(mc + mc'))
        end
    end

    @testset "non-hermitian 2d" begin
        for n=1:5
            angle = 2π * rand()
            rot = @SMatrix [cos(angle) -sin(angle); sin(angle) cos(angle)]

            vals, vecs = eigen(rot)

            @test norm(vals[1]) ≈ 1.0
            @test norm(vals[2]) ≈ 1.0

            @test vecs[:,1] ≈ conj.(vecs[:,2])
        end
    end

    @testset "non-hermitian 3d" begin
        for n=1:5
            angle = 2π * rand()
            rot = @SMatrix [cos(angle) 0.0 -sin(angle); 0.0 1.0 0.0; sin(angle) 0.0 cos(angle)]

            vals, vecs = eigen(rot)

            @test norm(vals[1]) ≈ 1.0
            @test norm(vals[2]) ≈ 1.0

            @test vecs[:,1] ≈ conj.(vecs[:,2])
        end
    end
end
