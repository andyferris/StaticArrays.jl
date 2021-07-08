using StaticArrays, Test, LinearAlgebra

tol = 1e-13

@testset "Moore–Penrose inverse (Peseudo-inverse)" begin
    M1 = @SMatrix [1.5 1.3; 1.2 1.9]
    N1 = pinv(M1)
    @test norm(M1*N1*M1 - M1) < tol
    @test norm(N1*M1*N1 - N1) < tol
    @test N1 isa SMatrix{2,2,Float64}
    @test N1 ≈ pinv(Matrix(M1))

    M2 = @SMatrix [1//2 0 0;3//2 5//3 8//7;9//4 -1//3 -8//7;0 0 0]
    N2 = pinv(M2)
    @test norm(M2*N2*M2 - M2) < tol
    @test norm(N2*M2*N2 - N2) < tol
    @test N2 isa SMatrix{3,4,Float64}
    @test N2 ≈ pinv(Matrix(M2))

    M3 = SDiagonal(0,1,2,3)
    N3 = pinv(M3)
    @test norm(M3*N3*M3 - M3) < tol
    @test norm(N3*M3*N3 - N3) < tol
    @test N3 isa Diagonal{Float64, <:SVector{4,Float64}}
    @test N3 ≈ pinv(Matrix(M3))

    M4 = @SMatrix randn(2,5)
    N4 = pinv(M4)
    @test norm(M4*N4*M4 - M4) < tol
    @test norm(N4*M4*N4 - N4) < tol
    @test N4 isa SMatrix{5,2,Float64}
    @test N4 ≈ pinv(Matrix(M4))

    M5 = SMatrix{0,5,Int}()
    N5 = pinv(M5)
    @test norm(M5*N5*M5 - M5) < tol
    @test norm(N5*(M5*N5) - N5) < tol
    @test N5 isa SMatrix{5,0,Float64}
    @test N5 ≈ pinv(Matrix(M5))

    M6 = @SMatrix [1/2 0 0;0 5/3 0;0 0 0;0 0 0]
    N6 = pinv(M6)
    @test norm(M6*N6*M6 - M6) < tol
    @test norm(N6*M6*N6 - N6) < tol
    @test N6 isa SMatrix{3,4,Float64}
    @test N6 ≈ pinv(Matrix(M6))

    M7 = M6'
    N7 = pinv(M7)
    @test norm(M7*N7*M7 - M7) < tol
    @test norm(N7*M7*N7 - N7) < tol
    @test N7 isa SMatrix{4,3,Float64}
    @test N7 ≈ pinv(Matrix(M7))

    M8 = @MMatrix [0.5 1.1 0.0;0.0 -2.8 0.0;0.0 0.0 0.0;0.0 0.0 0.0]
    N8 = pinv(M8)
    @test norm(M8*N8*M8 - M8) < tol
    @test norm(N8*M8*N8 - N8) < tol
    @test N8 isa MMatrix{3,4,Float64}
    @test N8 ≈ pinv(Matrix(M8))

    M9 = M8'
    N9 = pinv(M9)
    @test norm(M9*N9*M9 - M9) < tol
    @test norm(N9*M9*N9 - N9) < tol
    @test N9 isa MMatrix{4,3,Float64}
    @test N9 ≈ pinv(Matrix(M9))

    M10 = @SMatrix randn(3,3)
    N10 = pinv(M10)
    @test N10 ≈ inv(M10)
    @test norm(M10*N10*M10 - M10) < tol
    @test norm(N10*M10*N10 - N10) < tol
    @test N10 isa StaticMatrix
    @test N10 ≈ pinv(Matrix(M10))
end
