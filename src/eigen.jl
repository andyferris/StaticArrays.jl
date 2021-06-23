
@inline function eigvals(a::LinearAlgebra.RealHermSymComplexHerm{T,SA},; permute::Bool=true, scale::Bool=true) where {T <: Real, SA <: StaticArray}
    _eigvals(Size(SA), a, permute, scale)
end

@inline function eigvals(a::StaticArray; permute::Bool=true, scale::Bool=true)
    if ishermitian(a)
        _eigvals(Size(a), Hermitian(a), permute, scale)
    elseif Size(a) == Size{(1,1)}()
        _eigvals(Size(a), a, permute, scale)
    else
        error("Only hermitian matrices are diagonalizable by *StaticArrays*. Non-Hermitian matrices should be converted to `Array` first.")
    end
end

function eigvals(A::StaticMatrixLike, B::StaticMatrixLike; kwargs...)
    SA = Size(A)
    if SA != Size(B)
        throw(DimensionMismatch("Generalized eigenvalues can only be calculated for matrices of equal sizes: dimensions are $SA and $(Size(B))"))
    end
    checksquare(A)
    return _eigvals(SA, A, B; kwargs...)
end

@inline function _eigvals(s::Size, A::StaticMatrixLike, B::StaticMatrixLike; kwargs...)
    return SVector{s[1]}(eigvals(Array(A), Array(B); kwargs...))
end

@inline _eigvals(::Size{(1,1)}, a, permute, scale) = @inbounds return SVector(Tuple(a))
@inline _eigvals(::Size{(1, 1)}, a::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real} = @inbounds return SVector(real(parent(a).data[1]))

@inline function _eigvals(::Size{(2,2)}, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    a = A.data

    if A.uplo == 'U'
        @inbounds t_half = real(a[1] + a[4])/2
        @inbounds d = real(a[1]*a[4] - a[3]'*a[3]) # Should be real

        tmp2 = t_half*t_half - d
        tmp2 < 0 ? tmp = zero(tmp2) : tmp = sqrt(tmp2) # Numerically stable for identity matrices, etc.
        return SVector(t_half - tmp, t_half + tmp)
    else
        @inbounds t_half = real(a[1] + a[4])/2
        @inbounds d = real(a[1]*a[4] - a[2]'*a[2]) # Should be real

        tmp2 = t_half*t_half - d
        tmp2 < 0 ? tmp = zero(tmp2) : tmp = sqrt(tmp2) # Numerically stable for identity matrices, etc.
        return SVector(t_half - tmp, t_half + tmp)
    end
end

@inline function _eigvals(::Size{(3,3)}, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    S = arithmetic_closure(T)
    Sreal = real(S)

    @inbounds a11 = convert(Sreal, A.data[1])
    @inbounds a22 = convert(Sreal, A.data[5])
    @inbounds a33 = convert(Sreal, A.data[9])
    if A.uplo == 'U'
        @inbounds a12 = convert(S, A.data[4])
        @inbounds a13 = convert(S, A.data[7])
        @inbounds a23 = convert(S, A.data[8])
    else
        @inbounds a12 = conj(convert(S, A.data[2]))
        @inbounds a13 = conj(convert(S, A.data[3]))
        @inbounds a23 = conj(convert(S, A.data[6]))
    end

    p1 = abs2(a12) + abs2(a13) + abs2(a23)
    if (p1 == 0)
        # Matrix is diagonal
        if a11 < a22
            if a22 < a33
                return SVector(a11, a22, a33)
            elseif a33 < a11
                return SVector(a33, a11, a22)
            else
                return SVector(a11, a33, a22)
            end
        else #a22 < a11
            if a11 < a33
                return SVector(a22, a11, a33)
            elseif a33 < a22
                return SVector(a33, a22, a11)
            else
                return SVector(a22, a33, a11)
            end
        end
    end

    q = (a11 + a22 + a33) / 3
    p2 = abs2(a11 - q) + abs2(a22 - q) + abs2(a33 - q) + 2 * p1
    p = sqrt(p2 / 6)
    invp = inv(p)
    b11 = (a11 - q) * invp
    b22 = (a22 - q) * invp
    b33 = (a33 - q) * invp
    b12 = a12 * invp
    b13 = a13 * invp
    b23 = a23 * invp
    B = SMatrix{3,3,S}((b11, conj(b12), conj(b13), b12, b22, conj(b23), b13, b23, b33))
    r = real(det(B)) / 2

    # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    # but computation error can leave it slightly outside this range.
    if (r <= -1)
        phi = Sreal(pi) / 3
    elseif (r >= 1)
        phi = zero(Sreal)
    else
        phi = acos(r) / 3
    end

    eig3 = q + 2 * p * cos(phi)
    eig1 = q + 2 * p * cos(phi + (2*Sreal(pi)/3))
    eig2 = 3 * q - eig1 - eig3     # since tr(A) = eig1 + eig2 + eig3

    return SVector(eig1, eig2, eig3)
end

@inline function _eigvals(s::Size, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    vals = eigvals(Hermitian(Array(parent(A))))
    return SVector{s[1], T}(vals)
end

# Utility to rewrap `Eigen` of normal `Array` into an Eigen containing `SArray`.
@inline function _make_static(s::Size, E::Eigen{T,V}) where {T,V}
    Eigen(similar_type(SVector, V, Size(s[1]))(E.values),
          similar_type(SMatrix, T, s)(E.vectors))
end

@inline function _eig(s::Size, A::T, permute, scale) where {T <: StaticMatrix}
    if ishermitian(A)
        return _eig(s, Hermitian(A), permute, scale)
    else
        # For the non-hermitian branch fall back to LinearAlgebra eigen().
        # Eigenvalues could be real or complex so a Union of concrete types is
        # inferred. Having _make_static a separate function allows inference to
        # preserve the union of concrete types:
        #   Union{E{A,B},E{C,D}} -> Union{E{SA,SB},E{SC,SD}}
        _make_static(s, eigen(Array(A); permute = permute, scale = scale))
    end
end

@inline function _eig(s::Size, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    E = eigen(Hermitian(Array(parent(A))))
    return Eigen(SVector{s[1], T}(E.values), SMatrix{s[1], s[2], eltype(A)}(E.vectors))
end


@inline function _eig(::Size{(1,1)}, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    @inbounds return Eigen(SVector{1,T}((real(A[1]),)), SMatrix{1,1,eltype(A)}(I))
end

@inline function _eig(::Size{(2,2)}, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    a = A.data
    TA = eltype(A)

    A11 = real(a[1])
    A22 = real(a[4])
    if A.uplo == 'U'
        @inbounds A21 = a[3]'
    else
        @inbounds A21 = a[2]
    end
    @inbounds if !iszero(A21) # A is not diagonal
        t_half = (A11 + A22) / 2
        d_half = (A11 - A22) / 2

        tmp = hypot(d_half, A21)
        vals = SVector(t_half - tmp, t_half + tmp)

        v11 = d_half - tmp
        n1 = hypot(v11, A21) # always > 0
        if n1 < floatmin(T) # n1 subnormal
            scale = inv(floatmin(T))
            n1 *= scale
            v11 = (v11 * scale) / n1
            v12 = (A21 * scale) / n1
        else
            v11 = v11 / n1
            v12 = A21 / n1
        end

        vecs = @SMatrix [ v11  -v12' ;
                          v12  v11' ]

        return Eigen(vals, vecs)
    end

    # A must be diagonal if we reached this point;
    # treatment of uplo 'L' and 'U' is then identical
    if A11 < A22
        vals = SVector(A11, A22)
        vecs = @SMatrix [convert(TA, 1) convert(TA, 0);
                         convert(TA, 0) convert(TA, 1)]
    else # A22 <= A11
        vals = SVector(A22, A11)
        vecs = @SMatrix [convert(TA, 0) convert(TA, 1);
                         convert(TA, 1) convert(TA, 0)]
    end
    return Eigen(vals,vecs)
end

# Port of https://www.geometrictools.com/GTEngine/Include/Mathematics/GteSymmetricEigensolver3x3.h
# released by David Eberly, Geometric Tools, Redmond WA 98052
# under the Boost Software License, Version 1.0 (included at the end of this file)
# The original documentation states
# (see https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf )
# [This] is an implementation of Algorithm 8.2.3 (Symmetric QR Algorithm) described in
# Matrix Computations,2nd edition, by G. H. Golub and C. F. Van Loan, The Johns Hopkins
# University Press, Baltimore MD, Fourth Printing 1993. Algorithm 8.2.1 (Householder
# Tridiagonalization) is used to reduce matrix A to tridiagonal D′. Algorithm 8.2.2
# (Implicit Symmetric QR Step with Wilkinson Shift) is used for the iterative reduction
# from tridiagonal to diagonal. Numerically, we have errors E=RTAR−D. Algorithm 8.2.3
# mentions that one expects |E| is approximately μ|A|, where |M| denotes the Frobenius norm
# of M and where μ is the unit roundoff for the floating-point arithmetic: 2−23 for float,
# which is FLTEPSILON = 1.192092896e-7f, and 2−52 for double, which is
# DBLEPSILON = 2.2204460492503131e-16.
# TODO ensure right-handedness of the eigenvalue matrix
# TODO extend the method to complex hermitian
@inline function _eig(::Size{(3,3)}, A::LinearAlgebra.HermOrSym{T}, permute, scale) where {T <: Real}
    function converged(aggressive, bdiag0, bdiag1, bsuper)
        if aggressive
            bsuper == 0
        else
            diag_sum = abs(bdiag0) + abs(bdiag1)
            diag_sum + bsuper == diag_sum
        end
    end

    function get_cos_sin(u::T,v::T) where {T}
        max_abs = max(abs(u), abs(v))
        if max_abs > 0
            u,v = (u,v) ./ max_abs
            len = sqrt(u^2 + v^2)
            cs, sn = (u,v) ./ len
            if cs > 0
                cs = -cs
                sn = -sn
            end
            T(cs), T(sn)
        else
            T(-1), T(0)
        end
    end

    function _sortperm3(v)
        local perm = SVector(1,2,3)
        # unrolled bubble-sort
        (v[perm[1]] > v[perm[2]]) && (perm = SVector(perm[2], perm[1], perm[3]))
        (v[perm[2]] > v[perm[3]]) && (perm = SVector(perm[1], perm[3], perm[2]))
        (v[perm[1]] > v[perm[2]]) && (perm = SVector(perm[2], perm[1], perm[3]))
        perm
    end

    # Givens reflections
    update0(Q, c, s) = Q * @SMatrix [c 0 -s; s 0 c; 0 1 0]
    update1(Q, c, s) = Q * @SMatrix [0 1 0; c 0 s; -s 0 c]
    # Householder reflections
    update2(Q, c, s) = Q * @SMatrix [c s 0; s -c 0; 0 0 1]
    update3(Q, c, s) = Q * @SMatrix [1 0 0; 0 c s; 0 s -c]

    is_rotation = false

    # If `aggressive` is `true`, the iterations occur until a superdiagonal
    # entry is exactly zero, otherwise they occur until it is effectively zero
    # compared to the magnitude of its diagonal neighbors. Generally the non-
    # aggressive convergence is acceptable.
    #
    # Even with `aggressive = true` this method is faster than the one it
    # replaces and in order to keep the old interface, aggressive is set to true
    aggressive = true

    # the input is symmetric, so we only consider the unique elements:
    a00, a01, a02, a11, a12, a22 = A[1,1], A[1,2], A[1,3], A[2,2], A[2,3], A[3,3]

    # Compute the Householder reflection H and B = H * A * H where b02 = 0

    c,s = get_cos_sin(a12, -a02)

    Q = @SMatrix [c s 0; s -c 0; 0 0 1]

    term0 = c * a00 + s * a01
    term1 = c * a01 + s * a11
    b00 = c * term0 + s * term1
    b01 = s * term0 - c * term1
    term0 = s * a00 - c * a01
    term1 = s * a01 - c * a11
    b11 = s * term0 - c * term1
    b12 = s * a02 - c * a12
    b22 = a22

    # Givens reflections, B' = G^T * B * G, preserve tridiagonal matrices
    max_iteration = 2 * (1 + precision(T) - exponent(floatmin(T)))

    if abs(b12) <= abs(b01)
        saveB00, saveB01, saveB11 = b00, b01, b11
        for iteration in 1:max_iteration
            # compute the Givens reflection
            c2, s2 = get_cos_sin((b00 - b11) / 2, b01)
            s = sqrt((1 - c2) / 2)
            c = s2 / 2s

            # update Q by the Givens reflection
            Q = update0(Q, c, s)
            is_rotation = !is_rotation

            # update B ← Q^T * B * Q, ensuring that b02 is zero and |b12| has
            # strictly decreased
            saveB00, saveB01, saveB11 = b00, b01, b11
            term0 = c * saveB00 + s * saveB01
            term1 = c * saveB01 + s * saveB11
            b00 = c * term0 + s * term1
            b11 = b22
            term0 = c * saveB01 - s * saveB00
            term1 = c * saveB11 - s * saveB01
            b22 = c * term1 - s * term0
            b01 = s * b12
            b12 = c * b12

            if converged(aggressive, b00, b11, b01)
                # compute the Householder reflection
                c2, s2 = get_cos_sin((b00 - b11) / 2, b01)
                s = sqrt((1 - c2) / 2)
                c = s2 / 2s

                # update Q by the Householder reflection
                Q = update2(Q, c, s)
                is_rotation = !is_rotation

                # update D = Q^T * B * Q
                saveB00, saveB01, saveB11 = b00, b01, b11
                term0 = c * saveB00 + s * saveB01
                term1 = c * saveB01 + s * saveB11
                b00 = c * term0 + s * term1
                term0 = s * saveB00 - c * saveB01
                term1 = s * saveB01 - c * saveB11
                b11 = s * term0 - c * term1
                break
            end
        end
    else
        saveB11, saveB12, saveB22 = b11, b12, b22
        for iteration in 1:max_iteration
            # compute the Givens reflection
            c2, s2 = get_cos_sin((b22 - b11) / 2, b12)
            s = sqrt((1 - c2) / 2)
            c = s2 / 2s

            # update Q by the Givens reflection
            Q = update1(Q, c, s)
            is_rotation = !is_rotation

            # update B ← Q^T * B * Q ensuring that b02 is zero and |b12| has
            # strictly decreased.
            saveB11, saveB12, saveB22 = b11, b12, b22

            term0 = c * saveB22 + s * saveB12
            term1 = c * saveB12 + s * saveB11
            b22 = c * term0 + s * term1
            b11 = b00
            term0 = c * saveB12 - s * saveB22
            term1 = c * saveB11 - s * saveB12
            b00 = c * term1 - s * term0
            b12 = s * b01
            b01 = c * b01

            if converged(aggressive, b11, b22, b12)
                # compute the Householder reflection
                c2, s2 = get_cos_sin((b11 - b22) / 2, b12)
                s = sqrt((1 - c2) / 2)
                c = s2 / 2s

                # update Q by the Householder reflection
                Q = update3(Q, c, s)
                is_rotation = !is_rotation

                # update D = Q^T * B * Q
                saveB11, saveB12, saveB22 = b11, b12, b22
                term0 = c * saveB11 + s * saveB12
                term1 = c * saveB12 + s * saveB22
                b11 = c * term0 + s * term1
                term0 = s * saveB11 - c * saveB12
                term1 = s * saveB12 - c * saveB22
                b22 = s * term0 - c * term1
                break
            end
        end
    end

    evals = @SVector [b00, b11, b22]
    perm = _sortperm3(evals)
    Eigen(evals[perm], Q[:,perm])
end


@inline function eigen(A::StaticMatrix; permute::Bool=true, scale::Bool=true)
    _eig(Size(A), A, permute, scale)
end

# to avoid method ambiguity with LinearAlgebra
@inline eigen(A::Hermitian{<:Real,<:StaticMatrix}; kwargs...)    = _eigen(A; kwargs...)
@inline eigen(A::Hermitian{<:Complex,<:StaticMatrix}; kwargs...) = _eigen(A; kwargs...)
@inline eigen(A::Symmetric{<:Real,<:StaticMatrix}; kwargs...)    = _eigen(A; kwargs...)
@inline eigen(A::Symmetric{<:Complex,<:StaticMatrix}; kwargs...) = _eigen(A; kwargs...)

@inline function _eigen(A::LinearAlgebra.HermOrSym; permute::Bool=true, scale::Bool=true)
    _eig(Size(A), A, permute, scale)
end

# NOTE: The following Boost Software License applies to parts of the method:
#     _eig{T<:Real}(::Size{(3,3)}, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale)

#=
Boost Software License - Version 1.0 - August 17th, 2003



Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
=#
