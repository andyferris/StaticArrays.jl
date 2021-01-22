# Moore-Penrose pseudoinverse

"""
    pinv(M; atol::Real=0, rtol::Real=atol>0 ? 0 : n*ϵ)
    pinv(M, rtol::Real) = pinv(M; rtol=rtol) # to be deprecated in Julia 2.0

This function is a StaticMatrix version of `LinearAlgebra.pinv`.

# Examples
```jldoctest
julia> M1 = @SMatrix [1.5 1.3; 1.2 1.9]
2×2 SArray{Tuple{2,2},Float64,2,4} with indices SOneTo(2)×SOneTo(2):
 1.5  1.3
 1.2  1.9

julia> pinv(M1)
2×2 SArray{Tuple{2,2},Float64,2,4} with indices SOneTo(2)×SOneTo(2):
  1.47287   -1.00775
 -0.930233   1.16279

julia> M1 * pinv(M1)
2×2 SArray{Tuple{2,2},Float64,2,4} with indices SOneTo(2)×SOneTo(2):
 1.0          -2.22045e-16
 1.56636e-16   1.0

julia> M2 = @SMatrix [1//2 0 0;3//2 5//3 8//7;9//4 -1//3 -8//7;0 0 0]
4×3 SArray{Tuple{4,3},Rational{Int64},2,12} with indices SOneTo(4)×SOneTo(3):
 1//2   0//1   0//1
 3//2   5//3   8//7
 9//4  -1//3  -8//7
 0//1   0//1   0//1

julia> pinv(M2)
3×4 SArray{Tuple{3,4},Float64,2,12} with indices SOneTo(3)×SOneTo(4):
  2.0       4.05208e-17  -1.06076e-16  0.0
 -5.625     0.75          0.75         0.0
  5.57812  -0.21875      -1.09375      0.0
```
"""
@inline function pinv(A::StaticMatrix{m,n,T} where m where n; atol::Real = 0.0, rtol::Real = (eps(real(float(one(T))))*min(size(A)...))*iszero(atol)) where T
    S = typeof(zero(T)/sqrt(one(T) + one(T)))
    A_S = convert(similar_type(A,S),A)
    return _pinv(A_S, atol, rtol)
end

@inline function _pinv(A::StaticMatrix{m,n,T}, atol::Real, rtol::Real) where T where m where n
    if m == 0 || n == 0
        return similar_type(A, Size(n,m))()
    end
    if istril(A)
        if istriu(A)
            maxabsA = maximum(abs.(diag(A)))
            tol = max(rtol*maxabsA, atol)
            return _pinv_M(A, tol)
        end
    end
    ssvd = svd(A, full = false)
    tol = max(rtol*maximum(ssvd.S), atol)
    sinv = _pinv_V(ssvd.S, tol)
    return ssvd.Vt'*SDiagonal(sinv)*ssvd.U'
end

@inline function pinv(D::Diagonal{T,<:StaticVector}) where T
    V = D.diag
    S = typeof(zero(T)/sqrt(one(T) + one(T)))
    V_S = convert(similar_type(V,S),V)
    return Diagonal(_pinv_V(V_S))
end

@generated function _pinv_M(A::StaticMatrix{m,n,T}, tol) where m where n where T
    minlen = min(m,n)
    exprs = [:(zero($T)) for i in 1:n, j in 1:m]
    for i in 1:minlen
        exprs[i,i] = :(ifelse(A[$i,$i] > tol, inv(A[$i,$i]), zero($T)))
    end
    return quote
        Base.@_inline_meta
        @inbounds return similar_type(A, Size($n, $m))(tuple($(exprs...)))
    end
end

@generated function _pinv_V(v::StaticVector{n,T}, tol) where n where T
    exprs = [
        :(ifelse(v[$i] > tol, inv(v[$i]), zero(T)))
        for i in 1:n
    ]
    return quote
        Base.@_inline_meta
        @inbounds return similar_type(v, Size($n))(tuple($(exprs...)))
    end
end

@generated function _pinv_V(v::StaticVector{n,T}) where n where T
    exprs = [
        :(pinv(v[$i]))
        for i in 1:n
    ]
    return quote
        Base.@_inline_meta
        @inbounds return similar_type(v, Size($n))(tuple($(exprs...)))
    end
end
