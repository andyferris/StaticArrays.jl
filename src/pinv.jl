function pinv(A::StaticMatrix{m,n,T} where m where n; atol::Real = 0.0, rtol::Real = (eps(real(float(one(T))))*min(size(A)...))*iszero(atol)) where T
    S = StaticArrays.arithmetic_closure(T)
    A_S = convert(similar_type(A,S),A)
    _pinv(Size(A_S),A_S)
end

function _pinv(s::Size{sizes}, A::StaticMatrix{m,n,T}; atol::Real = 0.0, rtol::Real = (eps(real(float(one(T))))*min(size(A)...))*iszero(atol)) where T where sizes where m where n
    # This function is a StaticMatrix version of LinearAlgebra.pinv
    if m == 0 || n == 0
        Tout = typeof(zero(T)/sqrt(one(T) + one(T)))
        return similar_type(A, Size(n,m))()
    end
    if istril(A)
        if istriu(A)
            maxabsA = maximum(abs.(diag(A)))
            tol = max(rtol*maxabsA, atol)
            return _pinv_B(s, A, tol)
        end
    else
        _svd = svd(A, full = false)
        tol = max(rtol*maximum(_svd.S), atol)
        sinv = _pinv_A(s, A, _svd, tol)
        return _svd.Vt'*SDiagonal(sinv)*_svd.U'
    end
end

@generated function _pinv_B(::Size{sizes}, A::StaticMatrix{m,n,T} where m where n, tol) where sizes where T
    minlen = min(sizes[1],sizes[2])

    exprs = [:(zero($T)) for i in 1:sizes[2], j in 1:sizes[1]]
    for i in 1:minlen
        exprs[i,i] = :(ifelse(A[$i,$i]<tol, zero($T), inv(A[$i,$i])))
    end

    return quote
        Base.@_inline_meta
        @inbounds return similar_type(A, Size($sizes[2], $sizes[1]))(tuple($(exprs...)))
    end
end

@generated function _pinv_A(::Size{sizes}, A::StaticMatrix, _svd::StaticArrays.SVD, tol) where sizes
    minlen = min(sizes[1],sizes[2])

    exprs = [
        :(
            Stype = eltype(_svd.S);
            sinvi = ifelse(_svd.S[$i] > tol, one(Stype) / _svd.S[$i], zero(Stype));
            ifelse(isfinite(sinvi), sinvi, zero(Stype))
        )
        for i in 1:minlen
    ]

    return quote
        Base.@_inline_meta
        @inbounds return similar_type(A, Size($minlen))(tuple($(exprs...)))
    end
end