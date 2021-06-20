import LinearAlgebra: BlasFloat, matprod, mul!


# Manage dispatch of * and mul!
# TODO Adjoint? (Inner product?)

# *(A::StaticMatMulLike, B::AbstractVector) causes an ambiguity with SparseArrays
@inline *(A::StaticMatrix, B::AbstractVector) = _mul(Size(A), A, B)
@inline *(A::StaticMatMulLike, B::StaticVector) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticMatrix, B::StaticVector) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticMatMulLike, B::StaticMatMulLike) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticVector, B::StaticMatMulLike) = *(reshape(A, Size(Size(A)[1], 1)), B)
@inline *(A::StaticVector, B::Transpose{<:Any, <:StaticVector}) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticVector, B::Adjoint{<:Any, <:StaticVector}) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticArray{Tuple{N,1},<:Any,2}, B::Adjoint{<:Any,<:StaticVector}) where {N} = vec(A) * B
@inline *(A::StaticArray{Tuple{N,1},<:Any,2}, B::Transpose{<:Any,<:StaticVector}) where {N} = vec(A) * B

# Avoid LinearAlgebra._quad_matmul's order calculation on equal sizes
@inline *(A::StaticMatMulLike{N,N}, B::StaticMatMulLike{N,N}, C::StaticMatMulLike{N,N}) where {N} = (A*B)*C
@inline *(A::StaticMatMulLike{N,N}, B::StaticMatMulLike{N,N}, C::StaticMatMulLike{N,N}, D::StaticMatMulLike{N,N}) where {N} = ((A*B)*C)*D

"""
    mul_result_structure(a::Type, b::Type)

Get a structure wrapper that should be applied to the result of multiplication of matrices
of given types (a*b).
"""
function mul_result_structure(a, b)
    return identity
end
function mul_result_structure(::UpperTriangular{<:Any, <:StaticMatrix}, ::UpperTriangular{<:Any, <:StaticMatrix})
    return UpperTriangular
end
function mul_result_structure(::LowerTriangular{<:Any, <:StaticMatrix}, ::LowerTriangular{<:Any, <:StaticMatrix})
    return LowerTriangular
end
function mul_result_structure(::UpperTriangular{<:Any, <:StaticMatrix}, ::SDiagonal)
    return UpperTriangular
end
function mul_result_structure(::LowerTriangular{<:Any, <:StaticMatrix}, ::SDiagonal)
    return LowerTriangular
end
function mul_result_structure(::SDiagonal, ::UpperTriangular{<:Any, <:StaticMatrix})
    return UpperTriangular
end
function mul_result_structure(::SDiagonal, ::LowerTriangular{<:Any, <:StaticMatrix})
    return LowerTriangular
end
function mul_result_structure(::UnitUpperTriangular{<:Any, <:StaticMatrix}, ::SDiagonal)
    return UpperTriangular
end
function mul_result_structure(::UnitLowerTriangular{<:Any, <:StaticMatrix}, ::SDiagonal)
    return LowerTriangular
end
function mul_result_structure(::SDiagonal, ::UnitUpperTriangular{<:Any, <:StaticMatrix})
    return UpperTriangular
end
function mul_result_structure(::SDiagonal, ::UnitLowerTriangular{<:Any, <:StaticMatrix})
    return LowerTriangular
end
function mul_result_structure(::SDiagonal, ::SDiagonal)
    return Diagonal
end

# Implementations

function mul_smat_vec_exprs(sa, access_a)
    return [combine_products([:($(uplo_access(sa, :a, k, j, access_a))*b[$j]) for j = 1:sa[2]]) for k = 1:sa[1]]
end

@generated function _mul(::Size{sa}, wrapped_a::StaticMatMulLike{<:Any, <:Any, Ta}, b::AbstractVector{Tb}) where {sa, Ta, Tb}
    if sa[2] != 0
        retexpr = gen_by_access(wrapped_a) do access_a
            exprs = mul_smat_vec_exprs(sa, access_a)
            return :(@inbounds return similar_type(b, T, Size(sa[1]))(tuple($(exprs...))))
        end
    else
        exprs = [:(zero(T)) for k = 1:sa[1]]
        retexpr = :(@inbounds return similar_type(b, T, Size(sa[1]))(tuple($(exprs...))))
    end

    return quote
        @_inline_meta
        if length(b) != sa[2]
            throw(DimensionMismatch("Tried to multiply arrays of size $sa and $(size(b))"))
        end
        T = promote_op(matprod,Ta,Tb)
        a = mul_parent(wrapped_a)
        $retexpr
    end
end

@generated function _mul(::Size{sa}, ::Size{sb}, wrapped_a::StaticMatMulLike{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    if sa[2] != 0
        retexpr = gen_by_access(wrapped_a) do access_a
            exprs = mul_smat_vec_exprs(sa, access_a)
            return :(@inbounds return similar_type(b, T, Size(sa[1]))(tuple($(exprs...))))
        end
    else
        exprs = [:(zero(T)) for k = 1:sa[1]]
        retexpr = :(@inbounds return similar_type(b, T, Size(sa[1]))(tuple($(exprs...))))
    end

    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        a = mul_parent(wrapped_a)
        $retexpr
    end
end

# outer product
@generated function _mul(::Size{sa}, ::Size{sb}, a::StaticVector{<: Any, Ta},
        b::Union{Transpose{Tb, <:StaticVector}, Adjoint{Tb, <:StaticVector}}) where {sa, sb, Ta, Tb}
    newsize = (sa[1], sb[2])
    exprs = [:(a[$i]*b[$j]) for i = 1:sa[1], j = 1:sb[2]]
    return quote
        @_inline_meta
        T = promote_op(*, Ta, Tb)
        @inbounds return similar_type(b, T, Size($newsize))(tuple($(exprs...)))
    end
end

_unstatic_array(::Type{TSA}) where {S, T, N, TSA<:StaticArray{S,T,N}} = AbstractArray{T,N}
for TWR in [Adjoint, Transpose, Symmetric, Hermitian, LowerTriangular, UpperTriangular, UnitUpperTriangular, UnitLowerTriangular, Diagonal]
    @eval _unstatic_array(::Type{$TWR{T,TSA}}) where {S, T, N, TSA<:StaticArray{S,T,N}} = $TWR{T,<:AbstractArray{T,N}}
end

@generated function _mul(Sa::Size{sa}, Sb::Size{sb}, a::StaticMatMulLike{<:Any, <:Any, Ta}, b::StaticMatMulLike{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    S = Size(sa[1], sb[2])
    # Heuristic choice for amount of codegen
    a_tri_mul = a <: LinearAlgebra.AbstractTriangular ? 4 : 1
    b_tri_mul = b <: LinearAlgebra.AbstractTriangular ? 4 : 1
    ab_tri_mul = (a == 4 && b == 4) ? 2 : 1
    if a <: StaticMatrix && b <: StaticMatrix
        # Julia unrolls these loops pretty well
        return quote
            @_inline_meta
            return mul_loop(Sa, Sb, a, b)
        end
    elseif sa[1]*sa[2]*sb[2] <= 4*8*8*8*a_tri_mul*b_tri_mul*ab_tri_mul || a <: Diagonal || b <: Diagonal
        return quote
            @_inline_meta
            return mul_unrolled(Sa, Sb, a, b)
        end
    elseif (sa[1] <= 14 && sa[2] <= 14 && sb[2] <= 14) || !(a <: StaticMatrix) || !(b <: StaticMatrix)
        return quote
            @_inline_meta
            return mul_unrolled_chunks(Sa, Sb, a, b)
        end
    else
        # we don't have any special code for handling this case so let's fall back to
        # the generic implementation of matrix multiplication
        return quote
            @_inline_meta
            return mul_generic(Sa, Sb, a, b)
        end
    end
end

@generated function mul_unrolled(::Size{sa}, ::Size{sb}, wrapped_a::StaticMatMulLike{<:Any, <:Any, Ta}, wrapped_b::StaticMatMulLike{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    if sa[2] != 0
        retexpr = gen_by_access(wrapped_a, wrapped_b) do access_a, access_b
            exprs = [combine_products([:($(uplo_access(sa, :a, k1, j, access_a))*$(uplo_access(sb, :b, j, k2, access_b))) for j = 1:sa[2]]
                ) for k1 = 1:sa[1], k2 = 1:sb[2]]
            return :((mul_result_structure(wrapped_a, wrapped_b))(similar_type(a, T, $S)(tuple($(exprs...)))))
        end
    else
        exprs = [:(zero(T)) for k1 = 1:sa[1], k2 = 1:sb[2]]
        retexpr = :(return (mul_result_structure(wrapped_a, wrapped_b))(similar_type(a, T, $S)(tuple($(exprs...)))))
    end

    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        a = mul_parent(wrapped_a)
        b = mul_parent(wrapped_b)
        @inbounds $retexpr
    end
end

@generated function mul_loop(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    # optimal for AVX2 with `Float64
    # AVX512 would want something more like 16x14 or 24x9 with `Float64`
    M_r, N_r = 8, 6
    n = 0
    M, K = sa
    N = sb[2]
    q = Expr(:block)
    atemps = [Symbol(:a_, k1) for k1 = 1:M]
    tmps = [Symbol("tmp_$(k1)_$(k2)") for k1 = 1:M, k2 = 1:N]
    while n < N
        nu = min(N, n + N_r)
        nrange = n+1:nu
        m = 0
        while m < M
            mu = min(M, m + M_r)
            mrange = m+1:mu

            atemps_init = [:($(atemps[k1]) = a[$k1]) for k1 = mrange]
            exprs_init = [:($(tmps[k1,k2])  = $(atemps[k1]) * b[$(1 + (k2-1) * sb[1])]) for k1 = mrange, k2 = nrange]
            atemps_loop_init = [:($(atemps[k1]) = a[$(k1-sa[1]) + $(sa[1])*j]) for k1 = mrange]
            exprs_loop = [:($(tmps[k1,k2]) = muladd($(atemps[k1]), b[j + $((k2-1) * sb[1])], $(tmps[k1,k2]))) for k1 = mrange, k2 = nrange]
            qblock = quote
                @inbounds $(Expr(:block, atemps_init...))
                @inbounds $(Expr(:block, exprs_init...))
                for j = 2:$(sa[2])
                    @inbounds $(Expr(:block, atemps_loop_init...))
                    @inbounds $(Expr(:block, exprs_loop...))
                end
            end
            push!(q.args, qblock)
            m = mu
        end
        n = nu
    end
    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        $q
        @inbounds return similar_type(a, T, $S)(tuple($(tmps...)))
    end
end

@generated function mul_generic(::Size{sa}, ::Size{sb}, wrapped_a::StaticMatMulLike{<:Any, <:Any, Ta}, wrapped_b::StaticMatMulLike{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    return quote
        @_inline_meta
        T = promote_op(matprod, Ta, Tb)
        a = mul_parent(wrapped_a)
        b = mul_parent(wrapped_b)
        return (mul_result_structure(wrapped_a, wrapped_b))(similar_type(a, T, $S)(invoke(*, Tuple{$(_unstatic_array(a)),$(_unstatic_array(b))}, a, b)))
    end
end

# Concatenate a series of matrix-vector multiplications
# Each function is N^2 not N^3 - aids in compile time.
@generated function mul_unrolled_chunks(::Size{sa}, ::Size{sb}, wrapped_a::StaticMatMulLike{<:Any, <:Any, Ta}, wrapped_b::StaticMatMulLike{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    # Do a custom b[:, k2] to return a SVector (an isbitstype type) rather than (possibly) a mutable type. Avoids allocation == faster
    tmp_type_in = :(SVector{$(sb[1]), T})
    tmp_type_out = :(SVector{$(sa[1]), T})

    retexpr = gen_by_access(wrapped_a, wrapped_b) do access_a, access_b
        vect_exprs = [:($(Symbol("tmp_$k2")) = partly_unrolled_multiply($(Size{sa}()), $(Size{(sb[1],)}()),
            a, $(Expr(:call, tmp_type_in, [uplo_access(sb, :b, i, k2, access_b) for i = 1:sb[1]]...)), $(Val(access_a)))::$tmp_type_out) for k2 = 1:sb[2]]

        exprs = [:($(Symbol("tmp_$k2"))[$k1]) for k1 = 1:sa[1], k2 = 1:sb[2]]

        return quote
            @inbounds $(Expr(:block, vect_exprs...))
            $(Expr(:block,
                :(@inbounds return (mul_result_structure(wrapped_a, wrapped_b))(similar_type(a, T, $S)(tuple($(exprs...)))))
            ))
        end
    end
    return quote
        @_inline_meta
        T = promote_op(matprod, Ta, Tb)
        a = mul_parent(wrapped_a)
        b = mul_parent(wrapped_b)
        $retexpr
    end
end

# a special version for plain matrices
@generated function mul_unrolled_chunks(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    # optimal for AVX2 with `Float64
    # AVX512 would want something more like 16x14 or 24x9 with `Float64`
    M_r, N_r = 8, 6
    n = 0
    M, K = sa
    N = sb[2]
    q = Expr(:block)
    atemps = [Symbol(:a_, k1) for k1 = 1:M]
    tmps = [Symbol("tmp_$(k1)_$(k2)") for k1 = 1:M, k2 = 1:N]
    while n < N
        nu = min(N, n + N_r)
        nrange = n+1:nu
        m = 0
        while m < M
            mu = min(M, m + M_r)
            mrange = m+1:mu

            atemps_init = [:($(atemps[k1]) = a[$k1]) for k1 = mrange]
            exprs_init = [:($(tmps[k1,k2])  = $(atemps[k1]) * b[$(1 + (k2-1) * sb[1])]) for k1 = mrange, k2 = nrange]
            push!(q.args, :(@inbounds $(Expr(:block, atemps_init...))))
            push!(q.args, :(@inbounds $(Expr(:block, exprs_init...))))

            for j in 2:K
                atemps_loop_init = [:($(atemps[k1]) = a[$(LinearIndices(sa)[k1,j])]) for k1 = mrange]
                exprs_loop = [:($(tmps[k1,k2]) = muladd($(atemps[k1]), b[$(LinearIndices(sb)[j,k2])], $(tmps[k1,k2]))) for k1 = mrange, k2 = nrange]
                push!(q.args, :(@inbounds $(Expr(:block, atemps_loop_init...))))
                push!(q.args, :(@inbounds $(Expr(:block, exprs_loop...))))
            end
            m = mu
        end
        n = nu
    end
    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        $q
        @inbounds return similar_type(a, T, $S)(tuple($(tmps...)))
    end
end

#
