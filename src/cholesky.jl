# Generic Cholesky decomposition for fixed-size matrices, mostly unrolled
non_hermitian_error() = throw(LinearAlgebra.PosDefException(-1))
@inline function LinearAlgebra.cholesky(A::StaticMatrix; check=true)
    if ishermitian(A)
        return cholesky(Hermitian(A), check=check)
    else   # skip factorization if matrix isn't positive definite
        C = A
        loc = -1
        check && non_hermitian_error()
    end
    return Cholesky(C, 'U', loc)
end

@inline function LinearAlgebra.cholesky(A::LinearAlgebra.RealHermSymComplexHerm{<:Number, <:StaticMatrix}; check=true)
    C = _cholesky(Size(A), A.data)
    loc = _checkpsd(C)
    if check && loc != 0
        throw(PosDefException(loc))
    end
    return Cholesky(C, 'U', loc)
end

"""
    _checkpsd(C::StaticMatrix{M,M,T})

Checks if an upper-triangular Cholesky factor `C` was successfully factored.
If the original matrix `A = C'C` was positive definite, return 0. 
Otherwise, return the location of the first zero element on the diagonal of `C`.

Used to fill in the `info` field of the `LinearAlgebra.Cholesky` type.
"""
function _checkpsd(C::StaticMatrix{M,M,T}) where {M,T}
    ispsd = abs(prod(diag(C))) > zero(real(eltype(C))) 
    loc = ispsd ? 0 : findfirst(diag(C) .== zero(eltype(C)))
    return loc
end

@inline LinearAlgebra._chol!(A::StaticMatrix, ::Type{UpperTriangular}) = (cholesky(A).U, 0)

@generated function _cholesky(::Size{S}, A::StaticMatrix{M,M,TM}) where {S,M,TM}
    @assert (M,M) == S
    M > 24 && return :(_cholesky_large(Size{$S}(), A))
    q = Expr(:block)
    push!(q.args, :(T = promote_type(typeof(sqrt(one(eltype(A)))), Float32)))
    for n ∈ 1:M
        for m ∈ n:M
            L_m_n = Symbol(:L_,m,:_,n)
            push!(q.args, :($L_m_n = @inbounds A[$n, $m]))
        end
        for k ∈ 1:n-1, m ∈ n:M
            L_m_n = Symbol(:L_,m,:_,n)
            L_m_k = Symbol(:L_,m,:_,k)
            L_n_k = Symbol(:L_,n,:_,k)
            push!(q.args, :($L_m_n = muladd(-$L_m_k, $L_n_k', $L_m_n)))
        end
        L_n_n = Symbol(:L_,n,:_,n)
        push!(q.args, :($L_n_n = real($L_n_n) > 0 ? sqrt($L_n_n) : zero(T)))
        Linv_n_n = Symbol(:Linv_,n,:_,n)
        push!(q.args, :($Linv_n_n = inv($L_n_n)))
        for m ∈ n+1:M
            L_m_n = Symbol(:L_,m,:_,n)
            push!(q.args, :($L_m_n *= $Linv_n_n))
        end
    end
    ret = Expr(:tuple)
    for n ∈ 1:M
        for m ∈ 1:n
            push!(ret.args, Symbol(:L_,n,:_,m))
        end
        for m ∈ n+1:M
            push!(ret.args, :(zero(T)))
        end
    end
    push!(q.args, :(similar_type(A, T)($ret)))
    return Expr(:block, Expr(:meta, :inline), q)
end

# Otherwise default algorithm returning wrapped SizedArray
@inline _cholesky_large(::Size{S}, A::StaticArray) where {S} =
    similar_type(A)(cholesky(Hermitian(Matrix(A))).U)

LinearAlgebra.hermitian_type(::Type{SA}) where {T, S, SA<:SArray{S,T}} = Hermitian{T,SA}

function inv(A::Cholesky{T,<:StaticMatrix{N,N,T}}) where {N,T}
    return A.U \ (A.U' \ SDiagonal{N}(I))
end

function Base.:\(A::Cholesky{T,<:StaticMatrix{N,N,T}}, B::StaticVecOrMatLike) where {N,T}
    return A.U \ (A.U' \ B)
end

function Base.:/(B::StaticMatrixLike, A::Cholesky{T,<:StaticMatrix{N,N,T}}) where {N,T}
    return (B / A.U) / A.U'
end

