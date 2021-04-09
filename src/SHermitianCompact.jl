"""
    SHermitianCompact{N, T, L} <: StaticMatrix{N, N, T}

A [`StaticArray`](@ref) subtype that represents a Hermitian matrix. Unlike
`LinearAlgebra.Hermitian`, `SHermitianCompact` stores only the lower triangle
of the matrix (as an `SVector`). The lower triangle is stored in column-major order.
For example, for an `SHermitianCompact{3}`, the indices of the stored elements
can be visualized as follows:

```
┌ 1 ⋅ ⋅ ┐
| 2 4 ⋅ |
└ 3 5 6 ┘
```

Type parameters:
* `N`: matrix dimension;
* `T`: element type for lower triangle;
* `L`: length of the `SVector` storing the lower triangular elements.

Note that `L` is always the `N`th [triangular number](https://en.wikipedia.org/wiki/Triangular_number).

An `SHermitianCompact` may be constructed either:

* from an `AbstractVector` containing the lower triangular elements; or
* from a `Tuple` containing both upper and lower triangular elements in column major order; or
* from another `StaticMatrix`.

For the latter two cases, only the lower triangular elements are used; the upper triangular
elements are ignored.
"""
struct SHermitianCompact{N, T, L} <: StaticMatrix{N, N, T}
    lowertriangle::SVector{L, T}

    @inline function SHermitianCompact{N, T, L}(lowertriangle::SVector{L}) where {N, T, L}
        _check_hermitian_parameters(Val(N), Val(L))
        new{N, T, L}(lowertriangle)
    end
end

@inline function _check_hermitian_parameters(::Val{N}, ::Val{L}) where {N, L}
    if 2 * L !== N * (N + 1)
        throw(ArgumentError("Size mismatch in SHermitianCompact parameters. Got dimension $N and length $L."))
    end
end

Base.@pure triangularnumber(N::Int) = div(N * (N + 1), 2)
Base.@pure triangularroot(L::Int) = div(isqrt(8 * L + 1) - 1, 2) # from quadratic formula

lowertriangletype(::Type{SHermitianCompact{N, T, L}}) where {N, T, L} = SVector{L, T}
lowertriangletype(::Type{SHermitianCompact{N, T}}) where {N, T} = SVector{triangularnumber(N), T}
lowertriangletype(::Type{SHermitianCompact{N}}) where {N} = SVector{triangularnumber(N)}

@inline SHermitianCompact{N, T}(lowertriangle::SVector{L}) where {N, T, L} = SHermitianCompact{N, T, L}(lowertriangle)
@inline SHermitianCompact{N}(lowertriangle::SVector{L, T}) where {N, T, L} = SHermitianCompact{N, T, L}(lowertriangle)

@inline function SHermitianCompact(lowertriangle::SVector{L, T}) where {T, L}
    N = triangularroot(L)
    SHermitianCompact{N, T, L}(lowertriangle)
end

@generated function SHermitianCompact{N, T, L}(a::Tuple) where {N, T, L}
    expr = Vector{Expr}(undef, L)
    i = 0
    for col = 1 : N, row = col : N
        index = N * (col - 1) + row
        expr[i += 1] = :(a[$index])
    end
    quote
        @_inline_meta
        @inbounds return SHermitianCompact{N, T, L}(SVector{L, T}(tuple($(expr...))))
    end
end

@inline function SHermitianCompact{N, T}(a::Tuple) where {N, T}
    L = triangularnumber(N)
    SHermitianCompact{N, T, L}(a)
end

@inline SHermitianCompact{N}(a::Tuple) where {N} = SHermitianCompact{N, promote_tuple_eltype(a)}(a)
@inline SHermitianCompact{N}(a::Tuple{T,Vararg{T,M}}) where {N, T, M} = SHermitianCompact{N, T}(a)
@inline SHermitianCompact(a::StaticMatrix{N, N, T}) where {N, T} = SHermitianCompact{N, T}(a)

@inline (::Type{SSC})(a::SHermitianCompact) where {SSC <: SHermitianCompact} = SSC(a.lowertriangle)
@inline (::Type{SSC})(a::SSC) where {SSC <: SHermitianCompact} = a

@inline (::Type{SSC})(a::AbstractVector) where {SSC <: SHermitianCompact} = SSC(convert(lowertriangletype(SSC), a))

@generated function _hermitian_compact_indices(::Val{N}) where N
    # Returns a Tuple{Pair{Int, Bool}} I such that for linear index i,
    # * I[i][1] is the index into the lowertriangle field of an SHermitianCompact{N};
    # * I[i][2] is true iff i is an index into the lower triangle of an N × N matrix.
    indexmat = Matrix{Pair{Int, Bool}}(undef, N, N)
    i = 0
    for col = 1 : N, row = 1 : N
        indexmat[row, col] = if row >= col
            (i += 1) => true
        else
            indexmat[col, row][1] => false
        end
    end
    quote
        Base.@_inline_meta
        return $(tuple(indexmat...))
    end
end

Base.@propagate_inbounds function Base.getindex(a::SHermitianCompact{N}, i::Int) where {N}
    I = _hermitian_compact_indices(Val(N))
    j, lower = I[i]
    @inbounds value = a.lowertriangle[j]
    return lower ? value : adjoint(value)
end

Base.@propagate_inbounds function Base.setindex(a::SHermitianCompact{N, T, L}, x, i::Int) where {N, T, L}
    I = _hermitian_compact_indices(Val(N))
    j, lower = I[i]
    value = lower ? x : adjoint(x)
    return SHermitianCompact{N}(setindex(a.lowertriangle, value, j))
end

# needed because it is used in convert.jl and the generic fallback is slow
@generated function Base.Tuple(a::SHermitianCompact{N}) where N
    exprs = [:(a[$i]) for i = 1 : N^2]
    quote
        @_inline_meta
        tuple($(exprs...))
    end
end

LinearAlgebra.ishermitian(a::SHermitianCompact) = true
LinearAlgebra.issymmetric(a::SHermitianCompact) = true

# TODO: factorize?

@inline Base.:(==)(a::SHermitianCompact, b::SHermitianCompact) = a.lowertriangle == b.lowertriangle
@generated function _map(f, a::SHermitianCompact...)
    S = Size(a[1])
    N = S[1]
    L = triangularnumber(N)
    exprs = Vector{Expr}(undef, L)
    for i ∈ 1:L
        tmp = [:(a[$j].lowertriangle[$i]) for j ∈ 1:length(a)]
        exprs[i] = :(f($(tmp...)))
    end
    return quote
        @_inline_meta
        same_size(a...)
        @inbounds return SHermitianCompact(SVector(tuple($(exprs...))))
    end
end

@inline Base.:*(a::Number, b::SHermitianCompact) = SHermitianCompact(a * b.lowertriangle)
@inline Base.:*(a::SHermitianCompact, b::Number) = SHermitianCompact(a.lowertriangle * b)

@inline Base.:/(a::SHermitianCompact, b::Number) = SHermitianCompact(a.lowertriangle / b)
@inline Base.:\(a::Number, b::SHermitianCompact) = SHermitianCompact(a \ b.lowertriangle)

@generated function _plus_uniform(::Size{S}, a::SHermitianCompact{N, T, L}, λ) where {S, N, T, L}
    @assert S[1] == N
    @assert S[2] == N
    exprs = Vector{Expr}(undef, L)
    i = 0
    for col = 1 : N, row = col : N
        i += 1
        exprs[i] = row == col ? :(a.lowertriangle[$i] + λ) : :(a.lowertriangle[$i])
    end
    return quote
        @_inline_meta
        R = promote_type(eltype(a), typeof(λ))
        SHermitianCompact{N, R, L}(SVector{L, R}(tuple($(exprs...))))
    end
end

@generated function LinearAlgebra.transpose(a::SHermitianCompact{N, T, L}) where {N, T, L}
    # To conform with LinearAlgebra, the transpose should be recursive.
    # For this compact representation of a Hermitian matrix, that means that
    # we should recursively transpose of the diagonal elements, but only
    # conjugate the off-diagonal elements:
    # [A  Bᴴ]ᵀ  =  [Aᵀ  Bᵀ]  =  [Aᵀ      Bᵀ]
    # [B  C ]      [Bᴴᵀ Cᵀ]     [conj(B) Cᵀ]
    exprs = Vector{Expr}(undef, L)
    i = 0
    for col = 1 : N, row = col : N
        i += 1
        exprs[i] = row == col ? :(transpose(a.lowertriangle[$i])) : :(conj(a.lowertriangle[$i]))
    end
    return quote
        @_inline_meta
        SHermitianCompact{N}(SVector{L}(tuple($(exprs...))))
    end
end

@generated function LinearAlgebra.adjoint(a::SHermitianCompact{N, T, L}) where {N, T, L}
    # To conform with LinearAlgebra, the adjoint should be recursive.
    # Taking the adjoint of a Hermitian matrix is the identity, but
    # with this compact representation of a Hermitian matrix, care
    # should be taken that only the diagonal elements should be
    # recursively conjugate-transposed; the off-diagonal elements should
    # not:
    # [A  Bᴴ]ᴴ  =  [Aᴴ  Bᴴ]  =  [Aᴴ Bᴴ]
    # [B  C ]      [Bᴴᴴ Cᴴ]     [B  Cᴴ]
    exprs = Vector{Expr}(undef, L)
    i = 0
    for col = 1 : N, row = col : N
        i += 1
        exprs[i] = row == col ? :(adjoint(a.lowertriangle[$i])) : :(a.lowertriangle[$i])
    end
    return quote
        @_inline_meta
        SHermitianCompact{N}(SVector{L}(tuple($(exprs...))))
    end
end

@generated function _one(::Size{S}, ::Type{SSC}) where {S, SSC <: SHermitianCompact}
    N = S[1]
    L = triangularnumber(N)
    T = eltype(SSC)
    if T == Any
        T = Float64
    end
    exprs = Vector{Expr}(undef, L)
    i = 0
    for col = 1 : N, row = col : N
        exprs[i += 1] = row == col ? :(one($T)) : :(zero($T))
    end
    quote
        @_inline_meta
        return SHermitianCompact(SVector(tuple($(exprs...))))
    end
end

@inline _scalar_matrix(s::Size{S}, t::Type{SSC}) where {S, SSC <: SHermitianCompact} = _one(s, t)

# _fill covers fill, zeros, and ones:
@generated function _fill(val, ::Size{s}, ::Type{SSC}) where {s, SSC <: SHermitianCompact}
    N = s[1]
    L = triangularnumber(N)
    v = [:val for i = 1:L]
    return quote
        @_inline_meta
        $SSC(SVector(tuple($(v...))))
    end
end

@generated function _rand(randfun, rng::AbstractRNG, ::Type{SSC}) where {N, SSC <: SHermitianCompact{N}}
    T = eltype(SSC)
    if T == Any
        T = Float64
    end
    L = triangularnumber(N)
    v = [:(randfun(rng, $T)) for i = 1:L]
    return quote
        @_inline_meta
        $SSC(SVector(tuple($(v...))))
    end
end

@inline Random.rand(rng::AbstractRNG, ::Type{SSC}) where {SSC <: SHermitianCompact} = _rand(rand, rng, SSC)
@inline Random.randn(rng::AbstractRNG, ::Type{SSC}) where {SSC <: SHermitianCompact} = _rand(randn, rng, SSC)
@inline Random.randexp(rng::AbstractRNG, ::Type{SSC}) where {SSC <: SHermitianCompact} = _rand(randexp, rng, SSC)
