length(a::SA) where {SA <: StaticArray} = prod(Size(SA))
length(a::Type{SA}) where {SA <: StaticArray} = prod(Size(SA))

@pure size(::Type{<:StaticArray{S}}) where S = tuple(S.parameters...)
@inline function size(t::Type{<:StaticArray}, d::Int)
    S = size(t)
    d > length(S) ? 1 : S[d]
end
@inline size(a::StaticArray) = size(typeof(a))
@inline size(a::StaticArray, d::Int) = size(typeof(a), d)

# This seems to confuse Julia a bit in certain circumstances (specifically for trailing 1's)
@inline function Base.isassigned(a::StaticArray, i::Int...)
    ii = sub2ind(size(a), i...)
    1 <= ii <= length(a) ? true : false
end

Base.IndexStyle{T<:StaticArray}(::Type{T}) = IndexLinear()

@inline indices(a::StaticArray) = _indices(Size(a))
@inline indices(::Type{T}) where {T <: StaticArray} = _indices(Size(T))

@pure function _indices(::Size{S}) where {S}
    return map(s -> SUnitRange(1, s), S)
end

# Default type search for similar_type
"""
    similar_type(static_array)
    similar_type(static_array, T)
    similar_type(array, ::Size)
    similar_type(array, T, ::Size)

Returns a constructor for a statically-sized array similar to the input array
(or type) `static_array`/`array`, optionally with different element type `T` or size
`Size`. If the input `array` is not a `StaticArray` then the `Size` is mandatory.

This differs from `similar()` in that the resulting array type may not be
mutable (or define `setindex!()`), and therefore the returned type may need to
be *constructed* with its data.

Note that the (optional) size *must* be specified as a static `Size` object (so the compiler
can infer the result statically).

New types should define the signature `similar_type{A<:MyType,T,S}(::Type{A},::Type{T},::Size{S})`
if they wish to overload the default behavior.
"""
function similar_type end

similar_type(::SA) where {SA<:StaticArray} = similar_type(SA,eltype(SA))
similar_type(::Type{SA}) where {SA<:StaticArray} = similar_type(SA,eltype(SA))

similar_type(::SA,::Type{T}) where {SA<:StaticArray,T} = similar_type(SA,T,Size(SA))
similar_type(::Type{SA},::Type{T}) where {SA<:StaticArray,T} = similar_type(SA,T,Size(SA))

similar_type(::A,s::Size{S}) where {A<:AbstractArray,S} = similar_type(A,eltype(A),s)
similar_type(::Type{A},s::Size{S}) where {A<:AbstractArray,S} = similar_type(A,eltype(A),s)

similar_type(::A,::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = similar_type(A,T,s)

# Default types
# Generally, use SArray
similar_type{A<:AbstractArray,T,S}(::Type{A},::Type{T},s::Size{S}) = default_similar_type(T,s,length_val(s))
default_similar_type{T,S,D}(::Type{T}, s::Size{S}, ::Type{Val{D}}) = SArray{Tuple{S...},T,D,prod(s)}


# should mutable things stay mutable?
#similar_type{SA<:Union{MVector,MMatrix,MArray},T,S}(::Type{SA},::Type{T},s::Size{S}) = mutable_similar_type(T,s,length_val(s))

mutable_similar_type{T,S,D}(::Type{T}, s::Size{S}, ::Type{Val{D}}) = MArray{Tuple{S...},T,D,prod(s)}

# Should `SizedArray` stay the same, and also take over an `Array`?
#similar_type{SA<:SizedArray,T,S}(::Type{SA},::Type{T},s::Size{S}) = sizedarray_similar_type(T,s,length_val(s))
#similar_type{A<:Array,T,S}(::Type{A},::Type{T},s::Size{S}) = sizedarray_similar_type(T,s,length_val(s))

sizedarray_similar_type{T,S,D}(::Type{T},s::Size{S},::Type{Val{D}}) = SizedArray{Tuple{S...},T,D,D}

# Field vectors are user controlled, and currently default to SVector, etc

"""
    similar(static_array)
    similar(static_array, T)
    similar(array, ::Size)
    similar(array, T, ::Size)

Constructs and returns a mutable but statically-sized array (i.e. a `StaticArray`). If the
input `array` is not a `StaticArray`, then the `Size` is required to determine the output
size (or else a dynamically sized array will be returned).
"""
similar{SA<:StaticArray}(::SA) = similar(SA,eltype(SA))
similar{SA<:StaticArray}(::Type{SA}) = similar(SA,eltype(SA))

similar{SA<:StaticArray,T}(::SA,::Type{T}) = similar(SA,T,Size(SA))
similar{SA<:StaticArray,T}(::Type{SA},::Type{T}) = similar(SA,T,Size(SA))

similar{A<:AbstractArray,S}(::A,s::Size{S}) = similar(A,eltype(A),s)
similar{A<:AbstractArray,S}(::Type{A},s::Size{S}) = similar(A,eltype(A),s)

similar{A<:AbstractArray,T,S}(::A,::Type{T},s::Size{S}) = similar(A,T,s)

# defaults to built-in mutable types
similar{A<:AbstractArray,T,S}(::Type{A},::Type{T},s::Size{S}) = mutable_similar_type(T,s,length_val(s))()

# both SizedArray and Array return SizedArray
similar{SA<:SizedArray,T,S}(::Type{SA},::Type{T},s::Size{S}) = sizedarray_similar_type(T,s,length_val(s))()
similar{A<:Array,T,S}(::Type{A},::Type{T},s::Size{S}) = sizedarray_similar_type(T,s,length_val(s))()


@inline reshape(a::StaticArray, s::Size) = similar_type(a, s)(Tuple(a))
@inline reshape(a::AbstractArray, s::Size) = _reshape(s, IndexStyle(a), a)
@generated function _reshape(a::AbstractArray, indexstyle, s::Size{S}) where {S}
    if indexstyle == IndexLinear
        exprs = [:(a[$i]) for i = 1:prod(S)]
    else
        exprs = [:(a[$(inds...)]) for inds ∈ CartesianRange(S)]
    end

    return quote
        @_inline_meta
        if length(a) != prod(s)
            throw(DimensionMismatch("Tried to resize dynamic object of size $(size(a)) to $s"))
        end
        return similar_type(a, s)(tuple($(exprs...)))
    end
end

reshape(a::Array, s::Size{S}) where {S} = s(a)

@inline vec(a::StaticArray) = reshape(a, Size(prod(Size(typeof(a)))))

@inline copy(a::StaticArray) = similar_type(a)(Tuple(a))

# TODO permutedims?

# TODO perhaps could move `Symmetric`, etc into seperate files.

# This is used in Base.LinAlg quite a lot, and it impacts type stability
# since some functions like expm() branch on a check for Hermitian or Symmetric
# TODO much more work on type stability. Base functions are using similar() with
# size, which poses difficulties!!
@inline Base.full(sym::Symmetric{T,SM}) where {T,SM <: StaticMatrix} = _full(Size(SM), sym)

@generated function _full(::Size{S}, sym::Symmetric{T,SM}) where {S, T, SM <: StaticMatrix}
    exprs_up = [i <= j ? :(m[$(sub2ind(S, i, j))]) : :(m[$(sub2ind(S, j, i))]) for i = 1:S[1], j=1:S[2]]
    exprs_lo = [i >= j ? :(m[$(sub2ind(S, i, j))]) : :(m[$(sub2ind(S, j, i))]) for i = 1:S[1], j=1:S[2]]

    return quote
        @_inline_meta
        m = sym.data
        if sym.uplo == 'U'
            @inbounds return SM(tuple($(exprs_up...)))
        else
            @inbounds return SM(tuple($(exprs_lo...)))
        end
    end
end

@inline Base.full(herm::Hermitian{T,SM}) where {T,SM <: StaticMatrix} = _full(Size(SM), herm)

@generated function _full(::Size{S}, herm::Hermitian{T,SM}) where {S, T, SM <: StaticMatrix}
    exprs_up = [i <= j ? :(m[$(sub2ind(S, i, j))]) : :(conj(m[$(sub2ind(S, j, i))])) for i = 1:S[1], j=1:S[2]]
    exprs_lo = [i >= j ? :(m[$(sub2ind(S, i, j))]) : :(conj(m[$(sub2ind(S, j, i))])) for i = 1:S[1], j=1:S[2]]

    return quote
        @_inline_meta
        m = herm.data
        if herm.uplo == 'U'
            @inbounds return SM(tuple($(exprs_up...)))
        else
            @inbounds return SM(tuple($(exprs_lo...)))
        end
    end
end
