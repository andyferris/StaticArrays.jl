length(a::StaticArrayLike) = prod(Size(a))
length(a::Type{SA}) where {SA <: StaticArrayLike} = prod(Size(SA))

@pure size(::Type{SA}) where {SA <: StaticArrayLike} = Tuple(Size(SA))
@inline function size(t::Type{<:StaticArrayLike}, d::Int)
    S = size(t)
    d > length(S) ? 1 : S[d]
end
@inline size(a::StaticArrayLike) = Tuple(Size(a))

Base.axes(s::StaticArray) = _axes(Size(s))
@pure function _axes(::Size{sizes}) where {sizes}
    map(SOneTo, sizes)
end
Base.axes(rv::Adjoint{<:Any,<:StaticVector})   = (SOneTo(1), axes(rv.parent)...)
Base.axes(rv::Transpose{<:Any,<:StaticVector}) = (SOneTo(1), axes(rv.parent)...)

Base.eachindex(::IndexLinear, a::StaticArray) = SOneTo(length(a))

# Base.strides is intentionally not defined for SArray, see PR #658 for discussion
Base.strides(a::MArray) = Base.size_to_strides(1, size(a)...)
Base.strides(a::SizedArray) = strides(a.data)

function Base.summary(io::IO, a, inds::Tuple{SOneTo, Vararg{SOneTo}})
    print(io, Base.dims2string(length.(inds)), " ")
    Base.showarg(io, a, true)
end

# This seems to confuse Julia a bit in certain circumstances (specifically for trailing 1's)
@inline function Base.isassigned(a::StaticArray, i::Int...)
    ii = LinearIndices(size(a))[i...]
    1 <= ii <= length(a) ? true : false
end

Base.IndexStyle(::Type{T}) where {T<:StaticArray} = IndexLinear()

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

New types should define the signature `similar_type(::Type{A},::Type{T},::Size{S}) where {A<:MyType,T,S}`
if they wish to overload the default behavior.
"""
function similar_type end

similar_type(::SA) where {SA<:StaticArrayLike} = similar_type(SA,eltype(SA))
similar_type(::Type{SA}) where {SA<:StaticArrayLike} = similar_type(SA,eltype(SA))

similar_type(::SA,::Type{T}) where {SA<:StaticArrayLike,T} = similar_type(SA,T,Size(SA))
similar_type(::Type{SA},::Type{T}) where {SA<:StaticArrayLike,T} = similar_type(SA,T,Size(SA))

similar_type(::A,s::Size{S}) where {A<:AbstractArray,S} = similar_type(A,eltype(A),s)
similar_type(::Type{A},s::Size{S}) where {A<:AbstractArray,S} = similar_type(A,eltype(A),s)

similar_type(::A,::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = similar_type(A,T,s)

# We should be able to deal with SOneTo axes
similar_type(s::SOneTo) = similar_type(typeof(s))
similar_type(::Type{SOneTo{n}}) where {n} = similar_type(SOneTo{n}, Int, Size(n))

similar_type(::A, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray} = similar_type(A, eltype(A), shape)
similar_type(::Type{A}, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray} = similar_type(A, eltype(A), shape)

similar_type(::A,::Type{T}, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray,T} = similar_type(A, T, Size(last.(shape)))
similar_type(::Type{A},::Type{T}, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray,T} = similar_type(A, T, Size(last.(shape)))


# unwrapping arrays (see issue #828)
const SimilarTypeArrayWrapper{T,AW} = Union{
    Transpose{T,AW},
    Adjoint{T,AW},
    Symmetric{T,AW},
    Hermitian{T,AW},
    UpperTriangular{T,AW},
    LowerTriangular{T,AW},
    UnitUpperTriangular{T,AW},
    UnitLowerTriangular{T,AW},
    Diagonal{T,AW}}

function similar_type(::Type{A}, ::Type{T}, shape::Size) where {T,AW<:AbstractArray,A<:SimilarTypeArrayWrapper{<:Any,AW}}
    return similar_type(AW, T, shape)
end

# Default types
# Generally, use SArray
similar_type(::Type{A},::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = default_similar_type(T,s,length_val(s))
default_similar_type(::Type{T}, s::Size{S}, ::Type{Val{D}}) where {T,S,D} = SArray{Tuple{S...},T,D,prod(s)}

similar_type(::Type{SA},::Type{T},s::Size{S}) where {SA<:Union{MVector,MMatrix,MArray},T,S} = mutable_similar_type(T,s,length_val(s))

mutable_similar_type(::Type{T}, s::Size{S}, ::Type{Val{D}}) where {T,S,D} = MArray{Tuple{S...},T,D,prod(s)}

similar_type(::Type{<:SizedArray},::Type{T},s::Size{S}) where {S,T} = sizedarray_similar_type(T,s,length_val(s))
# Should SizedArray also be used for normal Array?
#similar_type(::Type{<:Array},::Type{T},s::Size{S}) where {S,T} = sizedarray_similar_type(T,s,length_val(s))

sizedarray_similar_type(::Type{T},s::Size{S},::Type{Val{D}}) where {T,S,D} = SizedArray{Tuple{S...},T,D,length(s)}

# Utility for computing the eltype of an array instance, type, or type
# constructor.  For type constructors without a definite eltype, the default
# value is returned.
Base.@pure _eltype_or(a::AbstractArray, default) = eltype(a)
Base.@pure _eltype_or(::Type{<:AbstractArray{T}}, default) where {T} = T
Base.@pure _eltype_or(::Type{<:AbstractArray}, default) = default # eltype not available

"""
    _construct_similar(a, ::Size, elements::NTuple)

Construct a static array of similar type to `a` with the given `elements`.

When `a` is an instance or a concrete type the element type `eltype(a)` is
used. However, when `a` is a `UnionAll` type such as `SMatrix{2,2}`, the
promoted type of `elements` is used instead.
"""
@inline function _construct_similar(a, s::Size, elements::Tuple{ET,Vararg{ET,L}}) where {L,ET}
    similar_type(a, _eltype_or(a, ET), s)(elements)
end


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
similar(::SA) where {SA<:StaticArray} = similar(SA,eltype(SA))
similar(::Type{SA}) where {SA<:StaticArray} = similar(SA,eltype(SA))

similar(::SA,::Type{T}) where {SA<:StaticArray,T} = similar(SA,T,Size(SA))
similar(::Type{SA},::Type{T}) where {SA<:StaticArray,T} = similar(SA,T,Size(SA))

# Cases where a Size is given as the dimensions
similar(::A,s::Size{S}) where {A<:AbstractArray,S} = similar(A,eltype(A),s)
similar(::Type{A},s::Size{S}) where {A<:AbstractArray,S} = similar(A,eltype(A),s)

similar(::A,::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = similar(A,T,s)

# defaults to built-in mutable types
similar(::Type{A},::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = mutable_similar_type(T,s,length_val(s))(undef)

# both SizedArray and Array return SizedArray
similar(::Type{SA},::Type{T},s::Size{S}) where {SA<:SizedArray,T,S} = sizedarray_similar_type(T,s,length_val(s))(undef)
similar(::Type{A},::Type{T},s::Size{S}) where {A<:Array,T,S} = sizedarray_similar_type(T,s,length_val(s))(undef)

# Support tuples of mixtures of `SOneTo`s alongside the normal `Integer` and `OneTo` options
# by simply converting them to either a tuple of Ints or a Size, re-dispatching to either one
# of the above methods (in the case of Size) or a base fallback (in the case of Ints).
const HeterogeneousBaseShape = Union{Integer, Base.OneTo}
const HeterogeneousShape = Union{Integer, Base.OneTo, SOneTo}
const HeterogeneousShapeTuple = Union{
    Tuple{SOneTo, Vararg{HeterogeneousShape}},
    Tuple{HeterogeneousBaseShape, SOneTo, Vararg{HeterogeneousShape}},
    Tuple{HeterogeneousBaseShape, HeterogeneousBaseShape, SOneTo, Vararg{HeterogeneousShape}}
}

similar(A::AbstractArray, ::Type{T}, shape::HeterogeneousShapeTuple) where {T} = similar(A, T, homogenize_shape(shape))
similar(::Type{A}, shape::HeterogeneousShapeTuple) where {A<:AbstractArray} = similar(A, homogenize_shape(shape))
# Use an Array for StaticArrays if we don't have a statically-known size
similar(::Type{A}, shape::Tuple{Int, Vararg{Int}}) where {A<:StaticArray} = Array{eltype(A)}(undef, shape)

homogenize_shape(::Tuple{}) = ()
homogenize_shape(shape::Tuple{Vararg{SOneTo}}) = Size(map(last, shape))
homogenize_shape(shape::Tuple{Vararg{HeterogeneousShape}}) = map(last, shape)


@inline reshape(a::StaticArray, s::Size) = similar_type(a, s)(Tuple(a))
@inline reshape(a::AbstractArray, s::Size) = _reshape(a, IndexStyle(a), s)
@inline reshape(a::StaticArray, s::Tuple{SOneTo,Vararg{SOneTo}}) = reshape(a, homogenize_shape(s))
@generated function _reshape(a::AbstractArray, indexstyle, s::Size{S}) where {S}
    if indexstyle == IndexLinear
        exprs = [:(a[$i]) for i = 1:prod(S)]
    else
        exprs = [:(a[$(inds)]) for inds ∈ CartesianIndices(S)]
    end

    return quote
        @_inline_meta
        if length(a) != prod(s)
            throw(DimensionMismatch("Tried to resize dynamic object of size $(size(a)) to $s"))
        end
        return similar_type(a, s)(tuple($(exprs...)))
    end
end

reshape(a::Array, ::Size{S}) where {S} = SizedArray{Tuple{S...}}(a)

@inline vec(a::StaticArray) = reshape(a, Size(prod(Size(typeof(a)))))

@inline copy(a::StaticArray) = typeof(a)(Tuple(a))
@inline copy(a::SizedArray) = typeof(a)(copy(a.data))

@inline reverse(v::StaticVector) = typeof(v)(_reverse(v))

@generated function _reverse(v::StaticVector{N,T}) where {N,T}
    return Expr(:tuple, (:(v[$i]) for i = N:(-1):1)...)
end

@generated function Base.rot180(A::SMatrix{M,N}) where {M,N}
    exs = rot180([:(getindex(A,$i,$j)) for i in 1:M, j in 1:N])
    return :(SMatrix{M,N}($(exs...)))
end
for rot in [:rotl90, :rotr90]
    @eval @generated function Base.$rot(A::SMatrix{M,N}) where {M,N}
        exs = $rot([:(getindex(A,$i,$j)) for i in 1:M, j in 1:N])
        return :(SMatrix{N,M}($(exs...)))
    end
end

# TODO permutedims?

#--------------------------------------------------
# Concatenation
@inline vcat(a::StaticMatrixLike) = a
@inline vcat(a::StaticVecOrMatLike, b::StaticVecOrMatLike) = _vcat(Size(a), Size(b), a, b)
@inline vcat(a::StaticVecOrMatLike, b::StaticVecOrMatLike, c::StaticVecOrMatLike...) = vcat(vcat(a,b), vcat(c...))
# A couple of hacky overloads to avoid some vcat surprises.
# We can't really make this work a lot better in general without Base providing
# a dispatch mechanism for output container type.
@inline vcat(a::StaticVector) = a
@inline vcat(a::StaticVector, bs::Number...) = vcat(a, SVector(bs))
@inline vcat(a::Number, b::StaticVector) = vcat(similar_type(b, typeof(a), Size(1))((a,)), b)

@generated function _vcat(::Size{Sa}, ::Size{Sb}, a::StaticVecOrMatLike, b::StaticVecOrMatLike) where {Sa, Sb}
    if Size(Sa)[2] != Size(Sb)[2]
        return :(throw(DimensionMismatch("Tried to vcat arrays of size $Sa and $Sb")))
    end

    # TODO cleanup?
    if a <: StaticVector && b <: StaticVector
        Snew = (Sa[1] + Sb[1],)
        exprs = vcat([:(a[$i]) for i = 1:Sa[1]],
                     [:(b[$i]) for i = 1:Sb[1]])
    else
        Snew = (Sa[1] + Sb[1], Size(Sa)[2])
        exprs = [((i <= size(a,1)) ? ((a <: StaticVector) ? :(a[$i]) : :(a[$i,$j]))
                                   : ((b <: StaticVector) ? :(b[$(i-size(a,1))]) : :(b[$(i-size(a,1)),$j])))
                                   for i = 1:(Sa[1]+Sb[1]), j = 1:Size(Sa)[2]]
    end

    return quote
        @_inline_meta
        @inbounds return similar_type(a, promote_type(eltype(a), eltype(b)), Size($Snew))(tuple($(exprs...)))
    end
end

@inline hcat(a::StaticVector) = similar_type(a, Size(Size(a)[1],1))(a)
@inline hcat(a::StaticMatrixLike) = a
@inline hcat(a::StaticVecOrMatLike, b::StaticVecOrMatLike) = _hcat(Size(a), Size(b), a, b)
@inline hcat(a::StaticVecOrMatLike, b::StaticVecOrMatLike, c::StaticVecOrMatLike...) = hcat(hcat(a,b), hcat(c...))
@inline hcat(a::StaticMatrix{1}) = a # disambiguation
@inline hcat(a::StaticMatrix{1}, bs::Number...) = hcat(a, SMatrix{1,length(bs)}(bs))
@inline hcat(a::Number, b::StaticMatrix{1}) = hcat(similar_type(b, typeof(a), Size(1))((a,)), b)

@generated function _hcat(::Size{Sa}, ::Size{Sb}, a::StaticVecOrMatLike, b::StaticVecOrMatLike) where {Sa, Sb}
    if Sa[1] != Sb[1]
        return :(throw(DimensionMismatch("Tried to hcat arrays of size $Sa and $Sb")))
    end

    exprs = vcat([:(a[$i]) for i = 1:prod(Sa)],
                 [:(b[$i]) for i = 1:prod(Sb)])

    Snew = (Sa[1], Size(Sa)[2] + Size(Sb)[2])

    return quote
        @_inline_meta
        @inbounds return similar_type(a, promote_type(eltype(a), eltype(b)), Size($Snew))(tuple($(exprs...)))
    end
end

if VERSION >= v"1.6.0-DEV.1334"
    # FIXME: This always assumes one-based linear indexing and that subtypes of StaticArray
    # don't overload iterate
    @inline function Base.rest(a::StaticArray{S}, (_, i) = (nothing, 0)) where {S}
        newlen = tuple_prod(S) - i
        return similar_type(typeof(a), Size(newlen))(Base.rest(Tuple(a), i + 1))
    end
end
