# Default error messages to help users with new types and to avoid subsequent stack overflows
getindex(a::StaticArray, i::Int) = error("getindex(::$(typeof(a)), ::Int) is not defined.")
setindex!(a::StaticArray, value, i::Int) = error("setindex!(::$(typeof(a)), value, ::Int) is not defined.")

#######################################
## Multidimensional scalar indexing  ##
#######################################

# Note: all indexing behavior defaults to dense, linear indexing

@propagate_inbounds function getindex(a::StaticArray, inds::Int...)
    @boundscheck checkbounds(a, inds...)
    _getindex_scalar(Size(a), a, inds...)
end

@generated function _getindex_scalar(::Size{S}, a::StaticArray, inds::Int...) where S
    if length(inds) == 0
        return quote
            @_propagate_inbounds_meta
            a[1]
        end
    end

    stride = 1
    ind_expr = :()
    for i ∈ 1:length(inds)
        if i == 1
            ind_expr = :(inds[1])
        else
            ind_expr = :($ind_expr + $stride * (inds[$i] - 1))
        end
        stride *= Size(S)[i]
    end
    return quote
        @_propagate_inbounds_meta
        a[$ind_expr]
    end
end

@propagate_inbounds function setindex!(a::StaticArray, value, inds::Int...)
    @boundscheck checkbounds(a, inds...)
    _setindex!_scalar(Size(a), a, value, inds...)
    return a
end

@generated function _setindex!_scalar(::Size{S}, a::StaticArray, value, inds::Int...) where S
    if length(inds) == 0
        return quote
            @_propagate_inbounds_meta
            a[1] = value
        end
    end

    stride = 1
    ind_expr = :()
    for i ∈ 1:length(inds)
        if i == 1
            ind_expr = :(inds[1])
        else
            ind_expr = :($ind_expr + $stride * (inds[$i] - 1))
        end
        stride *= S[i]
    end
    return quote
        @_propagate_inbounds_meta
        a[$ind_expr] = value
    end
end

#########################
## Indexing utilities  ##
#########################

@pure unpack_size(::Type{Size{S}}) where {S} = map(Size, S)

@inline index_size(::Size, ::Int) = Size()
@inline index_size(::Size, a::StaticArray) = Size(a)
@inline index_size(s::Size, ::Colon) = s
@inline index_size(s::Size, a::SOneTo{n}) where n = Size(n,)
if isdefined(Base, :IdentityUnitRange)
    @inline index_size(s::Size, a::Base.IdentityUnitRange{SOneTo{n}}) where n = Size(n,)
end

@inline index_sizes(::S, inds...) where {S<:Size} = map(index_size, unpack_size(S), inds)
@inline index_sizes(::S, inds) where {S<:Size} = map(index_size, map(Size, linear_index_size(S)), (inds,))

@inline index_sizes() = ()
@inline index_sizes(::Int, inds...) = (Size(), index_sizes(inds...)...)
@inline index_sizes(a::StaticArray, inds...) = (Size(a), index_sizes(inds...)...)

out_index_size(ind_sizes::Type{<:Size}...) = Size(_out_index_size((), ind_sizes...))
@inline _out_index_size(t::Tuple) = t
@inline _out_index_size(t::Tuple, ::Type{Size{S}}, ind_sizes...) where {S} = _out_index_size((t..., S...), ind_sizes...)

linear_index_size(ind_sizes::Type{<:Size}...) = _linear_index_size((), ind_sizes...)
@inline _linear_index_size(t::Tuple) = t
@inline _linear_index_size(t::Tuple, ::Type{Size{S}}, ind_sizes...) where {S} = _linear_index_size((t..., prod(S)), ind_sizes...)

_ind(i::Int, ::Int, ::Type{Int}) = :(inds[$i])
_ind(i::Int, j::Int, ::Type{<:StaticArray}) = :(inds[$i][$j])
_ind(i::Int, j::Int, ::Type{Colon}) = j
_ind(i::Int, j::Int, ::Type{<:SOneTo}) = j
if isdefined(Base, :IdentityUnitRange)
    _ind(i::Int, j::Int, ::Type{<:Base.IdentityUnitRange{<:SOneTo}}) = j
end

################################
## Non-scalar linear indexing ##
################################

@inline function getindex(a::StaticArray, ::Colon)
    _getindex(a::StaticArray, Length(a), :)
end

@generated function _getindex(a::StaticArray, ::Length{L}, ::Colon) where {L}
    exprs = [:(a[$i]) for i = 1:L]
    return quote
        @_inline_meta
        @inbounds return similar_type(a, Size(L))(tuple($(exprs...)))
    end
end

@propagate_inbounds function getindex(a::StaticArray, inds::StaticArray{<:Tuple, Int})
    _getindex(a, Size(inds), inds)
end

@generated function _getindex(a::StaticArray, s::Size{S}, inds::StaticArray{<:Tuple, Int}) where {S}
    exprs = [:(a[inds[$i]]) for i = 1:prod(S)]
    return quote
        @_propagate_inbounds_meta
        similar_type(a, s)(tuple($(exprs...)))
    end
end

@inline function setindex!(a::StaticArray, v, ::Colon)
    _setindex!(a::StaticArray, v, Length(a), :)
    return a
end

@generated function _setindex!(a::StaticArray, v, ::Length{L}, ::Colon) where {L}
    exprs = [:(a[$i] = v) for i = 1:L]
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@generated function _setindex!(a::StaticArray, v::AbstractArray, ::Length{L}, ::Colon) where {L}
    exprs = [:(a[$i] = v[$i]) for i = 1:L]
    return quote
        @_propagate_inbounds_meta
        @boundscheck if length(v) != L
            throw(DimensionMismatch("tried to assign $(length(v))-element array to length-$L destination"))
        end
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@generated function _setindex!(a::StaticArray, v::StaticArray, ::Length{L}, ::Colon) where {L}
    exprs = [:(a[$i] = v[$i]) for i = 1:L]
    return quote
        @_propagate_inbounds_meta
        @boundscheck if Length(typeof(v)) != L
            throw(DimensionMismatch("tried to assign $(length(v))-element array to length-$L destination"))
        end
        $(Expr(:block, exprs...))
        return a
    end
end

@propagate_inbounds function setindex!(a::StaticArray, v, inds::StaticArray{<:Tuple, Int})
    _setindex!(a, v, Size(inds), inds)
    return a
end

@generated function _setindex!(a::StaticArray, v, s::Size{S}, inds::StaticArray{<:Tuple, Int}) where {S}
    exprs = [:(a[inds[$i]] = v) for i = 1:prod(S)]
    return quote
        @_propagate_inbounds_meta
        similar_type(a, s)(tuple($(exprs...)))
        return a
    end
end

@generated function _setindex!(a::StaticArray, v::AbstractArray, s::Size{S}, inds::StaticArray{<:Tuple, Int}) where {S}
    exprs = [:(a[inds[$i]] = v[$i]) for i = 1:prod(S)]
    return quote
        @_propagate_inbounds_meta
        @boundscheck if length(v) != $(prod(S))
            throw(DimensionMismatch("tried to assign $(length(v))-element array to length-$(length(inds)) destination"))
        end
        $(Expr(:block, exprs...))
        return a
    end
end

@generated function _setindex!(a::StaticArray, v::StaticArray, s::Size{S}, inds::StaticArray{<:Tuple, Int}) where {S}
    exprs = [:(a[inds[$i]] = v[$i]) for i = 1:prod(S)]
    return quote
        @_propagate_inbounds_meta
        @boundscheck if Length(typeof(v)) != Length(s)
            throw(DimensionMismatch("tried to assign $(length(v))-element array to length-$(length(inds)) destination"))
        end
        $(Expr(:block, exprs...))
        return a
    end
end

###########################################
## Multidimensional non-scalar indexing  ##
###########################################

# To intercept `A[i1, ...]` where all `i` indexes have static sizes,
# create a wrapper used to mark non-scalar indexing operations.
# We insert this at a point in the dispatch hierarchy where we can intercept any
# `typeof(A)` (specifically, including dynamic arrays) without triggering ambiguities.

struct StaticIndexing{I}
    ind::I
end
unwrap(i::StaticIndexing) = i.ind

function Base.to_indices(A, I::Tuple{Vararg{Union{Integer, CartesianIndex, StaticArray{<:Tuple,Int}}}})
    inds = to_indices(A, axes(A), I)
    return map(StaticIndexing, inds)
end

# getindex

@propagate_inbounds function getindex(a::StaticArray, inds::Union{Int, StaticArray{<:Tuple, Int}, SOneTo, Colon}...)
    ar = reshape(a, Val(length(inds)))
    _getindex(ar, index_sizes(Size(ar), inds...), inds)
end

if isdefined(Base, :IdentityUnitRange)
    @propagate_inbounds function getindex(a::StaticArray, inds::Union{Int, StaticArray{<:Tuple, Int}, SOneTo, Colon, Base.IdentityUnitRange{<:SOneTo}}...)
        ar = reshape(a, Val(length(inds)))
        _getindex(ar, index_sizes(Size(ar), inds...), inds)
    end
end

function Base._getindex(::IndexStyle, A::AbstractArray, i1::StaticIndexing, I::StaticIndexing...)
    inds = (unwrap(i1), map(unwrap, I)...)
    return StaticArrays._getindex(A, index_sizes(inds...), inds)
end


@generated function _getindex(a::AbstractArray, ind_sizes::Tuple{Vararg{Size}}, inds)
    newsize = out_index_size(ind_sizes.parameters...)
    linearsizes = linear_index_size(ind_sizes.parameters...)
    exprs = Array{Expr}(undef, linearsizes)

    # Iterate over input indices
    ind_types = inds.parameters
    current_ind = ones(Int,length(linearsizes))
    more = !isempty(exprs)
    while more
        exprs_tmp = [_ind(i, current_ind[i], ind_types[i]) for i = 1:length(linearsizes)]
        exprs[current_ind...] = :(getindex(a, $(exprs_tmp...)))

        # increment current_ind
        current_ind[1] += 1
        for i ∈ 1:length(linearsizes)
            if current_ind[i] > linearsizes[i]
                if i == length(linearsizes)
                    more = false
                    break
                else
                    current_ind[i] = 1
                    current_ind[i+1] += 1
                end
            else
                break
            end
        end
    end

    quote
        @_propagate_inbounds_meta
        similar_type(a, $newsize)(tuple($(exprs...)))
    end
end

# setindex!

@propagate_inbounds function setindex!(a::StaticArray, value, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...)
    _setindex!(a, value, index_sizes(Size(a), inds...), inds)
end

function Base._setindex!(::IndexStyle, a::AbstractArray, value, i1::StaticIndexing, I::StaticIndexing...)
    inds = (unwrap(i1), map(unwrap, I)...)
    return StaticArrays._setindex!(a, value, index_sizes(inds...), inds)
end

# setindex! from a scalar
@generated function _setindex!(a::AbstractArray, value, ind_sizes::Tuple{Vararg{Size}}, inds)
    linearsizes = linear_index_size(ind_sizes.parameters...)
    exprs = Array{Expr}(undef, linearsizes)

    # Iterate over input indices
    ind_types = inds.parameters
    current_ind = ones(Int,length(ind_types))
    more = !isempty(exprs)
    while more
        exprs_tmp = [_ind(i, current_ind[i], ind_types[i]) for i = 1:length(ind_types)]
        exprs[current_ind...] = :(setindex!(a, value, $(exprs_tmp...)))

        # increment current_ind
        current_ind[1] += 1
        for i ∈ 1:length(linearsizes)
            if current_ind[i] > linearsizes[i]
                if i == length(linearsizes)
                    more = false
                    break
                else
                    current_ind[i] = 1
                    current_ind[i+1] += 1
                end
            else
                break
            end
        end
    end

    quote
        @_propagate_inbounds_meta
        $(exprs...)
        return a
    end
end

# setindex! from an array
@generated function _setindex!(a::AbstractArray, v::AbstractArray, ind_sizes::Tuple{Vararg{Size}}, inds)
    linearsizes = linear_index_size(ind_sizes.parameters...)
    exprs = Array{Expr}(undef, linearsizes)

    # Iterate over input indices
    ind_types = inds.parameters
    current_ind = ones(Int,length(ind_types))
    more = true
    j = 1
    while more
        exprs_tmp = [_ind(i, current_ind[i], ind_types[i]) for i = 1:length(ind_types)]
        exprs[current_ind...] = :(setindex!(a, v[$j], $(exprs_tmp...)))

        # increment current_ind
        current_ind[1] += 1
        for i ∈ 1:length(linearsizes)
            if current_ind[i] > linearsizes[i]
                if i == length(linearsizes)
                    more = false
                    break
                else
                    current_ind[i] = 1
                    current_ind[i+1] += 1
                end
            else
                break
            end
        end
        j += 1
    end

    quote
        @_propagate_inbounds_meta
        if length(v) != $(prod(linearsizes))
            newsize = $linearsizes
            throw(DimensionMismatch("tried to assign $(length(v))-element array to $newsize destination"))
        end
        $(exprs...)
        return a
    end
end

# checkindex

Base.checkindex(B::Type{Bool}, inds::AbstractUnitRange, i::StaticIndexing{T}) where T = Base.checkindex(B, inds, unwrap(i))

# unsafe_view

# unsafe_view need only deal with vargs of `StaticIndexing`, as wrapped by to_indices.
# i1 is explicitly specified to avoid ambiguities with Base
Base.unsafe_view(A::AbstractArray, i1::StaticIndexing, indices::StaticIndexing...) = Base.unsafe_view(A, unwrap(i1), map(unwrap, indices)...)

# Views of views need a new method for Base.SubArray because storing indices
# wrapped in StaticIndexing in field indices of SubArray causes all sorts of problems.
# Additionally, in some cases the SubArray constructor may be called directly
# instead of unsafe_view so we need this method too (Base._maybe_reindex
# is a good example)
# the tuple indices has to have at least one element to prevent infinite
# recursion when viewing a zero-dimensional array (see issue #705)
Base.SubArray(A::AbstractArray, indices::Tuple{StaticIndexing, Vararg{StaticIndexing}}) = Base.SubArray(A, map(unwrap, indices))
