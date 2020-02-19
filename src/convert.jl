(::Type{SA})(x::Tuple{Tuple{Tuple{<:Tuple}}}) where {SA <: StaticArray} =
    throw(DimensionMismatch("No precise constructor for $SA found. Length of input was $(length(x[1][1][1]))."))

@inline (::Type{SA})(x...) where {SA <: StaticArray} = SA(x)
@inline (::Type{SA})(a::StaticArray) where {SA<:StaticArray} = SA(Tuple(a))
@inline (::Type{SA})(a::StaticArray) where {SA<:SizedArray} = SA(a.data)
@propagate_inbounds (::Type{SA})(a::AbstractArray) where {SA <: StaticArray} = convert(SA, a)

# this covers most conversions and "statically-sized reshapes"
@inline convert(::Type{SA}, sa::StaticArray) where {SA<:StaticArray} = SA(Tuple(sa))
@inline convert(::Type{SA}, sa::SA) where {SA<:StaticArray} = sa
@inline convert(::Type{SA}, x::Tuple) where {SA<:StaticArray} = SA(x) # convert -> constructor. Hopefully no loops...

# Constructing a Tuple from a StaticArray
@inline Tuple(a::StaticArray) = unroll_tuple(a, Length(a))

@noinline function dimension_mismatch_fail(SA::Type, a::AbstractArray)
    throw(DimensionMismatch("expected input array of length $(length(SA)), got length $(length(a))"))
end

@propagate_inbounds function convert(::Type{SA}, a::AbstractArray) where {SA <: StaticArray}
    @boundscheck if length(a) != length(SA)
        dimension_mismatch_fail(SA, a)
    end

    return _convert(SA, a, Length(SA))
end

@inline _convert(SA, a, l::Length) = SA(unroll_tuple(a, l))
@inline _convert(SA::Type{<:StaticArray{<:Tuple,T}}, a, ::Length{0}) where T = similar_type(SA, T)(())
@inline _convert(SA, a, ::Length{0}) = similar_type(SA, eltype(a))(())

length_val(a::T) where {T <: StaticArrayLike} = length_val(Size(T))
length_val(a::Type{T}) where {T<:StaticArrayLike} = length_val(Size(T))

@generated function unroll_tuple(a::AbstractArray, ::Length{L}) where {L}
    exprs = [:(a[$j]) for j = 1:L]
    quote
        @_inline_meta
        @inbounds return $(Expr(:tuple, exprs...))
    end
end

# promote() to SArray
@inline Base.promote_rule(::Type{<: SVector{S, T} },
                        y::Type{<: AbstractArray{T} } ) where {S, T} = (@boundscheck length(y) == S || error();
                                                                        SVector{S, T} )
@inline Base.promote_rule(::Type{<: SVector{S, T1} },
                        ::Type{<: AbstractArray{T2} } ) where {S, T1, T2} = SVector{S, promote_type(T1, T2) }

@inline Base.promote_rule(::Type{<: SMatrix{S1, S2, T} },
                        y::Type{<: AbstractArray{T} } ) where {S1, S2, T} = (@boundscheck length(y) == S1 * S2 || error();
                                                                             SMatrix{S1, S2, T} )
@inline Base.promote_rule(::Type{<: SMatrix{S1, S2, T1} },
                        ::Type{<: AbstractArray{T2} } ) where {S1, S2, T1, T2} = SMatrix{S1, S2, promote_type(T1, T2) }

@inline Base.promote_rule(::Type{<: SArray{S, T} },
                        y::Type{<: AbstractArray{T} } ) where {S, T} = (@boundscheck length(y) == prod(size(S)) || error();
                                                                        SArray{S, T} )
@inline Base.promote_rule(::Type{<: SArray{S, T1} },
                        ::Type{<: AbstractArray{T2} } ) where {S, T1, T2} = SArray{S, promote_type(T1, T2) }
