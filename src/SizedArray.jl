
"""
    SizedArray{Tuple{dims...}}(array)

Wraps an `AbstractArray` with a static size, so to take advantage of the (faster)
methods defined by the static array package. The size is checked once upon
construction to determine if the number of elements (`length`) match, but the
array may be reshaped.

The aliases `SizedVector{N}` and `SizedMatrix{N,M}` are provided as more
convenient names for one and two dimensional `SizedArray`s. For example, to
wrap a 2x3 array `a` in a `SizedArray`, use `SizedMatrix{2,3}(a)`.
"""
struct SizedArray{S<:Tuple,T,N,M,TData<:AbstractArray{T,M}} <: StaticArray{S,T,N}
    data::TData

    function SizedArray{S,T,N,M,TData}(a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}}
        if size(a) != size_to_tuple(S) && size(a) != (tuple_prod(S),)
            throw(DimensionMismatch("Dimensions $(size(a)) don't match static size $S"))
        end
        return new{S,T,N,M,TData}(a)
    end

    function SizedArray{S,T,N,1,TData}(::UndefInitializer) where {S,T,N,TData<:AbstractArray{T,1}}
        return new{S,T,N,1,TData}(TData(undef, tuple_prod(S)))
    end
    function SizedArray{S,T,N,N,TData}(::UndefInitializer) where {S,T,N,TData<:AbstractArray{T,N}}
        return new{S,T,N,N,TData}(TData(undef, size_to_tuple(S)...))
    end
end

@inline function SizedArray{S,T,N}(
    a::TData,
) where {S,T,N,M,TData<:AbstractArray{T,M}}
    return SizedArray{S,T,N,M,TData}(a)
end
@inline function SizedArray{S,T}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}}
    return SizedArray{S,T,tuple_length(S),M,TData}(a)
end
@inline function SizedArray{S}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}}
    return SizedArray{S,T,tuple_length(S),M,TData}(a)
end
function SizedArray{S,T,N,N}(::UndefInitializer) where {S,T,N}
    return SizedArray{S,T,N,N,Array{T,N}}(undef)
end
function SizedArray{S,T,N,1}(::UndefInitializer) where {S,T,N}
    return SizedArray{S,T,N,1,Vector{T}}(undef)
end
@inline function SizedArray{S,T,N}(::UndefInitializer) where {S,T,N}
    return SizedArray{S,T,N,N}(undef)
end
@inline function SizedArray{S,T}(::UndefInitializer) where {S,T}
    return SizedArray{S,T,tuple_length(S)}(undef)
end
@generated function (::Type{SizedArray{S,T,N,M,TData}})(x::NTuple{L,Any}) where {S,T,N,M,TData<:AbstractArray{T,M},L}
    if L != tuple_prod(S)
        error("Dimension mismatch")
    end
    exprs = [:(a[$i] = x[$i]) for i = 1:L]
    return quote
        $(Expr(:meta, :inline))
        a = SizedArray{S,T,N,M}(undef)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end
@inline function SizedArray{S,T,N,M}(x::Tuple) where {S,T,N,M}
    return SizedArray{S,T,N,M,Array{T,M}}(x)
end
@inline function SizedArray{S,T,N}(x::Tuple) where {S,T,N}
    return SizedArray{S,T,N,N,Array{T,N}}(x)
end
@inline function SizedArray{S,T}(x::Tuple) where {S,T}
    return SizedArray{S,T,tuple_length(S)}(x)
end
@inline function SizedArray{S}(x::NTuple{L,T}) where {S,T,L}
    return SizedArray{S,T}(x)
end

# Overide some problematic default behaviour
@inline convert(::Type{SA}, sa::SizedArray) where {SA<:SizedArray} = SA(sa.data)
@inline convert(::Type{SA}, sa::SA) where {SA<:SizedArray} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
@inline function Base.Array(sa::SizedArray{S}) where {S}
    return Array(reshape(sa.data, size_to_tuple(S)))
end
@inline function Base.Array{T}(sa::SizedArray{S,T}) where {T,S}
    return Array(reshape(sa.data, size_to_tuple(S)))
end
@inline function Base.Array{T,N}(sa::SizedArray{S,T,N}) where {T,S,N}
    return Array(reshape(sa.data, size_to_tuple(S)))
end

@inline function convert(::Type{Array}, sa::SizedArray{S}) where {S}
    return Array(reshape(sa.data, size_to_tuple(S)))
end
@inline function convert(::Type{Array}, sa::SizedArray{S,T,N,M,Array{T,M}}) where {S,T,N,M}
    return sa.data
end
@inline function convert(::Type{Array{T}}, sa::SizedArray{S,T}) where {T,S}
    return Array(reshape(sa.data, size_to_tuple(S)))
end
@inline function convert(::Type{Array{T}}, sa::SizedArray{S,T,N,M,Array{T,M}}) where {S,T,N,M}
    return sa.data
end
@inline function convert(
    ::Type{Array{T,N}},
    sa::SizedArray{S,T,N},
) where {T,S,N}
    return Array(reshape(sa.data, size_to_tuple(S)))
end
@inline function convert(::Type{Array{T,N}}, sa::SizedArray{S,T,N,N,Array{T,N}}) where {S,T,N}
    return sa.data
end

@propagate_inbounds getindex(a::SizedArray, i::Int) = getindex(a.data, i)
@propagate_inbounds setindex!(a::SizedArray, v, i::Int) = setindex!(a.data, v, i)

const SizedVector{S,T,M} = SizedArray{Tuple{S},T,1,M,Array{T,M}}

@inline function SizedVector{S}(a::TData) where {S,T,TData<:AbstractVector{T}}
    return SizedArray{Tuple{S},T,1,1,TData}(a)
end
@inline function SizedVector(x::NTuple{S,T}) where {S,T}
    return SizedArray{Tuple{S},T,1,1,Vector{T}}(x)
end
@inline function SizedVector{S}(x::NTuple{S,T}) where {S,T}
    return SizedArray{Tuple{S},T,1,1,Vector{T}}(x)
end
@inline function SizedVector{S,T}(x::NTuple{S}) where {S,T}
    return SizedArray{Tuple{S},T,1,1,Vector{T}}(x)
end
# disambiguation
@inline function SizedVector{S}(a::StaticVector{S,T}) where {S,T}
    return SizedVector{S,T}(a.data)
end

const SizedMatrix{S1,S2,T,M} = SizedArray{Tuple{S1,S2},T,2,M,Array{T,M}}

@inline function SizedMatrix{S1,S2}(
    a::TData,
) where {S1,S2,T,M,TData<:AbstractArray{T,M}}
    return SizedArray{Tuple{S1,S2},T,2,M,TData}(a)
end
@inline function SizedMatrix{S1,S2}(x::NTuple{L,T}) where {S1,S2,T,L}
    return SizedArray{Tuple{S1,S2},T,2,2,Matrix{T}}(x)
end
@inline function SizedMatrix{S1,S2,T}(x::NTuple{L}) where {S1,S2,T,L}
    return SizedArray{Tuple{S1,S2},T,2,2,Matrix{T}}(x)
end
# disambiguation
@inline function SizedMatrix{S1,S2}(a::StaticMatrix{S1,S2,T}) where {S1,S2,T}
    return SizedMatrix{S1,S2,T}(a.data)
end

Base.dataids(sa::SizedArray) = Base.dataids(sa.data)

function promote_rule(
    ::Type{SizedArray{S,T,N,M,TDataA}},
    ::Type{SizedArray{S,U,N,M,TDataB}},
) where {S,T,U,N,M,TDataA,TDataB}
    TU = promote_type(T, U)
    return SizedArray{
        S,
        TU,
        N,
        M,
        promote_type(TDataA, TDataB),
    }
end
