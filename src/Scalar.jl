"""
    Scalar{T}(x::T)

Construct a statically-sized 0-dimensional array that contains a single element,
`x`. This type is particularly useful for influencing broadcasting operations.
"""
const Scalar{T} = SArray{Tuple{},T,0,1}

@inline Scalar(x::Tuple{T}) where {T} = Scalar{T}(x[1])
@inline Scalar(a::AbstractArray) = Scalar{typeof(a)}((a,))
@inline Scalar(a::AbstractScalar) = Scalar{eltype(a)}((a[],)) # Do we want this to convert or wrap?
@inline Scalar(a::SA) where {SA<:StaticArray} = Scalar{SA}(a) # solve ambiguity
@inline function convert(::Type{SA}, a::AbstractArray) where {SA <: Scalar}
    return SA((a[],))
end
@inline convert(::Type{SA}, sa::SA) where {SA <: Scalar} = sa

@propagate_inbounds function getindex(v::Scalar, i::Int)
    @boundscheck if i != 1
        error("Attempt to index Scalar at index $i")
    end
    v.data[1]
end

# A lot more compact than the default array show
Base.show(io::IO, ::MIME"text/plain", x::Scalar{T}) where {T} = print(io, "Scalar{$T}(", x.data, ")")

