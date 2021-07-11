"""
    SOneTo(n)

Return a statically-sized `AbstractUnitRange` starting at `1`, functioning as the `axes` of
a `StaticArray`.
"""
struct SOneTo{n} <: AbstractUnitRange{Int}
end

SOneTo(n::Int) = SOneTo{n}()
function SOneTo{n}(r::AbstractUnitRange) where n
    ((first(r) == 1) & (last(r) == n)) && return SOneTo{n}()

    errmsg(r) = throw(DimensionMismatch("$r is inconsistent with SOneTo{$n}")) # avoid GC frame
    errmsg(r)
end

Base.Tuple(::SOneTo{N}) where N = ntuple(identity, Val(N))

Base.axes(s::SOneTo) = (s,)
Base.size(s::SOneTo) = (length(s),)
Base.length(s::SOneTo{n}) where {n} = n

# The axes of a Slice'd SOneTo use the SOneTo itself
Base.axes(S::Base.Slice{<:SOneTo}) = (S.indices,)
Base.unsafe_indices(S::Base.Slice{<:SOneTo}) = (S.indices,)
Base.axes1(S::Base.Slice{<:SOneTo}) = S.indices

@propagate_inbounds function Base.getindex(s::SOneTo, i::Int)
    @boundscheck checkbounds(s, i)
    return i
end
@propagate_inbounds function Base.getindex(s::SOneTo, s2::SOneTo)
    @boundscheck checkbounds(s, s2)
    return s2
end
if isdefined(Base, :IdentityUnitRange)
    @propagate_inbounds function Base.getindex(s::SOneTo, s2::Base.IdentityUnitRange{<:AbstractUnitRange{<:Integer}})
        @boundscheck checkbounds(s, s2)
        return s2
    end
    Base.axes(::Base.IdentityUnitRange{A}) where {A <: SOneTo} = (A(),)
    Base.axes(r::Base.IdentityUnitRange{<:SOneTo}, d::Int) = d <= 1 ? axes(r)[d] : SOneTo(1)
    Base.axes1(r::Base.IdentityUnitRange{A}) where {A <: SOneTo} = A()
    Base.unsafe_indices(::Base.IdentityUnitRange{A}) where {A <: SOneTo} = (A(),)
end

Base.first(::SOneTo) = 1
Base.last(::SOneTo{n}) where {n} = n::Int

@pure function Base.iterate(::SOneTo{n}) where {n}
    if n::Int < 1
        return nothing
    else
        (1, 1)
    end
end
function Base.iterate(::SOneTo{n}, s::Int) where {n}
    if s < n::Int
        s2 = s + 1
        return (s2, s2)
    else
        return nothing
    end
end

function Base.getproperty(::SOneTo{n}, s::Symbol) where {n}
    if s === :start
        return 1
    elseif s === :stop
        return n::Int
    else
        error("type SOneTo has no property $s")
    end
end

function Base.show(io::IO, ::SOneTo{n}) where {n}
    print(io, "SOneTo(", n::Int, ")")
end

Base.@pure function Base.checkindex(::Type{Bool}, ::SOneTo{n1}, ::SOneTo{n2}) where {n1, n2}
    return n1::Int >= n2::Int
end

Base.promote_rule(a::Type{Base.OneTo{T}}, ::Type{SOneTo{n}}) where {T,n} =
    Base.OneTo{promote_type(T, Int)}
