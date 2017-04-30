# `Start` and `End` should be `Int`
struct SUnitRange{Start, L} <: StaticVector{L, Int}
    function SUnitRange{Start, L}() where {Start, L}
        check_sunitrange_params(L)
        new{Start, L}()
    end
end

@pure function check_sunitrange_params(L::Int)
    if L < 0
        error("Static unit range length is negative")
    end
end

function check_sunitrange_params(L)
    throw(TypeError(:SUnitRange, "type parameters must be `Int`s", Tuple{Int, Int, Int}, Tuple{typeof(a), typeof(b), typeof(c)}))
end

@pure SUnitRange(a::Int, b::Int) = SUnitRange{a, max(0, b - a + 1)}()

@pure @propagate_inbounds function getindex(x::SUnitRange{Start, L}, i::Int) where {Start, L}
    @boundscheck if i < Start || i >= (Start + L)
        throw(BoundsError(x, i))
    end
    return Start + i - 1
end

# Shorten show for REPL use.
show(io::IO, ::Type{SUnitRange}) = print(io, "SUnitRange")
function show(io::IO, ::MIME"text/plain", ::SUnitRange{Start, L}) where {Start, L}
    print(io, "SUnitRange($Start,$(Start + L - 1))")
end

# For this type to be usable as `indices`, they need to support some more stuff
Base.unsafe_length(r::SUnitRange) = length(r)
@inline first(r::SUnitRange{Start}) where {Start} = Start # matches Base.UnitRange when L == 0...
@inline endof(r::SUnitRange{Start, L}) where {Start, L} = L

(==)(::SUnitRange{Start, L}, ::SUnitRange{Start, L}) where {Start, L} = true
(==)(::SUnitRange{Start1, L1}, ::SUnitRange{Start2, L2}) where {Start1, Start2, L1, L2} = false

(==)(::SUnitRange{1, L}, r::Base.OneTo) where {L} = L == r.stop
(==)(::SUnitRange, ::Base.OneTo) = false
(==)(r::Base.OneTo, ::SUnitRange{1, L}) where {L} = L == r.stop
(==)(::Base.OneTo, ::SUnitRange) = false

start(r::SUnitRange) = 1
next(r::SUnitRange, i) = (r[i], i+1)
done(r::SUnitRange{Start, L}, i) where {Start, L} = i > L

@pure Base.UnitRange{Int}(::SUnitRange{Start, L}) where {Start, L} = Start : (Start + L - 1)
@pure Base.Slice(::SUnitRange{Start, L}) where {Start, L} = Base.Slice(Start : (Start + L - 1)) # TODO not optimal

function Base.checkindex(::Type{Bool}, ::SUnitRange{Start, L}, r::UnitRange{Int}) where {Start, L}
    return first(r) < Start | last(r) >= Start + L
end

function Base.checkindex(::Type{Bool}, ::SUnitRange{Start, L}, r::Base.Slice{UnitRange{Int}}) where {Start, L}
    return first(r) < Start | last(r) >= Start + L
end
