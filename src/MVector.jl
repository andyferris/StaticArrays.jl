"""
    MVector{S,T}(undef)
    MVector{S,T}(x::NTuple{S, T})
    MVector{S,T}(x1, x2, x3, ...)

Construct a statically-sized, mutable vector `MVector`. Data may optionally be
provided upon construction, and can be mutated later. Constructors may drop the
`T` and `S` parameters if they are inferrable from the input (e.g.
`MVector(1,2,3)` constructs an `MVector{3, Int}`).

    MVector{S}(vec::Vector)

Construct a statically-sized, mutable vector of length `S` using the data from
`vec`. The parameter `S` is mandatory since the length of `vec` is unknown to the
compiler (the element type may optionally also be specified).
"""
const MVector{S, T} = MArray{Tuple{S}, T, 1, S}

@inline MVector(a::StaticVector{N,T}) where {N,T} = MVector{N,T}(a)
@inline MVector(x::NTuple{S,Any}) where {S} = MVector{S}(x)
@inline MVector{S}(x::Tuple{T,Vararg{T}}) where {S, T} = MVector{S, T}(x)
@inline MVector{S}(x::NTuple{S,Any}) where {S} = MVector{S, promote_tuple_eltype(typeof(x))}(x)

# Some more advanced constructor-like functions
@inline zeros(::Type{MVector{N}}) where {N} = zeros(MVector{N,Float64})
@inline ones(::Type{MVector{N}}) where {N} = ones(MVector{N,Float64})

#####################
## MVector methods ##
#####################

macro MVector(ex)
    if isa(ex, Expr) && ex.head == :vect
        return esc(Expr(:call, MVector{length(ex.args)}, Expr(:tuple, ex.args...)))
    elseif isa(ex, Expr) && ex.head == :ref
        return esc(Expr(:call, Expr(:curly, :MVector, length(ex.args[2:end]), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif isa(ex, Expr) && ex.head == :comprehension
        if length(ex.args) != 1 || !isa(ex.args[1], Expr) || ex.args[1].head != :generator
            error("Expected generator in comprehension, e.g. [f(i) for i = 1:3]")
        end
        ex = ex.args[1]
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @MVector")
        end

        rng = Core.eval(__module__, ex.args[2].args[2])
        f = gensym()
        f_expr = :($f = ($(ex.args[2].args[1]) -> $(ex.args[1])))
        exprs = [:($f($j)) for j in rng]

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MVector, length(rng)), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) !! ex.args[2].head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i) for i = 1:3]")
        end
        T = ex.args[1]
        ex = ex.args[2]
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @MVector")
        end

        rng = Core.eval(__module__, ex.args[2].args[2])
        f = gensym()
        f_expr = :($f = ($(ex.args[2].args[1]) -> $(ex.args[1])))
        exprs = [:($f($j)) for j in rng]

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MVector, length(rng), T), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :call
        if ex.args[1] == :zeros || ex.args[1] == :ones || ex.args[1] == :rand || ex.args[1] == :randn || ex.args[1] == :randexp
            if length(ex.args) == 2
                return quote
                    $(esc(ex.args[1]))(MVector{$(esc(ex.args[2]))})
                end
            elseif length(ex.args) == 3
                return quote
                    $(esc(ex.args[1]))(MVector{$(esc(ex.args[3])), $(esc(ex.args[2]))})
                end
            else
                error("@MVector expected a 1-dimensional array expression")
            end
        elseif ex.args[1] == :fill
            if length(ex.args) == 3
                return quote
                    $(esc(ex.args[1]))($(esc(ex.args[2])), MVector{$(esc(ex.args[3]))})
                end
            else
                error("@MVector expected a 1-dimensional array expression")
            end
        else
            error("@MVector only supports the zeros(), ones(), rand(), randn(), and randexp() functions.")
        end
    else
        error("Use @MVector [a,b,c] or @MVector([a,b,c])")
    end
end
