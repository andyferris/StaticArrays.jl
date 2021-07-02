"""
    MArray{S, T, N, L}(undef)
    MArray{S, T, N, L}(x::NTuple{L})
    MArray{S, T, N, L}(x1, x2, x3, ...)


Construct a statically-sized, mutable array `MArray`. The data may optionally be
provided upon construction and can be mutated later. The `S` parameter is a Tuple-type
specifying the dimensions, or size, of the array - such as `Tuple{3,4,5}` for a 3×4×5-sized
array. The `N` parameter is the dimension of the array; the `L` parameter is the `length`
of the array and is always equal to `prod(S)`. Constructors may drop the `L`, `N` and `T`
parameters if they are inferrable from the input (e.g. `L` is always inferrable from `S`).

    MArray{S}(a::Array)

Construct a statically-sized, mutable array of dimensions `S` (expressed as a `Tuple{...}`)
using the data from `a`. The `S` parameter is mandatory since the size of `a` is unknown to
the compiler (the element type may optionally also be specified).
"""
mutable struct MArray{S <: Tuple, T, N, L} <: StaticArray{S, T, N}
    data::NTuple{L,T}

    function MArray{S,T,N,L}(x::NTuple{L,T}) where {S,T,N,L}
        check_array_parameters(S, T, Val{N}, Val{L})
        new{S,T,N,L}(x)
    end

    function MArray{S,T,N,L}(x::NTuple{L,Any}) where {S,T,N,L}
        check_array_parameters(S, T, Val{N}, Val{L})
        new{S,T,N,L}(convert_ntuple(T, x))
    end

    function MArray{S,T,N,L}(::UndefInitializer) where {S,T,N,L}
        check_array_parameters(S, T, Val{N}, Val{L})
        new{S,T,N,L}()
    end
end

@generated function (::Type{MArray{S,T,N}})(x::Tuple) where {S,T,N}
    return quote
        $(Expr(:meta, :inline))
        MArray{S,T,N,$(tuple_prod(S))}(x)
    end
end

@generated function (::Type{MArray{S,T}})(x::Tuple) where {S,T}
    return quote
        $(Expr(:meta, :inline))
        MArray{S,T,$(tuple_length(S)),$(tuple_prod(S))}(x)
    end
end

@generated function (::Type{MArray{S}})(x::T) where {S, T <: Tuple}
    return quote
        $(Expr(:meta, :inline))
        MArray{S,promote_tuple_eltype(T),$(tuple_length(S)),$(tuple_prod(S))}(x)
    end
end

@generated function (::Type{MArray{S,T,N}})(::UndefInitializer) where {S,T,N}
    return quote
        $(Expr(:meta, :inline))
        MArray{S, T, N, $(tuple_prod(S))}(undef)
    end
end

@generated function (::Type{MArray{S,T}})(::UndefInitializer) where {S,T}
    return quote
        $(Expr(:meta, :inline))
        MArray{S, T, $(tuple_length(S)), $(tuple_prod(S))}(undef)
    end
end

@inline MArray(a::StaticArray{S,T}) where {S<:Tuple,T} = MArray{S,T}(Tuple(a))

####################
## MArray methods ##
####################

@propagate_inbounds function getindex(v::MArray, i::Int)
    @boundscheck checkbounds(v,i)
    T = eltype(v)

    if isbitstype(T)
        return GC.@preserve v unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), i)
    end
    v.data[i]
end

@propagate_inbounds function setindex!(v::MArray, val, i::Int)
    @boundscheck checkbounds(v,i)
    T = eltype(v)

    if isbitstype(T)
        GC.@preserve v unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), convert(T, val), i)
    else
        # This one is unsafe (#27)
        # unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Nothing}}, pointer_from_objref(v.data)), pointer_from_objref(val), i)
        error("setindex!() with non-isbitstype eltype is not supported by StaticArrays. Consider using SizedArray.")
    end

    return v
end

@inline Tuple(v::MArray) = v.data

Base.dataids(ma::MArray) = (UInt(pointer(ma)),)

@inline function Base.unsafe_convert(::Type{Ptr{T}}, a::MArray{S,T}) where {S,T}
    Base.unsafe_convert(Ptr{T}, pointer_from_objref(a))
end

macro MArray(ex)
    if !isa(ex, Expr)
        error("Bad input for @MArray")
    end

    if ex.head == :vect  # vector
        return esc(Expr(:call, MArray{Tuple{length(ex.args)}}, Expr(:tuple, ex.args...)))
    elseif ex.head == :ref # typed, vector
        return esc(Expr(:call, Expr(:curly, :MArray, Tuple{length(ex.args)-1}, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return esc(Expr(:call, MArray{Tuple{s1, s2}}, Expr(:tuple, ex.args...)))
    elseif ex.head == :typed_hcat # typed, 1 x n
        s1 = 1
        s2 = length(ex.args) - 1
        return esc(Expr(:call, Expr(:curly, :MArray, Tuple{s1, s2}, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif ex.head == :vcat
        if isa(ex.args[1], Expr) && ex.args[1].head == :row # n x m
            # Validate
            s1 = length(ex.args)
            s2s = map(i -> ((isa(ex.args[i], Expr) && ex.args[i].head == :row) ? length(ex.args[i].args) : 1), 1:s1)
            s2 = minimum(s2s)
            if maximum(s2s) != s2
                throw(ArgumentError("Rows must be of matching lengths"))
            end

            exprs = [ex.args[i].args[j] for i = 1:s1, j = 1:s2]
            return esc(Expr(:call, MArray{Tuple{s1, s2}}, Expr(:tuple, exprs...)))
        else # n x 1
            return esc(Expr(:call, MArray{Tuple{length(ex.args), 1}}, Expr(:tuple, ex.args...)))
        end
    elseif ex.head == :typed_vcat
        if isa(ex.args[2], Expr) && ex.args[2].head == :row # typed, n x m
            # Validate
            s1 = length(ex.args) - 1
            s2s = map(i -> ((isa(ex.args[i+1], Expr) && ex.args[i+1].head == :row) ? length(ex.args[i+1].args) : 1), 1:s1)
            s2 = minimum(s2s)
            if maximum(s2s) != s2
                error("Rows must be of matching lengths")
            end

            exprs = [ex.args[i+1].args[j] for i = 1:s1, j = 1:s2]
            return esc(Expr(:call, Expr(:curly, :MArray, Tuple{s1, s2}, ex.args[1]), Expr(:tuple, exprs...)))
        else # typed, n x 1
            return esc(Expr(:call, Expr(:curly, :MArray, Tuple{length(ex.args)-1, 1}, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
        end
    elseif isa(ex, Expr) && ex.head == :comprehension
        if length(ex.args) != 1 || !isa(ex.args[1], Expr) || ex.args[1].head != :generator
            error("Expected generator in comprehension, e.g. [f(i,j) for i = 1:3, j = 1:3]")
        end
        ex = ex.args[1]
        n_rng = length(ex.args) - 1
        rng_args = [ex.args[i+1].args[1] for i = 1:n_rng]
        rngs = [Core.eval(__module__, ex.args[i+1].args[2]) for i = 1:n_rng]
        rng_lengths = map(length, rngs)

        f = gensym()
        f_expr = :($f = ($(Expr(:tuple, rng_args...)) -> $(ex.args[1])))

        # TODO figure out a generic way of doing this...
        if n_rng == 1
            exprs = [:($f($j1)) for j1 in rngs[1]]
        elseif n_rng == 2
            exprs = [:($f($j1, $j2)) for j1 in rngs[1], j2 in rngs[2]]
        elseif n_rng == 3
            exprs = [:($f($j1, $j2, $j3)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3]]
        elseif n_rng == 4
            exprs = [:($f($j1, $j2, $j3, $j4)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4]]
        elseif n_rng == 5
            exprs = [:($f($j1, $j2, $j3, $j4, $j5)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5]]
        elseif n_rng == 6
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6]]
        elseif n_rng == 7
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6, $j7)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6], j7 in rngs[7]]
        elseif n_rng == 8
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6, $j7, $j8)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6], j7 in rngs[7], j8 in rngs[8]]
        else
            error("@MArray only supports up to 8-dimensional comprehensions")
        end

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MArray, Tuple{rng_lengths...}), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) || ex.args[2].head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i,j) for i = 1:3, j = 1:3]")
        end
        T = ex.args[1]
        ex = ex.args[2]
        n_rng = length(ex.args) - 1
        rng_args = [ex.args[i+1].args[1] for i = 1:n_rng]
        rngs = [Core.eval(__module__, ex.args[i+1].args[2]) for i = 1:n_rng]
        rng_lengths = map(length, rngs)

        f = gensym()
        f_expr = :($f = ($(Expr(:tuple, rng_args...)) -> $(ex.args[1])))

        # TODO figure out a generic way of doing this...
        if n_rng == 1
            exprs = [:($f($j1)) for j1 in rngs[1]]
        elseif n_rng == 2
            exprs = [:($f($j1, $j2)) for j1 in rngs[1], j2 in rngs[2]]
        elseif n_rng == 3
            exprs = [:($f($j1, $j2, $j3)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3]]
        elseif n_rng == 4
            exprs = [:($f($j1, $j2, $j3, $j4)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4]]
        elseif n_rng == 5
            exprs = [:($f($j1, $j2, $j3, $j4, $j5)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5]]
        elseif n_rng == 6
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6]]
        elseif n_rng == 7
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6, $j7)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6], j7 in rngs[7]]
        elseif n_rng == 8
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6, $j7, $j8)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6], j7 in rngs[7], j8 in rngs[8]]
        else
            error("@MArray only supports up to 8-dimensional comprehensions")
        end

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MArray, Tuple{rng_lengths...}, T), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :call
        if ex.args[1] == :zeros || ex.args[1] == :ones || ex.args[1] == :rand || ex.args[1] == :randn || ex.args[1] == :randexp
            if length(ex.args) == 1
                error("@MArray got bad expression: $(ex.args[1])()")
            else
                return quote
                    if isa($(esc(ex.args[2])), DataType)
                        $(ex.args[1])($(esc(Expr(:curly, MArray, Expr(:curly, Tuple, ex.args[3:end]...), ex.args[2]))))
                    else
                        $(ex.args[1])($(esc(Expr(:curly, MArray, Expr(:curly, Tuple, ex.args[2:end]...)))))
                    end
                end
            end
        elseif ex.args[1] == :fill
            if length(ex.args) == 1
                error("@MArray got bad expression: $(ex.args[1])()")
            elseif length(ex.args) == 2
                error("@MArray got bad expression: $(ex.args[1])($(ex.args[2]))")
            else
                return quote
                    $(esc(ex.args[1]))($(esc(ex.args[2])), MArray{$(esc(Expr(:curly, Tuple, ex.args[3:end]...)))})
                end
            end
        else
            error("@MArray only supports the zeros(), ones(), rand(), randn(), and randexp() functions.")
        end
    else
        error("Bad input for @MArray")
    end

end

function promote_rule(::Type{<:MArray{S,T,N,L}}, ::Type{<:MArray{S,U,N,L}}) where {S,T,U,N,L}
    MArray{S,promote_type(T,U),N,L}
end

function Base.view(
    a::MArray{S},
    indices::Union{Integer, Colon, StaticVector, Base.Slice, SOneTo}...,
) where {S}
    new_size = new_out_size(S, indices...)
    view_from_invoke = invoke(view, Tuple{AbstractArray, typeof(indices).parameters...}, a, indices...)
    return SizedArray{new_size}(view_from_invoke)
end

Base.elsize(::Type{<:MArray{S,T}}) where {S,T} = sizeof(T)

function Base.show(io::IO, S::MArray)
    print(io, typeof(S), "(")
    show(io, Tuple(S))
    print(io, ")")
end
