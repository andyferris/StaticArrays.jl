import LinearAlgebra.MulAddMul

""" Size that stores whether a Matrix is a Transpose
Useful when selecting multiplication methods, and avoiding allocations when dealing with
the `Transpose` type by passing around the original matrix.
Should pair with `parent`.
"""
struct TSize{S,T}
    function TSize{S,T}() where {S,T}
        new{S::Tuple{Vararg{StaticDimension}},T::Bool}()
    end
end
TSize(A::Type{<:Transpose{<:Any,<:StaticArray}}) = TSize{size(A),true}()
TSize(A::Type{<:Adjoint{<:Real,<:StaticArray}}) = TSize{size(A),true}()  # can't handle complex adjoints yet
TSize(A::Type{<:StaticArray}) = TSize{size(A),false}()
TSize(A::StaticArrayLike) = TSize(typeof(A))
istranpose(::TSize{<:Any,T}) where T = T
size(::TSize{S}) where S = S
Size(::TSize{S}) where S = Size{S}()
Base.transpose(::TSize{S,T}) where {S,T} = TSize{reverse(S),!T}()

# Get the parent of transposed arrays, or the array itself if it has no parent
#   QUESTION: maybe call this something else?
Base.parent(A::Union{<:Transpose{<:Any,<:StaticArray}, <:Adjoint{<:Any,<:StaticArray}}) = A.parent
Base.parent(A::StaticArray) = A

# 5-argument matrix multiplication
#    To avoid allocations, strip away Transpose type and store tranpose info in Size
@inline LinearAlgebra.mul!(dest::StaticVecOrMatLike, A::StaticVecOrMatLike, B::StaticVecOrMatLike,
    α::Real, β::Real) = _mul!(TSize(dest), parent(dest), TSize(A), TSize(B), parent(A), parent(B), MulAddMul(α,β))

"Calculate the product of the dimensions being multiplied. Useful as a heuristic for unrolling."
@inline multiplied_dimension(A::Type{<:StaticVecOrMatLike}, B::Type{<:StaticVecOrMatLike}) =
    prod(size(A)) * size(B,2)

""" Combine left and right sides of an assignment expression, short-cutting
        lhs = α * rhs + β * lhs,
    element-wise.
If α = 1, the multiplication by α is removed. If β = 0, the second rhs term is removed.
"""
function _muladd_expr(lhs::Array{Expr}, rhs::Array{Expr}, ::Type{<:MulAddMul{Tα,Tβ}}) where {Tα,Tβ}
    @assert length(lhs) == length(rhs)
    n = length(rhs)
    if !Tα # not 1
        rhs = [:(α * $(expr)) for expr in rhs]
    end
    if !Tβ # not 0
        rhs = [:($(lhs[k]) * β + $(rhs[k])) for k = 1:n]
    end
    exprs = [:($(lhs[k]) = $(rhs[k])) for k = 1:n]
    return exprs
end

"Validate the dimensions of a matrix multiplication, including matrix-vector products"
function check_dims(::Size{sc}, ::Size{sa}, ::Size{sb}) where {sa,sb,sc}
    if sb[1] != sa[2] || sc[1] != sa[1]
        return false
    elseif length(sc) == 2 || length(sb) == 2
        sc2 = length(sc) == 1 ? 1 : sc[2]
        sb2 = length(sb) == 1 ? 1 : sb[2]
        if sc2 != sb2
            return false
        end
    end
    return true
end

"Obtain an expression for the linear index of var[k,j], taking transposes into account"
@inline _lind(A::Type{<:TSize}, k::Int, j::Int) = _lind(:a, A, k, j)
function _lind(var::Symbol, A::Type{TSize{sa,tA}}, k::Int, j::Int) where {sa,tA}
    if tA
        return :($var[$(LinearIndices(reverse(sa))[j, k])])
    else
        return :($var[$(LinearIndices(sa)[k, j])])
    end
end

# Matrix-vector multiplication
@generated function _mul!(Sc::TSize{sc}, c::StaticVecOrMatLike, Sa::TSize{sa}, Sb::TSize{sb},
        a::StaticMatrix, b::StaticVector, _add::MulAddMul,
        ::Val{col}=Val(1)) where {sa, sb, sc, col}
    if sa[2] != sb[1] || sc[1] != sa[1]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    if sa[2] != 0
        lhs = [:($(_lind(:c,Sc,k,col))) for k = 1:sa[1]]
        ab = [:($(reduce((ex1,ex2) -> :(+($ex1,$ex2)),
            [:($(_lind(Sa,k,j))*b[$j]) for j = 1:sa[2]]))) for k = 1:sa[1]]
        exprs = _muladd_expr(lhs, ab, _add)
    else
        exprs = [:(c[$k] = zero(eltype(c))) for k = 1:sa[1]]
    end

    return quote
        # @_inline_meta
        α = _add.alpha
        β = _add.beta
        @inbounds $(Expr(:block, exprs...))
        return c
    end
end

# Outer product
@generated function _mul!(::TSize{sc}, c::StaticMatrix, ::TSize{sa,false}, ::TSize{sb,true}, a::StaticVector,
        b::StaticVector,
        _add::MulAddMul) where {sa, sb, sc}
    if sc[1] != sa[1] || sc[2] != sb[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    lhs = [:(c[$(LinearIndices(sc)[i,j])]) for i = 1:sa[1], j = 1:sb[2]]
    ab = [:(a[$i] * b[$j]) for i = 1:sa[1], j = 1:sb[2]]
    exprs = _muladd_expr(lhs, ab, _add)

    return quote
        @_inline_meta
        α = _add.alpha
        β = _add.beta
        @inbounds $(Expr(:block, exprs...))
        return c
    end
end

# Matrix-matrix multiplication
@generated function _mul!(Sc::TSize{sc}, c::StaticMatrixLike,
        Sa::TSize{sa}, Sb::TSize{sb},
        a::StaticMatrixLike, b::StaticMatrixLike,
        _add::MulAddMul) where {sa, sb, sc}
    Ta,Tb,Tc = eltype(a), eltype(b), eltype(c)
    can_blas = Tc == Ta && Tc == Tb && Tc <: BlasFloat

    mult_dim = multiplied_dimension(a,b)
    if mult_dim < 4*4*4
        return quote
            @_inline_meta
            muladd_unrolled_all!(Sc, c, Sa, Sb, a, b, _add)
            # return c
        end
    elseif mult_dim < 14*14*14 # Something seems broken for this one with large matrices (becomes allocating)
        return quote
            @_inline_meta
            muladd_unrolled_chunks!(Sc, c, Sa, Sb, a, b, _add)
            # return c
        end
    else
        if can_blas
            return quote
                @_inline_meta
                mul_blas!(Sc, c, Sa, Sb, a, b, _add)
                return c
            end
        else
            return quote
                @_inline_meta
                muladd_unrolled_chunks!(Sc, c, Sa, Sb, a, b, _add)
                return c
            end
        end
    end
end


@generated function muladd_unrolled_all!(Sc::TSize{sc}, c::StaticMatrixLike, Sa::TSize{sa}, Sb::TSize{sb},
        a::StaticMatrixLike, b::StaticMatrixLike, _add::MulAddMul) where {sa, sb, sc}
    if !check_dims(Size(sc),Size(sa),Size(sb))
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    if sa[2] != 0
        lhs = [:($(_lind(:c, Sc, k1, k2))) for k1 = 1:sa[1], k2 = 1:sb[2]]
        ab = [:($(reduce((ex1,ex2) -> :(+($ex1,$ex2)),
                [:($(_lind(:a, Sa, k1, j)) * $(_lind(:b, Sb, j, k2))) for j = 1:sa[2]]
            ))) for k1 = 1:sa[1], k2 = 1:sb[2]]
        exprs = _muladd_expr(lhs, ab, _add)
    end

    return quote
        @_inline_meta
        α = _add.alpha
        β = _add.beta
        @inbounds $(Expr(:block, exprs...))
    end
end


@generated function muladd_unrolled_chunks!(Sc::TSize{sc}, c::StaticMatrix,
        Sa::TSize{sa}, Sb::TSize{sb},
        a::StaticMatrixLike, b::StaticMatrixLike,
        _add::MulAddMul) where {sa, sb, sc}
    if !check_dims(Size(sc),Size(sa),Size(sb))
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    # Do a custom b[:, k2] to return a SVector (an isbitstype type) rather than a mutable type. Avoids allocation == faster
    tmp_type = SVector{sb[1], eltype(c)}

    col_mult = [:(
        _mul!(Sc, c, Sa, Sb, a,
        $(Expr(:call, tmp_type,
        [:($(_lind(:b, Sb, i, k2))) for i = 1:sb[1]]...)),_add,Val($k2))) for k2 = 1:sb[2]]

    return quote
        α = _add.alpha
        β = _add.beta
        return $(Expr(:block, col_mult...))
    end
end

@inline _get_raw_data(A::SizedArray) = A.data
@inline _get_raw_data(A::StaticArray) = A

function mul_blas!(::TSize{<:Any,false}, c::StaticMatrix, ::TSize{<:Any,tA}, ::TSize{<:Any,tB},
        a::StaticMatrix, b::StaticMatrix, _add::MulAddMul) where {tA,tB}
    mat_char(tA) = tA ? 'T' : 'N'
    T = eltype(a)
    A = _get_raw_data(a)
    B = _get_raw_data(b)
    C = _get_raw_data(c)
    BLAS.gemm!(mat_char(tA), mat_char(tB), T(_add.alpha), A, B, T(_add.beta), C)
end

# if C is transposed, transpose the entire expression
@inline mul_blas!(Sc::TSize{<:Any,true}, c::StaticMatrix, Sa::TSize, Sb::TSize,
        a::StaticMatrix, b::StaticMatrix, _add::MulAddMul) =
    mul_blas!(transpose(Sc), c, transpose(Sb), transpose(Sa), b, a, _add)

# TODO: Get this version of mul_blas! working so there's backward-compatibility with older Julia versions

# @generated function mul_blas!(::TSize{s,false}, c::StaticMatrix{<:Any, <:Any, T},
#         ::TSize{sa,tA}, ::TSize{sb,tB},
#         a::StaticMatrix{<:Any, <:Any, T}, b::StaticMatrix{<:Any, <:Any, T},
#         _add::MulAddMul = MulAddMul()) where {s,sa,sb, T <: BlasFloat, tA, tB}
#     if sb[1] != sa[2] || sa[1] != s[1] || sb[2] != s[2]
#         throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $s"))
#     end
#
#     if sa[1] > 0 && sa[2] > 0 && sb[2] > 0
#         # This code adapted from `gemm!()` in base/linalg/blas.jl
#
#         if T == Float64
#             gemm = :dgemm_
#         elseif T == Float32
#             gemm = :sgemm_
#         elseif T == Complex{Float64}
#             gemm = :zgemm_
#         else # T == Complex{Float32}
#             gemm = :cgemm_
#         end
#
#         mat_char(tA) = tA ? 'T' : 'N'
#         transA = mat_char(tA)
#         transB = mat_char(tB)
#
#         m = sa[tA ? 2 : 1]
#         ka = sa[tA ? 1 : 2]
#         kb = sb[tB ? 2 : 1]
#         n = sb[tB ? 1 : 2]
#
#         blascall = quote
#              ccall((LinearAlgebra.BLAS.@blasfunc($gemm), LinearAlgebra.BLAS.libblas), Nothing,
#                  (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt},
#                   Ref{LinearAlgebra.BLAS.BlasInt}, Ref{$T}, Ptr{$T}, Ref{LinearAlgebra.BLAS.BlasInt},
#                   Ptr{$T}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{$T}, Ptr{$T},
#                   Ref{LinearAlgebra.BLAS.BlasInt}),
#                   $transA, $transB, $m, $n,
#                   $ka, alpha, A, strideA,
#                   B, strideB, beta, C,
#                   strideC)
#         end
#
#         return quote
#             println($transA)
#             println($transB)
#             alpha = _add.alpha
#             beta = _add.beta
#             # m = $(sa[1])
#             # ka = $(sa[2])
#             # kb = $(sb[1])
#             # n = $(sb[2])
#             strideA = $m  #$(sa[1])
#             strideB = $kb #$(sb[1])
#             strideC = $(s[1])
#             A = _get_raw_data(a)
#             B = _get_raw_data(b)
#             C = _get_raw_data(c)
#
#             $blascall
#
#             return c
#         end
#     else
#         throw(DimensionMismatch("Cannot call BLAS gemm with zero-dimension arrays, attempted $sa * $sb -> $s."))
#     end
# end
