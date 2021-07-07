module StaticArrays

import Base: @_inline_meta, @_propagate_inbounds_meta, @_pure_meta, @propagate_inbounds, @pure

import Base: getindex, setindex!, size, similar, vec, show, length, convert, promote_op,
             promote_rule, map, map!, reduce, mapreduce, foldl, mapfoldl, broadcast,
             broadcast!, conj, hcat, vcat, ones, zeros, one, reshape, fill, fill!, inv,
             iszero, sum, prod, count, any, all, minimum, maximum, extrema,
             copy, read, read!, write, reverse

import Statistics: mean

using Random
import Random: rand, randn, randexp, rand!, randn!, randexp!
using Core.Compiler: return_type
import Base: sqrt, exp, log
using LinearAlgebra
import LinearAlgebra: transpose, adjoint, dot, eigvals, eigen, lyap, tr,
                      kron, diag, norm, dot, diagm, lu, svd, svdvals,
                      factorize, ishermitian, issymmetric, isposdef, issuccess, normalize,
                      normalize!, Eigen, det, logdet, logabsdet, cross, diff, qr, \
using LinearAlgebra: checksquare

export SOneTo
export StaticScalar, StaticArray, StaticVector, StaticMatrix
export Scalar, SArray, SVector, SMatrix
export MArray, MVector, MMatrix
export FieldVector, FieldMatrix, FieldArray
export SizedArray, SizedVector, SizedMatrix
export SDiagonal
export SHermitianCompact

export Size, Length

export SA, SA_F32, SA_F64
export @SVector, @SMatrix, @SArray
export @MVector, @MMatrix, @MArray

export similar_type
export push, pop, pushfirst, popfirst, insert, deleteat, setindex
export packdims, unpackdims, unsafe_packdims, unsafe_unpackdims

include("SOneTo.jl")

"""
    abstract type StaticArray{S, T, N} <: AbstractArray{T, N} end
    StaticScalar{T}     = StaticArray{Tuple{}, T, 0}
    StaticVector{N,T}   = StaticArray{Tuple{N}, T, 1}
    StaticMatrix{N,M,T} = StaticArray{Tuple{N,M}, T, 2}

`StaticArray`s are Julia arrays with fixed, known size.

## Dev docs

They must define the following methods:
 - Constructors that accept a flat tuple of data.
 - `getindex()` with an integer (linear indexing) (preferably `@inline` with `@boundscheck`).
 - `Tuple()`, returning the data in a flat Tuple.

It may be useful to implement:

- `similar_type(::Type{MyStaticArray}, ::Type{NewElType}, ::Size{NewSize})`, returning a
  type (or type constructor) that accepts a flat tuple of data.

For mutable containers you may also need to define the following:

 - `setindex!` for a single element (linear indexing).
 - `similar(::Type{MyStaticArray}, ::Type{NewElType}, ::Size{NewSize})`.
 - In some cases, a zero-parameter constructor, `MyStaticArray{...}()` for unintialized data
   is assumed to exist.

(see also `SVector`, `SMatrix`, `SArray`, `MVector`, `MMatrix`, `MArray`, `SizedArray`, `FieldVector`, `FieldMatrix` and `FieldArray`)
"""
abstract type StaticArray{S <: Tuple, T, N} <: AbstractArray{T, N} end
const StaticScalar{T} = StaticArray{Tuple{}, T, 0}
const StaticVector{N, T} = StaticArray{Tuple{N}, T, 1}
const StaticMatrix{N, M, T} = StaticArray{Tuple{N, M}, T, 2}
const StaticVecOrMat{T} = Union{StaticVector{<:Any, T}, StaticMatrix{<:Any, <:Any, T}}

# Being a member of StaticMatrixLike, StaticVecOrMatLike, or StaticArrayLike implies that Size(A)
# returns a static Size instance (none of the dimensions are Dynamic). The converse may not be true.
# These are akin to aliases like StridedArray and in similarly bad taste, but the current approach
# in Base necessitates their existence.
const StaticMatrixLike{T} = Union{
    StaticMatrix{<:Any, <:Any, T},
    Transpose{T, <:StaticVecOrMat{T}},
    Adjoint{T, <:StaticVecOrMat{T}},
    Symmetric{T, <:StaticMatrix{<:Any, <:Any, T}},
    Hermitian{T, <:StaticMatrix{<:Any, <:Any, T}},
    Diagonal{T, <:StaticVector{<:Any, T}},
    # We specifically list *Triangular here rather than using
    # AbstractTriangular to avoid ambiguities in size() etc.
    UpperTriangular{T, <:StaticMatrix{<:Any, <:Any, T}},
    LowerTriangular{T, <:StaticMatrix{<:Any, <:Any, T}},
    UnitUpperTriangular{T, <:StaticMatrix{<:Any, <:Any, T}},
    UnitLowerTriangular{T, <:StaticMatrix{<:Any, <:Any, T}}
}
const StaticVecOrMatLike{T} = Union{StaticVector{<:Any, T}, StaticMatrixLike{T}}
const StaticArrayLike{T} = Union{StaticVecOrMatLike{T}, StaticArray{<:Tuple, T}}

const AbstractScalar{T} = AbstractArray{T, 0} # not exported, but useful none-the-less
const StaticArrayNoEltype{S, N, T} = StaticArray{S, T, N}

include("util.jl")
include("traits.jl")

include("SUnitRange.jl")
include("FieldArray.jl")
include("SArray.jl")
include("SMatrix.jl")
include("SVector.jl")
include("Scalar.jl")
include("MArray.jl")
include("MVector.jl")
include("MMatrix.jl")
include("SizedArray.jl")
include("SDiagonal.jl")
include("SHermitianCompact.jl")

include("initializers.jl")
include("convert.jl")

include("abstractarray.jl")
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("sort.jl")
include("arraymath.jl")
include("linalg.jl")
include("matrix_multiply_add.jl")
include("matrix_multiply.jl")
include("lu.jl")
include("det.jl")
include("inv.jl")
include("solve.jl")
include("eigen.jl")
include("expm.jl")
include("sqrtm.jl")
include("lyap.jl")
include("triangular.jl")
include("cholesky.jl")
include("svd.jl")
include("qr.jl")
include("deque.jl")
include("flatten.jl")
include("io.jl")

if Base.VERSION >= v"1.4.2"
    include("precompile.jl")
    _precompile_()
end

end # module
