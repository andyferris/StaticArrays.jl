# API
## Guide


### `SVector`

The simplest static array is the type `SVector{N,T}`, which provides an
immutable vector of fixed length `N` and type `T`.

`SVector` defines a series of convenience constructors, so you can just type
e.g. `SVector(1,2,3)`. Alternatively there is an intelligent `@SVector` macro
where you can use native Julia array literals syntax, comprehensions, and the
`zeros()`, `ones()`, `fill()`, `rand()` and `randn()` functions, such as `@SVector [1,2,3]`,
`@SVector Float64[1,2,3]`, `@SVector [f(i) for i = 1:10]`, `@SVector zeros(3)`,
`@SVector randn(Float32, 4)`, etc (Note: the range of a comprehension is evaluated at global scope by the
macro, and must be made of combinations of literal values, functions, or global
variables, but is not limited to just simple ranges. Extending this to
(hopefully statically known by type-inference) local-scope variables is hoped
for the future. The `zeros()`, `ones()`, `fill()`, `rand()` and `randn()` functions do not have this
limitation.)

### `SMatrix`

Statically sized `N×M` matrices are provided by `SMatrix{N,M,T,L}`.

Here `L` is the `length` of the matrix, such that `N × M = L`. However,
convenience constructors are provided, so that `L`, `T` and even `M` are
unnecessary. At minimum, you can type `SMatrix{2}(1,2,3,4)` to create a 2×2
matrix (the total number of elements must divide evenly into `N`). A
convenience macro `@SMatrix [1 2; 3 4]` is provided (which also accepts
comprehensions and the `zeros()`, `ones()`, `fill()`, `rand()`, `randn()` and `eye()`
functions).

### `SArray`

A container with arbitrarily many dimensions is defined as
`struct SArray{Size,T,N,L} <: StaticArray{Size,T,N}`, where
`Size = Tuple{S1, S2, ...}` is a tuple of `Int`s. You can easily construct one with
the `@SArray` macro, supporting all the features of `@SVector` and `@SMatrix`
(but with arbitrary dimension).

The main reason `SVector` and `SMatrix` are defined is to make it easier to
define the types without the extra tuple characters (compare `SVector{3}` to
`SArray{Tuple{3}}`).

### `Scalar`

Sometimes you want to broadcast an operation, but not over one of your inputs.
A classic example is attempting to displace a collection of vectors by the
same vector. We can now do this with the `Scalar` type:

```julia
[[1,2,3], [4,5,6]] .+ Scalar([1,0,-1]) # [[2,2,2], [5,5,5]]
```

`Scalar` is simply an implementation of an immutable, 0-dimensional `StaticArray`.

### The `Size` trait

The size of a statically sized array is a static parameter associated with the
type of the array. The `Size` trait is provided as an abstract representation of
the dimensions of a static array. An array `sa::SA` of size `(dims...)` is
associated with `Size{(dims...)}()`. The following are equivalent (`@pure`)
constructors:
```julia
Size{(dims...,)}()
Size(dims...)
Size(sa::StaticArray)
Size(SA) # SA <: StaticArray
```
This is extremely useful for (a) performing dispatch depending on the size of an
array, and (b) passing array dimensions that the compiler can reason about.

An example of size-based dispatch for the determinant of a matrix would be:
```julia
det(x::StaticMatrix) = _det(Size(x), x)
_det(::Size{(1,1)}, x::StaticMatrix) = x[1,1]
_det(::Size{(2,2)}, x::StaticMatrix) = x[1,1]*x[2,2] - x[1,2]*x[2,1]
# and other definitions as necessary
```

Examples of using `Size` as a compile-time constant include
```julia
reshape(svector, Size(2,2))  # Convert SVector{4} to SMatrix{2,2}
SizedMatrix{3,3}(rand(3,3))  # Construct a random 3×3 SizedArray (see below)
```

### Indexing

Statically sized indexing can be realized by indexing each dimension by a
scalar, a `StaticVector` or `:`. Indexing in this way will result a statically
sized array (even if the input was dynamically sized, in the case of
`StaticVector` indices) of the closest type (as defined by `similar_type`).

Conversely, indexing a statically sized array with a dynamically sized index
(such as a `Vector{Integer}` or `UnitRange{Integer}`) will result in a standard
(dynamically sized) `Array`.

### `similar_type()`

Since immutable arrays need to be constructed "all-at-once", we need a way of
obtaining an appropriate constructor if the element type or dimensions of the
output array differs from the input. To this end, `similar_type` is introduced,
behaving just like `similar`, except that it returns a type. Relevant methods
are:

```julia
similar_type(::Type{A}) where {A <: StaticArray} # defaults to A
similar_type(::Type{A}, ::Type{ElType}) where {A <: StaticArray, ElType} # Change element type
similar_type(::Type{A}, size::Size) where {A <: AbstractArray} # Change size
similar_type(::Type{A}, ::Type{ElType}, size::Size) where {A <: AbstractArray, ElType} # Change both
```

These setting will affect everything, from indexing, to matrix multiplication
and `broadcast`. Users wanting introduce a new array type should *only* overload
the last method in the above.

Use of `similar` will fall back to a mutable container, such as a `MVector`
(see below), and it requires use of the `Size` trait if you wish to set a new
static size (or else a dynamically sized `Array` will be generated when
specifying the size as plain integers).

### Collecting directly into static arrays

You can collect [iterators](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration) into static arrays directly with [`StaticArrays.sacollect`](@ref). The size needs to be specified, but the element type is optional.

### Mutable arrays: `MVector`, `MMatrix` and `MArray`

These statically sized arrays are identical to the above, but are defined as
`mutable struct`s, instead of immutable `struct`s. Because they are mutable, they
allow `setindex!` to be defined (achieved through pointer manipulation, into a
tuple).

As a consequence of Julia's internal implementation, these mutable containers
live on the heap, not the stack. Their memory must be allocated and tracked by
the garbage collector. Nevertheless, there is opportunity for speed
improvements relative to `Base.Array` because (a) there may be one less
pointer indirection, (b) their (typically small) static size allows for
additional loop unrolling and inlining, and consequentially (c) their mutating
methods like `map!` are extremely fast. Benchmarking shows that operations such
as addition and matrix multiplication are faster for `MMatrix` than `Matrix`,
at least for sizes up to 14 × 14, though keep in mind that optimal speed will
be obtained by using mutating functions (like `map!` or `A_mul_B!`) where
possible, rather than reallocating new memory.

Mutable static arrays also happen to be very useful containers that can be
constructed on the heap (with the ability to use `setindex!`, etc), and later
copied as e.g. an immutable `SVector` to the stack for use, or into e.g. an
`Array{SVector}` for storage.

Convenience macros `@MVector`, `@MMatrix` and `@MArray` are provided.

### `SizedArray`: a decorate size wrapper for `Array`

Another convenient mutable type is the `SizedArray`, which is just a wrapper-type
about a standard Julia `Array` which declares its known size. For example, if
we knew that `a` was a 2×2 `Matrix`, then we can type `sa = SizedArray{Tuple{2,2}}(a)`
to construct a new object which knows the type (the size will be verified
automatically). For one and two dimensions, a more convenient syntax for
obtaining a `SizedArray` is by using the `SizedMatrix` and `SizedVector`
aliases, e.g. `sa = SizedMatrix{2,2}(a)`.

Then, methods on `sa` will use the specialized code provided by the *StaticArrays*
package, which in many cases will be much, much faster. For example, calling
`eigen(sa)` will be signficantly faster than `eigen(a)` since it will perform a
specialized 2×2 matrix diagonalization rather than a general algorithm provided
by Julia and *LAPACK*.

In some cases it will make more sense to use a `SizedArray`, and in other cases
an `MArray` might be preferable.

### `FieldVector`

Sometimes it is useful to give your own struct types the properties of a vector.
*StaticArrays* can take care of this for you by allowing you to inherit from
`FieldVector{N, T}`. For example, consider:

```julia
struct Point3D <: FieldVector{3, Float64}
    x::Float64
    y::Float64
    z::Float64
end
```

With this type, users can easily access fields to `p = Point3D(x,y,z)` using
`p.x`, `p.y` or `p.z`, or alternatively via `p[1]`, `p[2]`, or `p[3]`. You may
even permute the coordinates with `p[SVector(3,2,1)]`). Furthermore, `Point3D`
is a complete `AbstractVector` implementation where you can add, subtract or
scale vectors, multiply them by matrices, etc.

It is also worth noting that `FieldVector`s may be mutable or immutable, and
that `setindex!` is defined for use on mutable types. For immutable containers,
you may want to define a method for `similar_type` so that operations leave the
type constant (otherwise they may fall back to `SVector`). For mutable
containers, you may want to define a default constructor (no inputs) and an
appropriate method for `similar`,

### Implementing your own types

You can easily create your own `StaticArray` type, by defining linear
`getindex` (and optionally `setindex!` for mutable types --- see
`setindex!(::MArray, val, i)` in *MArray.jl* for an example of how to
achieve this through pointer manipulation). Your type should define a constructor
that takes a tuple of the data (and mutable containers may want to define a
default constructor).

Other useful functions to overload may be `similar_type` (and `similar` for
mutable containers).

### Conversions from `Array`

In order to convert from a dynamically sized `AbstractArray` to one of the
statically sized array types, you must specify the size explicitly.  For
example,

```julia
v = [1,2]

m = [1 2;
     3 4]

# ... a lot of intervening code

sv = SVector{2}(v)
sm = SMatrix{2,2}(m)
sa = SArray{Tuple{2,2}}(m)

sized_v = SizedVector{2}(v)
sized_m = SizedMatrix{2,2}(m)
```

We have avoided adding `SVector(v::AbstractVector)` as a valid constructor to
help users avoid the type instability (and potential performance disaster, if
used without care) of this innocuous looking expression.

### Arrays of static arrays

Storing a large number of static arrays is convenient as an array of static
arrays. For example, a collection of positions (3D coordinates --- `SVector{3,Float64}`)
could be represented as a `Vector{SVector{3,Float64}}`.

Another common way of storing the same data is as a 3×`N` `Matrix{Float64}`.
Rather conveniently, such types have *exactly* the same binary layout in memory,
and therefore we can use `reinterpret` to convert between the two formats
```julia
function svectors(x::Matrix{T}, ::Val{N}) where {T,N}
    size(x,1) == N || error("sizes mismatch")
    isbitstype(T) || error("use for bitstypes only")
    reinterpret(SVector{N,T}, vec(x))
end
```
Such a conversion does not copy the data, rather it refers to the *same* memory.
Arguably, a `Vector` of `SVector`s is often preferable to a `Matrix` because it
provides a better abstraction of the objects contained in the array and it
allows the fast *StaticArrays* methods to act on elements.

However, the resulting object is a Base.ReinterpretArray, not an Array, which
carries some runtime penalty on every single access. If you can afford the
memory for a copy and can live with the non-shared mutation semantics, then it
is better to pull a copy by e.g.
```julia
function svectorscopy(x::Matrix{T}, ::Val{N}) where {T,N}
    size(x,1) == N || error("sizes mismatch")
    isbitstype(T) || error("use for bitstypes only")
    copy(reinterpret(SVector{N,T}, vec(x)))
end
```
For example:
```
julia> M=reshape(collect(1:6), (2,3))
2×3 Array{Int64,2}:
 1  3  5
 2  4  6

julia> svectors(M, Val{2}())
3-element reinterpret(SArray{Tuple{2},Int64,1,2}, ::Array{Int64,1}):
 [1, 2]
 [3, 4]
 [5, 6]

julia> svectorscopy(M, Val{2}())
3-element Array{SArray{Tuple{2},Int64,1,2},1}:
 [1, 2]
 [3, 4]
 [5, 6]
```

### Working with mutable and immutable arrays

Generally, it is performant to rebind an *immutable* array, such as
```julia
function average_position(positions::Vector{SVector{3,Float64}})
    x = zeros(SVector{3,Float64})
    for pos ∈ positions
        x = x + pos
    end
    return x / length(positions)
end
```
so long as the `Type` of the rebound variable (`x`, above) does not change.

On the other hand, the above code for mutable containers like `Array`, `MArray`
or `SizedArray` is *not* very efficient. Mutable containers in Julia 0.5 must
be *allocated* and later *garbage collected*, and for small, fixed-size arrays
this can be a leading contribution to the cost. In the above code, a new array
will be instantiated and allocated on each iteration of the loop. In order to
avoid unnecessary allocations, it is best to allocate an array only once and
apply mutating functions to it:
```julia
function average_position(positions::Vector{SVector{3,Float64}})
    x = zeros(MVector{3,Float64})
    for pos ∈ positions
        # Take advantage of Julia 0.5 broadcast fusion
        x .= (+).(x, pos) # same as broadcast!(+, x, x, positions[i])
    end
    x .= (/).(x, length(positions))
    return x
end
```
Keep in mind that Julia 0.5 does not fuse calls to `.+`, etc (or `.+=` etc),
however the `.=` and `(+).()` syntaxes are fused into a single, efficient call
to `broadcast!`. The simpler syntax `x .+= pos` is expected to be non-allocating
(and therefore faster) in Julia 0.6.

The functions `setindex`, `push`, `pop`, `pushfirst`, `popfirst`, `insert` and `deleteat`
are provided for performing certain specific operations on static arrays, in
analogy with the standard functions `setindex!`, `push!`, `pop!`, etc. (Note that
if the size of the static array changes, the type of the output will differ from
the input.)

When building static arrays iteratively, it is usually efficient to build up an `MArray` first and then convert. The allocation will be elided by recent Julia compilers, resulting in very efficient code:
```julia
function standard_basis_vector(T, ::Val{I}, ::Val{N}) where {I,N}
    v = zero(MVector{N,T})
    v[I] = one(T)
    SVector(v)
end
```

### SIMD optimizations

It seems Julia and LLVM are smart enough to use processor vectorization
extensions like SSE and AVX - however they are currently partially disabled by
default. Run Julia with `julia -O` or `julia -O3` to enable these optimizations,
and many of your (immutable) `StaticArray` methods *should* become significantly
faster!

## Docstrings

```@index
Pages   = ["api.md"]
```
```@autodocs
Modules = [StaticArrays]
```
