
# Some unbound_args are hard to avoid, e.g. `SVector{N}(::NTuple{N,T})`
# if `N == 0`, then `T` is unbound. We could disallow calling
# `SVector{0}(())`, but throwing a `T not defined` error seems benign.
# Working around this, e.g. with `SVector{N}(::Tuple{T,Vararg{T,Nm1}}) where {N,T,Nm1}`
# and then asserting `N == Nm1+1` also seems reasonable (the compiler should eliminate the assert).
const allowable_unbound_args = 0

@test length(detect_unbound_args(StaticArrays)) <= allowable_unbound_args
