using StaticArrays
using BenchmarkTools
using LinearAlgebra: qr

a = m = 0
for K = 1:22
    a = rand(SMatrix{K,K,Float64,K*K})
    m = Matrix(a)
    print("Size($K,$K)\n  Compilation time:")
    @time qr(a)
    @btime qr($a)
    @btime qr($m)
end
