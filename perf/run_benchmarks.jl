benchmark_suite = true
using LinearAlgebra

for N = 2:16
    include("benchmark2.jl")
end

benchmark_suite = false;
