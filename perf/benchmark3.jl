using StaticArrays
using BenchmarkTools

import BenchmarkTools: prettytime, prettymemory
using Statistics: mean

@noinline plus(a,b) = a+b
@noinline plus!(c,a,b) = broadcast!(+, c, a, b)

@noinline mul(a,b) = a*b

for T ∈ [Int64, Float64]
    for N ∈ [1,2,4,8,16,32,64,128,256]
        println("=====================================================================")
        println(" Vectors of length ", N, " and eltype ", T)
        println("=====================================================================")
        immutables = [rand(SVector{N,T})]
        mutables = [rand(T,N), rand(MVector{N,T}), SizedVector{N}(rand(T,N))]
        instances = vcat(immutables, mutables)

        namelengths = [length(string(typeof(v).name.name)) for v ∈ instances]
        maxnamelength = maximum(namelengths)

        for v ∈ instances
            result = mean(@benchmark plus($(copy(v)), $(copy(v))))
            padding = maxnamelength - length(string(typeof(v).name.name))
            println(typeof(v).name.name, ":", " " ^ padding, " v3 = v1 + v2 takes ", prettytime(time(result)), ", ", prettymemory(memory(result)), " (GC ", prettytime(gctime(result)) , ")")
        end

        println()

        for v ∈ mutables
            result = mean(@benchmark plus!($(copy(v)), $(copy(v)), $(copy(v))))
            padding = maxnamelength - length(string(typeof(v).name.name))
            println(typeof(v).name.name, ":", " " ^ padding, " v3 .= +.(v1, v2) takes ", prettytime(time(result)), ", ", prettymemory(memory(result)), " (GC ", prettytime(gctime(result)) , ")")
        end

        println()

        if N > 16
            continue
        end
        println("=====================================================================")
        println(" Matrices of size ", N, "×", N, " and eltype ", T)
        println("=====================================================================")
        immutables = [rand(SMatrix{N,N,T})]
        mutables = [rand(T,N,N), rand(MMatrix{N,N,T}), SizedMatrix{N,N}(rand(T,N,N))]
        instances = vcat(immutables, mutables)

        namelengths = [length(string(typeof(v).name.name)) for v ∈ instances]
        maxnamelength = maximum(namelengths)

        for m ∈ instances
            result = mean(@benchmark mul($(copy(m)), $(copy(m))))
            padding = maxnamelength - length(string(typeof(m).name.name))
            println(typeof(m).name.name, ":", " " ^ padding, " m3 = m1 * m2 takes ", prettytime(time(result)), ", ", prettymemory(memory(result)), " (GC ", prettytime(gctime(result)) , ")")
        end

        println()

        for m ∈ mutables
            result = mean(@benchmark mul!($(copy(m)), $(copy(m)), $(copy(m))))
            padding = maxnamelength - length(string(typeof(m).name.name))
            println(typeof(m).name.name, ":", " " ^ padding, " mul!(m3, m1, m2) takes ", prettytime(time(result)), ", ", prettymemory(memory(result)), " (GC ", prettytime(gctime(result)) , ")")
        end

        println()

    end
end
