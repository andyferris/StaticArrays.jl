fsa = false
all_methods = false

using StaticArrays
@static if fsa
    using FixedSizeArrays
end

if !@isdefined(N)
    N = 4
end

M_f = div(10^9, N^3)
M_g = div(2*10^8, N^2)

# Size

A = rand(Float64,N,N)
A = A*A'
As = SMatrix{N,N}(A)
Am = MMatrix{N,N}(A)
Az = SizedMatrix{N,N}(copy(A))
@static if fsa
    Af = Mat(ntuple(j -> ntuple(i->A[i,j], N), N)) # there is a bug in FixedSizeArrays Mat constructor (13 July 2016)
end


if !@isdefined(f_mut_marray) || !@isdefined(benchmark_suite) || benchmark_suite == false
    @generated f(n::Integer, A) = :(@inbounds (C = A; for i = 1:n; C = C*A; end; return C))
    @generated f_unrolled(n::Integer, A::Union{SMatrix{M,M},MMatrix{M,M}}) where {M} = :(@inbounds (C = A; for i = 1:n; C = StaticArrays.A_mul_B_unrolled(C,A); end; return C))
    @generated f_unrolled_chunks(n::Integer, A::Union{SMatrix{M,M},MMatrix{M,M}}) where {M} = :(@inbounds (C = A; for i = 1:n; C = StaticArrays.A_mul_B_unrolled_chunks(C,A); end; return C))
    @generated f_loop(n::Integer, A::Union{SMatrix{M,M},MMatrix{M,M}}) where {M} = :(@inbounds (C = A; for i = 1:n; C = StaticArrays.A_mul_B_loop(C,A); end; return C))
    @generated f_via_sarray(n::Integer, A::MMatrix{M,M}) where {M}= :(@inbounds (C = A; for i = 1:n; C = MMatrix{M,M}(SMatrix{M,M}(C)*SMatrix{M,M}(A)); end; return C))
    @generated f_mut_array(n::Integer, A) = :(@inbounds (C = copy(A); tmp = similar(A); for i = 1:n;  mul!(tmp, C, A); map!(identity, C, tmp); end; return C))
    @generated f_mut_marray(n::Integer, A) = :(@inbounds (C = similar(A); C[:] = A[:]; tmp = similar(A); for i = 1:n; StaticArrays.mul!(tmp, C, A); C.data = tmp.data; end; return C))
    @generated f_mut_unrolled(n::Integer, A) = :(@inbounds (C = similar(A); C[:] = A[:]; tmp = similar(A); for i = 1:n; StaticArrays.A_mul_B_unrolled!(tmp, C, A); C.data = tmp.data; end; return C))
    @generated f_mut_chunks(n::Integer, A) = :(@inbounds (C = similar(A); C[:] = A[:]; tmp = similar(A); for i = 1:n; StaticArrays.A_mul_B_unrolled_chunks!(tmp, C, A); C.data = tmp.data; end; return C))
    @generated f_blas_marray(n::Integer, A) = :(@inbounds (C = similar(A); C[:] = A[:]; tmp = similar(A); for i = 1:n; StaticArrays.A_mul_B_blas!(tmp, C, A); C.data = tmp.data; end; return C))

    @generated g(n::Integer, A) = :(@inbounds (C = A; for i = 1:n; C = C + A; end; return C))
    @generated g_mut(n::Integer, A) = :(@inbounds (C = copy(A); for i = 1:n; @inbounds map!(+, C, C, A); end; return C))
    @generated g_via_sarray(n::Integer, A::MMatrix{M,M}) where {M} = :(@inbounds (C = similar(A); C[:] = A[:]; for i = 1:n; C = MMatrix{M,M}(SMatrix{M,M}(C) + SMatrix{M,M}(A)); end; return C))

    @noinline _det(x) = det(x)
    @noinline _inv(x) = inv(x)
    @noinline _eig(x) = eigen(x)
    @noinline _chol(x) = cholesky(x)

    f_det(n::Int, A) = (for i = 1:n; _det(A); end)
    f_inv(n::Int, A) = (for i = 1:n; _inv(A); end)
    f_eig(n::Int, A) = (for i = 1:n; _eig(A); end)
    f_chol(n::Int, A) = (for i = 1:n; _chol(A); end)

    # Notes: - A[:] = B[:] is allocating in Base, unlike `map!`
    #        - Also, the same goes for Base's implementation of broadcast!(f, A, B, C) (but map! is OK).
    #        - I really need to implement copy() in StaticArrays... (and maybe a special instance of setindex!(C, :))
    #        - I tried generated functions here to see if they force the compiler to specialize more.

end

# Test and compile f()

C = f(2, A)
C_mut = f_mut_array(2, A)
if N <= 4;  @assert C_mut ≈ C; end

println("=====================================")
println("    Benchmarks for $N×$N matrices")
println("=====================================")
@static if all_methods
    print("SMatrix * SMatrix compilation time (unrolled):         ")
    @time eval(quote
        Cs_unrolled = f_unrolled(2, As)
        Cs_unrolled::SMatrix
        if N <= 4; @assert Cs_unrolled ≈ C; end
    end)

    print("SMatrix * SMatrix compilation time (chunks):           ")
    @time eval(quote
        Cs_chunks = f_unrolled_chunks(2, As)
        Cs_chunks::SMatrix
        if N <= 4; @assert Cs_chunks ≈ C; end
    end)

    print("SMatrix * SMatrix compilation time (loop):           ")
    @time eval(quote
        Cs_loop = f_loop(2, As)
        Cs_loop::SMatrix
        if N <= 4; @assert Cs_loop ≈ C; end
    end)

    print("MMatrix * MMatrix compilation time (unrolled):         ")
    @time eval(quote
        Cm_unrolled = f_unrolled(2, Am)
        Cm_unrolled::MMatrix
        if N <= 4; @assert Cm_unrolled ≈ C; end
    end)

    print("MMatrix * MMatrix compilation time (chunks):           ")
    @time eval(quote
        Cm_chunks = f_unrolled_chunks(2, Am)
        Cm_chunks::MMatrix
        if N <= 4; @assert Cm_chunks ≈ C; end
    end)

    print("MMatrix * MMatrix compilation time (loop):           ")
    @time eval(quote
        Cm_loop = f_loop(2, Am)
        Cm_loop::MMatrix
        if N <= 4; @assert Cm_loop ≈ C; end
    end)

    Cm_via_sarray = f_via_sarray(2, Am)
    Cm_via_sarray::MMatrix
    if N <= 4; @assert Cm_via_sarray ≈ C; end
else
    #Cs_unrolled = f_unrolled(2, As)
    #Cs_unrolled::SMatrix
    #if N <= 4; @assert Cs_unrolled ≈ C; end

    #Cs_chunks = f_unrolled_chunks(2, As)
    #Cs_chunks::SMatrix
    #if N <= 4; @assert Cs_chunks ≈ C; end

    #Cm_unrolled = f_unrolled(2, Am)
    #Cm_unrolled::MMatrix
    #if N <= 4; @assert Cm_unrolled ≈ C; end

    #Cm_chunks = f_unrolled_chunks(2, Am)
    #Cm_chunks::MMatrix
    #if N <= 4; @assert Cm_chunks ≈ C; end
end

# Warmup and some checks
Cs = f(2, As)
Cm = f(2, Am)
Cz = f(2, Az)

Cs::SMatrix
if N <= 4; @assert Cs ≈ C; end
Cm::MMatrix
if N <= 4; @assert Cm ≈ C; end
Cz::SArray
if N <= 4; @assert Cz ≈ C; end

@static if fsa
    @static if all_methods
        print("Mat * Mat compilation time:                            ")
        @time eval(quote
            Cf = f(2, Af)
            Cf::Mat
            if N <= 4; @assert Cf == C; end
        end)
    else
        Cf = f(2, Af)
        Cf::Mat
        if N <= 4; @assert Cf == C; end
    end
end

# Mutating versions

Cm_mut = f_mut_marray(2, Am)
Cm_mut::MMatrix
if N <= 4; @assert Cm_mut ≈ C; end

Cz_mut = f_mut_array(2, Az)
Cz_mut::SizedMatrix
if N <= 4; @assert Cz_mut ≈ C; end

@static if all_methods
    println()
    print("mul!(MMatrix, MMatrix) compilation time (unrolled):")
    @time eval(quote
        Cm_mut_unrolled = f_mut_unrolled(2, Am)
        Cm_mut_unrolled::MMatrix
        if N <= 4; @assert Cm_mut_unrolled ≈ C; end
    end)

    print("mul!(MMatrix, MMatrix) compilation time (chunks):  ")
    @time eval(quote
        Cm_mut_chunks = f_mut_chunks(2, Am)
        Cm_mut_chunks::MMatrix
        if N <= 4; @assert Cm_mut_chunks ≈ C; end
    end)

    print("mul!(MMatrix, MMatrix) compilation time (BLAS):    ")
    @time eval(quote
        Cm_blas = f_blas_marray(2, Am)
        Cm_blas::MMatrix
        if N <= 4; @assert Cm_blas ≈ C; end
    end)
else
    #Cm_mut_unrolled = f_mut_marray(2, Am)
    #Cm_mut_unrolled::MMatrix
    #if N <= 4; @assert Cm_mut_unrolled ≈ C; end

    #Cm_mut_chunks = f_mut_chunks(2, Am)
    #Cm_mut_chunks::MMatrix
    #if N <= 4; @assert Cm_mut_chunks ≈ C; end

    #Cm_blas = f_blas_marray(2, Am)
    #Cm_blas::MMatrix
    #if N <= 4; @assert Cm_blas ≈ C; end
end


# Test and compile g()

C = g(2, A)
C_mut = g_mut(2, A)

if N <= 4; @assert C_mut ≈ C; end

Cs = g(2, As)
Cm = g(2, Am)
Cm_mut = g_mut(2, Am)
Cz = g(2, Az)
Cz_mut = g_mut(2, Az)

Cs::SMatrix
if N <= 4; @assert Cs == C; end
Cm::MMatrix
if N <= 4; @assert Cm == C; end
Cm_mut::MMatrix
if N <= 4; @assert Cm_mut == C; end
Cz::SMatrix
if N <= 4; @assert Cz == C; end
Cz_mut::SizedMatrix
if N <= 4; @assert Cz_mut == C; end

@static if all_methods
    Cm_via_sarray = g_via_sarray(2, Am)
    Cm_via_sarray::MMatrix
    if N <= 4; @assert Cm_via_sarray == C; end
end

@static if fsa
    Cf = g(2, Af)
    Cf::Mat
    if N <= 4; @assert Cf == C; end
end

if N <= 3
    # det, eig etc
    C = f_det(2, A)
    Cs = f_det(2, As)
    Cm = f_det(2, Am)
    Cz = f_det(2, Az)

    C = f_inv(2, A)
    Cs = f_inv(2, As)
    Cm = f_inv(2, Am)
    Cz = f_inv(2, Az)

    C = f_eig(2, Symmetric(A))
    Cs = f_eig(2, Symmetric(As))
    Cm = f_eig(2, Symmetric(Am))
    Cz = f_eig(2, Symmetric(Az))

    C = f_chol(2, Symmetric(A))
    Cs = f_chol(2, Symmetric(As))
    Cm = f_chol(2, Symmetric(Am))
    Cz = f_chol(2, Symmetric(Az))
end

println()

# Do the performance tests

println("Matrix multiplication")
println("---------------------")
begin
   print("Array               ->"); @time f(M_f, A)
   @static if fsa
       print("Mat                 ->"); @time f(M_f, Af)
   end
   print("SArray              ->"); @time f(M_f, As)
   print("MArray              ->"); @time f(M_f, Am)
   print("SizedArray          ->"); @time f(M_f, Az)
   @static if all_methods
       print("SArray (unrolled)   ->"); @time f_unrolled(M_f, As)
       print("SArray (chunks)     ->"); @time f_unrolled_chunks(M_f, As)
       print("SArray (loop)       ->"); @time f_loop(M_f, As)
       print("MArray (unrolled)   ->"); @time f_unrolled(M_f, Am)
       print("MArray (chunks)     ->"); @time f_unrolled_chunks(M_f, Am)
       print("MArray (loop)       ->"); @time f_loop(M_f, Am)
       print("MArray (via SArray) ->"); @time f_via_sarray(M_f, Am)
   end
end
println()

println("Matrix multiplication (mutating)")
println("--------------------------------")
begin
   print("Array               ->"); @time f_mut_array(M_f, A)
   print("MArray              ->"); @time f_mut_marray(M_f, Am)
   print("SizedArray          ->"); @time f_mut_array(M_f, Az)
   @static if all_methods
       print("MArray (unrolled)   ->"); @time f_mut_unrolled(M_f, Am)
       print("MArray (chunks)     ->"); @time f_mut_chunks(M_f, Am)
       print("MArray (BLAS gemm!) ->"); @time f_blas_marray(M_f, Am)
   end
end
println()

println("Matrix addition")
println("---------------")
begin
   print("Array               ->"); @time g(M_g, A)
   @static if fsa
       print("Mat                 ->"); @time g(M_g, Af)
   end
   print("SArray              ->"); @time g(M_g, As)
   print("MArray              ->"); @time g(M_g, Am)
   print("SizedArray          ->"); @time g(M_g, Az)
   @static if all_methods
       print("MArray (via SArray) ->"); @time g_via_sarray(M_g, Am)
   end
end
println()

println("Matrix addition (mutating)")
println("--------------------------")
begin
   print("Array      ->"); @time g_mut(M_g, A) # broadcast! seems to be broken!
   print("MArray     ->"); @time g_mut(M_g, Am)
   print("SizedArray ->"); @time g_mut(M_g, Az)
end
println()

if N <= 3
    println("Matrix determinant")
    println("------------------")
    begin
       print("Array      ->"); @time f_det(M_f, A)
       print("SArray     ->"); @time f_det(M_f, As)
       print("MArray     ->"); @time f_det(M_f, Am)
       print("SizedArray ->"); @time f_det(M_f, Az)
    end
    println()

    println("Matrix inverse")
    println("--------------")
    begin
       print("Array      ->"); @time f_inv(M_f, A)
       print("SArray     ->"); @time f_inv(M_f, As)
       print("MArray     ->"); @time f_inv(M_f, Am)
       print("SizedArray ->"); @time f_inv(M_f, Az)
    end
    println()

    println("Matrix symmetric eigenvalue")
    println("---------------------------")
    begin
       print("Array      ->"); @time f_eig(M_f, Symmetric(A))
       print("SArray     ->"); @time f_eig(M_f, Symmetric(As))
       print("MArray     ->"); @time f_eig(M_f, Symmetric(Am))
       print("SizedArray ->"); @time f_eig(M_f, Symmetric(Az))
    end
    println()

    println("Matrix Cholesky")
    println("---------------")
    begin
       print("Array      ->"); @time f_chol(M_f, Symmetric(A))
       print("SArray     ->"); @time f_chol(M_f, Symmetric(As))
       print("MArray     ->"); @time f_chol(M_f, Symmetric(Am))
       print("SizedArray ->"); @time f_chol(M_f, Symmetric(Az))
    end
    println()
end #if N <= 3
