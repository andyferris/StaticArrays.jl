"""
    x ≊ y

Inexact equality comparison. Like `≈` this calls `isapprox`, but with a
tighter tolerance of `rtol=10*eps()`.  Input with "\\approxeq".
"""
≊(x,y) = isapprox(x, y, rtol=10*eps())

"""
    @testinf a op b

Test that the type of the first argument `a` is inferred, and that `a op b` is
true.  For example, the following are equivalent:

    @testinf SVector(1,2) + SVector(1,2) == SVector(2,4)
    @test @inferred(SVector(1,2) + SVector(1,2)) == SVector(2,4)
"""
macro testinf(ex)
    @assert ex.head == :call
    infarg = ex.args[2]
    if !(infarg isa Expr) || infarg.head != :call
        # Workaround for an oddity in @inferred
        infarg = :(identity($infarg))
    end
    ex.args[2] = :(@inferred($infarg))
    esc(:(@test $ex))
end

function test_expand_error(ex)
    @test_throws LoadError macroexpand(@__MODULE__, ex)
end

mutable struct ErrorCounterTestSet <: Test.AbstractTestSet
    passcount::Int
    errorcount::Int
    failcount::Int
end
ErrorCounterTestSet(args...; kws...) = ErrorCounterTestSet(0,0,0)
Test.finish(ts::ErrorCounterTestSet) = ts
Test.record(ts::ErrorCounterTestSet, ::Test.Pass)  = (ts.passcount += 1)
Test.record(ts::ErrorCounterTestSet, ::Test.Error) = (ts.errorcount += 1)
Test.record(ts::ErrorCounterTestSet, ::Test.Fail)  = (ts.failcount += 1)

"""
    @test_inlined f(x,y, ...)

Check that the (optimized) llvm code generated for the expression
`f(x,y,...)` contains no `call` instructions.

Note that LLVM IR can contain `call` instructions to intrinsics which don't
make it into the native code, so this can be overly eager in declaring a
a lack of complete inlining.
"""
macro test_inlined(ex, should_inline=true)
    ex_orig = ex
    ex = macroexpand(@__MODULE__, :(@code_llvm $ex))
    expr = quote
        code_str = sprint() do io
            code_llvm(io, $(map(esc, ex.args[2:end])...))
        end
        # Crude detection of call instructions remaining within what should be
        # fully inlined code.
        #
        # TODO: Figure out some better pattern matching; LLVM IR can contain
        # calls to intrinsics, so this will sometimes/often fail even when the
        # native code has no call instructions.
        $(should_inline ?
          :(@test !occursin("call", code_str)) :
          :(@test occursin("call", code_str))
        )
    end
    @assert expr.args[4].head == :macrocall
    expr.args[4].args[2] = __source__
    expr
end

should_be_inlined(x) = x*x
@noinline _should_not_be_inlined(x) = x*x
should_not_be_inlined(x) = _should_not_be_inlined(x)

"""
    @test_const_fold f(args...)

Test that constant folding works with a function call `f(args...)`.
"""
macro test_const_fold(ex)
    quote
        ci, = $(esc(:($InteractiveUtils.@code_typed optimize = true $ex)))
        @test $(esc(ex)) == constant_return(ci)
    end
end

struct NonConstantValue end

function constant_return(ci)
    if :rettype in fieldnames(typeof(ci))
        ci.rettype isa Core.Compiler.Const && return ci.rettype.val
        return NonConstantValue()
    else
        # for julia < 1.2
        ex = ci.code[end]
        Meta.isexpr(ex, :return) || return NonConstantValue()
        val = ex.args[1]
        return val isa QuoteNode ? val.value : val
    end
end

@testset "@test_const_fold" begin
    should_const_fold() = (1, 2, 3)
    @test_const_fold should_const_fold()

    x = Ref(1)
    should_not_const_fold() = x[]
    ts = @testset ErrorCounterTestSet "" begin
        @test_const_fold should_not_const_fold()
    end
    @test ts.errorcount == 0 && ts.failcount == 1 && ts.passcount == 0
end

"""
    @inferred_maybe_allow allow ex

Expands to `@inferred allow ex` on Julia 1.2 and newer and
`ex` on Julia 1.0 and 1.1.
"""
macro inferred_maybe_allow(allow, ex)
    if VERSION < v"1.2"
        return esc(:($ex))
    else
        return esc(:(@inferred $allow $ex))
    end
end

@testset "test utils" begin
    @testset "@testinf" begin
        @testinf [1,2] == [1,2]
        x = [1,2]
        @testinf x == [1,2]
        @testinf (@SVector [1,2]) == (@SVector [1,2])
    end

    @testset "@test_inlined" begin
        @test_inlined should_be_inlined(1)
        @test_inlined should_not_be_inlined(1) false
        ts = @testset ErrorCounterTestSet "" begin
            @test_inlined should_be_inlined(1) false
            @test_inlined should_not_be_inlined(1)
        end
        @test ts.errorcount == 0 && ts.failcount == 2 && ts.passcount == 0
    end
end
