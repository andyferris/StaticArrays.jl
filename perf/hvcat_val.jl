using StaticArray, BenchmarkTools

let
rows, cols = 4, 4
_dims = Expr(:tuple, [cols for _ in 1:rows]...)

for (f, wrap_val) in [(:f1, false), (:f2, true)]
    dims = wrap_val ? :(Val{$_dims}()) : _dims
    zeros_sa = :(Base.typed_hvcat(SA, $dims, $([0 for _ in 1:rows*cols]...)))
    xs = [Symbol(:x, i) for i in 1:rows*cols]
    is = [Symbol(:i, i) for i in 1:rows*cols]
    is_sa = :(Base.typed_hvcat(SA, $dims, $(is...)))
    @eval begin
        function $f($(xs...))
            r = $zeros_sa
            for ($(is...),) in Iterators.product($(xs...))
                r += $is_sa
            end
            r
        end
    end
end

xs = [:(1:2) for _ in 1:rows*cols]
display(@eval @benchmark f1($(xs...)))
display(@eval @benchmark f2($(xs...)))
end
