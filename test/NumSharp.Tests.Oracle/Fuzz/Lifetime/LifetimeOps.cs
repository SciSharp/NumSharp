using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Fuzz.Lifetime
{
    /// <summary>
    ///     The calls of the lifetime oracle: op key -> the NumSharp call, with every operand built by its maker INSIDE
    ///     the call expression. Pairs one to one with <c>OPS</c> in <c>test/oracle/gen_lifetime_oracle.py</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why the makers are invoked inline.</b> <c>np.take(a[0](), a[1]())</c> evaluates both makers into
    ///     evaluation-stack temporaries and passes them straight to the call; in optimized code nothing in this frame
    ///     reports them to the GC after the call starts, so inside NumSharp each operand is reachable only through the
    ///     call's own parameters — the situation of a caller that passes a temporary. A local holding the operand here
    ///     would keep it alive and hide exactly the bug the tier exists to catch, so the only operand ever stored in a
    ///     local is a TARGET the call writes in place (its result is the target itself, which the caller does hold).
    ///     </para>
    ///     <para>
    ///     <b>Why NoInlining.</b> <see cref="Invoke"/> is one large switch; inlining the per-op lambdas into the child's
    ///     repetition loop would let the JIT keep their temporaries in the loop's frame. Each op body is its own method
    ///     (a lambda), invoked through a delegate.
    ///     </para>
    /// </remarks>
    internal static class LifetimeOps
    {
        /// <summary>One op body: the operand makers and the case (its parameters) -> the call's result.</summary>
        private delegate object Body(Func<NDArray>[] a, LifetimeCase c);

        /// <summary>The registry. A key the corpus names but this table lacks fails the case loudly.</summary>
        private static readonly Dictionary<string, Body> Table = new(StringComparer.Ordinal)
        {
            // ---- elementwise unary
            ["sqrt"] = (a, c) => np.sqrt(a[0]()),
            ["abs"] = (a, c) => np.abs(a[0]()),
            ["negative"] = (a, c) => np.negative(a[0]()),
            ["floor"] = (a, c) => np.floor(a[0]()),
            ["ceil"] = (a, c) => np.ceil(a[0]()),
            ["trunc"] = (a, c) => np.trunc(a[0]()),
            ["rint"] = (a, c) => np.rint(a[0]()),
            ["square"] = (a, c) => np.square(a[0]()),
            ["reciprocal"] = (a, c) => np.reciprocal(a[0]()),
            ["isnan"] = (a, c) => np.isnan(a[0]()),
            ["isfinite"] = (a, c) => np.isfinite(a[0]()),
            ["signbit"] = (a, c) => np.signbit(a[0]()),
            ["invert"] = (a, c) => np.invert(a[0]()),
            ["logical_not"] = (a, c) => np.logical_not(a[0]()),
            ["fabs"] = (a, c) => np.fabs(a[0]()),
            ["sign"] = (a, c) => np.sign(a[0]()),
            ["positive"] = (a, c) => np.positive(a[0]()),
            ["conjugate"] = (a, c) => np.conjugate(a[0]()),
            ["real"] = (a, c) => np.real(a[0]()).copy(),
            ["nan_to_num"] = (a, c) => np.nan_to_num(a[0]()),
            ["round"] = (a, c) => np.round_(a[0](), c.IntParam("decimals") ?? 0),
            ["exp"] = (a, c) => np.exp(a[0]()),
            ["sin"] = (a, c) => np.sin(a[0]()),
            ["sqrt_out"] = (a, c) => { var o = a[1](); np.sqrt(a[0](), o); return o; },
            ["sqrt_where"] = (a, c) => { var o = a[1](); np.sqrt(a[0](), o, a[2]()); return o; },
            ["modf"] = (a, c) => np.modf(a[0]()),
            ["frexp"] = (a, c) => np.frexp(a[0]()),
            // ---- elementwise binary
            ["add"] = (a, c) => np.add(a[0](), a[1]()),
            ["subtract"] = (a, c) => np.subtract(a[0](), a[1]()),
            ["multiply"] = (a, c) => np.multiply(a[0](), a[1]()),
            ["divide"] = (a, c) => np.divide(a[0](), a[1]()),
            ["floor_divide"] = (a, c) => np.floor_divide(a[0](), a[1]()),
            ["mod"] = (a, c) => np.mod(a[0](), a[1]()),
            ["fmod"] = (a, c) => np.fmod(a[0](), a[1]()),
            ["power"] = (a, c) => np.power(a[0](), a[1]()),
            ["maximum"] = (a, c) => np.maximum(a[0](), a[1]()),
            ["minimum"] = (a, c) => np.minimum(a[0](), a[1]()),
            ["fmax"] = (a, c) => np.fmax(a[0](), a[1]()),
            ["copysign"] = (a, c) => np.copysign(a[0](), a[1]()),
            ["heaviside"] = (a, c) => np.heaviside(a[0](), a[1]()),
            ["ldexp"] = (a, c) => np.ldexp(a[0](), a[1]()),
            ["gcd"] = (a, c) => np.gcd(a[0](), a[1]()),
            ["lcm"] = (a, c) => np.lcm(a[0](), a[1]()),
            ["bitwise_and"] = (a, c) => np.bitwise_and(a[0](), a[1]()),
            ["bitwise_xor"] = (a, c) => np.bitwise_xor(a[0](), a[1]()),
            ["left_shift"] = (a, c) => np.left_shift(a[0](), a[1]()),
            ["right_shift"] = (a, c) => np.right_shift(a[0](), a[1]()),
            ["less"] = (a, c) => np.less(a[0](), a[1]()),
            ["equal"] = (a, c) => np.equal(a[0](), a[1]()),
            ["greater_equal"] = (a, c) => np.greater_equal(a[0](), a[1]()),
            ["logical_and"] = (a, c) => np.logical_and(a[0](), a[1]()),
            ["logical_xor"] = (a, c) => np.logical_xor(a[0](), a[1]()),
            ["isclose"] = (a, c) => np.isclose(a[0](), a[1]()),
            ["allclose"] = (a, c) => np.allclose(a[0](), a[1]()),
            ["array_equal"] = (a, c) => np.array_equal(a[0](), a[1]()),
            ["add_scalar"] = (a, c) => np.add(a[0](), 3.0),
            ["add_out"] = (a, c) => { var o = a[2](); np.add(a[0](), a[1](), o); return o; },
            ["add_where"] = (a, c) => { var o = a[2](); np.add(a[0](), a[1](), o, a[3]()); return o; },
            ["divmod"] = (a, c) => np.divmod(a[0](), a[1]()),
            ["op_add"] = (a, c) => a[0]() + a[1](),
            ["op_mul_scalar"] = (a, c) => a[0]() * 2.0,
            ["op_gt"] = (a, c) => a[0]() > a[1](),
            ["op_eq"] = (a, c) => a[0]() == a[1](),
            ["op_neg"] = (a, c) => -a[0](),
            ["op_not"] = (a, c) => !a[0](),
            ["op_and"] = (a, c) => a[0]() & a[1](),
            ["clip"] = (a, c) => np.clip(a[0](), a[1](), a[2]()),
            ["clip_scalars"] = (a, c) => np.clip(a[0](), 10.0, 500.0),
            ["where3"] = (a, c) => np.where(a[0](), a[1](), a[2]()),
            ["where3_scalar"] = (a, c) => np.where(a[0](), 7.0, a[1]()),
            ["where1"] = (a, c) => np.where(a[0]()),
            ["evaluate_fma"] = (a, c) => np.evaluate(NDExpr.Arr(a[0]()) * NDExpr.Arr(a[1]()) + NDExpr.Arr(a[2]())),
            ["evaluate_sum"] = (a, c) => np.evaluate(NDExpr.Sum(NDExpr.Arr(a[0]()) * NDExpr.Arr(a[1]()))),
            // ---- reductions / scans
            ["sum"] = (a, c) => c.IntParam("axis") is int ax ? np.sum(a[0](), ax) : np.sum(a[0]()),
            ["sum_dtype"] = (a, c) => np.sum(a[0](), dtype: np.int64),
            ["prod"] = (a, c) => np.prod(a[0]()),
            ["mean"] = (a, c) => c.IntParam("axis") is int ax ? np.mean(a[0](), ax) : np.mean(a[0]()),
            ["std"] = (a, c) => np.std(a[0]()),
            ["var"] = (a, c) => c.IntParam("axis") is int ax ? np.var(a[0](), ax) : np.var(a[0]()),
            ["max"] = (a, c) => c.IntParam("axis") is int ax ? np.max(a[0](), ax) : np.max(a[0]()),
            ["min"] = (a, c) => c.IntParam("axis") is int ax ? np.min(a[0](), ax) : np.min(a[0]()),
            ["max_initial_where"] = (a, c) => np.max(a[0](), null, null, false, -1.0, a[1]()),
            ["argmax"] = (a, c) => c.IntParam("axis") is int ax ? np.argmax(a[0](), ax) : (object)np.argmax(a[0]()),
            ["argmin"] = (a, c) => c.IntParam("axis") is int ax ? np.argmin(a[0](), ax) : (object)np.argmin(a[0]()),
            ["all"] = (a, c) => c.IntParam("axis") is int ax ? np.all(a[0](), ax) : (object)np.all(a[0]()),
            ["any"] = (a, c) => c.IntParam("axis") is int ax ? np.any(a[0](), ax) : (object)np.any(a[0]()),
            ["count_nonzero"] = (a, c) => c.IntParam("axis") is int ax ? np.count_nonzero(a[0](), ax) : (object)np.count_nonzero(a[0]()),
            ["nansum"] = (a, c) => np.nansum(a[0]()),
            ["nanmax"] = (a, c) => np.nanmax(a[0]()),
            ["nanmean"] = (a, c) => np.nanmean(a[0]()),
            ["nanvar"] = (a, c) => np.nanvar(a[0]()),
            ["ptp"] = (a, c) => np.ptp(a[0]()),
            ["average_weights"] = (a, c) => np.average(a[0](), weights: a[1]()),
            ["median"] = (a, c) => np.median(a[0]()),
            ["percentile"] = (a, c) => np.percentile(a[0](), 30.0),
            ["quantile_axis"] = (a, c) => np.quantile(a[0](), 0.4, 0),
            ["nanmedian"] = (a, c) => np.nanmedian(a[0]()),
            ["cumsum"] = (a, c) => c.IntParam("axis") is int ax ? np.cumsum(a[0](), ax) : np.cumsum(a[0]()),
            ["cumprod"] = (a, c) => np.cumprod(a[0]()),
            ["nancumsum"] = (a, c) => np.nancumsum(a[0]()),
            ["diff"] = (a, c) => np.diff(a[0]()),
            ["ediff1d"] = (a, c) => np.ediff1d(a[0]()),
            ["trapezoid"] = (a, c) => np.trapezoid(a[0]()),
            ["gradient"] = (a, c) => (NDArray)np.gradient(a[0]()),   // a 1-D input: NumPy returns the single array
            // ---- selection / indexing
            ["take"] = (a, c) => np.take(a[0](), a[1](), c.IntParam("axis"), null, c.StringParam("mode", "raise")),
            ["put"] = (a, c) => { var t = a[0](); np.put(t, a[1](), a[2]()); return t; },
            ["place"] = (a, c) => { var t = a[0](); np.place(t, a[1](), a[2]()); return t; },
            ["putmask"] = (a, c) => { var t = a[0](); np.putmask(t, a[1](), a[2]()); return t; },
            ["take_along_axis"] = (a, c) => np.take_along_axis(a[0](), a[1](), 0),
            ["put_along_axis"] = (a, c) => { var t = a[0](); np.put_along_axis(t, a[1](), a[2](), 0); return t; },
            ["choose"] = (a, c) => np.choose(a[0](), new[] { a[1](), a[2]() }),
            ["compress"] = (a, c) => np.compress(a[0](), a[1]()),
            ["extract"] = (a, c) => np.extract(a[0](), a[1]()),
            ["select"] = (a, c) => np.select(new[] { a[0](), a[1]() }, new object[] { a[2](), a[3]() }),
            ["copyto"] = (a, c) => { var t = a[0](); np.copyto(t, a[1]()); return t; },
            ["copyto_where"] = (a, c) => { var t = a[0](); np.copyto(t, a[1](), where: a[2]()); return t; },
            ["fill_diagonal"] = (a, c) => { var t = a[0](); np.fill_diagonal(t, a[1]()); return t; },
            ["diag"] = (a, c) => np.diag(a[0]()),
            ["diagonal_copy"] = (a, c) => np.diagonal(a[0]()).copy(),
            ["trace"] = (a, c) => np.trace(a[0]()),
            ["tril"] = (a, c) => np.tril(a[0]()),
            ["triu"] = (a, c) => np.triu(a[0]()),
            ["nonzero"] = (a, c) => np.nonzero(a[0]()),
            ["flatnonzero"] = (a, c) => np.flatnonzero(a[0]()),
            ["argwhere"] = (a, c) => np.argwhere(a[0]()),
            ["unravel_index"] = (a, c) => np.unravel_index(a[0](), c.IntArrayParam("dims")),
            ["ravel_multi_index"] = (a, c) => np.ravel_multi_index(new[] { a[0](), a[1]() }, new[] { 7, 7 }),
            ["fancy_get"] = (a, c) => a[0]()[a[1]()],
            ["fancy_get2"] = (a, c) => a[0]()[a[1](), a[2]()],
            ["fancy_get_cols"] = (a, c) => a[0]()[Slice.All, a[1]()],
            ["fancy_set"] = (a, c) => { var t = a[0](); t[a[1]()] = a[2](); return t; },
            ["bool_get"] = (a, c) => a[0]()[a[1]()],
            ["bool_set"] = (a, c) => { var t = a[0](); t[a[1]()] = a[2](); return t; },
            ["slice_set"] = (a, c) => { var t = a[0](); t["::2"] = a[1](); return t; },
            // ---- sort / search / sets / counting
            ["sort"] = (a, c) => np.sort(a[0]()),
            ["argsort"] = (a, c) => np.argsort(a[0]()),   // NumSharp's argsort is stable, as NumPy's kind="stable"
            ["partition"] = (a, c) => np.partition(a[0](), c.IntParam("kth").Value),
            ["argpartition"] = (a, c) => np.argpartition(a[0](), c.IntParam("kth").Value),
            ["lexsort"] = (a, c) => np.lexsort(new[] { a[0](), a[1]() }),
            ["searchsorted"] = (a, c) => np.searchsorted(a[0](), a[1]()),
            ["unique"] = (a, c) => (NDArray)np.unique(a[0]()),
            ["unique_counts"] = (a, c) => (NDArray[])np.unique_counts(a[0]()),
            ["isin"] = (a, c) => np.isin(a[0](), a[1]()),
            ["intersect1d"] = (a, c) => np.intersect1d(a[0](), a[1]()),
            ["union1d"] = (a, c) => np.union1d(a[0](), a[1]()),
            ["setdiff1d"] = (a, c) => np.setdiff1d(a[0](), a[1]()),
            ["setxor1d"] = (a, c) => np.setxor1d(a[0](), a[1]()),
            ["digitize"] = (a, c) => np.digitize(a[0](), a[1]()),
            ["bincount"] = (a, c) => np.bincount(a[0]()),
            ["bincount_weights"] = (a, c) => np.bincount(a[0](), a[1]()),
            ["histogram"] = (a, c) => np.histogram(a[0](), 10).hist,
            ["sort_complex"] = (a, c) => np.sort_complex(a[0]()),
            ["nanargmax"] = (a, c) => (object)np.nanargmax(a[0]()),
            // ---- linear algebra / products
            ["dot"] = (a, c) => np.dot(a[0](), a[1]()),
            ["matmul"] = (a, c) => np.matmul(a[0](), a[1]()),
            ["inner"] = (a, c) => np.inner(a[0](), a[1]()),
            ["vdot"] = (a, c) => np.vdot(a[0](), a[1]()),
            ["vecdot"] = (a, c) => np.vecdot(a[0](), a[1]()),
            ["matvec"] = (a, c) => np.matvec(a[0](), a[1]()),
            ["vecmat"] = (a, c) => np.vecmat(a[0](), a[1]()),
            ["outer"] = (a, c) => np.outer(a[0](), a[1]()),
            ["tensordot"] = (a, c) => np.tensordot(a[0](), a[1](), 1),
            ["einsum_mv"] = (a, c) => np.einsum("ij,j->i", a[0](), a[1]()),
            ["kron"] = (a, c) => np.kron(a[0](), a[1]()),
            ["cross"] = (a, c) => np.cross(a[0](), a[1]()),
            ["det"] = (a, c) => np.linalg.det(a[0]()),
            ["solve"] = (a, c) => np.linalg.solve(a[0](), a[1]()),
            ["inv"] = (a, c) => np.linalg.inv(a[0]()),
            ["norm"] = (a, c) => np.linalg.norm(a[0]()),
            ["matrix_power"] = (a, c) => np.linalg.matrix_power(a[0](), 3),
            ["cov"] = (a, c) => np.cov(a[0]()),
            ["corrcoef"] = (a, c) => np.corrcoef(a[0]()),
            // ---- sliding / fft
            ["convolve"] = (a, c) => np.convolve(a[0](), a[1]()),
            ["correlate"] = (a, c) => np.correlate(a[0](), a[1]()),
            ["fft"] = (a, c) => np.fft.fft(a[0]()),
            ["rfft"] = (a, c) => np.fft.rfft(a[0]()),
            ["ifft"] = (a, c) => np.fft.ifft(a[0]()),
            ["fft2"] = (a, c) => np.fft.fft2(a[0]()),
            ["fftshift"] = (a, c) => np.fft.fftshift(a[0]()),
            // ---- manipulation / copies / casts
            ["concatenate"] = (a, c) => np.concatenate(new[] { a[0](), a[1]() }, c.IntParam("axis") ?? 0),
            ["stack"] = (a, c) => np.stack(new[] { a[0](), a[1]() }),
            ["vstack"] = (a, c) => np.vstack(new[] { a[0](), a[1]() }),
            ["hstack"] = (a, c) => np.hstack(new[] { a[0](), a[1]() }),
            ["column_stack"] = (a, c) => np.column_stack(new[] { a[0](), a[1]() }),
            ["repeat"] = (a, c) => np.repeat(a[0](), 2),
            ["repeat_array"] = (a, c) => np.repeat(a[0](), a[1]()),
            ["tile"] = (a, c) => np.tile(a[0](), new[] { 2 }),
            ["roll"] = (a, c) => np.roll(a[0](), 3),
            ["pad"] = (a, c) => np.pad(a[0](), 2),
            ["insert"] = (a, c) => np.insert(a[0](), 5, a[1]()),
            ["delete"] = (a, c) => np.delete(a[0](), a[1]()),
            ["append"] = (a, c) => np.append(a[0](), a[1]()),
            ["resize"] = (a, c) => np.resize(a[0](), new Shape(c.IntParam("n").Value)),
            ["trim_zeros"] = (a, c) => np.trim_zeros(a[0]()).copy(),
            ["flatten"] = (a, c) => a[0]().flatten(),
            ["ravel"] = (a, c) => np.ravel(a[0]()).copy(),
            ["copy"] = (a, c) => np.copy(a[0]()),
            ["copy_f"] = (a, c) => a[0]().copy('F'),
            ["ascontiguousarray"] = (a, c) => np.ascontiguousarray(a[0]()),
            ["asfortranarray"] = (a, c) => np.asfortranarray(a[0]()),
            ["astype"] = (a, c) => a[0]().astype(np.dtype(c.StringParam("dtype"))),
            ["broadcast_to_copy"] = (a, c) => np.broadcast_to(a[0](), new Shape(c.IntParam("side").Value, c.IntParam("side").Value)).copy(),
            ["rot90_copy"] = (a, c) => np.rot90(a[0]()).copy(),
            ["flip_copy"] = (a, c) => np.flip(a[0]()).copy(),
            ["unique_axis"] = (a, c) => (NDArray)np.unique(a[0](), axis: 0),
            ["interp"] = (a, c) => np.interp(a[0](), a[1](), a[2]()),
            ["meshgrid"] = (a, c) => (NDArray[])np.meshgrid(a[0](), a[1]()),
            ["array_split_piece"] = (a, c) => np.array_split(a[0](), 3)[1].copy(),
            // ---- numpy.polynomial / legacy polynomial
            ["polyval"] = (a, c) => np.polynomial.polynomial.polyval(a[0](), a[1]()),
            ["chebval"] = (a, c) => np.polynomial.chebyshev.chebval(a[0](), a[1]()),
            ["polyder"] = (a, c) => np.polynomial.polynomial.polyder(a[0]()),
            ["polyint"] = (a, c) => np.polynomial.polynomial.polyint(a[0]()),
            ["polyadd"] = (a, c) => np.polynomial.polynomial.polyadd(a[0](), a[1]()),
            ["polymul"] = (a, c) => np.polynomial.polynomial.polymul(a[0](), a[1]()),
            ["polyvander"] = (a, c) => np.polynomial.polynomial.polyvander(a[0](), 3),
            ["mapdomain"] = (a, c) => np.polynomial.polyutils.mapdomain(a[0](), new[] { 0.0, 1024.0 }, new[] { -1.0, 1.0 }),
            ["legacy_polyval"] = (a, c) => np.polyval(a[0](), a[1]()),
            // ---- numpy.ma
            ["ma_sum"] = (a, c) => np.ma.array(a[0](), mask: a[1]()).sum().filled(0.0),
            ["ma_add"] = (a, c) => (np.ma.array(a[0](), mask: a[1]()) + np.ma.array(a[2](), mask: a[3]())).filled(0.0),
            ["ma_masked_where"] = (a, c) => np.ma.masked_where(a[0](), a[1]()).filled(-1.0),
            // ---- conversion / IO (the .NET results are compared as bytes)
            ["tobytes"] = (a, c) => a[0]().tobytes(),
            ["to_array"] = (a, c) => MemoryMarshal.AsBytes<double>(a[0]().ToArray<double>()).ToArray(),
            ["to_multidim"] = (a, c) => MultiDimBytes(a[0]().ToMuliDimArray<double>()),
            ["to_jagged"] = (a, c) => JaggedBytes((double[][])a[0]().ToJaggedArray<double>()),
            ["save_npy"] = (a, c) => SaveBytes(a[0]()),
            ["view_int64"] = (a, c) => a[0]().view(np.int64).copy(),
            ["nditer_sum"] = (a, c) => NditerSum(a[0]()),
            ["flat_sum"] = (a, c) => FlatSum(a[0]()),
        };

        /// <summary>
        ///     The op keys the registry implements; <c>LifetimeOracleTests.EveryCorpusOpHasABody</c> holds the corpus to
        ///     them in-process, so a key the generator adds without a body fails fast instead of inside the child.
        /// </summary>
        public static IReadOnlyCollection<string> Keys => Table.Keys;

        /// <summary>
        ///     Runs one case's call.
        /// </summary>
        /// <param name="c">The case (its op key and parameters).</param>
        /// <param name="makers">One maker per operand: a fresh temporary or the held instance.</param>
        /// <returns>The call's result, as the NumSharp API returns it.</returns>
        /// <exception cref="NotSupportedException">The registry has no body for the case's op key.</exception>
        /// <remarks>Any exception the call itself raises propagates; the child records it as the repetition's outcome.</remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static object Invoke(LifetimeCase c, Func<NDArray>[] makers)
        {
            if (!Table.TryGetValue(c.Op, out var body))
                throw new NotSupportedException($"lifetime op '{c.Op}' has no body in LifetimeOps");
            return body(makers, c);
        }

        /// <summary>The bytes of a <c>double[,]</c> in row-major order (NumPy's C order).</summary>
        /// <param name="md">The rectangular array <c>ToMuliDimArray</c> returned.</param>
        /// <returns>Its elements' bytes, row after row (a CLR rectangular array is row-major).</returns>
        private static byte[] MultiDimBytes(Array md)
        {
            var bytes = new byte[Buffer.ByteLength(md)];
            Buffer.BlockCopy(md, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        /// <summary>The rows of a jagged <c>double[][]</c> concatenated (NumPy's C order).</summary>
        /// <param name="rows">The jagged array <c>ToJaggedArray</c> returned for a 2-D operand.</param>
        /// <returns>The rows' bytes, concatenated.</returns>
        private static byte[] JaggedBytes(double[][] rows)
        {
            var ms = new MemoryStream();
            foreach (var row in rows)
                ms.Write(MemoryMarshal.AsBytes<double>(row));
            return ms.ToArray();
        }

        /// <summary>The bytes <c>np.save</c> writes for an array (NumSharp's writer is byte-identical to NumPy's).</summary>
        /// <param name="a">The array to save.</param>
        /// <returns>The whole <c>.npy</c> file.</returns>
        private static byte[] SaveBytes(NDArray a)
        {
            var ms = new MemoryStream();
            np.save(ms, a);
            return ms.ToArray();
        }

        /// <summary>A sequential sum through the typed by-ref iterator (memory order; a C-contiguous operand is linear).</summary>
        /// <param name="a">A float64 operand.</param>
        /// <returns>The sum, in visiting order (exact for the corpus's small-integer values).</returns>
        private static double NditerSum(NDArray a)
        {
            double s = 0;
            foreach (var x in np.nditer<double>(a))
                s += x;
            return s;
        }

        /// <summary>A sequential sum through the typed flat (C-order) iterator.</summary>
        /// <param name="a">A float64 operand.</param>
        /// <returns>The sum, in C order (exact for the corpus's small-integer values).</returns>
        private static double FlatSum(NDArray a)
        {
            double s = 0;
            foreach (var x in np.flat<double>(a))
                s += x;
            return s;
        }
    }
}
