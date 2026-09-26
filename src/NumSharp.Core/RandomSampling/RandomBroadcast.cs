using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's broadcast machinery for array-valued distribution parameters (<c>numpy/random/_common.pyx</c>: the
    ///     non-scalar branches of <c>cont</c> / <c>disc</c>, <c>cont_broadcast_N</c>, <c>discrete_broadcast_*</c> and
    ///     <c>validate_output_shape</c>) — the output shape and its checks, NumPy's ufunc-step error texts, and the scans
    ///     that decide whether a draw may read ahead.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The flow every array-parameter sampler follows, in NumPy's order: convert each parameter
    ///     (<see cref="RandomParam"/>, NumPy's <c>PyArray_FROM_OTF</c> with its <c>'safe'</c> casting gate); when every
    ///     converted parameter is 0-d take the scalar path; otherwise check each parameter's constraint over the whole
    ///     array (<see cref="RandomConstraints.CheckArray"/>), resolve the output shape (<see cref="OutputDims"/>),
    ///     allocate, and draw one value per output position in C order (<see cref="BroadcastWalk"/>) with that position's
    ///     parameters.
    ///     </para>
    ///     <para>
    ///     Shape errors carry NumPy's texts and <see cref="ValueError"/> type: NumPy's <c>MultiIterNew</c> mismatch
    ///     (<c>shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (2,)
    ///     and arg 1 with shape (3,).</c>), where arg 0 is the output when a size was given, and
    ///     <c>Output size (1,) is not compatible with broadcast dimensions of inputs (3,).</c> when a size the parameters
    ///     broadcast into is not the broadcast result itself.
    ///     </para>
    /// </remarks>
    internal static unsafe class RandomBroadcast
    {
        /// <summary>
        ///     The shortest run of equal means (<see cref="RunEnd(long, long, double*, long)"/>) that a broadcast
        ///     <c>poisson</c> builds a <see cref="PoissonSetup"/> for; a shorter run keeps NumPy's per-call statements in
        ///     locals (<see cref="Distributions.RandomPoisson(ref DrawBufferDouble, double)"/>).
        /// </summary>
        /// <remarks>
        ///     Both routes evaluate the same expressions, so the choice only moves time: building the 72-byte setup costs
        ///     more than the <c>exp</c> it holds (see <see cref="PoissonSetup"/>'s constructor), which a run of a few values
        ///     amortizes and a new mean at every position — NumPy's own per-call arithmetic — never does.
        /// </remarks>
        internal const long PoissonSetupMinRun = 4;

        /// <summary>
        ///     The shortest run of equal means for which a broadcast <c>poisson</c> also hoists PTRS's per-rejection
        ///     <c>log(invalpha)</c> into its <see cref="PoissonSetup"/> — the scalar fill's threshold for its memos.
        /// </summary>
        /// <remarks>
        ///     Bit-neutral (the same <c>log</c> of the same value); only about one PTRS trial in ten reaches the test that
        ///     needs it, so the hoist pays for itself only over a longer run.
        /// </remarks>
        internal const long PoissonRejectionLogMinRun = 16;

        /// <summary>
        ///     The output dimensions of a broadcast draw: <paramref name="size"/> when given (NumPy's <c>np.empty(size)</c>),
        ///     else the parameters' broadcast shape — then NumPy's <c>validate_output_shape</c>: the output and the parameters
        ///     must broadcast to exactly the output's shape.
        /// </summary>
        /// <param name="size">The requested size; ignored unless <paramref name="sizeGiven"/>.</param>
        /// <param name="sizeGiven">Whether the caller passed a size (NumPy's <c>size is not None</c>).</param>
        /// <param name="a">The first parameter (converted), in NumPy's <c>MultiIterNew</c> order.</param>
        /// <param name="b">The second parameter, or null.</param>
        /// <param name="c">The third parameter, or null.</param>
        /// <returns>The output dimensions.</returns>
        /// <exception cref="ValueError">The parameters (and the size) do not broadcast — NumPy's <c>shape mismatch</c>
        ///     text, where arg 0 is the output when a size was given — or a size that is not the broadcast shape itself
        ///     (<c>Output size (1,) is not compatible with broadcast dimensions of inputs (3,).</c>).</exception>
        /// <remarks>
        ///     The parameter ORDER is part of the contract: NumPy numbers the operands in its error text by their position
        ///     in <c>MultiIterNew</c>, which is not always the signature order (the binomial passes <c>p</c> before
        ///     <c>n</c>), so callers pass them exactly as NumPy's sampler does.
        /// </remarks>
        internal static long[] OutputDims(Shape size, bool sizeGiven, NDArray a, NDArray b = null, NDArray c = null)
        {
            long[] outDims = sizeGiven ? (long[])(size.dimensions ?? Array.Empty<long>()).Clone() : Broadcast(a, b, c);
            // NumPy's MultiIterNew(randoms, a, b, c): the output takes part as arg 0.
            var withOut = new System.Collections.Generic.List<Shape> { new Shape(outDims) };
            withOut.Add(a.Shape);
            if (b is not null)
                withOut.Add(b.Shape);
            if (c is not null)
                withOut.Add(c.Shape);
            Shape it;
            try
            {
                it = np.broadcast_shapes(withOut.ToArray());
            }
            catch (IncorrectShapeException e)
            {
                throw new ValueError(e.Message);
            }
            if (!SameDims(it.dimensions ?? Array.Empty<long>(), outDims))
                throw new ValueError($"Output size {PyTuple(outDims)} is not compatible with broadcast dimensions of inputs {PyTuple(it.dimensions ?? Array.Empty<long>())}.");
            return outDims;
        }

        /// <summary>The parameters' broadcast shape (NumPy's <c>MultiIterNew</c> over them alone), with NumPy's mismatch text.</summary>
        /// <param name="a">The first parameter.</param>
        /// <param name="b">The second parameter, or null.</param>
        /// <param name="c">The third parameter, or null.</param>
        /// <returns>The broadcast dimensions.</returns>
        /// <exception cref="ValueError">The parameters do not broadcast.</exception>
        internal static long[] Broadcast(NDArray a, NDArray b = null, NDArray c = null)
        {
            try
            {
                Shape s = b is null ? a.Shape
                    : c is null ? np.broadcast_shapes(a.Shape, b.Shape)
                    : np.broadcast_shapes(a.Shape, b.Shape, c.Shape);
                return (long[])(s.dimensions ?? Array.Empty<long>()).Clone();
            }
            catch (IncorrectShapeException e)
            {
                throw new ValueError(e.Message);
            }
        }

        /// <summary>
        ///     Allocates a broadcast draw's output — NumPy's <c>np.empty(shape, dtype)</c>: a fresh, uninitialized,
        ///     C-contiguous array every element of which the draw loop writes.
        /// </summary>
        /// <param name="typeCode">The output dtype.</param>
        /// <param name="dims">The output dimensions (from <see cref="OutputDims"/>).</param>
        /// <returns>The output.</returns>
        internal static NDArray NewOutput(NPTypeCode typeCode, long[] dims) => new NDArray(typeCode, new Shape(dims), false);

        /// <summary>
        ///     Runs a NumPy ufunc-style step on the parameters (NumPy's <c>np.subtract(high, low)</c>, <c>np.greater</c>, …),
        ///     turning NumSharp's broadcast failure into NumPy's <see cref="ValueError"/> with the same text
        ///     (<c>operands could not be broadcast together with shapes (3,) (2,) </c>).
        /// </summary>
        /// <param name="op">The step.</param>
        /// <returns>The step's result.</returns>
        /// <exception cref="ValueError">The step's operands do not broadcast.</exception>
        internal static NDArray Ufunc(Func<NDArray> op)
        {
            try
            {
                return op();
            }
            catch (IncorrectShapeException e)
            {
                throw new ValueError(e.Message);
            }
        }

        /// <summary>
        ///     NumPy's <c>np.any(compare(a, b))</c> for a pairwise parameter check (<c>np.greater(oleft, omode)</c>,
        ///     <c>np.equal(oleft, oright)</c>, …) as ONE fused pass that never materializes the broadcast boolean array.
        /// </summary>
        /// <param name="compare">The comparison, as an expression factory (<see cref="NDExpr.Greater"/>, <see cref="NDExpr.Equal"/>, …).</param>
        /// <param name="a">The first operand (converted parameter).</param>
        /// <param name="b">The second operand (converted parameter).</param>
        /// <returns>True when the comparison holds anywhere in the broadcast (false for an empty broadcast).</returns>
        /// <exception cref="ValueError">The operands do not broadcast — the ufunc's own text
        ///     (<c>operands could not be broadcast together with shapes (3,) (2,) </c>), which the fused pass reports
        ///     identically for two operands.</exception>
        /// <remarks>
        ///     A column against a row broadcasts to the whole output: materializing <c>np.greater</c>'s <c>(k, k)</c> bool
        ///     array and then scanning it measured 15x slower than the fused <c>np.evaluate</c>
        ///     (7.6 vs 0.5 ms at <c>k = 1000</c>), which alone put a broadcast <c>triangular</c> below NumPy.
        /// </remarks>
        internal static bool AnyCompare(Func<NDExpr, NDExpr, NDExpr> compare, NDArray a, NDArray b)
        {
            using var any = Ufunc(() => np.evaluate(NDExpr.Any(compare(NDExpr.Arr(a), NDExpr.Arr(b)))));
            return any.GetAtIndex<bool>(0);
        }

        /// <summary>Whether any element of a bool array is true — NumPy's <c>np.any</c>, disposing the mask.</summary>
        /// <param name="mask">The mask (consumed).</param>
        /// <returns>True when any element is true.</returns>
        internal static bool AnyAndDispose(NDArray mask)
        {
            using (mask)
                return np.any(mask);
        }

        /// <summary>
        ///     Whether some element <c>v</c> of a converted parameter lies in <c>[lo, hi]</c> (a NaN never does) — the scan
        ///     behind the array constraint checks and the read-ahead decisions (see the remarks).
        /// </summary>
        /// <param name="p">A converted parameter (<see cref="RandomParam.Array"/>): C-contiguous float64, int64 or float32.</param>
        /// <param name="lo">The inclusive lower bound (<see cref="double.NegativeInfinity"/> for none).</param>
        /// <param name="hi">The inclusive upper bound (<see cref="double.PositiveInfinity"/> for none).</param>
        /// <returns>True when some element satisfies <c>lo &lt;= v &lt;= hi</c>.</returns>
        /// <exception cref="ArgumentException"><paramref name="p"/> is not float64, int64 or float32 (a caller bug).</exception>
        /// <remarks>
        ///     <para>
        ///     Every rule the samplers scan for is a closed range once an EXCLUSIVE bound is moved to its neighbouring double
        ///     (<c>v &gt; 0</c> is <c>v &gt;= double.Epsilon</c>, <c>v &lt; 1</c> is <c>v &lt;= BitDecrement(1)</c>): exact for every
        ///     double, and a NaN fails both comparisons exactly as it fails NumPy's <c>np.less_equal</c>/<c>np.greater</c>. An
        ///     int64 or float32 element widens to double first — exact for float32, and for int64 exact against the small
        ///     bounds compared here (the ordering against them survives the rounding of values past <c>2^53</c>).
        ///     </para>
        ///     <para>
        ///     Why a read-ahead may trust a scan of the PARAMETERS: a read-ahead may only run ahead of NumPy's stream by
        ///     draws NumPy will certainly make, which a fill guarantees by keeping <c>Owed</c> at a lower bound of the draws
        ///     still to come — "one per value still owed" when every value draws. Every element of a broadcast parameter is
        ///     read by at least one output position (a non-empty output reaches every index of every operand), so "no
        ///     element draws nothing" over the parameters is exactly "every output position draws".
        ///     </para>
        ///     <para>
        ///     A plain loop with the bounds in registers, vectorized over float64 — NOT a per-element delegate, which
        ///     measured 2.8 ns an element (280 us per rule for a 100K-element parameter): more than a cheap sampler's whole
        ///     draw, where NumPy's ufunc checks cost about a tenth of that.
        ///     </para>
        /// </remarks>
        internal static bool AnyInRange(NDArray p, double lo, double hi)
        {
            long n = p.size;
            byte* first = (byte*)p.Storage.Address + p.Shape.offset * p.dtypesize;
            switch (p.typecode)
            {
                case NPTypeCode.Double:
                {
                    var d = (double*)first;
                    long i = 0;
                    if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                    {
                        var vlo = new Vector<double>(lo);
                        var vhi = new Vector<double>(hi);
                        for (; i <= n - Vector<double>.Count; i += Vector<double>.Count)
                        {
                            var v = Unsafe.ReadUnaligned<Vector<double>>(d + i);
                            // A lane is in range when both compares hold (a NaN lane fails both).
                            if (!Vector.EqualsAll(Vector.GreaterThanOrEqual(v, vlo) & Vector.LessThanOrEqual(v, vhi), Vector<long>.Zero))
                                return true;
                        }
                    }
                    for (; i < n; i++)
                        if (d[i] >= lo && d[i] <= hi)
                            return true;
                    return false;
                }
                case NPTypeCode.Int64:
                {
                    var d = (long*)first;
                    for (long i = 0; i < n; i++)
                    {
                        double v = d[i];
                        if (v >= lo && v <= hi)
                            return true;
                    }
                    return false;
                }
                case NPTypeCode.Single:
                {
                    var d = (float*)first;
                    for (long i = 0; i < n; i++)
                    {
                        double v = d[i];
                        if (v >= lo && v <= hi)
                            return true;
                    }
                    return false;
                }
                default:
                    throw new ArgumentException($"A converted random parameter is float64, int64 or float32, not {p.typecode}.", nameof(p));
            }
        }

        /// <summary>
        ///     Whether EVERY element <c>v</c> of a converted parameter lies in <c>[lo, hi]</c> — NumPy's <c>np.all</c> over a
        ///     range condition, so a NaN element fails it (an empty parameter satisfies it).
        /// </summary>
        /// <param name="p">A converted parameter: C-contiguous float64, int64 or float32.</param>
        /// <param name="lo">The inclusive lower bound (see <see cref="AnyInRange"/> for exclusive bounds).</param>
        /// <param name="hi">The inclusive upper bound.</param>
        /// <returns>True when no element falls outside the range or is NaN.</returns>
        /// <exception cref="ArgumentException"><paramref name="p"/> is not float64, int64 or float32 (a caller bug).</exception>
        internal static bool AllInRange(NDArray p, double lo, double hi)
        {
            long n = p.size;
            byte* first = (byte*)p.Storage.Address + p.Shape.offset * p.dtypesize;
            switch (p.typecode)
            {
                case NPTypeCode.Double:
                {
                    var d = (double*)first;
                    long i = 0;
                    if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                    {
                        var vlo = new Vector<double>(lo);
                        var vhi = new Vector<double>(hi);
                        for (; i <= n - Vector<double>.Count; i += Vector<double>.Count)
                        {
                            var v = Unsafe.ReadUnaligned<Vector<double>>(d + i);
                            // Every lane must pass both compares; a NaN lane passes neither.
                            if (!(Vector.GreaterThanOrEqualAll(v, vlo) && Vector.LessThanOrEqualAll(v, vhi)))
                                return false;
                        }
                    }
                    for (; i < n; i++)
                        if (!(d[i] >= lo && d[i] <= hi))
                            return false;
                    return true;
                }
                case NPTypeCode.Int64:
                {
                    var d = (long*)first;
                    for (long i = 0; i < n; i++)
                    {
                        double v = d[i];
                        if (!(v >= lo && v <= hi))
                            return false;
                    }
                    return true;
                }
                case NPTypeCode.Single:
                {
                    var d = (float*)first;
                    for (long i = 0; i < n; i++)
                    {
                        double v = d[i];
                        if (!(v >= lo && v <= hi))
                            return false;
                    }
                    return true;
                }
                default:
                    throw new ArgumentException($"A converted random parameter is float64, int64 or float32, not {p.typecode}.", nameof(p));
            }
        }

        /// <summary>Whether some element of a converted parameter is exactly zero, of either sign — a draw-free input's usual mark.</summary>
        /// <param name="p">A converted parameter: C-contiguous float64, int64 or float32.</param>
        /// <returns>True when some element equals 0 (<c>-0.0</c> included, as <c>v == 0</c> reads it).</returns>
        /// <exception cref="ArgumentException"><paramref name="p"/> is not float64, int64 or float32 (a caller bug).</exception>
        internal static bool AnyZero(NDArray p) => AnyInRange(p, 0.0, 0.0);

        /// <summary>Whether some element of a converted parameter is NaN (never, for int64).</summary>
        /// <param name="p">A converted parameter: C-contiguous float64, int64 or float32.</param>
        /// <returns>True when some element is NaN.</returns>
        /// <exception cref="ArgumentException"><paramref name="p"/> is not float64, int64 or float32 (a caller bug).</exception>
        internal static bool AnyNaN(NDArray p)
        {
            long n = p.size;
            byte* first = (byte*)p.Storage.Address + p.Shape.offset * p.dtypesize;
            switch (p.typecode)
            {
                case NPTypeCode.Double:
                {
                    var d = (double*)first;
                    long i = 0;
                    if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                        for (; i <= n - Vector<double>.Count; i += Vector<double>.Count)
                        {
                            var v = Unsafe.ReadUnaligned<Vector<double>>(d + i);
                            // v == v fails exactly on the NaN lanes.
                            if (!Vector.EqualsAll(v, v))
                                return true;
                        }
                    for (; i < n; i++)
                        if (double.IsNaN(d[i]))
                            return true;
                    return false;
                }
                case NPTypeCode.Int64:
                    return false;
                case NPTypeCode.Single:
                {
                    var d = (float*)first;
                    for (long i = 0; i < n; i++)
                        if (float.IsNaN(d[i]))
                            return true;
                    return false;
                }
                default:
                    throw new ArgumentException($"A converted random parameter is float64, int64 or float32, not {p.typecode}.", nameof(p));
            }
        }

        /// <summary>
        ///     Whether some NON-NaN element of a converted parameter has its sign bit set — NumPy's
        ///     <c>np.any(np.logical_and(np.logical_not(np.isnan(val)), np.signbit(val)))</c>, the <c>CONS_NON_NEGATIVE</c>
        ///     array test, which rejects <c>-0.0</c> along with every negative value.
        /// </summary>
        /// <param name="p">A converted parameter: C-contiguous float64, int64 or float32.</param>
        /// <returns>True when some element is negative or <c>-0.0</c>; a NaN of either sign never counts.</returns>
        /// <exception cref="ArgumentException"><paramref name="p"/> is not float64, int64 or float32 (a caller bug).</exception>
        /// <remarks>
        ///     <c>v &lt; 0</c> alone would miss <c>-0.0</c>, which NumPy's <c>signbit</c> catches; an int64 element's sign bit
        ///     IS <c>v &lt; 0</c>. Over float64 the bits are tested directly: the sign bit set and the magnitude no larger than
        ///     <c>+inf</c>'s (a larger magnitude is a NaN).
        /// </remarks>
        internal static bool AnyNegativeSign(NDArray p)
        {
            long n = p.size;
            byte* first = (byte*)p.Storage.Address + p.Shape.offset * p.dtypesize;
            switch (p.typecode)
            {
                case NPTypeCode.Double:
                {
                    var d = (long*)first; // the float64 bit patterns
                    const long magnitude = 0x7FFFFFFFFFFFFFFF, infBits = 0x7FF0000000000000;
                    for (long i = 0; i < n; i++)
                    {
                        long b = d[i];
                        if (b < 0 && (b & magnitude) <= infBits)
                            return true;
                    }
                    return false;
                }
                case NPTypeCode.Int64:
                {
                    var d = (long*)first;
                    for (long i = 0; i < n; i++)
                        if (d[i] < 0)
                            return true;
                    return false;
                }
                case NPTypeCode.Single:
                {
                    var d = (float*)first;
                    for (long i = 0; i < n; i++)
                        if (!float.IsNaN(d[i]) && float.IsNegative(d[i]))
                            return true;
                    return false;
                }
                default:
                    throw new ArgumentException($"A converted random parameter is float64, int64 or float32, not {p.typecode}.", nameof(p));
            }
        }

        /// <summary>
        ///     Whether some element of a converted float64 parameter HALVES to zero (<c>v / 2.0 == 0</c>) — the degrees of
        ///     freedom whose chi-square gamma (shape <c>df / 2</c>) returns 0 without drawing.
        /// </summary>
        /// <param name="p">A converted parameter: C-contiguous float64.</param>
        /// <returns>True when some element halves to zero.</returns>
        /// <exception cref="ArgumentException"><paramref name="p"/> is not float64, int64 or float32 (a caller bug).</exception>
        /// <remarks>
        ///     A validated positive degrees of freedom can still halve to zero: the smallest subnormal <c>5e-324</c>
        ///     (<c>2^-1074</c>, <see cref="double.Epsilon"/>) halves to a tie that rounds to even, i.e. to 0, while
        ///     <c>1e-323 / 2</c> is exactly <c>5e-324</c> — so the halving zeros are exactly <c>[-5e-324, 5e-324]</c>. "df &gt; 0"
        ///     does NOT imply "the chi-square draws", and a read-ahead that assumed it would run ahead of NumPy's stream.
        /// </remarks>
        internal static bool AnyHalvesToZero(NDArray p) => AnyInRange(p, -double.Epsilon, double.Epsilon);

        /// <summary>
        ///     The end of the parameter RUN that starts at position <paramref name="j"/> of a walk chunk: the first later
        ///     position whose parameter differs from position <paramref name="j"/>'s (bitwise), else <paramref name="count"/>
        ///     — so a sampler builds its per-call setup ONCE per run and draws the whole run over it.
        /// </summary>
        /// <param name="j">The run's first position in the chunk (<c>0 &lt;= j &lt; count</c>).</param>
        /// <param name="count">The chunk length (<see cref="BroadcastWalk.Count"/>).</param>
        /// <param name="a">The chunk's first element of the parameter (<see cref="BroadcastWalk.A"/>).</param>
        /// <param name="sa">The parameter's element stride in the chunk (0 along a broadcast axis).</param>
        /// <returns>The run's exclusive end, in <c>(j, count]</c>.</returns>
        /// <remarks>
        ///     <para>
        ///     Bit-neutral by construction: a setup (NumPy's per-call statements — <c>1/sqrt(9b)</c>, PTRS's constants,
        ///     HRUA's <c>loggam</c>s, …) is a pure function of its parameters evaluated with NumPy's expressions, so the one
        ///     built for a run's first position is the one every position of the run would rebuild. Parameters compare
        ///     BITWISE: <c>-0.0</c> and <c>+0.0</c> end a run (a setup may depend on the sign), and a NaN continues one only
        ///     as the same NaN pattern (whose setup is the same computation). It pays where broadcasting repeats a tuple
        ///     along the walk — a column parameter against a row, a parameter stretched over a larger <c>size</c> — and a
        ///     parameter broadcast along the chunk's axis (stride 0) is constant over the whole chunk, answered at once.
        ///     </para>
        ///     <para>
        ///     Why a scan that yields the run, rather than a per-position "same as the previous position?" memory: a setup
        ///     carried across the iterations of one loop and replaced only when the parameters change is loop-carried state
        ///     the JIT keeps in memory and copies on every rebuild — measured 3x slower for legacy <c>noncentral_f</c>
        ///     (a 22-double setup) on distinct per-position parameters. A setup declared inside the run's body is a fresh
        ///     local the run's inner loop reads, and the distinct-parameter case costs one compare per position.
        ///     </para>
        ///     <para>
        ///     Deliberately NOT <c>AggressiveInlining</c>: measured, inlining the scans into the samplers slowed the
        ///     Generator's broadcast <c>beta</c> 1.3x and legacy HRUA <c>hypergeometric</c> 1.3x — the samplers' own kernels
        ///     are what the JIT must inline into those loops, and one call per run is cheap.
        ///     </para>
        /// </remarks>
        internal static long RunEnd(long j, long count, double* a, long sa)
        {
            if (sa == 0)
                return count;
            long ba = BitConverter.DoubleToInt64Bits(a[j * sa]);
            long e = j + 1;
            while (e < count && BitConverter.DoubleToInt64Bits(a[e * sa]) == ba)
                e++;
            return e;
        }

        /// <summary>
        ///     The end of the run of positions repeating position <paramref name="j"/>'s two parameters, bitwise — the
        ///     two-parameter form of <see cref="RunEnd(long, long, double*, long)"/> (see its remarks).
        /// </summary>
        /// <param name="j">The run's first position in the chunk.</param>
        /// <param name="count">The chunk length.</param>
        /// <param name="a">The chunk's first element of the first parameter.</param>
        /// <param name="sa">The first parameter's element stride in the chunk.</param>
        /// <param name="b">The chunk's first element of the second parameter.</param>
        /// <param name="sb">The second parameter's element stride in the chunk.</param>
        /// <returns>The run's exclusive end, in <c>(j, count]</c>.</returns>
        internal static long RunEnd(long j, long count, double* a, long sa, double* b, long sb)
        {
            // Both constant along the chunk (broadcast): one run. A single stride-0 parameter only ever compares an
            // element with itself below, which the other parameter's scan decides.
            if (sa == 0 && sb == 0)
                return count;
            long ba = BitConverter.DoubleToInt64Bits(a[j * sa]), bb = BitConverter.DoubleToInt64Bits(b[j * sb]);
            long e = j + 1;
            while (e < count && BitConverter.DoubleToInt64Bits(a[e * sa]) == ba && BitConverter.DoubleToInt64Bits(b[e * sb]) == bb)
                e++;
            return e;
        }

        /// <summary>
        ///     The end of the run of positions repeating position <paramref name="j"/>'s three parameters, bitwise — the
        ///     three-parameter form of <see cref="RunEnd(long, long, double*, long)"/> (see its remarks).
        /// </summary>
        /// <param name="j">The run's first position in the chunk.</param>
        /// <param name="count">The chunk length.</param>
        /// <param name="a">The chunk's first element of the first parameter.</param>
        /// <param name="sa">The first parameter's element stride in the chunk.</param>
        /// <param name="b">The chunk's first element of the second parameter.</param>
        /// <param name="sb">The second parameter's element stride in the chunk.</param>
        /// <param name="c">The chunk's first element of the third parameter.</param>
        /// <param name="sc">The third parameter's element stride in the chunk.</param>
        /// <returns>The run's exclusive end, in <c>(j, count]</c>.</returns>
        internal static long RunEnd(long j, long count, double* a, long sa, double* b, long sb, double* c, long sc)
        {
            if (sa == 0 && sb == 0 && sc == 0)
                return count;
            long ba = BitConverter.DoubleToInt64Bits(a[j * sa]), bb = BitConverter.DoubleToInt64Bits(b[j * sb]),
                bc = BitConverter.DoubleToInt64Bits(c[j * sc]);
            long e = j + 1;
            while (e < count && BitConverter.DoubleToInt64Bits(a[e * sa]) == ba && BitConverter.DoubleToInt64Bits(b[e * sb]) == bb
                   && BitConverter.DoubleToInt64Bits(c[e * sc]) == bc)
                e++;
            return e;
        }

        /// <summary>
        ///     The end of the run of positions repeating position <paramref name="j"/>'s three INTEGER parameters — the
        ///     hypergeometric's <c>(ngood, nbad, nsample)</c> form of <see cref="RunEnd(long, long, double*, long)"/>
        ///     (see its remarks; integers need no bitwise care).
        /// </summary>
        /// <param name="j">The run's first position in the chunk.</param>
        /// <param name="count">The chunk length.</param>
        /// <param name="a">The chunk's first element of the first parameter.</param>
        /// <param name="sa">The first parameter's element stride in the chunk.</param>
        /// <param name="b">The chunk's first element of the second parameter.</param>
        /// <param name="sb">The second parameter's element stride in the chunk.</param>
        /// <param name="c">The chunk's first element of the third parameter.</param>
        /// <param name="sc">The third parameter's element stride in the chunk.</param>
        /// <returns>The run's exclusive end, in <c>(j, count]</c>.</returns>
        internal static long RunEnd(long j, long count, long* a, long sa, long* b, long sb, long* c, long sc)
        {
            if (sa == 0 && sb == 0 && sc == 0)
                return count;
            long va = a[j * sa], vb = b[j * sb], vc = c[j * sc];
            long e = j + 1;
            while (e < count && a[e * sa] == va && b[e * sb] == vb && c[e * sc] == vc)
                e++;
            return e;
        }

        /// <summary>Whether two dimension vectors are equal.</summary>
        /// <param name="x">The first.</param>
        /// <param name="y">The second.</param>
        /// <returns>True when equal in rank and extents.</returns>
        private static bool SameDims(long[] x, long[] y)
        {
            if (x.Length != y.Length)
                return false;
            for (int i = 0; i < x.Length; i++)
                if (x[i] != y[i])
                    return false;
            return true;
        }

        /// <summary>A Python tuple repr of dimensions — <c>()</c>, <c>(3,)</c>, <c>(4, 3)</c> — NumPy's error spelling.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>The repr.</returns>
        internal static string PyTuple(long[] dims)
        {
            var sb = new StringBuilder("(");
            for (int i = 0; i < dims.Length; i++)
            {
                if (i > 0)
                    sb.Append(", ");
                sb.Append(dims[i]);
            }
            if (dims.Length == 1)
                sb.Append(',');
            return sb.Append(')').ToString();
        }
    }

    /// <summary>
    ///     One distribution parameter after NumPy's conversion (<c>PyArray_FROM_OTF(x, NPY_DOUBLE / NPY_INT64,
    ///     NPY_ARRAY_ALIGNED)</c>): the caller's array itself when it already is a C-contiguous array of the target dtype,
    ///     else a private converted copy this value owns and disposes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A C# <c>null</c> stands for one of two Python arguments, decided per parameter by whether NumPy gives it a
    ///     default: for a parameter WITH a default (<c>loc=0.0</c>, <c>scale=1.0</c>, <c>lam=1.0</c>, …) null is "not
    ///     passed" — the default (<see cref="Float64(NDArray, double)"/>), since C# cannot spell an array default; for a
    ///     parameter WITHOUT one it is Python's <c>None</c> (<see cref="Float64(NDArray)"/> / <see cref="Int64"/>), which
    ///     NumPy converts to a float64 NaN (<c>np.asarray(None, float64)</c>) and rejects as an int64
    ///     (<c>int() argument must be … not 'NoneType'</c>). A NaN from <c>None</c> behaves like any NaN on the array path
    ///     but fails the scalar path's <c>PyFloat_AsDouble(None)</c> (<see cref="Scalar"/>), exactly as in NumPy.
    ///     </para>
    ///     <para>
    ///     The value is a <c>readonly struct</c> so a <c>using var</c> declaration (read-only by C# rules) never hides a
    ///     mutation behind a defensive copy: nothing here mutates after construction.
    ///     </para>
    /// </remarks>
    internal readonly struct RandomParam : IDisposable
    {
        /// <summary>The converted parameter: C-contiguous, float64 or int64 (float32 for <see cref="Float32Forced"/>).</summary>
        internal readonly NDArray Array;

        /// <summary>True when <see cref="Array"/> is a private copy this value disposes.</summary>
        private readonly bool _owned;

        /// <summary>True when the caller passed Python's <c>None</c> (a C# null for a parameter without a NumPy default).</summary>
        internal readonly bool IsNone;

        /// <summary>
        ///     True when the ORIGINAL 0-d value is complex — NumPy's scalar paths call <c>PyFloat_AsDouble</c> on the
        ///     argument itself, which rejects a complex even where a forced cast already made the array real.
        /// </summary>
        private readonly bool _complexScalar;

        /// <summary>Wraps a converted array.</summary>
        /// <param name="array">The converted array.</param>
        /// <param name="owned">Whether this value owns (and disposes) it.</param>
        /// <param name="isNone">Whether it stands for Python's <c>None</c>.</param>
        /// <param name="complexScalar">Whether the original was a 0-d complex.</param>
        private RandomParam(NDArray array, bool owned, bool isNone, bool complexScalar)
        {
            Array = array;
            _owned = owned;
            IsNone = isNone;
            _complexScalar = complexScalar;
        }

        /// <summary>Whether the converted parameter is 0-d (every 0-d parameter together selects NumPy's scalar path).</summary>
        internal bool IsScalar => Array.ndim == 0;

        /// <summary>
        ///     A float64 parameter WITHOUT a NumPy default: <paramref name="x"/> converted under the <c>'safe'</c> rule, or —
        ///     for null, Python's <c>None</c> — a 0-d NaN (NumPy's <c>np.asarray(None, np.float64)</c>).
        /// </summary>
        /// <param name="x">The argument as passed.</param>
        /// <returns>The converted parameter.</returns>
        /// <exception cref="TypeError">The dtype does not cast to float64 under <c>'safe'</c> (complex) — NumPy's
        ///     <c>Cannot cast {array data|scalar} from dtype('complex128') to dtype('float64') according to the rule 'safe'</c>.</exception>
        internal static RandomParam Float64(NDArray x)
        {
            // NumPy's None converts to Python's float('nan') — the POSITIVE quiet NaN 0x7ff8000000000000. .NET's
            // double.NaN is the negative one, and a NaN parameter's bits propagate into the draws (gamma(NaN) -> NaN
            // carries them), so the constant matters.
            if (x is null)
                return new RandomParam(NDArray.Scalar(Distributions.NPY_NAN), owned: true, isNone: true, complexScalar: false);
            RandomConstraints.CheckSafeCast(x, NPTypeCode.Double);
            return Convert(x, NPTypeCode.Double);
        }

        /// <summary>
        ///     A float64 parameter WITH a NumPy default: <paramref name="x"/> converted under the <c>'safe'</c> rule, or —
        ///     for null ("not passed") — the default as a 0-d array.
        /// </summary>
        /// <param name="x">The argument as passed, or null.</param>
        /// <param name="defaultValue">NumPy's default for the parameter.</param>
        /// <returns>The converted parameter.</returns>
        /// <exception cref="TypeError">The dtype does not cast to float64 under <c>'safe'</c> (complex).</exception>
        internal static RandomParam Float64(NDArray x, double defaultValue)
        {
            if (x is null)
                return new RandomParam(NDArray.Scalar(defaultValue), owned: true, isNone: false, complexScalar: false);
            RandomConstraints.CheckSafeCast(x, NPTypeCode.Double);
            return Convert(x, NPTypeCode.Double);
        }

        /// <summary>
        ///     An int64 parameter (always without a NumPy default): <paramref name="x"/> converted under the <c>'safe'</c>
        ///     rule, which rejects floats, uint64 and complex.
        /// </summary>
        /// <param name="x">The argument as passed.</param>
        /// <returns>The converted parameter.</returns>
        /// <exception cref="TypeError"><paramref name="x"/> is null (Python's <c>None</c>: NumPy's
        ///     <c>int() argument must be a string, a bytes-like object or a real number, not 'NoneType'</c>), or its dtype
        ///     does not cast to int64 under <c>'safe'</c>.</exception>
        internal static RandomParam Int64(NDArray x)
        {
            if (x is null)
                throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            RandomConstraints.CheckSafeCast(x, NPTypeCode.Int64);
            return Convert(x, NPTypeCode.Int64);
        }

        /// <summary>
        ///     A float32 parameter under NumPy's <c>PyArray_FROMANY(x, NPY_FLOAT32, 0, 0, NPY_ARRAY_ALIGNED |
        ///     NPY_ARRAY_FORCECAST)</c> — the float32 <c>standard_gamma</c>: EVERY dtype is cast (float64 rounds, complex
        ///     keeps its real part), none is refused.
        /// </summary>
        /// <param name="x">The argument as passed.</param>
        /// <returns>The converted parameter.</returns>
        /// <remarks>
        ///     Null (Python's <c>None</c>) becomes a 0-d NaN like the float64 conversion. A 0-d complex converts too, but
        ///     remembers it was complex: the scalar path reads the ORIGINAL through <c>PyFloat_AsDouble</c>, which refuses it
        ///     (<see cref="Scalar"/>).
        /// </remarks>
        internal static RandomParam Float32Forced(NDArray x)
        {
            if (x is null)
                // np.asarray(None, np.float32): the positive quiet NaN 0x7fc00000 (.NET's float.NaN is 0xffc00000).
                return new RandomParam(NDArray.Scalar(BitConverter.Int32BitsToSingle(0x7FC00000)), owned: true, isNone: true, complexScalar: false);
            bool complexScalar = x.ndim == 0 && x.typecode == NPTypeCode.Complex;
            var converted = Convert(x, NPTypeCode.Single);
            return new RandomParam(converted.Array, converted._owned, isNone: false, complexScalar);
        }

        /// <summary>
        ///     The conversion itself: the argument when it already is a C-contiguous array of <paramref name="to"/>, else an
        ///     owned copy in that dtype (C order, the logical element order NumPy's multi-iterator reads).
        /// </summary>
        /// <param name="x">The argument (not null; the safe-cast gate already ran where one applies).</param>
        /// <param name="to">The target dtype.</param>
        /// <returns>The converted parameter.</returns>
        private static RandomParam Convert(NDArray x, NPTypeCode to)
        {
            if (x.typecode == to && x.Shape.IsContiguous)
                return new RandomParam(x, owned: false, isNone: false, complexScalar: false);
            return new RandomParam(x.astype(to), owned: true, isNone: false, complexScalar: false);
        }

        /// <summary>
        ///     The scalar path's value of a 0-d parameter — NumPy's <c>PyFloat_AsDouble(arg)</c> on the ORIGINAL argument.
        /// </summary>
        /// <returns>The value as a double (an int64 or float32 parameter widened exactly).</returns>
        /// <exception cref="TypeError">The argument was Python's <c>None</c> (<c>must be real number, not NoneType</c>)
        ///     or a 0-d complex (<c>float() argument must be a string or a real number, not 'complex'</c>).</exception>
        internal double Scalar()
        {
            if (IsNone)
                throw new TypeError("must be real number, not NoneType");
            if (_complexScalar)
                throw new TypeError("float() argument must be a string or a real number, not 'complex'");
            return Array.typecode switch
            {
                NPTypeCode.Int64 => Array.GetAtIndex<long>(0),
                NPTypeCode.Single => Array.GetAtIndex<float>(0),
                _ => Array.GetAtIndex<double>(0),
            };
        }

        /// <summary>
        ///     The scalar path's value of a 0-d parameter together with its scalar constraint — NumPy's <c>cont</c>/<c>disc</c>
        ///     convert and check each parameter in turn, so a later parameter's <c>None</c> is reported only after an
        ///     earlier parameter's constraint passed.
        /// </summary>
        /// <param name="name">The parameter name in NumPy's message.</param>
        /// <param name="cons">The scalar constraint.</param>
        /// <returns>The value.</returns>
        /// <exception cref="TypeError">See <see cref="Scalar"/>.</exception>
        /// <exception cref="ValueError">The value violates <paramref name="cons"/> (NumPy's scalar text).</exception>
        internal double ScalarChecked(string name, ConstraintType cons)
        {
            double v = Scalar();
            RandomConstraints.Check(v, name, cons);
            return v;
        }

        /// <summary>The scalar path's value of a 0-d int64 parameter — NumPy's <c>&lt;int64_t&gt;arg</c>.</summary>
        /// <returns>The value.</returns>
        internal long ScalarInt64() => Array.GetAtIndex<long>(0);

        /// <summary>Disposes the converted copy when this value owns one; the caller's own array is never touched.</summary>
        public void Dispose()
        {
            if (_owned)
                Array.Dispose();
        }
    }

    /// <summary>
    ///     A C-order walk over a broadcast draw's output that reads up to three parameters through their broadcast strides —
    ///     the walk NumPy's <c>PyArray_MultiIter_NEXT</c> performs, driven by <see cref="NDIterRef"/> with an external
    ///     inner loop.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Each <see cref="Next"/> exposes one inner-loop chunk: <see cref="Count"/> positions whose output slot and
    ///     parameter elements sit at <c>pointer + j * stride</c> (element strides; a broadcast axis has stride 0). The
    ///     iterator runs in C order over the OUTPUT's shape (the output is operand 0, so its shape is the iteration shape
    ///     and the parameters broadcast into it), so position <c>i</c> of the walk is output element <c>i</c> — the order
    ///     NumPy's loop draws in. Coalescing may merge axes into longer chunks; it never reorders.
    ///     </para>
    ///     <para>
    ///     A class, not a ref struct: the chunk fields change on every step, and a <c>using var</c> struct would be
    ///     read-only (each call mutating a defensive copy). The iterator's heap state is detached into this owner and
    ///     re-borrowed per step (the <c>np.nditer</c> pattern), and <see cref="Dispose"/> frees it — always use it in a
    ///     <c>using</c>. An empty output builds no iterator (NumPy's zero-size operands would need <c>ZEROSIZE_OK</c>)
    ///     and yields no chunk.
    ///     </para>
    /// </remarks>
    internal sealed unsafe class BroadcastWalk : IDisposable
    {
        /// <summary>The detached iterator state (null for an empty output, and after disposal).</summary>
        private NDIterState* _state;

        /// <summary>The operands the iterator was built over (kept alive for its lifetime).</summary>
        [NDBorrowed] // the caller's output and converted parameters: the sampler owns and disposes them, never the walk
        private readonly NDArray[] _operands;

        /// <summary>The iterator's resolved advance function, cached across borrows.</summary>
        private NDIterNextFunc _next;

        /// <summary>Whether the first chunk has been handed out (the iterator starts ON it, so only later steps advance).</summary>
        private bool _started;

        /// <summary>Whether a second / third parameter takes part.</summary>
        private readonly bool _hasB, _hasC;

        /// <summary>The number of positions in the current chunk.</summary>
        internal long Count;

        /// <summary>The current chunk's first output slot.</summary>
        internal byte* Out;

        /// <summary>The current chunk's first element of the first / second / third parameter.</summary>
        internal byte* A, B, C;

        /// <summary>The output's element stride inside the chunk.</summary>
        internal long OutStride;

        /// <summary>The parameters' element strides inside the chunk (0 along a broadcast axis).</summary>
        internal long StrideA, StrideB, StrideC;

        /// <summary>Builds the walk over <paramref name="output"/>'s shape.</summary>
        /// <param name="output">The fresh output (operand 0; its shape is the iteration shape).</param>
        /// <param name="a">The first parameter (broadcasts into the output).</param>
        /// <param name="b">The second parameter, or null.</param>
        /// <param name="c">The third parameter, or null (only with <paramref name="b"/>).</param>
        /// <exception cref="ArgumentException"><paramref name="c"/> is given without <paramref name="b"/> (a caller bug).</exception>
        internal BroadcastWalk(NDArray output, NDArray a, NDArray b = null, NDArray c = null)
        {
            if (c is not null && b is null)
                throw new ArgumentException("A third parameter requires a second one.", nameof(c));
            _hasB = b is not null;
            _hasC = c is not null;
            if (output.size == 0)
                return;

            int nop = 2 + (_hasB ? 1 : 0) + (_hasC ? 1 : 0);
            var ops = new NDArray[nop];
            var flags = new NDIterPerOpFlags[nop];
            ops[0] = output;
            flags[0] = NDIterPerOpFlags.WRITEONLY;
            ops[1] = a;
            flags[1] = NDIterPerOpFlags.READONLY;
            if (_hasB)
            {
                ops[2] = b;
                flags[2] = NDIterPerOpFlags.READONLY;
            }
            if (_hasC)
            {
                ops[3] = c;
                flags[3] = NDIterPerOpFlags.READONLY;
            }

            // Every operand already has its loop dtype (the parameters were converted), so nothing casts or buffers:
            // the chunk pointers are absolute addresses into the operands.
            var it = NDIterRef.MultiNew(nop, ops, NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_NO_CASTING, flags);
            _state = it.Detach(out _operands, out _);
        }

        /// <summary>Moves to the next inner-loop chunk and loads its pointers, strides and length.</summary>
        /// <returns>False when the walk is complete (at once for an empty output).</returns>
        internal bool Next()
        {
            if (_state == null)
                return false;
            var it = NDIterRef.Borrow(_state, _operands, _next);
            if (_started && !it.Iternext())
                return false;
            _started = true;
            _next ??= it.PeekCachedIterNext();

            void** ptrs = it.GetDataPtrArray();
            Out = (byte*)ptrs[0];
            A = (byte*)ptrs[1];
            B = _hasB ? (byte*)ptrs[2] : null;
            C = _hasC ? (byte*)ptrs[3] : null;
            if (it.NDim == 0)
            {
                // A 0-d iteration has no inner axis to read (GetInnerLoopSizePtr would index Shape[-1]): one position.
                Count = 1;
                OutStride = StrideA = StrideB = StrideC = 0;
                return true;
            }
            Count = *it.GetInnerLoopSizePtr();
            OutStride = it.GetInnerLoopElementStride(0);
            StrideA = it.GetInnerLoopElementStride(1);
            StrideB = _hasB ? it.GetInnerLoopElementStride(2) : 0;
            StrideC = _hasC ? it.GetInnerLoopElementStride(3) : 0;
            return true;
        }

        /// <summary>Frees the iterator state (idempotent).</summary>
        public void Dispose()
        {
            if (_state == null)
                return;
            NDIterRef.FreeState(_state);
            _state = null;
        }
    }
}
