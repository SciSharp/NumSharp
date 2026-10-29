using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

// =============================================================================
// NDTextNumericReduction.cs — a NUMERIC add / multiply loop whose input or out is text
// =============================================================================
//
// NumPy's loop resolution (NDReduceLikeResolution) can pick a numeric loop X for a reduction whose operands hold text:
// np.trace(np.array([['3', '1'], ['1', '2']]), dtype='f8', out=f8_out) parses the text into float64,
// np.matrix(i8).prod(axis=1, dtype='i8', out=U6) writes the product's str() into the text out. NumPy then simply runs
// the numeric loop, and its iterator casts around it:
//
//   reduce (PyUFunc_ReduceWrapper): the loop's identity is written into the out through the X → out cast (a text out
//   gets '0.0', '0', 'False'), then a BUFFERED iterator reads the out into an X buffer at every fill's start (a parse:
//   'False' reads back as True, a truncated '12345.' as 12345.0) and writes it back at the fill's end — so the running
//   value round-trips through the out's text at every fill — while the input is parsed into X buffers.
//
//   accumulate (PyUFunc_Accumulate): the iterator cannot buffer, so the out's cells are COPIED into an X temporary first
//   (UPDATEIFCOPY: an out that does not parse raises before anything is computed — np.zeros(3, 'U6')'s '' does not parse
//   into int64), the input is copied into X, the accumulation runs in X and the temporary is cast back into the out.
//
// Both are reproduced here: the reduction runs on NumPyReduceIterator (NumPy's fill schedule, ported) with conversions
// that parse and format text, and NumPy's own inner loops for add / multiply — a float add call is the running value
// plus the call's pairwise sum, a float multiply call sequential, float16 computed in float32 and narrowed per call.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     A numeric <c>add</c> / <c>multiply</c> reduction or accumulation whose input or out is text (<c>S</c>, <c>U</c>,
    ///     <c>StringDType</c>): NumPy's buffered casts around its numeric loop (see the file header).
    /// </summary>
    internal static unsafe class NDTextNumericReduction
    {
        /// <summary>
        ///     <c>ufunc.reduce(a, axis, dtype=X, out=out, keepdims=keepdims)</c> after the loop resolution chose the numeric
        ///     loop <paramref name="loop"/>: <c>PyUFunc_ReduceWrapper</c>'s shape checks, the identity written into the out,
        ///     then the buffered walk (the text conversions at every fill).
        /// </summary>
        /// <param name="ufunc"><c>"add"</c> or <c>"multiply"</c>.</param>
        /// <param name="a">The reduced input (text or numbers).</param>
        /// <param name="axisFlags">The reduced axes (<see cref="ReductionArguments.AxisFlags(int, int?)"/>).</param>
        /// <param name="keepdims">Whether the out keeps the reduced axes.</param>
        /// <param name="out">The out (text or numbers; writeable — the caller checked).</param>
        /// <param name="loop">The loop's descriptor (a numeric dtype).</param>
        /// <returns><paramref name="out"/> itself.</returns>
        /// <exception cref="ValueError">An out of the wrong rank or shape (the reduction iterator's texts), or a text that does
        ///     not parse into the loop dtype (NumPy's cast texts, at the fill that meets it).</exception>
        /// <exception cref="IncorrectShapeException">The out does not broadcast against the result.</exception>
        /// <exception cref="OverflowException">An integer text outside the loop's integer type.</exception>
        [NDScoped] // the overlap copy and the identity scalar are this call's own
        internal static NDArray Reduce(string ufunc, NDArray a, bool[] axisFlags, bool keepdims, NDArray @out, DType loop)
        {
            // add / multiply are reorderable: several axes are allowed. The out's rank and the iterator's broadcast checks.
            ReductionArguments.ValidateOperands(a, axisFlags, keepdims, @out, null, ufunc);
            // COPY_IF_OVERLAP: an out that may share memory with the input reduces a copy of the input.
            NDArray input = NDMemOverlap.SolveMayShareMemory(@out, a, 0) != MemOverlap.No ? a.copy('K') : a;

            // raw_array_assign_scalar: the loop's identity (0 for add, 1 for multiply, in the loop dtype) cast into every out
            // element — for a text out, the identity's str() — before a single element is read.
            NDArray identity = ufunc == "add" ? new NDArray(loop, Shape.Scalar, true) : NDArray.Ones(Shape.Scalar, loop);
            UnmanagedStorage.Copyto(@out, identity, NPY_CASTING.NPY_UNSAFE_CASTING);
            if (input.size == 0 || @out.size == 0)
                return @out;

            NPTypeCode loopT = loop.GetTypeCode();
            bool outCast = !IsLoopDtype(@out, loopT);
            bool inCast = !IsLoopDtype(input, loopT);
            var it = NumPyReduceIterator.Build(input.Shape, axisFlags, keepdims, @out.Shape, null, outCast, inCast, np.getbufsize());
            Run(it, ufunc, @out, input, loopT, new Transfers(input, @out, loop));
            return @out;
        }

        /// <summary>
        ///     <c>ufunc.accumulate(a, axis, dtype=X, out=out)</c> after the loop resolution chose the numeric loop
        ///     <paramref name="loop"/> (the out or the input is text): <c>PyUFunc_Accumulate</c>'s outer-iterator path — its
        ///     shape checks, the out's cells copied into the loop dtype (a parse), the input copied into it, the accumulation in
        ///     the loop dtype, and the cast back into the out.
        /// </summary>
        /// <param name="ufunc"><c>"add"</c> or <c>"multiply"</c>.</param>
        /// <param name="a">The accumulated input.</param>
        /// <param name="axis">The axis, or null (the input is raveled in C order and accumulated along axis 0).</param>
        /// <param name="out">The out (writeable — the caller checked).</param>
        /// <param name="loop">The loop's descriptor (a numeric dtype).</param>
        /// <returns><paramref name="out"/> itself.</returns>
        /// <exception cref="IncorrectShapeException">The iterator's shape errors (an out with too few dimensions, a shape
        ///     that does not broadcast, an out that would have to broadcast).</exception>
        /// <exception cref="ValueError">The out's text, then the input's, does not parse into the loop dtype (NumPy copies the
        ///     out first: its error wins).</exception>
        /// <exception cref="OverflowException">An integer text outside the loop's integer type.</exception>
        [NDScoped] // the raveled input, the views, the loop-dtype copies and the accumulation are this call's own
        internal static NDArray Accumulate(string ufunc, NDArray a, int? axis, NDArray @out, DType loop)
        {
            // The np.* calls below compute the accumulation itself: a chararray or numpy.matrix operand (or out) must not meet
            // the class rules those functions apply to a USER's call (the boundaries' audits require library code to compute
            // plain).
            using var plainRegion = NDCharArrayBoundary.Inside();
            NDArray input = a;
            int ax;
            if (axis is not { } given || a.ndim == 0)
            {
                // np.cumsum / np.cumprod ravel in C order (a copy for a non-contiguous input) and accumulate along axis 0.
                input = (a.Shape.IsContiguous ? a : np.ascontiguousarray(a)).reshape(a.size);
                ax = 0;
            }
            else
            {
                ax = given < 0 ? given + a.ndim : given;
            }

            // A cast always takes PyUFunc_Accumulate's outer iterator: its op_axes / broadcast / no-broadcast checks first.
            if (NDVStringReduction.AccumulateIteratorShapeError(@out, input) is { } shapeError)
                throw shapeError;
            long count = input.Shape.dimensions[ax];
            if (count == 0 || @out.size == 0)
                return @out;

            // The cells the iterator walks: the out's leading input-rank axes (a longer out's trailing axes stay at index 0)
            // and, along the axis, as many as the input has.
            NDArray outView = NDVStringReduction.LeadingAxesView(@out, input.ndim);
            NDArray target = outView.Shape.dimensions[ax] == count ? outView : NDVStringReduction.AxisPrefixView(outView, ax, count);
            NDArray src = input;
            if (!SameDims(target, input))
                src = NDArray.BroadcastTo(input, new Shape((long[])target.Shape.dimensions.Clone()));

            // UPDATEIFCOPY: the out's cells are converted into the loop dtype BEFORE the input (NpyIter allocates its
            // operands in order) — '' does not parse, so NumPy's zeros-initialized text outs refuse here. The converted
            // values themselves are overwritten (the accumulation copies each lane's first element): only the errors count.
            if (!IsLoopDtype(target, loop.GetTypeCode()))
                target.astype(loop, true, 'K', "unsafe");
            // COPY: the input in the loop dtype (a text input parses here).
            NDArray inLoop = src.astype(loop, true, 'K', "unsafe");
            // The accumulation in the loop dtype (NumPy's elementwise loop along the axis), cast back into the out.
            NDArray acc = loop.GetTypeCode() == NPTypeCode.Boolean
                ? LogicalAccumulate(ufunc, inLoop, ax)
                : ufunc == "add"
                    ? inLoop.TensorEngine.ReduceCumAdd(inLoop, ax, loop)
                    : inLoop.TensorEngine.ReduceCumMul(inLoop, ax, loop);
            UnmanagedStorage.Copyto(target, acc, NPY_CASTING.NPY_UNSAFE_CASTING);
            return @out;
        }

        /// <summary>
        ///     NumPy's bool <c>add</c> / <c>multiply</c> loops as an accumulation: <c>BOOL_add</c> is <c>logical_or</c> and
        ///     <c>BOOL_multiply</c> is <c>logical_and</c>, so element <c>k</c> of the result is whether ANY (add) or ALL
        ///     (multiply) of the elements up to <c>k</c> are true. The engine's scans have no bool loop (a bool scan widens to
        ///     int64), so this counts instead: a running int64 sum of the truth values is nonzero exactly when one was true, and
        ///     a running int64 product of them is one exactly when all were (both exact: a sum of ones never passes the element
        ///     count, a product of zeros and ones stays zero or one).
        /// </summary>
        /// <param name="ufunc"><c>"add"</c> or <c>"multiply"</c>.</param>
        /// <param name="x">The bool input (the loop dtype).</param>
        /// <param name="axis">The normalized axis.</param>
        /// <returns>The bool accumulation, shaped like <paramref name="x"/>.</returns>
        [NDScopedCovered] // the int64 scan is its caller's scope's temporary
        private static NDArray LogicalAccumulate(string ufunc, NDArray x, int axis)
        {
            NDArray counts = ufunc == "add"
                ? x.TensorEngine.ReduceCumAdd(x, axis, np.int64)
                : x.TensorEngine.ReduceCumMul(x, axis, np.int64);
            return counts.TensorEngine.NotEqual(counts, NDArray.Scalar(0L));
        }

        // ---------------------------------------------------------------------------------------------
        // The walk
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Runs the iterator with NumPy's add / multiply loop for the loop dtype.
        /// </summary>
        /// <param name="it">The iterator.</param>
        /// <param name="ufunc"><c>"add"</c> or <c>"multiply"</c>.</param>
        /// <param name="out">The out.</param>
        /// <param name="input">The input.</param>
        /// <param name="loopT">The loop dtype.</param>
        /// <param name="t">The operands' conversions.</param>
        /// <exception cref="NotSupportedException">A loop dtype without a NumPy add / multiply loop (cannot happen: the
        ///     resolver only picks the numeric lanes).</exception>
        private static void Run(NumPyReduceIterator it, string ufunc, NDArray @out, NDArray input, NPTypeCode loopT, Transfers t)
        {
            byte* o = (byte*)@out.Address + @out.Shape.offset * @out.Storage.ItemBytes;
            byte* x = (byte*)input.Address + input.Shape.offset * input.Storage.ItemBytes;
            int oSz = @out.Storage.ItemBytes, xSz = input.Storage.ItemBytes;
            bool add = ufunc == "add";
            switch (loopT)
            {
                case NPTypeCode.Double:
                    it.Run(o, oSz, t.OutToLoop, t.LoopToOut, x, xSz, t.InToLoop, 0,
                        add ? new NumPyReduceIterator.Loop<double>(&AddReduceDouble, &AddBinary<double>)
                            : new NumPyReduceIterator.Loop<double>(&MulReduceFloat<double>, &MulBinary<double>));
                    break;
                case NPTypeCode.Single:
                    it.Run(o, oSz, t.OutToLoop, t.LoopToOut, x, xSz, t.InToLoop, 0,
                        add ? new NumPyReduceIterator.Loop<float>(&AddReduceSingle, &AddBinary<float>)
                            : new NumPyReduceIterator.Loop<float>(&MulReduceFloat<float>, &MulBinary<float>));
                    break;
                case NPTypeCode.Half:
                    it.Run(o, oSz, t.OutToLoop, t.LoopToOut, x, xSz, t.InToLoop, 0,
                        add ? new NumPyReduceIterator.Loop<Half>(&AddReduceHalf, &AddBinaryHalf)
                            : new NumPyReduceIterator.Loop<Half>(&MulReduceHalf, &MulBinaryHalf));
                    break;
                case NPTypeCode.Complex:
                    it.Run(o, oSz, t.OutToLoop, t.LoopToOut, x, xSz, t.InToLoop, 0,
                        add ? new NumPyReduceIterator.Loop<Complex>(&AddReduceComplex, &AddBinaryComplex)
                            : new NumPyReduceIterator.Loop<Complex>(&MulReduceComplex, &MulBinaryComplex));
                    break;
                case NPTypeCode.Boolean:
                    it.Run(o, oSz, t.OutToLoop, t.LoopToOut, x, xSz, t.InToLoop, 0,
                        add ? new NumPyReduceIterator.Loop<bool>(&OrReduce, &OrBinary)
                            : new NumPyReduceIterator.Loop<bool>(&AndReduce, &AndBinary));
                    break;
                case NPTypeCode.SByte: RunInteger<sbyte>(it, add, o, oSz, x, xSz, t); break;
                case NPTypeCode.Byte: RunInteger<byte>(it, add, o, oSz, x, xSz, t); break;
                case NPTypeCode.Int16: RunInteger<short>(it, add, o, oSz, x, xSz, t); break;
                case NPTypeCode.UInt16: RunInteger<ushort>(it, add, o, oSz, x, xSz, t); break;
                case NPTypeCode.Char: RunInteger<char>(it, add, o, oSz, x, xSz, t); break;
                case NPTypeCode.Int32: RunInteger<int>(it, add, o, oSz, x, xSz, t); break;
                case NPTypeCode.UInt32: RunInteger<uint>(it, add, o, oSz, x, xSz, t); break;
                case NPTypeCode.Int64: RunInteger<long>(it, add, o, oSz, x, xSz, t); break;
                case NPTypeCode.UInt64: RunInteger<ulong>(it, add, o, oSz, x, xSz, t); break;
                case NPTypeCode.Decimal:
                    it.Run(o, oSz, t.OutToLoop, t.LoopToOut, x, xSz, t.InToLoop, 0,
                        add ? new NumPyReduceIterator.Loop<decimal>(&AddReduceDecimal, &AddBinaryDecimal)
                            : new NumPyReduceIterator.Loop<decimal>(&MulReduceDecimal, &MulBinaryDecimal));
                    break;
                default:
                    throw new NotSupportedException($"No {ufunc} loop for {loopT}.");
            }
        }

        /// <summary>The integer lanes' loops (modular arithmetic: the call partition cannot change the result).</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="it">The iterator.</param>
        /// <param name="add">True for add, false for multiply.</param>
        /// <param name="o">The out's element 0.</param>
        /// <param name="oSz">The out's element size.</param>
        /// <param name="x">The input's element 0.</param>
        /// <param name="xSz">The input's element size.</param>
        /// <param name="t">The conversions.</param>
        private static void RunInteger<T>(NumPyReduceIterator it, bool add, byte* o, int oSz, byte* x, int xSz, Transfers t)
            where T : unmanaged, IBinaryInteger<T>
        {
            it.Run(o, oSz, t.OutToLoop, t.LoopToOut, x, xSz, t.InToLoop, 0,
                add ? new NumPyReduceIterator.Loop<T>(&AddReduceInteger<T>, &AddBinary<T>)
                    : new NumPyReduceIterator.Loop<T>(&MulReduceInteger<T>, &MulBinary<T>));
        }

        /// <summary>Whether an array's storage IS the loop dtype (a numeric lane, not text): it is read in place.</summary>
        /// <param name="a">The array.</param>
        /// <param name="loopT">The loop dtype.</param>
        /// <returns>True for the loop dtype itself.</returns>
        private static bool IsLoopDtype(NDArray a, NPTypeCode loopT) => a.typecode == loopT && !NDStringUfuncs.IsText(a);

        /// <summary>Whether two arrays have the same dimensions.</summary>
        /// <param name="x">The first array.</param>
        /// <param name="y">The second array.</param>
        /// <returns>True for equal ranks and extents.</returns>
        private static bool SameDims(NDArray x, NDArray y)
        {
            if (x.ndim != y.ndim)
                return false;
            for (int i = 0; i < x.ndim; i++)
                if (x.Shape.dimensions[i] != y.Shape.dimensions[i])
                    return false;
            return true;
        }

        // ---------------------------------------------------------------------------------------------
        // The conversions
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The buffer conversions of one reduction: the out to and from the loop dtype, the input to it — a parse /
        ///     <c>str()</c> for a fixed-width text operand, the <c>StringDType</c> cast loops (the cells' allocator locked per
        ///     run) for a <c>StringDType</c> one, the IL cast kernels for a number.
        /// </summary>
        private sealed class Transfers
        {
            /// <summary>The out's <c>StringDType</c> allocator (locked around each run), or null.</summary>
            private readonly VStringAllocator _outAllocator;

            /// <summary>The input's <c>StringDType</c> allocator (locked around each run), or null.</summary>
            private readonly VStringAllocator _inAllocator;

            /// <summary>Builds the conversions.</summary>
            /// <param name="input">The input.</param>
            /// <param name="out">The out.</param>
            /// <param name="loop">The loop's descriptor.</param>
            internal Transfers(NDArray input, NDArray @out, DType loop)
            {
                _outAllocator = NDStringUfuncs.IsVString(@out) ? @out.Storage.VStringSlice.Allocator : null;
                _inAllocator = NDStringUfuncs.IsVString(input) ? input.Storage.VStringSlice.Allocator : null;
                OutToLoop = ToLoop(@out, loop, _outAllocator);
                LoopToOut = FromLoop(loop, @out, _outAllocator);
                InToLoop = ToLoop(input, loop, _inAllocator);
            }

            /// <summary>The out → loop conversion (a fill's read of the out).</summary>
            internal NumPyReduceIterator.Transfer OutToLoop { get; }

            /// <summary>The loop → out conversion (a fill's write-back).</summary>
            internal NumPyReduceIterator.Transfer LoopToOut { get; }

            /// <summary>The input → loop conversion.</summary>
            internal NumPyReduceIterator.Transfer InToLoop { get; }

            /// <summary>The conversion of an operand into the loop dtype.</summary>
            /// <param name="a">The operand.</param>
            /// <param name="loop">The loop's descriptor.</param>
            /// <param name="allocator">The operand's <c>StringDType</c> allocator, or null.</param>
            /// <returns>The conversion.</returns>
            private static NumPyReduceIterator.Transfer ToLoop(NDArray a, DType loop, VStringAllocator allocator)
            {
                NPTypeCode loopT = loop.GetTypeCode();
                if (allocator is not null)
                {
                    var run = NDVStringCasting.TransferRun(a.dtype, allocator, loop, null);
                    return (s, ss, d, ds, n) =>
                    {
                        allocator.acquire_allocator();
                        try { run(s, ss, d, ds, n); }
                        finally { allocator.release_allocator(); }
                    };
                }
                if (NDStringUfuncs.IsString(a))
                {
                    bool unicode = a.dtype.kind == 'U';
                    int chars = unicode ? a.dtype.itemsize / 4 : a.dtype.itemsize;
                    // ParseRun's work space: the widest element's parse (one buffer per conversion; runs are sequential).
                    var scratch = new byte[NDTextParse.ScratchBytes(chars)];
                    return (s, ss, d, ds, n) =>
                        NDTextConversion.ParseRun(s, ss, unicode, chars, d, ds, loopT, n, scratch, TextReprStyle.ArrayElement);
                }
                return NumberRun(a.typecode, loopT);
            }

            /// <summary>The conversion of the loop dtype into an operand.</summary>
            /// <param name="loop">The loop's descriptor.</param>
            /// <param name="a">The operand.</param>
            /// <param name="allocator">The operand's <c>StringDType</c> allocator, or null.</param>
            /// <returns>The conversion.</returns>
            private static NumPyReduceIterator.Transfer FromLoop(DType loop, NDArray a, VStringAllocator allocator)
            {
                NPTypeCode loopT = loop.GetTypeCode();
                if (allocator is not null)
                {
                    var run = NDVStringCasting.TransferRun(loop, null, a.dtype, allocator);
                    return (s, ss, d, ds, n) =>
                    {
                        allocator.acquire_allocator();
                        try { run(s, ss, d, ds, n); }
                        finally { allocator.release_allocator(); }
                    };
                }
                if (NDStringUfuncs.IsString(a))
                {
                    bool unicode = a.dtype.kind == 'U';
                    int chars = unicode ? a.dtype.itemsize / 4 : a.dtype.itemsize;
                    return (s, ss, d, ds, n) => NDTextConversion.FormatRun(s, ss, loopT, d, ds, unicode, chars, n);
                }
                return NumberRun(loopT, a.typecode);
            }

            /// <summary>A numeric cast run (the IL inner cast kernel, the scalar converter without dynamic code).</summary>
            /// <param name="from">The source dtype.</param>
            /// <param name="to">The destination dtype.</param>
            /// <returns>The conversion.</returns>
            private static NumPyReduceIterator.Transfer NumberRun(NPTypeCode from, NPTypeCode to)
            {
                var kernel = DirectILKernelGenerator.TryGetInnerCastKernel(from, to);
                if (kernel is not null)
                    return (s, ss, d, ds, n) => kernel(s, ss, d, ds, n);
                return (s, ss, d, ds, n) =>
                {
                    for (long i = 0; i < n; i++, s += ss, d += ds)
                        NDIterCasting.ConvertValue(s, d, from, to);
                };
            }
        }

        // ---------------------------------------------------------------------------------------------
        // NumPy's add / multiply inner loops (loops.c.src, loops_arithm_fp.dispatch.c.src)
        // ---------------------------------------------------------------------------------------------

        /// <summary>The cached float64 pairwise fold (NumPy's <c>DOUBLE_pairwise_sum</c>), or null without dynamic code.</summary>
        private static readonly PairwiseFoldDouble s_pairwiseDouble = ILKernelGenerator.TryGetPairwiseFoldDouble();

        /// <summary>The cached float32 pairwise fold (NumPy's <c>FLOAT_pairwise_sum</c>), or null without dynamic code.</summary>
        private static readonly PairwiseFoldSingle s_pairwiseSingle = ILKernelGenerator.TryGetPairwiseFoldSingle();

        /// <summary>The cached complex128 pairwise fold (NumPy's <c>CDOUBLE_pairwise_sum</c>), or null without dynamic code.</summary>
        private static readonly PairwiseFoldComplex s_pairwiseComplex = ILKernelGenerator.TryGetPairwiseFoldComplex();

        /// <summary><c>DOUBLE_add</c>'s reduce form: the running value plus the call's pairwise sum.</summary>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's stride in elements.</param>
        /// <param name="n">The call's element count.</param>
        /// <returns>The new running value.</returns>
        private static double AddReduceDouble(double acc, double* x, long xs, long n)
            => acc + (s_pairwiseDouble is { } f ? f(x, n, xs) : PairwiseDoubleManaged(x, n, xs));

        /// <summary><c>FLOAT_add</c>'s reduce form (float32 throughout).</summary>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride in elements.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static float AddReduceSingle(float acc, float* x, long xs, long n)
            => acc + (s_pairwiseSingle is { } f ? f(x, n, xs) : PairwiseSingleManaged(x, n, xs));

        /// <summary><c>HALF_add</c>'s reduce form: the running value widened to float32, plus the call's float32 pairwise sum,
        ///     narrowed once at the call's end.</summary>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride in elements.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static Half AddReduceHalf(Half acc, Half* x, long xs, long n)
            => (Half)((float)acc + ILKernelGenerator.PairwiseFoldHalf((ushort*)x, n, xs));

        /// <summary><c>CDOUBLE_add</c>'s reduce form: each part plus the call's pairwise part sum.</summary>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride in elements.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static Complex AddReduceComplex(Complex acc, Complex* x, long xs, long n)
        {
            double rr, ri;
            if (s_pairwiseComplex is { } f)
            {
                var v = f(x, n, xs * 16);
                rr = v.GetElement(0);
                ri = v.GetElement(1);
            }
            else
            {
                PairwiseComplexManaged((double*)x, 2 * n, xs * 8, out rr, out ri);
            }
            return new Complex(acc.Real + rr, acc.Imaginary + ri);
        }

        /// <summary>The elementwise add of a float / integer lane (<c>o[i] = o[i] + x[i]</c>, wrapping for integers).</summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's stride in elements.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride in elements.</param>
        /// <param name="n">The element count.</param>
        private static void AddBinary<T>(T* o, long os, T* x, long xs, long n) where T : unmanaged, IAdditionOperators<T, T, T>
        {
            for (long i = 0; i < n; i++)
                o[i * os] = unchecked(o[i * os] + x[i * xs]);
        }

        /// <summary>The elementwise multiply of a float / integer lane (<c>o[i] = o[i] * x[i]</c>).</summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        private static void MulBinary<T>(T* o, long os, T* x, long xs, long n) where T : unmanaged, IMultiplyOperators<T, T, T>
        {
            for (long i = 0; i < n; i++)
                o[i * os] = unchecked(o[i * os] * x[i * xs]);
        }

        /// <summary><c>DOUBLE_multiply</c> / <c>FLOAT_multiply</c>'s reduce form: sequential products (no pairwise for multiply).</summary>
        /// <typeparam name="T">float or double.</typeparam>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static T MulReduceFloat<T>(T acc, T* x, long xs, long n) where T : unmanaged, IMultiplyOperators<T, T, T>
        {
            for (long i = 0; i < n; i++)
                acc *= x[i * xs];
            return acc;
        }

        /// <summary>The integer add reduce (modular).</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static T AddReduceInteger<T>(T acc, T* x, long xs, long n) where T : unmanaged, IBinaryInteger<T>
        {
            for (long i = 0; i < n; i++)
                acc = unchecked(acc + x[i * xs]);
            return acc;
        }

        /// <summary>The integer multiply reduce (modular).</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static T MulReduceInteger<T>(T acc, T* x, long xs, long n) where T : unmanaged, IBinaryInteger<T>
        {
            for (long i = 0; i < n; i++)
                acc = unchecked(acc * x[i * xs]);
            return acc;
        }

        /// <summary><c>HALF_add</c>'s elementwise form: float32 add, narrowed per element.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        private static void AddBinaryHalf(Half* o, long os, Half* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                o[i * os] = (Half)((float)o[i * os] + (float)x[i * xs]);
        }

        /// <summary><c>HALF_multiply</c>'s reduce form: float32 products, narrowed once at the call's end.</summary>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static Half MulReduceHalf(Half acc, Half* x, long xs, long n)
        {
            float f = (float)acc;
            for (long i = 0; i < n; i++)
                f *= (float)x[i * xs];
            return (Half)f;
        }

        /// <summary><c>HALF_multiply</c>'s elementwise form.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        private static void MulBinaryHalf(Half* o, long os, Half* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                o[i * os] = (Half)((float)o[i * os] * (float)x[i * xs]);
        }

        /// <summary><c>CDOUBLE_add</c>'s elementwise form: part by part.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        private static void AddBinaryComplex(Complex* o, long os, Complex* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
            {
                Complex a = o[i * os], b = x[i * xs];
                o[i * os] = new Complex(a.Real + b.Real, a.Imaginary + b.Imaginary);
            }
        }

        /// <summary>
        ///     <c>CDOUBLE_multiply</c>'s reduce form: an out of stride 0 takes the ufunc's scalar fallback loop (its SIMD path
        ///     refuses a zero output stride), MSVC-contracted (<see cref="ILKernelGenerator.PolyLoopScalarComplexMultiply"/>).
        /// </summary>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static Complex MulReduceComplex(Complex acc, Complex* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                acc = ILKernelGenerator.PolyLoopScalarComplexMultiply(acc, x[i * xs]);
            return acc;
        }

        /// <summary><c>CDOUBLE_multiply</c>'s elementwise form: the fused <c>simd_cmul</c> (<see cref="NDComplexMath.Multiply"/>).</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        private static void MulBinaryComplex(Complex* o, long os, Complex* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                o[i * os] = NDComplexMath.Multiply(o[i * os], x[i * xs]);
        }

        /// <summary><c>BOOL_add</c> (logical or) reduce.</summary>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static bool OrReduce(bool acc, bool* x, long xs, long n)
        {
            for (long i = 0; i < n && !acc; i++)
                acc = x[i * xs];
            return acc;
        }

        /// <summary><c>BOOL_add</c> (logical or) elementwise.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        private static void OrBinary(bool* o, long os, bool* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                o[i * os] = o[i * os] || x[i * xs];
        }

        /// <summary><c>BOOL_multiply</c> (logical and) reduce.</summary>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static bool AndReduce(bool acc, bool* x, long xs, long n)
        {
            for (long i = 0; i < n && acc; i++)
                acc = x[i * xs];
            return acc;
        }

        /// <summary><c>BOOL_multiply</c> (logical and) elementwise.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        private static void AndBinary(bool* o, long os, bool* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                o[i * os] = o[i * os] && x[i * xs];
        }

        /// <summary>NumSharp's decimal add reduce (sequential; no NumPy dtype).</summary>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static decimal AddReduceDecimal(decimal acc, decimal* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                acc += x[i * xs];
            return acc;
        }

        /// <summary>NumSharp's decimal add elementwise.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        private static void AddBinaryDecimal(decimal* o, long os, decimal* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                o[i * os] += x[i * xs];
        }

        /// <summary>NumSharp's decimal multiply reduce (sequential).</summary>
        /// <param name="acc">The running value.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new running value.</returns>
        private static decimal MulReduceDecimal(decimal acc, decimal* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                acc *= x[i * xs];
            return acc;
        }

        /// <summary>NumSharp's decimal multiply elementwise.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="n">The element count.</param>
        private static void MulBinaryDecimal(decimal* o, long os, decimal* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                o[i * os] *= x[i * xs];
        }

        // ---------------------------------------------------------------------------------------------
        // NumPy's pairwise sums without dynamic code (the SIMD folds are the same recursion, bit for bit)
        // ---------------------------------------------------------------------------------------------

        /// <summary><c>DOUBLE_pairwise_sum</c>: sequential below 8, eight accumulators up to 128, halves (kept a multiple of
        ///     8) above.</summary>
        /// <param name="a">The first element.</param>
        /// <param name="n">The element count.</param>
        /// <param name="stride">The stride in elements.</param>
        /// <returns>The sum (-0.0 for an empty range).</returns>
        private static double PairwiseDoubleManaged(double* a, long n, long stride)
        {
            if (n < 8)
            {
                double res = -0.0;
                for (long i = 0; i < n; i++)
                    res += a[i * stride];
                return res;
            }
            if (n <= 128)
            {
                double r0 = a[0], r1 = a[stride], r2 = a[2 * stride], r3 = a[3 * stride];
                double r4 = a[4 * stride], r5 = a[5 * stride], r6 = a[6 * stride], r7 = a[7 * stride];
                long i;
                for (i = 8; i < n - (n % 8); i += 8)
                {
                    r0 += a[(i + 0) * stride]; r1 += a[(i + 1) * stride]; r2 += a[(i + 2) * stride]; r3 += a[(i + 3) * stride];
                    r4 += a[(i + 4) * stride]; r5 += a[(i + 5) * stride]; r6 += a[(i + 6) * stride]; r7 += a[(i + 7) * stride];
                }
                double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
                for (; i < n; i++)
                    res += a[i * stride];
                return res;
            }
            long n2 = n / 2;
            n2 -= n2 % 8;
            return PairwiseDoubleManaged(a, n2, stride) + PairwiseDoubleManaged(a + n2 * stride, n - n2, stride);
        }

        /// <summary><c>FLOAT_pairwise_sum</c> (float32 accumulation), the same recursion.</summary>
        /// <param name="a">The first element.</param>
        /// <param name="n">The element count.</param>
        /// <param name="stride">The stride in elements.</param>
        /// <returns>The sum (-0.0f for an empty range).</returns>
        private static float PairwiseSingleManaged(float* a, long n, long stride)
        {
            if (n < 8)
            {
                float res = -0.0f;
                for (long i = 0; i < n; i++)
                    res += a[i * stride];
                return res;
            }
            if (n <= 128)
            {
                float r0 = a[0], r1 = a[stride], r2 = a[2 * stride], r3 = a[3 * stride];
                float r4 = a[4 * stride], r5 = a[5 * stride], r6 = a[6 * stride], r7 = a[7 * stride];
                long i;
                for (i = 8; i < n - (n % 8); i += 8)
                {
                    r0 += a[(i + 0) * stride]; r1 += a[(i + 1) * stride]; r2 += a[(i + 2) * stride]; r3 += a[(i + 3) * stride];
                    r4 += a[(i + 4) * stride]; r5 += a[(i + 5) * stride]; r6 += a[(i + 6) * stride]; r7 += a[(i + 7) * stride];
                }
                float res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
                for (; i < n; i++)
                    res += a[i * stride];
                return res;
            }
            long n2 = n / 2;
            n2 -= n2 % 8;
            return PairwiseSingleManaged(a, n2, stride) + PairwiseSingleManaged(a + n2 * stride, n - n2, stride);
        }

        /// <summary>
        ///     <c>CDOUBLE_pairwise_sum</c> over <paramref name="n"/> interleaved doubles (twice the complex count) at a stride of
        ///     <paramref name="strideBytes"/> per double (half the complex stride): four real and four imaginary accumulators.
        /// </summary>
        /// <param name="a">The first double.</param>
        /// <param name="n">The double count (even).</param>
        /// <param name="strideBytes">Half the complex stride, in bytes.</param>
        /// <param name="rr">The real sum.</param>
        /// <param name="ri">The imaginary sum.</param>
        private static void PairwiseComplexManaged(double* a, long n, long strideBytes, out double rr, out double ri)
        {
            byte* b = (byte*)a;
            if (n < 8)
            {
                rr = -0.0;
                ri = -0.0;
                for (long i = 0; i < n; i += 2)
                {
                    rr += *(double*)(b + i * strideBytes);
                    ri += *(double*)(b + i * strideBytes + 8);
                }
                return;
            }
            if (n <= 128)
            {
                double r0 = *(double*)b, r1 = *(double*)(b + 8);
                double r2 = *(double*)(b + 2 * strideBytes), r3 = *(double*)(b + 2 * strideBytes + 8);
                double r4 = *(double*)(b + 4 * strideBytes), r5 = *(double*)(b + 4 * strideBytes + 8);
                double r6 = *(double*)(b + 6 * strideBytes), r7 = *(double*)(b + 6 * strideBytes + 8);
                long i;
                for (i = 8; i < n - (n % 8); i += 8)
                {
                    r0 += *(double*)(b + (i + 0) * strideBytes); r1 += *(double*)(b + (i + 0) * strideBytes + 8);
                    r2 += *(double*)(b + (i + 2) * strideBytes); r3 += *(double*)(b + (i + 2) * strideBytes + 8);
                    r4 += *(double*)(b + (i + 4) * strideBytes); r5 += *(double*)(b + (i + 4) * strideBytes + 8);
                    r6 += *(double*)(b + (i + 6) * strideBytes); r7 += *(double*)(b + (i + 6) * strideBytes + 8);
                }
                rr = (r0 + r2) + (r4 + r6);
                ri = (r1 + r3) + (r5 + r7);
                for (; i < n; i += 2)
                {
                    rr += *(double*)(b + i * strideBytes);
                    ri += *(double*)(b + i * strideBytes + 8);
                }
                return;
            }
            long n2 = n / 2;
            n2 -= n2 % 8;
            PairwiseComplexManaged(a, n2, strideBytes, out double rr1, out double ri1);
            PairwiseComplexManaged((double*)(b + n2 * strideBytes), n - n2, strideBytes, out double rr2, out double ri2);
            rr = rr1 + rr2;
            ri = ri1 + ri2;
        }
    }
}
