using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Utilities;
using NumSharp.Utilities.Maths;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// N-D x M-D dot product: sum product over last axis of lhs and second-to-last axis of rhs.
        /// dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        /// </summary>
        /// <remarks>
        /// This replaces the original ~15,880 line implementation that used nested switch statements
        /// for every combination of dimension counts. Every layout and dtype runs one walk (<see cref="DotNDMDStrided"/>):
        /// an odometer over both operands' iterated axes and the dtype's dot function per output, four outputs at a time
        /// where a row allows. Contiguous float32 / float64 operands had a walker of their own that computed each output's
        /// offsets with an index division and read a view from its storage's start; the walk measured 1.05-2.07x faster
        /// on them, never slower, with the same bits, and reads a C-contiguous view at its own offset (an np.split /
        /// np.unstack child), which the former walker did not.
        /// </remarks>
        public NDArray DotNDMD(NDArray lhs, NDArray rhs)
        {
#if MINIMAL
            return null;
#else
            // Validate contracting dimension
            long contractDim = lhs.Shape[-1];
            if (contractDim != rhs.Shape[-2])
                throw new IncorrectShapeException(
                    string.Create(CultureInfo.InvariantCulture, $"shapes {lhs.Shape.ToPythonTuple()} and {rhs.Shape.ToPythonTuple()} not aligned: {contractDim} (dim {lhs.ndim - 1}) != {rhs.Shape[-2]} (dim {rhs.ndim - 2})"));

            // Compute output shape:
            // lhs shape without last dim + rhs shape without second-to-last dim
            long[] lshape = lhs.Shape.dimensions.RemoveAt(lhs.ndim - 1);  // All but last
            long[] rshape = rhs.Shape.dimensions.RemoveAt(rhs.ndim - 2);  // All but second-to-last

            var retShape = new long[lshape.Length + rshape.Length];
            Array.Copy(lshape, 0, retShape, 0, lshape.Length);
            Array.Copy(rshape, 0, retShape, lshape.Length, rshape.Length);

            var resultType = np._FindCommonArrayType(lhs.typecode, rhs.typecode);
            var result = NewArray(resultType, Shape.Adopt(retShape), true);

            // NumPy's N-D route (PyArray_MatrixProduct2): both operands converted to the result dtype (PyArray_FromAny),
            // then the dtype's dot function once per output element over the contracted axes (DotNDMDStrided). An operand
            // that already has the result dtype is used as it is - what astype(copy: false) returns too, without its
            // argument parsing, which a sub-microsecond dot would feel.
            NDArray ap1 = lhs.typecode == result.typecode ? lhs : lhs.astype(result.dtype, false);
            NDArray ap2 = rhs.typecode == result.typecode ? rhs : rhs.astype(result.dtype, false);
            try
            {
                DotNDMDStrided(ap1, ap2, result, lshape, rshape, contractDim);
            }
            finally
            {
                // astype(copy: false) is the operand itself when it already has the result dtype; a converted copy is
                // this call's intermediate, released here because the engine may run outside any NDScope.
                if (!ReferenceEquals(ap1, lhs))
                    ap1.Dispose();
                if (!ReferenceEquals(ap2, rhs))
                    ap2.Dispose();
            }
            return result;
#endif
        }

        /// <summary>
        ///     NumPy's N-D route of <c>PyArray_MatrixProduct2</c> for every operand pair:
        ///     the dtype's dot function (<c>PyArray_ArrFuncs.dotfunc</c>) once per output element, over the contracted
        ///     axes' strides - the last axis of <paramref name="ap1"/> and the second-to-last of <paramref name="ap2"/> - into
        ///     the C-ordered result, in the order NumPy's two all-but-axis iterators visit them.
        /// </summary>
        /// <param name="ap1">The first operand, already of the result dtype (NumPy's <c>PyArray_FromAny</c>).</param>
        /// <param name="ap2">The second operand, already of the result dtype.</param>
        /// <param name="result">The fresh C-contiguous result.</param>
        /// <param name="lshape"><paramref name="ap1"/>'s dims without the last.</param>
        /// <param name="rshape"><paramref name="ap2"/>'s dims without the second-to-last.</param>
        /// <param name="K">The contracted length.</param>
        /// <exception cref="NotSupportedException">A dtype with no dot function.</exception>
        /// <remarks>
        ///     The dot functions (<c>arraytypes.c.src</c>, and scipy-openblas behind the BLAS ones):
        ///     <list type="bullet">
        ///         <item>bool: <c>BOOL_dot</c>, true at the first pair of true elements;</item>
        ///         <item>the integers: the products summed in a C <c>long</c> / <c>long long</c> and the sum truncated to
        ///         the dtype, which is the sum modulo 2^bits: summed here in the dtype's own wrapping arithmetic, the same
        ///         value (the former route summed in double, so a sum past 2^53, or a uint64 product, lost its bits);</item>
        ///         <item>float16: <c>HALF_dot</c>, a sequential float32 sum of float32 products rounded once
        ///         (<see cref="NDHalfDot"/>);</item>
        ///         <item>float32, float64, complex128: <c>FLOAT_dot</c>, <c>DOUBLE_dot</c>, <c>CDOUBLE_dot</c> with the
        ///         scipy-openblas kernels behind them (<see cref="NDDot"/>: the stride class of the contracted runs picks
        ///         unit-stride cblas, strided cblas or the dotfunc's own loop, once per call); a row of at least four outputs
        ///         along <paramref name="ap2"/>'s innermost iterated axis goes to a row function that runs four outputs'
        ///         sums side by side (<see cref="WalkRows{T}"/>), each output keeping NumPy's order, and when one vector
        ///         pass over the result finds a NaN output, the per-output function computes it again with NumPy's NaN
        ///         (<see cref="WalkFixNaNs{T}"/>: the rows leave which NaN survives to RyuJIT);</item>
        ///         <item>decimal (no NumPy analog): an exact decimal sum.</item>
        ///     </list>
        /// </remarks>
        private static unsafe void DotNDMDStrided(NDArray ap1, NDArray ap2, NDArray result, long[] lshape, long[] rshape,
            long K)
        {
            // The contracted runs' strides are the same for every output: the BLAS dtypes pick their dot function once.
            long ls = ap1.Shape.strides[ap1.ndim - 1], rs = ap2.Shape.strides[ap2.ndim - 2];
            // A row of at least four outputs along ap2's innermost iterated axis lets a row function run several
            // outputs' sums side by side (NDDot.*RowFor); a shorter row gains nothing over one call per output.
            bool rows = rshape.Length > 0 && rshape[rshape.Length - 1] >= 4;
            // Whether the row walk has to scan its result for NaN (the rows leave which NaN survives to RyuJIT).
            bool scan = rows && OperandsMayHoldNaN(ap1, ap2, result.size);
            switch (result.typecode)
            {
                case NPTypeCode.Boolean: Walk<bool>(ap1, ap2, result, lshape, rshape, K, &BoolDot); return;
                case NPTypeCode.Byte: Walk<byte>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<byte>); return;
                case NPTypeCode.SByte: Walk<sbyte>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<sbyte>); return;
                case NPTypeCode.Int16: Walk<short>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<short>); return;
                case NPTypeCode.UInt16: Walk<ushort>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<ushort>); return;
                case NPTypeCode.Int32: Walk<int>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<int>); return;
                case NPTypeCode.UInt32: Walk<uint>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<uint>); return;
                case NPTypeCode.Int64: Walk<long>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<long>); return;
                case NPTypeCode.UInt64: Walk<ulong>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<ulong>); return;
                case NPTypeCode.Char: Walk<char>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<char>); return;
                case NPTypeCode.Half: Walk<Half>(ap1, ap2, result, lshape, rshape, K, &HalfDot); return;
                case NPTypeCode.Single:
                {
                    var row = rows ? NDDot.FloatDotRowFor(ls, rs) : null;
                    if (row != null)
                    {
                        // The row functions leave which NaN survives to RyuJIT's operand order; the walk scans what they
                        // write, and a NaN output is computed again by the per-output dot function with NumPy's NaN.
                        if (WalkRows<float>(ap1, ap2, result, lshape, rshape, K, row, scan))
                            WalkFixNaNs<float>(ap1, ap2, result, lshape, rshape, K, NDDot.FloatDotFor(ls, rs));
                    }
                    else
                        Walk<float>(ap1, ap2, result, lshape, rshape, K, NDDot.FloatDotFor(ls, rs));
                    return;
                }
                case NPTypeCode.Double:
                {
                    var row = rows ? NDDot.DoubleDotRowFor(ls, rs) : null;
                    if (row != null)
                    {
                        if (WalkRows<double>(ap1, ap2, result, lshape, rshape, K, row, scan))
                            WalkFixNaNs<double>(ap1, ap2, result, lshape, rshape, K, NDDot.DoubleDotFor(ls, rs));
                    }
                    else
                        Walk<double>(ap1, ap2, result, lshape, rshape, K, NDDot.DoubleDotFor(ls, rs));
                    return;
                }
                case NPTypeCode.Decimal: Walk<decimal>(ap1, ap2, result, lshape, rshape, K, &DecimalDot); return;
                case NPTypeCode.Complex:
                {
                    var row = rows ? NDDot.ComplexDotRowFor(ls, rs) : null;
                    if (row != null)
                    {
                        if (WalkRows<Complex>(ap1, ap2, result, lshape, rshape, K, row, scan))
                            WalkFixNaNs<Complex>(ap1, ap2, result, lshape, rshape, K, NDDot.ComplexDotFor(ls, rs));
                    }
                    else
                        Walk<Complex>(ap1, ap2, result, lshape, rshape, K, NDDot.ComplexDotFor(ls, rs));
                    return;
                }
                default: throw new NotSupportedException($"np.dot of {result.typecode} operands");
            }
        }

        /// <summary>
        ///     The output walk of <see cref="DotNDMDStrided"/>: for each position of <paramref name="ap1"/> but its last axis
        ///     (C order), each position of <paramref name="ap2"/> but its second-to-last (C order), one dot over the contracted
        ///     axes written to the next element of the C-ordered result - NumPy's <c>op += os</c> under its two iterators.
        /// </summary>
        /// <typeparam name="T">The element type of the result dtype.</typeparam>
        /// <param name="ap1">The first operand (of <typeparamref name="T"/>).</param>
        /// <param name="ap2">The second operand (of <typeparamref name="T"/>).</param>
        /// <param name="result">The fresh C-contiguous result (of <typeparamref name="T"/>).</param>
        /// <param name="lshape"><paramref name="ap1"/>'s dims without the last.</param>
        /// <param name="rshape"><paramref name="ap2"/>'s dims without the second-to-last.</param>
        /// <param name="K">The contracted length.</param>
        /// <param name="dot">The dtype's dot function: first operand, its stride, second operand, its stride, length (strides
        ///     in elements).</param>
        /// <remarks>
        ///     The operands are addressed from their storage plus the view's offset, so any layout - strided, reversed,
        ///     broadcast - is read through its own strides; the arrays are the caller's parameters and locals for the whole
        ///     walk, so the raw pointers stay valid. Each operand's position moves like NumPy's <c>PyArray_ITER_NEXT</c>: a
        ///     coordinate odometer over the iterated axes that adds a stride per step and takes back a whole axis on a carry,
        ///     so an output costs one dot call and no division.
        /// </remarks>
        private static unsafe void Walk<T>(NDArray ap1, NDArray ap2, NDArray result, long[] lshape, long[] rshape, long K,
            delegate*<T*, long, T*, long, long, T> dot) where T : unmanaged
        {
            int lnd = ap1.ndim, rnd = ap2.ndim;
            long[] lst = ap1.Shape.strides, rst = ap2.Shape.strides;
            T* lp = (T*)ap1.Storage.Address + ap1.Shape.offset;
            T* rp = (T*)ap2.Storage.Address + ap2.Shape.offset;
            T* op = (T*)result.Storage.Address;
            long ls = lst[lnd - 1];
            long rs = rst[rnd - 2];

            long totalL = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalL *= lshape[i];
            long totalR = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalR *= rshape[i];
            if (totalL == 0 || totalR == 0)
                return;   // an empty result: nothing to write

            // The strides of the iterated axes in C order, as PyArray_IterAllButAxis leaves them: ap1's axes but the last
            // (lshape), ap2's axes but the second-to-last (rshape). Ranks past 16 take heap arrays (NumSharp has no rank
            // limit); the coordinates start at zero and are reset at every pass of the inner walk.
            int lk = lshape.Length, rk = rshape.Length;
            Span<long> lStride = lk <= 16 ? stackalloc long[16] : new long[lk];
            Span<long> rStride = rk <= 16 ? stackalloc long[16] : new long[rk];
            Span<long> lc = lk <= 16 ? stackalloc long[16] : new long[lk];
            Span<long> rc = rk <= 16 ? stackalloc long[16] : new long[rk];
            for (int d = 0; d < lk; d++)
                lStride[d] = lst[d];
            for (int d = 0, j = 0; d < rnd; d++)
                if (d != rnd - 2)
                    rStride[j++] = rst[d];
            lc.Clear();

            long lOff = 0;
            for (long li = 0; li < totalL; li++)
            {
                rc.Clear();
                long rOff = 0;
                for (long ri = 0; ri < totalR; ri++)
                {
                    *op++ = dot(lp + lOff, ls, rp + rOff, rs, K);
                    for (int d = rk - 1; d >= 0; d--)
                    {
                        rOff += rStride[d];
                        if (++rc[d] < rshape[d])
                            break;
                        rOff -= rStride[d] * rshape[d];
                        rc[d] = 0;
                    }
                }

                for (int d = lk - 1; d >= 0; d--)
                {
                    lOff += lStride[d];
                    if (++lc[d] < lshape[d])
                        break;
                    lOff -= lStride[d] * lshape[d];
                    lc[d] = 0;
                }
            }
        }

        /// <summary>
        ///     <see cref="Walk{T}"/> with the innermost iterated axis of <paramref name="ap2"/> handed to a row function in one
        ///     call: the same outputs in the same C order, each computed by the row function as the per-output dot function
        ///     would, so a row function may interleave several outputs' arithmetic without changing any one of them.
        /// </summary>
        /// <typeparam name="T">The element type of the result dtype.</typeparam>
        /// <param name="ap1">The first operand (of <typeparamref name="T"/>).</param>
        /// <param name="ap2">The second operand (of <typeparamref name="T"/>).</param>
        /// <param name="result">The fresh C-contiguous result (of <typeparamref name="T"/>).</param>
        /// <param name="lshape"><paramref name="ap1"/>'s dims without the last.</param>
        /// <param name="rshape"><paramref name="ap2"/>'s dims without the second-to-last (at least one axis).</param>
        /// <param name="K">The contracted length.</param>
        /// <param name="row">The row function: first operand, its stride, the row's first second operand, its stride, the
        ///     element distance between consecutive outputs' second operands, the row's first output, the output count, the
        ///     length (strides in elements). The row's outputs are consecutive in the C-ordered result.</param>
        /// <param name="scan">Whether to scan the outputs for NaN (false when <see cref="OperandsMayHoldNaN"/> has shown no
        ///     operand holds one).</param>
        /// <returns>Whether an output is NaN (the caller computes those again with NumPy's NaN, <see cref="WalkFixNaNs{T}"/>:
        ///     the row functions leave which NaN survives to RyuJIT's operand order); false when not scanning.</returns>
        /// <remarks>
        ///     <para>The positions advance as in <see cref="Walk{T}"/>: a coordinate odometer over the iterated axes, here
        ///     over <paramref name="ap2"/>'s axes but the innermost, which each row call covers whole.</para>
        ///     <para>The outputs are scanned for NaN 512 at a time, as soon as the rows have written them (still in L1):
        ///     one scan of the whole result after the walk re-read it from L2, about 5% of np.dot of a (300, 3, 3)
        ///     complex array by a transposed (8, 3), whose 3-term dots cost little else.</para>
        /// </remarks>
        private static unsafe bool WalkRows<T>(NDArray ap1, NDArray ap2, NDArray result, long[] lshape, long[] rshape,
            long K, delegate*<T*, long, T*, long, long, T*, long, long, void> row, bool scan) where T : unmanaged
        {
            int lnd = ap1.ndim, rnd = ap2.ndim;
            long[] lst = ap1.Shape.strides, rst = ap2.Shape.strides;
            T* lp = (T*)ap1.Storage.Address + ap1.Shape.offset;
            T* rp = (T*)ap2.Storage.Address + ap2.Shape.offset;
            T* op = (T*)result.Storage.Address;
            long ls = lst[lnd - 1];
            long rs = rst[rnd - 2];

            long totalL = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalL *= lshape[i];
            long totalR = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalR *= rshape[i];
            if (totalL == 0 || totalR == 0)
                return false;   // an empty result: nothing to write

            int lk = lshape.Length, rk = rshape.Length;
            Span<long> lStride = lk <= 16 ? stackalloc long[16] : new long[lk];
            Span<long> rStride = rk <= 16 ? stackalloc long[16] : new long[rk];
            Span<long> lc = lk <= 16 ? stackalloc long[16] : new long[lk];
            Span<long> rc = rk <= 16 ? stackalloc long[16] : new long[rk];
            for (int d = 0; d < lk; d++)
                lStride[d] = lst[d];
            for (int d = 0, j = 0; d < rnd; d++)
                if (d != rnd - 2)
                    rStride[j++] = rst[d];
            lc.Clear();

            long inner = rshape[rk - 1];          // the row: every position of ap2's innermost iterated axis
            long innerStride = rStride[rk - 1];
            long rows = totalR / inner;
            T* unscanned = op;                    // the first output not yet scanned for NaN
            bool anyNaN = false;

            long lOff = 0;
            for (long li = 0; li < totalL; li++)
            {
                rc.Clear();
                long rOff = 0;
                for (long ro = 0; ro < rows; ro++)
                {
                    row(lp + lOff, ls, rp + rOff, rs, innerStride, op, inner, K);
                    op += inner;
                    if (scan && op - unscanned >= 512)
                    {
                        anyNaN |= AnyNaNOf(unscanned, op - unscanned);
                        unscanned = op;
                    }
                    for (int d = rk - 2; d >= 0; d--)
                    {
                        rOff += rStride[d];
                        if (++rc[d] < rshape[d])
                            break;
                        rOff -= rStride[d] * rshape[d];
                        rc[d] = 0;
                    }
                }

                for (int d = lk - 1; d >= 0; d--)
                {
                    lOff += lStride[d];
                    if (++lc[d] < lshape[d])
                        break;
                    lOff -= lStride[d] * lshape[d];
                    lc[d] = 0;
                }
            }

            return scan && (anyNaN | AnyNaNOf(unscanned, op - unscanned));
        }

        /// <summary>
        ///     Whether an output of the row walk can hold a NaN whose bits depend on the operand order: only when an operand
        ///     holds a NaN - otherwise every NaN output comes from an invalid <c>inf * 0</c> / <c>inf - inf</c>, which is
        ///     x86's default NaN whichever operand order. The operands are scanned instead of the result when they are
        ///     smaller and each is one contiguous block (a short contracted axis: np.dot of (300, 3, 3) by (3, 8) reads
        ///     2,724 elements and writes 7,200); otherwise the walk scans the result.
        /// </summary>
        /// <param name="ap1">The first operand.</param>
        /// <param name="ap2">The second operand.</param>
        /// <param name="outputs">The result's size.</param>
        /// <returns>False when neither operand holds a NaN (the walk need not scan); true when one does, or when the result
        ///     scan is the cheaper check.</returns>
        /// <remarks>Only on an x86 host: elsewhere an invalid operation makes another default NaN (Arm64's is positive),
        ///     which the walk's scan and the per-output models turn into x86's.</remarks>
        private static unsafe bool OperandsMayHoldNaN(NDArray ap1, NDArray ap2, long outputs)
        {
            if (!System.Runtime.Intrinsics.X86.X86Base.IsSupported || ap1.size + ap2.size >= outputs)
                return true;
            return BlockMayHoldNaN(ap1) || BlockMayHoldNaN(ap2);
        }

        /// <summary>Whether an operand may hold a NaN: a C- or F-contiguous float32 / float64 / complex128 block is
        ///     scanned (<see cref="NDDot.AnyNaN(double*, long)"/>); any other view answers true (unknown without a walk).</summary>
        /// <param name="x">The operand (the result dtype's element type: the walk's operands are converted first).</param>
        /// <returns>True unless the block was scanned and holds no NaN.</returns>
        private static unsafe bool BlockMayHoldNaN(NDArray x)
        {
            if (!x.Shape.IsContiguous && !x.Shape.IsFContiguous)
                return true;
            switch (x.typecode)
            {
                case NPTypeCode.Single:
                    return NDDot.AnyNaN((float*)x.Storage.Address + x.Shape.offset, x.size);
                case NPTypeCode.Double:
                    return NDDot.AnyNaN((double*)x.Storage.Address + x.Shape.offset, x.size);
                case NPTypeCode.Complex:
                    return NDDot.AnyNaN((double*)((Complex*)x.Storage.Address + x.Shape.offset), 2 * x.size);
                default:
                    return true;
            }
        }

        /// <summary>Whether any of <paramref name="count"/> consecutive outputs is NaN (<see cref="NDDot.AnyNaN(double*, long)"/>
        ///     for float32, float64 and complex128 - two doubles an output; never for the other dtypes).</summary>
        /// <typeparam name="T">The result's element type (a constant per instantiation: the JIT keeps one branch).</typeparam>
        /// <param name="p">The first output.</param>
        /// <param name="count">The number of outputs.</param>
        /// <returns>True when one holds a NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool AnyNaNOf<T>(T* p, long count) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
                return NDDot.AnyNaN((float*)p, count);
            if (typeof(T) == typeof(double))
                return NDDot.AnyNaN((double*)p, count);
            if (typeof(T) == typeof(Complex))
                return NDDot.AnyNaN((double*)p, 2 * count);
            return false;
        }

        /// <summary>
        ///     After <see cref="WalkRows{T}"/>: every NaN output of <paramref name="result"/> computed again by the per-output
        ///     dot function, which gives NumPy's NaN where several meet (NDDot.NaN.cs) - the row functions' value is the same,
        ///     but which NaN survives follows RyuJIT's operand order there. The outputs are visited as
        ///     <see cref="Walk{T}"/> visits them (the same odometer), and only a NaN one costs a dot.
        /// </summary>
        /// <typeparam name="T">float, double or Complex.</typeparam>
        /// <param name="ap1">The first operand.</param>
        /// <param name="ap2">The second operand.</param>
        /// <param name="result">The C-contiguous result the row walk filled.</param>
        /// <param name="lshape"><paramref name="ap1"/>'s dims without the last.</param>
        /// <param name="rshape"><paramref name="ap2"/>'s dims without the second-to-last.</param>
        /// <param name="K">The contracted length.</param>
        /// <param name="dot">The per-output dot function for the call's strides (NDDot.*DotFor).</param>
        /// <remarks>Why not a check inside the row functions: their four-output loops are short of registers, and a NaN
        ///     branch there (or a scan per row) cost short rows of tiny dots up to 14% (np.dot of (300, 3, 3) by a
        ///     transposed (8, 3)); one vector pass over the whole result costs about a quarter of a cycle per output.</remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe void WalkFixNaNs<T>(NDArray ap1, NDArray ap2, NDArray result, long[] lshape, long[] rshape,
            long K, delegate*<T*, long, T*, long, long, T> dot) where T : unmanaged, INumberBase<T>
        {
            int lnd = ap1.ndim, rnd = ap2.ndim;
            long[] lst = ap1.Shape.strides, rst = ap2.Shape.strides;
            T* lp = (T*)ap1.Storage.Address + ap1.Shape.offset;
            T* rp = (T*)ap2.Storage.Address + ap2.Shape.offset;
            T* op = (T*)result.Storage.Address;
            long ls = lst[lnd - 1];
            long rs = rst[rnd - 2];

            long totalL = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalL *= lshape[i];
            long totalR = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalR *= rshape[i];

            int lk = lshape.Length, rk = rshape.Length;
            var lStride = new long[Math.Max(lk, 1)];
            var rStride = new long[Math.Max(rk, 1)];
            var lc = new long[Math.Max(lk, 1)];
            var rc = new long[Math.Max(rk, 1)];
            for (int d = 0; d < lk; d++)
                lStride[d] = lst[d];
            for (int d = 0, j = 0; d < rnd; d++)
                if (d != rnd - 2)
                    rStride[j++] = rst[d];

            long lOff = 0;
            for (long li = 0; li < totalL; li++)
            {
                Array.Clear(rc);
                long rOff = 0;
                for (long ri = 0; ri < totalR; ri++, op++)
                {
                    if (T.IsNaN(*op))
                        *op = dot(lp + lOff, ls, rp + rOff, rs, K);
                    for (int d = rk - 1; d >= 0; d--)
                    {
                        rOff += rStride[d];
                        if (++rc[d] < rshape[d])
                            break;
                        rOff -= rStride[d] * rshape[d];
                        rc[d] = 0;
                    }
                }

                for (int d = lk - 1; d >= 0; d--)
                {
                    lOff += lStride[d];
                    if (++lc[d] < lshape[d])
                        break;
                    lOff -= lStride[d] * lshape[d];
                    lc[d] = 0;
                }
            }
        }

        /// <summary>NumPy's <c>BOOL_dot</c>: true at the first index where both elements are true.</summary>
        /// <param name="a">The first operand.</param><param name="sa">Its stride (elements).</param>
        /// <param name="b">The second operand.</param><param name="sb">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>Whether any pair is (true, true).</returns>
        private static unsafe bool BoolDot(bool* a, long sa, bool* b, long sb, long n)
        {
            for (long i = 0; i < n; i++, a += sa, b += sb)
                if (*a && *b)
                    return true;
            return false;
        }

        /// <summary>
        ///     NumPy's integer dot functions (<c>BYTE_dot</c> … <c>ULONGLONG_dot</c>): the products summed in a C
        ///     <c>long</c> (<c>long long</c> for the 64-bit types) and the sum cast to the dtype. Integer arithmetic wraps,
        ///     so that is the exact sum of the products modulo 2^bits of the dtype, whatever the width of the accumulator:
        ///     summed here in <typeparamref name="T"/>'s own wrapping arithmetic.
        /// </summary>
        /// <typeparam name="T">The integer element type (char: NumSharp's unsigned 16-bit dtype).</typeparam>
        /// <param name="a">The first operand.</param><param name="sa">Its stride (elements).</param>
        /// <param name="b">The second operand.</param><param name="sb">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>The sum of the products, wrapped to <typeparamref name="T"/>.</returns>
        private static unsafe T IntegerDot<T>(T* a, long sa, T* b, long sb, long n) where T : unmanaged, IBinaryInteger<T>
        {
            // Four independent sums, so consecutive multiply-adds do not wait on one accumulator. Wrapping addition is
            // associative and commutative (arithmetic modulo 2^bits), so any grouping gives NumPy's exact value.
            T s0 = T.Zero, s1 = T.Zero, s2 = T.Zero, s3 = T.Zero;
            long sa2 = 2 * sa, sa3 = 3 * sa, sa4 = 4 * sa, sb2 = 2 * sb, sb3 = 3 * sb, sb4 = 4 * sb;
            long i = 0;
            for (; i + 4 <= n; i += 4, a += sa4, b += sb4)
            {
                s0 = unchecked(s0 + a[0] * b[0]);
                s1 = unchecked(s1 + a[sa] * b[sb]);
                s2 = unchecked(s2 + a[sa2] * b[sb2]);
                s3 = unchecked(s3 + a[sa3] * b[sb3]);
            }

            for (; i < n; i++, a += sa, b += sb)
                s0 = unchecked(s0 + *a * *b);
            return unchecked(s0 + s1 + (s2 + s3));
        }

        /// <summary>NumPy's <c>HALF_dot</c>: float32 products summed in float32, rounded once to float16
        ///     (<see cref="NDHalfDot.Sum"/>, which keeps HALF_dot's NaN operand order).</summary>
        /// <param name="a">The first operand.</param><param name="sa">Its stride (elements).</param>
        /// <param name="b">The second operand.</param><param name="sb">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>The dot product.</returns>
        private static unsafe Half HalfDot(Half* a, long sa, Half* b, long sb, long n) => (Half)NDHalfDot.Sum(a, sa, b, sb, n);

        /// <summary>The decimal dot (no NumPy analog): an exact decimal sum of the products.</summary>
        /// <param name="a">The first operand.</param><param name="sa">Its stride (elements).</param>
        /// <param name="b">The second operand.</param><param name="sb">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>The dot product.</returns>
        /// <exception cref="OverflowException">The sum leaves decimal's range.</exception>
        private static unsafe decimal DecimalDot(decimal* a, long sa, decimal* b, long sb, long n)
        {
            decimal s = 0m;
            for (long i = 0; i < n; i++, a += sa, b += sb)
                s += *a * *b;
            return s;
        }

    }
}
