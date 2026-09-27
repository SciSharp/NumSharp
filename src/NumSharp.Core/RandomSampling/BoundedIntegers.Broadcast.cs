using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    // The ARRAY-bounds half of BoundedIntegers (the type's documentation lives on the main part, BoundedIntegers.cs):
    // NumPy's _bounded_integers.pyx.in `_rand_<dtype>` entry for array bounds — the scalar-vs-broadcast dispatch, the
    // per-dtype bound checks in NumPy's order and words, the element-wise Python-int() conversion of 64-bit bounds (with
    // NumPy's K-order scramble for non-C float bounds), and the draw loop that carries the sub-word buffer across
    // positions — plus ClampToInt128, which lets a Python-int (BigInteger) bound too wide for any dtype reach the
    // scalar path's out-of-bounds check. Entry points: Generator.integers(NDArray …)/(BigInteger …) and
    // NumPyRandom.randint(NDArray …)/(BigInteger …).
    internal static partial class BoundedIntegers
    {
        /// <summary>
        ///     NumPy's <c>_rand_&lt;dtype&gt;</c> for ARRAY-valued bounds (<c>Generator.integers(low_arr, high_arr)</c>,
        ///     <c>RandomState.randint(low_arr, high_arr)</c>): the scalar path when both bounds are 0-d, else
        ///     <c>_rand_&lt;dtype&gt;_broadcast</c> — per-element bounds checked NumPy's way, one bounded draw per output
        ///     position in C order, the sub-word buffer carried from one position to the next.
        /// </summary>
        /// <param name="bg">The bit generator; its lock is taken for the draw loop.</param>
        /// <param name="low">The first bound as received (null is Python's <c>None</c>).</param>
        /// <param name="high">The second bound as received; null is NumPy's <c>high=None</c> — the bounds become
        ///     <c>[0, low)</c>.</param>
        /// <param name="size">The output shape (default = NumPy's <c>size=None</c>: the bounds' broadcast shape, or one value
        ///     when both are 0-d).</param>
        /// <param name="dtype">The requested dtype (null = the caller's default is already substituted).</param>
        /// <param name="endpoint">Whether <paramref name="high"/> is inclusive.</param>
        /// <param name="useMasked">True for the legacy masked-rejection sampler, false for the Generator's Lemire sampler.</param>
        /// <param name="methodName">The public method named in the unsupported-dtype message.</param>
        /// <param name="legacyByteOrder">True for <c>RandomState.randint</c> (a non-native byte order draws the native twin).</param>
        /// <returns>The draws, of the resolved native dtype.</returns>
        /// <exception cref="TypeError">The dtype is not one of the nine integer/bool dtypes; a bound is <c>None</c> or a 0-d
        ///     complex (Python's <c>int()</c> refuses both).</exception>
        /// <exception cref="ValueError">A non-native dtype (Generator); a bound outside the dtype (<c>low/high is out of
        ///     bounds for &lt;dtype&gt;</c>); an empty interval anywhere (<c>low &gt;= high</c>, or <c>high &lt;= 0</c> when
        ///     every low is 0); a NaN reaching Python's <c>int()</c> (<c>cannot convert float NaN to integer</c>); bounds
        ///     that do not broadcast (the comparison ufunc's <c>operands could not be broadcast together with shapes …</c>)
        ///     or a size they do not broadcast with (<c>shape mismatch: … arg 0 … arg 2 …</c>); a negative size.</exception>
        /// <exception cref="OverflowException">An infinity reaching Python's <c>int()</c> (NumPy's
        ///     <c>OverflowError: cannot convert float infinity to integer</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's check ORDER is reproduced because it decides which error a bad request reports: the zero-size
        ///     shortcut (an empty size returns before any bound is looked at), then for 8/16/32-bit dtypes and bool the low
        ///     bound, the high bound and the ordering (each over the whole array, each skipped where the bound's dtype alone
        ///     guarantees it), for 64-bit dtypes the per-element conversions first; then the output and the broadcast.
        ///     </para>
        ///     <para>
        ///     Two NumPy quirks are kept on purpose, since they change the numbers a caller gets. (1) A size SMALLER than the
        ///     bounds' broadcast shape but broadcastable into it (<c>integers(zeros((2, 3)), 10, size=3)</c>) is not an error:
        ///     NumPy's legacy multi-iterator broadcasts the output too, and its loop runs <c>size</c> positions — so the result
        ///     takes the first <c>size</c> positions of the bounds' broadcast in C order. (2) On the 64-bit dtypes a bound that
        ///     does not cast safely (a float, a uint64 bound for int64, …) is converted element by element into
        ///     <c>np.empty_like(bound)</c> — which keeps a non-C layout — while the elements are read in C order and written
        ///     in memory order, so an F-ordered or broadcast float bound arrives scrambled. The numbers NumPy draws are those
        ///     of the scrambled bounds, and so are NumSharp's.
        ///     </para>
        /// </remarks>
        internal static NDArray DrawArray(BitGenerator bg, NDArray low, NDArray high, Shape size, DType dtype, bool endpoint,
                                          bool useMasked, string methodName, bool legacyByteOrder)
        {
            ResolveDtype(dtype, methodName, legacyByteOrder, out NPTypeCode tc, out int width, out Int128 lb, out Int128 ub);
            DType resultType = DType.From(tc);

            // `if size is not None and np.prod(size) == 0: return np.empty(size, dtype)` — before either bound is read.
            if (!size.IsEmpty && size.size == 0)
                return new NDArray(resultType, size);

            // `if high is None: high = low; low = 0` — the Python int 0, whose np.asarray is a 0-d int64.
            bool oneArg = high is null;
            using var zero = oneArg ? NDArray.Scalar(0L) : null;
            NDArray lo = oneArg ? zero : low;
            NDArray hi = oneArg ? low : high;

            // Both 0-d: `low = int(low_arr); high = int(high_arr)` then the scalar path (int() of a 0-d array: a float
            // truncates, NaN / inf / complex / None raise).
            if ((lo is null || lo.ndim == 0) && (hi is null || hi.ndim == 0))
                return Draw(bg, PyInt(lo), PyInt(hi), size, dtype, endpoint, useMasked, methodName, legacyByteOrder);

            return width == 64
                ? Broadcast64(bg, lo, hi, size, tc, lb, ub, endpoint, useMasked)
                : BroadcastSmall(bg, lo, hi, size, tc, width, lb, ub, endpoint, useMasked);
        }

        /// <summary>
        ///     <c>_rand_&lt;dtype&gt;_broadcast</c> for the 8/16/32-bit dtypes and bool: whole-array bound checks, then both
        ///     bounds FORCE-cast to the next larger type (a float truncates, a NaN takes the C cast's value) and one
        ///     <c>random_buffered_bounded_&lt;utype&gt;</c> draw per position.
        /// </summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="lo">The low bound (not both 0-d with <paramref name="hi"/>; null is Python's <c>None</c>).</param>
        /// <param name="hi">The high bound.</param>
        /// <param name="size">The output shape, or default for the bounds' broadcast shape.</param>
        /// <param name="tc">The output dtype.</param>
        /// <param name="width">1 (bool), 8, 16 or 32.</param>
        /// <param name="lb">The dtype's inclusive minimum.</param>
        /// <param name="ub">The dtype's inclusive maximum.</param>
        /// <param name="closed">Whether the interval is closed (<c>endpoint=True</c>).</param>
        /// <param name="useMasked">Masked rejection (legacy) instead of Lemire.</param>
        /// <returns>The draws.</returns>
        /// <exception cref="TypeError">A bound is <c>None</c> (the comparison refuses it, as Python's does).</exception>
        /// <exception cref="ValueError">See <see cref="DrawArray"/>.</exception>
        private static NDArray BroadcastSmall(BitGenerator bg, NDArray lo, NDArray hi, Shape size, NPTypeCode tc, int width,
                                              Int128 lb, Int128 ub, bool closed, bool useMasked)
        {
            DType otype = DType.From(tc);
            NPTypeCode up = UpType(tc);
            string name = tc.AsNumpyDtypeName();

            // `if np.can_cast(low_arr, otype): pass  elif np.any(np.less(low_arr, otype(lb))): raise` — the dtype alone
            // may already rule the value out of range; the bound is compared as a STRONG scalar of the output dtype.
            if (lo is null)
                throw new TypeError("'<' not supported between instances of 'NoneType' and 'int'");
            if (!np.can_cast(lo.dtype, otype))
            {
                using var lbArr = ScalarOf(lb, tc);
                if (RandomBroadcast.AnyCompare(NDExpr.Less, lo, lbArr))
                    throw new ValueError($"low is out of bounds for {name}");
            }
            if (hi is null)
                throw new TypeError($"'{(closed ? ">=" : ">")}' not supported between instances of 'NoneType' and 'int'");
            if (!np.can_cast(hi.dtype, otype))
            {
                // The EXCLUSIVE bound as the next larger type (np.int64(0x80000000) for int32, np.uint8(2) for bool).
                using var ubArr = ScalarOf(ub + 1, up);
                if (RandomBroadcast.AnyCompare(closed ? NDExpr.GreaterEqual : NDExpr.Greater, hi, ubArr))
                    throw new ValueError($"high is out of bounds for {name}");
            }
            if (RandomBroadcast.AnyCompare(closed ? NDExpr.Greater : NDExpr.GreaterEqual, lo, hi))
                throw new ValueError(FormatBoundsError(closed, lo));

            // PyArray_FROM_OTF(x, <up>, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST): the unsafe cast NumPy's astype does,
            // then widened exactly to an 8-byte carrier the loop reads as raw bits (wrapping arithmetic then truncating
            // to the draw width gives the C loop's bits whatever the intermediate width).
            using var loW = WidenToCarrier(lo, up);
            using var hiW = WidenToCarrier(hi, up);
            return DrawLoop(bg, loW, hiW, size, tc, width, closed ? 0UL : 1UL, useMasked);
        }

        /// <summary>
        ///     <c>_rand_&lt;dtype&gt;_broadcast</c> for int64/uint64: each bound cast safely when its dtype allows, else
        ///     converted element by element through Python's <c>int()</c> with a range check (the high one already made
        ///     closed), then the ordering check and one <c>random_bounded_uint64</c> draw per position.
        /// </summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="lo">The low bound (null is Python's <c>None</c>).</param>
        /// <param name="hi">The high bound.</param>
        /// <param name="size">The output shape, or default for the bounds' broadcast shape.</param>
        /// <param name="tc">Int64 or UInt64.</param>
        /// <param name="lb">The dtype's inclusive minimum.</param>
        /// <param name="ub">The dtype's inclusive maximum.</param>
        /// <param name="closed">Whether the interval is closed.</param>
        /// <param name="useMasked">Masked rejection (legacy) instead of Lemire.</param>
        /// <returns>The draws.</returns>
        /// <exception cref="TypeError">A bound is <c>None</c> or holds a value <c>int()</c> refuses.</exception>
        /// <exception cref="ValueError">See <see cref="DrawArray"/>.</exception>
        /// <exception cref="OverflowException">An infinite float bound.</exception>
        private static NDArray Broadcast64(BitGenerator bg, NDArray lo, NDArray hi, Shape size, NPTypeCode tc, Int128 lb,
                                           Int128 ub, bool closed, bool useMasked)
        {
            DType otype = DType.From(tc);
            string name = tc.AsNumpyDtypeName();
            ulong isOpen = closed ? 0UL : 1UL;

            NDArray loW = null, hiW = null;
            try
            {
                if (lo is null)
                    throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
                loW = np.can_cast(lo.dtype, otype)
                    ? lo.astype(tc)
                    : ConvertElementwise(lo, tc, v => v < lb || v > ub ? $"low is out of bounds for {name}" : null);

                if (hi is null)
                    throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
                if (np.can_cast(hi.dtype, otype))
                    hiW = hi.astype(tc);
                else
                {
                    // `closed_upper = int(flat[i]) - is_open`, range-checked, and the interval made closed right here.
                    NDArray lowForMessage = loW;
                    hiW = ConvertElementwise(hi, tc, v => v - (Int128)isOpen > ub ? $"high is out of bounds for {name}"
                                                        : v - (Int128)isOpen < lb ? FormatBoundsError(closed, lowForMessage)
                                                        : null, (Int128)isOpen);
                    isOpen = 0;
                }

                // `low_high_comp = greater if not is_open else greater_equal` (is_open is 0 once the high was closed above).
                if (RandomBroadcast.AnyCompare(isOpen == 0 ? NDExpr.Greater : NDExpr.GreaterEqual, loW, hiW))
                    throw new ValueError(FormatBoundsError(closed, loW));

                return DrawLoop(bg, loW, hiW, size, tc, 64, isOpen, useMasked);
            }
            finally
            {
                if (!ReferenceEquals(loW, lo))
                    loW?.Dispose();
                if (!ReferenceEquals(hiW, hi))
                    hiW?.Dispose();
            }
        }

        /// <summary>
        ///     The draw loop shared by both broadcast paths: allocates the output (NumPy's <c>np.empty(size)</c>, or the
        ///     bounds' broadcast shape), walks <c>MultiIterNew3(low, high, out)</c> for <c>out.size</c> positions in C order
        ///     and draws <c>[low, high - is_open]</c> at each.
        /// </summary>
        /// <param name="bg">The bit generator; its lock is held for the loop.</param>
        /// <param name="loW">The low bounds, as an 8-byte carrier (int64/uint64) read as raw bits.</param>
        /// <param name="hiW">The high bounds, likewise.</param>
        /// <param name="size">The output shape, or default.</param>
        /// <param name="tc">The output dtype.</param>
        /// <param name="width">1 (bool), 8, 16, 32 or 64.</param>
        /// <param name="isOpen">1 when the high bound is still exclusive, 0 when already closed.</param>
        /// <param name="useMasked">Masked rejection (legacy) instead of Lemire.</param>
        /// <returns>The draws.</returns>
        /// <exception cref="ValueError">A negative size (<c>negative dimensions are not allowed</c>), or a size the bounds do
        ///     not broadcast with (<c>shape mismatch: … arg 0 … arg 2 …</c>: low, high, output).</exception>
        private static unsafe NDArray DrawLoop(BitGenerator bg, NDArray loW, NDArray hiW, Shape size, NPTypeCode tc, int width,
                                               ulong isOpen, bool useMasked)
        {
            DType resultType = DType.From(tc);
            // np.empty(size) — or np.empty(MultiIterNew2(low, high).shape) — built from FRESH dimensions (a caller's view
            // Shape carries strides an allocation must not inherit).
            long[] outDims = size.IsEmpty
                ? BroadcastDims(loW.Shape, hiW.Shape)
                : (long[])(size.dimensions ?? Array.Empty<long>()).Clone();
            var ret = new NDArray(resultType, new Shape(outDims), false);
            NDArray loIt = null, hiIt = null, outIt = null;
            try
            {
                // MultiIterNew3(low, high, out): an incompatible size is its mismatch error (arg 2 is the output); a size
                // that broadcasts INTO a larger bounds shape is legal, and the loop covers only out.size positions.
                long[] itDims = BroadcastDims(loW.Shape, hiW.Shape, ret.Shape);
                if (SameDimensions(itDims, outDims))
                {
                    loIt = loW;
                    hiIt = hiW;
                    outIt = ret;
                }
                else
                {
                    // The first out.size positions of the bounds' broadcast, in C order (NumPy's out_data[i] is written by
                    // the flat index while the iterator walks the larger broadcast shape).
                    long n = ret.size;
                    loIt = FirstPositions(loW, itDims, n);
                    hiIt = FirstPositions(hiW, itDims, n);
                    outIt = ret.reshape(n);
                }

                ulong widthMask = width == 64 ? ulong.MaxValue : width == 32 ? 0xFFFFFFFFUL : width == 16 ? 0xFFFFUL : 0xFFUL;
                int itemsize = width == 64 ? 8 : width == 32 ? 4 : width == 16 ? 2 : 1;
                lock (bg.@lock)
                {
                    // NumPy declares the sub-word buffer once, outside the loop: one 32-bit word feeds several positions.
                    uint buf = 0;
                    int bcnt = 0;
                    using var walk = new BroadcastWalk(outIt, loIt, hiIt);
                    while (walk.Next())
                    {
                        // The chunk's pointers and BYTE strides, read once per chunk into locals: read through the walk
                        // object inside the loop they were reloaded for every position (the virtual NextUInt32/NextUInt64
                        // calls may, as far as the JIT can tell, write the object's fields).
                        byte* pLo = walk.A, pHi = walk.B, pOut = walk.Out;
                        long sLo = walk.StrideA * 8, sHi = walk.StrideB * 8, sOut = walk.OutStride * itemsize;
                        long count = walk.Count;
                        for (long j = 0; j < count; j++, pLo += sLo, pHi += sHi, pOut += sOut)
                        {
                            ulong l = *(ulong*)pLo;
                            ulong h = *(ulong*)pHi;
                            // rng = <utype>((high_v - is_open) - low_v); off = <utype>low_v — wrapping 64-bit arithmetic
                            // truncated to the draw width is bit-identical to the C loop's.
                            ulong rng = unchecked(h - isOpen - l) & widthMask;
                            ulong off = l & widthMask;
                            ulong v = DrawOne(bg, width, off, rng, useMasked, ref buf, ref bcnt);
                            switch (itemsize)
                            {
                                case 1: *pOut = (byte)v; break;
                                case 2: *(ushort*)pOut = (ushort)v; break;
                                case 4: *(uint*)pOut = (uint)v; break;
                                default: *(ulong*)pOut = v; break;
                            }
                        }
                    }
                }
                return ret;
            }
            catch
            {
                // The output never leaves on failure: free it rather than leave unmanaged memory to the finalizer.
                ret.Dispose();
                throw;
            }
            finally
            {
                // The quirk path's first-positions copies and output view are this method's own.
                if (!ReferenceEquals(loIt, loW))
                    loIt?.Dispose();
                if (!ReferenceEquals(hiIt, hiW))
                    hiIt?.Dispose();
                if (!ReferenceEquals(outIt, ret))
                    outIt?.Dispose();
            }
        }

        /// <summary>
        ///     One bounded value in <c>[off, off + rng]</c> at a draw width — NumPy's
        ///     <c>random_buffered_bounded_{bool,uint8,uint16,uint32}</c> and <c>random_bounded_uint64</c>.
        /// </summary>
        /// <param name="bg">The bit generator (its lock held by the caller).</param>
        /// <param name="width">1 (bool), 8, 16, 32 or 64.</param>
        /// <param name="off">The low bound's bits at the draw width.</param>
        /// <param name="rng">The closed-interval range at the draw width.</param>
        /// <param name="useMasked">Masked rejection (legacy) instead of Lemire.</param>
        /// <param name="buf">The sub-word buffer shared by consecutive positions (bool / 8 / 16-bit).</param>
        /// <param name="bcnt">The number of sub-words still buffered.</param>
        /// <returns>The value, in the low <paramref name="width"/> bits.</returns>
        /// <remarks>
        ///     The width dispatch is the stream contract, not a speed switch: the same range consumes 1, 8, 16, 32 or 64
        ///     bits per draw depending on the dtype. A zero range returns the low bound without touching the stream at
        ///     every width.
        /// </remarks>
        private static ulong DrawOne(BitGenerator bg, int width, ulong off, ulong rng, bool useMasked, ref uint buf, ref int bcnt)
        {
            switch (width)
            {
                case 1:
                    // buffered_bounded_bool: one bit of a buffered 32-bit word.
                    if (rng == 0)
                        return off;
                    if (bcnt == 0)
                    {
                        buf = bg.NextUInt32();
                        bcnt = 31;
                    }
                    else
                    {
                        buf >>= 1;
                        bcnt -= 1;
                    }
                    return buf & 1u;
                case 8:
                    if (rng == 0)
                        return off;
                    if (rng == 0xFF)
                        return (byte)(off + BufferedUInt8(bg, ref buf, ref bcnt));   // Lemire8 lacks the full range
                    if (useMasked)
                    {
                        byte mask8 = (byte)GenMask(rng);
                        byte val8;
                        while ((val8 = (byte)(BufferedUInt8(bg, ref buf, ref bcnt) & mask8)) > rng) { }
                        return (byte)(off + val8);
                    }
                    return (byte)(off + LemireUInt8(bg, (byte)rng, ref buf, ref bcnt));
                case 16:
                    if (rng == 0)
                        return off;
                    if (rng == 0xFFFF)
                        return (ushort)(off + BufferedUInt16(bg, ref buf, ref bcnt));
                    if (useMasked)
                    {
                        ushort mask16 = (ushort)GenMask(rng);
                        ushort val16;
                        while ((val16 = (ushort)(BufferedUInt16(bg, ref buf, ref bcnt) & mask16)) > rng) { }
                        return (ushort)(off + val16);
                    }
                    return (ushort)(off + LemireUInt16(bg, (ushort)rng, ref buf, ref bcnt));
                case 32:
                    // random_buffered_bounded_uint32 ignores the buffer (every draw is a whole word).
                    if (rng == 0)
                        return off;
                    if (rng == 0xFFFFFFFFUL)
                        return (uint)(off + bg.NextUInt32());
                    return (uint)(off + (useMasked ? MaskedUInt32(bg, (uint)rng, (uint)GenMask(rng)) : LemireUInt32(bg, (uint)rng)));
                default:
                    // Only the masked sampler reads the mask (NumPy's Lemire callers pass 0), so Lemire skips the smear.
                    return RandomBoundedUInt64(bg, off, rng, useMasked ? GenMask(rng) : 0UL, useMasked);
            }
        }

        /// <summary>
        ///     A Python int as the range checks see it: exact within <c>±2**100</c>, clamped beyond — past every dtype's
        ///     range, where each check (against a bound within <c>±2**64</c>, or against the other bound) is decided the same
        ///     way for the clamped value as for the exact one.
        /// </summary>
        /// <param name="v">The integer.</param>
        /// <returns>The value for <see cref="Draw"/>.</returns>
        internal static Int128 ClampToInt128(BigInteger v)
        {
            BigInteger limit = BigInteger.One << 100;
            if (v >= limit)
                return (Int128)1 << 100;
            if (v <= -limit)
                return -((Int128)1 << 100);
            return (Int128)v;
        }

        /// <summary>The next larger type <c>_rand_&lt;dtype&gt;_broadcast</c> checks and reads the bounds in.</summary>
        /// <param name="tc">The 8/16/32-bit output dtype or bool.</param>
        /// <returns>uint8 for bool, 16-bit for 8-bit, 32-bit for 16-bit, 64-bit for 32-bit (same signedness).</returns>
        private static NPTypeCode UpType(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => NPTypeCode.Byte,
            NPTypeCode.Byte => NPTypeCode.UInt16,
            NPTypeCode.SByte => NPTypeCode.Int16,
            NPTypeCode.UInt16 => NPTypeCode.UInt32,
            NPTypeCode.Int16 => NPTypeCode.Int32,
            NPTypeCode.UInt32 => NPTypeCode.UInt64,
            _ => NPTypeCode.Int64,
        };

        /// <summary>A 0-d array holding an in-range value of a dtype — NumPy's <c>np.&lt;dtype&gt;(value)</c> strong scalar.</summary>
        /// <param name="value">The value (representable in <paramref name="tc"/>).</param>
        /// <param name="tc">The scalar's dtype.</param>
        /// <returns>The 0-d array; the caller disposes it.</returns>
        private static NDArray ScalarOf(Int128 value, NPTypeCode tc)
        {
            // Every bound fits a 64-bit carrier; the 0-d array then takes the dtype through NumPy's own cast.
            using var carrier = value < 0 ? NDArray.Scalar((long)value) : NDArray.Scalar((ulong)value);
            return carrier.astype(tc);
        }

        /// <summary>
        ///     A bound FORCE-cast to <paramref name="up"/> (NumPy's <c>PyArray_FROM_OTF(x, up, FORCECAST)</c>) and widened
        ///     exactly to its 8-byte carrier: int64 for a signed type, uint64 for an unsigned one.
        /// </summary>
        /// <param name="x">The bound.</param>
        /// <param name="up">The next larger type.</param>
        /// <returns>A new array the caller disposes.</returns>
        private static NDArray WidenToCarrier(NDArray x, NPTypeCode up)
        {
            NPTypeCode carrier = up is NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64
                ? NPTypeCode.UInt64
                : NPTypeCode.Int64;
            using var cast = x.astype(up);
            return cast.astype(carrier);
        }

        /// <summary>
        ///     The 64-bit path's element-by-element conversion of a bound whose dtype does not cast safely — NumPy's
        ///     <c>arr = np.empty_like(orig, dtype=otype); for i: arr_data[i] = int(orig.flat[i])</c> with a range check.
        /// </summary>
        /// <param name="x">The bound.</param>
        /// <param name="tc">Int64 or UInt64.</param>
        /// <param name="check">The range check of one converted value: an error message, or null when it is in range.</param>
        /// <param name="subtract">Subtracted from each stored value (the high bound's <c>is_open</c>; 0 for the low one).</param>
        /// <returns>A new C-contiguous array (logical values in NumPy's scrambled order, see the remarks) the caller disposes.</returns>
        /// <exception cref="ValueError">A value fails <paramref name="check"/>, or is NaN.</exception>
        /// <exception cref="OverflowException">A value is infinite.</exception>
        /// <exception cref="TypeError">A value <c>int()</c> refuses.</exception>
        /// <remarks>
        ///     The elements are read in C order (<c>flat</c>) but written to MEMORY index <c>i</c> of an <c>empty_like</c>
        ///     that keeps the source's layout (NumPy's K order: C for C-contiguous or 1-D sources, F for F-contiguous ones,
        ///     else the source's strides sorted by magnitude). For every non-C layout the logical array therefore holds the
        ///     values in a different arrangement than the source; the result reproduces that arrangement as a C-contiguous
        ///     array (logical element <c>m</c> = the value written at memory position <c>mem(m)</c>), so the draws that follow
        ///     see exactly the bounds NumPy's do. The checks run in C order, so the first offending value is NumPy's.
        /// </remarks>
        private static unsafe NDArray ConvertElementwise(NDArray x, NPTypeCode tc, Func<Int128, string> check, Int128 subtract = default)
        {
            long n = x.size;
            var values = new ulong[n];
            for (long i = 0; i < n; i++)
            {
                Int128 v = PyIntOfElement(x.GetAtIndex(i));
                string err = check(v);
                if (err is not null)
                    throw new ValueError(err);
                values[i] = unchecked((ulong)(long)(v - subtract));
            }

            var ret = new NDArray(DType.From(tc), new Shape((long[])x.shape.Clone()), false);
            var dst = (ulong*)ret.Address;
            long[] mem = KeepOrderMemoryIndex(x.Shape);
            if (mem is null)
            {
                for (long i = 0; i < n; i++)
                    dst[i] = values[i];
            }
            else
            {
                for (long c = 0; c < n; c++)
                    dst[c] = values[mem[c]];
            }
            return ret;
        }

        /// <summary>
        ///     For each C-order position of <paramref name="src"/>, its memory index in NumPy's
        ///     <c>np.empty_like(src)</c> (order K) layout — or null when that layout is C order (no rearrangement).
        /// </summary>
        /// <param name="src">The prototype's shape.</param>
        /// <returns>The mapping, or null.</returns>
        /// <remarks>
        ///     NumPy's <c>PyArray_NewLikeArrayWithShape</c>: a C-contiguous or at-most-1-D prototype gives C order, an
        ///     F-contiguous one F order, anything else the prototype's axes sorted by DESCENDING absolute stride (ties by
        ///     axis, <c>PyArray_CreateSortedStridePerm</c>) with contiguous strides assigned in that order — a broadcast
        ///     axis (stride 0) becoming the innermost.
        /// </remarks>
        private static long[] KeepOrderMemoryIndex(Shape src)
        {
            int nd = src.NDim;
            if (nd <= 1 || src.IsContiguous || src.size <= 1)
                return null;
            long[] dims = src.dimensions;
            int[] perm = new int[nd];
            for (int k = 0; k < nd; k++)
                perm[k] = k;
            if (src.IsFContiguous)
                Array.Reverse(perm);
            else
            {
                long[] st = src.strides;
                // Stable: equal magnitudes keep axis order.
                Array.Sort(perm, (a, b) =>
                {
                    long sa = Math.Abs(st[a]), sb = Math.Abs(st[b]);
                    return sa != sb ? sb.CompareTo(sa) : a.CompareTo(b);
                });
            }
            var kstride = new long[nd];
            long s = 1;
            for (int k = nd - 1; k >= 0; k--)
            {
                kstride[perm[k]] = s;
                s *= dims[perm[k]];
            }

            long n = src.size;
            var mem = new long[n];
            var idx = new long[nd];
            for (long c = 0; c < n; c++)
            {
                long p = 0;
                for (int k = 0; k < nd; k++)
                    p += idx[k] * kstride[k];
                mem[c] = p;
                // C-order odometer.
                for (int k = nd - 1; k >= 0; k--)
                {
                    if (++idx[k] < dims[k])
                        break;
                    idx[k] = 0;
                }
            }
            return mem;
        }

        /// <summary>
        ///     Python's <c>int()</c> of a 0-d bound (NumPy's scalar path: <c>int(np.asarray(low))</c>): an integer as is, a
        ///     bool as 0/1, a float truncated toward zero.
        /// </summary>
        /// <param name="x">The 0-d bound, or null for Python's <c>None</c>.</param>
        /// <returns>The value (clamped to <c>±2**100</c> past every dtype's range — see <see cref="PyIntOfFloat"/>).</returns>
        /// <exception cref="TypeError"><paramref name="x"/> is null (<c>… not 'NoneType'</c>) or complex (<c>int()</c> of a
        ///     0-d complex ARRAY refuses it, unlike a complex scalar element).</exception>
        /// <exception cref="ValueError">A NaN.</exception>
        /// <exception cref="OverflowException">An infinity.</exception>
        private static Int128 PyInt(NDArray x)
        {
            if (x is null)
                throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            if (x.typecode == NPTypeCode.Complex)
                throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'complex'");
            return PyIntOfElement(x.GetAtIndex(0));
        }

        /// <summary>
        ///     Python's <c>int()</c> of one element NumPy's <c>flat</c> hands out — a NumPy scalar, whose complex flavor
        ///     converts its real part (with a <c>ComplexWarning</c> NumSharp does not model).
        /// </summary>
        /// <param name="v">The boxed element.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ValueError">A NaN.</exception>
        /// <exception cref="OverflowException">An infinity.</exception>
        private static Int128 PyIntOfElement(object v) => v switch
        {
            bool b => b ? 1 : 0,
            byte u8 => u8,
            sbyte i8 => i8,
            short i16 => i16,
            ushort u16 => u16,
            int i32 => i32,
            uint u32 => u32,
            long i64 => i64,
            ulong u64 => u64,
            char ch => ch,
            Half h => PyIntOfFloat((double)h),
            float f => PyIntOfFloat(f),
            double d => PyIntOfFloat(d),
            decimal m => (Int128)decimal.Truncate(m),
            Complex z => PyIntOfFloat(z.Real),
            _ => throw new TypeError($"int() argument must be a string, a bytes-like object or a real number, not '{v?.GetType().Name ?? "NoneType"}'"),
        };

        /// <summary>Python's <c>int(float)</c>: truncation toward zero, NaN and infinity refused.</summary>
        /// <param name="d">The float.</param>
        /// <returns>The integer, clamped to <c>±2**100</c>.</returns>
        /// <exception cref="ValueError">NaN (<c>cannot convert float NaN to integer</c>).</exception>
        /// <exception cref="OverflowException">Infinity (NumPy's <c>OverflowError: cannot convert float infinity to integer</c>).</exception>
        /// <remarks>
        ///     Python's result is exact at any magnitude; the clamp stands in for it beyond <c>2**100</c>, where every
        ///     comparison the checks make (against a dtype bound within <c>±2**64</c>, or against the other bound) is
        ///     decided by the first check that sees it, identically for the true value and the clamped one.
        /// </remarks>
        private static Int128 PyIntOfFloat(double d)
        {
            if (double.IsNaN(d))
                throw new ValueError("cannot convert float NaN to integer");
            if (double.IsInfinity(d))
                throw new OverflowException("cannot convert float infinity to integer");
            const double Limit = 1267650600228229401496703205376.0;   // 2**100
            if (d >= Limit)
                return (Int128)1 << 100;
            if (d <= -Limit)
                return -((Int128)1 << 100);
            return (Int128)Math.Truncate(d);
        }

        /// <summary>
        ///     NumPy's <c>format_bounds_error</c> over an array low bound: the <c>high</c> wording when EVERY low is 0 (the
        ///     one-argument form's), else the <c>low</c> wording.
        /// </summary>
        /// <param name="closed">Whether the interval is closed.</param>
        /// <param name="low">The low bounds.</param>
        /// <returns>The message.</returns>
        private static string FormatBoundsError(bool closed, NDArray low)
            => FormatBoundsError(closed, (Int128)(np.any(low) ? 1 : 0));

        /// <summary>The broadcast of two or three shapes with NumPy's multi-iterator mismatch text, as fresh dimensions.</summary>
        /// <param name="shapes">The shapes, in NumPy's argument order.</param>
        /// <returns>The broadcast dimensions.</returns>
        /// <exception cref="ValueError">The shapes do not broadcast.</exception>
        private static long[] BroadcastDims(params Shape[] shapes)
        {
            try
            {
                Shape s = np.broadcast_shapes(shapes);
                return (long[])(s.dimensions ?? Array.Empty<long>()).Clone();
            }
            catch (IncorrectShapeException e)
            {
                throw new ValueError(e.Message);
            }
        }

        /// <summary>Whether two dimension vectors are equal.</summary>
        /// <param name="a">The first.</param>
        /// <param name="b">The second.</param>
        /// <returns>True when equal.</returns>
        private static bool SameDimensions(long[] a, long[] b)
        {
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     The first <paramref name="n"/> elements, in C order, of <paramref name="x"/> broadcast to
        ///     <paramref name="dims"/> — the bounds NumPy's loop reads when the output is smaller than their broadcast.
        /// </summary>
        /// <param name="x">The bound carrier.</param>
        /// <param name="dims">The (larger) broadcast shape.</param>
        /// <param name="n">The output size.</param>
        /// <returns>A new 1-D array of <paramref name="n"/> elements the caller disposes.</returns>
        private static NDArray FirstPositions(NDArray x, long[] dims, long n)
        {
            using var b = np.broadcast_to(x, new Shape(dims));
            using var flat = b.flatten();
            using var head = flat[$":{n}"];
            return head.copy();
        }
    }
}
