using System;
using NumSharp.Generic;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Return random integers from <paramref name="low"/> (inclusive) to <paramref name="high"/>
        ///     (exclusive, or inclusive when <paramref name="endpoint"/> is true).
        /// </summary>
        /// <param name="low">Lowest integer drawn (or, when <paramref name="high"/> is null, one above the highest with low = 0).</param>
        /// <param name="high">If provided, one above the largest integer drawn (or the largest when <paramref name="endpoint"/>).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value; <c>()</c> returns a 0-d array.</param>
        /// <param name="dtype">Desired native integer or bool dtype. Default is int64.</param>
        /// <param name="endpoint">If true, sample from the closed interval <c>[low, high]</c>.</param>
        /// <returns>The draws, of <paramref name="dtype"/>.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a native non-integer dtype (<c>Unsupported dtype dtype('float64') for integers</c>).</exception>
        /// <exception cref="ValueError">
        ///     <paramref name="dtype"/> has a non-native byte order; or the bounds fall outside the dtype
        ///     (<c>low/high is out of bounds for &lt;dtype&gt;</c>); or the interval is empty (<c>low &gt;= high</c>,
        ///     <c>high &lt;= 0</c> and their closed-interval forms).
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
        ///     <br/>
        ///     Uses Lemire's method (NumPy's Generator default, <c>use_masked=False</c>) — NOT the
        ///     legacy masked rejection of <c>RandomState.randint</c> — so the stream is byte-identical
        ///     to <c>default_rng(seed).integers(...)</c>. A zero-size request returns an empty array before
        ///     the bounds are checked and draws nothing, exactly as NumPy's <c>_rand_*</c> does.
        /// </remarks>
        public NDArray integers(long low, long? high = null, Shape size = default, DType dtype = null, bool endpoint = false)
            => IntegersCore(low, high.HasValue ? (Int128)high.Value : (Int128?)null, size, dtype, endpoint);

        /// <summary>
        ///     Unsigned overload of <see cref="integers(long, long?, Shape, DType, bool)"/> — the C# spelling of a
        ///     bound above <see cref="long.MaxValue"/>, which NumPy expresses with arbitrary-precision Python ints:
        ///     the upper half of the <c>uint64</c> range (full range: <c>integers(0UL, ulong.MaxValue,
        ///     dtype: np.uint64, endpoint: true)</c>) and the exclusive int64 high <c>2**63</c>
        ///     (<c>integers(0UL, 9223372036854775808UL, dtype: np.int64)</c>).
        /// </summary>
        /// <param name="low">Lowest integer drawn (or, when <paramref name="high"/> is null, one above the highest with low = 0).</param>
        /// <param name="high">If provided, one above the largest integer drawn (or the largest when <paramref name="endpoint"/>).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value; <c>()</c> returns a 0-d array.</param>
        /// <param name="dtype">Desired native integer or bool dtype. Default is int64.</param>
        /// <param name="endpoint">If true, sample from the closed interval <c>[low, high]</c>.</param>
        /// <returns>The draws, of <paramref name="dtype"/>.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a native non-integer dtype.</exception>
        /// <exception cref="ValueError">Non-native byte order, bounds outside the dtype, or an empty interval — as for the signed overload.</exception>
        /// <remarks>
        ///     Both overloads share one arbitrary-precision (<see cref="Int128"/>) range check, so every value
        ///     expressible in either spelling is validated exactly as NumPy validates the Python int.
        /// </remarks>
        public NDArray integers(ulong low, ulong? high = null, Shape size = default, DType dtype = null, bool endpoint = false)
            => IntegersCore(low, high.HasValue ? (Int128)high.Value : (Int128?)null, size, dtype, endpoint);

        /// <summary>
        ///     The shared body of both <c>integers</c> overloads: NumPy's <c>Generator.integers</c> +
        ///     <c>_bounded_integers._rand_&lt;dtype&gt;</c> scalar path, in NumPy's validation order.
        /// </summary>
        /// <param name="low">The first bound as received.</param>
        /// <param name="high">The second bound as received (null = NumPy's <c>high=None</c>).</param>
        /// <param name="size">The output shape (default = scalar request).</param>
        /// <param name="dtype">The requested dtype (null = int64).</param>
        /// <param name="endpoint">Whether <paramref name="high"/> is inclusive.</param>
        /// <returns>The draws.</returns>
        /// <exception cref="TypeError">Unsupported native dtype.</exception>
        /// <exception cref="ValueError">Non-native dtype, out-of-bounds low/high, or an empty interval.</exception>
        /// <remarks>
        ///     The range arithmetic is done in <see cref="Int128"/> — NumPy works on unbounded Python ints, so
        ///     <c>high - 1</c> can never wrap and a bound one past a dtype's range is REPORTED rather than
        ///     silently masked. Each dtype's upper bound is its INCLUSIVE maximum (NumPy's template <c>ub</c>:
        ///     <c>0x1</c> for bool, <c>0xFF</c> for uint8, <c>0x7FFFFFFF</c> for int32, …), compared against the
        ///     already-inclusive high; comparing against an exclusive limit accepted <c>high = max + 2</c>, masked
        ///     the range to zero and returned a constant array.
        /// </remarks>
        private NDArray IntegersCore(Int128 low, Int128? high, Shape size, DType dtype, bool endpoint)
        {
            Int128 lo, hi;
            if (high is null)
            {
                hi = low;
                lo = 0;
            }
            else
            {
                lo = low;
                hi = high.Value;
            }

            dtype ??= DType.Int64;
            ResolveIntegerDtype(dtype, out NPTypeCode tc, out int width, out Int128 lb, out Int128 ub);

            // NumPy _rand_*: `if size is not None and np.prod(size) == 0: return np.empty(size, dtype)` runs
            // BEFORE the bounds are looked at, and consumes nothing from the stream.
            if (!IsNoSize(size) && size.size == 0)
                return new NDArray(dtype, size);

            // The internal generator produces on the closed interval [low, high]; subtract 1 for the
            // half-open case. Int128 keeps this exact for both overloads (no wrap at 0 or at 2**63).
            if (!endpoint)
                hi -= 1;

            string name = tc.AsNumpyDtypeName();
            if (lo < lb)
                throw new ValueError($"low is out of bounds for {name}");
            if (hi > ub)
                throw new ValueError($"high is out of bounds for {name}");
            if (lo > hi)
                throw new ValueError(FormatBoundsError(endpoint, lo));

            // rng = high - low fits the dtype's unsigned width once the bounds hold; off is the low bound's
            // two's-complement bit pattern at that width (NumPy's `<utype>(<nptype>low)` cast).
            ulong widthMask = width >= 64 ? ulong.MaxValue : ((1UL << width) - 1UL);
            ulong rng = unchecked((ulong)(hi - lo)) & widthMask;
            ulong off = unchecked((ulong)(lo & ulong.MaxValue)) & widthMask;

            lock (_bitGenerator.@lock)
                return FillIntegers(dtype, tc, size, width, off, rng);
        }

        /// <summary>
        ///     Maps a requested dtype onto NumPy's nine <c>integers</c> loops (the <c>_dtype == np.int32 …</c>
        ///     dispatch), with each loop's draw width and INCLUSIVE value range.
        /// </summary>
        /// <param name="dtype">The requested dtype.</param>
        /// <param name="tc">The resolved type code.</param>
        /// <param name="width">The bounded-draw width: 1 (bool), 8, 16, 32 or 64 — it decides how the stream is consumed.</param>
        /// <param name="lb">The inclusive minimum of the dtype.</param>
        /// <param name="ub">The inclusive maximum of the dtype.</param>
        /// <exception cref="ValueError">The dtype's byte order is not native (NumPy checks this after the nine equality tests fail, so it wins over "unsupported").</exception>
        /// <exception cref="TypeError">The dtype is a native non-integer type.</exception>
        private static void ResolveIntegerDtype(DType dtype, out NPTypeCode tc, out int width, out Int128 lb, out Int128 ub)
        {
            tc = dtype.GetTypeCode();
            if (dtype.isnative)
            {
                switch (tc)
                {
                    case NPTypeCode.Boolean: width = 1; lb = 0; ub = 1; return;
                    case NPTypeCode.Byte: width = 8; lb = 0; ub = byte.MaxValue; return;
                    case NPTypeCode.SByte: width = 8; lb = sbyte.MinValue; ub = sbyte.MaxValue; return;
                    case NPTypeCode.UInt16: width = 16; lb = 0; ub = ushort.MaxValue; return;
                    case NPTypeCode.Int16: width = 16; lb = short.MinValue; ub = short.MaxValue; return;
                    case NPTypeCode.UInt32: width = 32; lb = 0; ub = uint.MaxValue; return;
                    case NPTypeCode.Int32: width = 32; lb = int.MinValue; ub = int.MaxValue; return;
                    case NPTypeCode.UInt64: width = 64; lb = 0; ub = ulong.MaxValue; return;
                    case NPTypeCode.Int64: width = 64; lb = long.MinValue; ub = long.MaxValue; return;
                }
            }
            else
            {
                throw new ValueError("Providing a dtype with a non-native byteorder is not supported. If you require platform-independent byteorder, call byteswap when required.");
            }

            throw new TypeError($"Unsupported dtype {dtype.ToString(true)} for integers");
        }

        /// <summary>
        ///     Builds the result: a scalar request returns a 0-d array of the dtype, anything else a filled array.
        /// </summary>
        /// <param name="dtype">The result dtype.</param>
        /// <param name="tc">The result type code.</param>
        /// <param name="size">The output shape (default = scalar request).</param>
        /// <param name="width">The bounded-draw width (1/8/16/32/64).</param>
        /// <param name="off">The low bound, as the unsigned bit pattern of the dtype.</param>
        /// <param name="rng">The closed-interval range <c>high - low</c>.</param>
        /// <returns>The draws.</returns>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        private NDArray FillIntegers(DType dtype, NPTypeCode tc, Shape size, int width, ulong off, ulong rng)
        {
            // size == 0 -> empty array of the requested dtype (draws no state). IntegersCore already
            // returned for this case; kept so the helper is self-contained.
            if (!IsNoSize(size) && size.size == 0)
                return new NDArray(dtype, size);

            if (IsNoSize(size))
            {
                var scalar = new NDArray(dtype, Shape.Vector(1));
                FillBounded(scalar, 1, width, off, rng);
                var value = scalar.GetAtIndex(0);
                return NDArray.Scalar(value, tc);
            }

            var nd = new NDArray(dtype, size);
            FillBounded(nd, nd.size, width, off, rng);
            return nd;
        }

        /// <summary>
        ///     Return random bytes.
        /// </summary>
        /// <param name="length">Number of random bytes.</param>
        /// <returns>
        ///     A 1-D <see cref="NDArray{T}"/> of <see cref="byte"/> (dtype <c>uint8</c>), length
        ///     <paramref name="length"/> — the NumSharp analogue of NumPy's <c>bytes</c> object.
        /// </returns>
        /// <exception cref="ValueError"><paramref name="length"/> is below the smallest length NumPy accepts (<c>negative dimensions are not allowed</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.bytes.html
        ///     <br/>
        ///     Byte-identical to NumPy: draws <c>ceil(length/4)</c> uint32 words from PCG64 (via the
        ///     32-bit buffered path), packs them little-endian, and truncates to <paramref name="length"/>.
        ///     As with <see cref="NumPyRandom.bytes"/>, the result is an unmanaged-backed
        ///     <see cref="NDArray{T}"/>, so — matching NumPy's 64-bit <c>npy_intp</c> length — it is
        ///     NOT capped at <see cref="Array.MaxLength"/> and a request over 2 GiB still succeeds.
        /// </remarks>
        public NDArray<byte> bytes(long length)
        {
            lock (_bitGenerator.@lock)
                return NumPyRandom.BytesCore(length, static bg => bg.NextUInt32(), _bitGenerator);
        }

        /// <summary>
        ///     NumPy's <c>format_bounds_error</c>: the special wording for the default single-argument
        ///     (<c>low == 0</c>) case, else the low-versus-high wording.
        /// </summary>
        /// <param name="closed">Whether the interval is closed (<c>endpoint=True</c>).</param>
        /// <param name="low">The low bound.</param>
        /// <returns>The message text.</returns>
        private static string FormatBoundsError(bool closed, Int128 low)
        {
            if (low == 0)
                return closed ? "high < 0" : "high <= 0";
            return closed ? "low > high" : "low >= high";
        }

        // ---- the per-width bounded fills (numpy random_bounded_uintX_fill, use_masked=False) ----

        /// <summary>Dispatches a bounded fill to the per-width loop that consumes the stream the way NumPy's does.</summary>
        /// <param name="nd">The freshly allocated C-contiguous destination.</param>
        /// <param name="cnt">The number of values to draw.</param>
        /// <param name="width">1 (bool), 8, 16, 32 or 64.</param>
        /// <param name="off">The low bound's unsigned bit pattern.</param>
        /// <param name="rng">The closed-interval range.</param>
        /// <remarks>
        ///     The width dispatch is REQUIRED for parity, not a performance switch: the 8/16-bit and bool loops
        ///     split one 32-bit word into several values, so the same range drawn at another width consumes the
        ///     stream differently.
        /// </remarks>
        private unsafe void FillBounded(NDArray nd, long cnt, int width, ulong off, ulong rng)
        {
            void* addr = (void*)nd.Address;
            switch (width)
            {
                case 1: FillBoundedBool((byte*)addr, cnt, (byte)off, (byte)rng); break;
                case 8: FillBoundedUInt8((byte*)addr, cnt, (byte)off, (byte)rng); break;
                case 16: FillBoundedUInt16((ushort*)addr, cnt, (ushort)off, (ushort)rng); break;
                case 32: FillBoundedUInt32((uint*)addr, cnt, (uint)off, (uint)rng); break;
                default: FillBoundedUInt64((ulong*)addr, cnt, off, rng); break;
            }
        }

        /// <summary>NumPy's <c>random_bounded_uint64_fill</c> (Lemire): 32-bit draws for ranges that fit, 64-bit otherwise.</summary>
        /// <param name="outp">Destination.</param>
        /// <param name="cnt">Value count.</param>
        /// <param name="off">Low bound.</param>
        /// <param name="rng">Closed-interval range.</param>
        private unsafe void FillBoundedUInt64(ulong* outp, long cnt, ulong off, ulong rng)
        {
            if (rng == 0)
            {
                for (long i = 0; i < cnt; i++) outp[i] = off;
            }
            else if (rng <= 0xFFFFFFFFUL)
            {
                if (rng == 0xFFFFFFFFUL)
                    for (long i = 0; i < cnt; i++) outp[i] = off + _bitGenerator.NextUInt32();
                else
                {
                    uint r = (uint)rng;
                    for (long i = 0; i < cnt; i++) outp[i] = off + LemireUint32(r);
                }
            }
            else if (rng == 0xFFFFFFFFFFFFFFFFUL)
            {
                for (long i = 0; i < cnt; i++) outp[i] = off + _bitGenerator.NextUInt64();
            }
            else
            {
                for (long i = 0; i < cnt; i++) outp[i] = off + LemireUint64(rng);
            }
        }

        /// <summary>NumPy's <c>random_bounded_uint32_fill</c> (Lemire).</summary>
        /// <param name="outp">Destination.</param>
        /// <param name="cnt">Value count.</param>
        /// <param name="off">Low bound.</param>
        /// <param name="rng">Closed-interval range.</param>
        private unsafe void FillBoundedUInt32(uint* outp, long cnt, uint off, uint rng)
        {
            if (rng == 0)
                for (long i = 0; i < cnt; i++) outp[i] = off;
            else if (rng == 0xFFFFFFFFu)
                for (long i = 0; i < cnt; i++) outp[i] = off + _bitGenerator.NextUInt32();
            else
                for (long i = 0; i < cnt; i++) outp[i] = off + LemireUint32(rng);
        }

        /// <summary>NumPy's <c>random_bounded_uint16_fill</c> (Lemire over a 32-bit word split in two halves).</summary>
        /// <param name="outp">Destination.</param>
        /// <param name="cnt">Value count.</param>
        /// <param name="off">Low bound.</param>
        /// <param name="rng">Closed-interval range.</param>
        private unsafe void FillBoundedUInt16(ushort* outp, long cnt, ushort off, ushort rng)
        {
            uint buf = 0;
            int bcnt = 0;
            if (rng == 0)
                for (long i = 0; i < cnt; i++) outp[i] = off;
            else if (rng == 0xFFFF)
                for (long i = 0; i < cnt; i++) outp[i] = (ushort)(off + BufferedUint16(ref buf, ref bcnt));
            else
                for (long i = 0; i < cnt; i++) outp[i] = (ushort)(off + LemireUint16(rng, ref buf, ref bcnt));
        }

        /// <summary>NumPy's <c>random_bounded_uint8_fill</c> (Lemire over a 32-bit word split in four bytes).</summary>
        /// <param name="outp">Destination.</param>
        /// <param name="cnt">Value count.</param>
        /// <param name="off">Low bound.</param>
        /// <param name="rng">Closed-interval range.</param>
        private unsafe void FillBoundedUInt8(byte* outp, long cnt, byte off, byte rng)
        {
            uint buf = 0;
            int bcnt = 0;
            if (rng == 0)
                for (long i = 0; i < cnt; i++) outp[i] = off;
            else if (rng == 0xFF)
                for (long i = 0; i < cnt; i++) outp[i] = (byte)(off + BufferedUint8(ref buf, ref bcnt));
            else
                for (long i = 0; i < cnt; i++) outp[i] = (byte)(off + LemireUint8(rng, ref buf, ref bcnt));
        }

        /// <summary>NumPy's <c>random_bounded_bool_fill</c>: one bit per value from a buffered 32-bit word (the low bound is ignored once the range is 1).</summary>
        /// <param name="outp">Destination.</param>
        /// <param name="cnt">Value count.</param>
        /// <param name="off">Low bound (the value when the range is 0).</param>
        /// <param name="rng">Closed-interval range (0 or 1).</param>
        private unsafe void FillBoundedBool(byte* outp, long cnt, byte off, byte rng)
        {
            uint buf = 0;
            int bcnt = 0;
            for (long i = 0; i < cnt; i++)
            {
                if (rng == 0) { outp[i] = off; continue; }
                if (bcnt == 0) { buf = _bitGenerator.NextUInt32(); bcnt = 31; }
                else { buf >>= 1; bcnt -= 1; }
                outp[i] = (byte)((buf & 0x1u) != 0 ? 1 : 0);
            }
        }

        // ---- 32-bit buffer splitters (numpy buffered_uint16 / buffered_uint8) ----

        /// <summary>NumPy's <c>buffered_uint16</c>: serves the two 16-bit halves of one 32-bit word, low half first.</summary>
        /// <param name="buf">The buffered word (state across calls within one fill).</param>
        /// <param name="bcnt">The number of halves still buffered.</param>
        /// <returns>The next 16-bit value.</returns>
        private ushort BufferedUint16(ref uint buf, ref int bcnt)
        {
            if (bcnt == 0) { buf = _bitGenerator.NextUInt32(); bcnt = 1; }
            else { buf >>= 16; bcnt -= 1; }
            return (ushort)buf;
        }

        /// <summary>NumPy's <c>buffered_uint8</c>: serves the four bytes of one 32-bit word, low byte first.</summary>
        /// <param name="buf">The buffered word (state across calls within one fill).</param>
        /// <param name="bcnt">The number of bytes still buffered.</param>
        /// <returns>The next 8-bit value.</returns>
        private byte BufferedUint8(ref uint buf, ref int bcnt)
        {
            if (bcnt == 0) { buf = _bitGenerator.NextUInt32(); bcnt = 3; }
            else { buf >>= 8; bcnt -= 1; }
            return (byte)buf;
        }

        // ---- Lemire bounded generators (numpy bounded_lemire_uintX) ----

        /// <summary>NumPy's <c>bounded_lemire_uint64</c>: unbiased draw in <c>[0, rng]</c> by 128-bit multiply-and-reject.</summary>
        /// <param name="rng">Closed-interval range (below 2**64-1).</param>
        /// <returns>The draw.</returns>
        private ulong LemireUint64(ulong rng)
        {
            ulong rngExcl = rng + 1;
            UInt128 m = (UInt128)_bitGenerator.NextUInt64() * rngExcl;
            ulong leftover = (ulong)m;
            if (leftover < rngExcl)
            {
                ulong threshold = (ulong.MaxValue - rng) % rngExcl;
                while (leftover < threshold)
                {
                    m = (UInt128)_bitGenerator.NextUInt64() * rngExcl;
                    leftover = (ulong)m;
                }
            }
            return (ulong)(m >> 64);
        }

        /// <summary>NumPy's <c>bounded_lemire_uint32</c>: unbiased draw in <c>[0, rng]</c> by 64-bit multiply-and-reject.</summary>
        /// <param name="rng">Closed-interval range (below 2**32-1).</param>
        /// <returns>The draw.</returns>
        private uint LemireUint32(uint rng)
        {
            uint rngExcl = rng + 1;
            ulong m = (ulong)_bitGenerator.NextUInt32() * rngExcl;
            uint leftover = (uint)m;
            if (leftover < rngExcl)
            {
                uint threshold = (uint.MaxValue - rng) % rngExcl;
                while (leftover < threshold)
                {
                    m = (ulong)_bitGenerator.NextUInt32() * rngExcl;
                    leftover = (uint)m;
                }
            }
            return (uint)(m >> 32);
        }

        /// <summary>NumPy's <c>buffered_bounded_lemire_uint16</c>: Lemire over buffered 16-bit halves.</summary>
        /// <param name="rng">Closed-interval range (below 0xFFFF).</param>
        /// <param name="buf">The buffered word.</param>
        /// <param name="bcnt">The number of halves still buffered.</param>
        /// <returns>The draw.</returns>
        private ushort LemireUint16(ushort rng, ref uint buf, ref int bcnt)
        {
            ushort rngExcl = (ushort)(rng + 1);
            uint m = (uint)BufferedUint16(ref buf, ref bcnt) * rngExcl;
            ushort leftover = (ushort)m;
            if (leftover < rngExcl)
            {
                ushort threshold = (ushort)((ushort)(ushort.MaxValue - rng) % rngExcl);
                while (leftover < threshold)
                {
                    m = (uint)BufferedUint16(ref buf, ref bcnt) * rngExcl;
                    leftover = (ushort)m;
                }
            }
            return (ushort)(m >> 16);
        }

        /// <summary>NumPy's <c>buffered_bounded_lemire_uint8</c>: Lemire over buffered bytes.</summary>
        /// <param name="rng">Closed-interval range (below 0xFF).</param>
        /// <param name="buf">The buffered word.</param>
        /// <param name="bcnt">The number of bytes still buffered.</param>
        /// <returns>The draw.</returns>
        private byte LemireUint8(byte rng, ref uint buf, ref int bcnt)
        {
            byte rngExcl = (byte)(rng + 1);
            uint m = (uint)BufferedUint8(ref buf, ref bcnt) * rngExcl;
            byte leftover = (byte)m;
            if (leftover < rngExcl)
            {
                byte threshold = (byte)((byte)(byte.MaxValue - rng) % rngExcl);
                while (leftover < threshold)
                {
                    m = (uint)BufferedUint8(ref buf, ref bcnt) * rngExcl;
                    leftover = (byte)m;
                }
            }
            return (byte)(m >> 8);
        }
    }
}
