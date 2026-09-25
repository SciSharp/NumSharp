using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's bounded-integer machinery (<c>_bounded_integers.pyx.in</c> + the bounded helpers of
    ///     <c>src/distributions/distributions.c</c>) over any <see cref="BitGenerator"/>: the range validation
    ///     shared by <c>Generator.integers</c> and <c>RandomState.randint</c>, the per-width fills in both of
    ///     NumPy's strategies (Lemire for the Generator, masked rejection for the legacy RandomState), and the
    ///     scalar <c>random_interval</c> / <c>random_bounded_uint64</c> draws the shuffles and <c>choice</c> use.
    /// </summary>
    /// <remarks>
    ///     Every helper consumes the stream exactly as its C counterpart does — including the sub-word
    ///     buffering (one 32-bit word feeds four uint8 or two uint16 values) and the 32-bit shortcut for ranges
    ///     that fit 32 bits even when the bounds need 64 — so a generator's integer draws stay byte-identical to
    ///     NumPy's for every dtype. Callers hold the bit generator's lock around a fill.
    /// </remarks>
    internal static class BoundedIntegers
    {
        /// <summary>
        ///     Validates a bounded-integer request and draws it: NumPy's <c>Generator.integers</c> /
        ///     <c>RandomState.randint</c> dispatch followed by <c>_rand_&lt;dtype&gt;</c>'s scalar path.
        /// </summary>
        /// <param name="bg">The bit generator; its lock is taken for the fill.</param>
        /// <param name="low">The first bound as received.</param>
        /// <param name="high">The second bound as received (null = NumPy's <c>high=None</c>: draw from <c>[0, low)</c>).</param>
        /// <param name="size">The output shape (default = NumPy's <c>size=None</c>: a single value, returned 0-d).</param>
        /// <param name="dtype">The requested dtype (null = int64 for the Generator; the legacy caller passes its own default).</param>
        /// <param name="endpoint">Whether <paramref name="high"/> is inclusive.</param>
        /// <param name="useMasked">True for the legacy masked-rejection sampler, false for the Generator's Lemire sampler.</param>
        /// <param name="methodName">The public method name used in the unsupported-dtype message (<c>integers</c> / <c>randint</c>).</param>
        /// <param name="legacyByteOrder">True for <c>RandomState.randint</c>, which only WARNS about a non-native byte order and
        /// draws the native dtype; false for <c>Generator.integers</c>, which raises.</param>
        /// <returns>The draws, of the resolved native dtype.</returns>
        /// <exception cref="TypeError">The dtype is not one of the nine integer/bool dtypes.</exception>
        /// <exception cref="ValueError">A non-native dtype (Generator), bounds outside the dtype, or an empty interval.</exception>
        /// <remarks>
        ///     The range arithmetic is <see cref="Int128"/> — NumPy works on unbounded Python ints, so
        ///     <c>high - 1</c> never wraps — and each dtype's upper bound is its INCLUSIVE maximum, compared with
        ///     the already-inclusive high. A zero-size request returns an empty array BEFORE the bounds are looked
        ///     at and draws nothing.
        /// </remarks>
        internal static NDArray Draw(BitGenerator bg, Int128 low, Int128? high, Shape size, DType dtype, bool endpoint,
                                     bool useMasked, string methodName, bool legacyByteOrder)
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

            ResolveDtype(dtype, methodName, legacyByteOrder, out NPTypeCode tc, out int width, out Int128 lb, out Int128 ub);
            DType resultType = DType.From(tc); // the native builtin (a non-native legacy request draws native)

            // NumPy _rand_*: `if size is not None and np.prod(size) == 0: return np.empty(size, dtype)` runs
            // BEFORE the bounds are looked at, and consumes nothing from the stream.
            if (!size.IsEmpty && size.size == 0)
                return new NDArray(resultType, size);

            // The internal generator produces on the closed interval [low, high]; subtract 1 for half-open.
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

            lock (bg.@lock)
            {
                if (size.IsEmpty)
                {
                    // NumPy fills a 1-element buffer (fresh sub-word buffer) and returns it as a scalar.
                    using var one = new NDArray(resultType, Shape.Vector(1));
                    Fill(bg, one, 1, width, off, rng, useMasked);
                    return NDArray.Scalar(one.GetAtIndex(0), tc);
                }

                var nd = new NDArray(resultType, size);
                Fill(bg, nd, nd.size, width, off, rng, useMasked);
                return nd;
            }
        }

        /// <summary>
        ///     Maps a requested dtype onto NumPy's nine bounded-integer loops, with each loop's draw width and
        ///     INCLUSIVE value range.
        /// </summary>
        /// <param name="dtype">The requested dtype.</param>
        /// <param name="methodName">The method named in the unsupported-dtype message.</param>
        /// <param name="legacyByteOrder">Whether a non-native byte order is accepted and drawn native (RandomState) rather than rejected (Generator).</param>
        /// <param name="tc">The resolved type code.</param>
        /// <param name="width">1 (bool), 8, 16, 32 or 64 — decides how the stream is consumed.</param>
        /// <param name="lb">The inclusive minimum.</param>
        /// <param name="ub">The inclusive maximum.</param>
        /// <exception cref="ValueError">A non-native dtype on the Generator path (NumPy tests this after the nine equality checks fail).</exception>
        /// <exception cref="TypeError">A native non-integer dtype.</exception>
        private static void ResolveDtype(DType dtype, string methodName, bool legacyByteOrder,
                                         out NPTypeCode tc, out int width, out Int128 lb, out Int128 ub)
        {
            tc = dtype.GetTypeCode();
            if (!dtype.isnative)
            {
                // RandomState.randint: DeprecationWarning + `_dtype = _dtype.newbyteorder()` — i.e. the native
                // twin is drawn. Generator.integers: every equality test fails, then `not isnative` raises.
                if (!legacyByteOrder)
                    throw new ValueError("Providing a dtype with a non-native byteorder is not supported. If you require platform-independent byteorder, call byteswap when required.");
                dtype = DType.From(tc);
            }

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

            throw new TypeError($"Unsupported dtype {dtype.ToString(true)} for {methodName}");
        }

        /// <summary>
        ///     NumPy's <c>format_bounds_error</c>: the special wording for the default single-argument
        ///     (<c>low == 0</c>) case, else the low-versus-high wording.
        /// </summary>
        /// <param name="closed">Whether the interval is closed (<c>endpoint=True</c>).</param>
        /// <param name="low">The low bound.</param>
        /// <returns>The message text.</returns>
        internal static string FormatBoundsError(bool closed, Int128 low)
        {
            if (low == 0)
                return closed ? "high < 0" : "high <= 0";
            return closed ? "low > high" : "low >= high";
        }

        /// <summary>NumPy's <c>gen_mask</c>: the smallest all-ones bit mask covering <paramref name="max"/>.</summary>
        /// <param name="max">The value to cover.</param>
        /// <returns>The mask.</returns>
        internal static ulong GenMask(ulong max)
        {
            ulong mask = max;
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;
            mask |= mask >> 32;
            return mask;
        }

        /// <summary>
        ///     NumPy's <c>random_interval</c>: a uniform integer in <c>[0, max]</c> by mask-rejection — the sampler
        ///     every Fisher–Yates shuffle (Generator and legacy) uses.
        /// </summary>
        /// <param name="bg">The bit generator (the caller holds its lock).</param>
        /// <param name="max">The inclusive upper bound.</param>
        /// <returns>The draw (0 without consuming the stream when <paramref name="max"/> is 0).</returns>
        /// <remarks>A 32-bit word is drawn whenever <paramref name="max"/> fits 32 bits, a 64-bit word otherwise.</remarks>
        internal static ulong RandomInterval(BitGenerator bg, ulong max)
        {
            if (max == 0)
                return 0;
            ulong mask = GenMask(max);
            ulong value;
            if (max <= 0xffffffffUL)
                while ((value = bg.NextUInt32() & mask) > max) { }
            else
                while ((value = bg.NextUInt64() & mask) > max) { }
            return value;
        }

        /// <summary>
        ///     NumPy's <c>random_bounded_uint64</c>: a single value in <c>[off, off + rng]</c>, with a 32-bit draw
        ///     for ranges that fit 32 bits.
        /// </summary>
        /// <param name="bg">The bit generator (the caller holds its lock).</param>
        /// <param name="off">The low bound.</param>
        /// <param name="rng">The closed-interval range.</param>
        /// <param name="mask">The masked sampler's mask (ignored by Lemire; NumPy's <c>choice</c> passes 0 with Lemire).</param>
        /// <param name="useMasked">Masked rejection (legacy) instead of Lemire.</param>
        /// <returns>The draw.</returns>
        internal static ulong RandomBoundedUInt64(BitGenerator bg, ulong off, ulong rng, ulong mask, bool useMasked)
        {
            if (rng == 0)
                return off;
            if (rng <= 0xFFFFFFFFUL)
            {
                // The 32-bit Lemire method does not handle rng = 0xFFFFFFFF; a plain 32-bit word covers it (both samplers).
                if (rng == 0xFFFFFFFFUL)
                    return off + bg.NextUInt32();
                return off + (useMasked ? MaskedUInt32(bg, (uint)rng, (uint)mask) : LemireUInt32(bg, (uint)rng));
            }
            // Lemire64 doesn't support the inclusive rng 0xFFFFFFFFFFFFFFFF.
            if (rng == 0xFFFFFFFFFFFFFFFFUL)
                return off + bg.NextUInt64();
            return off + (useMasked ? MaskedUInt64(bg, rng, mask) : LemireUInt64(bg, rng));
        }

        /// <summary>Dispatches a bounded fill to the per-width loop that consumes the stream the way NumPy's does.</summary>
        /// <param name="bg">The bit generator (the caller holds its lock).</param>
        /// <param name="nd">The freshly allocated C-contiguous destination.</param>
        /// <param name="cnt">The number of values to draw.</param>
        /// <param name="width">1 (bool), 8, 16, 32 or 64.</param>
        /// <param name="off">The low bound's unsigned bit pattern.</param>
        /// <param name="rng">The closed-interval range.</param>
        /// <param name="useMasked">Masked rejection (legacy) instead of Lemire.</param>
        /// <remarks>
        ///     The width dispatch is REQUIRED for parity, not a performance switch: the 8/16-bit and bool loops split
        ///     one 32-bit word into several values, so the same range drawn at another width consumes the stream
        ///     differently.
        /// </remarks>
        internal static unsafe void Fill(BitGenerator bg, NDArray nd, long cnt, int width, ulong off, ulong rng, bool useMasked)
        {
            void* addr = (void*)(nd.Storage.Address + nd.Shape.offset * nd.dtypesize);
            switch (width)
            {
                case 1: FillBool(bg, (byte*)addr, cnt, (byte)off, (byte)rng); break;
                case 8: FillUInt8(bg, (byte*)addr, cnt, (byte)off, (byte)rng, useMasked); break;
                case 16: FillUInt16(bg, (ushort*)addr, cnt, (ushort)off, (ushort)rng, useMasked); break;
                case 32: FillUInt32(bg, (uint*)addr, cnt, (uint)off, (uint)rng, useMasked); break;
                default: FillUInt64(bg, (ulong*)addr, cnt, off, rng, useMasked); break;
            }
        }

        /// <summary>NumPy's <c>random_bounded_uint64_fill</c>: 32-bit draws for ranges that fit, 64-bit otherwise.</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="outp">Destination.</param>
        /// <param name="cnt">Value count.</param>
        /// <param name="off">Low bound.</param>
        /// <param name="rng">Closed-interval range.</param>
        /// <param name="useMasked">Masked rejection (legacy) instead of Lemire.</param>
        private static unsafe void FillUInt64(BitGenerator bg, ulong* outp, long cnt, ulong off, ulong rng, bool useMasked)
        {
            if (rng == 0)
            {
                for (long i = 0; i < cnt; i++) outp[i] = off;
            }
            else if (rng <= 0xFFFFFFFFUL)
            {
                // Call the 32-bit generator when the RANGE fits 32 bits — even if the bounds need 64.
                if (rng == 0xFFFFFFFFUL)
                {
                    for (long i = 0; i < cnt; i++) outp[i] = off + bg.NextUInt32();
                }
                else if (useMasked)
                {
                    uint r = (uint)rng, mask = (uint)GenMask(rng);
                    for (long i = 0; i < cnt; i++) outp[i] = off + MaskedUInt32(bg, r, mask);
                }
                else
                {
                    uint r = (uint)rng;
                    for (long i = 0; i < cnt; i++) outp[i] = off + LemireUInt32(bg, r);
                }
            }
            else if (rng == 0xFFFFFFFFFFFFFFFFUL)
            {
                for (long i = 0; i < cnt; i++) outp[i] = off + bg.NextUInt64();
            }
            else if (useMasked)
            {
                ulong mask = GenMask(rng);
                for (long i = 0; i < cnt; i++) outp[i] = off + MaskedUInt64(bg, rng, mask);
            }
            else
            {
                for (long i = 0; i < cnt; i++) outp[i] = off + LemireUInt64(bg, rng);
            }
        }

        /// <summary>NumPy's <c>random_bounded_uint32_fill</c>.</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="outp">Destination.</param>
        /// <param name="cnt">Value count.</param>
        /// <param name="off">Low bound.</param>
        /// <param name="rng">Closed-interval range.</param>
        /// <param name="useMasked">Masked rejection (legacy) instead of Lemire.</param>
        private static unsafe void FillUInt32(BitGenerator bg, uint* outp, long cnt, uint off, uint rng, bool useMasked)
        {
            if (rng == 0)
            {
                for (long i = 0; i < cnt; i++) outp[i] = off;
            }
            else if (rng == 0xFFFFFFFFu)
            {
                // Lemire32 doesn't support rng = 0xFFFFFFFF.
                for (long i = 0; i < cnt; i++) outp[i] = off + bg.NextUInt32();
            }
            else if (useMasked)
            {
                uint mask = (uint)GenMask(rng);
                for (long i = 0; i < cnt; i++) outp[i] = off + MaskedUInt32(bg, rng, mask);
            }
            else
            {
                for (long i = 0; i < cnt; i++) outp[i] = off + LemireUInt32(bg, rng);
            }
        }

        /// <summary>NumPy's <c>random_bounded_uint16_fill</c>: one 32-bit word feeds two values (low half first).</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="outp">Destination.</param>
        /// <param name="cnt">Value count.</param>
        /// <param name="off">Low bound.</param>
        /// <param name="rng">Closed-interval range.</param>
        /// <param name="useMasked">Masked rejection (legacy) instead of Lemire.</param>
        private static unsafe void FillUInt16(BitGenerator bg, ushort* outp, long cnt, ushort off, ushort rng, bool useMasked)
        {
            uint buf = 0;
            int bcnt = 0;
            if (rng == 0)
            {
                for (long i = 0; i < cnt; i++) outp[i] = off;
            }
            else if (rng == 0xFFFF)
            {
                // Lemire16 doesn't support rng = 0xFFFF.
                for (long i = 0; i < cnt; i++) outp[i] = (ushort)(off + BufferedUInt16(bg, ref buf, ref bcnt));
            }
            else if (useMasked)
            {
                ushort mask = (ushort)GenMask(rng);
                for (long i = 0; i < cnt; i++)
                {
                    ushort val;
                    while ((val = (ushort)(BufferedUInt16(bg, ref buf, ref bcnt) & mask)) > rng) { }
                    outp[i] = (ushort)(off + val);
                }
            }
            else
            {
                for (long i = 0; i < cnt; i++) outp[i] = (ushort)(off + LemireUInt16(bg, rng, ref buf, ref bcnt));
            }
        }

        /// <summary>NumPy's <c>random_bounded_uint8_fill</c>: one 32-bit word feeds four values (low byte first).</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="outp">Destination.</param>
        /// <param name="cnt">Value count.</param>
        /// <param name="off">Low bound.</param>
        /// <param name="rng">Closed-interval range.</param>
        /// <param name="useMasked">Masked rejection (legacy) instead of Lemire.</param>
        private static unsafe void FillUInt8(BitGenerator bg, byte* outp, long cnt, byte off, byte rng, bool useMasked)
        {
            uint buf = 0;
            int bcnt = 0;
            if (rng == 0)
            {
                for (long i = 0; i < cnt; i++) outp[i] = off;
            }
            else if (rng == 0xFF)
            {
                // Lemire8 doesn't support rng = 0xFF.
                for (long i = 0; i < cnt; i++) outp[i] = (byte)(off + BufferedUInt8(bg, ref buf, ref bcnt));
            }
            else if (useMasked)
            {
                byte mask = (byte)GenMask(rng);
                for (long i = 0; i < cnt; i++)
                {
                    byte val;
                    while ((val = (byte)(BufferedUInt8(bg, ref buf, ref bcnt) & mask)) > rng) { }
                    outp[i] = (byte)(off + val);
                }
            }
            else
            {
                for (long i = 0; i < cnt; i++) outp[i] = (byte)(off + LemireUInt8(bg, rng, ref buf, ref bcnt));
            }
        }

        /// <summary>
        ///     NumPy's <c>random_bounded_bool_fill</c>: one bit per value from a buffered 32-bit word (the same for
        ///     both samplers; the low bound is used only when the range is 0).
        /// </summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="outp">Destination.</param>
        /// <param name="cnt">Value count.</param>
        /// <param name="off">Low bound (the value when the range is 0).</param>
        /// <param name="rng">Closed-interval range (0 or 1).</param>
        private static unsafe void FillBool(BitGenerator bg, byte* outp, long cnt, byte off, byte rng)
        {
            uint buf = 0;
            int bcnt = 0;
            for (long i = 0; i < cnt; i++)
            {
                if (rng == 0) { outp[i] = off; continue; }
                if (bcnt == 0) { buf = bg.NextUInt32(); bcnt = 31; }
                else { buf >>= 1; bcnt -= 1; }
                outp[i] = (byte)((buf & 0x1u) != 0 ? 1 : 0);
            }
        }

        // ---- 32-bit buffer splitters (numpy buffered_uint16 / buffered_uint8) ----

        /// <summary>NumPy's <c>buffered_uint16</c>: serves the two 16-bit halves of one 32-bit word, low half first.</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="buf">The buffered word (state across calls within one fill).</param>
        /// <param name="bcnt">The number of halves still buffered.</param>
        /// <returns>The next 16-bit value.</returns>
        private static ushort BufferedUInt16(BitGenerator bg, ref uint buf, ref int bcnt)
        {
            if (bcnt == 0) { buf = bg.NextUInt32(); bcnt = 1; }
            else { buf >>= 16; bcnt -= 1; }
            return (ushort)buf;
        }

        /// <summary>NumPy's <c>buffered_uint8</c>: serves the four bytes of one 32-bit word, low byte first.</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="buf">The buffered word (state across calls within one fill).</param>
        /// <param name="bcnt">The number of bytes still buffered.</param>
        /// <returns>The next 8-bit value.</returns>
        private static byte BufferedUInt8(BitGenerator bg, ref uint buf, ref int bcnt)
        {
            if (bcnt == 0) { buf = bg.NextUInt32(); bcnt = 3; }
            else { buf >>= 8; bcnt -= 1; }
            return (byte)buf;
        }

        // ---- masked rejection (numpy bounded_masked_uint64 / buffered_bounded_masked_uint32) ----

        /// <summary>NumPy's <c>bounded_masked_uint64</c>: 64-bit words under <paramref name="mask"/> until one is &lt;= <paramref name="rng"/>.</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="rng">The inclusive range.</param>
        /// <param name="mask">The covering bit mask.</param>
        /// <returns>The draw.</returns>
        private static ulong MaskedUInt64(BitGenerator bg, ulong rng, ulong mask)
        {
            ulong val;
            while ((val = bg.NextUInt64() & mask) > rng) { }
            return val;
        }

        /// <summary>NumPy's <c>buffered_bounded_masked_uint32</c>: 32-bit words under <paramref name="mask"/> until one is &lt;= <paramref name="rng"/>.</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="rng">The inclusive range.</param>
        /// <param name="mask">The covering bit mask.</param>
        /// <returns>The draw.</returns>
        private static uint MaskedUInt32(BitGenerator bg, uint rng, uint mask)
        {
            uint val;
            while ((val = bg.NextUInt32() & mask) > rng) { }
            return val;
        }

        // ---- Lemire bounded generators (numpy bounded_lemire_uintX) ----

        /// <summary>NumPy's <c>bounded_lemire_uint64</c>: unbiased draw in <c>[0, rng]</c> by 128-bit multiply-and-reject.</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="rng">Closed-interval range (below 2**64-1).</param>
        /// <returns>The draw.</returns>
        internal static ulong LemireUInt64(BitGenerator bg, ulong rng)
        {
            ulong rngExcl = rng + 1;
            UInt128 m = (UInt128)bg.NextUInt64() * rngExcl;
            ulong leftover = (ulong)m;
            if (leftover < rngExcl)
            {
                ulong threshold = (ulong.MaxValue - rng) % rngExcl;
                while (leftover < threshold)
                {
                    m = (UInt128)bg.NextUInt64() * rngExcl;
                    leftover = (ulong)m;
                }
            }
            return (ulong)(m >> 64);
        }

        /// <summary>NumPy's <c>buffered_bounded_lemire_uint32</c>: unbiased draw in <c>[0, rng]</c> by 64-bit multiply-and-reject.</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="rng">Closed-interval range (below 2**32-1).</param>
        /// <returns>The draw.</returns>
        internal static uint LemireUInt32(BitGenerator bg, uint rng)
        {
            uint rngExcl = rng + 1;
            ulong m = (ulong)bg.NextUInt32() * rngExcl;
            uint leftover = (uint)m;
            if (leftover < rngExcl)
            {
                uint threshold = (uint.MaxValue - rng) % rngExcl;
                while (leftover < threshold)
                {
                    m = (ulong)bg.NextUInt32() * rngExcl;
                    leftover = (uint)m;
                }
            }
            return (uint)(m >> 32);
        }

        /// <summary>NumPy's <c>buffered_bounded_lemire_uint16</c>: Lemire over buffered 16-bit halves.</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="rng">Closed-interval range (below 0xFFFF).</param>
        /// <param name="buf">The buffered word.</param>
        /// <param name="bcnt">The number of halves still buffered.</param>
        /// <returns>The draw.</returns>
        private static ushort LemireUInt16(BitGenerator bg, ushort rng, ref uint buf, ref int bcnt)
        {
            ushort rngExcl = (ushort)(rng + 1);
            uint m = (uint)BufferedUInt16(bg, ref buf, ref bcnt) * rngExcl;
            ushort leftover = (ushort)m;
            if (leftover < rngExcl)
            {
                ushort threshold = (ushort)((ushort)(ushort.MaxValue - rng) % rngExcl);
                while (leftover < threshold)
                {
                    m = (uint)BufferedUInt16(bg, ref buf, ref bcnt) * rngExcl;
                    leftover = (ushort)m;
                }
            }
            return (ushort)(m >> 16);
        }

        /// <summary>NumPy's <c>buffered_bounded_lemire_uint8</c>: Lemire over buffered bytes.</summary>
        /// <param name="bg">The bit generator.</param>
        /// <param name="rng">Closed-interval range (below 0xFF).</param>
        /// <param name="buf">The buffered word.</param>
        /// <param name="bcnt">The number of bytes still buffered.</param>
        /// <returns>The draw.</returns>
        private static byte LemireUInt8(BitGenerator bg, byte rng, ref uint buf, ref int bcnt)
        {
            byte rngExcl = (byte)(rng + 1);
            uint m = (uint)BufferedUInt8(bg, ref buf, ref bcnt) * rngExcl;
            byte leftover = (byte)m;
            if (leftover < rngExcl)
            {
                byte threshold = (byte)((byte)(byte.MaxValue - rng) % rngExcl);
                while (leftover < threshold)
                {
                    m = (uint)BufferedUInt8(bg, ref buf, ref bcnt) * rngExcl;
                    leftover = (byte)m;
                }
            }
            return (byte)(m >> 8);
        }

        /// <summary>
        ///     NumPy's <c>_shuffle_raw</c>: swap element <c>i</c> with <c>random_interval(i)</c> for
        ///     <c>i = n-1 … 1</c>, over <paramref name="n"/> elements <paramref name="strideBytes"/> apart.
        /// </summary>
        /// <param name="bg">The bit generator (the caller holds its lock).</param>
        /// <param name="basePtr">Address of element 0.</param>
        /// <param name="n">Element count.</param>
        /// <param name="strideBytes">Byte distance between consecutive elements (may be negative).</param>
        /// <param name="itemsize">Element size in bytes (at most 16).</param>
        /// <remarks>
        ///     Every <c>i</c> draws, and <c>i == j</c> skips only the copy — the stream consumption is NumPy's for
        ///     both the Generator's and the legacy RandomState's shuffles.
        /// </remarks>
        internal static unsafe void ShuffleRaw(BitGenerator bg, byte* basePtr, long n, long strideBytes, int itemsize)
        {
            byte* buf = stackalloc byte[16]; // widest dtype (Complex/Decimal)
            for (long i = n - 1; i >= 1; i--)
            {
                ulong j = RandomInterval(bg, (ulong)i);
                if ((long)j == i)
                    continue;
                byte* pi = basePtr + i * strideBytes;
                byte* pj = basePtr + (long)j * strideBytes;
                Buffer.MemoryCopy(pj, buf, 16, itemsize);
                Buffer.MemoryCopy(pi, pj, itemsize, itemsize);
                Buffer.MemoryCopy(buf, pi, itemsize, itemsize);
            }
        }

        /// <summary>
        ///     The permutation an in-place Fisher–Yates (random_interval) over <c>[0, m)</c> leaves behind — the index
        ///     order an N-D sub-array swap loop produces.
        /// </summary>
        /// <param name="bg">The bit generator (the caller holds its lock).</param>
        /// <param name="m">The length of the permuted axis.</param>
        /// <returns>The permuted index order.</returns>
        /// <remarks>Draws for every <c>i</c> (even when the draw equals <c>i</c>), as NumPy does.</remarks>
        internal static long[] FisherYatesIndices(BitGenerator bg, long m)
        {
            var idx = new long[m];
            for (long k = 0; k < m; k++)
                idx[k] = k;
            for (long i = m - 1; i >= 1; i--)
            {
                ulong j = RandomInterval(bg, (ulong)i);
                if ((long)j != i)
                    (idx[i], idx[j]) = (idx[j], idx[i]);
            }
            return idx;
        }
    }
}
