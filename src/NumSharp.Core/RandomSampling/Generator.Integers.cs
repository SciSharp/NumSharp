using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Return random integers from <paramref name="low"/> (inclusive) to <paramref name="high"/>
        ///     (exclusive, or inclusive when <paramref name="endpoint"/> is true).
        /// </summary>
        /// <param name="low">Lowest integer drawn (or the highest, one above, when <paramref name="high"/> is null).</param>
        /// <param name="high">If provided, one above the largest integer drawn (or the largest when <paramref name="endpoint"/>).</param>
        /// <param name="size">Output shape. If default/scalar a single value is returned.</param>
        /// <param name="dtype">Desired integer dtype. Default is int64.</param>
        /// <param name="endpoint">If true, sample from the closed interval <c>[low, high]</c>.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
        ///     <br/>
        ///     Uses Lemire's method (NumPy's Generator default, <c>use_masked=False</c>) — NOT the
        ///     legacy masked rejection of <c>RandomState.randint</c> — so the stream is byte-identical
        ///     to <c>default_rng(seed).integers(...)</c>.
        /// </remarks>
        public NDArray integers(long low, long? high = null, Shape size = default, Type dtype = null, bool endpoint = false)
        {
            dtype = dtype ?? typeof(long);
            NPTypeCode tc = dtype.GetTypeCode();

            long lo, hiArg;
            if (high == null)
            {
                hiArg = low;
                lo = 0;
            }
            else
            {
                lo = low;
                hiArg = high.Value;
            }

            // Internal generator produces on the closed interval; subtract 1 for the half-open case.
            long highInclusive = endpoint ? hiArg : hiArg - 1;

            ComputeOffRng(tc, dtype, lo, highInclusive, endpoint, out int width, out ulong off, out ulong rng);

            // size == 0 -> empty array of the requested dtype (drawn no state).
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
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.bytes.html
        ///     <br/>
        ///     Byte-identical to NumPy: draws <c>ceil(length/4)</c> uint32 words from PCG64 (via the
        ///     32-bit buffered path), packs them little-endian, and truncates to <paramref name="length"/>.
        /// </remarks>
        public byte[] bytes(long length)
        {
            long nUint32 = (length - 1) / 4 + 1; // C truncation, as npy_intp
            if (nUint32 < 0)
                throw new ValueError("negative dimensions are not allowed");

            long totalBytes = nUint32 * 4;
            var full = new byte[totalBytes];
            long pos = 0;
            for (long w = 0; w < nUint32; w++)
            {
                uint r = _bitGenerator.NextUInt32();
                full[pos++] = (byte)r;
                full[pos++] = (byte)(r >> 8);
                full[pos++] = (byte)(r >> 16);
                full[pos++] = (byte)(r >> 24);
            }

            long end = length >= 0 ? Math.Min(length, totalBytes) : Math.Max(0, totalBytes + length);
            if (end == totalBytes)
                return full;
            var result = new byte[end];
            Array.Copy(full, result, end);
            return result;
        }

        // ---- off/rng computation + validation (numpy _bounded_integers.pyx.in scalar path) ----

        private static void ComputeOffRng(NPTypeCode tc, Type dtype, long lo, long highInclusive, bool endpoint,
                                          out int width, out ulong off, out ulong rng)
        {
            // (lb, ub, width). ub is the EXCLUSIVE upper bound; only meaningful (and checkable
            // against a long input) for the <= 32-bit widths.
            long lb;
            long ub;
            switch (tc)
            {
                case NPTypeCode.Boolean: lb = 0; ub = 2; width = 1; break;
                case NPTypeCode.Byte: lb = 0; ub = 0x100L; width = 8; break;
                case NPTypeCode.SByte: lb = -0x80L; ub = 0x80L; width = 8; break;
                case NPTypeCode.UInt16: lb = 0; ub = 0x10000L; width = 16; break;
                case NPTypeCode.Int16: lb = -0x8000L; ub = 0x8000L; width = 16; break;
                case NPTypeCode.UInt32: lb = 0; ub = 0x100000000L; width = 32; break;
                case NPTypeCode.Int32: lb = -0x80000000L; ub = 0x80000000L; width = 32; break;
                case NPTypeCode.UInt64: lb = 0; ub = long.MaxValue; width = 64; break; // ub 2^64 unreachable by long
                case NPTypeCode.Int64: lb = long.MinValue; ub = long.MaxValue; width = 64; break; // ub 2^63 unreachable
                default:
                    throw new TypeError($"Unsupported dtype {dtype.Name} for integers");
            }

            if (lo < lb)
                throw new ValueError($"low is out of bounds for {tc.AsNumpyDtypeName()}");
            // For 64-bit widths a long can never exceed the exclusive bound, so skip that check.
            if (width <= 32 && highInclusive > ub)
                throw new ValueError($"high is out of bounds for {tc.AsNumpyDtypeName()}");
            if (lo > highInclusive)
                throw new ValueError(FormatBoundsError(endpoint, lo));

            ulong widthMask = width >= 64 ? ulong.MaxValue : ((1UL << width) - 1UL);
            off = unchecked((ulong)lo) & widthMask;
            rng = unchecked((ulong)(highInclusive - lo)) & widthMask;
        }

        private static string FormatBoundsError(bool closed, long low)
        {
            if (low == 0)
                return closed ? "high < 0" : "high <= 0";
            return closed ? "low > high" : "low >= high";
        }

        // ---- the per-width bounded fills (numpy random_bounded_uintX_fill, use_masked=False) ----

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

        private unsafe void FillBoundedUInt32(uint* outp, long cnt, uint off, uint rng)
        {
            if (rng == 0)
                for (long i = 0; i < cnt; i++) outp[i] = off;
            else if (rng == 0xFFFFFFFFu)
                for (long i = 0; i < cnt; i++) outp[i] = off + _bitGenerator.NextUInt32();
            else
                for (long i = 0; i < cnt; i++) outp[i] = off + LemireUint32(rng);
        }

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

        private ushort BufferedUint16(ref uint buf, ref int bcnt)
        {
            if (bcnt == 0) { buf = _bitGenerator.NextUInt32(); bcnt = 1; }
            else { buf >>= 16; bcnt -= 1; }
            return (ushort)buf;
        }

        private byte BufferedUint8(ref uint buf, ref int bcnt)
        {
            if (bcnt == 0) { buf = _bitGenerator.NextUInt32(); bcnt = 3; }
            else { buf >>= 8; bcnt -= 1; }
            return (byte)buf;
        }

        // ---- Lemire bounded generators (numpy bounded_lemire_uintX) ----

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
