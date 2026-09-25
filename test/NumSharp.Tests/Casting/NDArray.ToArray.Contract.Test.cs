using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     Pins <see cref="NDArray.ToArray{T}"/> — the copy of an NDArray into a flat .NET <c>T[]</c> in logical C
    ///     (row-major) order — for all 15 dtypes over every shape size and memory layout its dispatch tells apart, for
    ///     all 210 (source dtype, T) conversions over every conversion route, against NumPy 2.4.2 on representative
    ///     conversions, and on its contract: a fresh copy, an untouched source, the shared empty array, and its refusals.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Why so wide: ToArray is not one loop but a dispatch over element width, layout and size, and every route is
    ///         separate code per width. Same dtype: a C-contiguous source (any offset) is one allocation plus one copy —
    ///         up to 64 bytes by at most four overlapping 16-byte moves, beyond that a block copy, the result zeroed below
    ///         8 KB and uninitialized from it; one element; a single-value broadcast (one fill); 1- and 2-byte stride-2 or
    ///         reversed walks of at least 2 048 elements (NumSharp's iterator); a rank-2 view canonicalized into a line
    ///         (an extent-1 axis, rows that merge) or a plane; a line by its stride (block copy, fill, vector reverse —
    ///         streamed by whole cache lines from a mebibyte, with a separate 16-byte-element streamer —, AVX2 deinterleave
    ///         for stride 2, hardware gathers for other strides); a plane by its strides (row copies, row fills, reversed
    ///         rows — streamed per row or through an L1 scratch from a mebibyte —, 8×8 / 8×4 / 4×4 register-block
    ///         transposes with overlapping ragged tails up to 32 KB, float64-width transposes streamed from a mebibyte by
    ///         whole lines when the column count is a multiple of 8 and by 32-byte segments for a multiple of 4, 64-column
    ///         transpose bands, deinterleaves and gathers); a rank ≥ 3 view transposed over an outer unit-stride axis; and
    ///         an address-order pre-fault before every transposing fill of a large-object result (85 000 bytes and up).
    ///         Conversions: AVX2 converters for 24 pairs, whole-line non-temporal converters for 7 pairs from a mebibyte,
    ///         the generator's cast kernel (remembered per pair from its second call on), a scalar odometer up to 1 024
    ///         elements, the iterator's cast copy beyond that, and a bit-exact decimal → float64 kernel. A size or a layout
    ///         that no test reaches is a route no test reaches, so the catalogs below walk every size threshold from both
    ///         sides, every ragged register-block tail shape, and every element width.
    ///     </para>
    ///     <para>
    ///         Why the oracle is independent: other suites (the tobytes and ToMuliDimArray contracts) use ToArray as THEIR
    ///         oracle, so this one must not. <see cref="LogicalBytes"/> reads a view's own buffer through its dimensions,
    ///         element strides and offset with a plain odometer, one element at a time — no NumSharp copy, iterator or
    ///         kernel. A same-dtype result is compared with it as RAW BYTES, so NaN payloads, signalling NaNs, signed zeros,
    ///         decimal scales and every other bit pattern must survive every lane move; the sources are random bit patterns
    ///         with those specials planted at a fixed period (<see cref="RawPattern"/>). A conversion is compared with
    ///         <c>astype</c> — the documented contract — read through the same oracle, as raw bytes (a decimal target by
    ///         value, the project's oracle convention: 1.0m and 1.00m are the same number), over values chosen so that
    ///         every conversion is defined (<see cref="ConversionPattern"/>: no NaN, infinity or out-of-range float →
    ///         integer, which C leaves undefined and hosts answer differently); and
    ///         <see cref="Conversion_RepresentativeValues_MatchNumPyOnEveryRoute"/> anchors representative conversions to
    ///         NumPy 2.4.2's own output on every route.
    ///     </para>
    ///     <para>
    ///         Three known divergences are pinned as <c>[OpenBugs]</c> rather than hidden, each with NumPy's exact bits on
    ///         every route: a signalling NaN converted into or out of float16 is quieted by ToArray's scalar routes (while
    ///         astype keeps it signalling, as NumPy does) — the one input the conversion matrix quiets beforehand, and
    ///         only for float16 conversions; astype's own float16 → float64 / complex128 conversion quiets it on every
    ///         route; and on .NET 8 uint64 ≥ 2⁶³ → float64 rounds twice (the runtime's own cast).
    ///     </para>
    ///     <para>
    ///         Every case also checks that the source's buffer is byte-for-byte unchanged afterwards (a mis-aimed store in
    ///         a transpose block or a streamed line would land there), that a second call returns an equal but distinct
    ///         array, and that an empty result is the shared <see cref="Array.Empty{T}"/> instance. Failures are collected
    ///         rather than thrown at the first one, so a broken route reports every (dtype, layout) cell it breaks.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ToArrayContractTests
    {
        /// <summary>The 15 NumSharp dtypes.</summary>
        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        /// <summary>One mebibyte: the size from which ToArray streams reversed runs, transposes and conversions.</summary>
        private const long Mebibyte = 1L << 20;

        // ============================================================================================== tests

        /// <summary>
        ///     Every dtype, through every shape size and memory layout that selects a distinct copy route — both sides of
        ///     every size threshold, every ragged register-block tail — copies exactly the view's logical C-order elements,
        ///     bit for bit, and leaves its source untouched.
        /// </summary>
        /// <param name="dtype">The dtype under test (also the requested element type: no conversion).</param>
        /// <exception cref="AssertFailedException">
        ///     A case returned the wrong length or bytes, reused an array, wrote into its source, or threw; or the catalog
        ///     ran fewer cases than it lists.
        /// </exception>
        /// <remarks>
        ///     1 915–1 975 cases per dtype, by element width (see <see cref="RunSameDtypeCatalog{T}"/>): small and mid-size
        ///     views — among them 0-d, empty and one-element views, contiguous views at a non-zero offset (split children),
        ///     broadcasts, views of views, F-ordered copies and an array over managed memory (<c>np.frombuffer</c>) — and 21
        ///     large ones at the mebibyte and large-object sizes.
        /// </remarks>
        [TestMethod]
        [DataRow(NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Complex)]
        public void SameDtype_EveryShapeAndLayout_CopiesTheLogicalElementsBitExact(NPTypeCode dtype)
        {
            var log = new CaseLog();
            Visit(dtype, new SameDtypeVisitor(log));
            AssertNoFailures(log, $"{dtype} same-dtype catalog", minimumCases: 1900);
        }

        /// <summary>
        ///     From every source dtype into every other dtype, over every conversion route, ToArray converts exactly as
        ///     <c>astype</c> does — on the first call of a pair and on the next (which runs the remembered kernel) — and
        ///     leaves its source untouched.
        /// </summary>
        /// <param name="from">The source dtype; every other dtype is a target.</param>
        /// <exception cref="AssertFailedException">
        ///     A case differed from astype, reused an array, wrote into its source, or threw; or the catalog ran fewer cases
        ///     than it lists.
        /// </exception>
        /// <remarks>
        ///     61 cases per target (see <see cref="RunConversionCatalog{T}"/>): contiguous runs from 1 element to 5 000 (the
        ///     scalar converter, the AVX2 converters from one destination vector on, the cast kernel), contiguous views at a
        ///     zero and at a non-zero offset (slices and split children), 0-d, empty and one-element views, non-contiguous
        ///     views of at most 1 024 elements (the scalar odometer) and of more (the iterator's cast copy); plus, per
        ///     source, one transposed large-object result (the pre-fault; its target rotates so every target gets one) and,
        ///     for the four targets with dedicated converters, a mebibyte of contiguous result (the line converters) — 864
        ///     or 867 cases per source dtype.
        /// </remarks>
        [TestMethod]
        [DataRow(NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Complex)]
        public void Conversion_EveryTargetAndLayout_MatchesAstype(NPTypeCode from)
        {
            var log = new CaseLog();
            var visitor = new ConversionVisitor(from, log);
            foreach (NPTypeCode to in AllDtypes)
                if (to != from)
                    Visit(to, visitor);
            AssertNoFailures(log, $"{from} → every other dtype", minimumCases: 14 * 61);
        }

        /// <summary>
        ///     Representative conversions produce NumPy 2.4.2's exact output — truncation toward zero, modular wrap,
        ///     round-to-nearest-even, overflow to infinity, signed zeros, subnormals, complex → real and complex → bool,
        ///     bool both ways — on every route a converting call can take: contiguous, at an offset, reversed, stepped,
        ///     transposed, broadcast, 0-d, and large enough for the mebibyte converters and the iterator's cast copy.
        /// </summary>
        /// <exception cref="AssertFailedException">A route's result differs from NumPy's bytes.</exception>
        /// <remarks>
        ///     Each expected array is what NumPy 2.4.2 printed for <c>np.array(input, from_dtype).astype(to_dtype)</c>
        ///     (floats checked by their bits). The last two cases are NumSharp-only dtypes: a char converts as its UTF-16
        ///     code unit (NumPy's uint16) and a decimal truncates toward zero (NumPy's float rule) — each checked against
        ///     the NumPy statement of that model.
        /// </remarks>
        [TestMethod]
        public void Conversion_RepresentativeValues_MatchNumPyOnEveryRoute()
        {
            var log = new CaseLog();

            // numpy: np.array([1.9, -2.9, 0.5, -0.5, 127.99, -128.99]).astype(np.int8) → [1, -2, 0, 0, 127, -128]
            LikeNumPy(new[] { 1.9, -2.9, 0.5, -0.5, 127.99, -128.99 }, new sbyte[] { 1, -2, 0, 0, 127, -128 },
                "float64 → int8 truncates toward zero", log);
            // numpy: np.array([0.99, 255.5, 3.7, 0.0]).astype(np.uint8) → [0, 255, 3, 0]
            LikeNumPy(new[] { 0.99, 255.5, 3.7, 0.0 }, new byte[] { 0, 255, 3, 0 }, "float64 → uint8 truncates", log);
            // numpy: np.array([65535.9, -0.5, 0.99], np.float32).astype(np.uint16) → [65535, 0, 0]
            LikeNumPy(new[] { 65535.9f, -0.5f, 0.99f }, new ushort[] { 65535, 0, 0 }, "float32 → uint16 truncates", log);
            // numpy: np.array([4294967295.5, 0.99, -0.75]).astype(np.uint32) → [4294967295, 0, 0]
            LikeNumPy(new[] { 4294967295.5, 0.99, -0.75 }, new uint[] { 4294967295, 0, 0 }, "float64 → uint32 truncates", log);
            // numpy: np.array([1.8446744073709550e19, 0.5, 9.2e18]).astype(np.uint64) → [18446744073709549568, 0, 9200000000000000000]
            LikeNumPy(new[] { 1.8446744073709550e19, 0.5, 9.2e18 }, new ulong[] { 18446744073709549568, 0, 9200000000000000000 },
                "float64 → uint64 truncates", log);
            // numpy: np.array([-1.5, 2.5, 32752], np.float16).astype(np.int16) → [-1, 2, 32752]
            LikeNumPy(new[] { (Half)(-1.5), (Half)2.5, (Half)32752 }, new short[] { -1, 2, 32752 }, "float16 → int16 truncates", log);
            // numpy: np.array([-0.9, -1.1, 1.5e10], np.float32).astype(np.int64) → [0, -1, 15000000512]
            LikeNumPy(new[] { -0.9f, -1.1f, 1.5e10f }, new long[] { 0, -1, 15000000512 }, "float32 → int64 truncates", log);
            // numpy: np.array([300, -1, 256, 255, -129], np.int32).astype(np.uint8) → [44, 255, 0, 255, 127]
            LikeNumPy(new[] { 300, -1, 256, 255, -129 }, new byte[] { 44, 255, 0, 255, 127 }, "int32 → uint8 wraps modulo 256", log);
            // numpy: np.array([300, -1, 256, 255, -129, 128], np.int32).astype(np.int8) → [44, -1, 0, -1, 127, -128]
            LikeNumPy(new[] { 300, -1, 256, 255, -129, 128 }, new sbyte[] { 44, -1, 0, -1, 127, -128 }, "int32 → int8 wraps", log);
            // numpy: np.array([1, 5_000_000_000, -1, 2**31, -2**31 - 1], np.int64).astype(np.int32)
            //        → [1, 705032704, -1, -2147483648, 2147483647]
            LikeNumPy(new long[] { 1, 5_000_000_000, -1, 2147483648, -2147483649 },
                new[] { 1, 705032704, -1, -2147483648, 2147483647 }, "int64 → int32 keeps the low 32 bits", log);
            // numpy: np.array([65535, 32768, 1], np.uint16).astype(np.int16) → [-1, -32768, 1]
            LikeNumPy(new ushort[] { 65535, 32768, 1 }, new short[] { -1, -32768, 1 }, "uint16 → int16 wraps", log);
            // numpy: np.array([-1, 5], np.int8).astype(np.uint64) → [18446744073709551615, 5]
            LikeNumPy(new sbyte[] { -1, 5 }, new ulong[] { 18446744073709551615, 5 }, "int8 → uint64 sign-extends, then wraps", log);
            // numpy: np.array([-2**31, -1, 2**31 - 1], np.int32).astype(np.int64) → [-2147483648, -1, 2147483647]
            LikeNumPy(new[] { int.MinValue, -1, int.MaxValue }, new long[] { -2147483648, -1, 2147483647 }, "int32 → int64 sign-extends", log);
            // numpy: np.array([4294967295, 0], np.uint32).astype(np.int64) → [4294967295, 0]
            LikeNumPy(new uint[] { 4294967295, 0 }, new long[] { 4294967295, 0 }, "uint32 → int64 zero-extends", log);
            // numpy: np.array([-2**31, -1, 2**31 - 1], np.int32).astype(np.float64) → [-2147483648.0, -1.0, 2147483647.0]
            LikeNumPy(new[] { int.MinValue, -1, int.MaxValue }, new[] { -2147483648.0, -1.0, 2147483647.0 }, "int32 → float64 is exact", log);
            // numpy: np.array([16777217, 2**31 - 1, -16777219], np.int32).astype(np.float32).view(np.uint32)
            //        → [0x4b800000, 0x4f000000, 0xcb800002]
            LikeNumPy(new[] { 16777217, int.MaxValue, -16777219 }, Singles(0x4B800000, 0x4F000000, 0xCB800002),
                "int32 → float32 rounds to nearest-even", log);
            // numpy: np.array([0, 200, 255], np.uint8).astype(np.float32) → [0.0, 200.0, 255.0]
            LikeNumPy(new byte[] { 0, 200, 255 }, new[] { 0f, 200f, 255f }, "uint8 → float32 is exact", log);
            // numpy: np.array([2**53 + 1, 2**64 - 1, 2**63 + 1], np.uint64).astype(np.float64).view(np.uint64)
            //        → [0x4340000000000000, 0x43f0000000000000, 0x43e0000000000000]
            LikeNumPy(new ulong[] { 9007199254740993, 18446744073709551615, 9223372036854775809 },
                Doubles(0x4340000000000000, 0x43F0000000000000, 0x43E0000000000000), "uint64 → float64 rounds to nearest-even", log);
            // numpy: np.array([2049, 65519, 65520, -70000, 2050, 3], np.int64).astype(np.float16).view(np.uint16)
            //        → [0x6800, 0x7bff, 0x7c00, 0xfc00, 0x6801, 0x4200]
            LikeNumPy(new long[] { 2049, 65519, 65520, -70000, 2050, 3 }, Halves(0x6800, 0x7BFF, 0x7C00, 0xFC00, 0x6801, 0x4200),
                "int64 → float16 rounds to nearest-even and overflows to ±inf", log);
            // numpy: np.array([1e40, 1.0000000596046448, 3.4028235677973366e38, -1e-50, 1.401298464324817e-45])
            //        .astype(np.float32).view(np.uint32) → [0x7f800000, 0x3f800000, 0x7f800000, 0x80000000, 0x1]
            LikeNumPy(new[] { 1e40, 1.0000000596046448, 3.4028235677973366e38, -1e-50, 1.401298464324817e-45 },
                Singles(0x7F800000, 0x3F800000, 0x7F800000, 0x80000000, 0x00000001),
                "float64 → float32 rounds to nearest-even, overflows to inf, keeps -0 and subnormals", log);
            // numpy: np.array([1.5, -0.0, np.inf, 2**-149], np.float32).astype(np.float64).view(np.uint64)
            //        → [0x3ff8000000000000, 0x8000000000000000, 0x7ff0000000000000, 0x36a0000000000000]
            LikeNumPy(new[] { 1.5f, -0.0f, float.PositiveInfinity, 1.401298464324817e-45f },
                Doubles(0x3FF8000000000000, 0x8000000000000000, 0x7FF0000000000000, 0x36A0000000000000), "float32 → float64 is exact", log);
            // numpy: np.array([65504, 2**-24, -0.0], np.float16).astype(np.float64).view(np.uint64)
            //        → [0x40effc0000000000, 0x3e70000000000000, 0x8000000000000000]
            LikeNumPy(Halves(0x7BFF, 0x0001, 0x8000), Doubles(0x40EFFC0000000000, 0x3E70000000000000, 0x8000000000000000),
                "float16 → float64 is exact, subnormals and -0 included", log);
            // numpy: np.array([1+2j, -3.5-1j, -7j]).astype(np.float64) → [1.0, -3.5, 0.0]
            LikeNumPy(new[] { new Complex(1, 2), new Complex(-3.5, -1), new Complex(0, -7) }, new[] { 1.0, -3.5, 0.0 },
                "complex128 → float64 keeps the real part", log);
            // numpy: np.array([0j, 1j, 2+0j, complex(nan, 0), complex(0, -0.0)]).astype(bool) → [False, True, True, True, False]
            LikeNumPy(new[] { new Complex(0, 0), new Complex(0, 1), new Complex(2, 0), new Complex(double.NaN, 0), new Complex(0, -0.0) },
                new[] { false, true, true, true, false }, "complex128 → bool tests both parts", log);
            // numpy: np.array([0.0, -0.0, nan, 1e-320, inf]).astype(bool) → [False, False, True, True, True]
            LikeNumPy(new[] { 0.0, -0.0, double.NaN, 1e-320, double.PositiveInfinity }, new[] { false, false, true, true, true },
                "float64 → bool: NaN, subnormals and infinities are true, ±0 false", log);
            // numpy: np.array([nan, -0.0, 2**-24], np.float16).astype(bool) → [True, False, True]
            LikeNumPy(Halves(0x7E00, 0x8000, 0x0001), new[] { true, false, true }, "float16 → bool", log);
            // numpy: np.array([True, False]).astype(np.float64) → [1.0, 0.0]
            LikeNumPy(new[] { true, false }, new[] { 1.0, 0.0 }, "bool → float64", log);
            // numpy: np.array([1.5, -0.0]).astype(np.complex128) → [(1.5+0j), (-0+0j)] (real bits 0x3ff8…, 0x8000…; imaginary +0)
            LikeNumPy(new[] { 1.5, -0.0 }, new[] { new Complex(1.5, 0), new Complex(-0.0, 0) },
                "float64 → complex128 keeps -0 in the real part, +0 imaginary", log);
            // NumSharp-only char: its UTF-16 code unit, i.e. numpy's np.array([65, 65535, 55296], np.uint16).astype(np.int32)
            // → [65, 65535, 55296].
            // The code units as casts, not escapes: U+FFFF (a noncharacter) and U+D800 (a lone high surrogate).
            LikeNumPy(new[] { 'A', (char)0xFFFF, (char)0xD800 }, new[] { 65, 65535, 55296 }, "char → int32 reads the UTF-16 code unit", log);
            // NumSharp-only decimal: truncation toward zero, numpy's float rule —
            // np.array([2.5, 3.5, -2.5, 2.7, -2.7]).astype(np.int32) → [2, 3, -2, 2, -2].
            LikeNumPy(new[] { 2.5m, 3.5m, -2.5m, 2.7m, -2.7m }, new[] { 2, 3, -2, 2, -2 }, "decimal → int32 truncates toward zero", log);

            // 30 conversions × 8 layouts, plus one 0-d view per input value (109 of them).
            AssertNoFailures(log, "NumPy's representative conversions", minimumCases: 30 * 8 + 100);
        }

        /// <summary>
        ///     uint64 values at or above 2⁶³ convert to float64 with ONE rounding, as NumPy's correctly rounded cast does.
        ///     Fails on .NET 8 only, and not in ToArray: there the runtime's own <c>(double)ulong</c> converts as signed and
        ///     adds 2⁶⁴, rounding twice, and astype — which ToArray matches by contract — inherits it, so 2⁶³ + 1025 lands
        ///     on 2⁶³ instead of 2⁶³ + 2048.
        /// </summary>
        /// <exception cref="AssertFailedException">A route's float64 differs from NumPy's bits (on .NET 8: every route, for 2⁶³ + 1025).</exception>
        /// <remarks>
        ///     Probed: <c>np.array([2**63 + 1025, 2**63 + 2049, 2**63 + 1], np.uint64).astype(np.float64).view(np.uint64)</c>
        ///     → <c>[0x43e0000000000001, 0x43e0000000000001, 0x43e0000000000000]</c>; .NET 8.0.29 gives
        ///     <c>0x43e0000000000000</c> for the first through astype and ToArray alike (contiguous and strided), .NET 10
        ///     matches NumPy. The decimal → float64 kernel already guards the same runtime difference with a start-up
        ///     probe; the uint64 cast has no such guard. Remove <c>[OpenBugs]</c> when astype rounds once on every runtime.
        /// </remarks>
        [TestMethod]
        [OpenBugs]
        public void Conversion_UInt64AtOrAboveTwoTo63ToFloat64_RoundsOnceLikeNumPy()
        {
            var log = new CaseLog();
            LikeNumPy(new ulong[] { 9223372036854776833, 9223372036854777857, 9223372036854775809 },
                Doubles(0x43E0000000000001, 0x43E0000000000001, 0x43E0000000000000), "uint64 ≥ 2^63 → float64", log);
            AssertNoFailures(log, "uint64 ≥ 2^63 → float64", minimumCases: 9);
        }

        /// <summary>
        ///     A signalling NaN converted into or out of float16 stays SIGNALLING with its payload, as NumPy's bit-level
        ///     <c>npy_halfbits_to_floatbits</c> / <c>npy_floatbits_to_halfbits</c> / <c>npy_doublebits_to_halfbits</c>
        ///     keep it — on every route. Fails today on ToArray's scalar routes only: the small non-contiguous views and
        ///     the 0-d elements (contiguous and large views match NumPy).
        /// </summary>
        /// <exception cref="AssertFailedException">A route set the quiet bit, or lost the payload (on the scalar routes: every signalling input).</exception>
        /// <remarks>
        ///     <para>
        ///         Root cause: those routes convert one element at a time through <c>NDIterCasting.ConvertValue</c>, which
        ///         reads a float16 through .NET's <c>(double)Half</c> and writes one through <c>(Half)double</c> — both set
        ///         the NaN quiet bit (and a signalling NaN whose payload does not survive the narrowing becomes the canonical
        ///         0x7E00, where NumPy keeps it signalling as 0x7C01). astype's cast kernels do the bit conversion NumPy does,
        ///         so ToArray breaks its own contract ("converts exactly as astype") on these values. The conversion matrix
        ///         quiets the signalling NaNs of every float16 conversion source until this is fixed.
        ///     </para>
        ///     <para>
        ///         Probed (numpy 2.4.2): float16 [0x7d01, 0xfd01, 0x7e01, 0x7c01] → float32 [0x7fa02000, 0xffa02000,
        ///         0x7fc02000, 0x7f802000]; float32 [0x7fa00001, 0xffa00001, 0x7fc0beef, 0x7f800001] → float16 [0x7d00,
        ///         0xfd00, 0x7e05, 0x7c01]; float64 [0x7ff4000000000001, 0xfff4000000000001, 0x7ff8deadbeef0001,
        ///         0x7ff0000000000001] (and complex128 with those real parts) → float16 [0x7d00, 0xfd00, 0x7e37, 0x7c01].
        ///         Remove <c>[OpenBugs]</c> when the scalar converter keeps signalling NaNs as NumPy does.
        ///     </para>
        /// </remarks>
        [TestMethod]
        [OpenBugs]
        public void Conversion_SignallingNaNIntoOrOutOfFloat16_StaysSignallingOnEveryRoute()
        {
            var log = new CaseLog();
            LikeNumPy(Halves(0x7D01, 0xFD01, 0x7E01, 0x7C01), Singles(0x7FA02000, 0xFFA02000, 0x7FC02000, 0x7F802000),
                "float16 signalling NaN → float32", log);
            LikeNumPy(Singles(0x7FA00001, 0xFFA00001, 0x7FC0BEEF, 0x7F800001), Halves(0x7D00, 0xFD00, 0x7E05, 0x7C01),
                "float32 signalling NaN → float16", log);
            double[] doubles = Doubles(0x7FF4000000000001, 0xFFF4000000000001, 0x7FF8DEADBEEF0001, 0x7FF0000000000001);
            LikeNumPy(doubles, Halves(0x7D00, 0xFD00, 0x7E37, 0x7C01), "float64 signalling NaN → float16", log);
            LikeNumPy(doubles.Select(d => new Complex(d, 0)).ToArray(), Halves(0x7D00, 0xFD00, 0x7E37, 0x7C01),
                "complex128 signalling NaN → float16", log);
            AssertNoFailures(log, "signalling NaNs into or out of float16", minimumCases: 4 * 12);
        }

        /// <summary>
        ///     A float16 signalling NaN converted to float64 or complex128 stays SIGNALLING with its payload, as NumPy's
        ///     <c>npy_halfbits_to_doublebits</c> keeps it. Fails today on EVERY route — not a ToArray inconsistency but an
        ///     astype parity bug ToArray inherits by contract: astype's own float16 → float64 / complex128 conversion sets
        ///     the quiet bit.
        /// </summary>
        /// <exception cref="AssertFailedException">A route set the quiet bit (today: every route, for every signalling input).</exception>
        /// <remarks>
        ///     Probed (numpy 2.4.2): float16 [0x7d01, 0xfd01, 0x7e01, 0x7c01] → float64 [0x7ff4040000000000,
        ///     0xfff4040000000000, 0x7ff8040000000000, 0x7ff0040000000000] (complex128: the same real parts, +0 imaginary);
        ///     NumSharp gives 0x7ffc040000000000 / 0xfffc040000000000 / … / 0x7ff8040000000000 through astype and ToArray
        ///     alike (float16 → float32 keeps the payload; the float32 → float64 widening then quiets it, where NumPy converts
        ///     float16 → float64 directly). Remove <c>[OpenBugs]</c> when astype keeps the NaN signalling.
        /// </remarks>
        [TestMethod]
        [OpenBugs]
        public void Conversion_SignallingFloat16NaNToFloat64_StaysSignallingLikeNumPy()
        {
            var log = new CaseLog();
            Half[] halves = Halves(0x7D01, 0xFD01, 0x7E01, 0x7C01);
            double[] numpy = Doubles(0x7FF4040000000000, 0xFFF4040000000000, 0x7FF8040000000000, 0x7FF0040000000000);
            LikeNumPy(halves, numpy, "float16 signalling NaN → float64", log);
            LikeNumPy(halves, numpy.Select(d => new Complex(d, 0)).ToArray(), "float16 signalling NaN → complex128", log);
            AssertNoFailures(log, "float16 signalling NaN → float64 / complex128", minimumCases: 2 * 12);
        }

        /// <summary>
        ///     A 0-d array yields a ONE-element <c>T[]</c> holding its value — a fresh scalar, a 0-d view into a larger
        ///     buffer (at a non-zero offset, and inside a reversed view), and converted.
        /// </summary>
        /// <exception cref="AssertFailedException">The result was not a one-element array of the right value.</exception>
        [TestMethod]
        public void ZeroDimensional_YieldsAOneElementArray()
        {
            using var scalar = NDArray.Scalar(3.5);
            scalar.ToArray<double>().Should().Equal(3.5);
            scalar.ToArray<int>().Should().Equal(3);

            using var m = np.arange(12.0).reshape(3, 4);
            using var element = m["1, 2"];
            element.ToArray<double>().Should().Equal(6.0);
            element.ToArray<long>().Should().Equal(6L);

            using var reversed = m["::-1, ::-1"];
            using var inner = reversed["1, 2"];
            // m[::-1, ::-1][1, 2] is m[1, 1].
            inner.ToArray<double>().Should().Equal(5.0);

            using var flag = NDArray.Scalar(true);
            flag.ToArray<bool>().Should().Equal(true);
            flag.ToArray<double>().Should().Equal(1.0);

            using var z = NDArray.Scalar(new Complex(-2.5, 4));
            z.ToArray<Complex>().Should().Equal(new Complex(-2.5, 4));
            z.ToArray<double>().Should().Equal(-2.5);

            using var d = NDArray.Scalar(12.75m);
            d.ToArray<decimal>().Should().Equal(12.75m);
            d.ToArray<short>().Should().Equal((short)12);
        }

        /// <summary>
        ///     An empty array — owning or a view, of any rank, same dtype or converted — yields the shared
        ///     <see cref="Array.Empty{T}"/> instance, for every dtype.
        /// </summary>
        /// <exception cref="AssertFailedException">An empty source returned something other than the shared empty array.</exception>
        /// <remarks>
        ///     The documented return: an empty result allocates nothing. Both the same-dtype path and the conversion path
        ///     are asserted, for owning empties of three ranks and for empty views (an empty slice, an empty slice of a
        ///     reversed view, an empty column block of a transposed view, and an empty broadcast).
        /// </remarks>
        [TestMethod]
        public void EmptyArray_ReturnsTheSharedEmptyArray()
        {
            var log = new CaseLog();
            foreach (NPTypeCode dtype in AllDtypes)
                Visit(dtype, new EmptyVisitor(log));
            AssertNoFailures(log, "empty sources", minimumCases: 15 * 8);
        }

        /// <summary>
        ///     The result is a private copy: every call allocates its own array, writing it never reaches the source, and
        ///     writing the source never reaches an earlier result — on the inline tiny copy, the block copy, a transposed
        ///     fill, a scalar broadcast and a conversion.
        /// </summary>
        /// <exception cref="AssertFailedException">Two calls shared an array, or a write crossed between result and source.</exception>
        [TestMethod]
        public void Result_IsAFreshCopy_ThatSharesNoMemoryWithTheSource()
        {
            using var a = np.arange(12.0).reshape(3, 4);
            double[] first = a.ToArray<double>(), second = a.ToArray<double>();
            ReferenceEquals(first, second).Should().BeFalse("every call allocates its own result");
            first[6] = -1.0;
            a.GetDouble(1, 2).Should().Be(6.0, "writing the result must not reach the source");
            a.SetDouble(100.0, 0, 0);
            second[0].Should().Be(0.0, "writing the source must not reach an earlier result");

            // The inline copy (at most 64 bytes, no block copy involved).
            using var tiny = np.arange(3.0);
            double[] t = tiny.ToArray<double>();
            t[1] = -1.0;
            tiny.GetDouble(1).Should().Be(1.0, "the inline tiny copy is a private copy");

            // A transposed fill.
            using var transposed = a.T;
            double[] tr = transposed.ToArray<double>();
            tr[1] = -2.0;
            a.GetDouble(1, 0).Should().Be(4.0, "a transposed view's result is a private copy");

            // A scalar broadcast (a fill from one element).
            using var one = np.arange(5.0);
            using var broadcast = np.broadcast_to(one["2"], new Shape(4, 3));
            double[] b = broadcast.ToArray<double>();
            b[5] = -3.0;
            one.GetDouble(2).Should().Be(2.0, "a broadcast's result is a private copy");
            b.Count(v => v == 2.0).Should().Be(11, "only the written element changed");

            // A conversion.
            using var ints = np.arange(6);
            float[] f = ints.ToArray<float>();
            f[0] = 42f;
            ints.GetAtIndex<long>(0).Should().Be(0L, "a converted result is a private copy");
        }

        /// <summary>
        ///     An element type that is not a NumSharp dtype has no astype conversion to perform, so the call refuses it up
        ///     front with <see cref="NotSupportedException"/> naming the type — for every dtype and layout, an empty and a
        ///     0-d source included (refused before anything is read or allocated).
        /// </summary>
        /// <exception cref="AssertFailedException">A non-dtype element type did not throw as specified.</exception>
        [TestMethod]
        public void NonDtypeElementType_ThrowsNotSupportedException()
        {
            foreach (NPTypeCode dtype in AllDtypes)
            {
                using var line = RawPattern(dtype, 24, 0x5EED_0100);
                using var grid = line.reshape(4, 6);
                var sources = new (string Name, NDArray Array)[]
                {
                    ("contiguous", line), ("transposed", grid.T), ("stepped", line["::-3"]),
                    ("0-d", line["5"]), ("empty", line["7:7"]),
                };
                foreach (var (name, source) in sources)
                {
                    Action guid = () => source.ToArray<Guid>();
                    guid.Should().Throw<NotSupportedException>($"{dtype} {name}").WithMessage("*Guid is not a NumSharp dtype*");
                    Action dateTime = () => source.ToArray<DateTime>();
                    dateTime.Should().Throw<NotSupportedException>($"{dtype} {name}").WithMessage("*DateTime is not a NumSharp dtype*");
                    Action pointer = () => source.ToArray<nint>();
                    pointer.Should().Throw<NotSupportedException>($"{dtype} {name}");
                }
                // The views (every source but the line itself, which its using disposes).
                foreach (var (_, source) in sources)
                    if (!ReferenceEquals(source, line))
                        source.Dispose();
            }
        }

        /// <summary>
        ///     A view of more than <see cref="int.MaxValue"/> elements — which no .NET array can index — is refused with
        ///     <see cref="InvalidOperationException"/> naming its size, before anything is allocated, same dtype or
        ///     converted, 1-D or 2-D.
        /// </summary>
        /// <exception cref="AssertFailedException">The wrong exception type or text surfaced.</exception>
        /// <remarks>A broadcast view has that many elements without owning that much memory, so the refusal costs nothing.</remarks>
        [TestMethod]
        public void MoreThanIntMaxValueElements_ThrowsInvalidOperationException()
        {
            using var one = NDArray.Scalar(7.0);
            using var line = np.broadcast_to(one, new Shape((long)int.MaxValue + 1));
            Action same = () => line.ToArray<double>();
            same.Should().Throw<InvalidOperationException>().WithMessage("*2147483648 exceeds int.MaxValue*");
            Action converted = () => line.ToArray<float>();
            converted.Should().Throw<InvalidOperationException>().WithMessage("*2147483648 exceeds int.MaxValue*");

            using var plane = np.broadcast_to(one, new Shape(65536, 32769));
            Action samePlane = () => plane.ToArray<double>();
            samePlane.Should().Throw<InvalidOperationException>().WithMessage("*2147549184 exceeds int.MaxValue*");
            Action convertedPlane = () => plane.ToArray<int>();
            convertedPlane.Should().Throw<InvalidOperationException>().WithMessage("*2147549184 exceeds int.MaxValue*");
        }

        // ============================================================================================== catalogs

        /// <summary>
        ///     The same-dtype catalog for one element type: every shape size and memory layout that selects a distinct
        ///     route of ToArray's exact copy, every size threshold from both sides, and every ragged register-block tail.
        /// </summary>
        /// <typeparam name="T">The element type (the dtype's: no conversion).</typeparam>
        /// <param name="log">Collects the cases and their failures.</param>
        /// <remarks>
        ///     Small and mid-size cases are views of four shared patterns (a 15 000-element line, a 17 × 17 grid, a
        ///     1 100-element cube and a 150-element hyper-cube pool) and of one array over managed memory, each checked
        ///     for being untouched after every case; every large case gets a pattern of its own. The sizes that matter are all functions of the element width:
        ///     the inline copy ends at 64 bytes, the zeroed allocation at 8 KB, the streamed routes start at a mebibyte and
        ///     the pre-fault at 85 000 bytes, so each is placed per width rather than by element count.
        /// </remarks>
        private static void RunSameDtypeCatalog<T>(CaseLog log) where T : unmanaged
        {
            NPTypeCode dtype = InfoOf<T>.NPTypeCode;
            int size = Unsafe.SizeOf<T>();
            string tag = dtype.ToString();

            // ---- 1-D: the hot frame, strided lines, and the degenerate shapes --------------------------------------
            using (var line = RawPattern(dtype, 15_000, 0x5EED_0001))
            {
                var c = new CaseRunner<T>(line, log, $"{tag} 1-D");

                // Contiguous from the logical start: every byte count of the inline copy (1 … 64 bytes and just past,
                // covering its 1 / 2–3 / 4–7 / 8–15 / 16–32 / 33–64-byte move shapes), the block copy, both sides of the
                // 8 KB zeroed / uninitialized allocation switch, and an odd size past it.
                for (long n = 1; n <= 64 / size + 2; n++)
                    c.Check($"[0:{n}]", () => line[$"0:{n}"]);
                foreach (long n in new long[] { 100, 1000, 8192 / size - 1, 8192 / size, 8192 / size + 1, 4097 })
                    c.Check($"[0:{n}]", () => line[$"0:{n}"]);
                // Contiguous at an offset (the view's buffer address moves; the copy must start at ITS element 0), at
                // phases that misalign every vector width, and an owning copy.
                foreach (long off in new long[] { 1, 3, 7 })
                    foreach (long n in new long[] { 1, 7, 33, 1000 })
                        c.Check($"[{off}:{off + n}]", () => line[$"{off}:{off + n}"]);
                c.Check("owning copy of [0:5000]", () => line["0:5000"].copy());

                // Contiguous at a NON-ZERO Shape offset: a slice re-seats its buffer address and keeps offset 0, so only
                // an np.split / np.unstack child reaches the frames' "address + offset" term — at every copy size.
                foreach (long n in new long[] { 1, 2, 3, 5, 8, 9, 16, 17, 33, 64 / size, 64 / size + 1, 1000, 8192 / size + 1, 4097 })
                    c.Check($"split child [5:{5 + n}]", () => SplitChild(line, 5, n));
                c.Check("split child rows [4:6] of (6, 5)", () => np.split(Grid(line, 6, 5), new[] { 4 })[1]);
                c.Check("unstack child [2] of (4, 6)", () => np.unstack(Grid(line, 4, 6))[2]);
                c.Check("split child [7:47][::-1]", () => SplitChild(line, 7, 40)["::-1"]);
                c.Check("split child [11:83] as (8, 9).T", () => SplitChild(line, 11, 72).reshape(8, 9).T);
                c.Check("split child [7:47][3] (0-d)", () => SplitChild(line, 7, 40)["3"]);
                c.Check("split child [9:10] broadcast (4, 3)", () => np.broadcast_to(SplitChild(line, 9, 1), new Shape(4, 3)));

                // Strided lines: gathers (any stride), AVX2 deinterleave (stride 2), vector reverse (stride −1), and the
                // plain walk, at lengths around every vector width; 1- and 2-byte walks of 2 048+ elements go through
                // the iterator's subword kernels, so both sides of that threshold are listed.
                foreach (long step in new long[] { 2, 3, 7, -1, -2, -3 })
                    foreach (long n in new long[] { 1, 2, 3, 5, 8, 9, 16, 17, 33, 100, 2049 })
                        c.Check($"step {step} n={n}", () => Stepped(line, step, n, lead: 1 + n % 5));
                foreach (long step in new long[] { 2, -1 })
                    foreach (long n in new long[] { 2047, 2048 })
                        c.Check($"step {step} n={n}", () => Stepped(line, step, n, lead: 2));
                foreach (long n in new long[] { 2, 5, 14 })
                    c.Check($"step 1000 n={n}", () => Stepped(line, 1000, n, lead: 3));

                // 0-d: a view at a non-zero offset, a view inside a reversed view, an owning 0-d copy, a fresh scalar.
                c.Check("0-d [3]", () => line["3"]);
                c.Check("0-d [::-1][2]", () => line["::-1"]["2"]);
                c.Check("0-d owning copy", () => line["3"].copy());
                c.Check("0-d NDArray.Scalar", () => NDArray.Scalar(line.GetAtIndex<T>(9)));

                // Empty: owning (three ranks) and views (an empty slice, of a reversed view, of a stepped view, an empty
                // broadcast).
                c.Check("empty owning (0)", () => np.empty(new Shape(0), dtype));
                c.Check("empty owning (0, 5)", () => np.empty(new Shape(0, 5), dtype));
                c.Check("empty owning (4, 0)", () => np.empty(new Shape(4, 0), dtype));
                c.Check("empty owning (2, 0, 3)", () => np.empty(new Shape(2, 0, 3), dtype));
                c.Check("empty [5:5]", () => line["5:5"]);
                c.Check("empty [::-1][0:0]", () => line["::-1"]["0:0"]);
                c.Check("empty [::2][3:3]", () => line["::2"]["3:3"]);
                c.Check("empty broadcast (0, 4)", () => np.broadcast_to(line["0:1"], new Shape(0, 4)));

                // One element through a non-contiguous route: stepped, reversed, a broadcast of one.
                c.Check("one element [2:1000:1000]", () => line["2:1000:1000"]);
                c.Check("one element [::-1][0:1]", () => line["::-1"]["0:1"]);
                c.Check("one element broadcast (1,)", () => np.broadcast_to(line["3:4"], new Shape(1)));

                // Single-value broadcasts (every stride 0: one fill), of every rank.
                c.Check("scalar broadcast (7,)", () => np.broadcast_to(line["3"], new Shape(7)));
                c.Check("scalar broadcast (5, 9)", () => np.broadcast_to(line["3"], new Shape(5, 9)));
                c.Check("scalar broadcast (2, 3, 4)", () => np.broadcast_to(line["3"], new Shape(2, 3, 4)));
                c.Check("scalar broadcast (3, 1, 5)", () => np.broadcast_to(line["3"], new Shape(3, 1, 5)));
                c.Check("one-element broadcast (4097,)", () => np.broadcast_to(line["3:4"], new Shape(4097)));
                c.Check("one-element broadcast (2, 1)", () => np.broadcast_to(line["3:4"], new Shape(2, 1)));

                // Other broadcasts: a strided row, a strided column, reversed / transposed broadcasts, leading axes.
                c.Check("row broadcast of [0:40:2] → (7, 20)", () => np.broadcast_to(line["0:40:2"], new Shape(7, 20)));
                c.Check("column broadcast of [0:14:2] → (7, 20)", () => np.broadcast_to(np.expand_dims(line["0:14:2"], 1), new Shape(7, 20)));
                c.Check("row broadcast (7, 20)[:, ::-1]", () => np.broadcast_to(line["0:20"], new Shape(7, 20))[":, ::-1"]);
                c.Check("row broadcast (7, 20).T", () => np.broadcast_to(line["0:20"], new Shape(7, 20)).T);
                c.Check("column broadcast (7, 20).T", () => np.broadcast_to(np.expand_dims(line["0:7"], 1), new Shape(7, 20)).T);
                c.Check("broadcast (6,) → (2, 3, 6)", () => np.broadcast_to(line["0:6"], new Shape(2, 3, 6)));

                // Views of views, and 2-D views of the line with non-unit strides on both axes.
                c.Check("[::2][::-3]", () => line["::2"]["::-3"]);
                c.Check("[100:10:-3][::2]", () => line["100:10:-3"]["::2"]);
                c.Check("(8, 9)[1:7, ::-1][::2, 1:]", () => Grid(line, 8, 9)["1:7, ::-1"]["::2, 1:"]);
                c.Check("(8, 9).T[::-1][1:, ::2]", () => Grid(line, 8, 9).T["::-1"]["1:, ::2"]);
                c.Check("(8, 9)[5:1:-1, 7:2:-2]", () => Grid(line, 8, 9)["5:1:-1, 7:2:-2"]);

                // 2-D forms beyond the tail sweep, on three shapes: (13, 17) and (8, 9) small, and (70, 150) whose
                // transpose spans 18 whole 8-row strips + 6 ragged rows and a full 64-column band + a 6-column band.
                foreach (var (r, cols) in new (long, long)[] { (13, 17), (8, 9), (70, 150) })
                {
                    string s = $"({r}, {cols})";
                    c.Check($"{s} C-contiguous", () => Grid(line, r, cols));
                    c.Check($"{s} owning copy", () => Grid(line, r, cols).copy());
                    c.Check($"{s} F-contiguous copy", () => np.asfortranarray(Grid(line, r, cols)));
                    c.Check($"{s}.T", () => Grid(line, r, cols).T);
                    c.Check($"{s}[::-1, :]", () => Grid(line, r, cols)["::-1, :"]);
                    c.Check($"{s}[::-1, ::-1]", () => Grid(line, r, cols)["::-1, ::-1"]);
                    c.Check($"{s}[::2, :]", () => Grid(line, r, cols)["::2, :"]);
                    c.Check($"{s}[::2, ::2]", () => Grid(line, r, cols)["::2, ::2"]);
                    c.Check($"{s}[1:, 1:]", () => Grid(line, r, cols)["1:, 1:"]);
                    c.Check($"{s}[1:-1, 2:-1]", () => Grid(line, r, cols)["1:-1, 2:-1"]);
                    c.Check($"{s}[:, 1]", () => Grid(line, r, cols)[":, 1"]);
                    c.Check($"{s}[:, 1:2]", () => Grid(line, r, cols)[":, 1:2"]);
                    c.Check($"{s}[1:2, :]", () => Grid(line, r, cols)["1:2, :"]);
                    c.Check($"{s}.T[:, ::-1]", () => Grid(line, r, cols).T[":, ::-1"]);
                    c.Check($"{s}.T[::-1]", () => Grid(line, r, cols).T["::-1"]);
                    c.Check($"{s}[:, ::2]", () => Grid(line, r, cols)[":, ::2"]);
                    c.Check($"{s}[:, ::3]", () => Grid(line, r, cols)[":, ::3"]);
                    c.Check($"{s}[:, ::-1]", () => Grid(line, r, cols)[":, ::-1"]);
                    c.Check($"{s} row broadcast", () => np.broadcast_to(Grid(line, r, cols)["0:1, :"], new Shape(r, cols)));
                    c.Check($"{s} column broadcast", () => np.broadcast_to(Grid(line, r, cols)[":, 0:1"], new Shape(r, cols)));
                    c.Check($"{s} row broadcast .T", () => np.broadcast_to(Grid(line, r, cols)["0:1, :"], new Shape(r, cols)).T);
                    c.Check($"{s} column broadcast .T", () => np.broadcast_to(Grid(line, r, cols)[":, 0:1"], new Shape(r, cols)).T);
                    c.Check($"{s} expand_dims(.T, 1)", () => np.expand_dims(Grid(line, r, cols).T, 1));
                }
            }

            // ---- 2-D tail sweep: every row count × column count the register blocks and bands treat differently -----
            // Rows / columns 1–3 (element moves), 4–7 (4-row bands and 5–7-column tails), 8 (one strip / block), 9, 12,
            // 13, 16, 17 (strips + ragged remainders ending in an overlapping block).
            long[] sweep = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17 };
            using (var grid = RawPattern(dtype, 17 * 17, 0x5EED_0002))
            {
                var c = new CaseRunner<T>(grid, log, $"{tag} 2-D sweep");
                foreach (long r in sweep)
                    foreach (long cols in sweep)
                    {
                        string s = $"({r}, {cols})";
                        c.Check($"{s} C-contiguous", () => Grid(grid, r, cols));
                        c.Check($"{s}.T", () => Grid(grid, r, cols).T);
                        c.Check($"{s} F-contiguous copy", () => np.asfortranarray(Grid(grid, r, cols)));
                        c.Check($"{s}[:, ::2]", () => Grid(grid, r, cols)[":, ::2"]);
                        c.Check($"{s}[:, ::3]", () => Grid(grid, r, cols)[":, ::3"]);
                        c.Check($"{s}[:, ::-1]", () => Grid(grid, r, cols)[":, ::-1"]);
                        c.Check($"{s}[::-1, :].T", () => Grid(grid, r, cols)["::-1, :"].T);
                        c.Check($"{s}[::2, :].T", () => Grid(grid, r, cols)["::2, :"].T);
                        c.Check($"{s}[:, ::2].T", () => Grid(grid, r, cols)[":, ::2"].T);
                    }
            }

            // ---- 3-D: every axis permutation (the outer-axis transpose and plane walks), steps, broadcasts ----------
            using (var cube = RawPattern(dtype, 1100, 0x5EED_0003))
            {
                var c = new CaseRunner<T>(cube, log, $"{tag} 3-D");
                int[][] permutations = { new[] { 0, 1, 2 }, new[] { 0, 2, 1 }, new[] { 1, 0, 2 }, new[] { 1, 2, 0 }, new[] { 2, 0, 1 }, new[] { 2, 1, 0 } };
                foreach (var (a, b, cc) in new (long, long, long)[] { (2, 3, 4), (4, 5, 6), (5, 8, 9), (3, 1, 7), (1, 6, 1), (6, 7, 25), (4, 4, 4), (8, 3, 5) })
                {
                    string s = $"({a}, {b}, {cc})";
                    foreach (int[] p in permutations)
                        c.Check($"{s} transpose({string.Join(",", p)})", () => np.transpose(Cube(cube, a, b, cc), p));
                    c.Check($"{s}[..., ::2]", () => Cube(cube, a, b, cc)["..., ::2"]);
                    c.Check($"{s}[:, ::-1, :]", () => Cube(cube, a, b, cc)[":, ::-1, :"]);
                    c.Check($"{s}[::2, :, 1:]", () => Cube(cube, a, b, cc)["::2, :, 1:"]);
                    c.Check($"{s}[:, 1:, ::-1]", () => Cube(cube, a, b, cc)[":, 1:, ::-1"]);
                    c.Check($"{s} middle-axis broadcast", () => np.broadcast_to(Cube(cube, a, b, cc)[":, 0:1, :"], new Shape(a, b, cc)));
                    c.Check($"{s}.T[..., ::2]", () => Cube(cube, a, b, cc).T["..., ::2"]);
                    c.Check($"{s} swapaxes(0, 2)[::-1]", () => np.swapaxes(Cube(cube, a, b, cc), 0, 2)["::-1"]);
                    c.Check($"{s} expand_dims(.T, 1)", () => np.expand_dims(Cube(cube, a, b, cc).T, 1));
                    c.Check($"{s} F-contiguous copy", () => np.asfortranarray(Cube(cube, a, b, cc)));
                    c.Check($"{s} moveaxis(0, -1)[::2]", () => np.moveaxis(Cube(cube, a, b, cc), 0, -1)["::2"]);
                    c.Check($"{s} one element [1:2, 0:1, 0:1]", () => Cube(cube, a, b, cc)[$"{a / 2}:{a / 2 + 1}, 0:1, 0:1"]);
                }
            }

            // ---- rank 4–6: permutations, steps and broadcasts the normalization and outer odometer must carry -------
            using (var hyper = RawPattern(dtype, 150, 0x5EED_0004))
            {
                var c = new CaseRunner<T>(hyper, log, $"{tag} N-D");
                NDArray H4() => hyper["0:120"].reshape(2, 3, 4, 5);
                NDArray H5() => hyper["0:144"].reshape(2, 3, 2, 3, 4);
                NDArray H6() => hyper["0:24"].reshape(2, 1, 2, 1, 2, 3);
                c.Check("4-D C-contiguous", H4);
                c.Check("4-D transpose(3,2,1,0)", () => np.transpose(H4(), new[] { 3, 2, 1, 0 }));
                c.Check("4-D transpose(0,2,3,1)", () => np.transpose(H4(), new[] { 0, 2, 3, 1 }));
                c.Check("4-D transpose(1,3,0,2)", () => np.transpose(H4(), new[] { 1, 3, 0, 2 }));
                c.Check("4-D transpose(2,0,3,1)", () => np.transpose(H4(), new[] { 2, 0, 3, 1 }));
                c.Check("4-D [:, ::-1, 1:, ::2]", () => H4()[":, ::-1, 1:, ::2"]);
                c.Check("4-D [..., ::-1]", () => H4()["..., ::-1"]);
                c.Check("4-D broadcast (2,1,4,5) → (2,3,4,5)", () => np.broadcast_to(H4()[":, 0:1"], new Shape(2, 3, 4, 5)));
                c.Check("5-D C-contiguous", H5);
                c.Check("5-D .T", () => H5().T);
                c.Check("5-D transpose(4,0,3,1,2)", () => np.transpose(H5(), new[] { 4, 0, 3, 1, 2 }));
                c.Check("5-D [:, :, ::-1, :, ::2]", () => H5()[":, :, ::-1, :, ::2"]);
                c.Check("6-D .T", () => H6().T);
                c.Check("6-D [:, :, ::-1, :, :, ::2]", () => H6()[":, :, ::-1, :, :, ::2"]);
            }

            // ---- an array over managed memory (np.frombuffer): a different storage owner, same routes ----------------
            using (var pattern = RawPattern(dtype, 3000, 0x5EED_0005))
            {
                byte[] managed = pattern.Unsafe.ReadOnlyBytes().ToArray();
                using var wrapped = np.frombuffer(managed, dtype);
                var c = new CaseRunner<T>(wrapped, log, $"{tag} frombuffer");
                c.Check("whole", () => wrapped[":"]);
                c.Check("[::-1]", () => wrapped["::-1"]);
                c.Check("[1::3]", () => wrapped["1::3"]);
                c.Check("(50, 60).T", () => wrapped.reshape(50, 60).T);
            }

            // ---- large sources: the mebibyte and large-object thresholds of every route, per element width ---------
            long mebElems = Mebibyte / size;
            // At least a mebibyte, contiguous: the block copy into an uninitialized large object (owning, at an offset);
            // reversed runs streamed by whole lines (at two phases); a stride-2 walk; a mebibyte-sized fill.
            Large<T>(log, tag, "line", mebElems + 13, 0x5EED_0010,
                ("owning", b => b[":"]),
                ("[3:]", b => b["3:"]),
                ("[::-1]", b => b["::-1"]),
                ($"[{mebElems + 9}::-1]", b => b[$"{mebElems + 9}::-1"]),
                ("[::2]", b => b["::2"]),
                ("scalar broadcast", b => np.broadcast_to(b["7"], new Shape(mebElems + 3))));
            // Transposes of a mebibyte and more for a float64-width element: whole-line streaming (columns % 8 == 0),
            // 32-byte streaming (columns % 8 == 4) and bands (otherwise); every other width takes the bands — all after
            // the large-object pre-fault.
            foreach (var (r, cols) in new (long, long)[] { (368, 360), (364, 362), (365, 363) })
                Large<T>(log, tag, $"({r}, {cols})", r * cols, 0x5EED_0011, ($"({r}, {cols}).T", b => b.reshape(r, cols).T));
            // A transpose past the large-object size but under a mebibyte: bands behind the pre-fault.
            long side = (long)System.Math.Ceiling(System.Math.Sqrt(90_000.0 / size));
            Large<T>(log, tag, $"({side}, {side + 3})", side * (side + 3), 0x5EED_0012,
                ($"({side}, {side + 3}).T", b => b.reshape(side, side + 3).T),
                ($"({side}, {side + 3})[:, ::-1].T", b => b.reshape(side, side + 3)[":, ::-1"].T));
            // Reversed-row planes of a mebibyte and more: rows of at least 16 KB stream one by one from registers,
            // shorter rows through the L1 scratch.
            long longCols = 16_384 / size + 3, longRows = Mebibyte / (longCols * size) + 2;
            Large<T>(log, tag, $"({longRows}, {longCols})", longRows * longCols, 0x5EED_0013,
                ($"({longRows}, {longCols})[:, ::-1]", b => b.reshape(longRows, longCols)[":, ::-1"]));
            long shortRows = Mebibyte / (37 * size) + 2;
            Large<T>(log, tag, $"({shortRows}, 37)", shortRows * 37, 0x5EED_0014,
                ($"({shortRows}, 37)[:, ::-1]", b => b.reshape(shortRows, 37)[":, ::-1"]));
            // Rank-3 permutations of a large object: the outer-axis transpose and the transposing plane walk, pre-faulted.
            Large<T>(log, tag, "(40, 30, 20)", 24_000, 0x5EED_0015,
                ("transpose(2,0,1)", b => np.transpose(b.reshape(40, 30, 20), new[] { 2, 0, 1 })),
                ("swapaxes(0,2)", b => np.swapaxes(b.reshape(40, 30, 20), 0, 2)),
                ("transpose(1,2,0)", b => np.transpose(b.reshape(40, 30, 20), new[] { 1, 2, 0 })));
            // 2-D stride-2 and reversed rows of 2 048+ elements under a mebibyte: the iterator's subword kernels for 1-
            // and 2-byte elements, the deinterleave / vector reverse for 4 and 8.
            Large<T>(log, tag, "(64, 130)", 64 * 130, 0x5EED_0016,
                ("[:, ::2]", b => b.reshape(64, 130)[":, ::2"]),
                ("[:, ::-1]", b => b.reshape(64, 130)[":, ::-1"]),
                ("[::2, ::-2]", b => b.reshape(64, 130)["::2, ::-2"]));
            // Large-object broadcasts: one row repeated (row copies), one column repeated (row fills).
            Large<T>(log, tag, "broadcast source", 1000, 0x5EED_0017,
                ("row broadcast (200, 1000)", b => np.broadcast_to(b, new Shape(200, 1000))),
                ("column broadcast (1000, 200)", b => np.broadcast_to(np.expand_dims(b, 1), new Shape(1000, 200))));
        }

        /// <summary>
        ///     The conversion catalog for one (source dtype, target) pair: every route a converting call can take, each
        ///     compared with astype.
        /// </summary>
        /// <typeparam name="T">The target element type (a dtype other than <paramref name="from"/>).</typeparam>
        /// <param name="from">The source dtype.</param>
        /// <param name="log">Collects the cases and their failures.</param>
        /// <remarks>
        ///     The small and mid-size views come from one 12 000-element pattern whose values suit the target
        ///     (<see cref="ConversionPattern"/>). One transposed large-object result per source dtype (its target chosen
        ///     so that the 15 sources reach the 15 targets once each) reaches the iterator's pre-faulted cast copy, and a
        ///     mebibyte-sized contiguous source for the four targets this file's AVX2 converters and line converters
        ///     write (float64, float32, int64, int32).
        /// </remarks>
        private static void RunConversionCatalog<T>(NPTypeCode from, CaseLog log) where T : unmanaged
        {
            NPTypeCode to = InfoOf<T>.NPTypeCode;
            int size = Unsafe.SizeOf<T>();
            string tag = $"{from} → {to}";
            ulong seed = 0xC0DE_0000UL + (ulong)from * 31 + (ulong)to;

            using (var line = ConversionPattern(from, to, 12_000, seed))
            {
                var c = new CaseRunner<T>(line, log, tag);

                // Contiguous: one element (the scalar converter), then sizes around one 32-byte destination vector (the
                // AVX2 converters start there), the kernel, and past the iterator threshold; at an offset; 2-D and 3-D;
                // an owning copy.
                foreach (long n in new long[] { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1000, 1025, 5000 })
                    c.Check($"contiguous [0:{n}]", () => line[$"0:{n}"]);
                foreach (long n in new long[] { 2, 9, 33, 1000 })
                    c.Check($"contiguous [3:{3 + n}]", () => line[$"3:{3 + n}"]);
                // A NON-ZERO Shape offset (a slice re-seats its address and keeps offset 0; a split child keeps the
                // offset): both of the lean frame's branches — the AVX2 converters and the remembered kernel — and the
                // general frame add it to the storage address.
                foreach (long n in new long[] { 2, 3, 9, 16, 17, 33, 1000, 5000 })
                    c.Check($"split child [3:{3 + n}]", () => SplitChild(line, 3, n));
                c.Check("split child rows [4:6] of (6, 7)", () => np.split(Grid(line, 6, 7), new[] { 4 })[1]);
                c.Check("split child [5:45][::-1]", () => SplitChild(line, 5, 40)["::-1"]);
                c.Check("split child [5:45][2] (0-d)", () => SplitChild(line, 5, 40)["2"]);
                c.Check("contiguous (7, 9)", () => Grid(line, 7, 9));
                c.Check("contiguous (3, 4, 5)", () => Cube(line, 3, 4, 5));
                c.Check("owning copy", () => line["0:12000"].copy());

                // 0-d and one element (the scalar converter), empty (the shared empty array).
                c.Check("0-d [3]", () => line["3"]);
                c.Check("0-d owning copy", () => line["3"].copy());
                c.Check("one element [5:6]", () => line["5:6"]);
                c.Check("one element [2:1000:1000]", () => line["2:1000:1000"]);
                c.Check("empty [5:5]", () => line["5:5"]);
                c.Check("empty owning (3, 0)", () => np.empty(new Shape(3, 0), from));

                // Non-contiguous, at most 1 024 elements (the scalar odometer), the last one exactly 1 024.
                c.Check("(7, 9).T", () => Grid(line, 7, 9).T);
                c.Check("[0:33:2]", () => line["0:33:2"]);
                c.Check("[19::-1]", () => line["19::-1"]);
                c.Check("(5, 6)[:, ::-1]", () => Grid(line, 5, 6)[":, ::-1"]);
                c.Check("(8, 9)[1:-1, 2:-1]", () => Grid(line, 8, 9)["1:-1, 2:-1"]);
                c.Check("(3, 4, 5) transpose(2,0,1)", () => np.transpose(Cube(line, 3, 4, 5), new[] { 2, 0, 1 }));
                c.Check("row broadcast (4, 5)", () => np.broadcast_to(line["0:5"], new Shape(4, 5)));
                c.Check("column broadcast (5, 4)", () => np.broadcast_to(np.expand_dims(line["0:5"], 1), new Shape(5, 4)));
                c.Check("scalar broadcast (6,)", () => np.broadcast_to(line["7"], new Shape(6)));
                c.Check("(32, 32).T (1 024)", () => Grid(line, 32, 32).T);

                // Non-contiguous, more than 1 024 elements (the iterator's cast copy), the first one exactly 1 025.
                c.Check("(25, 41).T (1 025)", () => Grid(line, 25, 41).T);
                c.Check("(40, 50).T", () => Grid(line, 40, 50).T);
                c.Check("[0:5001:2]", () => line["0:5001:2"]);
                c.Check("[2999::-1]", () => line["2999::-1"]);
                c.Check("(10, 12, 14) transpose(2,0,1)", () => np.transpose(Cube(line, 10, 12, 14), new[] { 2, 0, 1 }));
                c.Check("row broadcast (6, 400)", () => np.broadcast_to(line["0:400"], new Shape(6, 400)));
                c.Check("column broadcast (400, 6)", () => np.broadcast_to(np.expand_dims(line["0:400"], 1), new Shape(400, 6)));
                c.Check("scalar broadcast (1500,)", () => np.broadcast_to(line["7"], new Shape(1500)));
                c.Check("(60, 90)[::-1, ::3]", () => Grid(line, 60, 90)["::-1, ::3"]);
            }

            // A transposed large-object result: the iterator's cast copy behind the address-order pre-fault. One target
            // per source (the next dtype in the list) keeps it to 15 cases while every target and source is reached.
            int fromIndex = Array.IndexOf(AllDtypes, from);
            if (AllDtypes[(fromIndex + 1) % AllDtypes.Length] == to)
            {
                long side = (long)System.Math.Ceiling(System.Math.Sqrt(85_000.0 / size)) + 1;
                using var big = ConversionPattern(from, to, side * (side + 1), seed + 1);
                new CaseRunner<T>(big, log, tag).Check($"large-object ({side}, {side + 1}).T", () => big.reshape(side, side + 1).T);
            }

            // At least a mebibyte of contiguous result for the four targets with dedicated converters: the line
            // converters' seven pairs, the AVX2 converters' other pairs in the general frame, the kernel for the rest.
            if (to is NPTypeCode.Double or NPTypeCode.Single or NPTypeCode.Int64 or NPTypeCode.Int32)
            {
                long n = Mebibyte / size + 7;
                using var big = ConversionPattern(from, to, n + 3, seed + 2);
                var r = new CaseRunner<T>(big, log, tag);
                r.Check($"mebibyte [0:{n}]", () => big[$"0:{n}"]);
                r.Check($"mebibyte [3:{n + 3}]", () => big[$"3:{n + 3}"]);
                r.Check($"mebibyte split child [3:{n + 3}]", () => SplitChild(big, 3, n));
            }
        }

        // ============================================================================================== helpers

        /// <summary>
        ///     Checks representative values against NumPy's output on every route a converting call can take: the input
        ///     laid out contiguous, at an offset, reversed, stepped, transposed, broadcast, element by element as 0-d
        ///     views, and tiled past the size thresholds (a mebibyte of result for a 4- or 8-byte target, else past the
        ///     iterator threshold) both contiguous and stepped.
        /// </summary>
        /// <typeparam name="TFrom">The source element type.</typeparam>
        /// <typeparam name="TTo">The target element type.</typeparam>
        /// <param name="input">The source values in logical order.</param>
        /// <param name="numpy">NumPy's converted values for <paramref name="input"/>, element for element.</param>
        /// <param name="what">Description for failure messages.</param>
        /// <param name="log">Collects the cases and their failures.</param>
        /// <exception cref="ArgumentException"><paramref name="input"/> and <paramref name="numpy"/> differ in length.</exception>
        /// <remarks>
        ///     Every layout's logical C-order elements are <paramref name="input"/> (or a repetition of it), so NumPy's
        ///     answer is known for each: the stepped layout interleaves each value with a copy of itself and keeps every
        ///     second one, the transposed layout stores the values as the columns of an (n, 2) array so its transpose's
        ///     rows are two copies of the input, and the broadcast repeats the input as three rows.
        /// </remarks>
        private static void LikeNumPy<TFrom, TTo>(TFrom[] input, TTo[] numpy, string what, CaseLog log)
            where TFrom : unmanaged where TTo : unmanaged
        {
            if (input.Length != numpy.Length)
                throw new ArgumentException($"{what}: {input.Length} inputs but {numpy.Length} expected values.", nameof(numpy));
            int n = input.Length;
            TFrom[] pairs = Interleave(input);

            ExpectBytes(log, what, "contiguous", () => np.array(input), numpy);
            ExpectBytes(log, what, "offset [3:]", () => np.array(Repeat(input, 3)[..3].Concat(input).ToArray())["3:"], numpy);
            // Enumerable.Reverse spelled out: under C# 14's first-class spans `input.Reverse()` binds to the in-place
            // MemoryExtensions.Reverse(Span<T>), which returns nothing and would reverse the input itself.
            ExpectBytes(log, what, "reversed [::-1]", () => np.array(Enumerable.Reverse(input).ToArray())["::-1"], numpy);
            ExpectBytes(log, what, "stepped [::2]", () => np.array(pairs)["::2"], numpy);
            ExpectBytes(log, what, "transposed (n, 2).T", () => np.array(pairs).reshape(n, 2).T, Repeat(numpy, 2));
            ExpectBytes(log, what, "row broadcast (3, n)", () => np.broadcast_to(np.array(input), new Shape(3, n)), Repeat(numpy, 3));
            // Every element alone, as a 0-d view at its own offset: the one-element scalar converter.
            for (int i = 0; i < n; i++)
                ExpectBytes(log, what, $"0-d [{i}]", () => np.array(input)[$"{i}"], numpy[i..(i + 1)]);

            // Past the thresholds: a mebibyte of 4- or 8-byte result reaches the line / AVX2 converters, anything past
            // 1 024 elements the kernel (contiguous) or the iterator's cast copy (stepped).
            int size = Unsafe.SizeOf<TTo>();
            long target = size is 4 or 8 ? Mebibyte / size + 7 : 2100;
            int reps = (int)((target + n - 1) / n);
            TFrom[] tiled = Repeat(input, reps);
            TTo[] tiledNumPy = Repeat(numpy, reps);
            ExpectBytes(log, what, $"large contiguous ({tiled.Length})", () => np.array(tiled), tiledNumPy);
            ExpectBytes(log, what, $"large stepped ({tiled.Length})", () => np.array(Interleave(tiled))["::2"], tiledNumPy);
        }

        /// <summary>
        ///     Builds a source, converts it with ToArray, and records a failure unless the result's bytes equal the
        ///     expected elements' bytes (a failure to build or convert is recorded too, with its exception).
        /// </summary>
        /// <typeparam name="TTo">The requested element type.</typeparam>
        /// <param name="log">Collects the case and its failure.</param>
        /// <param name="what">The conversion under test.</param>
        /// <param name="layout">The layout of the source.</param>
        /// <param name="make">Builds the source (disposed here).</param>
        /// <param name="expected">The expected elements in logical C order.</param>
        private static void ExpectBytes<TTo>(CaseLog log, string what, string layout, Func<NDArray> make, TTo[] expected)
            where TTo : unmanaged
        {
            log.Cases++;
            string label = $"{what} — {layout}";
            try
            {
                using NDArray source = make();
                TTo[] got = source.ToArray<TTo>();
                string difference = FirstDifference(got, MemoryMarshal.AsBytes(expected.AsSpan()).ToArray(), byValue: false);
                if (difference != null)
                    log.Fail($"{label}: {difference}");
            }
            catch (Exception e)
            {
                log.Fail($"{label}: threw {e.GetType().Name}: {e.Message}");
            }
        }

        /// <summary>
        ///     The oracle: the logical C-order bytes of <paramref name="view"/>, read straight from its buffer through its
        ///     own dimensions, element strides and offset by a plain odometer, one element at a time — no NumSharp copy,
        ///     iterator or kernel is involved, so it shares no code with <see cref="NDArray.ToArray{T}"/> beyond the
        ///     view's own metadata.
        /// </summary>
        /// <param name="view">Any array or view: any rank, any strides (negative and broadcast zero included), 0-d or empty.</param>
        /// <returns><c>size × itemsize</c> bytes; logical element k at <c>[k · itemsize, (k + 1) · itemsize)</c>.</returns>
        private static unsafe byte[] LogicalBytes(NDArray view)
        {
            int itemSize = view.dtypesize;
            long count = view.size;
            var bytes = new byte[count * itemSize];
            if (count == 0)
                return bytes;
            Shape shape = view.Shape;
            long[] dims = shape.dimensions, strides = shape.strides;
            int nd = dims.Length;
            // Logical element 0 of ANY layout: the storage's base address plus the view's offset in elements.
            byte* first = view.Storage.Address + shape.offset * itemSize;
            var coord = new long[nd];
            long offset = 0;
            fixed (byte* start = bytes)
            {
                byte* d = start;
                for (long k = 0; k < count; k++, d += itemSize)
                {
                    CopyElement(first + offset * itemSize, d, itemSize);
                    // Odometer, last axis fastest: step that axis; an axis that wraps rewinds its whole run and carries.
                    // A 0-d view (nd = 0) has its one element and no axis to step.
                    for (int ax = nd - 1; ax >= 0; ax--)
                    {
                        offset += strides[ax];
                        if (++coord[ax] < dims[ax])
                            break;
                        offset -= strides[ax] * dims[ax];
                        coord[ax] = 0;
                    }
                }
            }
            // The raw pointer read the view's buffer: keep the view (and so its buffer) alive until the walk is done.
            GC.KeepAlive(view);
            return bytes;
        }

        /// <summary>Copies one element of <paramref name="itemSize"/> bytes (1, 2, 4, 8 or 16), any alignment.</summary>
        /// <param name="s">Source element.</param>
        /// <param name="d">Destination element.</param>
        /// <param name="itemSize">Element size in bytes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void CopyElement(byte* s, byte* d, int itemSize)
        {
            // Typed moves keep the oracle fast on the mebibyte cases (a million elements) without a copy call each.
            switch (itemSize)
            {
                case 1: *d = *s; break;
                case 2: Unsafe.WriteUnaligned(d, Unsafe.ReadUnaligned<ushort>(s)); break;
                case 4: Unsafe.WriteUnaligned(d, Unsafe.ReadUnaligned<uint>(s)); break;
                case 8: Unsafe.WriteUnaligned(d, Unsafe.ReadUnaligned<ulong>(s)); break;
                default: Buffer.MemoryCopy(s, d, itemSize, itemSize); break;
            }
        }

        /// <summary>
        ///     Describes the first element whose bytes (or, for a decimal compared by value, whose value) differ, or null
        ///     when the result holds exactly the expected elements.
        /// </summary>
        /// <typeparam name="T">The result's element type.</typeparam>
        /// <param name="got">The result.</param>
        /// <param name="want">The expected elements as raw bytes (element k at <c>[k · size, (k + 1) · size)</c>).</param>
        /// <param name="byValue">Compare decimals by value (a conversion's result: 1.0m and 1.00m are equal) instead of by bits.</param>
        /// <returns>Null on a match; else the length mismatch, or the first differing index with both elements' bytes.</returns>
        private static string FirstDifference<T>(T[] got, byte[] want, bool byValue) where T : unmanaged
        {
            int size = Unsafe.SizeOf<T>();
            if ((long)got.Length * size != want.Length)
                return $"length {got.Length} but {want.Length / size} elements were expected";
            ReadOnlySpan<byte> gotBytes = MemoryMarshal.AsBytes(got.AsSpan());
            if (!byValue && gotBytes.SequenceEqual(want))
                return null;
            ReadOnlySpan<T> wantElements = MemoryMarshal.Cast<byte, T>(want.AsSpan());
            for (int i = 0; i < got.Length; i++)
            {
                bool same = byValue && typeof(T) == typeof(decimal)
                    ? (decimal)(object)got[i] == (decimal)(object)wantElements[i]
                    : gotBytes.Slice(i * size, size).SequenceEqual(want.AsSpan(i * size, size));
                if (!same)
                    return $"element {i} (C order) is {got[i]} [0x{Convert.ToHexString(gotBytes.Slice(i * size, size))}] " +
                           $"but {wantElements[i]} [0x{Convert.ToHexString(want.AsSpan(i * size, size))}] was expected";
            }
            return null;
        }

        /// <summary>
        ///     A strided 1-D view of exactly <paramref name="n"/> elements with element stride <paramref name="step"/>,
        ///     starting <paramref name="lead"/> elements into the line (a descending view starts high enough to fit).
        /// </summary>
        /// <param name="line">The 1-D source.</param>
        /// <param name="step">The element stride (non-zero, either sign).</param>
        /// <param name="n">The element count.</param>
        /// <param name="lead">The first element's index for an ascending view; the lowest index read for a descending one.</param>
        /// <returns>A view of <paramref name="line"/>.</returns>
        private static NDArray Stepped(NDArray line, long step, long n, long lead)
        {
            long start = step > 0 ? lead : lead + (n - 1) * -step;
            return line[$"{start}::{step}"][$"0:{n}"];
        }

        /// <summary>
        ///     The C-contiguous view of elements [start, start + n) of a 1-D array as an <c>np.split</c> child: unlike the
        ///     equivalent slice (which re-seats its buffer address and keeps offset 0), it keeps the base address and
        ///     records <paramref name="start"/> as its <see cref="Shape"/> offset.
        /// </summary>
        /// <param name="line">The 1-D source.</param>
        /// <param name="start">The first element's index (the child's offset).</param>
        /// <param name="n">The element count.</param>
        /// <returns>A view of <paramref name="line"/>.</returns>
        private static NDArray SplitChild(NDArray line, long start, long n) => np.split(line, new[] { start, start + n })[1];

        /// <summary>The (rows, cols) C-contiguous view over the first rows · cols elements of a 1-D pattern.</summary>
        /// <param name="line">The 1-D source.</param>
        /// <param name="rows">Row count.</param>
        /// <param name="cols">Column count.</param>
        /// <returns>A view of <paramref name="line"/>.</returns>
        private static NDArray Grid(NDArray line, long rows, long cols) => line[$"0:{rows * cols}"].reshape(rows, cols);

        /// <summary>The (a, b, c) C-contiguous view over the first a · b · c elements of a 1-D pattern.</summary>
        /// <param name="line">The 1-D source.</param>
        /// <param name="a">Extent of axis 0.</param>
        /// <param name="b">Extent of axis 1.</param>
        /// <param name="c">Extent of axis 2.</param>
        /// <returns>A view of <paramref name="line"/>.</returns>
        private static NDArray Cube(NDArray line, long a, long b, long c) => line[$"0:{a * b * c}"].reshape(a, b, c);

        /// <summary>
        ///     Runs large same-dtype cases over a pattern of their own (created and disposed here): one runner, so every
        ///     view is checked against the pattern's untouched bytes.
        /// </summary>
        /// <typeparam name="T">The element type (the dtype's).</typeparam>
        /// <param name="log">Collects the cases and their failures.</param>
        /// <param name="tag">The dtype, for failure messages.</param>
        /// <param name="name">The pattern's description.</param>
        /// <param name="count">The pattern's element count.</param>
        /// <param name="seed">The pattern's seed.</param>
        /// <param name="views">The views to check, each built from the pattern.</param>
        private static void Large<T>(CaseLog log, string tag, string name, long count, ulong seed,
            params (string Name, Func<NDArray, NDArray> Make)[] views) where T : unmanaged
        {
            using var pattern = RawPattern(InfoOf<T>.NPTypeCode, count, seed);
            var runner = new CaseRunner<T>(pattern, log, $"{tag} large {name}");
            foreach (var (viewName, make) in views)
                runner.Check(viewName, () => make(pattern));
        }

        /// <summary>
        ///     Fails the test with every recorded failure (the first 30 listed), or — when there is none — asserts that the
        ///     catalog actually ran at least <paramref name="minimumCases"/> cases, so a catalog emptied by mistake cannot
        ///     pass vacuously.
        /// </summary>
        /// <param name="log">The finished log.</param>
        /// <param name="what">The catalog's description.</param>
        /// <param name="minimumCases">The fewest cases the catalog lists.</param>
        /// <exception cref="AssertFailedException">A case failed, or too few cases ran.</exception>
        private static void AssertNoFailures(CaseLog log, string what, int minimumCases)
        {
            if (log.Failures.Count > 0)
                Assert.Fail($"{what}: {log.Failures.Count} of {log.Cases} cases failed; first {System.Math.Min(30, log.Failures.Count)}:\n  "
                            + string.Join("\n  ", log.Failures.Take(30)));
            log.Cases.Should().BeGreaterThanOrEqualTo(minimumCases, $"{what} lists at least that many cases");
        }

        // ============================================================================================== patterns

        /// <summary>
        ///     A fresh C-contiguous 1-D array of random element bit patterns (deterministic per seed) with specials planted
        ///     at a fixed period: so a misplaced element changes the bytes with near certainty, and every special bit
        ///     pattern must survive every copy route.
        /// </summary>
        /// <param name="dtype">Element type.</param>
        /// <param name="n">Element count (0 allowed).</param>
        /// <param name="seed">Seed of the bit stream.</param>
        /// <returns>A fresh owning array; the caller disposes it.</returns>
        /// <remarks>
        ///     Booleans are 0 or 1 (NumSharp's bool storage), decimals are random valid decimals (96-bit mantissa, sign,
        ///     scale 0–28 — raw bits would not be decimals), every other dtype takes raw random bits. Every 13th element is
        ///     then a special: NaNs quiet and signalling, both signs, with payloads; ±inf; ±0; subnormals; extremes — for
        ///     float16 / float32 / float64 and both parts of a complex; 0, −1, the minimum and the maximum for the
        ///     integers; zeros of both signs and other scales, the extremes and the smallest step for decimal. Every 29th
        ///     complex is a signed zero in both parts (complex → bool reads both).
        /// </remarks>
        private static NDArray RawPattern(NPTypeCode dtype, long n, ulong seed)
        {
            var a = np.empty(new Shape(n), dtype);
            Span<byte> bytes = a.Unsafe.Bytes();
            ulong state = seed;
            switch (dtype)
            {
                case NPTypeCode.Boolean:
                    for (int i = 0; i < bytes.Length; i++)
                        bytes[i] = (byte)(NextBits(ref state) & 1);
                    return a;
                case NPTypeCode.Decimal:
                {
                    Span<decimal> d = MemoryMarshal.Cast<byte, decimal>(bytes);
                    for (int i = 0; i < d.Length; i++)
                        d[i] = i % 13 == 6 ? DecimalSpecials[i / 13 % DecimalSpecials.Length] : RandomDecimal(ref state);
                    return a;
                }
            }

            Span<ulong> words = MemoryMarshal.Cast<byte, ulong>(bytes);
            for (int i = 0; i < words.Length; i++)
                words[i] = NextBits(ref state);
            for (int i = words.Length * 8; i < bytes.Length; i++)
                bytes[i] = (byte)NextBits(ref state);

            int size = bytes.Length / System.Math.Max(1, (int)n);
            for (int i = 6; i < n; i += 13)
            {
                int k = i / 13;
                switch (dtype)
                {
                    case NPTypeCode.Half:
                        MemoryMarshal.Cast<byte, ushort>(bytes)[i] = HalfSpecials[k % HalfSpecials.Length];
                        break;
                    case NPTypeCode.Single:
                        MemoryMarshal.Cast<byte, uint>(bytes)[i] = SingleSpecials[k % SingleSpecials.Length];
                        break;
                    case NPTypeCode.Double:
                        MemoryMarshal.Cast<byte, ulong>(bytes)[i] = DoubleSpecials[k % DoubleSpecials.Length];
                        break;
                    case NPTypeCode.Complex:
                        // Real and imaginary parts drawn from the table at different phases, so each special meets others.
                        MemoryMarshal.Cast<byte, ulong>(bytes)[2 * i] = DoubleSpecials[k % DoubleSpecials.Length];
                        MemoryMarshal.Cast<byte, ulong>(bytes)[2 * i + 1] = DoubleSpecials[(k + 5) % DoubleSpecials.Length];
                        break;
                    default:
                        // Integers (char included): 0, all ones (−1), the minimum (sign bit only) and the maximum.
                        ulong special = (k % 4) switch
                        {
                            0 => 0UL,
                            1 => ulong.MaxValue,
                            2 => 1UL << (size * 8 - 1),
                            _ => (1UL << (size * 8 - 1)) - 1,
                        };
                        for (int b = 0; b < size; b++)
                            bytes[i * size + b] = (byte)(special >> (8 * b));
                        break;
                }
            }
            if (dtype == NPTypeCode.Complex)
                for (int i = 11; i < n; i += 29)
                {
                    // (±0, ±0): false as a bool only when BOTH parts are zero.
                    MemoryMarshal.Cast<byte, ulong>(bytes)[2 * i] = i % 2 == 0 ? 0UL : 0x8000_0000_0000_0000UL;
                    MemoryMarshal.Cast<byte, ulong>(bytes)[2 * i + 1] = i % 3 == 0 ? 0x8000_0000_0000_0000UL : 0UL;
                }
            return a;
        }

        /// <summary>
        ///     A fresh C-contiguous 1-D source of <paramref name="from"/> whose every value converts into
        ///     <paramref name="to"/> by a DEFINED rule, so astype and ToArray must agree on every host: raw bit patterns
        ///     wherever every value is defined, else values drawn inside the target's range.
        /// </summary>
        /// <param name="from">The source dtype.</param>
        /// <param name="to">The target dtype (≠ <paramref name="from"/>).</param>
        /// <param name="n">Element count.</param>
        /// <param name="seed">Seed of the value stream.</param>
        /// <returns>A fresh owning array; the caller disposes it.</returns>
        /// <remarks>
        ///     <para>
        ///         Integer and bool sources convert by a defined rule into every target (modular narrowing, IEEE rounding —
        ///         overflow to ±inf included —, ≠ 0), and every source converts by one into bool, float16, float32, float64
        ///         and complex: those take <see cref="RawPattern"/>, specials and all.
        ///     </para>
        ///     <para>
        ///         A float, complex or decimal source into an integer (or char) target is defined only where the value's
        ///         integral part fits the target (C leaves the rest undefined, and NumSharp's scalar and vector paths —
        ///         like NumPy's own — answer it differently): values anywhere in the range, small ones with fractions
        ///         (negative fractions truncate toward zero), and values just inside both ends, each re-checked after
        ///         rounding to the source's precision (a float16 holds at most 65 504). Into decimal, the source must be
        ///         finite and within the decimal range: magnitudes from 1e-12 to 1e20, exact binary fractions, 53-bit
        ///         integers and signed zeros. A complex source's real part follows those rules; its imaginary part is
        ///         random bits, which such a conversion discards.
        ///     </para>
        /// </remarks>
        private static NDArray ConversionPattern(NPTypeCode from, NPTypeCode to, long n, ulong seed)
        {
            bool bounded = IsIntegerTarget(to, out double lo, out double hi);
            bool floating = from is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex or NPTypeCode.Decimal;
            if (!floating || (!bounded && to != NPTypeCode.Decimal))
            {
                var raw = RawPattern(from, n, seed);
                // A known divergence, pinned (not hidden) by the two [OpenBugs] signalling-NaN tests: a signalling NaN
                // converted into or out of float16 keeps its quiet bit clear in NumPy, but ToArray's scalar routes set
                // it (and astype's float16 → float64 / complex128 kernels do too). Every such conversion here gets
                // QUIET NaNs — payloads and signs intact — so the matrix stays strict on everything else.
                if (from == NPTypeCode.Half || to == NPTypeCode.Half)
                    QuietSignallingNaNs(raw, from);
                return raw;
            }
            if (from == NPTypeCode.Half)
            {
                // A float16 holds at most ±65 504: draw inside that too, or nearly every value would be replaced.
                lo = System.Math.Max(lo, -65504);
                hi = System.Math.Min(hi, 65504);
            }

            var a = np.empty(new Shape(n), from);
            Span<byte> bytes = a.Unsafe.Bytes();
            ulong state = seed;
            for (int i = 0; i < n; i++)
            {
                double v = bounded ? DrawInRange(ref state, i, lo, hi) : DrawDecimalSafe(ref state, i);
                // The value as the source dtype holds it, re-checked; a value that rounded out of the domain is replaced.
                double fallback = i % 7 + 0.25;
                switch (from)
                {
                    case NPTypeCode.Half:
                    {
                        Half h = (Half)v;
                        MemoryMarshal.Cast<byte, Half>(bytes)[i] = Fits((double)h, bounded, lo, hi) ? h : (Half)fallback;
                        break;
                    }
                    case NPTypeCode.Single:
                    {
                        float f = (float)v;
                        MemoryMarshal.Cast<byte, float>(bytes)[i] = Fits(f, bounded, lo, hi) ? f : (float)fallback;
                        break;
                    }
                    case NPTypeCode.Double:
                        MemoryMarshal.Cast<byte, double>(bytes)[i] = Fits(v, bounded, lo, hi) ? v : fallback;
                        break;
                    case NPTypeCode.Decimal:
                    {
                        // Two decimal places: fractions that truncate, at a scale other than the double's shortest form.
                        decimal d = (decimal)System.Math.Round(v, 2);
                        MemoryMarshal.Cast<byte, decimal>(bytes)[i] = Fits((double)d, bounded, lo, hi) ? d : (decimal)fallback;
                        break;
                    }
                    default:
                    {
                        double imaginary = BitConverter.UInt64BitsToDouble(NextBits(ref state));
                        MemoryMarshal.Cast<byte, Complex>(bytes)[i] = new Complex(Fits(v, bounded, lo, hi) ? v : fallback, imaginary);
                        break;
                    }
                }
            }
            return a;
        }

        /// <summary>
        ///     Sets the quiet bit of every signalling NaN in a float16, float32, float64 or complex128 array (both parts of
        ///     a complex), leaving every other value — quiet NaNs, their payloads and signs included — untouched; any
        ///     other dtype is left as it is.
        /// </summary>
        /// <param name="a">A C-contiguous array, modified in place.</param>
        /// <param name="dtype">Its dtype.</param>
        /// <remarks>
        ///     Used only for conversions into or out of float16, where ToArray's scalar routes disagree with astype on
        ///     signalling NaNs (see <see cref="Conversion_SignallingNaNIntoOrOutOfFloat16_StaysSignallingOnEveryRoute"/>).
        /// </remarks>
        private static void QuietSignallingNaNs(NDArray a, NPTypeCode dtype)
        {
            Span<byte> bytes = a.Unsafe.Bytes();
            switch (dtype)
            {
                case NPTypeCode.Half:
                    foreach (ref ushort h in MemoryMarshal.Cast<byte, ushort>(bytes))
                        // Exponent all ones, a non-zero significand, the quiet bit (the significand's top bit) clear.
                        if ((h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0 && (h & 0x0200) == 0)
                            h |= 0x0200;
                    break;
                case NPTypeCode.Single:
                    foreach (ref uint u in MemoryMarshal.Cast<byte, uint>(bytes))
                        if ((u & 0x7F80_0000) == 0x7F80_0000 && (u & 0x007F_FFFF) != 0 && (u & 0x0040_0000) == 0)
                            u |= 0x0040_0000;
                    break;
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    // A complex is two doubles: both parts are quieted alike.
                    foreach (ref ulong x in MemoryMarshal.Cast<byte, ulong>(bytes))
                        if ((x & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000 && (x & 0x000F_FFFF_FFFF_FFFF) != 0
                            && (x & 0x0008_0000_0000_0000) == 0)
                            x |= 0x0008_0000_0000_0000;
                    break;
            }
        }

        /// <summary>
        ///     Whether <paramref name="to"/> is an integer (or char) target, and if so the range a float's integral part
        ///     must fall in for the conversion to be defined.
        /// </summary>
        /// <param name="to">The target dtype.</param>
        /// <param name="lo">The smallest allowed integral part (as a double).</param>
        /// <param name="hi">The largest allowed integral part that is a double (int64 / uint64 maxima are not doubles: 2⁶³ − 1024 and 2⁶⁴ − 2048).</param>
        /// <returns>True for the eight integer dtypes and char; false for bool, the floats, complex and decimal.</returns>
        private static bool IsIntegerTarget(NPTypeCode to, out double lo, out double hi)
        {
            // Every arm a (double, double): the switch's type must not depend on inferring a common tuple type.
            (lo, hi) = to switch
            {
                NPTypeCode.Byte => (0.0, (double)byte.MaxValue),
                NPTypeCode.SByte => ((double)sbyte.MinValue, (double)sbyte.MaxValue),
                NPTypeCode.Int16 => ((double)short.MinValue, (double)short.MaxValue),
                NPTypeCode.UInt16 or NPTypeCode.Char => (0.0, (double)ushort.MaxValue),
                NPTypeCode.Int32 => ((double)int.MinValue, (double)int.MaxValue),
                NPTypeCode.UInt32 => (0.0, (double)uint.MaxValue),
                NPTypeCode.Int64 => (-9223372036854775808.0, 9223372036854774784.0),
                NPTypeCode.UInt64 => (0.0, 18446744073709549568.0),
                _ => (double.NaN, double.NaN),
            };
            return !double.IsNaN(lo);
        }

        /// <summary>
        ///     Whether a source value (as a double) converts by a defined rule: its integral part inside [lo, hi] for an
        ///     integer target, finite and within ±1e21 for a decimal target.
        /// </summary>
        /// <param name="w">The value as the source dtype holds it.</param>
        /// <param name="bounded">An integer target (else a decimal target).</param>
        /// <param name="lo">The integer target's smallest integral part.</param>
        /// <param name="hi">The integer target's largest integral part.</param>
        /// <returns>True when the conversion is defined; false for NaN, ±inf and out-of-range values.</returns>
        private static bool Fits(double w, bool bounded, double lo, double hi)
        {
            if (!bounded)
                return double.IsFinite(w) && System.Math.Abs(w) <= 1e21;
            double t = System.Math.Truncate(w);
            // NaN fails both comparisons; ±inf fails one.
            return t >= lo && t <= hi;
        }

        /// <summary>
        ///     Draws a value whose integral part lies in [lo, hi], cycling through four kinds: anywhere in the range, small
        ///     with an exact binary fraction, just under the top, just above the bottom.
        /// </summary>
        /// <param name="state">The bit stream's state.</param>
        /// <param name="i">The element index (selects the kind).</param>
        /// <param name="lo">The smallest integral part.</param>
        /// <param name="hi">The largest integral part.</param>
        /// <returns>The value (the caller re-checks it after rounding to the source dtype).</returns>
        private static double DrawInRange(ref ulong state, long i, double lo, double hi)
        {
            ulong bits = NextBits(ref state);
            // 53 random bits → a uniform double in [0, 1).
            double u = (bits >> 11) * (1.0 / (1UL << 53));
            return (i % 4) switch
            {
                0 => lo + (hi - lo) * u,
                1 => ((long)(bits % 4001) - 2000) / 8.0,
                // hi + [0, 0.875) truncates to hi; lo − [0, 0.875) to lo (to 0 for an unsigned target: −0.875 … 0).
                2 => hi + 0.875 * u,
                _ => lo - 0.875 * u,
            };
        }

        /// <summary>
        ///     Draws a finite value inside the decimal range, cycling through magnitudes from 1e-12 to 1e20, small exact
        ///     binary fractions, 53-bit integers, and a table of specials (signed zeros, 0.1, a double's smallest
        ///     subnormal — which rounds to 0m —, ±1e20).
        /// </summary>
        /// <param name="state">The bit stream's state.</param>
        /// <param name="i">The element index (selects the kind).</param>
        /// <returns>The value (the caller re-checks it after rounding to the source dtype).</returns>
        private static double DrawDecimalSafe(ref ulong state, long i)
        {
            ulong bits = NextBits(ref state);
            double sign = (bits & 1) == 0 ? 1.0 : -1.0;
            double u = (bits >> 11) * (1.0 / (1UL << 53));
            return (i % 4) switch
            {
                0 => sign * System.Math.Pow(10, -12 + 32 * u),
                1 => ((long)(bits % 4001) - 2000) / 8.0,
                2 => sign * (bits >> 11),
                _ => DecimalSafeSpecials[(int)(i / 4 % DecimalSafeSpecials.Length)],
            };
        }

        /// <summary>A random valid decimal: a 96-bit mantissa (a third of them 64-bit, a third 32-bit), a random sign and scale 0–28.</summary>
        /// <param name="state">The bit stream's state.</param>
        /// <returns>The decimal.</returns>
        private static decimal RandomDecimal(ref ulong state)
        {
            ulong low = NextBits(ref state), high = NextBits(ref state);
            int lo = (int)low, mid = (int)(low >> 32), hi = (int)high;
            // Mix mantissa widths: full 96-bit ones only would make every decimal enormous.
            if (high % 3 == 1)
                hi = 0;
            else if (high % 3 == 2)
                hi = mid = 0;
            return new decimal(lo, mid, hi, ((high >> 40) & 1) == 1, (byte)((high >> 48) % 29));
        }

        /// <summary>
        ///     SplitMix64: a tiny, fast, well-mixed deterministic bit stream — every output bit depends on the seed, so
        ///     the patterns are reproducible and a misplaced element is detected with near certainty.
        /// </summary>
        /// <param name="state">The stream's state, advanced by one step.</param>
        /// <returns>The next 64 random bits.</returns>
        private static ulong NextBits(ref ulong state)
        {
            ulong z = state += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }

        /// <summary>float64 specials planted by <see cref="RawPattern"/> (bits): NaNs quiet / signalling / signed / with payloads, ±inf, ±0, subnormals, the extremes, 1 + ulp.</summary>
        private static readonly ulong[] DoubleSpecials =
        {
            0x7FF8_0000_0000_0000, 0xFFF8_0000_0000_0000, 0x7FF8_DEAD_BEEF_0001, 0x7FF4_0000_0000_0001,
            0x7FF0_0000_0000_0001, 0xFFF0_DEAD_BEEF_0002, 0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000,
            0x8000_0000_0000_0000, 0x0000_0000_0000_0000, 0x0000_0000_0000_0001, 0x800F_FFFF_FFFF_FFFF,
            0x7FEF_FFFF_FFFF_FFFF, 0x3FF0_0000_0000_0001,
        };

        /// <summary>float32 specials planted by <see cref="RawPattern"/> (bits), the float64 table's counterparts.</summary>
        private static readonly uint[] SingleSpecials =
        {
            0x7FC0_0000, 0xFFC0_0000, 0x7FC0_BEEF, 0x7FA0_0001, 0x7F80_0001, 0xFF80_0002, 0x7F80_0000,
            0xFF80_0000, 0x8000_0000, 0x0000_0000, 0x0000_0001, 0x807F_FFFF, 0x7F7F_FFFF, 0x3F80_0001,
        };

        /// <summary>float16 specials planted by <see cref="RawPattern"/> (bits), the float64 table's counterparts.</summary>
        private static readonly ushort[] HalfSpecials =
        {
            0x7E00, 0xFE00, 0x7E01, 0x7D01, 0x7C01, 0xFC02, 0x7C00, 0xFC00, 0x8000, 0x0000, 0x0001, 0x83FF, 0x7BFF, 0x3C01,
        };

        /// <summary>decimal specials planted by <see cref="RawPattern"/>: the extremes, zeros of both signs and scales, 1.00, the smallest step, −1, the largest mantissa at scale 28.</summary>
        private static readonly decimal[] DecimalSpecials =
        {
            decimal.MaxValue, decimal.MinValue, 0m, new decimal(0, 0, 0, true, 3), 1.00m,
            0.0000000000000000000000000001m, -1m, new decimal(-1, -1, -1, false, 28),
        };

        /// <summary>Finite doubles every decimal conversion accepts: signed zeros, 0.1, −123.456, the smallest subnormal (→ 0m), ±1e20, 7.9e27.</summary>
        private static readonly double[] DecimalSafeSpecials = { 0.0, -0.0, 0.1, -123.456, double.Epsilon, 1e20, -1e20, 7.9e27 };

        /// <summary>float16 values from their bits.</summary>
        /// <param name="bits">The IEEE binary16 encodings.</param>
        /// <returns>The values.</returns>
        private static Half[] Halves(params ushort[] bits) => bits.Select(BitConverter.UInt16BitsToHalf).ToArray();

        /// <summary>float32 values from their bits.</summary>
        /// <param name="bits">The IEEE binary32 encodings.</param>
        /// <returns>The values.</returns>
        private static float[] Singles(params uint[] bits) => bits.Select(BitConverter.UInt32BitsToSingle).ToArray();

        /// <summary>float64 values from their bits.</summary>
        /// <param name="bits">The IEEE binary64 encodings.</param>
        /// <returns>The values.</returns>
        private static double[] Doubles(params ulong[] bits) => bits.Select(BitConverter.UInt64BitsToDouble).ToArray();

        /// <summary>The values repeated back to back <paramref name="times"/> times.</summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="values">The values.</param>
        /// <param name="times">How many copies (≥ 1).</param>
        /// <returns>A new array of <c>values.Length × times</c> elements.</returns>
        private static T[] Repeat<T>(T[] values, int times)
        {
            var result = new T[values.Length * times];
            for (int t = 0; t < times; t++)
                values.CopyTo(result, t * values.Length);
            return result;
        }

        /// <summary>Each value followed by a copy of itself: <c>[v0, v0, v1, v1, …]</c>, whose <c>[::2]</c> is the values.</summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="values">The values.</param>
        /// <returns>A new array of twice the length.</returns>
        private static T[] Interleave<T>(T[] values)
        {
            var result = new T[values.Length * 2];
            for (int i = 0; i < values.Length; i++)
                result[2 * i] = result[2 * i + 1] = values[i];
            return result;
        }

        // ============================================================================================== machinery

        /// <summary>The cases a catalog ran and the failures it recorded.</summary>
        private sealed class CaseLog
        {
            /// <summary>Cases run so far (a failing case counts too).</summary>
            public int Cases;

            /// <summary>One message per failing case (a case records at most one per kind of check).</summary>
            public readonly List<string> Failures = new List<string>();

            /// <summary>Records a failure.</summary>
            /// <param name="message">What failed, with the case's description.</param>
            public void Fail(string message) => Failures.Add(message);
        }

        /// <summary>
        ///     Runs cases over views of ONE C-contiguous source array: each <see cref="Check"/> builds a view, calls
        ///     <see cref="NDArray.ToArray{T}"/> twice, compares both results with the oracle, checks the source's buffer is
        ///     untouched, and disposes the view — recording failures in the <see cref="CaseLog"/> rather than stopping.
        /// </summary>
        /// <typeparam name="T">The requested element type: the source's dtype (an exact copy) or a conversion target.</typeparam>
        private sealed class CaseRunner<T> where T : unmanaged
        {
            /// <summary>The array every view reads.</summary>
            private readonly NDArray _source;

            /// <summary>The source's bytes when the runner was made: what they must still be after every case.</summary>
            private readonly byte[] _snapshot;

            /// <summary>The shared log.</summary>
            private readonly CaseLog _log;

            /// <summary>Prefix of every failure message (dtype, pair, pattern).</summary>
            private readonly string _prefix;

            /// <summary>Creates a runner over <paramref name="source"/>, snapshotting its bytes.</summary>
            /// <param name="source">A C-contiguous array (its bytes are read whole).</param>
            /// <param name="log">The shared log.</param>
            /// <param name="prefix">Prefix of every failure message.</param>
            /// <exception cref="InvalidOperationException"><paramref name="source"/> is not C-contiguous.</exception>
            public CaseRunner(NDArray source, CaseLog log, string prefix)
            {
                _source = source;
                _log = log;
                _prefix = prefix;
                _snapshot = source.Unsafe.ReadOnlyBytes().ToArray();
            }

            /// <summary>
            ///     Checks one case: the view's ToArray result — twice — against the oracle (the view's own logical bytes,
            ///     or those of its astype result for a conversion), the identity of the two results, and the source's
            ///     bytes afterwards.
            /// </summary>
            /// <param name="name">The case's description.</param>
            /// <param name="make">Builds the view (disposed here, with any astype result).</param>
            /// <remarks>
            ///     Two calls because a conversion's first call for a (source, target) pair runs the general frame, which
            ///     remembers the pair's cast kernel, and later calls run the lean frame on that kernel — both must match.
            ///     An empty result must be the shared <see cref="Array.Empty{T}"/> instance both times; any other result
            ///     must be a distinct array each call. A source found modified is reported and restored from the snapshot,
            ///     so later cases start from the pattern again.
            /// </remarks>
            public void Check(string name, Func<NDArray> make)
            {
                _log.Cases++;
                string what = $"{_prefix} {name}";
                NDArray view = null, converted = null;
                try
                {
                    view = make();
                    what += $" shape ({string.Join(", ", view.shape)})";
                    bool conversion = view.typecode != InfoOf<T>.NPTypeCode;
                    // The oracle: the view's own logical bytes; for a conversion, those of astype's result (the contract).
                    if (conversion)
                        converted = view.astype(InfoOf<T>.NPTypeCode);
                    byte[] want = LogicalBytes(converted ?? view);
                    T[] first = view.ToArray<T>();
                    T[] second = view.ToArray<T>();
                    // A decimal RESULT OF A CONVERSION compares by value; an exact decimal copy must keep every bit.
                    bool byValue = conversion && typeof(T) == typeof(decimal);
                    string difference = FirstDifference(first, want, byValue) ?? FirstDifference(second, want, byValue);
                    if (difference != null)
                        _log.Fail($"{what}: {difference}");
                    else if (view.size == 0 && !(ReferenceEquals(first, Array.Empty<T>()) && ReferenceEquals(second, Array.Empty<T>())))
                        _log.Fail($"{what}: an empty result must be the shared Array.Empty<{typeof(T).Name}>() instance");
                    else if (view.size != 0 && ReferenceEquals(first, second))
                        _log.Fail($"{what}: two calls returned the same array");
                }
                catch (Exception e)
                {
                    _log.Fail($"{what}: threw {e.GetType().Name}: {e.Message}");
                }
                finally
                {
                    converted?.Dispose();
                    view?.Dispose();
                }

                if (!_source.Unsafe.ReadOnlyBytes().SequenceEqual(_snapshot))
                {
                    _log.Fail($"{what}: ToArray wrote into its source's buffer");
                    _snapshot.CopyTo(_source.Unsafe.Bytes());
                }
            }
        }

        /// <summary>A body run once per element type: C# cannot pass an open generic method as a delegate.</summary>
        private interface IElementTypeVisitor
        {
            /// <summary>Runs the body for element type <typeparamref name="T"/>.</summary>
            /// <typeparam name="T">The CLR type of a NumSharp dtype.</typeparam>
            void Visit<T>() where T : unmanaged;
        }

        /// <summary>Calls <paramref name="visitor"/> with the CLR element type of <paramref name="dtype"/>.</summary>
        /// <param name="dtype">One of the 15 dtypes.</param>
        /// <param name="visitor">The body to run.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="dtype"/> is not one of the 15 dtypes.</exception>
        private static void Visit(NPTypeCode dtype, IElementTypeVisitor visitor)
        {
            switch (dtype)
            {
                case NPTypeCode.Boolean: visitor.Visit<bool>(); break;
                case NPTypeCode.Byte: visitor.Visit<byte>(); break;
                case NPTypeCode.SByte: visitor.Visit<sbyte>(); break;
                case NPTypeCode.Int16: visitor.Visit<short>(); break;
                case NPTypeCode.UInt16: visitor.Visit<ushort>(); break;
                case NPTypeCode.Int32: visitor.Visit<int>(); break;
                case NPTypeCode.UInt32: visitor.Visit<uint>(); break;
                case NPTypeCode.Int64: visitor.Visit<long>(); break;
                case NPTypeCode.UInt64: visitor.Visit<ulong>(); break;
                case NPTypeCode.Char: visitor.Visit<char>(); break;
                case NPTypeCode.Half: visitor.Visit<Half>(); break;
                case NPTypeCode.Single: visitor.Visit<float>(); break;
                case NPTypeCode.Double: visitor.Visit<double>(); break;
                case NPTypeCode.Decimal: visitor.Visit<decimal>(); break;
                case NPTypeCode.Complex: visitor.Visit<Complex>(); break;
                default: throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Not a NumSharp dtype.");
            }
        }

        /// <summary>Runs <see cref="RunSameDtypeCatalog{T}"/> for the visited element type.</summary>
        private sealed class SameDtypeVisitor : IElementTypeVisitor
        {
            /// <summary>The shared log.</summary>
            private readonly CaseLog _log;

            /// <summary>Creates the visitor.</summary>
            /// <param name="log">The log the catalog records into.</param>
            public SameDtypeVisitor(CaseLog log) => _log = log;

            /// <inheritdoc/>
            public void Visit<T>() where T : unmanaged => RunSameDtypeCatalog<T>(_log);
        }

        /// <summary>Runs <see cref="RunConversionCatalog{T}"/> from one source dtype into the visited target.</summary>
        private sealed class ConversionVisitor : IElementTypeVisitor
        {
            /// <summary>The source dtype.</summary>
            private readonly NPTypeCode _from;

            /// <summary>The shared log.</summary>
            private readonly CaseLog _log;

            /// <summary>Creates the visitor.</summary>
            /// <param name="from">The source dtype.</param>
            /// <param name="log">The log the catalog records into.</param>
            public ConversionVisitor(NPTypeCode from, CaseLog log)
            {
                _from = from;
                _log = log;
            }

            /// <inheritdoc/>
            public void Visit<T>() where T : unmanaged => RunConversionCatalog<T>(_from, _log);
        }

        /// <summary>
        ///     Checks that empty sources of the visited element type — owning ones of three ranks and empty views — yield
        ///     the shared <see cref="Array.Empty{T}"/> instance, same dtype and converted.
        /// </summary>
        private sealed class EmptyVisitor : IElementTypeVisitor
        {
            /// <summary>The shared log.</summary>
            private readonly CaseLog _log;

            /// <summary>Creates the visitor.</summary>
            /// <param name="log">The log the checks record into.</param>
            public EmptyVisitor(CaseLog log) => _log = log;

            /// <inheritdoc/>
            public void Visit<T>() where T : unmanaged
            {
                NPTypeCode dtype = InfoOf<T>.NPTypeCode;
                using var line = RawPattern(dtype, 40, 0x5EED_0200);
                var sources = new (string Name, Func<NDArray> Make)[]
                {
                    ("owning (0,)", () => np.empty(new Shape(0), dtype)),
                    ("owning (3, 0)", () => np.empty(new Shape(3, 0), dtype)),
                    ("owning (2, 0, 4)", () => np.empty(new Shape(2, 0, 4), dtype)),
                    ("[4:4]", () => line["4:4"]),
                    ("[::-1][0:0]", () => line["::-1"]["0:0"]),
                    ("(5, 8).T[:, 3:3]", () => Grid(line, 5, 8).T[":, 3:3"]),
                    ("broadcast (0, 6)", () => np.broadcast_to(line["0:1"], new Shape(0, 6))),
                    ("[::3][2:2]", () => line["::3"]["2:2"]),
                };
                foreach (var (name, make) in sources)
                {
                    _log.Cases++;
                    try
                    {
                        using NDArray source = make();
                        // Same dtype, then two conversions (a float and an integer target, never the dtype itself).
                        if (!ReferenceEquals(source.ToArray<T>(), Array.Empty<T>()))
                            _log.Fail($"{dtype} {name}: same-dtype result is not Array.Empty<{typeof(T).Name}>()");
                        if (dtype != NPTypeCode.Double && !ReferenceEquals(source.ToArray<double>(), Array.Empty<double>()))
                            _log.Fail($"{dtype} {name}: float64 result is not Array.Empty<Double>()");
                        if (dtype != NPTypeCode.Int32 && !ReferenceEquals(source.ToArray<int>(), Array.Empty<int>()))
                            _log.Fail($"{dtype} {name}: int32 result is not Array.Empty<Int32>()");
                    }
                    catch (Exception e)
                    {
                        _log.Fail($"{dtype} {name}: threw {e.GetType().Name}: {e.Message}");
                    }
                }
            }
        }
    }
}
