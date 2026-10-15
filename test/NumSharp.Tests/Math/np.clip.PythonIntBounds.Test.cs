using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.clip with integer bounds that may not fit the array's dtype — NumPy's <c>_methods._clip</c>
    ///     (numpy/_core/_methods.py) and the ufunc it routes to. A C# integer literal is NumPy's Python int:
    ///     on an integer array a bound at or past the dtype's limit is DROPPED (it cannot clip anything), and a
    ///     kept bound that does not fit the loop dtype raises OverflowError instead of wrapping. Every
    ///     expectation below was probed against NumPy 2.4.2. The texts past ±2^31 are the LP64 (Linux/macOS)
    ///     ones, probed through WSL, because Windows NumPy's 32-bit C long reports those values differently.
    /// </summary>
    [TestClass]
    public class np_clip_python_int_bounds_tests
    {
        /// <summary>The int8 test array: both limits, a negative, zero and a positive value.</summary>
        /// <returns>A fresh int8 array [-128, -5, 0, 5, 127].</returns>
        private static NDArray I8() => np.array(new sbyte[] { -128, -5, 0, 5, 127 });

        /// <summary>The uint8 test array from the bug report: zero, a small value, a large value and the limit.</summary>
        /// <returns>A fresh uint8 array [0, 5, 200, 255].</returns>
        private static NDArray U8() => np.array(new byte[] { 0, 5, 200, 255 });

        /// <summary>Asserts the call raises NumPy's OverflowError with the exact NumPy message.</summary>
        /// <param name="call">The clip call expected to raise.</param>
        /// <param name="text">NumPy's message, verbatim.</param>
        private static void ExpectOverflow(Func<object> call, string text)
        {
            var e = Assert.ThrowsExactly<OverflowException>(() => call());
            e.Message.Should().Be(text);
        }

        /// <summary>Asserts the call raises the house ArgumentException carrying NumPy's UFuncTypeError text.</summary>
        /// <param name="call">The clip call expected to raise.</param>
        /// <param name="text">NumPy's message, verbatim.</param>
        private static void ExpectCastError(Func<object> call, string text)
        {
            var e = Assert.ThrowsExactly<ArgumentException>(() => call());
            e.Message.Should().Be(text);
        }

        /// <summary>The three reported cases: before the fix the bound was cast to the array's dtype and wrapped.</summary>
        [TestMethod]
        public void ReportedCases_DropTheBoundInsteadOfWrapping()
        {
            // NumPy: np.clip(uint8[0,5,200,255], 0, 256) -> [0 5 200 255] (both bounds dropped; NumSharp gave [0 0 0 0]).
            var r1 = np.clip(U8(), 0, 256);
            r1.dtype.Should().Be(np.uint8);
            r1.ToArray<byte>().Should().Equal(0, 5, 200, 255);

            // NumPy: np.clip(uint8, -1, 100) -> [0 5 100 100] (min dropped; NumSharp gave all 100).
            var r2 = np.clip(U8(), -1, 100);
            r2.dtype.Should().Be(np.uint8);
            r2.ToArray<byte>().Should().Equal(0, 5, 100, 100);

            // NumPy: np.clip(int8, -1000, 1000) -> unchanged (NumSharp gave all -24).
            var r3 = np.clip(I8(), -1000, 1000);
            r3.dtype.Should().Be(np.int8);
            r3.ToArray<sbyte>().Should().Equal(-128, -5, 0, 5, 127);
        }

        /// <summary>A kept bound on the impossible side raises; when both are kept, min is reported first (operand order).</summary>
        [TestMethod]
        public void ImpossibleSide_RaisesOverflowError_MinFirst()
        {
            ExpectOverflow(() => np.clip(I8(), 1000, 2000), "Python integer 1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), -2000, -1000), "Python integer -1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), 1000, null), "Python integer 1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), null, -1000), "Python integer -1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), 1000, -1000), "Python integer 1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), -1000, -200), "Python integer -200 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), 200, 1000), "Python integer 200 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), 200, 50), "Python integer 200 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), -50, -200), "Python integer -200 out of bounds for int8");
            ExpectOverflow(() => np.clip(U8(), -5, -1), "Python integer -1 out of bounds for uint8");
            ExpectOverflow(() => np.clip(U8(), 300, 400), "Python integer 300 out of bounds for uint8");
            ExpectOverflow(() => np.clip(U8(), 300, 100), "Python integer 300 out of bounds for uint8");
            ExpectOverflow(() => np.clip(U8(), 100, -1), "Python integer -1 out of bounds for uint8");
            ExpectOverflow(() => np.clip(I8(), 128, 128), "Python integer 128 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), -129, -129), "Python integer -129 out of bounds for int8");
        }

        /// <summary>A one-sided bound past the limit is dropped too, leaving positive (a copy).</summary>
        [TestMethod]
        public void OneSided_PastTheLimit_IsTheIdentity()
        {
            np.clip(I8(), -1000, null).ToArray<sbyte>().Should().Equal(-128, -5, 0, 5, 127);
            np.clip(I8(), null, 1000).ToArray<sbyte>().Should().Equal(-128, -5, 0, 5, 127);
            np.clip(I8(), -5, 3).ToArray<sbyte>().Should().Equal(-5, -5, 0, 3, 3);
        }

        /// <summary>
        ///     Every integer dtype at its exact limits. A bound AT the limit is dropped (<c>min &lt;= iinfo.min</c>), one step
        ///     inside clips, and the limit used on the other side is kept and clips everything to it.
        /// </summary>
        [TestMethod]
        public void ExactLimits_EveryIntegerDtype()
        {
            // int8: array [min, min+1, 1, max-1, max]
            var a8 = np.array(new sbyte[] { -128, -127, 1, 126, 127 });
            np.clip(a8, -128, 127).ToArray<sbyte>().Should().Equal(-128, -127, 1, 126, 127);
            np.clip(a8, -127, 126).ToArray<sbyte>().Should().Equal(-127, -127, 1, 126, 126);
            np.clip(a8, -129, 128).ToArray<sbyte>().Should().Equal(-128, -127, 1, 126, 127);
            np.clip(a8, 127, null).ToArray<sbyte>().Should().Equal(127, 127, 127, 127, 127);
            np.clip(a8, null, -128).ToArray<sbyte>().Should().Equal(-128, -128, -128, -128, -128);

            var u8 = np.array(new byte[] { 0, 1, 1, 254, 255 });
            np.clip(u8, 0, 255).ToArray<byte>().Should().Equal(0, 1, 1, 254, 255);
            np.clip(u8, 1, 254).ToArray<byte>().Should().Equal(1, 1, 1, 254, 254);
            np.clip(u8, -1, 256).ToArray<byte>().Should().Equal(0, 1, 1, 254, 255);
            np.clip(u8, 255, null).ToArray<byte>().Should().Equal(255, 255, 255, 255, 255);
            np.clip(u8, null, 0).ToArray<byte>().Should().Equal(0, 0, 0, 0, 0);
            np.clip(u8, 255, 0).ToArray<byte>().Should().Equal(0, 0, 0, 0, 0);

            var i16 = np.array(new short[] { -32768, -32767, 1, 32766, 32767 });
            np.clip(i16, -32768, 32767).ToArray<short>().Should().Equal(-32768, -32767, 1, 32766, 32767);
            np.clip(i16, -32767, 32766).ToArray<short>().Should().Equal(-32767, -32767, 1, 32766, 32766);
            np.clip(i16, -32769, 32768).ToArray<short>().Should().Equal(-32768, -32767, 1, 32766, 32767);

            var u16 = np.array(new ushort[] { 0, 1, 1, 65534, 65535 });
            np.clip(u16, 0, 65535).ToArray<ushort>().Should().Equal(0, 1, 1, 65534, 65535);
            np.clip(u16, -1, 65536).ToArray<ushort>().Should().Equal(0, 1, 1, 65534, 65535);
            np.clip(u16, 1, 65534).ToArray<ushort>().Should().Equal(1, 1, 1, 65534, 65534);

            var i32 = np.array(new int[] { int.MinValue, int.MinValue + 1, 1, int.MaxValue - 1, int.MaxValue });
            np.clip(i32, int.MinValue, int.MaxValue).ToArray<int>().Should().Equal(int.MinValue, int.MinValue + 1, 1, int.MaxValue - 1, int.MaxValue);
            np.clip(i32, (long)int.MinValue - 1, (long)int.MaxValue + 1).ToArray<int>().Should().Equal(int.MinValue, int.MinValue + 1, 1, int.MaxValue - 1, int.MaxValue);
            np.clip(i32, int.MinValue + 1, int.MaxValue - 1).ToArray<int>().Should().Equal(int.MinValue + 1, int.MinValue + 1, 1, int.MaxValue - 1, int.MaxValue - 1);

            var u32 = np.array(new uint[] { 0, 1, 1, uint.MaxValue - 1, uint.MaxValue });
            np.clip(u32, 0, uint.MaxValue).ToArray<uint>().Should().Equal(0u, 1u, 1u, uint.MaxValue - 1, uint.MaxValue);
            np.clip(u32, -1, (long)uint.MaxValue + 1).ToArray<uint>().Should().Equal(0u, 1u, 1u, uint.MaxValue - 1, uint.MaxValue);
            np.clip(u32, 1, uint.MaxValue - 1).ToArray<uint>().Should().Equal(1u, 1u, 1u, uint.MaxValue - 1, uint.MaxValue - 1);

            var i64 = np.array(new long[] { long.MinValue, long.MinValue + 1, 1, long.MaxValue - 1, long.MaxValue });
            np.clip(i64, long.MinValue, long.MaxValue).ToArray<long>().Should().Equal(long.MinValue, long.MinValue + 1, 1L, long.MaxValue - 1, long.MaxValue);
            np.clip(i64, long.MinValue, ulong.MaxValue).ToArray<long>().Should().Equal(long.MinValue, long.MinValue + 1, 1L, long.MaxValue - 1, long.MaxValue);
            np.clip(i64, long.MinValue + 1, long.MaxValue - 1).ToArray<long>().Should().Equal(long.MinValue + 1, long.MinValue + 1, 1L, long.MaxValue - 1, long.MaxValue - 1);

            var u64 = np.array(new ulong[] { 0, 1, 1, ulong.MaxValue - 1, ulong.MaxValue });
            np.clip(u64, 0, ulong.MaxValue).ToArray<ulong>().Should().Equal(0UL, 1UL, 1UL, ulong.MaxValue - 1, ulong.MaxValue);
            np.clip(u64, -1, ulong.MaxValue).ToArray<ulong>().Should().Equal(0UL, 1UL, 1UL, ulong.MaxValue - 1, ulong.MaxValue);
            np.clip(u64, long.MinValue, ulong.MaxValue).ToArray<ulong>().Should().Equal(0UL, 1UL, 1UL, ulong.MaxValue - 1, ulong.MaxValue);
            np.clip(u64, 1, ulong.MaxValue - 1).ToArray<ulong>().Should().Equal(1UL, 1UL, 1UL, ulong.MaxValue - 1, ulong.MaxValue - 1);
            np.clip(u64, null, 0).ToArray<ulong>().Should().Equal(0UL, 0UL, 0UL, 0UL, 0UL);
        }

        /// <summary>
        ///     The LP64 texts (Linux/macOS NumPy 2.4.2, probed through WSL). NumPy converts a Python int through a C long for
        ///     every loop except uint32/uint64 (C unsigned long), so a value past long.MaxValue reports CPython's
        ///     "Python int too large to convert to C long" there, and the plain out-of-bounds text everywhere else.
        /// </summary>
        [TestMethod]
        public void PastInt32_UsesTheLp64Texts()
        {
            var i8 = I8();
            ExpectOverflow(() => np.clip(i8, 2147483648L, null), "Python integer 2147483648 out of bounds for int8");
            ExpectOverflow(() => np.clip(i8, null, -2147483649L), "Python integer -2147483649 out of bounds for int8");
            ExpectOverflow(() => np.clip(i8, long.MaxValue, null), "Python integer 9223372036854775807 out of bounds for int8");
            ExpectOverflow(() => np.clip(i8, null, long.MinValue), "Python integer -9223372036854775808 out of bounds for int8");
            ExpectOverflow(() => np.clip(i8, 9223372036854775808UL, null), "Python int too large to convert to C long");
            ExpectOverflow(() => np.clip(np.array(new byte[] { 5 }), ulong.MaxValue, null), "Python int too large to convert to C long");
            ExpectOverflow(() => np.clip(np.array(new ushort[] { 5 }), 9223372036854775808UL, null), "Python int too large to convert to C long");
            ExpectOverflow(() => np.clip(np.array(new ushort[] { 5 }), null, long.MinValue), "Python integer -9223372036854775808 out of bounds for uint16");
            ExpectOverflow(() => np.clip(np.array(new int[] { 5 }), 2147483648L, null), "Python integer 2147483648 out of bounds for int32");
            ExpectOverflow(() => np.clip(np.array(new long[] { 5 }), 9223372036854775808UL, null), "Python int too large to convert to C long");

            var u32 = np.array(new uint[] { 0, 5 });
            ExpectOverflow(() => np.clip(u32, 4294967296L, null), "Python integer 4294967296 out of bounds for uint32");
            ExpectOverflow(() => np.clip(u32, 9223372036854775808UL, null), "Python integer 9223372036854775808 out of bounds for uint32");
            ExpectOverflow(() => np.clip(u32, ulong.MaxValue, null, dtype: np.uint32), "Python integer 18446744073709551615 out of bounds for uint32");

            ExpectOverflow(() => np.clip(np.array(new ulong[] { 5 }), null, -1), "Python integer -1 out of bounds for uint64");
            ExpectOverflow(() => np.clip(np.array(new ulong[] { 5 }), null, long.MinValue), "Python integer -9223372036854775808 out of bounds for uint64");
        }

        /// <summary>
        ///     A bool array has no drop rule ('iu' kinds only). A bool loop packs nothing, but a Python int past long.MaxValue
        ///     still fails NumPy's int64 conversion; inside int64 the dtype= cast check reports it as int64.
        /// </summary>
        [TestMethod]
        public void BoolArray_PromotesAndChecksAsNumPy()
        {
            var b = np.array(new[] { true, false });
            var r = np.clip(b, -1000, 1000);
            r.dtype.Should().Be(np.int64);
            r.ToArray<long>().Should().Equal(1L, 0L);
            np.clip(b, false, true).ToArray<bool>().Should().Equal(true, false);

            ExpectOverflow(() => np.clip(b, 0, 9223372036854775808UL, dtype: np.@bool), "Python int too large to convert to C long");
            ExpectOverflow(() => np.clip(I8(), 9223372036854775808UL, null, dtype: np.@bool), "Python int too large to convert to C long");
            ExpectCastError(() => np.clip(b, long.MinValue, 1, dtype: np.@bool),
                "Cannot cast ufunc 'clip' input 1 from dtype('int64') to dtype('bool') with casting rule 'same_kind'");
        }

        /// <summary>NumPy packs the bounds before it looks at any element, so an empty array raises too.</summary>
        [TestMethod]
        public void EmptyArray_StillRaises()
        {
            var e = np.zeros(new Shape(0), np.int8);
            ExpectOverflow(() => np.clip(e, 1000, 2000), "Python integer 1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(e, null, -1000), "Python integer -1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(e, 9223372036854775808UL, null), "Python int too large to convert to C long");

            var r = np.clip(e, -1000, 1000);
            r.dtype.Should().Be(np.int8);
            r.size.Should().Be(0);
        }

        /// <summary>A 0-d input clips as a scalar: dropped bounds leave it, an impossible bound raises.</summary>
        [TestMethod]
        public void ZeroDInput()
        {
            var z = NDArray.Scalar((sbyte)5);
            var r = np.clip(z, -1000, 1000);
            r.ndim.Should().Be(0);
            r.dtype.Should().Be(np.int8);
            r.GetAtIndex<sbyte>(0).Should().Be(5);
            ExpectOverflow(() => np.clip(z, 1000, null), "Python integer 1000 out of bounds for int8");
        }

        /// <summary>Every C# integer literal type is a Python int, and so is a char; each takes the same drop/raise rules.</summary>
        [TestMethod]
        public void EveryCSharpIntegerLiteralType_IsAPythonInt()
        {
            sbyte[] identity = { -128, -5, 0, 5, 127 };
            np.clip(I8(), (short)-1000, (short)1000).ToArray<sbyte>().Should().Equal(identity);
            np.clip(I8(), (sbyte)-128, (sbyte)127).ToArray<sbyte>().Should().Equal(identity);
            np.clip(I8(), null, (byte)200).ToArray<sbyte>().Should().Equal(identity);
            np.clip(I8(), null, (ushort)60000).ToArray<sbyte>().Should().Equal(identity);
            np.clip(I8(), null, 3000000000u).ToArray<sbyte>().Should().Equal(identity);
            np.clip(I8(), long.MinValue, ulong.MaxValue).ToArray<sbyte>().Should().Equal(identity);
            np.clip(I8(), null, 'é').ToArray<sbyte>().Should().Equal(identity);            // char 233 >= 127: dropped

            ExpectOverflow(() => np.clip(I8(), (byte)200, null), "Python integer 200 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), (ushort)1000, null), "Python integer 1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), 'é', null), "Python integer 233 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), null, (short)-1000), "Python integer -1000 out of bounds for int8");
        }

        /// <summary>A 0-d VIEW bound (offset into a bigger buffer) is read at its own element.</summary>
        [TestMethod]
        public void ZeroDViewBound_ReadsItsOwnElement()
        {
            var big = np.array(new long[] { 0, 0, 1000 });
            np.clip(I8(), null, big[2]).ToArray<sbyte>().Should().Equal(-128, -5, 0, 5, 127);
            var neg = np.array(new long[] { 0, 0, -1000 });
            ExpectOverflow(() => np.clip(I8(), null, neg[2]), "Python integer -1000 out of bounds for int8");
        }

        /// <summary>
        ///     The drop rule uses the INPUT array's range; the overflow check uses the LOOP dtype (dtype= or promoted). NumPy
        ///     packs the scalars before it validates the casts, so the bound's error wins over an input cast error.
        /// </summary>
        [TestMethod]
        public void DtypeKeyword_DropByInputRange_PackIntoLoop()
        {
            var r1 = np.clip(I8(), 1000, 2000, dtype: np.int16);
            r1.dtype.Should().Be(np.int16);
            r1.ToArray<short>().Should().Equal(1000, 1000, 1000, 1000, 1000);
            np.clip(I8(), -1000, 1000, dtype: np.int16).ToArray<short>().Should().Equal(-128, -5, 0, 5, 127);
            np.clip(I8(), -1000, 50, dtype: np.int16).ToArray<short>().Should().Equal(-128, -5, 0, 5, 50);
            np.clip(I8(), 1000, -1000, dtype: np.int16).ToArray<short>().Should().Equal(-1000, -1000, -1000, -1000, -1000);

            ExpectOverflow(() => np.clip(I8(), 1000, 2000, dtype: np.int8), "Python integer 1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), 1000, 2000, dtype: np.uint8), "Python integer 1000 out of bounds for uint8");
            ExpectOverflow(() => np.clip(np.array(new byte[] { 1, 2, 3 }), 300, null, dtype: np.int8), "Python integer 300 out of bounds for int8");
            np.clip(np.array(new byte[] { 1, 2, 3 }), 300, null, dtype: np.uint16).ToArray<ushort>().Should().Equal(300, 300, 300);

            // clip(int8, -1, 5, dtype=uint64): -1 is kept (it is inside int8) and cannot pack into uint64 — reported
            // before the int8 -> uint64 input cast, which is not same_kind either.
            ExpectOverflow(() => np.clip(I8(), -1, 5, dtype: np.uint64), "Python integer -1 out of bounds for uint64");
            ExpectOverflow(() => np.clip(I8(), -1, null, dtype: np.uint64), "Python integer -1 out of bounds for uint64");
            // On uint8 the same -1 is dropped first: minimum(uint8, 5, dtype=uint64).
            np.clip(np.array(new byte[] { 1, 2, 3 }), -1, 5, dtype: np.uint64).ToArray<ulong>().Should().Equal(1UL, 2UL, 3UL);
        }

        /// <summary>
        ///     dtype= is NumPy's loop request: every input must reach it by a same_kind cast, checked in operand order, and
        ///     the message names the ROUTED ufunc. A Python int bound reaches any integer or inexact loop, a Python float only
        ///     inexact ones, a Python complex only complex.
        /// </summary>
        [TestMethod]
        public void DtypeKeyword_InputCasts_NameTheRoutedUfunc()
        {
            ExpectCastError(() => np.clip(I8(), 5, 10, dtype: np.uint8),
                "Cannot cast ufunc 'clip' input 0 from dtype('int8') to dtype('uint8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(I8(), -1000, 1000, dtype: np.uint8),
                "Cannot cast ufunc 'positive' input from dtype('int8') to dtype('uint8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(I8(), -1000, 50, dtype: np.uint8),
                "Cannot cast ufunc 'minimum' input 0 from dtype('int8') to dtype('uint8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(I8(), 1, 300, dtype: np.uint8),
                "Cannot cast ufunc 'maximum' input 0 from dtype('int8') to dtype('uint8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(I8(), 0.5, 2.5, dtype: np.int8),
                "Cannot cast ufunc 'clip' input 1 from dtype('float64') to dtype('int8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(I8(), 0, new Complex(1, 2), dtype: np.int8),
                "Cannot cast ufunc 'clip' input 2 from dtype('complex128') to dtype('int8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(np.array(new[] { 0.5, 2.5 }), 0, new Complex(1, 2), dtype: np.float64),
                "Cannot cast ufunc 'clip' input 2 from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");

            var b = np.array(new[] { true, false });
            ExpectCastError(() => np.clip(b, 0, 2, dtype: np.@bool),
                "Cannot cast ufunc 'clip' input 1 from dtype('int64') to dtype('bool') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(b, null, 0, dtype: np.@bool),
                "Cannot cast ufunc 'minimum' input 1 from dtype('int64') to dtype('bool') with casting rule 'same_kind'");

            var u8 = np.array(new byte[] { 1, 2, 3 });
            ExpectCastError(() => np.clip(u8, np.array(new long[] { 1, 1, 1 }), 2, dtype: np.uint8),
                "Cannot cast ufunc 'clip' input 1 from dtype('int64') to dtype('uint8') with casting rule 'same_kind'");

            // Legal: a Python int reaches uint8 even though int64 -> uint8 is not same_kind; float64 -> float16 is.
            np.clip(u8, 5, 10, dtype: np.uint8).ToArray<byte>().Should().Equal(5, 5, 5);
            np.clip(np.array(new[] { 0.5, 2.5 }), 1, 2, dtype: np.float16).ToArray<Half>().Should().Equal((Half)1, (Half)2);
            np.clip(np.array(new sbyte[] { -1, 2, 3 }), true, 2, dtype: np.int8).ToArray<sbyte>().Should().Equal(1, 2, 2);
            np.clip(np.array(new[] { 0.5, 2.5 }), 0, new Complex(1, 2), dtype: np.complex128).ToArray<Complex>()
                .Should().Equal(new Complex(0.5, 0), new Complex(1, 2));
        }

        /// <summary>
        ///     With no bound left NumPy calls positive, which has no bool loop; dtype= is positive's own loop request.
        /// </summary>
        [TestMethod]
        public void NoBoundLeft_IsPositive()
        {
            var b = np.array(new[] { true, false });
            var e1 = Assert.ThrowsExactly<TypeError>(() => np.clip(b));
            e1.Message.Should().Be("ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.BoolDType'> -> None");
            Assert.ThrowsExactly<TypeError>(() => np.clip(b, null, null));
            Assert.ThrowsExactly<TypeError>(() => b.clip());

            var e2 = Assert.ThrowsExactly<TypeError>(() => np.clip(I8(), dtype: np.@bool));
            e2.Message.Should().Be("ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.Int8DType'> -> <class 'numpy.dtypes.BoolDType'>");
            var e3 = Assert.ThrowsExactly<TypeError>(() => np.clip(b, dtype: np.@bool));
            e3.Message.Should().Be("ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.BoolDType'> -> <class 'numpy.dtypes.BoolDType'>");

            np.clip(b, dtype: np.float64).ToArray<double>().Should().Equal(1.0, 0.0);
            ExpectCastError(() => np.clip(I8(), dtype: np.uint8),
                "Cannot cast ufunc 'positive' input from dtype('int8') to dtype('uint8') with casting rule 'same_kind'");

            // Char has its own DType class (np.dtypes.CharDType), so its no-loop text names it; the oracle's uint16
            // proxy would say UInt16DType, which is why gen_oracle.py carves these rows out of the char tier.
            var c = np.array(new[] { 'a', 'b' });
            var e4 = Assert.ThrowsExactly<TypeError>(() => np.clip(c, dtype: np.@bool));
            e4.Message.Should().Be("ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.CharDType'> -> <class 'numpy.dtypes.BoolDType'>");
            // The bool check only applies when nothing is left to clip with: clip(b, False, True) has a bool loop.
            np.clip(b, false, true).dtype.Should().Be(np.@bool);
        }

        /// <summary>
        ///     out= follows NumPy's ufunc rule: any out the loop dtype reaches by a same_kind cast is legal and is returned
        ///     itself; a refused one names the routed ufunc.
        /// </summary>
        [TestMethod]
        public void Out_SameKindCastIsLegal_ErrorsNameTheRoutedUfunc()
        {
            var o1 = np.empty(new Shape(4), np.int8);
            var r1 = np.clip(U8(), 0, 256, o1);
            ReferenceEquals(r1, o1).Should().BeTrue();
            o1.ToArray<sbyte>().Should().Equal(0, 5, -56, -1);               // positive(u8, out=i8) wraps 200/255

            var o2 = np.empty(new Shape(4), np.float64);
            np.clip(U8(), -1, 100, o2).ToArray<double>().Should().Equal(0.0, 5.0, 100.0, 100.0);

            var o3 = np.empty(new Shape(4), np.int8);
            np.clip(U8(), 5, 100, o3).ToArray<sbyte>().Should().Equal(5, 5, 100, 100);

            var o4 = np.empty(new Shape(5), np.int16);
            np.clip(I8(), -1000, 50, o4).ToArray<short>().Should().Equal(-128, -5, 0, 5, 50);

            var o5 = np.empty(new Shape(2), np.float64);
            np.clip(np.array(new[] { 1.5f, -2f }), 0, 1, o5).ToArray<double>().Should().Equal(1.0, 0.0);

            ExpectCastError(() => np.clip(I8(), 5, 10, np.empty(new Shape(5), np.uint8)),
                "Cannot cast ufunc 'clip' output from dtype('int8') to dtype('uint8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(I8(), -1000, 1000, np.empty(new Shape(5), np.uint8)),
                "Cannot cast ufunc 'positive' output from dtype('int8') to dtype('uint8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(I8(), 1, null, np.empty(new Shape(5), np.uint8)),
                "Cannot cast ufunc 'maximum' output from dtype('int8') to dtype('uint8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(I8(), null, 1, np.empty(new Shape(5), np.uint8)),
                "Cannot cast ufunc 'minimum' output from dtype('int8') to dtype('uint8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(np.array(new[] { 0.5 }), 0, 1, np.empty(new Shape(1), np.int8)),
                "Cannot cast ufunc 'clip' output from dtype('float64') to dtype('int8') with casting rule 'same_kind'");
        }

        /// <summary>
        ///     NumPy's error order: read-only out → the Python-int packing → dtype= input casts → the out cast → the out shape.
        /// </summary>
        [TestMethod]
        public void ErrorOrder_MatchesNumPy()
        {
            var ro = np.broadcast_to(NDArray.Scalar((sbyte)0), new Shape(5));
            var e = Assert.ThrowsExactly<ValueError>(() => np.clip(I8(), 1000, 2000, ro));
            e.Message.Should().Be("output array is read-only");

            ExpectOverflow(() => np.clip(I8(), 1000, 2000, np.empty(new Shape(5), np.@bool)), "Python integer 1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), 1000, 2000, np.empty(new Shape(3), np.int8)), "Python integer 1000 out of bounds for int8");
            ExpectOverflow(() => np.clip(I8(), 1000, 2000, np.empty(new Shape(3), np.uint8), np.uint8), "Python integer 1000 out of bounds for uint8");

            ExpectCastError(() => np.clip(np.array(new[] { 0.5, 2.5, 3.0 }), 1, 2, np.empty(new Shape(3), np.@bool), np.int8),
                "Cannot cast ufunc 'clip' input 0 from dtype('float64') to dtype('int8') with casting rule 'same_kind'");
            ExpectCastError(() => np.clip(I8(), 5, 10, np.empty(new Shape(7), np.uint8)),
                "Cannot cast ufunc 'clip' output from dtype('int8') to dtype('uint8') with casting rule 'same_kind'");
            var shape = Assert.ThrowsExactly<ValueError>(() => np.clip(I8(), 5, 10, np.empty(new Shape(7), np.int8)));
            shape.Message.Should().Be("operands could not be broadcast together with shapes (5,) () () (7,) ");
        }

        /// <summary>Dropped bounds keep NumPy's output layout: K order for a fresh result, a fresh writeable array for a broadcast input.</summary>
        [TestMethod]
        public void Layouts_DroppedBounds()
        {
            var mm = np.arange(12).astype(np.int8).reshape(3, 4);

            var t1 = np.clip(mm.T, -1000, 5);
            t1.Shape.IsFContiguous.Should().BeTrue();
            t1.Should().BeShaped(4, 3);
            t1.ToArray<sbyte>().Should().Equal(0, 4, 5, 1, 5, 5, 2, 5, 5, 3, 5, 5);

            var t2 = np.clip(mm.T, -1000, 1000);
            t2.Shape.IsFContiguous.Should().BeTrue();
            t2.ToArray<sbyte>().Should().Equal(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11);

            var s = np.clip(mm[":, ::2"], -1000, 1000);
            s.Shape.IsContiguous.Should().BeTrue();
            s.ToArray<sbyte>().Should().Equal(0, 2, 4, 6, 8, 10);

            np.clip(mm["::-1, ::-1"], 2, 1000).ToArray<sbyte>().Should().Equal(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 2);

            // A permuted 3-D input keeps its memory order (NumPy strides (4, 12, 1)).
            var p = np.clip(np.arange(24).astype(np.int8).reshape(2, 3, 4).transpose(new[] { 1, 0, 2 }), -1000, 1000);
            p.strides.Should().Equal(4L, 12L, 1L);

            var bc = np.clip(np.broadcast_to(NDArray.Scalar((sbyte)3), new Shape(2, 3)), -1000, 1000);
            bc.Shape.IsWriteable.Should().BeTrue();
            bc.Shape.IsContiguous.Should().BeTrue();
            bc.ToArray<sbyte>().Should().Equal(3, 3, 3, 3, 3, 3);
        }

        /// <summary>A given out is returned itself, also for an F-contiguous input (NumPy returns out).</summary>
        [TestMethod]
        public void Out_IsReturnedItself_ForAnFContiguousInput()
        {
            var f = np.asfortranarray(np.arange(12.0).reshape(3, 4));
            var o = np.empty(new Shape(3, 4), np.float64);
            var r = np.clip(f, 2, 8, o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.ToArray<double>().Should().Equal(2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8);
        }

        /// <summary>
        ///     Char rides uint16 (NumSharp's NumPy proxy): the drop rule uses [0, 65535], an impossible bound names uint16, and a
        ///     float bound promotes to float64 instead of being truncated into char.
        /// </summary>
        [TestMethod]
        public void CharArray_RidesTheUInt16Proxy()
        {
            var c = np.array(new[] { (char)0, (char)5, (char)200, (char)65535 });
            var r1 = np.clip(c, -1, 70000);
            r1.dtype.Should().Be(np.@char);
            r1.ToArray<char>().Should().Equal((char)0, (char)5, (char)200, (char)65535);
            np.clip(c, 5, 233).ToArray<char>().Should().Equal((char)5, (char)5, (char)200, (char)233);

            ExpectOverflow(() => np.clip(c, 70000, null), "Python integer 70000 out of bounds for uint16");
            ExpectOverflow(() => np.clip(c, null, -1), "Python integer -1 out of bounds for uint16");

            var r2 = np.clip(c, 0.5, 100.5);
            r2.dtype.Should().Be(np.float64);
            r2.ToArray<double>().Should().Equal(0.5, 5.0, 100.5, 100.5);
        }

        /// <summary>ndarray.clip and np.ma.clip are the same _clip.</summary>
        [TestMethod]
        public void NDArrayClipAndMaskedClip()
        {
            I8().clip(-1000, 1000).ToArray<sbyte>().Should().Equal(-128, -5, 0, 5, 127);
            ExpectOverflow(() => I8().clip(1000), "Python integer 1000 out of bounds for int8");
            ExpectOverflow(() => I8().clip(null, -1000), "Python integer -1000 out of bounds for int8");

            var m = np.ma.array(I8(), mask: new[] { false, true, false, false, false });
            var mc = np.ma.clip(m, -1000, 1000);
            mc.data.ToArray<sbyte>().Should().Equal(-128, -5, 0, 5, 127);
            mc.mask.ToArray<bool>().Should().Equal(false, true, false, false, false);
            ExpectOverflow(() => np.ma.clip(m, 1000, null), "Python integer 1000 out of bounds for int8");
        }

        /// <summary>
        ///     NumPy's clip has a CONST fast path for two scalar float32/float64 bounds (clip.cpp
        ///     <c>_npy_clip_const_minmax_</c>): <c>x &lt; lo ? lo : x</c>, then <c>x &gt; hi ? hi : x</c>, which keeps the element
        ///     on a tie. Its general path (array bounds) and one-sided clips (maximum / minimum) take the bound instead. So
        ///     the sign of a zero tied with a ±0 bound depends on the path. Bytes probed against NumPy 2.4.2. The arrays are
        ///     long enough to cross the SIMD body, its remainder and the scalar tail. A strided view takes the iterator
        ///     route; F-ordered and transposed views are dense runs and take the flat route after the layout permutation.
        /// </summary>
        [TestMethod]
        public void SignedZeroTies_FollowNumPysPathPerBoundKind()
        {
            const ulong NegZero = 0x8000000000000000UL;
            // The raw float64 bit patterns of r in C order: a float32 widens exactly, so -0.0 stays 0x8000000000000000 and
            // +0.0 stays 0, which a value comparison (-0.0 == +0.0) could not tell apart.
            static ulong[] Bits(NDArray r) => r.astype(np.float64).ToArray<double>().Select(d => (ulong)BitConverter.DoubleToInt64Bits(d)).ToArray();

            foreach (int n in new[] { 1, 7, 37, 100 })
            {
                var x = np.full(new Shape(n), -0.0);

                // Two scalar bounds: the const fast path keeps -0.0 at a 0 bound (NumPy: -0.0).
                Bits(np.clip(x, 0, 1)).Should().AllBeEquivalentTo(NegZero, $"n={n}: clip(-0.0, 0, 1) keeps the element");
                Bits(np.clip(x, 0.0, 1.0)).Should().AllBeEquivalentTo(NegZero);
                Bits(np.clip(x.astype(np.float32), 0, 1)).Should().AllBeEquivalentTo(NegZero);
                // clip(+0.0, lo, -0.0) keeps +0.0 too (NumPy: +0.0, where Min() would give -0.0).
                Bits(np.clip(np.zeros(new Shape(n), np.float64), -1, -0.0)).Should().AllBeEquivalentTo(0UL);

                // One-sided: maximum / minimum, whose ties take the bound (NumPy: +0.0).
                Bits(np.clip(x, 0, null)).Should().AllBeEquivalentTo(0UL, "maximum(-0.0, 0) is the bound");
                // Array bounds: the general MIN(MAX()) path takes the bound (NumPy: +0.0).
                Bits(np.clip(x, np.zeros(new Shape(n), np.float64), np.ones(new Shape(n), np.float64))).Should().AllBeEquivalentTo(0UL);
            }

            // Every layout agrees (the strided loop makes the same switch as the contiguous kernel).
            var m = np.full(new Shape(4, 6), -0.0);
            Bits(np.clip(m.T, 0, 1)).Should().AllBeEquivalentTo(NegZero);
            Bits(np.clip(m[":, ::2"], 0, 1)).Should().AllBeEquivalentTo(NegZero);
            Bits(np.clip(m["::-1"], 0, 1)).Should().AllBeEquivalentTo(NegZero);

            // NaN elements pass through, min > max still yields max, and ordinary clamping is unchanged.
            var v = np.array(new[] { double.NaN, -5.0, 0.5, 5.0 });
            var r = np.clip(v, 0, 1).ToArray<double>();
            double.IsNaN(r[0]).Should().BeTrue();
            r.Skip(1).Should().Equal(0.0, 0.5, 1.0);
            np.clip(v, 3, 1).ToArray<double>().Skip(1).Should().Equal(1.0, 1.0, 1.0);
        }

        /// <summary>Float and complex arrays keep both bounds: a big Python int converts with rounding (float16 → inf).</summary>
        [TestMethod]
        public void InexactArrays_ConvertBigIntsWithRounding()
        {
            var f16 = np.array(new[] { (Half)1.5, (Half)(-2.0), (Half)60000 });
            np.clip(f16, 0, 100000).ToArray<Half>().Should().Equal((Half)1.5, (Half)0, (Half)60000);
            np.clip(f16, 100000, null).ToArray<Half>().Should().Equal(Half.PositiveInfinity, Half.PositiveInfinity, Half.PositiveInfinity);
            np.clip(f16, null, -100000).ToArray<Half>().Should().Equal(Half.NegativeInfinity, Half.NegativeInfinity, Half.NegativeInfinity);

            var f32 = np.array(new[] { 1.5f, -2f });
            const float TwoTo63 = 9223372036854775808f;                       // 2^63 is exact in float32
            np.clip(f32, 9223372036854775808UL, null).ToArray<float>().Should().Equal(TwoTo63, TwoTo63);
            np.clip(np.array(new[] { 1.0 }), ulong.MaxValue, null).ToArray<double>().Should().Equal(1.8446744073709552e19);

            var c = np.array(new[] { new Complex(1, 2), new Complex(-3, 0) });
            np.clip(c, -1000, 1000).ToArray<Complex>().Should().Equal(new Complex(1, 2), new Complex(-3, 0));
        }
    }
}
