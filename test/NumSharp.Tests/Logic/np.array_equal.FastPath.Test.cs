using System;
using System.Numerics;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    /// Pins the fused fast path (<c>EqualityScan</c>) that <c>np.array_equal</c> / <c>np.array_equiv</c>
    /// take for same-shape, same-dtype, dense-contiguous operands — a single early-exiting SIMD pass
    /// instead of the <c>np.all(a == b)</c> composition. These tests target the edges the fast path could
    /// get wrong: the NaN / signed-zero traps (a raw-byte scan would mis-handle both, so float/complex
    /// use IEEE value comparison), the both-F-contiguous layout, early-exit correctness, and the
    /// Half/Decimal fallback. Every expected value equals what the composition (and NumPy 2.4.2) produce.
    /// </summary>
    [TestClass]
    public class np_array_equal_FastPath_Test
    {
        // The fast path must agree with the composition for EVERY comparison — this is the invariant.
        private static void AssertMatchesComposition(NDArray a, NDArray b)
        {
            bool fused = np.array_equal(a, b);           // fast path when eligible
            bool composition = np.all(a == b);           // reference (equal shapes here)
            Assert.AreEqual(composition, fused,
                $"fused array_equal disagreed with np.all(a==b): fused={fused}, composition={composition}");
        }

        [TestMethod]
        public void FastPath_Float_NaN_IsNotEqual()
        {
            // The raw-byte trap: two NaNs share a bit pattern but must compare UNEQUAL (equal_nan=False).
            var a = np.array(new[] { 1.0, double.NaN, 3.0 });
            AssertMatchesComposition(a, np.array(new[] { 1.0, double.NaN, 3.0 }));
            Assert.IsFalse(np.array_equal(a, np.array(new[] { 1.0, double.NaN, 3.0 })));
            // float32 too
            Assert.IsFalse(np.array_equal(np.array(new[] { 1.0f, float.NaN }), np.array(new[] { 1.0f, float.NaN })));
        }

        [TestMethod]
        public void FastPath_Float_SignedZero_IsEqual()
        {
            // The other byte trap: -0.0 and +0.0 differ in bits but must compare EQUAL.
            AssertMatchesComposition(np.array(new[] { -0.0, 0.0 }), np.array(new[] { 0.0, -0.0 }));
            Assert.IsTrue(np.array_equal(np.array(new[] { -0.0, 0.0 }), np.array(new[] { 0.0, -0.0 })));
            Assert.IsTrue(np.array_equal(np.array(new[] { -0.0f, 0.0f }), np.array(new[] { 0.0f, -0.0f })));
        }

        [TestMethod]
        public void FastPath_Complex_NaNComponent_And_SignedZero()
        {
            // Complex rides the 2×double reinterpret: a NaN in either component ⇒ unequal; ±0 components equal.
            Assert.IsFalse(np.array_equal(np.array(new[] { new Complex(double.NaN, 2) }),
                                          np.array(new[] { new Complex(double.NaN, 2) })));
            Assert.IsTrue(np.array_equal(np.array(new[] { new Complex(1, -0.0) }),
                                         np.array(new[] { new Complex(1, 0.0) })));
        }

        [TestMethod]
        public void FastPath_Infinity()
        {
            Assert.IsTrue(np.array_equal(np.array(new[] { double.PositiveInfinity, 1.0 }),
                                         np.array(new[] { double.PositiveInfinity, 1.0 })));
            Assert.IsFalse(np.array_equal(np.array(new[] { double.PositiveInfinity }),
                                          np.array(new[] { double.NegativeInfinity })));
        }

        [TestMethod]
        public void FastPath_EarlyExit_MismatchAtVariousPositions()
        {
            // Correctness of the early-exit: a single differing element anywhere returns false.
            foreach (int pos in new[] { 0, 1, 500, 4095, 4096, 9999 })
            {
                var a = np.arange(10000).astype(NPTypeCode.Int32);
                var b = np.arange(10000).astype(NPTypeCode.Int32);
                b[pos] = np.array(new[] { -1 })[0];
                Assert.IsFalse(np.array_equal(a, b), $"expected false with a diff at {pos}");
            }
        }

        [TestMethod]
        public void FastPath_BothFContiguous()
        {
            var c = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
            var f1 = np.asfortranarray(c);
            var f2 = np.asfortranarray(np.arange(12).astype(NPTypeCode.Double).reshape(3, 4));
            Assert.IsTrue(np.array_equal(f1, f2));
            var f3 = np.asfortranarray(np.arange(12).astype(NPTypeCode.Double).reshape(3, 4));
            f3[1, 2] = np.array(new[] { -1.0 })[0];
            Assert.IsFalse(np.array_equal(f1, f3));
        }

        [TestMethod]
        public void FastPath_AllFastDtypes_Contiguous()
        {
            // Every dtype the fast path handles (all but Half/Decimal): equal → true, one diff → false.
            foreach (var tc in new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex
            })
            {
                var a = np.arange(1, 13).astype(tc).reshape(3, 4);
                var b = np.arange(1, 13).astype(tc).reshape(3, 4);
                Assert.IsTrue(np.array_equal(a, b), $"{tc} contiguous-equal should be true");
                if (tc == NPTypeCode.Boolean) continue; // arange 1..12 → all-true bools; a diff can't show
                var c = np.arange(1, 13).astype(tc).reshape(3, 4);
                c[0, 0] = np.array(new[] { 99 }).astype(tc)[0];
                Assert.IsFalse(np.array_equal(a, c), $"{tc} one-diff should be false");
            }
        }

        [TestMethod]
        public void Fallback_HalfAndDecimal_StillCorrect()
        {
            // Half and Decimal are NOT eligible for the fast path (fall back to the composition) — still correct.
            Assert.IsTrue(np.array_equal(np.array(new[] { (Half)1, (Half)2 }), np.array(new[] { (Half)1, (Half)2 })));
            Assert.IsFalse(np.array_equal(np.array(new[] { (Half)1, (Half)2 }), np.array(new[] { (Half)1, (Half)9 })));
            // Half NaN still not equal (equal_nan=False), via the composition.
            Assert.IsFalse(np.array_equal(np.array(new[] { (Half)1, Half.NaN }), np.array(new[] { (Half)1, Half.NaN })));
            Assert.IsTrue(np.array_equal(np.array(new decimal[] { 1m, 2m }), np.array(new decimal[] { 1m, 2m })));
            Assert.IsFalse(np.array_equal(np.array(new decimal[] { 1m, 2m }), np.array(new decimal[] { 1m, 9m })));
        }

        [TestMethod]
        public void FastPath_ArrayEquiv_EqualShape_Vs_Broadcast()
        {
            // Equal shapes take the fast path; a genuine broadcast (different shapes) falls back — both correct.
            Assert.IsTrue(np.array_equiv(np.arange(12).reshape(3, 4), np.arange(12).reshape(3, 4)));
            Assert.IsFalse(np.array_equiv(np.arange(12).reshape(3, 4), Add1(np.arange(12).reshape(3, 4))));
            // broadcast (fallback path)
            Assert.IsTrue(np.array_equiv(np.array(new[] { 1, 2 }), np.array(new[,] { { 1, 2 }, { 1, 2 } })));
            Assert.IsFalse(np.array_equiv(np.array(new[] { 1, 2 }), np.array(new[,] { { 1, 2 }, { 1, 3 } })));
        }

        private static NDArray Add1(NDArray a) { a[0, 0] = np.array(new[] { 999 })[0]; return a; }

        [TestMethod]
        public void FastPath_ContiguousSlice_WithOffset()
        {
            // A contiguous slice has a non-zero Shape.offset; the fast path must read from Address + offset.
            var big = np.arange(100).astype(NPTypeCode.Int32);
            var s1 = big["10:20"];                 // contiguous, offset 10
            var s2 = np.arange(10, 20).astype(NPTypeCode.Int32);
            Assert.IsTrue(np.array_equal(s1, s2));
            var s3 = np.arange(10, 20).astype(NPTypeCode.Int32);
            s3[5] = np.array(new[] { -1 })[0];
            Assert.IsFalse(np.array_equal(s1, s3));
        }
    }
}
