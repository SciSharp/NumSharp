using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    ///     <c>np.matmul</c> on a stacked (≥3-D) operand carrying a zero-sized extent. Every
    ///     expectation here was read off NumPy 2.4.2, not inferred.
    ///     <para>
    ///     NumPy's gufunc signature is <c>(n?,k),(k,m?)->(n?,m?)</c>, so a zero lands in one of two
    ///     places and the two behave differently:
    ///     </para>
    ///     <list type="bullet">
    ///       <item>a zero in a STACK dim, or in <c>n</c> / <c>m</c>, makes the RESULT empty —
    ///             shape carries the 0 through, size is 0, nothing is computed;</item>
    ///       <item>a zero in <c>k</c> alone leaves the result NON-empty and every entry is an
    ///             EMPTY SUM, which is exactly zero. NumPy's <c>matmul_inner_noblas</c> stores 0
    ///             into the output cell before its (zero-trip) accumulation loop, so this holds
    ///             even when the operands are nan/inf.</item>
    ///     </list>
    ///     The 2-D and <c>np.dot</c> paths already got this right; the whole family used to throw
    ///     from <c>DefaultEngine.BatchedMatmul</c> — <c>InvalidOperationException</c> for an empty
    ///     batch odometer, <c>IndexOutOfRangeException</c> for a sub-view of a zero-sized operand,
    ///     and <c>IncorrectShapeException</c> where <c>0</c> broadcast against <c>1</c> gave 1.
    /// </summary>
    [TestClass]
    public class np_matmul_zerosized_test
    {
        #region shape parity — probed against numpy 2.4.2

        [TestMethod]
        public void ThreeD_ZeroInEveryPosition_MatchesNumpyShape()
        {
            // numpy: a.shape @ b.shape -> expected
            AssertMatmulShape(new long[] { 0, 3, 4 }, new long[] { 0, 4, 5 }, new long[] { 0, 3, 5 }); // zero batch
            AssertMatmulShape(new long[] { 2, 3, 0 }, new long[] { 2, 0, 5 }, new long[] { 2, 3, 5 }); // zero k
            AssertMatmulShape(new long[] { 2, 0, 4 }, new long[] { 2, 4, 5 }, new long[] { 2, 0, 5 }); // zero n
            AssertMatmulShape(new long[] { 2, 3, 4 }, new long[] { 2, 4, 0 }, new long[] { 2, 3, 0 }); // zero m
            AssertMatmulShape(new long[] { 0, 0, 0 }, new long[] { 0, 0, 0 }, new long[] { 0, 0, 0 });
            AssertMatmulShape(new long[] { 0, 3, 0 }, new long[] { 0, 0, 5 }, new long[] { 0, 3, 5 });
            AssertMatmulShape(new long[] { 2, 0, 0 }, new long[] { 2, 0, 5 }, new long[] { 2, 0, 5 });
            AssertMatmulShape(new long[] { 2, 0, 4 }, new long[] { 2, 4, 0 }, new long[] { 2, 0, 0 });
            AssertMatmulShape(new long[] { 2, 3, 0 }, new long[] { 2, 0, 0 }, new long[] { 2, 3, 0 });
        }

        [TestMethod]
        public void FourD_ZeroInEveryPosition_MatchesNumpyShape()
        {
            AssertMatmulShape(new long[] { 0, 2, 3, 4 }, new long[] { 0, 2, 4, 5 }, new long[] { 0, 2, 3, 5 });
            AssertMatmulShape(new long[] { 2, 0, 3, 4 }, new long[] { 2, 0, 4, 5 }, new long[] { 2, 0, 3, 5 });
            AssertMatmulShape(new long[] { 2, 3, 4, 0 }, new long[] { 2, 3, 0, 5 }, new long[] { 2, 3, 4, 5 }); // zero k
            AssertMatmulShape(new long[] { 2, 3, 0, 4 }, new long[] { 2, 3, 4, 5 }, new long[] { 2, 3, 0, 5 });
            AssertMatmulShape(new long[] { 2, 3, 4, 5 }, new long[] { 2, 3, 5, 0 }, new long[] { 2, 3, 4, 0 });
        }

        [TestMethod]
        public void ZeroStackDim_AgainstABroadcast2DOperand_BothOrders()
        {
            // The 2-D operand has NO stack dim at all, so it is right-aligned against a batch of 0.
            AssertMatmulShape(new long[] { 0, 3, 4 }, new long[] { 4, 5 }, new long[] { 0, 3, 5 });
            AssertMatmulShape(new long[] { 3, 4 }, new long[] { 0, 4, 5 }, new long[] { 0, 3, 5 });
            AssertMatmulShape(new long[] { 0, 3, 0 }, new long[] { 0, 5 }, new long[] { 0, 3, 5 });
            AssertMatmulShape(new long[] { 3, 0 }, new long[] { 0, 0, 5 }, new long[] { 0, 3, 5 });
            AssertMatmulShape(new long[] { 0, 2, 3, 4 }, new long[] { 4, 5 }, new long[] { 0, 2, 3, 5 });
            AssertMatmulShape(new long[] { 2, 0, 3, 4 }, new long[] { 4, 5 }, new long[] { 2, 0, 3, 5 });
        }

        [TestMethod]
        public void ZeroAgainstOne_InAStackDim_StretchesToZeroNotOne()
        {
            // np.broadcast_shapes((0,), (1,)) is (0,) — the length-1 axis takes the OTHER extent,
            // and that extent may be 0. Using Math.Max here yielded 1 and then demanded a (0,n,k)
            // operand be broadcast to (1,n,k), which is not a legal stretch.
            AssertMatmulShape(new long[] { 0, 3, 4 }, new long[] { 1, 4, 5 }, new long[] { 0, 3, 5 });
            AssertMatmulShape(new long[] { 1, 3, 4 }, new long[] { 0, 4, 5 }, new long[] { 0, 3, 5 });
            AssertMatmulShape(new long[] { 1, 1, 3, 4 }, new long[] { 0, 1, 4, 5 }, new long[] { 0, 1, 3, 5 });
            // a 0 and a >1 stretch in the SAME call, on different axes
            AssertMatmulShape(new long[] { 1, 0, 3, 4 }, new long[] { 2, 1, 4, 5 }, new long[] { 2, 0, 3, 5 });
        }

        [TestMethod]
        public void ZeroAgainstMoreThanOne_InAStackDim_StillRaises()
        {
            // 0 is compatible with 0 and 1 only. NumPy raises ValueError; NumSharp's house type for
            // a broadcast failure is IncorrectShapeException.
            AssertMatmulRaises(new long[] { 0, 3, 4 }, new long[] { 2, 4, 5 });
            AssertMatmulRaises(new long[] { 2, 3, 4 }, new long[] { 0, 4, 5 });
            AssertMatmulRaises(new long[] { 3, 0, 3, 4 }, new long[] { 3, 2, 4, 5 });
        }

        [TestMethod]
        public void OneDPromotion_AroundAZeroExtent_MatchesNumpy()
        {
            // A 1-D lhs is prepended a 1 and a 1-D rhs appended a 1; the inserted axis is squeezed
            // back out afterwards, so these land on the batched path with a degenerate core.
            AssertMatmulShape(new long[] { 0, 3, 4 }, new long[] { 4 }, new long[] { 0, 3 });
            AssertMatmulShape(new long[] { 3 }, new long[] { 0, 3, 4 }, new long[] { 0, 4 });
            AssertMatmulShape(new long[] { 2, 3, 0 }, new long[] { 0 }, new long[] { 2, 3 });    // zero k
            AssertMatmulShape(new long[] { 0 }, new long[] { 2, 0, 5 }, new long[] { 2, 5 });    // zero k
            AssertMatmulShape(new long[] { 2, 0, 4 }, new long[] { 4 }, new long[] { 2, 0 });
            AssertMatmulShape(new long[] { 4 }, new long[] { 2, 4, 0 }, new long[] { 2, 0 });
        }

        #endregion

        #region values — the k == 0 empty sum

        [TestMethod]
        public void ZeroK_ProducesExactZeros_AtEveryDtype()
        {
            // NumPy gives an all-zero result for every dtype it has; Char and Decimal have no NumPy
            // analog but follow the same law (the empty sum is default(T), whose bits are all-zero).
            foreach (var dtype in AllDtypes)
            {
                var r = np.matmul(Ramp(dtype, 2, 3, 0), Ramp(dtype, 2, 0, 5));
                Assert.AreEqual(dtype, r.typecode, $"{dtype}: dtype");
                CollectionAssert.AreEqual(new long[] { 2, 3, 5 }, r.shape, $"{dtype}: shape");
                Assert.AreEqual(30, r.size, $"{dtype}: size");
                AssertEveryElementIsZero(r, $"{dtype}: values");
            }
        }

        [TestMethod]
        public void ZeroK_WithNanAndInfOperands_IsStillExactlyPositiveZero()
        {
            // An empty sum never touches the operands, so nan @ inf is +0.0 — NOT nan. Probed.
            var a = np.full(new Shape(2, 3, 0), double.NaN, NPTypeCode.Double);
            var b = np.full(new Shape(2, 0, 5), double.PositiveInfinity, NPTypeCode.Double);

            var r = np.matmul(a, b);

            CollectionAssert.AreEqual(new long[] { 2, 3, 5 }, r.shape);
            for (long i = 0; i < r.size; i++)
                Assert.AreEqual(0L, BitConverter.DoubleToInt64Bits((double)r.GetAtIndex(i)),
                    $"element {i} is not +0.0");
        }

        [TestMethod]
        public void ZeroK_IsZeroEvenWhenTheAllocatorHandsBackDirtyMemory()
        {
            // No kernel runs on the k == 0 path, so the result comes straight from the allocation.
            // A dirty-buffer bug passes on the first run and fails under load — so run it many
            // times, interleaved with allocations that deliberately poison memory of the same size.
            for (int rep = 0; rep < 400; rep++)
            {
                var poison = new NDArray(NPTypeCode.Single, new Shape(2, 3, 5), fillZeros: false);
                for (long i = 0; i < poison.size; i++)
                    poison.SetAtIndex(-7777.5f, i);
                GC.KeepAlive(poison);

                var r = np.matmul(Ramp(NPTypeCode.Single, 2, 3, 0), Ramp(NPTypeCode.Single, 2, 0, 5));

                Assert.AreEqual(30, r.size, $"rep {rep}: size");
                for (long i = 0; i < r.size; i++)
                    Assert.AreEqual(0, BitConverter.SingleToInt32Bits((float)r.GetAtIndex(i)),
                        $"rep {rep}, element {i} is not exactly 0.0f");
            }
        }

        [TestMethod]
        public void ZeroK_PromotesTheResultDtype_LikeAnOrdinaryProduct()
        {
            // A degenerate product still goes through result_type — the shortcut must not skip it.
            AssertPromotes(NPTypeCode.Int32, NPTypeCode.Double, NPTypeCode.Double);
            AssertPromotes(NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Double);
            AssertPromotes(NPTypeCode.Boolean, NPTypeCode.Int32, NPTypeCode.Int32);
            AssertPromotes(NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.Int16);
            AssertPromotes(NPTypeCode.Single, NPTypeCode.Complex, NPTypeCode.Complex);
        }

        [TestMethod]
        public void EmptyResult_IsEmptyAtEveryDtype()
        {
            foreach (var dtype in AllDtypes)
            {
                var r = np.matmul(Ramp(dtype, 0, 3, 4), Ramp(dtype, 0, 4, 5));
                Assert.AreEqual(dtype, r.typecode, $"{dtype}: dtype");
                CollectionAssert.AreEqual(new long[] { 0, 3, 5 }, r.shape, $"{dtype}: shape");
                Assert.AreEqual(0, r.size, $"{dtype}: size");
            }
        }

        #endregion

        #region the ordinary path must not regress

        [TestMethod]
        public void NonDegenerateBatchedMatmul_StillEqualsThePerBatch2DProducts()
        {
            var a = Ramp(NPTypeCode.Double, 2, 3, 4);
            var b = Ramp(NPTypeCode.Double, 2, 4, 5);

            var stacked = np.matmul(a, b);

            CollectionAssert.AreEqual(new long[] { 2, 3, 5 }, stacked.shape);
            for (long t = 0; t < 2; t++)
            {
                var expected = np.matmul(a.GetData(new[] { t }), b.GetData(new[] { t }));
                var actual = stacked.GetData(new[] { t });
                for (long i = 0; i < expected.size; i++)
                    Assert.AreEqual(BitConverter.DoubleToInt64Bits((double)expected.GetAtIndex(i)),
                                    BitConverter.DoubleToInt64Bits((double)actual.GetAtIndex(i)),
                                    $"batch {t}, element {i}");
            }
        }

        [TestMethod]
        public void NonDegenerateBroadcastStack_StillBroadcastsTheOneAxis()
        {
            // The stack-broadcast rule changed from Math.Max to "the one that isn't 1" — for every
            // non-zero pair those agree, and this pins that they still do.
            AssertMatmulShape(new long[] { 1, 3, 4 }, new long[] { 5, 4, 5 }, new long[] { 5, 3, 5 });
            AssertMatmulShape(new long[] { 5, 3, 4 }, new long[] { 1, 4, 5 }, new long[] { 5, 3, 5 });
            AssertMatmulShape(new long[] { 5, 3, 4 }, new long[] { 4, 5 }, new long[] { 5, 3, 5 });
            AssertMatmulShape(new long[] { 2, 1, 3, 4 }, new long[] { 1, 6, 4, 5 }, new long[] { 2, 6, 3, 5 });

            var r = np.matmul(Ramp(NPTypeCode.Int32, 5, 3, 4), Ramp(NPTypeCode.Int32, 1, 4, 5));
            var one = np.matmul(Ramp(NPTypeCode.Int32, 5, 3, 4).GetData(new long[] { 2 }),
                                Ramp(NPTypeCode.Int32, 1, 4, 5).GetData(new long[] { 0 }));
            var slice = r.GetData(new long[] { 2 });
            for (long i = 0; i < one.size; i++)
                Assert.AreEqual((int)one.GetAtIndex(i), (int)slice.GetAtIndex(i), $"element {i}");
        }

        [TestMethod]
        public void TwoDAndDot_ZeroExtentPaths_AreUnchanged()
        {
            // These were already correct before the fix — pinned so the shortcut cannot capture them.
            var twoDZeroK = np.matmul(Ramp(NPTypeCode.Single, 3, 0), Ramp(NPTypeCode.Single, 0, 5));
            CollectionAssert.AreEqual(new long[] { 3, 5 }, twoDZeroK.shape);
            AssertEveryElementIsZero(twoDZeroK, "2-D zero-k");

            var twoDZeroM = np.matmul(Ramp(NPTypeCode.Single, 0, 4), Ramp(NPTypeCode.Single, 4, 5));
            CollectionAssert.AreEqual(new long[] { 0, 5 }, twoDZeroM.shape);

            var dotEmpty = np.dot(Ramp(NPTypeCode.Single, 0, 3, 4), Ramp(NPTypeCode.Single, 4, 5));
            CollectionAssert.AreEqual(new long[] { 0, 3, 5 }, dotEmpty.shape);

            var dotZeroK = np.dot(Ramp(NPTypeCode.Single, 2, 3, 0), Ramp(NPTypeCode.Single, 0, 5));
            CollectionAssert.AreEqual(new long[] { 2, 3, 5 }, dotZeroK.shape);
            AssertEveryElementIsZero(dotZeroK, "dot zero-k");
        }

        #endregion

        #region helpers

        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
        };

        /// <summary>0,1,2,… reshaped — a zero-sized shape simply yields an empty array.</summary>
        private static NDArray Ramp(NPTypeCode dtype, params long[] dims)
        {
            long n = 1;
            foreach (var d in dims)
                n *= d;
            return np.arange(n).astype(dtype).reshape(dims);
        }

        private static void AssertMatmulShape(long[] a, long[] b, long[] expected)
        {
            var r = np.matmul(Ramp(NPTypeCode.Single, a), Ramp(NPTypeCode.Single, b));
            CollectionAssert.AreEqual(expected, r.shape,
                $"({string.Join(",", a)}) @ ({string.Join(",", b)}) gave ({string.Join(",", r.shape)}), " +
                $"numpy gives ({string.Join(",", expected)})");

            long expectedSize = 1;
            foreach (var d in expected)
                expectedSize *= d;
            Assert.AreEqual(expectedSize, r.size,
                $"({string.Join(",", a)}) @ ({string.Join(",", b)}): size");
        }

        private static void AssertMatmulRaises(long[] a, long[] b)
        {
            Assert.ThrowsExactly<IncorrectShapeException>(
                () => np.matmul(Ramp(NPTypeCode.Single, a), Ramp(NPTypeCode.Single, b)),
                $"({string.Join(",", a)}) @ ({string.Join(",", b)}) should not be broadcastable");
        }

        private static void AssertPromotes(NPTypeCode lhs, NPTypeCode rhs, NPTypeCode expected)
        {
            Assert.AreEqual(expected, np.matmul(Ramp(lhs, 2, 3, 0), Ramp(rhs, 2, 0, 5)).typecode,
                $"{lhs} @ {rhs}: zero-k result dtype");
            Assert.AreEqual(expected, np.matmul(Ramp(lhs, 0, 3, 4), Ramp(rhs, 0, 4, 5)).typecode,
                $"{lhs} @ {rhs}: empty result dtype");
        }

        /// <summary>Compares through Decimal so every one of the 15 dtypes reads exactly.</summary>
        private static void AssertEveryElementIsZero(NDArray r, string because)
        {
            var asDecimal = r.typecode == NPTypeCode.Complex
                ? null
                : r.astype(NPTypeCode.Decimal);

            for (long i = 0; i < r.size; i++)
            {
                if (asDecimal is null)
                    Assert.AreEqual(System.Numerics.Complex.Zero,
                        (System.Numerics.Complex)r.GetAtIndex(i), $"{because}: element {i}");
                else
                    Assert.AreEqual(0m, (decimal)asDecimal.GetAtIndex(i), $"{because}: element {i}");
            }
        }

        #endregion
    }
}
