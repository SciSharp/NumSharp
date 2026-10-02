using System;
using System.Numerics;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    /// Tests for <c>np.array_equiv</c> and the <c>np.array_equal</c> family (including the
    /// <c>equal_nan</c> parameter). All expected values probed against NumPy 2.4.2.
    ///
    /// <para><c>array_equal</c> requires an EXACT shape match; <c>array_equiv</c> only requires the
    /// shapes to be BROADCAST-consistent. Neither treats NaN as equal by default — <c>array_equal</c>
    /// gains that behaviour through <c>equal_nan=True</c>, while <c>array_equiv</c> has no such option
    /// (matching NumPy).</para>
    /// </summary>
    [TestClass]
    public class np_array_equiv_Test
    {
        // ---------------------------------------------------------------- array_equiv ----

        [TestMethod]
        public void array_equiv_SameShape()
        {
            // >>> np.array_equiv([1,2],[1,2]) -> True ; np.array_equiv([1,2],[1,3]) -> False
            Assert.IsTrue(np.array_equiv(new[] { 1, 2 }, new[] { 1, 2 }));
            Assert.IsFalse(np.array_equiv(new[] { 1, 2 }, new[] { 1, 3 }));
        }

        [TestMethod]
        public void array_equiv_BroadcastConsistent()
        {
            // >>> np.array_equiv([1,2], [[1,2],[1,2]]) -> True  (shape-consistent + all equal)
            Assert.IsTrue(np.array_equiv(np.array(new[] { 1, 2 }), np.array(new[,] { { 1, 2 }, { 1, 2 } })));
            // >>> np.array_equiv([1,2], [[1,2],[1,3]]) -> False (broadcastable, but a value differs)
            Assert.IsFalse(np.array_equiv(np.array(new[] { 1, 2 }), np.array(new[,] { { 1, 2 }, { 1, 3 } })));
        }

        [TestMethod]
        public void array_equiv_NotBroadcastable_IsFalse()
        {
            // >>> np.array_equiv([1,2], [[1,2,1,2],[1,2,1,2]]) -> False (shapes not broadcast-consistent)
            Assert.IsFalse(np.array_equiv(np.array(new[] { 1, 2 }), np.array(new[,] { { 1, 2, 1, 2 }, { 1, 2, 1, 2 } })));
            // >>> np.array_equiv([1,2,3], [1,2]) -> False (1-D lengths differ, not broadcastable)
            Assert.IsFalse(np.array_equiv(new[] { 1, 2, 3 }, new[] { 1, 2 }));
        }

        [TestMethod]
        public void array_equiv_ScalarBroadcast()
        {
            // >>> np.array_equiv(5, [5,5,5]) -> True ; np.array_equiv(5, [5,5,6]) -> False
            Assert.IsTrue(np.array_equiv(5, np.array(new[] { 5, 5, 5 })));
            Assert.IsFalse(np.array_equiv(5, np.array(new[] { 5, 5, 6 })));
        }

        [TestMethod]
        public void array_equiv_ColumnBroadcast()
        {
            // >>> np.array_equiv([[1],[1]], [[1,1],[1,1]]) -> True  ((2,1) broadcasts to (2,2))
            Assert.IsTrue(np.array_equiv(np.array(new[,] { { 1 }, { 1 } }), np.array(new[,] { { 1, 1 }, { 1, 1 } })));
        }

        [TestMethod]
        public void array_equiv_Empty_IsTrue()
        {
            // >>> np.array_equiv([], []) -> True (all() over the empty comparison is vacuously true)
            Assert.IsTrue(np.array_equiv(np.array(new int[0]), np.array(new int[0])));
            // >>> np.array_equiv(np.zeros((0,3)), np.zeros((0,3))) -> True
            Assert.IsTrue(np.array_equiv(np.zeros(new Shape(0, 3)), np.zeros(new Shape(0, 3))));
        }

        [TestMethod]
        public void array_equiv_NaN_NeverEqual()
        {
            // >>> a = np.array([1.,np.nan]); np.array_equiv(a, a) -> False (NaN != NaN, no equal_nan option)
            var a = np.array(new[] { 1.0, np.nan });
            Assert.IsFalse(np.array_equiv(a, a));
        }

        [TestMethod]
        public void array_equiv_MixedDtype_ComparesByValue()
        {
            // >>> np.array_equiv(np.array([1,2],np.int32), np.array([1.,2.])) -> True
            Assert.IsTrue(np.array_equiv(np.array(new[] { 1, 2 }), np.array(new[] { 1.0, 2.0 })));
        }

        // ---------------------------------------------------- array_equal (default) ----

        [TestMethod]
        public void array_equal_ExactShapeRequired()
        {
            // >>> np.array_equal([1,2],[1,2]) -> True
            Assert.IsTrue(np.array_equal(new[] { 1, 2 }, new[] { 1, 2 }));
            // >>> np.array_equal([1,2],[1,2,3]) -> False  (different length)
            Assert.IsFalse(np.array_equal(np.array(new[] { 1, 2 }), np.array(new[] { 1, 2, 3 })));
            // >>> np.array_equal([1,2],[[1,2]]) -> False  (broadcastable but NOT the same shape)
            Assert.IsFalse(np.array_equal(np.array(new[] { 1, 2 }), np.array(new[,] { { 1, 2 } })));
        }

        [TestMethod]
        public void array_equal_NaN_DefaultIsFalse_EvenForSameArray()
        {
            // This pins the NumPy semantics that the old NumSharp shortcut violated:
            // >>> a = np.array([1., np.nan]); np.array_equal(a, a) -> False  (NaN != NaN, no equal_nan)
            var a = np.array(new[] { 1.0, np.nan });
            Assert.IsFalse(np.array_equal(a, a));
            // >>> np.array_equal([1,nan],[1,nan]) -> False
            Assert.IsFalse(np.array_equal(np.array(new[] { 1.0, np.nan }), np.array(new[] { 1.0, np.nan })));
        }

        [TestMethod]
        public void array_equal_Empty_And_Scalar()
        {
            // >>> np.array_equal([], []) -> True
            Assert.IsTrue(np.array_equal(np.array(new int[0]), np.array(new int[0])));
            // >>> np.array_equal(np.array(5), np.array(5)) -> True ; vs np.array([5]) -> False (shape)
            Assert.IsTrue(np.array_equal(NDArray.Scalar(5), NDArray.Scalar(5)));
            Assert.IsFalse(np.array_equal(NDArray.Scalar(5), np.array(new[] { 5 })));
        }

        [TestMethod]
        public void array_equal_MixedDtype_SameValues()
        {
            // >>> np.array_equal(np.array([1,2],np.int32), np.array([1,2],np.int64)) -> True
            Assert.IsTrue(np.array_equal(np.array(new[] { 1, 2 }), np.array(new long[] { 1, 2 })));
        }

        // ------------------------------------------------- array_equal (equal_nan) ----

        [TestMethod]
        public void array_equal_EqualNan_SamePositions()
        {
            // >>> np.array_equal([1,nan],[1,nan], equal_nan=True) -> True
            Assert.IsTrue(np.array_equal(np.array(new[] { 1.0, np.nan }), np.array(new[] { 1.0, np.nan }), equal_nan: true));
        }

        [TestMethod]
        public void array_equal_EqualNan_SameObject()
        {
            // >>> a = np.array([1.,np.nan]); np.array_equal(a, a, equal_nan=True) -> True
            var a = np.array(new[] { 1.0, np.nan });
            Assert.IsTrue(np.array_equal(a, a, equal_nan: true));
        }

        [TestMethod]
        public void array_equal_EqualNan_DifferentPositions_IsFalse()
        {
            // >>> np.array_equal([1,nan],[nan,1], equal_nan=True) -> False (NaN at different slots)
            Assert.IsFalse(np.array_equal(np.array(new[] { 1.0, np.nan }), np.array(new[] { np.nan, 1.0 }), equal_nan: true));
        }

        [TestMethod]
        public void array_equal_EqualNan_FiniteDiffers_IsFalse()
        {
            // >>> np.array_equal([1,nan,3],[1,nan,4], equal_nan=True) -> False (NaN slots match, 3 != 4)
            Assert.IsFalse(np.array_equal(np.array(new[] { 1.0, np.nan, 3.0 }), np.array(new[] { 1.0, np.nan, 4.0 }), equal_nan: true));
        }

        [TestMethod]
        public void array_equal_EqualNan_IntegerDtype()
        {
            // Integers cannot hold NaN, so equal_nan has no effect (fast path):
            // >>> np.array_equal([1,2],[1,2], equal_nan=True) -> True ; ...[1,3] -> False
            Assert.IsTrue(np.array_equal(np.array(new[] { 1, 2 }), np.array(new[] { 1, 2 }), equal_nan: true));
            Assert.IsFalse(np.array_equal(np.array(new[] { 1, 2 }), np.array(new[] { 1, 3 }), equal_nan: true));
        }

        [TestMethod]
        public void array_equal_EqualNan_Float32AndHalf()
        {
            // >>> np.array_equal([1.,nan],[1.,nan], equal_nan=True) -> True at float32 and float16 too
            Assert.IsTrue(np.array_equal(np.array(new[] { 1.0f, float.NaN }), np.array(new[] { 1.0f, float.NaN }), equal_nan: true));
            Assert.IsTrue(np.array_equal(np.array(new[] { (Half)1, Half.NaN }), np.array(new[] { (Half)1, Half.NaN }), equal_nan: true));
        }

        [TestMethod]
        public void array_equal_EqualNan_Complex_EitherComponentNaN()
        {
            // NumPy: a complex value counts as NaN when EITHER component is NaN, so nan+1j and 1+nanj
            // are considered equal under equal_nan.
            // >>> np.array_equal([nan+1j],[1+nanj], equal_nan=True) -> True ; equal_nan=False -> False
            var a = np.array(new[] { new Complex(double.NaN, 1.0) });
            var b = np.array(new[] { new Complex(1.0, double.NaN) });
            Assert.IsTrue(np.array_equal(a, b, equal_nan: true));
            Assert.IsFalse(np.array_equal(a, b, equal_nan: false));

            // A NaN-holding complex is not equal to a fully finite one (isnan positions differ).
            // >>> np.array_equal([nan+2j],[3+4j], equal_nan=True) -> False
            Assert.IsFalse(np.array_equal(np.array(new[] { new Complex(double.NaN, 2.0) }),
                                          np.array(new[] { new Complex(3.0, 4.0) }), equal_nan: true));
        }

        // -------------------------------------------------------------- layouts ----

        [TestMethod]
        public void array_equal_NonContiguousLayouts()
        {
            var src = np.arange(1, 13).reshape(3, 4);

            // Transposed (strided) view equals its own copy.
            Assert.IsTrue(np.array_equal(src.T, src.T.copy()));
            // Reversed (negative-stride) view equals its copy.
            Assert.IsTrue(np.array_equal(src["::-1"], src["::-1"].copy()));
            // Column-strided view equals its copy.
            Assert.IsTrue(np.array_equal(src[":, ::2"], src[":, ::2"].copy()));
            // A transposed view is NOT the same SHAPE as the original (3,4) vs (4,3).
            Assert.IsFalse(np.array_equal(src, src.T));
        }

        [TestMethod]
        public void array_equiv_BroadcastRowView()
        {
            // A (1,4) row is equivalent to a (3,4) block that is that row tiled down.
            var row = np.arange(1, 5).reshape(1, 4);
            var tiled = np.array(new[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 }).reshape(3, 4);
            Assert.IsTrue(np.array_equiv(row, tiled));
        }

        // ---------------------------------------------------------------- null ----

        [TestMethod]
        public void array_equal_And_equiv_NullOperands()
        {
            // Null "arrays" mirror the NDArray == null convention: both-null is equal, one-null is not.
            Assert.IsTrue(np.array_equal((NDArray)null, (NDArray)null));
            Assert.IsFalse(np.array_equal(np.array(new[] { 1 }), (NDArray)null));
            Assert.IsTrue(np.array_equiv((NDArray)null, (NDArray)null));
            Assert.IsFalse(np.array_equiv((NDArray)null, np.array(new[] { 1 })));
        }
    }
}
