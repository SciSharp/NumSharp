using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Logic
{
    public class NpLogicalReductionIteratorTests
    {
        [Test]
        public void All_Axis_OnTransposedView_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([[True, False, True], [True, True, False]]).T
            // >>> np.all(arr, axis=1)
            // array([ True, False, False])
            var arr = np.array(new bool[,] { { true, false, true }, { true, true, false } }).T;

            var result = np.all(arr, axis: 1);
            var expected = np.array(new[] { true, false, false });

            Assert.IsTrue(np.array_equal(result, expected));
        }

        [Test]
        public void All_Axis_OnTransposedView_Keepdims_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([[True, False, True], [True, True, False]]).T
            // >>> np.all(arr, axis=1, keepdims=True)
            // array([[ True],
            //        [False],
            //        [False]])
            var arr = np.array(new bool[,] { { true, false, true }, { true, true, false } }).T;

            var result = np.all(arr, axis: 1, keepdims: true);
            var expected = np.array(new bool[,] { { true }, { false }, { false } });

            Assert.AreEqual(2, result.ndim);
            Assert.IsTrue(np.array_equal(result, expected));
        }

        [Test]
        public void Any_Axis_OnTransposedView_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([[True, False, True], [True, True, False]]).T
            // >>> np.any(arr, axis=0)
            // array([ True,  True])
            var arr = np.array(new bool[,] { { true, false, true }, { true, true, false } }).T;

            var result = np.any(arr, axis: 0);
            var expected = np.array(new[] { true, true });

            Assert.IsTrue(np.array_equal(result, expected));
        }

        [Test]
        public void Any_Axis_OnTransposedView_Keepdims_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([[True, False, True], [True, True, False]]).T
            // >>> np.any(arr, axis=0, keepdims=True)
            // array([[ True,  True]])
            var arr = np.array(new bool[,] { { true, false, true }, { true, true, false } }).T;

            var result = np.any(arr, axis: 0, keepdims: true);
            var expected = np.array(new bool[,] { { true, true } });

            Assert.AreEqual(2, result.ndim);
            Assert.IsTrue(np.array_equal(result, expected));
        }

        [Test]
        public void All_EmptyAxisReduction_UsesIdentity()
        {
            // NumPy 2.4.2:
            // >>> a = np.zeros((0, 3), dtype=np.bool_)
            // >>> np.all(a, axis=0)
            // array([ True,  True,  True])
            // >>> b = np.zeros((2, 0), dtype=np.bool_)
            // >>> np.all(b, axis=1)
            // array([ True,  True])
            var a = np.zeros(new long[] { 0, 3 }, NPTypeCode.Boolean);
            var b = np.zeros(new long[] { 2, 0 }, NPTypeCode.Boolean);

            var result0 = np.all(a, axis: 0);
            var result1 = np.all(b, axis: 1);

            Assert.IsTrue(np.array_equal(result0, np.array(new[] { true, true, true })));
            Assert.IsTrue(np.array_equal(result1, np.array(new[] { true, true })));
        }

        [Test]
        public void Any_EmptyAxisReduction_UsesIdentity()
        {
            // NumPy 2.4.2:
            // >>> a = np.zeros((0, 3), dtype=np.bool_)
            // >>> np.any(a, axis=0)
            // array([False, False, False])
            // >>> b = np.zeros((2, 0), dtype=np.bool_)
            // >>> np.any(b, axis=1)
            // array([False, False])
            var a = np.zeros(new long[] { 0, 3 }, NPTypeCode.Boolean);
            var b = np.zeros(new long[] { 2, 0 }, NPTypeCode.Boolean);

            var result0 = np.any(a, axis: 0);
            var result1 = np.any(b, axis: 1);

            Assert.IsTrue(np.array_equal(result0, np.array(new[] { false, false, false })));
            Assert.IsTrue(np.array_equal(result1, np.array(new[] { false, false })));
        }

        [Test]
        public void All_BroadcastColumn_Axis1_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.broadcast_to(np.array([[True], [False], [True]], dtype=np.bool_), (3, 4))
            // >>> np.all(arr, axis=1)
            // array([ True, False,  True])
            var col = np.array(new bool[,] { { true }, { false }, { true } });
            var arr = np.broadcast_to(col, new Shape(3, 4));

            var result = np.all(arr, axis: 1);
            var expected = np.array(new[] { true, false, true });

            Assert.IsTrue(np.array_equal(result, expected));
        }

        [Test]
        public void Any_BroadcastColumn_Axis0_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.broadcast_to(np.array([[True], [False], [True]], dtype=np.bool_), (3, 4))
            // >>> np.any(arr, axis=0)
            // array([ True,  True,  True,  True])
            var col = np.array(new bool[,] { { true }, { false }, { true } });
            var arr = np.broadcast_to(col, new Shape(3, 4));

            var result = np.any(arr, axis: 0);
            var expected = np.array(new[] { true, true, true, true });

            Assert.IsTrue(np.array_equal(result, expected));
        }

        [Test]
        public void All_ChainedTransposedReversedView_Axis1_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([[True, False, True], [True, True, False], [False, False, False]]).T[:, ::-1]
            // >>> np.all(arr, axis=1)
            // array([False, False, False])
            var arr = np.array(new bool[,]
            {
                { true, false, true },
                { true, true, false },
                { false, false, false }
            }).T[":, ::-1"];

            var result = np.all(arr, axis: 1);
            var expected = np.array(new[] { false, false, false });

            Assert.IsTrue(np.array_equal(result, expected));
        }

        [Test]
        public void Any_ChainedTransposedReversedView_Axis0_Keepdims_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([[True, False, True], [True, True, False], [False, False, False]]).T[:, ::-1]
            // >>> np.any(arr, axis=0, keepdims=True)
            // array([[False,  True,  True]])
            var arr = np.array(new bool[,]
            {
                { true, false, true },
                { true, true, false },
                { false, false, false }
            }).T[":, ::-1"];

            var result = np.any(arr, axis: 0, keepdims: true);
            var expected = np.array(new bool[,] { { false, true, true } });

            Assert.AreEqual(2, result.ndim);
            Assert.IsTrue(np.array_equal(result, expected));
        }

        [Test]
        public void All_EmptySliceView_Axis1_UsesIdentity()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([[True, False, True], [False, True, False], [True, True, True]]).T[:, :0]
            // >>> np.all(arr, axis=1)
            // array([ True,  True,  True])
            var arr = np.array(new bool[,]
            {
                { true, false, true },
                { false, true, false },
                { true, true, true }
            }).T[":, :0"];

            var result = np.all(arr, axis: 1);
            var expected = np.array(new[] { true, true, true });

            Assert.IsTrue(np.array_equal(result, expected));
        }

        [Test]
        public void Any_EmptySliceView_Axis1_UsesIdentity()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([[True, False, True], [False, True, False], [True, True, True]]).T[:, :0]
            // >>> np.any(arr, axis=1)
            // array([False, False, False])
            var arr = np.array(new bool[,]
            {
                { true, false, true },
                { false, true, false },
                { true, true, true }
            }).T[":, :0"];

            var result = np.any(arr, axis: 1);
            var expected = np.array(new[] { false, false, false });

            Assert.IsTrue(np.array_equal(result, expected));
        }
    }
}
