using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.APIs
{
    /// <summary>
    /// np.nested_iters parity gate. The multi_index path (the documented primary use) is bit-exact
    /// with NumPy 2.4.2; every expectation was probed against it. The value-stream ORDER without
    /// multi_index is a documented [Misaligned] divergence (a pre-existing NDIter op_axes+order
    /// limitation, not a property of the nesting).
    /// </summary>
    [TestClass]
    public class np_nested_iters_tests
    {
        private static long V(np.NDIterator it, int op = 0) => Convert.ToInt64(it[op].GetAtIndex(0));

        [TestMethod]
        public void SingleOperand_MultiIndex_MatchesNumPy()
        {
            var a = np.arange(12).reshape(2, 3, 2);
            var p = np.nested_iters(a, new[] { new[] { 1 }, new[] { 0, 2 } }, flags: new[] { "multi_index" });
            try
            {
                var outerMi = new List<string>();
                var innerMi = new List<string>();
                var values = new List<long>();
                foreach (var _ in p[0])
                {
                    outerMi.Add(string.Join(",", p[0].multi_index));
                    foreach (var __ in p[1])
                    {
                        innerMi.Add(string.Join(",", p[1].multi_index));
                        values.Add(V(p[1]));
                    }
                }

                CollectionAssert.AreEqual(new[] { "0", "1", "2" }, outerMi);
                // inner multi_index cycles (0,0),(0,1),(1,0),(1,1) for each of the 3 outer positions
                Assert.AreEqual("0,0|0,1|1,0|1,1|0,0|0,1|1,0|1,1|0,0|0,1|1,0|1,1", string.Join("|", innerMi));
                CollectionAssert.AreEqual(new long[] { 0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11 }, values.ToArray());
            }
            finally { foreach (var it in p) it.Dispose(); }
        }

        [TestMethod]
        public void ThreeLevels_MatchesNumPy()
        {
            var a = np.arange(12).reshape(2, 3, 2);
            var p = np.nested_iters(a, new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } }, flags: new[] { "multi_index" });
            try
            {
                var vals = new List<long>();
                foreach (var _ in p[0])
                    foreach (var __ in p[1])
                        foreach (var ___ in p[2])
                            vals.Add(V(p[2]));
                CollectionAssert.AreEqual(Enumerable.Range(0, 12).Select(x => (long)x).ToArray(), vals.ToArray());
            }
            finally { foreach (var it in p) it.Dispose(); }
        }

        [TestMethod]
        public void TwoOperands_ReversedAxes_MultiIndex_MatchesNumPy()
        {
            var a = np.arange(12).reshape(2, 3, 2);
            var b = np.arange(12, 24).reshape(2, 3, 2);
            var p = np.nested_iters(new[] { a, b }, new[] { new[] { 0, 2 }, new[] { 1 } }, flags: new[] { "multi_index" });
            try
            {
                var rows = new List<string>();
                foreach (var _ in p[0])
                    foreach (var __ in p[1])
                        rows.Add($"{string.Join(",", p[0].multi_index)};{string.Join(",", p[1].multi_index)};{V(p[1], 0)};{V(p[1], 1)}");

                var expected = new[]
                {
                    "0,0;0;0;12", "0,0;1;2;14", "0,0;2;4;16",
                    "0,1;0;1;13", "0,1;1;3;15", "0,1;2;5;17",
                    "1,0;0;6;18", "1,0;1;8;20", "1,0;2;10;22",
                    "1,1;0;7;19", "1,1;1;9;21", "1,1;2;11;23",
                };
                CollectionAssert.AreEqual(expected, rows);
            }
            finally { foreach (var it in p) it.Dispose(); }
        }

        [TestMethod]
        public void Errors_MatchNumPy()
        {
            var a = np.arange(12).reshape(2, 3, 2);

            var e1 = Assert.ThrowsException<ValueError>(() => np.nested_iters(a, new[] { new[] { 0 } }));
            StringAssert.Contains(e1.Message, "axes must have at least 2 entries for nested iteration");

            var e2 = Assert.ThrowsException<ValueError>(() => np.nested_iters(a, new[] { new[] { 0 }, new[] { 0, 1 } }));
            StringAssert.Contains(e2.Message, "An axis is used more than once");

            var e3 = Assert.ThrowsException<ValueError>(() => np.nested_iters(a, new[] { new[] { -1 }, new[] { 1 } }));
            StringAssert.Contains(e3.Message, "An axis is out of bounds");
        }

        [TestMethod]
        public void OutOfRangeAxis_Rejected()
        {
            // Axis past the operand's ndim: NumSharp rejects at iterator construction (IncorrectShapeException)
            // where NumPy raises ValueError("...is not a valid axis of op[0]..."). Both refuse; text/type differ.
            var a = np.arange(12).reshape(2, 3, 2);
            Assert.ThrowsException<IncorrectShapeException>(
                () => np.nested_iters(a, new[] { new[] { 0 }, new[] { 9 } }));
        }

        [TestMethod]
        [Misaligned]
        public void ValueStreamOrder_WithoutMultiIndex_DivergesFromNumPy()
        {
            // NumPy follows `order` for the value stream (C -> 0,1,6,7,...). NumSharp's NDIter reorders
            // op_axes iteration by memory (F-like) regardless of `order`, so without multi_index the pure
            // value order is NumPy's F-order. Pinned to make the divergence explicit; the multi_index
            // path above is the bit-exact, recommended surface.
            var a = np.arange(12).reshape(2, 3, 2);
            var p = np.nested_iters(a, new[] { new[] { 1 }, new[] { 0, 2 } });   // no multi_index
            try
            {
                var vals = new List<long>();
                foreach (var _ in p[0])
                    foreach (var __ in p[1])
                        vals.Add(V(p[1]));
                // NumSharp (current): F-order. NumPy order='K'/'C' on this C-array: 0,1,6,7,2,3,8,9,4,5,10,11.
                CollectionAssert.AreEqual(new long[] { 0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11 }, vals.ToArray());
            }
            finally { foreach (var it in p) it.Dispose(); }
        }
    }
}
