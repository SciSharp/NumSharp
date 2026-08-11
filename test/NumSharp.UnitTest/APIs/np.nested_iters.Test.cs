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
    /// multi_index now ALSO matches NumPy — the op_axes contiguity check judges the iterator's
    /// remapped layout instead of the full operand, so a non-contiguous axis subset no longer
    /// coalesces into F-order (verified across the 2- and 3-level axes partitions of a (2,3,2) array).
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
        public void ValueStreamOrder_WithoutMultiIndex_MatchesNumPy()
        {
            // The bare value stream (no multi_index) now matches NumPy's memory-order traversal for an
            // op_axes subset. The inner iterator spans axes [0,2] of a C-contiguous (2,3,2) — strides
            // (6,1), NON-contiguous — so it must NOT coalesce; previously the contiguity check read the
            // full operand (C-contiguous) and coalesced it into F-order (0,6,1,7,...). Now it judges the
            // iterator's op_axes layout, canCoalesce stays false, and the descending K-sort yields NumPy's
            // order. This is exactly the value+coord order the multi_index test above already proved.
            var a = np.arange(12).reshape(2, 3, 2);
            var p = np.nested_iters(a, new[] { new[] { 1 }, new[] { 0, 2 } });   // no multi_index
            try
            {
                var vals = new List<long>();
                foreach (var _ in p[0])
                    foreach (var __ in p[1])
                        vals.Add(V(p[1]));
                CollectionAssert.AreEqual(new long[] { 0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11 }, vals.ToArray());
            }
            finally { foreach (var it in p) it.Dispose(); }
        }

        [TestMethod]
        public void ValueStreamOrder_OuterOpAxesSubset_MatchesNumPy()
        {
            // The OUTER iterator carries the non-contiguous subset here (axes [0,2], strides (6,1)); the
            // same coalescing bug drove it F-order without multi_index. NumPy's value stream is memory
            // order: outer (i,k) with k fastest, inner j -> 0,2,4, 1,3,5, 6,8,10, 7,9,11.
            var a = np.arange(12).reshape(2, 3, 2);
            var p = np.nested_iters(a, new[] { new[] { 0, 2 }, new[] { 1 } });   // no multi_index
            try
            {
                var vals = new List<long>();
                foreach (var _ in p[0])
                    foreach (var __ in p[1])
                        vals.Add(V(p[1]));
                CollectionAssert.AreEqual(new long[] { 0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11 }, vals.ToArray());
            }
            finally { foreach (var it in p) it.Dispose(); }
        }
    }
}
