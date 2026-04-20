using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest suite for NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE per-operand flag.
    /// NumPy: ndarraytypes.h:1170 (flag 0x40000000), nditer_constr.c:3130-3137 (short-circuit logic).
    ///
    /// Semantics: a hint used when COPY_IF_OVERLAP is set. If set on an operand AND
    /// both operands point to the same buffer with identical layout and no internal
    /// overlap, the overlap check can short-circuit (no copy needed) because the caller's
    /// inner loop accesses data element-by-element in iterator order.
    ///
    /// For NumSharp (which does not yet implement full COPY_IF_OVERLAP), this flag is
    /// accepted syntactically as a marker.
    /// </summary>
    [TestClass]
    public class NpyIterOverlapAssumeElementwiseTests
    {
        [TestMethod]
        public void OverlapAssumeElementwise_PerOpFlag_Value()
        {
            Assert.AreEqual(0x40000000u,
                (uint)NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP);
        }

        [TestMethod]
        public void OverlapAssumeElementwise_OnPerOpFlags_Accepted()
        {
            var arr = np.arange(5).astype(np.int32);
            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP
            };

            using var it = NpyIterRef.MultiNew(
                1, new[] { arr },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags);

            // Iteration should work normally
            int count = 0;
            do { count++; } while (it.Iternext());
            Assert.AreEqual(5, count);
        }

        [TestMethod]
        public void OverlapAssumeElementwise_MultiOp_AllAccepted()
        {
            var x = np.arange(4).astype(np.int32);
            var y = np.zeros(new int[] { 4 }, np.int32);

            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
                NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            };

            using var it = NpyIterRef.MultiNew(
                2, new[] { x, y },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags);

            do
            {
                int v = it.GetValue<int>(0);
                it.SetValue(v * 2, 1);
            } while (it.Iternext());

            CollectionAssert.AreEqual(new[] { 0, 2, 4, 6 }, y.ToArray<int>());
        }

        [TestMethod]
        public void OverlapAssumeElementwise_With_COPY_IF_OVERLAP_Global()
        {
            // When paired with the global COPY_IF_OVERLAP flag, this hint marks the
            // operand as safe for element-wise elision. We don't implement the elision
            // yet, but the combination should construct without error.
            var arr = np.arange(5).astype(np.int32);

            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP
            };

            using var it = NpyIterRef.MultiNew(
                1, new[] { arr },
                NpyIterGlobalFlags.COPY_IF_OVERLAP,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags);

            // Should iterate correctly
            int count = 0;
            do { count++; } while (it.Iternext());
            Assert.AreEqual(5, count);
        }

        [TestMethod]
        public void OverlapAssumeElementwise_PerOpFlag_IsHighBit()
        {
            // Verify bit position (top bit of the 16-bit per-op flag region)
            uint raw = (uint)NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
            Assert.AreEqual(30, (int)Math.Log2(raw));
        }
    }
}
