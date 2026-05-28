using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest suite for WRITEMASKED + ARRAYMASK support.
    /// NumPy: nditer_constr.c:1176-1230 (pairing validation),
    ///        nditer_constr.c:1328-1377 (check_mask_for_writemasked_reduction).
    ///
    /// Validation rules (verified against NumPy 2.4.2):
    /// - WRITEMASKED operand requires an ARRAYMASK operand.
    /// - ARRAYMASK operand requires at least one WRITEMASKED operand.
    /// - Only one operand may be ARRAYMASK.
    /// - An operand cannot be both WRITEMASKED and ARRAYMASK.
    /// - For a WRITEMASKED REDUCE operand: the mask must not vary while the operand is broadcast.
    /// </summary>
    [TestClass]
    public class NpyIterWriteMaskedTests
    {
        // ========= Validation: pairing rules =========

        [TestMethod]
        public void WriteMasked_WithArrayMask_Succeeds()
        {
            var arr = np.arange(5).astype(np.int32);
            var mask = np.array(new[] { true, false, true, false, true });
            var outArr = np.zeros(new int[] { 5 }, np.int32);

            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY,
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
            };

            using var it = NpyIterRef.MultiNew(
                3, new[] { arr, mask, outArr },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags);

            Assert.AreEqual(1, it.MaskOp);  // mask is operand index 1
            Assert.IsTrue(it.HasWriteMaskedOperand);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void WriteMasked_WithoutArrayMask_Throws()
        {
            var arr = np.arange(5).astype(np.int32);
            var outArr = np.zeros(new int[] { 5 }, np.int32);

            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY,
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
            };

            using var it = NpyIterRef.MultiNew(
                2, new[] { arr, outArr },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ArrayMask_WithoutWriteMasked_Throws()
        {
            var arr = np.arange(5).astype(np.int32);
            var mask = np.array(new[] { true, false, true, false, true });

            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY,
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
            };

            using var it = NpyIterRef.MultiNew(
                2, new[] { arr, mask },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TwoArrayMask_Throws()
        {
            var arr = np.arange(5).astype(np.int32);
            var mask1 = np.array(new[] { true, false, true, false, true });
            var mask2 = np.array(new[] { true, true, false, false, true });
            var outArr = np.zeros(new int[] { 5 }, np.int32);

            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY,
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
            };

            using var it = NpyIterRef.MultiNew(
                4, new[] { arr, mask1, mask2, outArr },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void WriteMaskedAndArrayMaskSameOperand_Throws()
        {
            var arr = np.arange(5).astype(np.int32);
            var outArr = np.zeros(new int[] { 5 }, np.int32);

            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY,
                NpyIterPerOpFlags.READWRITE |
                NpyIterPerOpFlags.WRITEMASKED |
                NpyIterPerOpFlags.ARRAYMASK,
            };

            using var it = NpyIterRef.MultiNew(
                2, new[] { arr, outArr },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags);
        }

        // ========= MaskOp tracking =========

        [TestMethod]
        public void MaskOp_MinusOne_WhenNoMask()
        {
            var arr = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(arr);
            Assert.AreEqual(-1, it.MaskOp);
            Assert.IsFalse(it.HasWriteMaskedOperand);
        }

        [TestMethod]
        public void MaskOp_CorrectlyTracksArrayMaskIndex()
        {
            var arr = np.arange(5).astype(np.int32);
            var mask = np.array(new[] { true, false, true, false, true });
            var outArr = np.zeros(new int[] { 5 }, np.int32);

            // Mask is at index 0, out at index 1, input at index 2
            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
                NpyIterPerOpFlags.READONLY,
            };

            using var it = NpyIterRef.MultiNew(
                3, new[] { mask, outArr, arr },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags);

            Assert.AreEqual(0, it.MaskOp);
        }

        // ========= Iteration works when WRITEMASKED set =========

        [TestMethod]
        public void WriteMasked_BasicIteration_AllElementsVisited()
        {
            var arr = np.arange(5).astype(np.int32);
            var mask = np.array(new[] { true, false, true, false, true });
            var outArr = np.zeros(new int[] { 5 }, np.int32);

            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY,
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
            };

            using var it = NpyIterRef.MultiNew(
                3, new[] { arr, mask, outArr },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags);

            // Iteration should visit all 5 elements (WRITEMASKED is just a marker;
            // actual masked writes are the responsibility of the higher-level code)
            int count = 0;
            do { count++; } while (it.Iternext());
            Assert.AreEqual(5, count);
        }

        // ========= check_mask_for_writemasked_reduction =========

        [TestMethod]
        public void MaskForWriteMaskedReduction_ValidPattern_Succeeds()
        {
            // WRITEMASKED reduction where mask has same shape as operand (no broadcast conflict).
            // Shape: (3, 4). Input: (3, 4). Output: (4,) with op_axes=[[-1, 0]]. Mask: (4,) with op_axes=[[-1, 0]].
            // Reduction axis is 0. Mask is aligned with output (same broadcast pattern).
            var x = np.arange(12).reshape(3, 4).astype(np.int32);
            var mask = np.array(new[] { true, false, true, false });
            var y = np.zeros(new int[] { 4 }, np.int32);

            var opAxes = new[]
            {
                new[] { 0, 1 },
                new[] { -1, 0 },  // mask: no axis 0 (broadcast), axis 0→1 (aligned with output)
                new[] { -1, 0 },  // output: same alignment
            };

            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY,
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
            };

            using var it = NpyIterRef.AdvancedNew(
                nop: 3,
                op: new[] { x, mask, y },
                flags: NpyIterGlobalFlags.REDUCE_OK | NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: opFlags,
                opDtypes: null,
                opAxesNDim: 2,
                opAxes: opAxes);

            Assert.IsTrue(it.IsReduction);
            Assert.AreEqual(1, it.MaskOp);
        }
    }
}
