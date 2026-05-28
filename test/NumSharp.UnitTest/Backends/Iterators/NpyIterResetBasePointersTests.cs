using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest suite for NpyIter_ResetBasePointers (nditer_api.c:314).
    ///
    /// Semantics: Replaces the reset data pointers with baseptrs[iop] + baseoffsets[iop],
    /// then repositions to IterStart. Used in nested iteration (NumPy mapping.c, ufunc_object.c).
    ///
    /// Expected values verified against NumPy 2.4.2 on 2026-04-17.
    /// </summary>
    [TestClass]
    public class NpyIterResetBasePointersTests
    {
        // ================================================================
        // Basic: swap single operand's underlying array
        // ================================================================

        [TestMethod]
        public unsafe void ResetBasePointers_1D_Int32_SwapsData()
        {
            // Two arrays with same shape+dtype
            var a = np.arange(5).astype(np.int32);       // [0,1,2,3,4]
            var b = (np.arange(5) * 10).astype(np.int32); // [0,10,20,30,40]

            using var it = NpyIterRef.New(a);
            // Initial iteration reads from a
            var first = new List<int>();
            do { first.Add(it.GetValue<int>(0)); } while (it.Iternext());
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3, 4 }, first.ToArray());

            // Swap to b via ResetBasePointers
            byte* bBase = (byte*)b.Array.Address + b.Shape.offset * b.dtypesize;
            Span<IntPtr> ptrs = stackalloc IntPtr[] { (IntPtr)bBase };
            Assert.IsTrue(it.ResetBasePointers(ptrs));

            // Now iteration reads from b
            var second = new List<int>();
            do { second.Add(it.GetValue<int>(0)); } while (it.Iternext());
            CollectionAssert.AreEqual(new[] { 0, 10, 20, 30, 40 }, second.ToArray());
        }

        // ================================================================
        // Neg-stride: BaseOffsets must route new baseptr to flipped end
        //
        // NumPy: nditer_constr.c:2579-2605 accumulates baseoffsets during
        // flip, then ResetBasePointers uses resetdataptr = baseptrs + baseoffsets.
        // ================================================================

        [TestMethod]
        public unsafe void ResetBasePointers_1D_NegStride_PreservesMemoryOrder()
        {
            // a_rev is a reversed view — K-order flips negative stride
            var a = np.arange(5).astype(np.int32);        // memory: [0,1,2,3,4]
            var a_rev = a["::-1"];                         // logical: [4,3,2,1,0], stride = -4
            var b = (np.arange(5) * 10).astype(np.int32); // memory: [0,10,20,30,40]
            var b_rev = b["::-1"];                         // logical: [40,30,20,10,0]

            using var it = NpyIterRef.New(a_rev, order: NPY_ORDER.NPY_KEEPORDER);
            var first = new List<int>();
            do { first.Add(it.GetValue<int>(0)); } while (it.Iternext());
            // K-order flips negative stride: iterates in memory order [0,1,2,3,4]
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3, 4 }, first.ToArray());

            // Swap underlying to b_rev — baseptr points to logical start of b_rev
            // (which is memory end). BaseOffset should have been recorded during flip.
            byte* bRevBase = (byte*)b_rev.Array.Address + b_rev.Shape.offset * b_rev.dtypesize;
            Span<IntPtr> ptrs = stackalloc IntPtr[] { (IntPtr)bRevBase };
            Assert.IsTrue(it.ResetBasePointers(ptrs));

            var second = new List<int>();
            do { second.Add(it.GetValue<int>(0)); } while (it.Iternext());
            // Should iterate b in memory order: [0,10,20,30,40]
            CollectionAssert.AreEqual(new[] { 0, 10, 20, 30, 40 }, second.ToArray());
        }

        // ================================================================
        // Mid-iteration reset — must fully restart
        // ================================================================

        [TestMethod]
        public unsafe void ResetBasePointers_MidIteration_RestartsFromBeginning()
        {
            var a = np.arange(6).astype(np.int32);
            var b = (np.arange(6) + 100).astype(np.int32);

            using var it = NpyIterRef.New(a);
            // Advance 3 steps
            for (int i = 0; i < 3; i++) it.Iternext();

            // ResetBasePointers to b, iterate fully — should yield [100,101,102,103,104,105]
            byte* bBase = (byte*)b.Array.Address + b.Shape.offset * b.dtypesize;
            Span<IntPtr> ptrs = stackalloc IntPtr[] { (IntPtr)bBase };
            it.ResetBasePointers(ptrs);

            var vals = new List<int>();
            do { vals.Add(it.GetValue<int>(0)); } while (it.Iternext());
            CollectionAssert.AreEqual(new[] { 100, 101, 102, 103, 104, 105 }, vals.ToArray());
        }

        // ================================================================
        // Multi-operand
        // ================================================================

        [TestMethod]
        public unsafe void ResetBasePointers_MultiOperand_SwapsBoth()
        {
            var x1 = np.arange(4).astype(np.int32);
            var y1 = np.zeros(new int[] { 4 }, np.int32);
            var x2 = (np.arange(4) + 10).astype(np.int32);
            var y2 = np.zeros(new int[] { 4 }, np.int32);

            var opFlags = new[]
            {
                NpyIterPerOpFlags.READONLY,
                NpyIterPerOpFlags.WRITEONLY,
            };

            using var it = NpyIterRef.MultiNew(2, new[] { x1, y1 },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING, opFlags);
            // Write y1[i] = x1[i] * 2
            do
            {
                int v = it.GetValue<int>(0);
                it.SetValue(v * 2, 1);
            } while (it.Iternext());

            CollectionAssert.AreEqual(new[] { 0, 2, 4, 6 }, y1.ToArray<int>());

            // Swap both operands
            byte* x2Base = (byte*)x2.Array.Address + x2.Shape.offset * x2.dtypesize;
            byte* y2Base = (byte*)y2.Array.Address + y2.Shape.offset * y2.dtypesize;
            Span<IntPtr> ptrs = stackalloc IntPtr[] { (IntPtr)x2Base, (IntPtr)y2Base };
            it.ResetBasePointers(ptrs);

            do
            {
                int v = it.GetValue<int>(0);
                it.SetValue(v * 3, 1);
            } while (it.Iternext());

            // y2 should be 3 * x2 = [30, 33, 36, 39]
            CollectionAssert.AreEqual(new[] { 30, 33, 36, 39 }, y2.ToArray<int>());
            // y1 should be unchanged from first pass
            CollectionAssert.AreEqual(new[] { 0, 2, 4, 6 }, y1.ToArray<int>());
        }

        // ================================================================
        // 2D
        // ================================================================

        [TestMethod]
        public unsafe void ResetBasePointers_2D_RowMajor_SwapsData()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);     // [[0,1,2],[3,4,5]]
            var b = (np.arange(6) + 100).reshape(2, 3).astype(np.int32); // [[100,101,102],[103,104,105]]

            using var it = NpyIterRef.New(a);
            var first = new List<int>();
            do { first.Add(it.GetValue<int>(0)); } while (it.Iternext());
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3, 4, 5 }, first.ToArray());

            byte* bBase = (byte*)b.Array.Address + b.Shape.offset * b.dtypesize;
            Span<IntPtr> ptrs = stackalloc IntPtr[] { (IntPtr)bBase };
            it.ResetBasePointers(ptrs);

            var second = new List<int>();
            do { second.Add(it.GetValue<int>(0)); } while (it.Iternext());
            CollectionAssert.AreEqual(new[] { 100, 101, 102, 103, 104, 105 }, second.ToArray());
        }

        // ================================================================
        // 2D negative stride — both axes flipped
        // NumPy: c = np.arange(6).reshape(2,3)[::-1,::-1]
        //        nditer iterates in memory order [0,1,2,3,4,5]
        // ================================================================

        [TestMethod]
        public unsafe void ResetBasePointers_2D_BothAxesReversed_PreservesMemoryOrder()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            var a_rev = a["::-1, ::-1"];  // logical [[5,4,3],[2,1,0]]
            var b = (np.arange(6) * 10).reshape(2, 3).astype(np.int32);
            var b_rev = b["::-1, ::-1"];

            using var it = NpyIterRef.New(a_rev, order: NPY_ORDER.NPY_KEEPORDER);
            var first = new List<int>();
            do { first.Add(it.GetValue<int>(0)); } while (it.Iternext());
            // NumPy output: memory-order iteration
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3, 4, 5 }, first.ToArray());

            byte* bRevBase = (byte*)b_rev.Array.Address + b_rev.Shape.offset * b_rev.dtypesize;
            Span<IntPtr> ptrs = stackalloc IntPtr[] { (IntPtr)bRevBase };
            it.ResetBasePointers(ptrs);

            var second = new List<int>();
            do { second.Add(it.GetValue<int>(0)); } while (it.Iternext());
            CollectionAssert.AreEqual(new[] { 0, 10, 20, 30, 40, 50 }, second.ToArray());
        }

        // ================================================================
        // Error path: length mismatch
        // ================================================================

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ResetBasePointers_WrongLength_Throws()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(a);
            Span<IntPtr> ptrs = stackalloc IntPtr[] { IntPtr.Zero, IntPtr.Zero };
            it.ResetBasePointers(ptrs);
        }

        // ================================================================
        // NDArray convenience overload
        // ================================================================

        [TestMethod]
        public unsafe void ResetBasePointers_NDArrayOverload_Works()
        {
            var a = np.arange(5).astype(np.int32);
            var b = (np.arange(5) + 50).astype(np.int32);

            using var it = NpyIterRef.New(a);
            // Consume one element so we know Reset works
            it.Iternext();

            Assert.IsTrue(it.ResetBasePointers(new[] { b }));

            var vals = new List<int>();
            do { vals.Add(it.GetValue<int>(0)); } while (it.Iternext());
            CollectionAssert.AreEqual(new[] { 50, 51, 52, 53, 54 }, vals.ToArray());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ResetBasePointers_NDArrayOverload_NullArray_Throws()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(a);
            it.ResetBasePointers((NDArray[])null);
        }

        // ================================================================
        // Repeated resets (nested iteration pattern)
        // ================================================================

        [TestMethod]
        public unsafe void ResetBasePointers_RepeatedResets_Work()
        {
            var arrays = new[]
            {
                np.arange(4).astype(np.int32),
                (np.arange(4) + 100).astype(np.int32),
                (np.arange(4) * 7).astype(np.int32),
            };
            var expected = new[]
            {
                new[] { 0, 1, 2, 3 },
                new[] { 100, 101, 102, 103 },
                new[] { 0, 7, 14, 21 },
            };

            using var it = NpyIterRef.New(arrays[0]);

            Span<IntPtr> ptrs = stackalloc IntPtr[1];
            for (int r = 0; r < arrays.Length; r++)
            {
                if (r > 0)
                {
                    byte* basePtr = (byte*)arrays[r].Array.Address + arrays[r].Shape.offset * arrays[r].dtypesize;
                    ptrs[0] = (IntPtr)basePtr;
                    it.ResetBasePointers(ptrs);
                }

                var vals = new List<int>();
                do { vals.Add(it.GetValue<int>(0)); } while (it.Iternext());
                CollectionAssert.AreEqual(expected[r], vals.ToArray(), $"Pass {r}");
            }
        }
    }
}
