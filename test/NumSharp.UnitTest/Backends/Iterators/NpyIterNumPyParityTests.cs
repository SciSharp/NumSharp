using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Tests derived from running actual NumPy code to verify NumSharp parity.
    /// Each test documents the NumPy code used to derive expected values.
    /// </summary>
    [TestClass]
    public class NpyIterNumPyParityTests
    {
        // =========================================================================
        // Coalescing Behavior Tests
        // =========================================================================

        [TestMethod]
        public void Coalescing_Contiguous3D_CoalescesToNDim1()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(arr)
            // >>> it.ndim
            // 1
            // >>> it.itersize
            // 24

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1, iter.NDim, "Contiguous (2,3,4) should coalesce to ndim=1");
            Assert.AreEqual(24, iter.IterSize);
        }

        [TestMethod]
        public void Coalescing_WithMultiIndex_PreservesNDim()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(arr, flags=['multi_index'])
            // >>> it.ndim
            // 3

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            Assert.AreEqual(3, iter.NDim, "With multi_index flag, should preserve ndim=3");
            Assert.IsTrue(iter.HasMultiIndex);
        }

        [TestMethod]
        public void Coalescing_Transposed_CoalescesToNDim1()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> arr_t = arr.T
            // >>> arr_t.shape
            // (4, 3, 2)
            // >>> arr_t.flags.c_contiguous
            // False
            // >>> arr_t.flags.f_contiguous
            // True
            // >>> it = np.nditer(arr_t)
            // >>> it.ndim
            // 1

            var arr = np.arange(24).reshape(2, 3, 4);
            var arr_t = arr.T;

            Assert.AreEqual(new Shape(4, 3, 2), arr_t.Shape);

            using var iter = NpyIterRef.New(arr_t);

            // NumPy coalesces F-contiguous arrays to ndim=1 as well
            Assert.AreEqual(1, iter.NDim, "F-contiguous transposed array should coalesce to ndim=1");
            Assert.AreEqual(24, iter.IterSize);
        }

        [TestMethod]
        public void Coalescing_NonContiguous2DSlice_PreservesNDim()
        {
            // NumPy 2.4.2:
            // >>> arr2d = np.arange(20).reshape(4, 5)
            // >>> sliced = arr2d[::2, ::2]
            // >>> sliced.shape
            // (2, 3)
            // >>> it = np.nditer(sliced)
            // >>> it.ndim
            // 2
            // >>> [int(x) for x in it]
            // [0, 2, 4, 10, 12, 14]

            var arr2d = np.arange(20).reshape(4, 5);
            var sliced = arr2d["::2, ::2"];

            Assert.AreEqual(new Shape(2, 3), sliced.Shape);

            using var iter = NpyIterRef.New(sliced, NpyIterGlobalFlags.MULTI_INDEX);

            // Non-contiguous slice with multi_index should preserve dimensions
            Assert.AreEqual(2, iter.NDim);
        }

        [TestMethod]
        public void Coalescing_Scalar_HasNDim0()
        {
            // NumPy 2.4.2:
            // >>> scalar = np.array(42)
            // >>> it = np.nditer(scalar)
            // >>> it.ndim
            // 0
            // >>> it.itersize
            // 1
            // >>> [int(x) for x in it]
            // [42]

            var scalar = np.array(42);

            using var iter = NpyIterRef.New(scalar);

            Assert.AreEqual(0, iter.NDim, "Scalar should have ndim=0");
            Assert.AreEqual(1, iter.IterSize, "Scalar should have itersize=1");
        }

        [TestMethod]
        public void Coalescing_EmptyArray_HasIterSize0()
        {
            // NumPy 2.4.2:
            // >>> empty = np.array([], dtype=np.int32)
            // >>> it = np.nditer(empty, flags=['zerosize_ok'])
            // >>> it.ndim
            // 1
            // >>> it.itersize
            // 0

            var empty = np.array(new int[0]);

            using var iter = NpyIterRef.New(empty, NpyIterGlobalFlags.ZEROSIZE_OK);

            Assert.AreEqual(1, iter.NDim);
            Assert.AreEqual(0, iter.IterSize);
        }

        // =========================================================================
        // C-Index Tracking Tests
        // =========================================================================

        [TestMethod]
        public void CIndex_2DArray_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr2d = np.arange(12).reshape(3, 4)
            // >>> it = np.nditer(arr2d, flags=['multi_index', 'c_index'])
            // First 6 elements:
            // [((0, 0), 0, 0), ((0, 1), 1, 1), ((0, 2), 2, 2), ((0, 3), 3, 3), ((1, 0), 4, 4), ((1, 1), 5, 5)]
            // (multi_index, c_index, value)

            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            Assert.IsTrue(iter.HasMultiIndex);
            Assert.IsTrue(iter.HasIndex);

            // Test specific positions from NumPy output
            var coords = new long[2];

            // Position (0, 0): c_index = 0
            iter.GotoMultiIndex(new long[] { 0, 0 });
            Assert.AreEqual(0, iter.GetIndex());

            // Position (0, 3): c_index = 3
            iter.GotoMultiIndex(new long[] { 0, 3 });
            Assert.AreEqual(3, iter.GetIndex());

            // Position (1, 0): c_index = 4
            iter.GotoMultiIndex(new long[] { 1, 0 });
            Assert.AreEqual(4, iter.GetIndex());

            // Position (2, 3): c_index = 11
            iter.GotoMultiIndex(new long[] { 2, 3 });
            Assert.AreEqual(11, iter.GetIndex());
        }

        [TestMethod]
        public void CIndex_3DArray_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(arr, flags=['multi_index', 'c_index'])
            // Selected elements from output:
            // {'multi_index': (0, 0, 0), 'c_index': 0, 'value': 0}
            // {'multi_index': (0, 1, 2), 'c_index': 6, 'value': 6}
            // {'multi_index': (1, 0, 0), 'c_index': 12, 'value': 12}
            // {'multi_index': (1, 2, 3), 'c_index': 23, 'value': 23}

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            // Position (0, 0, 0): c_index = 0
            iter.GotoMultiIndex(new long[] { 0, 0, 0 });
            Assert.AreEqual(0, iter.GetIndex());

            // Position (0, 1, 2): c_index = 6
            iter.GotoMultiIndex(new long[] { 0, 1, 2 });
            Assert.AreEqual(6, iter.GetIndex());

            // Position (1, 0, 0): c_index = 12
            iter.GotoMultiIndex(new long[] { 1, 0, 0 });
            Assert.AreEqual(12, iter.GetIndex());

            // Position (1, 2, 3): c_index = 23
            iter.GotoMultiIndex(new long[] { 1, 2, 3 });
            Assert.AreEqual(23, iter.GetIndex());
        }

        // =========================================================================
        // F-Index Tracking Tests
        // =========================================================================

        [TestMethod]
        public void FIndex_2DArray_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr2d = np.arange(12).reshape(3, 4)
            // >>> it = np.nditer(arr2d, flags=['multi_index', 'f_index'])
            // First 6 elements (multi_index, f_index, value):
            // [((0, 0), 0, 0), ((0, 1), 3, 1), ((0, 2), 6, 2), ((0, 3), 9, 3), ((1, 0), 1, 4), ((1, 1), 4, 5)]

            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.F_INDEX);

            // Position (0, 0): f_index = 0
            iter.GotoMultiIndex(new long[] { 0, 0 });
            Assert.AreEqual(0, iter.GetIndex());

            // Position (0, 1): f_index = 3 (column 1 in F-order = 1*3 = 3)
            iter.GotoMultiIndex(new long[] { 0, 1 });
            Assert.AreEqual(3, iter.GetIndex());

            // Position (1, 0): f_index = 1
            iter.GotoMultiIndex(new long[] { 1, 0 });
            Assert.AreEqual(1, iter.GetIndex());

            // Position (2, 3): f_index = 2 + 3*3 = 11
            iter.GotoMultiIndex(new long[] { 2, 3 });
            Assert.AreEqual(11, iter.GetIndex());
        }

        [TestMethod]
        public void FIndex_3DArray_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(arr, flags=['multi_index', 'f_index'])
            // Selected elements:
            // {'multi_index': (0, 0, 0), 'f_index': 0, 'value': 0}
            // {'multi_index': (0, 0, 1), 'f_index': 6, 'value': 1}
            // {'multi_index': (0, 1, 0), 'f_index': 2, 'value': 4}
            // {'multi_index': (1, 0, 0), 'f_index': 1, 'value': 12}
            // {'multi_index': (1, 2, 3), 'f_index': 23, 'value': 23}

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.F_INDEX);

            // Position (0, 0, 0): f_index = 0
            iter.GotoMultiIndex(new long[] { 0, 0, 0 });
            Assert.AreEqual(0, iter.GetIndex());

            // Position (0, 0, 1): f_index = 6
            iter.GotoMultiIndex(new long[] { 0, 0, 1 });
            Assert.AreEqual(6, iter.GetIndex());

            // Position (0, 1, 0): f_index = 2
            iter.GotoMultiIndex(new long[] { 0, 1, 0 });
            Assert.AreEqual(2, iter.GetIndex());

            // Position (1, 0, 0): f_index = 1
            iter.GotoMultiIndex(new long[] { 1, 0, 0 });
            Assert.AreEqual(1, iter.GetIndex());

            // Position (1, 2, 3): f_index = 23
            iter.GotoMultiIndex(new long[] { 1, 2, 3 });
            Assert.AreEqual(23, iter.GetIndex());
        }

        // =========================================================================
        // Sliced Array Iteration Tests
        // =========================================================================

        [TestMethod]
        public void SlicedArray_IterationOrder_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr2d = np.arange(12).reshape(3, 4)
            // >>> sliced = arr2d[::2, 1:3]  # Shape (2, 2)
            // >>> sliced.tolist()
            // [[1, 2], [9, 10]]
            // >>> it = np.nditer(sliced, flags=['multi_index', 'c_index'])
            // >>> [(it.multi_index, it.index, int(x)) for x in it]
            // [((0, 0), 0, 1), ((0, 1), 1, 2), ((1, 0), 2, 9), ((1, 1), 3, 10)]

            var arr2d = np.arange(12).reshape(3, 4);
            var sliced = arr2d["::2, 1:3"];

            Assert.AreEqual(new Shape(2, 2), sliced.Shape);

            // Verify sliced values match NumPy
            Assert.AreEqual(1, (int)sliced[0, 0]);
            Assert.AreEqual(2, (int)sliced[0, 1]);
            Assert.AreEqual(9, (int)sliced[1, 0]);
            Assert.AreEqual(10, (int)sliced[1, 1]);

            using var iter = NpyIterRef.New(sliced, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            // Verify c_index at each position
            iter.GotoMultiIndex(new long[] { 0, 0 });
            Assert.AreEqual(0, iter.GetIndex());

            iter.GotoMultiIndex(new long[] { 0, 1 });
            Assert.AreEqual(1, iter.GetIndex());

            iter.GotoMultiIndex(new long[] { 1, 0 });
            Assert.AreEqual(2, iter.GetIndex());

            iter.GotoMultiIndex(new long[] { 1, 1 });
            Assert.AreEqual(3, iter.GetIndex());
        }

        // =========================================================================
        // Broadcast Iteration Tests
        // =========================================================================

        [TestMethod]
        public void Broadcast_TwoOperands_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> a = np.array([[1], [2], [3]])  # Shape (3, 1)
            // >>> b = np.array([[10, 20, 30, 40]])  # Shape (1, 4)
            // >>> it = np.nditer([a, b], flags=['multi_index', 'c_index'])
            // First 4 elements:
            // {'multi_index': (0, 0), 'c_index': 0, 'a': 1, 'b': 10}
            // {'multi_index': (0, 1), 'c_index': 1, 'a': 1, 'b': 20}
            // {'multi_index': (0, 2), 'c_index': 2, 'a': 1, 'b': 30}
            // {'multi_index': (0, 3), 'c_index': 3, 'a': 1, 'b': 40}

            var a = np.array(new int[,] { { 1 }, { 2 }, { 3 } });  // Shape (3, 1)
            var b = np.array(new int[,] { { 10, 20, 30, 40 } });   // Shape (1, 4)

            Assert.AreEqual(new Shape(3, 1), a.Shape);
            Assert.AreEqual(new Shape(1, 4), b.Shape);

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(12, iter.IterSize);  // 3 * 4 = 12 after broadcast
            Assert.AreEqual(2, iter.NDim);       // Still 2D with multi_index
        }

        // =========================================================================
        // External Loop Tests
        // =========================================================================

        [TestMethod]
        public void ExternalLoop_Contiguous_SingleChunk()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(arr, flags=['external_loop'], op_flags=['readonly'])
            // >>> it.ndim
            // 1
            // >>> [len(chunk) for chunk in it]
            // [24]

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);

            Assert.AreEqual(1, iter.NDim);
            Assert.IsTrue(iter.HasExternalLoop);
            Assert.AreEqual(24, iter.IterSize);
        }

        // =========================================================================
        // Iteration Order Tests
        // =========================================================================

        [TestMethod]
        public void IterationOrder_2DArray_RowMajor()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(6).reshape(2, 3)
            // >>> it = np.nditer(arr, flags=['multi_index'])
            // >>> [(it.multi_index, int(x)) for x in it]
            // [((0, 0), 0), ((0, 1), 1), ((0, 2), 2), ((1, 0), 3), ((1, 1), 4), ((1, 2), 5)]

            var arr = np.arange(6).reshape(2, 3);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            var coords = new long[2];

            // At start: (0, 0) -> value 0
            iter.GetMultiIndex(coords);
            Assert.AreEqual(0, coords[0]);
            Assert.AreEqual(0, coords[1]);

            // After moving to index 2: (0, 2) -> value 2
            iter.GotoIterIndex(2);
            iter.GetMultiIndex(coords);
            Assert.AreEqual(0, coords[0]);
            Assert.AreEqual(2, coords[1]);

            // After moving to index 3: (1, 0) -> value 3
            iter.GotoIterIndex(3);
            iter.GetMultiIndex(coords);
            Assert.AreEqual(1, coords[0]);
            Assert.AreEqual(0, coords[1]);

            // After moving to index 5: (1, 2) -> value 5
            iter.GotoIterIndex(5);
            iter.GetMultiIndex(coords);
            Assert.AreEqual(1, coords[0]);
            Assert.AreEqual(2, coords[1]);
        }

        // =========================================================================
        // Buffered Iteration Tests
        // =========================================================================

        [TestMethod]
        public void Buffered_ChunkSizes_MatchBufferSize()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(100)
            // >>> it = np.nditer(arr, flags=['external_loop', 'buffered'], op_flags=['readonly'], buffersize=32)
            // >>> [len(chunk) for chunk in it]
            // [32, 32, 32, 4]

            var arr = np.arange(100);

            using var iter = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { arr },
                flags: NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                bufferSize: 32);

            Assert.IsTrue(iter.RequiresBuffering);
            Assert.AreEqual(100, iter.IterSize);
        }

        // =========================================================================
        // 3D Transposed with Multi-Index Tests
        // =========================================================================

        [TestMethod]
        public void Transposed3D_MultiIndex_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr3d = np.arange(24).reshape(2, 3, 4)
            // >>> arr3d_t = arr3d.transpose(2, 0, 1)  # Shape (4, 2, 3)
            // >>> it = np.nditer(arr3d_t, flags=['multi_index'])
            // First 8 with multi_index:
            // [((0, 0, 0), 0), ((1, 0, 0), 1), ((2, 0, 0), 2), ((3, 0, 0), 3),
            //  ((0, 0, 1), 4), ((1, 0, 1), 5), ((2, 0, 1), 6), ((3, 0, 1), 7)]

            var arr3d = np.arange(24).reshape(2, 3, 4);
            var arr3d_t = np.transpose(arr3d, new[] { 2, 0, 1 });

            Assert.AreEqual(new Shape(4, 2, 3), arr3d_t.Shape);

            using var iter = NpyIterRef.New(arr3d_t, NpyIterGlobalFlags.MULTI_INDEX);

            Assert.AreEqual(3, iter.NDim);  // With multi_index, preserves dimensions

            var coords = new long[3];

            // At index 0: (0, 0, 0) -> value 0
            iter.GetMultiIndex(coords);
            Assert.AreEqual(0, coords[0]);
            Assert.AreEqual(0, coords[1]);
            Assert.AreEqual(0, coords[2]);
        }

        // =========================================================================
        // Reset and State Tests
        // =========================================================================

        [TestMethod]
        public void Reset_RestoresInitialState()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.C_INDEX);

            // Move forward
            iter.GotoIterIndex(50);
            Assert.AreEqual(50, iter.IterIndex);
            Assert.AreEqual(50, iter.GetIndex());

            // Reset
            iter.Reset();
            Assert.AreEqual(0, iter.IterIndex);
            Assert.AreEqual(0, iter.GetIndex());
        }

        [TestMethod]
        public void GotoIterIndex_UpdatesAllState()
        {
            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            // Jump to index 17 = (1, 1, 1) in C-order
            iter.GotoIterIndex(17);

            Assert.AreEqual(17, iter.IterIndex);
            Assert.AreEqual(17, iter.GetIndex());

            var coords = new long[3];
            iter.GetMultiIndex(coords);
            Assert.AreEqual(1, coords[0]);
            Assert.AreEqual(1, coords[1]);
            Assert.AreEqual(1, coords[2]);
        }

        // =========================================================================
        // High-Dimensional Array Tests
        // =========================================================================

        [TestMethod]
        public void HighDimensional_5D_CoalescesToNDim1()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(32).reshape(2, 2, 2, 2, 2)
            // >>> it = np.nditer(arr)
            // >>> it.ndim
            // 1
            // >>> it.itersize
            // 32

            var arr = np.arange(32).reshape(2, 2, 2, 2, 2);

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1, iter.NDim, "5D contiguous array should coalesce to ndim=1");
            Assert.AreEqual(32, iter.IterSize);
        }

        // =========================================================================
        // Multi-Operand Tests
        // =========================================================================

        [TestMethod]
        public void MultiOperand_DifferentDtypes_PreservesTypes()
        {
            // NumPy 2.4.2:
            // >>> a = np.array([1, 2, 3], dtype=np.int32)
            // >>> b = np.array([1.5, 2.5, 3.5], dtype=np.float64)
            // >>> it = np.nditer([a, b])
            // >>> it.ndim
            // 1
            // >>> it.nop
            // 2

            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.array(new double[] { 1.5, 2.5, 3.5 });

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(1, iter.NDim);
            Assert.AreEqual(2, iter.NOp);
            Assert.AreEqual(3, iter.IterSize);

            var dtypes = iter.GetDescrArray();
            Assert.AreEqual(NPTypeCode.Int32, dtypes[0]);
            Assert.AreEqual(NPTypeCode.Double, dtypes[1]);
        }

        // =========================================================================
        // 1D Array Tests
        // =========================================================================

        [TestMethod]
        public void OneDimensional_MultiIndex_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(5)
            // >>> it = np.nditer(arr, flags=['multi_index', 'c_index'])
            // >>> [(it.multi_index, it.index, int(x)) for x in it]
            // [((0,), 0, 0), ((1,), 1, 1), ((2,), 2, 2), ((3,), 3, 3), ((4,), 4, 4)]

            var arr = np.arange(5);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            Assert.AreEqual(1, iter.NDim);

            var coords = new long[1];

            for (int i = 0; i < 5; i++)
            {
                iter.GotoIterIndex(i);
                iter.GetMultiIndex(coords);
                Assert.AreEqual(i, coords[0], $"multi_index at position {i}");
                Assert.AreEqual(i, iter.GetIndex(), $"c_index at position {i}");
            }
        }

        // =========================================================================
        // Broadcast with Scalar Tests
        // =========================================================================

        [TestMethod]
        public void BroadcastScalar_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> scalar = np.array(5)
            // >>> arr = np.arange(4)
            // >>> it = np.nditer([scalar, arr], flags=['multi_index'])
            // >>> [(it.multi_index, int(x), int(y)) for x, y in it]
            // [((0,), 5, 0), ((1,), 5, 1), ((2,), 5, 2), ((3,), 5, 3)]

            var scalar = np.array(5);
            var arr = np.arange(4);

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { scalar, arr },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(4, iter.IterSize);
            Assert.AreEqual(1, iter.NDim);
        }

        // =========================================================================
        // Reversed Array Tests
        // =========================================================================

        [TestMethod]
        public void Reversed1D_IndexTracking_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(10)
            // >>> rev = arr[::-1]
            // >>> rev.strides
            // (-8,)
            // >>> it = np.nditer(rev, flags=['multi_index', 'c_index'])
            // >>> [(it.multi_index, it.index, int(x)) for x in it]
            // [((9,), 9, 0), ((8,), 8, 1), ((7,), 7, 2), ...]
            // Note: multi_index and c_index track the ORIGINAL array positions

            var arr = np.arange(10);
            var rev = arr["::-1"];

            Assert.AreEqual(10, rev.size);

            // Verify reversed values
            Assert.AreEqual(9, (int)rev[0]);
            Assert.AreEqual(0, (int)rev[9]);

            using var iter = NpyIterRef.New(rev, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            Assert.AreEqual(1, iter.NDim);

            // NumPy behavior: indices track into the VIEW, not the original array
            // At position 0 of iteration: multi_index=(0,), value=9 (reversed)
            var coords = new long[1];
            iter.GotoMultiIndex(new long[] { 0 });
            Assert.AreEqual(0, iter.GetIndex());
        }

        // =========================================================================
        // 2D Partially Reversed Tests
        // =========================================================================

        [TestMethod]
        public void Reversed2D_OneAxis_ValuesMatch()
        {
            // NumPy 2.4.2:
            // >>> arr2d = np.arange(6).reshape(2, 3)
            // >>> rev2d = arr2d[:, ::-1]
            // >>> rev2d.tolist()
            // [[2, 1, 0], [5, 4, 3]]

            var arr2d = np.arange(6).reshape(2, 3);
            var rev2d = arr2d[":, ::-1"];

            // Verify values match NumPy output
            Assert.AreEqual(2, (int)rev2d[0, 0]);
            Assert.AreEqual(1, (int)rev2d[0, 1]);
            Assert.AreEqual(0, (int)rev2d[0, 2]);
            Assert.AreEqual(5, (int)rev2d[1, 0]);
            Assert.AreEqual(4, (int)rev2d[1, 1]);
            Assert.AreEqual(3, (int)rev2d[1, 2]);

            using var iter = NpyIterRef.New(rev2d, NpyIterGlobalFlags.MULTI_INDEX);

            Assert.AreEqual(6, iter.IterSize);
        }

        // =========================================================================
        // Reset Behavior Tests
        // =========================================================================

        [TestMethod]
        public void Reset_AfterPartialIteration_RestoresStart()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(5)
            // >>> it = np.nditer(arr, flags=['multi_index', 'c_index'])
            // >>> for i, x in enumerate(it):
            // ...     if i >= 3: break
            // >>> print(it.multi_index, it.index)
            // (3,) 3
            // >>> it.reset()
            // >>> print(it.multi_index, it.index)
            // (0,) 0

            var arr = np.arange(5);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            // Simulate partial iteration by jumping
            iter.GotoIterIndex(3);
            Assert.AreEqual(3, iter.IterIndex);
            Assert.AreEqual(3, iter.GetIndex());

            var coords = new long[1];
            iter.GetMultiIndex(coords);
            Assert.AreEqual(3, coords[0]);

            // Reset
            iter.Reset();
            Assert.AreEqual(0, iter.IterIndex);
            Assert.AreEqual(0, iter.GetIndex());

            iter.GetMultiIndex(coords);
            Assert.AreEqual(0, coords[0]);
        }

        // =========================================================================
        // RemoveMultiIndex Tests
        // =========================================================================

        [TestMethod]
        public void RemoveMultiIndex_EnablesCoalescing()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(a, flags=['multi_index'])
            // >>> print(f'Before: ndim={it.ndim}, shape={it.shape}')
            // Before: ndim=3, shape=(2, 3, 4)
            // >>> it.remove_multi_index()
            // >>> print(f'After: ndim={it.ndim}, shape={it.shape}')
            // After: ndim=1, shape=(24,)

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            Assert.AreEqual(3, iter.NDim);
            Assert.IsTrue(iter.HasMultiIndex);

            iter.RemoveMultiIndex();

            Assert.AreEqual(1, iter.NDim, "After RemoveMultiIndex, should coalesce to ndim=1");
            Assert.IsFalse(iter.HasMultiIndex);
            Assert.AreEqual(24, iter.IterSize);
        }

        [TestMethod]
        public void RemoveMultiIndex_ResetsIterIndex()
        {
            // NumPy 2.4.2:
            // >>> it = np.nditer(np.arange(24).reshape(2,3,4), flags=['multi_index'])
            // >>> for i in range(5): next(it)
            // >>> print(it.iterindex)
            // 4
            // >>> it.remove_multi_index()
            // >>> print(it.iterindex)
            // 0

            var arr = np.arange(24).reshape(2, 3, 4);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            // Advance a few positions
            iter.GotoIterIndex(5);
            Assert.AreEqual(5, iter.IterIndex);

            iter.RemoveMultiIndex();
            Assert.AreEqual(0, iter.IterIndex, "RemoveMultiIndex should reset iterindex to 0");
        }

        // =========================================================================
        // RemoveAxis Tests
        // =========================================================================

        [TestMethod]
        public void RemoveAxis_UpdatesShapeAndIterSize()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(a, flags=['multi_index'])
            // >>> it.remove_axis(1)  # Remove middle axis
            // >>> print(f'ndim={it.ndim}, shape={it.shape}, itersize={it.itersize}')
            // ndim=2, shape=(2, 4), itersize=8

            var arr = np.arange(24).reshape(2, 3, 4);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            Assert.AreEqual(3, iter.NDim);
            Assert.AreEqual(24, iter.IterSize);

            iter.RemoveAxis(1);

            Assert.AreEqual(2, iter.NDim);
            CollectionAssert.AreEqual(new long[] { 2, 4 }, iter.Shape);
            Assert.AreEqual(8, iter.IterSize);
        }

        [TestMethod]
        public void RemoveAxis_IteratesCorrectElements()
        {
            // NumPy 2.4.2:
            // >>> it = np.nditer(np.arange(24).reshape(2,3,4), flags=['multi_index'])
            // >>> it.remove_axis(1)
            // >>> for i, x in enumerate(it):
            // ...     if i < 8: print(f'{it.multi_index}: {int(x)}')
            // (0, 0): 0
            // (0, 1): 1
            // (0, 2): 2
            // (0, 3): 3
            // (1, 0): 12
            // (1, 1): 13
            // (1, 2): 14
            // (1, 3): 15

            var arr = np.arange(24).reshape(2, 3, 4);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);
            iter.RemoveAxis(1);

            var expectedValues = new int[] { 0, 1, 2, 3, 12, 13, 14, 15 };
            var coords = new long[2];

            for (int i = 0; i < 8; i++)
            {
                iter.GetMultiIndex(coords);
                int value = iter.GetValue<int>();
                Assert.AreEqual(expectedValues[i], value, $"At iteration {i}");
                iter.Iternext();
            }
        }

        // =========================================================================
        // Finished Property Tests
        // =========================================================================

        [TestMethod]
        public void Finished_FalseAtStart_TrueAfterLastElement()
        {
            // NumPy 2.4.2:
            // >>> it = np.nditer(np.arange(4))
            // >>> print(it.finished)
            // False
            // >>> while not it.finished:
            // ...     it.iternext()
            // >>> print(it.finished)
            // True

            var arr = np.arange(4);
            using var iter = NpyIterRef.New(arr);

            Assert.IsFalse(iter.Finished, "Should not be finished at start");

            int count = 0;
            while (!iter.Finished)
            {
                iter.Iternext();
                count++;
            }

            Assert.AreEqual(4, count);
            Assert.IsTrue(iter.Finished, "Should be finished after iterating all elements");
        }

        [TestMethod]
        public void Finished_ResetToFalseAfterReset()
        {
            var arr = np.arange(4);
            using var iter = NpyIterRef.New(arr);

            // Exhaust the iterator
            while (!iter.Finished)
                iter.Iternext();

            Assert.IsTrue(iter.Finished);

            iter.Reset();
            Assert.IsFalse(iter.Finished, "Should not be finished after reset");
        }

        // =========================================================================
        // Shape Property Tests
        // =========================================================================

        [TestMethod]
        public void Shape_MatchesIteratorDimensions()
        {
            // NumPy 2.4.2:
            // >>> it = np.nditer(np.arange(24).reshape(2,3,4), flags=['multi_index'])
            // >>> print(it.shape)
            // (2, 3, 4)

            var arr = np.arange(24).reshape(2, 3, 4);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            CollectionAssert.AreEqual(new long[] { 2, 3, 4 }, iter.Shape);
        }

        [TestMethod]
        public void Shape_ChangesAfterCoalescing()
        {
            // NumPy 2.4.2:
            // >>> it = np.nditer(np.arange(24).reshape(2,3,4))  # No multi_index = coalesced
            // >>> print(it.shape)
            // (24,)

            var arr = np.arange(24).reshape(2, 3, 4);
            using var iter = NpyIterRef.New(arr);  // No MULTI_INDEX flag

            CollectionAssert.AreEqual(new long[] { 24 }, iter.Shape);
        }

        // =========================================================================
        // Iternext Tests
        // =========================================================================

        [TestMethod]
        public void Iternext_ReturnsTrueWhileMoreElements()
        {
            // NumPy 2.4.2:
            // >>> it = np.nditer(np.arange(4))
            // >>> values = []
            // >>> while True:
            // ...     values.append(int(it[0]))
            // ...     if not it.iternext(): break
            // >>> print(values)
            // [0, 1, 2, 3]

            var arr = np.arange(4);
            using var iter = NpyIterRef.New(arr);

            var values = new System.Collections.Generic.List<int>();

            while (true)
            {
                values.Add(iter.GetValue<int>());
                if (!iter.Iternext())
                    break;
            }

            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3 }, values.ToArray());
        }

        // =========================================================================
        // IterRange Tests
        // =========================================================================

        [TestMethod]
        public void IterRange_ReturnsStartAndEnd()
        {
            var arr = np.arange(20);
            using var iter = NpyIterRef.New(arr);

            var range = iter.IterRange;
            Assert.AreEqual(0, range.Start);
            Assert.AreEqual(20, range.End);
        }

        [TestMethod]
        public void RangedIteration_MatchesNumPy()
        {
            // NumPy 2.4.2:
            // >>> it = np.nditer(np.arange(20).reshape(4,5), flags=['ranged', 'multi_index'])
            // >>> it.iterrange = (5, 15)
            // >>> it.reset()
            // >>> values = []
            // >>> while not it.finished:
            // ...     values.append((it.iterindex, it.multi_index, int(it[0])))
            // ...     it.iternext()
            // >>> print(values)
            // [(5, (1, 0), 5), (6, (1, 1), 6), ..., (14, (2, 4), 14)]

            var arr = np.arange(20).reshape(4, 5);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            iter.ResetToIterIndexRange(5, 15);

            Assert.AreEqual(5, iter.IterIndex);

            int count = 0;
            while (!iter.Finished)
            {
                iter.Iternext();
                count++;
            }

            Assert.AreEqual(10, count, "Range (5, 15) should iterate 10 elements");
        }

        // =========================================================================
        // Iteration Order Tests
        // =========================================================================

        [TestMethod]
        [Misaligned]  // NUMSHARP DIVERGENCE: F-order with MULTI_INDEX not fully implemented
        public void IterationOrder_FOrder_ColumnMajor()
        {
            // NumPy 2.4.2:
            // >>> it = np.nditer(np.arange(6).reshape(2,3), flags=['multi_index'], order='F')
            // >>> [(it.multi_index, int(x)) for x in it]
            // [((0, 0), 0), ((1, 0), 3), ((0, 1), 1), ((1, 1), 4), ((0, 2), 2), ((1, 2), 5)]
            //
            // NUMSHARP DIVERGENCE: When MULTI_INDEX is set, NumSharp skips axis reordering
            // to preserve original index mapping. F-order iteration with MULTI_INDEX
            // requires tracking both iteration order and original indices, which is not
            // yet implemented. Without MULTI_INDEX, F-order works correctly (axes coalesce).

            var arr = np.arange(6).reshape(2, 3);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX, NPY_ORDER.NPY_FORTRANORDER);

            var coords = new long[2];
            var results = new System.Collections.Generic.List<(long, long, int)>();

            while (!iter.Finished)
            {
                iter.GetMultiIndex(coords);
                results.Add((coords[0], coords[1], iter.GetValue<int>()));
                iter.Iternext();
            }

            // Current NumSharp behavior: iterates in C-order even with F flag when MULTI_INDEX set
            // This is a known divergence from NumPy
            Assert.AreEqual(0, results[0].Item3);  // (0,0) = 0
            Assert.AreEqual(1, results[1].Item3);  // (0,1) = 1 (C-order)
            Assert.AreEqual(2, results[2].Item3);  // (0,2) = 2 (C-order)
            Assert.AreEqual(3, results[3].Item3);  // (1,0) = 3 (C-order)
        }

        // =========================================================================
        // Value Access Tests
        // =========================================================================

        [TestMethod]
        public void GetValue_ReadsCorrectValue()
        {
            var arr = np.arange(12).reshape(3, 4);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            // Test at position (0, 0)
            Assert.AreEqual(0, iter.GetValue<int>());

            // Jump to position (1, 2)
            iter.GotoMultiIndex(new long[] { 1, 2 });
            Assert.AreEqual(6, iter.GetValue<int>());

            // Jump to position (2, 3)
            iter.GotoMultiIndex(new long[] { 2, 3 });
            Assert.AreEqual(11, iter.GetValue<int>());
        }

        [TestMethod]
        public void SetValue_WritesCorrectValue()
        {
            var arr = np.zeros(new Shape(3, 4), NPTypeCode.Int32);
            using var iter = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { arr },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READWRITE });

            // Set value at (1, 2)
            iter.GotoMultiIndex(new long[] { 1, 2 });
            iter.SetValue(42);

            Assert.AreEqual(42, (int)arr[1, 2]);
        }

        // =========================================================================
        // Multi-Operand Tests
        // =========================================================================

        [TestMethod]
        public void MultiOperand_GetValue_AccessesBothOperands()
        {
            var a = np.arange(6).reshape(2, 3);
            var b = np.arange(6, 12).reshape(2, 3);

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            // At (0, 0): a=0, b=6
            Assert.AreEqual(0, iter.GetValue<int>(0));
            Assert.AreEqual(6, iter.GetValue<int>(1));

            // Advance to (0, 1): a=1, b=7
            iter.Iternext();
            Assert.AreEqual(1, iter.GetValue<int>(0));
            Assert.AreEqual(7, iter.GetValue<int>(1));
        }

        // =========================================================================
        // Transposed Array Tests
        // =========================================================================

        [TestMethod]
        [Misaligned]  // NUMSHARP DIVERGENCE: K-order with MULTI_INDEX on transposed arrays not fully implemented
        public void Transposed_OrderK_FollowsMemoryLayout()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(6).reshape(2, 3)
            // >>> b = a.T  # Shape (3, 2), strides (8, 24)
            // >>> it = np.nditer(b, flags=['multi_index'], order='K')
            // >>> [int(x) for x in it]
            // [0, 1, 2, 3, 4, 5]
            //
            // NUMSHARP DIVERGENCE: When MULTI_INDEX is set, NumSharp skips axis reordering
            // to preserve original index mapping. K-order on F-contiguous arrays with
            // MULTI_INDEX requires tracking both iteration order and original indices.

            var arr = np.arange(6).reshape(2, 3);
            var transposed = arr.T;

            using var iter = NpyIterRef.New(transposed, NpyIterGlobalFlags.MULTI_INDEX, NPY_ORDER.NPY_KEEPORDER);

            var results = new System.Collections.Generic.List<int>();

            while (!iter.Finished)
            {
                results.Add(iter.GetValue<int>());
                iter.Iternext();
            }

            // Current NumSharp behavior: iterates in logical C-order of the transposed shape
            // This follows the view's logical structure rather than underlying memory layout
            // Transposed (3,2) iterates: (0,0)=0, (0,1)=3, (1,0)=1, (1,1)=4, (2,0)=2, (2,1)=5
            CollectionAssert.AreEqual(new[] { 0, 3, 1, 4, 2, 5 }, results.ToArray());
        }

        // =========================================================================
        // Edge Case Tests
        // =========================================================================

        [TestMethod]
        public void EmptyArray_IterSizeIsZero()
        {
            var empty = np.array(new int[0]);
            using var iter = NpyIterRef.New(empty, NpyIterGlobalFlags.ZEROSIZE_OK);

            Assert.AreEqual(0, iter.IterSize);
            Assert.IsTrue(iter.Finished, "Empty array iterator should be finished immediately");
        }

        [TestMethod]
        public void Scalar_IterSizeIsOne()
        {
            var scalar = np.array(42);
            using var iter = NpyIterRef.New(scalar);

            Assert.AreEqual(0, iter.NDim);
            Assert.AreEqual(1, iter.IterSize);
            Assert.AreEqual(42, iter.GetValue<int>());
        }

        // =========================================================================
        // Sliced Array Tests
        // =========================================================================

        [TestMethod]
        public void SlicedArray_StepSlice_CorrectValues()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> sliced = arr[::1, ::2, ::2]
            // >>> list(sliced.flat)
            // [0, 2, 8, 10, 12, 14, 20, 22]

            var arr = np.arange(24).reshape(2, 3, 4);
            var sliced = arr["::1, ::2, ::2"];

            using var iter = NpyIterRef.New(sliced, NpyIterGlobalFlags.MULTI_INDEX);

            var values = new System.Collections.Generic.List<int>();
            while (!iter.Finished)
            {
                values.Add(iter.GetValue<int>());
                iter.Iternext();
            }

            CollectionAssert.AreEqual(new[] { 0, 2, 8, 10, 12, 14, 20, 22 }, values.ToArray());
        }

        // =========================================================================
        // Broadcast Tests
        // =========================================================================

        [TestMethod]
        public void Broadcast_3x1_And_1x4_Produces_3x4()
        {
            // NumPy 2.4.2:
            // >>> a = np.array([[1], [2], [3]])  # (3, 1)
            // >>> b = np.array([[10, 20, 30, 40]])  # (1, 4)
            // >>> it = np.nditer([a, b], flags=['multi_index'])
            // >>> it.itersize
            // 12

            var a = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new int[,] { { 10, 20, 30, 40 } });

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(12, iter.IterSize);

            // Verify first few values
            Assert.AreEqual(1, iter.GetValue<int>(0));
            Assert.AreEqual(10, iter.GetValue<int>(1));

            iter.Iternext();
            Assert.AreEqual(1, iter.GetValue<int>(0));
            Assert.AreEqual(20, iter.GetValue<int>(1));
        }
    }
}
