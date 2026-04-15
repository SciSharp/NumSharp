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
        public void IterationOrder_FOrder_ColumnMajor()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(6).reshape(2, 3)
            // >>> it = np.nditer(a, flags=['multi_index'], order='F')
            // >>> [(it.multi_index, int(x)) for x in it]
            // [((0, 0), 0), ((1, 0), 3), ((0, 1), 1), ((1, 1), 4), ((0, 2), 2), ((1, 2), 5)]
            //
            // F-order iteration: first axis changes fastest (column-major)

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

            // F-order: iterates column by column (first axis changes fastest)
            Assert.AreEqual(6, results.Count);
            Assert.AreEqual(0, results[0].Item3);  // (0,0) = 0
            Assert.AreEqual(3, results[1].Item3);  // (1,0) = 3
            Assert.AreEqual(1, results[2].Item3);  // (0,1) = 1
            Assert.AreEqual(4, results[3].Item3);  // (1,1) = 4
            Assert.AreEqual(2, results[4].Item3);  // (0,2) = 2
            Assert.AreEqual(5, results[5].Item3);  // (1,2) = 5
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
        public void Transposed_OrderK_FollowsMemoryLayout()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(6).reshape(2, 3)
            // >>> b = a.T  # Shape (3, 2), strides (8, 24) - effectively F-contiguous
            // >>> it = np.nditer(b, flags=['multi_index'], order='K')
            // >>> [int(x) for x in it]
            // [0, 1, 2, 3, 4, 5]
            //
            // K-order follows memory layout: smallest stride (8) is axis 0, so iterate axis 0 first
            // Values are accessed in memory order: 0, 1, 2, 3, 4, 5

            var arr = np.arange(6).reshape(2, 3);
            var transposed = arr.T;  // (3, 2) with strides [1, 3] in element units

            using var iter = NpyIterRef.New(transposed, NpyIterGlobalFlags.MULTI_INDEX, NPY_ORDER.NPY_KEEPORDER);

            var results = new System.Collections.Generic.List<int>();

            while (!iter.Finished)
            {
                results.Add(iter.GetValue<int>());
                iter.Iternext();
            }

            // K-order on transposed: follows memory layout (values 0,1,2,3,4,5)
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3, 4, 5 }, results.ToArray());
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

        // =========================================================================
        // GotoIndex Tests
        // =========================================================================

        [TestMethod]
        public void GotoIndex_CIndex_JumpsToCorrectPosition()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(12).reshape(3, 4)
            // >>> it = np.nditer(a, flags=['c_index', 'multi_index'])
            // C_INDEX formula: c_index = row * 4 + col

            var arr = np.arange(12).reshape(3, 4);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.C_INDEX | NpyIterGlobalFlags.MULTI_INDEX);

            var coords = new long[2];

            // Jump to c_index=5 -> (1, 1) = value 5
            iter.GotoIndex(5);
            iter.GetMultiIndex(coords);
            Assert.AreEqual(5, iter.GetIndex());
            Assert.AreEqual(1, coords[0]);
            Assert.AreEqual(1, coords[1]);
            Assert.AreEqual(5, iter.GetValue<int>());

            // Jump to c_index=11 -> (2, 3) = value 11
            iter.GotoIndex(11);
            iter.GetMultiIndex(coords);
            Assert.AreEqual(11, iter.GetIndex());
            Assert.AreEqual(2, coords[0]);
            Assert.AreEqual(3, coords[1]);
            Assert.AreEqual(11, iter.GetValue<int>());

            // Jump back to c_index=0 -> (0, 0) = value 0
            iter.GotoIndex(0);
            iter.GetMultiIndex(coords);
            Assert.AreEqual(0, iter.GetIndex());
            Assert.AreEqual(0, coords[0]);
            Assert.AreEqual(0, coords[1]);
            Assert.AreEqual(0, iter.GetValue<int>());
        }

        [TestMethod]
        public void GotoIndex_FIndex_JumpsToCorrectPosition()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(12).reshape(3, 4)
            // >>> it = np.nditer(a, flags=['f_index', 'multi_index'])
            // F_INDEX formula: f_index = col * 3 + row

            var arr = np.arange(12).reshape(3, 4);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.F_INDEX | NpyIterGlobalFlags.MULTI_INDEX);

            var coords = new long[2];

            // F_INDEX=5 -> row = 5 % 3 = 2, col = 5 / 3 = 1 -> (2, 1) = value 9
            iter.GotoIndex(5);
            iter.GetMultiIndex(coords);
            Assert.AreEqual(5, iter.GetIndex());
            Assert.AreEqual(2, coords[0]);
            Assert.AreEqual(1, coords[1]);
            Assert.AreEqual(9, iter.GetValue<int>());

            // F_INDEX=7 -> row = 7 % 3 = 1, col = 7 / 3 = 2 -> (1, 2) = value 6
            iter.GotoIndex(7);
            iter.GetMultiIndex(coords);
            Assert.AreEqual(7, iter.GetIndex());
            Assert.AreEqual(1, coords[0]);
            Assert.AreEqual(2, coords[1]);
            Assert.AreEqual(6, iter.GetValue<int>());
        }

        [TestMethod]
        public void GotoIndex_3D_CIndex()
        {
            // NumPy 2.4.2:
            // >>> b = np.arange(24).reshape(2, 3, 4)
            // C_INDEX formula: c_index = d0 * 12 + d1 * 4 + d2

            var arr = np.arange(24).reshape(2, 3, 4);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.C_INDEX | NpyIterGlobalFlags.MULTI_INDEX);

            var coords = new long[3];

            // c_index=13 -> (1, 0, 1) = value 13
            iter.GotoIndex(13);
            iter.GetMultiIndex(coords);
            Assert.AreEqual(13, iter.GetIndex());
            Assert.AreEqual(1, coords[0]);
            Assert.AreEqual(0, coords[1]);
            Assert.AreEqual(1, coords[2]);
            Assert.AreEqual(13, iter.GetValue<int>());

            // c_index=23 -> (1, 2, 3) = value 23
            iter.GotoIndex(23);
            iter.GetMultiIndex(coords);
            Assert.AreEqual(23, iter.GetIndex());
            Assert.AreEqual(1, coords[0]);
            Assert.AreEqual(2, coords[1]);
            Assert.AreEqual(3, coords[2]);
            Assert.AreEqual(23, iter.GetValue<int>());
        }

        [TestMethod]
        public void CIndex_FOrderIteration_TracksOriginalArrayIndex()
        {
            // NumPy 2.4.2:
            // >>> it = np.nditer(np.arange(12).reshape(3,4), flags=['c_index', 'multi_index'], order='F')
            // >>> [(it.index, it.multi_index, int(it[0])) for i in range(6) if not it.iternext() or True]
            // [(0, (0, 0), 0), (4, (1, 0), 4), (8, (2, 0), 8), (1, (0, 1), 1), (5, (1, 1), 5), (9, (2, 1), 9)]
            //
            // Note: c_index tracks position in ORIGINAL array's C-order, not iteration order

            var arr = np.arange(12).reshape(3, 4);
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.C_INDEX | NpyIterGlobalFlags.MULTI_INDEX, NPY_ORDER.NPY_FORTRANORDER);

            var expected = new[] {
                (0, (0L, 0L), 0),
                (4, (1L, 0L), 4),
                (8, (2L, 0L), 8),
                (1, (0L, 1L), 1),
                (5, (1L, 1L), 5),
                (9, (2L, 1L), 9)
            };

            var coords = new long[2];
            for (int i = 0; i < 6; i++)
            {
                iter.GetMultiIndex(coords);
                Assert.AreEqual(expected[i].Item1, iter.GetIndex(), $"c_index mismatch at iteration {i}");
                Assert.AreEqual(expected[i].Item2.Item1, coords[0], $"row mismatch at iteration {i}");
                Assert.AreEqual(expected[i].Item2.Item2, coords[1], $"col mismatch at iteration {i}");
                Assert.AreEqual(expected[i].Item3, iter.GetValue<int>(), $"value mismatch at iteration {i}");
                iter.Iternext();
            }
        }

        // =========================================================================
        // Copy Tests
        // =========================================================================

        [TestMethod]
        public void Copy_CreatesIndependentIterator()
        {
            // NumPy 2.4.2:
            // >>> it1 = np.nditer(np.arange(12).reshape(3,4), flags=['multi_index'])
            // >>> for i in range(5): it1.iternext()
            // >>> it2 = it1.copy()
            // >>> it1.multi_index, it2.multi_index
            // ((1, 1), (1, 1))
            // >>> it1.iternext()
            // >>> it1.multi_index, it2.multi_index
            // ((1, 2), (1, 1))

            var arr = np.arange(12).reshape(3, 4);
            using var it1 = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            // Advance 5 positions
            for (int i = 0; i < 5; i++)
                it1.Iternext();

            var coords1 = new long[2];
            var coords2 = new long[2];

            it1.GetMultiIndex(coords1);
            Assert.AreEqual(1, coords1[0]);
            Assert.AreEqual(1, coords1[1]);

            // Copy
            using var it2 = it1.Copy();
            it2.GetMultiIndex(coords2);
            Assert.AreEqual(1, coords2[0]);
            Assert.AreEqual(1, coords2[1]);

            // Advance original only
            it1.Iternext();
            it1.GetMultiIndex(coords1);
            it2.GetMultiIndex(coords2);

            // Original advanced
            Assert.AreEqual(1, coords1[0]);
            Assert.AreEqual(2, coords1[1]);

            // Copy unchanged
            Assert.AreEqual(1, coords2[0]);
            Assert.AreEqual(1, coords2[1]);
        }

        [TestMethod]
        public void Copy_PreservesFlags()
        {
            var arr = np.arange(12).reshape(3, 4);
            using var it1 = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            it1.GotoIndex(5);

            using var it2 = it1.Copy();

            Assert.AreEqual(it1.HasMultiIndex, it2.HasMultiIndex);
            Assert.AreEqual(it1.HasIndex, it2.HasIndex);
            Assert.AreEqual(it1.GetIndex(), it2.GetIndex());
            Assert.AreEqual(it1.GetValue<int>(), it2.GetValue<int>());
        }

        [TestMethod]
        public void Copy_ResetDoesNotAffectOriginal()
        {
            var arr = np.arange(12).reshape(3, 4);
            using var it1 = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            // Advance to position 6
            for (int i = 0; i < 6; i++)
                it1.Iternext();

            using var it2 = it1.Copy();

            // Reset copy
            it2.Reset();

            var coords1 = new long[2];
            var coords2 = new long[2];

            it1.GetMultiIndex(coords1);
            it2.GetMultiIndex(coords2);

            // Original still at (1, 2)
            Assert.AreEqual(1, coords1[0]);
            Assert.AreEqual(2, coords1[1]);

            // Copy at (0, 0)
            Assert.AreEqual(0, coords2[0]);
            Assert.AreEqual(0, coords2[1]);
        }

        // =========================================================================
        // Negative Stride Flipping Tests (NumPy Parity)
        // =========================================================================
        // NumPy flips negative strides for memory-order iteration while tracking
        // flipped coordinates via negative Perm entries. These tests verify NumSharp
        // matches NumPy's behavior exactly.
        // =========================================================================

        [TestMethod]
        public void NegativeStride_1D_IteratesMemoryOrder()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(5)
            // >>> rev = arr[::-1]  # strides: (-8,)
            // >>> it = np.nditer(rev, flags=['multi_index', 'c_index'])
            // >>> [(it.multi_index, it.index, int(x)) for x in it]
            // [((4,), 4, 0), ((3,), 3, 1), ((2,), 2, 2), ((1,), 1, 3), ((0,), 0, 4)]
            //
            // Key behavior:
            // - Iterates in MEMORY order (values 0,1,2,3,4)
            // - multi_index reports ORIGINAL coordinates (4,3,2,1,0)
            // - c_index is flat index in original array (4,3,2,1,0)

            var arr = np.arange(5);
            var rev = arr["::-1"];

            // NumSharp uses element strides, not byte strides like NumPy
            // NumPy: -8 bytes = -1 element (sizeof(long) = 8)
            Assert.AreEqual(-1, rev.strides[0], "Reversed array should have negative stride");

            using var iter = NpyIterRef.New(rev, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            var coords = new long[1];
            var expectedValues = new int[] { 0, 1, 2, 3, 4 };  // Memory order
            var expectedMultiIndex = new long[] { 4, 3, 2, 1, 0 };  // Flipped
            var expectedCIndex = new long[] { 4, 3, 2, 1, 0 };  // Original positions

            for (int i = 0; i < 5; i++)
            {
                iter.GetMultiIndex(coords);
                var value = iter.GetValue<long>(0);
                var cIndex = iter.GetIndex();

                Assert.AreEqual(expectedValues[i], value, $"Value at iteration {i}");
                Assert.AreEqual(expectedMultiIndex[i], coords[0], $"MultiIndex at iteration {i}");
                Assert.AreEqual(expectedCIndex[i], cIndex, $"C_INDEX at iteration {i}");

                if (i < 4) iter.Iternext();
            }
        }

        [TestMethod]
        public void NegativeStride_2D_RowReversed_IteratesMemoryOrder()
        {
            // NumPy 2.4.2:
            // >>> arr2d = np.arange(6).reshape(2, 3)
            // >>> rev2d = arr2d[::-1, :]  # strides: (-24, 8)
            // >>> it = np.nditer(rev2d, flags=['multi_index', 'c_index'])
            // >>> [(it.multi_index, it.index, int(x)) for x in it]
            // [((1, 0), 3, 0), ((1, 1), 4, 1), ((1, 2), 5, 2),
            //  ((0, 0), 0, 3), ((0, 1), 1, 4), ((0, 2), 2, 5)]
            //
            // Values 0,1,2,3,4,5 in memory order
            // multi_index: first axis flipped

            var arr2d = np.arange(6).reshape(2, 3);
            var rev2d = arr2d["::-1, :"];

            // NumSharp uses element strides: -24 bytes / 8 = -3 elements, 8 bytes / 8 = 1 element
            Assert.AreEqual(-3, rev2d.strides[0], "First axis should have negative stride");
            Assert.AreEqual(1, rev2d.strides[1], "Second axis should have positive stride");

            using var iter = NpyIterRef.New(rev2d, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            var coords = new long[2];
            var expectedValues = new int[] { 0, 1, 2, 3, 4, 5 };
            var expectedMultiIndex = new long[,] { { 1, 0 }, { 1, 1 }, { 1, 2 }, { 0, 0 }, { 0, 1 }, { 0, 2 } };
            var expectedCIndex = new long[] { 3, 4, 5, 0, 1, 2 };

            for (int i = 0; i < 6; i++)
            {
                iter.GetMultiIndex(coords);
                var value = iter.GetValue<long>(0);
                var cIndex = iter.GetIndex();

                Assert.AreEqual(expectedValues[i], value, $"Value at iteration {i}");
                Assert.AreEqual(expectedMultiIndex[i, 0], coords[0], $"MultiIndex[0] at iteration {i}");
                Assert.AreEqual(expectedMultiIndex[i, 1], coords[1], $"MultiIndex[1] at iteration {i}");
                Assert.AreEqual(expectedCIndex[i], cIndex, $"C_INDEX at iteration {i}");

                if (i < 5) iter.Iternext();
            }
        }

        [TestMethod]
        public void NegativeStride_2D_ColReversed_IteratesMemoryOrder()
        {
            // NumPy 2.4.2:
            // >>> arr2d = np.arange(6).reshape(2, 3)
            // >>> rev2d = arr2d[:, ::-1]  # strides: (24, -8)
            // >>> it = np.nditer(rev2d, flags=['multi_index', 'c_index'])
            // >>> [(it.multi_index, it.index, int(x)) for x in it]
            // [((0, 2), 2, 0), ((0, 1), 1, 1), ((0, 0), 0, 2),
            //  ((1, 2), 5, 3), ((1, 1), 4, 4), ((1, 0), 3, 5)]
            //
            // Values 0,1,2,3,4,5 in memory order
            // multi_index: second axis flipped

            var arr2d = np.arange(6).reshape(2, 3);
            var rev2d = arr2d[":, ::-1"];

            // NumSharp uses element strides: 24 bytes / 8 = 3 elements, -8 bytes / 8 = -1 element
            Assert.AreEqual(3, rev2d.strides[0], "First axis should have positive stride");
            Assert.AreEqual(-1, rev2d.strides[1], "Second axis should have negative stride");

            using var iter = NpyIterRef.New(rev2d, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            var coords = new long[2];
            var expectedValues = new int[] { 0, 1, 2, 3, 4, 5 };
            var expectedMultiIndex = new long[,] { { 0, 2 }, { 0, 1 }, { 0, 0 }, { 1, 2 }, { 1, 1 }, { 1, 0 } };
            var expectedCIndex = new long[] { 2, 1, 0, 5, 4, 3 };

            for (int i = 0; i < 6; i++)
            {
                iter.GetMultiIndex(coords);
                var value = iter.GetValue<long>(0);
                var cIndex = iter.GetIndex();

                Assert.AreEqual(expectedValues[i], value, $"Value at iteration {i}");
                Assert.AreEqual(expectedMultiIndex[i, 0], coords[0], $"MultiIndex[0] at iteration {i}");
                Assert.AreEqual(expectedMultiIndex[i, 1], coords[1], $"MultiIndex[1] at iteration {i}");
                Assert.AreEqual(expectedCIndex[i], cIndex, $"C_INDEX at iteration {i}");

                if (i < 5) iter.Iternext();
            }
        }

        [TestMethod]
        public void NegativeStride_2D_BothReversed_IteratesMemoryOrder()
        {
            // NumPy 2.4.2:
            // >>> arr2d = np.arange(6).reshape(2, 3)
            // >>> rev2d = arr2d[::-1, ::-1]  # strides: (-24, -8)
            // >>> it = np.nditer(rev2d, flags=['multi_index', 'c_index'])
            // >>> [(it.multi_index, it.index, int(x)) for x in it]
            // [((1, 2), 5, 0), ((1, 1), 4, 1), ((1, 0), 3, 2),
            //  ((0, 2), 2, 3), ((0, 1), 1, 4), ((0, 0), 0, 5)]
            //
            // Values 0,1,2,3,4,5 in memory order
            // multi_index: both axes flipped

            var arr2d = np.arange(6).reshape(2, 3);
            var rev2d = arr2d["::-1, ::-1"];

            // NumSharp uses element strides: -24 bytes / 8 = -3 elements, -8 bytes / 8 = -1 element
            Assert.AreEqual(-3, rev2d.strides[0], "First axis should have negative stride");
            Assert.AreEqual(-1, rev2d.strides[1], "Second axis should have negative stride");

            using var iter = NpyIterRef.New(rev2d, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            var coords = new long[2];
            var expectedValues = new int[] { 0, 1, 2, 3, 4, 5 };
            var expectedMultiIndex = new long[,] { { 1, 2 }, { 1, 1 }, { 1, 0 }, { 0, 2 }, { 0, 1 }, { 0, 0 } };
            var expectedCIndex = new long[] { 5, 4, 3, 2, 1, 0 };

            for (int i = 0; i < 6; i++)
            {
                iter.GetMultiIndex(coords);
                var value = iter.GetValue<long>(0);
                var cIndex = iter.GetIndex();

                Assert.AreEqual(expectedValues[i], value, $"Value at iteration {i}");
                Assert.AreEqual(expectedMultiIndex[i, 0], coords[0], $"MultiIndex[0] at iteration {i}");
                Assert.AreEqual(expectedMultiIndex[i, 1], coords[1], $"MultiIndex[1] at iteration {i}");
                Assert.AreEqual(expectedCIndex[i], cIndex, $"C_INDEX at iteration {i}");

                if (i < 5) iter.Iternext();
            }
        }

        [TestMethod]
        public void NegativeStride_WithDontNegateStrides_PreservesViewOrder()
        {
            // NumPy 2.4.2:
            // When DONT_NEGATE_STRIDES is set, NumPy does NOT flip negative strides
            // and iterates in view logical order instead of memory order.
            //
            // >>> arr = np.arange(5)
            // >>> rev = arr[::-1]
            // >>> # With DONT_NEGATE_STRIDES, iteration follows view order
            // >>> # Values would be: 4, 3, 2, 1, 0 (view logical order)
            // >>> # multi_index: (0,), (1,), (2,), (3,), (4,) (no flipping)

            var arr = np.arange(5);
            var rev = arr["::-1"];

            using var iter = NpyIterRef.New(rev,
                NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX | NpyIterGlobalFlags.DONT_NEGATE_STRIDES);

            var coords = new long[1];
            var expectedValues = new int[] { 4, 3, 2, 1, 0 };  // View logical order
            var expectedMultiIndex = new long[] { 0, 1, 2, 3, 4 };  // No flipping

            for (int i = 0; i < 5; i++)
            {
                iter.GetMultiIndex(coords);
                var value = iter.GetValue<long>(0);

                Assert.AreEqual(expectedValues[i], value, $"Value at iteration {i}");
                Assert.AreEqual(expectedMultiIndex[i], coords[0], $"MultiIndex at iteration {i}");

                if (i < 4) iter.Iternext();
            }
        }

        [TestMethod]
        public void NegativeStride_GotoMultiIndex_WorksWithFlippedAxes()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(6).reshape(2, 3)
            // >>> rev = arr[::-1, :]
            // >>> it = np.nditer(rev, flags=['multi_index'])
            // >>> it[0]  # Access value at current position
            // array(0)
            // >>> # After GotoMultiIndex([0, 0]), we should be at original position (0,0)
            // >>> # which contains value 3 in the reversed view

            var arr2d = np.arange(6).reshape(2, 3);
            var rev2d = arr2d["::-1, :"];

            using var iter = NpyIterRef.New(rev2d, NpyIterGlobalFlags.MULTI_INDEX);

            // In NumPy, multi_index=(0,0) refers to original array position (0,0)
            // After flipping, this is at the "end" of memory iteration
            iter.GotoMultiIndex(new long[] { 0, 0 });

            var value = iter.GetValue<long>(0);
            Assert.AreEqual(3, value, "GotoMultiIndex([0,0]) should give original value at (0,0)");

            iter.GotoMultiIndex(new long[] { 1, 0 });
            value = iter.GetValue<long>(0);
            Assert.AreEqual(0, value, "GotoMultiIndex([1,0]) should give original value at (1,0)");
        }

        [TestMethod]
        public void NegativeStride_GotoIndex_WorksWithFlippedAxes()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(6).reshape(2, 3)
            // >>> rev = arr[::-1, :]
            // >>> it = np.nditer(rev, flags=['multi_index', 'c_index'])
            // >>> # GotoIndex(0) should go to original flat index 0
            // >>> # which is multi_index=(0,0) containing value 3

            var arr2d = np.arange(6).reshape(2, 3);
            var rev2d = arr2d["::-1, :"];

            using var iter = NpyIterRef.New(rev2d, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX);

            // C_INDEX=0 means original position (0,0) which has value 3
            iter.GotoIndex(0);
            var value = iter.GetValue<long>(0);
            Assert.AreEqual(3, value, "GotoIndex(0) should give value at original flat index 0");

            // C_INDEX=3 means original position (1,0) which has value 0
            iter.GotoIndex(3);
            value = iter.GetValue<long>(0);
            Assert.AreEqual(0, value, "GotoIndex(3) should give value at original flat index 3");
        }

        [TestMethod]
        public void NegativeStride_3D_PartiallyReversed_IteratesMemoryOrder()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> rev = arr[::-1, :, ::-1]  # Reverse first and last axes
            // >>> rev.strides
            // (-96, 32, -8)
            // >>> it = np.nditer(rev, flags=['multi_index'])
            // First few iterations...

            var arr = np.arange(24).reshape(2, 3, 4);
            var rev = arr["::-1, :, ::-1"];

            // NumSharp uses element strides: -96/8=-12, 32/8=4, -8/8=-1
            Assert.AreEqual(-12, rev.strides[0], "First axis should have negative stride");
            Assert.AreEqual(4, rev.strides[1], "Second axis should have positive stride");
            Assert.AreEqual(-1, rev.strides[2], "Third axis should have negative stride");

            using var iter = NpyIterRef.New(rev, NpyIterGlobalFlags.MULTI_INDEX);

            var coords = new long[3];

            // First iteration should be at memory position 0
            iter.GetMultiIndex(coords);
            var value = iter.GetValue<long>(0);

            // At memory position 0: original (0,0,0) = value 0
            // With axes 0 and 2 flipped: multi_index = (1, 0, 3)
            Assert.AreEqual(0, value, "First value should be 0 (memory order)");
            Assert.AreEqual(1, coords[0], "First axis flipped: multi_index[0] = 1");
            Assert.AreEqual(0, coords[1], "Second axis not flipped: multi_index[1] = 0");
            Assert.AreEqual(3, coords[2], "Third axis flipped: multi_index[2] = 3");
        }

        [TestMethod]
        public void NegativeStride_MixedOperands_OnlyFlipsWhenAllNegative()
        {
            // NumPy only flips strides when ALL operands have negative or zero stride
            // for a given axis. If one operand has positive stride, no flipping occurs.
            //
            // This test uses two operands: one reversed, one not reversed on same axis.

            var arr1 = np.arange(6).reshape(2, 3);  // strides (24, 8)
            var arr2 = arr1["::-1, :"];  // strides (-24, 8)

            using var iter = NpyIterRef.MultiNew(
                2,
                new[] { arr1, arr2 },
                NpyIterGlobalFlags.MULTI_INDEX,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY },
                null);

            var coords = new long[2];

            // Since arr1 has positive stride on axis 0 and arr2 has negative,
            // no flipping should occur (one positive prevents flip).
            // Iteration should follow arr1's order (values 0,1,2,3,4,5)
            iter.GetMultiIndex(coords);
            var v1 = iter.GetValue<long>(0);
            var v2 = iter.GetValue<long>(1);

            // At (0,0): arr1=0, arr2=3 (arr2 is reversed so sees row 1)
            Assert.AreEqual(0, v1, "arr1 value at (0,0)");
            Assert.AreEqual(3, v2, "arr2 value at (0,0) from reversed view");
        }

        [TestMethod]
        public void NegativeStride_NEGPERM_FlagIsSet()
        {
            // Verify that the NEGPERM flag is set when axes are flipped

            var arr = np.arange(5);
            var rev = arr["::-1"];

            using var iter = NpyIterRef.New(rev, NpyIterGlobalFlags.MULTI_INDEX);

            // When negative strides are flipped, NEGPERM should be set
            // and IDENTPERM should be cleared
            Assert.IsTrue(iter.HasNegPerm, "NEGPERM flag should be set for flipped axes");
            Assert.IsFalse(iter.HasIdentPerm, "IDENTPERM flag should be cleared when NEGPERM is set");
        }

        [TestMethod]
        public void NegativeStride_WithoutMultiIndex_StillIteratesMemoryOrder()
        {
            // Even without MULTI_INDEX flag, iteration should be in memory order
            // for cache efficiency.

            var arr = np.arange(5);
            var rev = arr["::-1"];

            using var iter = NpyIterRef.New(rev);  // No flags

            var values = new List<long>();
            do
            {
                values.Add(iter.GetValue<long>(0));
            } while (iter.Iternext());

            // Should iterate in memory order: 0, 1, 2, 3, 4
            CollectionAssert.AreEqual(new long[] { 0, 1, 2, 3, 4 }, values.ToArray(),
                "Without MULTI_INDEX, should still iterate memory order");
        }

        // =========================================================================
        // GetIterView Tests
        // =========================================================================
        // GetIterView returns an NDArray view with the iterator's internal axes
        // ordering. A C-order iteration of this view is equivalent to the
        // iterator's iteration order.
        // =========================================================================

        [TestMethod]
        public void GetIterView_ContiguousArray_ReturnsCoalescedView()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(arr)
            // >>> it.ndim, it.shape
            // (1, (24,))
            //
            // GetIterView should return a 1D view of 24 elements
            // (coalesced from 2x3x4)

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1, iter.NDim, "Contiguous 2x3x4 should coalesce to ndim=1");

            var view = iter.GetIterView(0);

            Assert.AreEqual(1, view.ndim, "View should be 1D");
            Assert.AreEqual(24, view.size, "View should have 24 elements");
            Assert.AreEqual(24, view.shape[0], "View shape should be (24,)");

            // C-order iteration of view should give 0, 1, 2, ..., 23
            for (int i = 0; i < 24; i++)
            {
                Assert.AreEqual(i, (int)view[i], $"View element {i}");
            }
        }

        [TestMethod]
        public void GetIterView_WithMultiIndex_PreservesOriginalShape()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(arr, flags=['multi_index'])
            // >>> it.ndim, it.shape
            // (3, (2, 3, 4))
            //
            // With MULTI_INDEX, no coalescing occurs

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            Assert.AreEqual(3, iter.NDim, "With MULTI_INDEX, should preserve ndim=3");

            var view = iter.GetIterView(0);

            Assert.AreEqual(3, view.ndim, "View should be 3D");
            Assert.AreEqual(2, view.shape[0]);
            Assert.AreEqual(3, view.shape[1]);
            Assert.AreEqual(4, view.shape[2]);
        }

        [TestMethod]
        public void GetIterView_TransposedArray_ReflectsInternalOrder()
        {
            // NumPy 2.4.2:
            // >>> arr = np.arange(24).reshape(2, 3, 4).T  # Shape (4, 3, 2)
            // >>> it = np.nditer(arr, order='K')
            // >>> it.ndim, it.shape
            // (1, (24,))  # Coalesced because K-order follows memory layout
            //
            // The view should reflect the iterator's internal reordering

            var arr = np.arange(24).reshape(2, 3, 4).T;  // Shape (4, 3, 2)

            // Without MULTI_INDEX, should coalesce
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER);

            // K-order on transposed array should coalesce to 1D
            var view = iter.GetIterView(0);

            // C-order iteration of view should match iterator order
            var iterValues = new List<long>();
            do
            {
                iterValues.Add(iter.GetValue<long>(0));
            } while (iter.Iternext());

            // View iteration should match
            iter.Reset();
            for (int i = 0; i < view.size; i++)
            {
                Assert.AreEqual(iterValues[i], (long)view.flat[i], $"View[{i}] should match iterator value");
            }
        }

        [TestMethod]
        public void GetIterView_SlicedArray_HasCorrectStrides()
        {
            // Sliced arrays have non-contiguous strides
            // GetIterView should return a view with the iterator's internal strides

            var arr = np.arange(24).reshape(2, 3, 4);
            var sliced = arr[":, ::2, :"];  // Shape (2, 2, 4), non-contiguous

            using var iter = NpyIterRef.New(sliced, NpyIterGlobalFlags.MULTI_INDEX);

            var view = iter.GetIterView(0);

            Assert.AreEqual(3, view.ndim);
            Assert.AreEqual(2, view.shape[0]);
            Assert.AreEqual(2, view.shape[1]);
            Assert.AreEqual(4, view.shape[2]);

            // View should have same values as sliced array
            Assert.AreEqual((int)sliced[0, 0, 0], (int)view[0, 0, 0]);
            Assert.AreEqual((int)sliced[0, 1, 0], (int)view[0, 1, 0]);
            Assert.AreEqual((int)sliced[1, 0, 0], (int)view[1, 0, 0]);
        }

        [TestMethod]
        public void GetIterView_MultipleOperands_ReturnsCorrectView()
        {
            // With multiple operands, each GetIterView(i) returns the i-th operand's view

            var arr1 = np.arange(6).reshape(2, 3);
            var arr2 = np.arange(6, 12).reshape(2, 3);

            using var iter = NpyIterRef.MultiNew(
                2,
                new[] { arr1, arr2 },
                NpyIterGlobalFlags.MULTI_INDEX,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY },
                null);

            var view0 = iter.GetIterView(0);
            var view1 = iter.GetIterView(1);

            // view0 should have arr1's data
            Assert.AreEqual(0, (int)view0[0, 0]);
            Assert.AreEqual(5, (int)view0[1, 2]);

            // view1 should have arr2's data
            Assert.AreEqual(6, (int)view1[0, 0]);
            Assert.AreEqual(11, (int)view1[1, 2]);
        }

        [TestMethod]
        public void GetIterView_BufferedIterator_ThrowsException()
        {
            // NumPy: Cannot provide an iterator view when buffering is enabled

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.BUFFERED);

            bool threw = false;
            try
            {
                iter.GetIterView(0);
            }
            catch (InvalidOperationException)
            {
                threw = true;
            }

            Assert.IsTrue(threw, "GetIterView should throw when buffering is enabled");
        }

        [TestMethod]
        public void GetIterView_InvalidOperandIndex_ThrowsException()
        {
            var arr = np.arange(24);

            using var iter = NpyIterRef.New(arr);

            bool threwNegative = false;
            try
            {
                iter.GetIterView(-1);
            }
            catch (ArgumentOutOfRangeException)
            {
                threwNegative = true;
            }
            Assert.IsTrue(threwNegative, "Should throw for negative operand index");

            bool threwOutOfRange = false;
            try
            {
                iter.GetIterView(1);
            }
            catch (ArgumentOutOfRangeException)
            {
                threwOutOfRange = true;
            }
            Assert.IsTrue(threwOutOfRange, "Should throw for operand index >= NOp");
        }

        [TestMethod]
        public void GetIterView_ReversedArray_ReflectsFlippedStrides()
        {
            // After negative stride flipping, GetIterView should return a view
            // with the flipped (positive) strides

            var arr = np.arange(6).reshape(2, 3);
            var rev = arr["::-1, :"];  // Reversed first axis

            using var iter = NpyIterRef.New(rev, NpyIterGlobalFlags.MULTI_INDEX);

            var view = iter.GetIterView(0);

            // The view should iterate in memory order (values 0,1,2,3,4,5)
            // even though the original reversed view would iterate 3,4,5,0,1,2
            var viewValues = new List<long>();
            for (int i = 0; i < view.size; i++)
                viewValues.Add((long)view.flat[i]);

            // After flipping, iteration is in memory order
            CollectionAssert.AreEqual(new long[] { 0, 1, 2, 3, 4, 5 }, viewValues.ToArray());
        }

        // =========================================================================
        // Cast Support Tests (Type Conversion During Iteration)
        // =========================================================================
        // NumPy nditer supports automatic type conversion when op_dtypes differ
        // from the actual array dtypes. This requires BUFFERED flag and respects
        // the casting parameter (no_casting, safe, same_kind, unsafe).
        // =========================================================================

        [TestMethod]
        public void Cast_Int32ToFloat64_SafeCasting()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([1, 2, 3], dtype=np.int32)
            // >>> it = np.nditer([arr], flags=['buffered'],
            // ...                op_flags=[['readonly']],
            // ...                op_dtypes=['float64'],
            // ...                casting='safe')
            // >>> [float(x) for x in it]
            // [1.0, 2.0, 3.0]

            var arr = np.array(new int[] { 1, 2, 3 });
            Assert.AreEqual(NPTypeCode.Int32, arr.typecode);

            using var iter = NpyIterRef.New(
                arr,
                NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                NPTypeCode.Double);

            var values = new List<double>();
            do
            {
                values.Add(iter.GetValue<double>(0));
            } while (iter.Iternext());

            CollectionAssert.AreEqual(new double[] { 1.0, 2.0, 3.0 }, values.ToArray());
        }

        [TestMethod]
        public void Cast_Float64ToInt32_UnsafeCasting()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
            // >>> it = np.nditer([arr], flags=['buffered'],
            // ...                op_flags=[['readonly']],
            // ...                op_dtypes=['int32'],
            // ...                casting='unsafe')
            // >>> [int(x) for x in it]
            // [1, 2, 3]  # Truncated

            var arr = np.array(new double[] { 1.5, 2.5, 3.5 });
            Assert.AreEqual(NPTypeCode.Double, arr.typecode);

            using var iter = NpyIterRef.New(
                arr,
                NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_UNSAFE_CASTING,
                NPTypeCode.Int32);

            var values = new List<int>();
            do
            {
                values.Add(iter.GetValue<int>(0));
            } while (iter.Iternext());

            // Values should be truncated
            CollectionAssert.AreEqual(new int[] { 1, 2, 3 }, values.ToArray());
        }

        [TestMethod]
        public void Cast_Float64ToInt32_SafeCasting_Throws()
        {
            // NumPy 2.4.2:
            // >>> arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
            // >>> it = np.nditer([arr], flags=['buffered'],
            // ...                op_flags=[['readonly']],
            // ...                op_dtypes=['int32'],
            // ...                casting='safe')
            // TypeError: Iterator operand 0 dtype could not be cast from dtype('float64')
            //            to dtype('int32') according to the rule 'safe'

            var arr = np.array(new double[] { 1.5, 2.5, 3.5 });

            bool threw = false;
            try
            {
                using var iter = NpyIterRef.New(
                    arr,
                    NpyIterGlobalFlags.BUFFERED,
                    NPY_ORDER.NPY_KEEPORDER,
                    NPY_CASTING.NPY_SAFE_CASTING,
                    NPTypeCode.Int32);
            }
            catch (InvalidCastException)
            {
                threw = true;
            }

            Assert.IsTrue(threw, "Should throw InvalidCastException for unsafe cast with safe casting rule");
        }

        [TestMethod]
        public void Cast_Int16ToInt32_SafeCasting()
        {
            // Safe widening cast: int16 -> int32

            var arr = np.array(new short[] { 100, 200, 300 });
            Assert.AreEqual(NPTypeCode.Int16, arr.typecode);

            using var iter = NpyIterRef.New(
                arr,
                NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                NPTypeCode.Int32);

            var values = new List<int>();
            do
            {
                values.Add(iter.GetValue<int>(0));
            } while (iter.Iternext());

            CollectionAssert.AreEqual(new int[] { 100, 200, 300 }, values.ToArray());
        }

        [TestMethod]
        public void Cast_CommonDtype_TwoOperands()
        {
            // NumPy 2.4.2:
            // >>> a = np.array([1, 2, 3], dtype=np.int32)
            // >>> b = np.array([1.5, 2.5, 3.5], dtype=np.float64)
            // >>> it = np.nditer([a, b], flags=['common_dtype', 'buffered'])
            // >>> print([str(d) for d in it.dtypes])
            // ['float64', 'float64']

            var arrInt = np.array(new int[] { 1, 2, 3 });
            var arrFloat = np.array(new double[] { 1.5, 2.5, 3.5 });

            using var iter = NpyIterRef.MultiNew(
                2,
                new[] { arrInt, arrFloat },
                NpyIterGlobalFlags.COMMON_DTYPE | NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY },
                null);  // null opDtypes = use common dtype

            // Both operands should be promoted to float64
            var dtypes = iter.GetDescrArray();
            Assert.AreEqual(NPTypeCode.Double, dtypes[0], "First operand should be cast to float64");
            Assert.AreEqual(NPTypeCode.Double, dtypes[1], "Second operand should be float64");

            // Verify values
            var vals0 = new List<double>();
            var vals1 = new List<double>();
            do
            {
                vals0.Add(iter.GetValue<double>(0));
                vals1.Add(iter.GetValue<double>(1));
            } while (iter.Iternext());

            CollectionAssert.AreEqual(new double[] { 1.0, 2.0, 3.0 }, vals0.ToArray());
            CollectionAssert.AreEqual(new double[] { 1.5, 2.5, 3.5 }, vals1.ToArray());
        }

        [TestMethod]
        public void Cast_WriteOutput_WithConversion()
        {
            // NumPy 2.4.2:
            // >>> out = np.zeros(3, dtype=np.float64)
            // >>> arr = np.array([10, 20, 30], dtype=np.int32)
            // >>> it = np.nditer([arr, out], flags=['buffered'],
            // ...                op_flags=[['readonly'], ['writeonly']],
            // ...                op_dtypes=['float64', 'float64'],
            // ...                casting='safe')
            // >>> for x, y in it:
            // ...     y[...] = x * 2.5
            // >>> out
            // array([25., 50., 75.])

            var arrIn = np.array(new int[] { 10, 20, 30 });
            var arrOut = np.zeros(3, NPTypeCode.Double);

            using var iter = NpyIterRef.MultiNew(
                2,
                new[] { arrIn, arrOut },
                NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY },
                new[] { NPTypeCode.Double, NPTypeCode.Double });

            do
            {
                var x = iter.GetValue<double>(0);
                iter.SetValue(x * 2.5, 1);  // SetValue<T>(value, operand)
            } while (iter.Iternext());

            // Verify output
            Assert.AreEqual(25.0, (double)arrOut[0], 0.001);
            Assert.AreEqual(50.0, (double)arrOut[1], 0.001);
            Assert.AreEqual(75.0, (double)arrOut[2], 0.001);
        }

        [TestMethod]
        public void Cast_SameKindCasting_IntToInt()
        {
            // Same-kind casting allows int32 -> int64 (both integers)

            var arr = np.array(new int[] { 1, 2, 3 });

            using var iter = NpyIterRef.New(
                arr,
                NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAME_KIND_CASTING,
                NPTypeCode.Int64);

            var values = new List<long>();
            do
            {
                values.Add(iter.GetValue<long>(0));
            } while (iter.Iternext());

            CollectionAssert.AreEqual(new long[] { 1, 2, 3 }, values.ToArray());
        }

        [TestMethod]
        public void Cast_SameKindCasting_IntToFloat_Throws()
        {
            // Same-kind casting does NOT allow int32 -> float64 (different kinds)
            // NumPy: "Cannot cast array data from dtype('int32') to dtype('float64')
            //         according to the rule 'same_kind'"

            var arr = np.array(new int[] { 1, 2, 3 });

            bool threw = false;
            try
            {
                using var iter = NpyIterRef.New(
                    arr,
                    NpyIterGlobalFlags.BUFFERED,
                    NPY_ORDER.NPY_KEEPORDER,
                    NPY_CASTING.NPY_SAME_KIND_CASTING,
                    NPTypeCode.Double);
            }
            catch (InvalidCastException)
            {
                threw = true;
            }

            Assert.IsTrue(threw, "Same-kind casting should not allow int -> float");
        }

        [TestMethod]
        public void Cast_NoCasting_SameType_Allowed()
        {
            // No casting: same type should be allowed

            var arr = np.array(new int[] { 1, 2, 3 });

            using var iter = NpyIterRef.New(
                arr,
                NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                NPTypeCode.Int32);  // Same as source

            var values = new List<int>();
            do
            {
                values.Add(iter.GetValue<int>(0));
            } while (iter.Iternext());

            CollectionAssert.AreEqual(new int[] { 1, 2, 3 }, values.ToArray());
        }

        [TestMethod]
        public void Cast_NoCasting_DifferentType_Throws()
        {
            // No casting: different type should throw

            var arr = np.array(new int[] { 1, 2, 3 });

            bool threw = false;
            try
            {
                using var iter = NpyIterRef.New(
                    arr,
                    NpyIterGlobalFlags.BUFFERED,
                    NPY_ORDER.NPY_KEEPORDER,
                    NPY_CASTING.NPY_NO_CASTING,
                    NPTypeCode.Int64);  // Different from source
            }
            catch (InvalidCastException)
            {
                threw = true;
            }

            Assert.IsTrue(threw, "No casting should not allow different types");
        }

        [TestMethod]
        public void Cast_RequiresBuffered_ThrowsWithoutBuffer()
        {
            // Casting requires BUFFERED flag

            var arr = np.array(new int[] { 1, 2, 3 });

            bool threw = false;
            try
            {
                // Try to cast without BUFFERED flag
                using var iter = NpyIterRef.New(
                    arr,
                    NpyIterGlobalFlags.None,  // No BUFFERED
                    NPY_ORDER.NPY_KEEPORDER,
                    NPY_CASTING.NPY_SAFE_CASTING,
                    NPTypeCode.Double);  // Different dtype
            }
            catch (ArgumentException)
            {
                threw = true;
            }

            Assert.IsTrue(threw, "Casting without BUFFERED should throw");
        }

        // =========================================================================
        // Reduction Support Tests
        // =========================================================================
        // NumPy nditer supports reduction operations where output operands have
        // fewer dimensions than inputs. This is achieved using op_axes with -1
        // entries for reduction dimensions. The iterator marks such operands
        // with stride=0 for reduction axes.
        // =========================================================================

        [TestMethod]
        public void Reduction_1DToScalar_IteratesCorrectly()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(6)
            // >>> it = np.nditer([a, None], ['reduce_ok'],
            // ...                [['readonly'], ['readwrite', 'allocate']],
            // ...                op_axes=[[0], [-1]])
            // >>> it.operands[1][...] = 0
            // >>> for x, y in it:
            // ...     y[...] += x
            // >>> int(it.operands[1])
            // 15
            //
            // -1 in op_axes means "newaxis" / broadcast / reduce on that axis

            var a = np.arange(6);
            var result = np.array(new long[] { 0 });  // Scalar output (1D of size 1)

            using var iter = NpyIterRef.AdvancedNew(
                2,
                new[] { a, result },
                NpyIterGlobalFlags.REDUCE_OK,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
                null,
                1,  // opAxesNDim = 1
                new[] { new[] { 0 }, new[] { -1 } });  // op_axes

            // Verify reduction is detected
            Assert.IsTrue(iter.IsReduction, "Should detect reduction");
            Assert.IsTrue(iter.IsOperandReduction(1), "Output operand should be marked as reduction");

            // Iterate and accumulate
            do
            {
                var x = iter.GetValue<long>(0);
                var y = iter.GetValue<long>(1);
                iter.SetValue(y + x, 1);
            } while (iter.Iternext());

            // Sum of 0+1+2+3+4+5 = 15
            Assert.AreEqual(15L, (long)result[0]);
        }

        [TestMethod]
        public void Reduction_2DToScalar_IteratesCorrectly()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(6).reshape(2, 3)
            // >>> it = np.nditer([a, None], ['reduce_ok', 'external_loop'],
            // ...                [['readonly'], ['readwrite', 'allocate']],
            // ...                op_axes=[[0, 1], [-1, -1]])
            // >>> it.operands[1][...] = 0
            // >>> for x, y in it:
            // ...     for j in range(len(y)):
            // ...         y[j] += x[j]
            // >>> int(it.operands[1])
            // 15

            var a = np.arange(6).reshape(2, 3);
            var result = np.array(new long[] { 0 });

            using var iter = NpyIterRef.AdvancedNew(
                2,
                new[] { a, result },
                NpyIterGlobalFlags.REDUCE_OK,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
                null,
                2,  // opAxesNDim = 2
                new[] { new[] { 0, 1 }, new[] { -1, -1 } });

            // Iterate and accumulate
            do
            {
                var x = iter.GetValue<long>(0);
                var y = iter.GetValue<long>(1);
                iter.SetValue(y + x, 1);
            } while (iter.Iternext());

            Assert.AreEqual(15L, (long)result[0]);
        }

        [TestMethod]
        public void Reduction_2DAlongAxis1_ProducesCorrectResult()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(6).reshape(2, 3)
            // >>> b = np.zeros(2, dtype=np.int64)
            // >>> it = np.nditer([a, b], ['reduce_ok'],
            // ...                [['readonly'], ['readwrite']],
            // ...                op_axes=[[0, 1], [0, -1]])
            // >>> for x, y in it:
            // ...     y[...] += x
            // >>> b
            // array([ 3, 12])  # Sum along axis 1: [0+1+2, 3+4+5]

            var a = np.arange(6).reshape(2, 3);
            var b = np.zeros(new Shape(2), NPTypeCode.Int64);

            using var iter = NpyIterRef.AdvancedNew(
                2,
                new[] { a, b },
                NpyIterGlobalFlags.REDUCE_OK,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
                null,
                2,
                new[] { new[] { 0, 1 }, new[] { 0, -1 } },  // axis 1 reduced
                new long[] { 2, 3 });  // Explicit iterShape needed when operands don't broadcast

            do
            {
                var x = iter.GetValue<long>(0);
                var y = iter.GetValue<long>(1);
                iter.SetValue(y + x, 1);
            } while (iter.Iternext());

            Assert.AreEqual(3L, (long)b[0], "Sum of row 0: 0+1+2=3");
            Assert.AreEqual(12L, (long)b[1], "Sum of row 1: 3+4+5=12");
        }

        [TestMethod]
        public void Reduction_IsFirstVisit_ReturnsTrueOnFirstElement()
        {
            // NumPy's IsFirstVisit() returns true when the current element of
            // a reduction operand is being visited for the first time.
            // This is used for initialization (e.g., set to 0 before summing).
            //
            // NumPy 2.4.2:
            // >>> a = np.arange(6).reshape(2, 3)
            // >>> b = np.zeros(2)
            // >>> it = np.nditer([a, b], ['reduce_ok', 'external_loop'],
            // ...                [['readonly'], ['readwrite']],
            // ...                op_axes=[[0, 1], [0, -1]])
            // >>> # At start, IsFirstVisit(1) is True for first row
            // >>> # After iterating past axis 1 values, IsFirstVisit(1) becomes False
            // >>> # When we move to row 1, IsFirstVisit(1) becomes True again

            var a = np.arange(6).reshape(2, 3);
            var b = np.zeros(new Shape(2), NPTypeCode.Int64);

            using var iter = NpyIterRef.AdvancedNew(
                2,
                new[] { a, b },
                NpyIterGlobalFlags.REDUCE_OK,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
                null,
                2,
                new[] { new[] { 0, 1 }, new[] { 0, -1 } },
                new long[] { 2, 3 });  // Explicit iterShape

            // First element (0,0): should be first visit to output[0]
            Assert.IsTrue(iter.IsFirstVisit(1), "First visit to output[0] at (0,0)");

            iter.Iternext();  // Move to (0,1)
            Assert.IsFalse(iter.IsFirstVisit(1), "Not first visit to output[0] at (0,1)");

            iter.Iternext();  // Move to (0,2)
            Assert.IsFalse(iter.IsFirstVisit(1), "Not first visit to output[0] at (0,2)");

            iter.Iternext();  // Move to (1,0) - first visit to output[1]
            Assert.IsTrue(iter.IsFirstVisit(1), "First visit to output[1] at (1,0)");

            iter.Iternext();  // Move to (1,1)
            Assert.IsFalse(iter.IsFirstVisit(1), "Not first visit to output[1] at (1,1)");
        }

        [TestMethod]
        public void Reduction_WithoutReduceOK_Throws()
        {
            // NumPy 2.4.2:
            // >>> a = np.arange(6)
            // >>> it = np.nditer([a, None], [],  # No reduce_ok
            // ...                [['readonly'], ['readwrite', 'allocate']],
            // ...                op_axes=[[0], [-1]])
            // ValueError: output operand requires a reduction along dimension 0,
            //             but the reduction is not enabled

            var a = np.arange(6);
            var result = np.array(new long[] { 0 });

            bool threw = false;
            try
            {
                using var iter = NpyIterRef.AdvancedNew(
                    2,
                    new[] { a, result },
                    NpyIterGlobalFlags.None,  // No REDUCE_OK
                    NPY_ORDER.NPY_KEEPORDER,
                    NPY_CASTING.NPY_NO_CASTING,
                    new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
                    null,
                    1,
                    new[] { new[] { 0 }, new[] { -1 } });
            }
            catch (ArgumentException)
            {
                threw = true;
            }

            Assert.IsTrue(threw, "Should throw when reduction detected but REDUCE_OK not set");
        }

        [TestMethod]
        public void Reduction_ReadOnlyOperand_DoesNotThrow()
        {
            // Reduction axes on READONLY operands should not require REDUCE_OK
            // because it's just broadcasting, not accumulation
            //
            // NumPy 2.4.2:
            // >>> a = np.arange(6).reshape(2, 3)
            // >>> scalar = np.array(10)
            // >>> it = np.nditer([a, scalar], [],  # No reduce_ok needed
            // ...                [['readonly'], ['readonly']],
            // ...                op_axes=[[0, 1], [-1, -1]])
            // >>> # Works fine - scalar is just broadcast

            var a = np.arange(6).reshape(2, 3);
            var scalar = np.array(new long[] { 10 });

            // Should not throw - readonly operand with stride 0 is just broadcasting
            using var iter = NpyIterRef.AdvancedNew(
                2,
                new[] { a, scalar },
                NpyIterGlobalFlags.None,  // No REDUCE_OK - should be fine for readonly
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY },
                null,
                2,
                new[] { new[] { 0, 1 }, new[] { -1, -1 } });

            // Verify scalar broadcasts correctly
            Assert.AreEqual(10L, iter.GetValue<long>(1));
            iter.Iternext();
            Assert.AreEqual(10L, iter.GetValue<long>(1));  // Same value due to stride 0
        }

        [TestMethod]
        public void Reduction_HasReduceFlag_WhenReductionDetected()
        {
            // The REDUCE flag should be set when reduction is detected

            var a = np.arange(6);
            var result = np.array(new long[] { 0 });

            using var iter = NpyIterRef.AdvancedNew(
                2,
                new[] { a, result },
                NpyIterGlobalFlags.REDUCE_OK,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
                null,
                1,
                new[] { new[] { 0 }, new[] { -1 } });

            Assert.IsTrue(iter.IsReduction, "REDUCE flag should be set");
            Assert.IsTrue(iter.IsOperandReduction(1), "Output operand should be marked as reduction");
            Assert.IsFalse(iter.IsOperandReduction(0), "Input operand should not be reduction");
        }
    }
}
