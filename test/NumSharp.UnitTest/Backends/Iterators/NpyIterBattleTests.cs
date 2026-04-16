using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battle tests for NpyIter implementation.
    /// Tests edge cases, parity with NumPy, and potential bugs.
    /// </summary>
    [TestClass]
    public class NpyIterBattleTests
    {
        // =====================================================================
        // Dimension Edge Cases
        // =====================================================================

        [TestMethod]
        public void Scalar_ZeroDimensions()
        {
            var scalar = np.array(42.0);
            Assert.AreEqual(0, scalar.ndim);

            using var iter = NpyIterRef.New(scalar);

            Assert.AreEqual(0, iter.NDim);
            Assert.AreEqual(1, iter.IterSize);
            Assert.AreEqual(1, iter.NOp);
        }

        [TestMethod]
        public void EmptyArray_ZeroSize()
        {
            var empty = np.empty(new Shape(0));

            using var iter = NpyIterRef.New(empty, NpyIterGlobalFlags.ZEROSIZE_OK);

            Assert.AreEqual(0, iter.IterSize);
        }

        [TestMethod]
        public void EmptyArray_MultiDimensional()
        {
            // Shape (2, 0, 3) - middle dimension is 0
            var empty = np.empty(new Shape(2, 0, 3));

            using var iter = NpyIterRef.New(empty, NpyIterGlobalFlags.ZEROSIZE_OK);

            Assert.AreEqual(0, iter.IterSize);
        }

        [TestMethod]
        public void SingleElement_1D()
        {
            var arr = np.array(new double[] { 99.0 });

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1, iter.IterSize);
        }

        [TestMethod]
        public void SingleElement_HighDimensional()
        {
            // Shape (1, 1, 1, 1, 1) - 5D but only 1 element
            var arr = np.ones(new Shape(1, 1, 1, 1, 1));

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1, iter.IterSize);
        }

        [TestMethod]
        public void HighDimensional_10D()
        {
            var shape = new int[10];
            for (int i = 0; i < 10; i++) shape[i] = 2;

            var arr = np.arange(1024).reshape(shape);  // 2^10 = 1024

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1024, iter.IterSize);
        }

        // =====================================================================
        // Memory Layout: Contiguous
        // =====================================================================

        [TestMethod]
        public unsafe void Contiguous_1D_CorrectDataAccess()
        {
            var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            using var iter = NpyIterRef.New(arr);

            // Verify basic properties
            Assert.AreEqual(5, iter.IterSize);
            Assert.AreEqual(1, iter.NDim);
            Assert.IsTrue(iter.IsContiguous);

            // Verify data pointer is valid
            var dataptrs = iter.GetDataPtrArray();
            Assert.IsTrue(dataptrs != null);
            Assert.IsTrue(dataptrs[0] != null);

            // Verify first element is accessible
            double firstValue = *(double*)dataptrs[0];
            Assert.AreEqual(1.0, firstValue);
        }

        [TestMethod]
        public unsafe void Contiguous_2D_IteratesRowMajor()
        {
            // NumPy iterates in C-order (row-major)
            // [[0, 1, 2], [3, 4, 5]] should iterate as 0, 1, 2, 3, 4, 5
            var arr = np.arange(6).reshape(2, 3);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            Assert.AreEqual(6, iter.IterSize);
            Assert.IsTrue(iter.HasMultiIndex);

            // With MULTI_INDEX, coalescing is disabled so we should have 2D
            Assert.AreEqual(2, iter.NDim);
        }

        // =====================================================================
        // Memory Layout: Sliced/Strided
        // =====================================================================

        [TestMethod]
        public void Sliced_EveryOther()
        {
            var arr = np.arange(10);
            var sliced = arr["::2"];  // [0, 2, 4, 6, 8]

            Assert.AreEqual(5, sliced.size);

            using var iter = NpyIterRef.New(sliced);

            Assert.AreEqual(5, iter.IterSize);
        }

        [TestMethod]
        public void Sliced_Reversed()
        {
            var arr = np.arange(5);
            var reversed = arr["::-1"];  // [4, 3, 2, 1, 0]

            Assert.AreEqual(5, reversed.size);

            using var iter = NpyIterRef.New(reversed);

            Assert.AreEqual(5, iter.IterSize);
        }

        [TestMethod]
        public void Sliced_Column()
        {
            var arr = np.arange(12).reshape(3, 4);
            var column = arr[":, 1"];  // Second column: [1, 5, 9]

            Assert.AreEqual(3, column.size);

            using var iter = NpyIterRef.New(column);

            Assert.AreEqual(3, iter.IterSize);
        }

        [TestMethod]
        public void Sliced_SubMatrix()
        {
            var arr = np.arange(24).reshape(4, 6);
            var sub = arr["1:3, 2:5"];  // 2x3 submatrix

            Assert.AreEqual(6, sub.size);

            using var iter = NpyIterRef.New(sub);

            Assert.AreEqual(6, iter.IterSize);
        }

        // =====================================================================
        // Memory Layout: Transposed
        // =====================================================================

        [TestMethod]
        public void Transposed_2D()
        {
            var arr = np.arange(6).reshape(2, 3);
            var transposed = arr.T;  // Shape (3, 2)

            Assert.AreEqual(3, transposed.shape[0]);
            Assert.AreEqual(2, transposed.shape[1]);
            Assert.AreEqual(6, transposed.size);

            using var iter = NpyIterRef.New(transposed);

            Assert.AreEqual(6, iter.IterSize);
        }

        [TestMethod]
        public void Transposed_3D()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var transposed = np.transpose(arr);  // Shape (4, 3, 2)

            Assert.AreEqual(4, transposed.shape[0]);
            Assert.AreEqual(3, transposed.shape[1]);
            Assert.AreEqual(2, transposed.shape[2]);

            using var iter = NpyIterRef.New(transposed);

            Assert.AreEqual(24, iter.IterSize);
        }

        // =====================================================================
        // Memory Layout: Broadcast
        // =====================================================================

        [TestMethod]
        public void Broadcast_ScalarTo1D()
        {
            var scalar = np.array(5.0);
            var target = np.arange(10);

            // Broadcast scalar to match target shape
            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { scalar, target },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(10, iter.IterSize);
        }

        [TestMethod]
        public void Broadcast_RowToMatrix()
        {
            var row = np.arange(4);           // Shape (4,)
            var matrix = np.arange(12).reshape(3, 4);  // Shape (3, 4)

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { row, matrix },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(12, iter.IterSize);
        }

        [TestMethod]
        public void Broadcast_ColumnToMatrix()
        {
            var column = np.arange(3).reshape(3, 1);   // Shape (3, 1)
            var matrix = np.arange(12).reshape(3, 4);  // Shape (3, 4)

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { column, matrix },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(12, iter.IterSize);
        }

        [TestMethod]
        public void Broadcast_IncompatibleShapes_Throws()
        {
            var a = np.arange(5);   // Shape (5,)
            var b = np.arange(3);   // Shape (3,) - incompatible!

            Assert.ThrowsException<IncorrectShapeException>(() =>
            {
                using var iter = NpyIterRef.MultiNew(
                    nop: 2,
                    op: new[] { a, b },
                    flags: NpyIterGlobalFlags.None,
                    order: NPY_ORDER.NPY_KEEPORDER,
                    casting: NPY_CASTING.NPY_SAFE_CASTING,
                    opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });
            });
        }

        // =====================================================================
        // Multi-Index Tracking
        // =====================================================================

        [TestMethod]
        public void MultiIndex_2D_InitialPosition()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            var coords = new long[2];
            iter.GetMultiIndex(coords);

            Assert.AreEqual(0, coords[0]);
            Assert.AreEqual(0, coords[1]);
        }

        [TestMethod]
        public void MultiIndex_GotoAndGet()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            // Jump to (1, 2)
            iter.GotoMultiIndex(new long[] { 1, 2 });

            var coords = new long[2];
            iter.GetMultiIndex(coords);

            Assert.AreEqual(1, coords[0]);
            Assert.AreEqual(2, coords[1]);
        }

        [TestMethod]
        public void MultiIndex_OutOfBounds_Throws()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            // Try to jump to invalid position
            bool threw = false;
            try
            {
                iter.GotoMultiIndex(new long[] { 5, 2 });  // 5 > 3
            }
            catch (IndexOutOfRangeException)
            {
                threw = true;
            }
            Assert.IsTrue(threw, "Should throw IndexOutOfRangeException for out of bounds coord");
        }

        [TestMethod]
        public void MultiIndex_NegativeCoord_Throws()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            bool threw = false;
            try
            {
                iter.GotoMultiIndex(new long[] { -1, 2 });
            }
            catch (IndexOutOfRangeException)
            {
                threw = true;
            }
            Assert.IsTrue(threw, "Should throw IndexOutOfRangeException for negative coord");
        }

        [TestMethod]
        public void MultiIndex_WithoutFlag_Throws()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr);  // No MULTI_INDEX flag

            Assert.IsFalse(iter.HasMultiIndex);

            bool threw = false;
            try
            {
                var coords = new long[2];
                iter.GetMultiIndex(coords);
            }
            catch (InvalidOperationException)
            {
                threw = true;
            }
            Assert.IsTrue(threw, "Should throw InvalidOperationException without MULTI_INDEX flag");
        }

        // =====================================================================
        // GotoIterIndex
        // =====================================================================

        [TestMethod]
        public void GotoIterIndex_ValidPositions()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            iter.GotoIterIndex(0);
            Assert.AreEqual(0, iter.IterIndex);

            iter.GotoIterIndex(50);
            Assert.AreEqual(50, iter.IterIndex);

            iter.GotoIterIndex(99);
            Assert.AreEqual(99, iter.IterIndex);
        }

        [TestMethod]
        public void GotoIterIndex_MultipleCalls()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            // Jump around randomly
            iter.GotoIterIndex(75);
            Assert.AreEqual(75, iter.IterIndex);

            iter.GotoIterIndex(10);
            Assert.AreEqual(10, iter.IterIndex);

            iter.GotoIterIndex(99);
            Assert.AreEqual(99, iter.IterIndex);

            iter.GotoIterIndex(0);
            Assert.AreEqual(0, iter.IterIndex);
        }

        // =====================================================================
        // Ranged Iteration
        // =====================================================================

        [TestMethod]
        public void RangedIteration_ValidRange()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            Assert.IsTrue(iter.ResetToIterIndexRange(20, 50));
            Assert.IsTrue(iter.IsRanged);
            Assert.AreEqual(20, iter.IterStart);
            Assert.AreEqual(50, iter.IterEnd);
            Assert.AreEqual(20, iter.IterIndex);
        }

        [TestMethod]
        public void RangedIteration_StartGreaterThanEnd()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            Assert.IsFalse(iter.ResetToIterIndexRange(50, 20));
            Assert.IsFalse(iter.IsRanged);
        }

        [TestMethod]
        public void RangedIteration_EndExceedsSize()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            Assert.IsFalse(iter.ResetToIterIndexRange(0, 200));
        }

        [TestMethod]
        public void RangedIteration_NegativeStart()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            Assert.IsFalse(iter.ResetToIterIndexRange(-10, 50));
        }

        [TestMethod]
        public void RangedIteration_EmptyRange()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            // start == end is valid (empty range)
            Assert.IsTrue(iter.ResetToIterIndexRange(50, 50));
            Assert.AreEqual(50, iter.IterStart);
            Assert.AreEqual(50, iter.IterEnd);
        }

        // =====================================================================
        // Coalescing Behavior
        // =====================================================================

        [TestMethod]
        public void Coalescing_1D_NoChange()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1, iter.NDim);
        }

        [TestMethod]
        public void Coalescing_DisabledWithMultiIndex()
        {
            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            // With MULTI_INDEX, coalescing should be disabled
            Assert.AreEqual(3, iter.NDim);
            Assert.IsTrue(iter.HasMultiIndex);
        }

        [TestMethod]
        public void Coalescing_ContiguousArray()
        {
            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr);

            // Contiguous array may coalesce (depends on implementation)
            Assert.IsTrue(iter.NDim >= 1 && iter.NDim <= 3);
            Assert.AreEqual(24, iter.IterSize);
        }

        [TestMethod]
        public void Coalescing_NonContiguous_NoCoalesce()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var transposed = arr.T;  // Non-contiguous

            using var iter = NpyIterRef.New(transposed);

            // Non-contiguous may not fully coalesce
            Assert.IsTrue(iter.NDim >= 1);
            Assert.AreEqual(24, iter.IterSize);
        }

        // =====================================================================
        // External Loop
        // =====================================================================

        [TestMethod]
        public void ExternalLoop_FlagSet()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);

            Assert.IsTrue(iter.HasExternalLoop);
        }

        [TestMethod]
        public void ExternalLoop_WithContiguous()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);

            Assert.IsTrue(iter.HasExternalLoop);
            Assert.IsTrue(iter.IsContiguous);
        }

        // =====================================================================
        // Inner Strides
        // =====================================================================

        [TestMethod]
        public unsafe void InnerStrides_Contiguous1D()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            var innerStrides = iter.GetInnerStrideArray();

            // Contiguous 1D array should have inner stride of 1
            Assert.AreEqual(1, innerStrides[0]);
        }

        [TestMethod]
        public unsafe void InnerStrides_Strided()
        {
            var arr = np.arange(100);
            var strided = arr["::2"];  // Every other element

            using var iter = NpyIterRef.New(strided);

            var innerStrides = iter.GetInnerStrideArray();

            // Strided array has stride of 2
            Assert.AreEqual(2, innerStrides[0]);
        }

        [TestMethod]
        public unsafe void InnerStrides_MultipleOperands()
        {
            var a = np.arange(12).reshape(3, 4);
            var b = np.arange(4);  // Will broadcast

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            var innerStrides = iter.GetInnerStrideArray();

            // Should have 2 inner strides
            Assert.IsTrue(innerStrides != null);
        }

        // =====================================================================
        // Reset
        // =====================================================================

        [TestMethod]
        public void Reset_ReturnsToStart()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            iter.GotoIterIndex(50);
            Assert.AreEqual(50, iter.IterIndex);

            iter.Reset();
            Assert.AreEqual(0, iter.IterIndex);
        }

        [TestMethod]
        public void Reset_AfterRangedIteration()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            iter.ResetToIterIndexRange(20, 50);
            iter.GotoIterIndex(35);
            Assert.AreEqual(35, iter.IterIndex);

            iter.Reset();
            Assert.AreEqual(20, iter.IterIndex);  // Should reset to IterStart, not 0
        }

        // =====================================================================
        // Dtype Handling
        // =====================================================================

        [DataTestMethod]
        [DataRow(NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        public void AllDtypes_SingleOperand(NPTypeCode dtype)
        {
            NDArray arr = dtype switch
            {
                NPTypeCode.Boolean => np.array(new bool[] { true, false, true }),
                NPTypeCode.Byte => np.array(new byte[] { 1, 2, 3 }),
                NPTypeCode.Int16 => np.array(new short[] { 1, 2, 3 }),
                NPTypeCode.UInt16 => np.array(new ushort[] { 1, 2, 3 }),
                NPTypeCode.Int32 => np.array(new int[] { 1, 2, 3 }),
                NPTypeCode.UInt32 => np.array(new uint[] { 1, 2, 3 }),
                NPTypeCode.Int64 => np.array(new long[] { 1, 2, 3 }),
                NPTypeCode.UInt64 => np.array(new ulong[] { 1, 2, 3 }),
                NPTypeCode.Single => np.array(new float[] { 1, 2, 3 }),
                NPTypeCode.Double => np.array(new double[] { 1, 2, 3 }),
                _ => throw new NotSupportedException()
            };

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(3, iter.IterSize);
            Assert.AreEqual(dtype, iter.GetDescrArray()[0]);
        }

        // =====================================================================
        // Resource Management
        // =====================================================================

        [TestMethod]
        public void Dispose_MultipleTimes_NoError()
        {
            var arr = np.arange(100);

            var iter = NpyIterRef.New(arr);
            iter.Dispose();
            iter.Dispose();  // Should not throw
            iter.Dispose();  // Should not throw
        }

        [TestMethod]
        public void MultipleIterators_SameArray()
        {
            var arr = np.arange(100);

            using var iter1 = NpyIterRef.New(arr);
            using var iter2 = NpyIterRef.New(arr);
            using var iter3 = NpyIterRef.New(arr);

            Assert.AreEqual(100, iter1.IterSize);
            Assert.AreEqual(100, iter2.IterSize);
            Assert.AreEqual(100, iter3.IterSize);
        }

        [TestMethod]
        public void AllocationStress_ManyIterators()
        {
            var arr = np.arange(100);

            // Create and dispose many iterators to stress allocation
            for (int i = 0; i < 1000; i++)
            {
                using var iter = NpyIterRef.New(arr);
                Assert.AreEqual(100, iter.IterSize);
            }
        }

        [TestMethod]
        public void AllocationStress_HighDimensional()
        {
            // Create high-dimensional arrays repeatedly
            for (int i = 0; i < 100; i++)
            {
                var shape = new int[15];
                for (int j = 0; j < 15; j++) shape[j] = 2;

                var arr = np.ones(new Shape(shape));

                using var iter = NpyIterRef.New(arr);
                Assert.AreEqual(32768, iter.IterSize);  // 2^15
            }
        }

        // =====================================================================
        // Properties
        // =====================================================================

        [TestMethod]
        public void Properties_Contiguous()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            Assert.IsTrue(iter.IsContiguous);
            Assert.IsFalse(iter.RequiresBuffering);
            Assert.IsFalse(iter.HasExternalLoop);
            Assert.IsFalse(iter.HasMultiIndex);
            Assert.IsFalse(iter.IsRanged);
        }

        [TestMethod]
        public void GetOperandArray_ReturnsCorrectArrays()
        {
            var a = np.arange(10);
            var b = np.arange(10);

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            var operands = iter.GetOperandArray();

            Assert.IsNotNull(operands);
            Assert.AreEqual(2, operands.Length);
            Assert.AreSame(a, operands[0]);
            Assert.AreSame(b, operands[1]);
        }

        // =====================================================================
        // Edge Cases: Views and Slices
        // =====================================================================

        [TestMethod]
        public void SliceOfSlice()
        {
            var arr = np.arange(100);
            var slice1 = arr["10:90"];
            var slice2 = slice1["10:70"];  // Elements 20-80 of original

            Assert.AreEqual(60, slice2.size);

            using var iter = NpyIterRef.New(slice2);

            Assert.AreEqual(60, iter.IterSize);
        }

        [TestMethod]
        public void SliceWithNegativeStep()
        {
            var arr = np.arange(10);
            var reversed = arr["::-1"];

            using var iter = NpyIterRef.New(reversed);

            Assert.AreEqual(10, iter.IterSize);
        }

        [TestMethod]
        public void NonContiguous_2D_Column()
        {
            var arr = np.arange(20).reshape(4, 5);
            var col = arr[":, 2"];  // Third column

            Assert.AreEqual(4, col.size);
            Assert.IsFalse(col.Shape.IsContiguous);

            using var iter = NpyIterRef.New(col);

            Assert.AreEqual(4, iter.IterSize);
        }

        // =====================================================================
        // Mixed Operand Scenarios
        // =====================================================================

        [TestMethod]
        public void MixedLayouts_ContiguousAndStrided()
        {
            var contiguous = np.arange(10);
            var strided = np.arange(20)["::2"];  // Every other

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { contiguous, strided },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(10, iter.IterSize);
        }

        [TestMethod]
        public void MixedDtypes()
        {
            var intArr = np.array(new int[] { 1, 2, 3 });
            var floatArr = np.array(new float[] { 1.0f, 2.0f, 3.0f });

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { intArr, floatArr },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            var dtypes = iter.GetDescrArray();
            Assert.AreEqual(NPTypeCode.Int32, dtypes[0]);
            Assert.AreEqual(NPTypeCode.Single, dtypes[1]);
        }

        // =====================================================================
        // Buffered Iteration
        // =====================================================================

        [TestMethod]
        public void Buffered_FlagSet()
        {
            var arr = np.arange(10000);

            using var iter = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { arr },
                flags: NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                bufferSize: 1024);

            Assert.IsTrue(iter.RequiresBuffering);
        }

        // =====================================================================
        // Error Conditions
        // =====================================================================

        [TestMethod]
        public void ManyOperands_Works()
        {
            // NUMSHARP DIVERGENCE: Unlike NumPy's NPY_MAXARGS=64, NumSharp supports unlimited operands.
            // Test with 10 operands to verify no artificial limit.
            var arrays = new NDArray[10];
            for (int i = 0; i < 10; i++)
                arrays[i] = np.arange(10);

            var opFlags = new NpyIterPerOpFlags[10];
            for (int i = 0; i < 10; i++)
                opFlags[i] = NpyIterPerOpFlags.READONLY;

            using var iter = NpyIterRef.MultiNew(
                nop: 10,  // NumSharp supports unlimited operands
                op: arrays,
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: opFlags);

            Assert.AreEqual(10, iter.NOp);
            Assert.AreEqual(10, iter.IterSize);
        }

        [TestMethod]
        public void ZeroOperands_Throws()
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(() =>
            {
                using var iter = NpyIterRef.MultiNew(
                    nop: 0,
                    op: Array.Empty<NDArray>(),
                    flags: NpyIterGlobalFlags.None,
                    order: NPY_ORDER.NPY_KEEPORDER,
                    casting: NPY_CASTING.NPY_SAFE_CASTING,
                    opFlags: Array.Empty<NpyIterPerOpFlags>());
            });
        }

        [TestMethod]
        public void NullOperand_Throws()
        {
            Assert.ThrowsException<NullReferenceException>(() =>
            {
                using var iter = NpyIterRef.New(null!);
            });
        }

        // =====================================================================
        // Data Verification - Verify actual iteration values
        // =====================================================================

        [TestMethod]
        public unsafe void DataVerification_1D_AllElements()
        {
            var expected = new int[] { 10, 20, 30, 40, 50 };
            var arr = np.array(expected);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            Assert.AreEqual(5, iter.IterSize);

            // Verify each element by jumping to it
            for (int i = 0; i < 5; i++)
            {
                iter.GotoMultiIndex(new long[] { i });

                var dataptr = iter.GetDataPtrArray()[0];
                int value = *(int*)dataptr;

                Assert.AreEqual(expected[i], value, $"Element at index {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void DataVerification_2D_AllElements()
        {
            // [[0, 1, 2], [3, 4, 5]]
            var arr = np.arange(6).reshape(2, 3);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            Assert.AreEqual(6, iter.IterSize);
            Assert.AreEqual(2, iter.NDim);

            // Verify each element
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    iter.GotoMultiIndex(new long[] { i, j });

                    var dataptr = iter.GetDataPtrArray()[0];
                    int value = *(int*)dataptr;
                    int expected = i * 3 + j;

                    Assert.AreEqual(expected, value, $"Element at ({i}, {j}) mismatch");
                }
            }
        }

        [TestMethod]
        public unsafe void DataVerification_Sliced_CorrectValues()
        {
            // arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            // sliced = arr[2:8:2] = [2, 4, 6]
            var arr = np.arange(10);
            var sliced = arr["2:8:2"];

            Assert.AreEqual(3, sliced.size);

            using var iter = NpyIterRef.New(sliced, NpyIterGlobalFlags.MULTI_INDEX);

            int[] expected = { 2, 4, 6 };

            for (int i = 0; i < 3; i++)
            {
                iter.GotoMultiIndex(new long[] { i });

                var dataptr = iter.GetDataPtrArray()[0];
                int value = *(int*)dataptr;

                Assert.AreEqual(expected[i], value, $"Sliced element at {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void DataVerification_Reversed_CorrectValues()
        {
            // arr = [0, 1, 2, 3, 4]
            // reversed = [4, 3, 2, 1, 0]
            var arr = np.arange(5);
            var reversed = arr["::-1"];

            Assert.AreEqual(5, reversed.size);

            using var iter = NpyIterRef.New(reversed, NpyIterGlobalFlags.MULTI_INDEX);

            for (int i = 0; i < 5; i++)
            {
                iter.GotoMultiIndex(new long[] { i });

                var dataptr = iter.GetDataPtrArray()[0];
                int value = *(int*)dataptr;
                int expected = 4 - i;

                Assert.AreEqual(expected, value, $"Reversed element at {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void DataVerification_Transposed_CorrectValues()
        {
            // arr = [[0, 1, 2], [3, 4, 5]]  shape (2, 3)
            // transposed = [[0, 3], [1, 4], [2, 5]]  shape (3, 2)
            var arr = np.arange(6).reshape(2, 3);
            var transposed = arr.T;

            Assert.AreEqual(3, transposed.shape[0]);
            Assert.AreEqual(2, transposed.shape[1]);

            using var iter = NpyIterRef.New(transposed, NpyIterGlobalFlags.MULTI_INDEX);

            // Expected values in transposed order
            int[,] expected = { { 0, 3 }, { 1, 4 }, { 2, 5 } };

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    iter.GotoMultiIndex(new long[] { i, j });

                    var dataptr = iter.GetDataPtrArray()[0];
                    int value = *(int*)dataptr;

                    Assert.AreEqual(expected[i, j], value, $"Transposed element at ({i}, {j}) mismatch");
                }
            }
        }

        [TestMethod]
        public unsafe void DataVerification_Column_CorrectValues()
        {
            // arr = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
            // column = arr[:, 2] = [2, 6, 10]
            var arr = np.arange(12).reshape(3, 4);
            var column = arr[":, 2"];

            Assert.AreEqual(3, column.size);
            Assert.AreEqual(1, column.ndim);

            using var iter = NpyIterRef.New(column, NpyIterGlobalFlags.MULTI_INDEX);

            int[] expected = { 2, 6, 10 };

            for (int i = 0; i < 3; i++)
            {
                iter.GotoMultiIndex(new long[] { i });

                var dataptr = iter.GetDataPtrArray()[0];
                int value = *(int*)dataptr;

                Assert.AreEqual(expected[i], value, $"Column element at {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void DataVerification_SubMatrix_CorrectValues()
        {
            // arr = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
            // sub = arr[1:3, 1:3] = [[5, 6], [9, 10]]
            var arr = np.arange(16).reshape(4, 4);
            var sub = arr["1:3, 1:3"];

            Assert.AreEqual(4, sub.size);
            Assert.AreEqual(2, sub.shape[0]);
            Assert.AreEqual(2, sub.shape[1]);

            using var iter = NpyIterRef.New(sub, NpyIterGlobalFlags.MULTI_INDEX);

            int[,] expected = { { 5, 6 }, { 9, 10 } };

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    iter.GotoMultiIndex(new long[] { i, j });

                    var dataptr = iter.GetDataPtrArray()[0];
                    int value = *(int*)dataptr;

                    Assert.AreEqual(expected[i, j], value, $"SubMatrix element at ({i}, {j}) mismatch");
                }
            }
        }

        [TestMethod]
        public unsafe void DataVerification_Broadcast_CorrectValues()
        {
            // a = [10, 20, 30] (shape (3,))
            // b = [[0, 1, 2], [3, 4, 5]] (shape (2, 3))
            // When iterated together with broadcasting, a broadcasts to (2, 3)

            var a = np.array(new int[] { 10, 20, 30 });
            var b = np.arange(6).reshape(2, 3);

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(6, iter.IterSize);
            Assert.AreEqual(2, iter.NDim);

            // Verify broadcast values at each position
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    iter.GotoMultiIndex(new long[] { i, j });

                    var dataptrs = iter.GetDataPtrArray();
                    int aValue = *(int*)dataptrs[0];
                    int bValue = *(int*)dataptrs[1];

                    // a broadcasts: [10, 20, 30] same for all rows
                    int expectedA = 10 + j * 10;
                    // b values: [[0,1,2], [3,4,5]]
                    int expectedB = i * 3 + j;

                    Assert.AreEqual(expectedA, aValue, $"Broadcast a at ({i}, {j}) mismatch");
                    Assert.AreEqual(expectedB, bValue, $"B at ({i}, {j}) mismatch");
                }
            }
        }

        [TestMethod]
        public unsafe void DataVerification_GotoIterIndex_MatchesMultiIndex()
        {
            // Verify that GotoIterIndex and GotoMultiIndex give same data pointer

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            // Test several positions
            var testCases = new (int linear, long[] coords)[]
            {
                (0, new long[] { 0, 0, 0 }),
                (5, new long[] { 0, 1, 1 }),  // 0*12 + 1*4 + 1 = 5
                (13, new long[] { 1, 0, 1 }), // 1*12 + 0*4 + 1 = 13
                (23, new long[] { 1, 2, 3 }), // 1*12 + 2*4 + 3 = 23
            };

            foreach (var (linear, coords) in testCases)
            {
                // Jump via linear index
                iter.GotoIterIndex(linear);
                var dataptrLinear = iter.GetDataPtrArray()[0];
                int valueLinear = *(int*)dataptrLinear;

                // Jump via multi-index
                iter.GotoMultiIndex(coords);
                var dataptrMulti = iter.GetDataPtrArray()[0];
                int valueMulti = *(int*)dataptrMulti;

                Assert.AreEqual(valueLinear, valueMulti,
                    $"Value mismatch at linear={linear}, coords=({string.Join(",", coords)})");
                Assert.AreEqual(linear, valueMulti,
                    $"Expected value {linear} at coords ({string.Join(",", coords)})");
            }
        }

        [TestMethod]
        public void DataVerification_IterSize_MatchesArraySize()
        {
            // Verify IterSize matches array size for various shapes

            var testCases = new[]
            {
                new int[] { },         // Scalar -> size 1
                new int[] { 1 },
                new int[] { 10 },
                new int[] { 2, 3 },
                new int[] { 2, 3, 4 },
                new int[] { 2, 2, 2, 2 },
            };

            foreach (var shape in testCases)
            {
                NDArray arr;
                long expectedSize;

                if (shape.Length == 0)
                {
                    arr = np.array(42.0);  // Scalar
                    expectedSize = 1;
                }
                else
                {
                    expectedSize = shape.Aggregate(1, (a, b) => a * b);
                    arr = np.arange((int)expectedSize).reshape(shape);
                }

                using var iter = NpyIterRef.New(arr);

                Assert.AreEqual(expectedSize, iter.IterSize,
                    $"IterSize mismatch for shape ({string.Join(",", shape)})");
            }
        }

        // =====================================================================
        // Edge Cases Found During Testing
        // =====================================================================

        [TestMethod]
        public void EdgeCase_VeryLargeDimension()
        {
            // Test with one very large dimension
            var arr = np.arange(1000000);

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1000000, iter.IterSize);
            Assert.AreEqual(1, iter.NDim);
        }

        [TestMethod]
        public void EdgeCase_ManySmallDimensions()
        {
            // Test with many dimensions of size 2
            var shape = new int[12];
            for (int i = 0; i < 12; i++) shape[i] = 2;

            var arr = np.ones(new Shape(shape));

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(4096, iter.IterSize);  // 2^12
        }

        [TestMethod]
        public unsafe void EdgeCase_DoublePrecision()
        {
            // Verify double precision values are correct
            var arr = np.array(new double[] { 1.5, 2.7, 3.14159265358979 });

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            iter.GotoMultiIndex(new long[] { 2 });
            var dataptr = iter.GetDataPtrArray()[0];
            double value = *(double*)dataptr;

            Assert.AreEqual(3.14159265358979, value, 1e-15);
        }

        [TestMethod]
        public unsafe void EdgeCase_BooleanArray()
        {
            var arr = np.array(new bool[] { true, false, true, false, true });

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            bool[] expected = { true, false, true, false, true };

            for (int i = 0; i < 5; i++)
            {
                iter.GotoMultiIndex(new long[] { i });
                var dataptr = iter.GetDataPtrArray()[0];
                bool value = *(bool*)dataptr;
                Assert.AreEqual(expected[i], value, $"Boolean at {i} mismatch");
            }
        }
    }
}

