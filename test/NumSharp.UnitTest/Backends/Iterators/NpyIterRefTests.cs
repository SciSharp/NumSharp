using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    [TestClass]
    public class NpyIterRefTests
    {
        [TestMethod]
        public void New_SingleOperand_Contiguous()
        {
            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);

            // Contiguous arrays fully coalesce to ndim=1 (NumPy parity)
            Assert.AreEqual(1, iter.NDim, "Contiguous array should coalesce to ndim=1");
            Assert.AreEqual(24, iter.IterSize);
            Assert.IsTrue(iter.IsContiguous);
        }

        [TestMethod]
        public void New_SingleOperand_Sliced()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var sliced = arr["0:2, 1:3, ::2"];

            using var iter = NpyIterRef.New(sliced);

            Assert.AreEqual(8, iter.IterSize);  // 2 * 2 * 2
            Assert.AreEqual(1, iter.NOp);
        }

        [TestMethod]
        public void MultiNew_TwoOperands_SameShape()
        {
            var a = np.arange(12).reshape(3, 4);
            var b = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(12, iter.IterSize);
            Assert.AreEqual(2, iter.NOp);
        }

        [TestMethod]
        public void MultiNew_TwoOperands_Broadcasting()
        {
            var a = np.arange(12).reshape(3, 4);
            var b = np.arange(4);  // Will broadcast to (3, 4)

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(12, iter.IterSize);
            Assert.AreEqual(2, iter.NDim);
        }

        [TestMethod]
        public void MultiNew_ThreeOperands_OutputArray()
        {
            var a = np.arange(12).reshape(3, 4);
            var b = np.arange(4);
            var c = np.empty((3, 4));

            using var iter = NpyIterRef.MultiNew(
                nop: 3,
                op: new[] { a, b, c },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY
                });

            Assert.AreEqual(12, iter.IterSize);
            Assert.AreEqual(3, iter.NOp);
        }

        [TestMethod]
        public void GetIterNext_ReturnsValidDelegate()
        {
            var arr = np.array(new double[] { 1, 2, 3, 4, 5 });

            using var iter = NpyIterRef.New(arr);

            var iternext = iter.GetIterNext();

            // Verify it was created
            Assert.IsNotNull(iternext);
        }

        [TestMethod]
        public void Reset_ResetsIteration()
        {
            var arr = np.arange(10);

            using var iter = NpyIterRef.New(arr);

            // Move forward
            iter.GotoIterIndex(5);
            Assert.AreEqual(5, iter.IterIndex);

            // Reset
            iter.Reset();
            Assert.AreEqual(0, iter.IterIndex);
        }

        [TestMethod]
        public void GotoIterIndex_JumpsToPosition()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            iter.GotoIterIndex(42);
            Assert.AreEqual(42, iter.IterIndex);

            iter.GotoIterIndex(99);
            Assert.AreEqual(99, iter.IterIndex);

            iter.GotoIterIndex(0);
            Assert.AreEqual(0, iter.IterIndex);
        }

        [TestMethod]
        public void Properties_ReturnCorrectValues()
        {
            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);

            Assert.AreEqual(1, iter.NOp);
            Assert.AreEqual(24, iter.IterSize);
            Assert.AreEqual(0, iter.IterIndex);
            Assert.IsFalse(iter.RequiresBuffering);
        }

        [TestMethod]
        public void GetDescrArray_ReturnsCorrectDtypes()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.array(new double[] { 1.0, 2.0, 3.0 });

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            var dtypes = iter.GetDescrArray();

            Assert.AreEqual(2, dtypes.Length);
            Assert.AreEqual(NPTypeCode.Int32, dtypes[0]);
            Assert.AreEqual(NPTypeCode.Double, dtypes[1]);
        }

        [TestMethod]
        public void ZeroSizeArray_HandledCorrectly()
        {
            var arr = np.empty(new Shape(0));

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.ZEROSIZE_OK);

            Assert.AreEqual(0, iter.IterSize);
        }

        [TestMethod]
        public void ScalarArray_HandledCorrectly()
        {
            var arr = np.array(42.0);

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1, iter.IterSize);
            Assert.AreEqual(0, iter.NDim);
        }

        [TestMethod]
        public void EnableExternalLoop_ModifiesFlags()
        {
            var arr = np.arange(10);

            using var iter = NpyIterRef.New(arr);

            Assert.IsFalse(iter.HasExternalLoop);

            iter.EnableExternalLoop();

            Assert.IsTrue(iter.HasExternalLoop);
        }

        [TestMethod]
        public void AdvancedNew_WithBuffering()
        {
            var arr = np.arange(1000);

            using var iter = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { arr },
                flags: NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                bufferSize: 256);

            Assert.IsTrue(iter.RequiresBuffering);
            Assert.AreEqual(1000, iter.IterSize);
        }

        [TestMethod]
        public void Coalescing_ReducesDimensions()
        {
            var arr = np.arange(24).reshape(2, 3, 4);

            // Coalescing always runs (unless MULTI_INDEX is set)
            // Contiguous arrays fully coalesce to 1D
            using var iter1 = NpyIterRef.New(arr);
            Assert.AreEqual(1, iter1.NDim, "Contiguous array should coalesce to ndim=1");

            // With external loop, same behavior (coalescing already ran)
            using var iter2 = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
            Assert.IsTrue(iter2.HasExternalLoop);
            Assert.IsTrue(iter2.IsContiguous);
            Assert.AreEqual(1, iter2.NDim, "With EXTERNAL_LOOP, still ndim=1");
        }

        [TestMethod]
        public void BroadcastError_ThrowsException()
        {
            var a = np.arange(12).reshape(3, 4);
            var b = np.arange(5);  // Cannot broadcast (5,) to (3, 4)

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

        // =========================================================================
        // Fix #1: Coalescing Always Runs Tests
        // =========================================================================

        [TestMethod]
        public void Coalescing_AlwaysRunsWithoutMultiIndex()
        {
            // NumPy behavior: contiguous arrays fully coalesce to ndim=1
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(arr)
            // >>> it.ndim  # Returns 1 (fully coalesced)
            //
            // NumSharp now matches this behavior by:
            // 1. Reordering axes by stride (smallest first) before coalescing
            // 2. Then coalescing adjacent axes with compatible strides
            //
            // For C-contiguous (2,3,4) with strides [12,4,1]:
            // - Reorder to [4,3,2] with strides [1,4,12]
            // - Coalesce: 1*4=4==4 ✓ → [12,2] with strides [1,12]
            // - Coalesce: 1*12=12==12 ✓ → [24] with strides [1]

            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr);

            // Contiguous array should fully coalesce to 1D (NumPy parity)
            Assert.AreEqual(1, iter.NDim, "Contiguous array should coalesce to ndim=1 (NumPy behavior)");
            Assert.AreEqual(24, iter.IterSize, "IterSize should be 24");
        }

        [TestMethod]
        public void Coalescing_1DArray_StaysAt1D()
        {
            // 1D arrays should remain at ndim=1
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1, iter.NDim, "1D array should have ndim=1");
            Assert.AreEqual(100, iter.IterSize);
        }

        [TestMethod]
        public void Coalescing_DisabledWithMultiIndex()
        {
            // NumPy behavior: MULTI_INDEX prevents coalescing
            // >>> it = np.nditer(arr, flags=['multi_index'])
            // >>> it.ndim
            // 3

            var arr = np.arange(24).reshape(2, 3, 4);

            // With MULTI_INDEX flag, should NOT coalesce
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            // Original dimensions preserved
            Assert.AreEqual(3, iter.NDim, "MULTI_INDEX should prevent coalescing");
            Assert.IsTrue(iter.HasMultiIndex);
        }

        [TestMethod]
        public void Coalescing_PartialForStridedArrays()
        {
            // Non-contiguous arrays may partially coalesce
            var arr = np.arange(24).reshape(2, 3, 4);
            var transposed = arr.T;  // (4, 3, 2) with non-contiguous strides

            using var iter = NpyIterRef.New(transposed);

            // After coalescing, dimensions may reduce but typically not to 1 for transposed
            Assert.IsTrue(iter.NDim >= 1 && iter.NDim <= 3);
            Assert.AreEqual(24, iter.IterSize);
        }

        // =========================================================================
        // Fix #4: Multi-Index Support Tests
        // =========================================================================

        [TestMethod]
        public void MultiIndex_GetCoordinates()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            Assert.IsTrue(iter.HasMultiIndex);

            var coords = new long[iter.NDim];
            iter.GetMultiIndex(coords);

            // At start, coordinates should be (0, 0)
            Assert.AreEqual(0, coords[0]);
            Assert.AreEqual(0, coords[1]);
        }

        [TestMethod]
        public void MultiIndex_GotoPosition()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.MULTI_INDEX);

            // Jump to position (1, 2) - element at index 6
            iter.GotoMultiIndex(new long[] { 1, 2 });

            var coords = new long[iter.NDim];
            iter.GetMultiIndex(coords);

            Assert.AreEqual(1, coords[0]);
            Assert.AreEqual(2, coords[1]);
        }

        [TestMethod]
        public void MultiIndex_ThrowsWithoutFlag()
        {
            var arr = np.arange(12);

            using var iter = NpyIterRef.New(arr);  // No MULTI_INDEX flag

            Assert.IsFalse(iter.HasMultiIndex);

            // Direct call to verify exception (can't use lambda with ref struct)
            bool threwException = false;
            try
            {
                var coords = new long[1];
                iter.GetMultiIndex(coords);
            }
            catch (InvalidOperationException)
            {
                threwException = true;
            }
            Assert.IsTrue(threwException, "Should throw InvalidOperationException when MULTI_INDEX flag not set");
        }

        // =========================================================================
        // Fix #5: Ranged Iteration Tests
        // =========================================================================

        [TestMethod]
        public void RangedIteration_ValidRange()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            // Set up ranged iteration for elements 20-50
            var success = iter.ResetToIterIndexRange(20, 50);

            Assert.IsTrue(success);
            Assert.IsTrue(iter.IsRanged);
            Assert.AreEqual(20, iter.IterStart);
            Assert.AreEqual(50, iter.IterEnd);
            Assert.AreEqual(20, iter.IterIndex);
        }

        [TestMethod]
        public void RangedIteration_InvalidRange()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            // Invalid: end > size
            Assert.IsFalse(iter.ResetToIterIndexRange(0, 200));

            // Invalid: start > end
            Assert.IsFalse(iter.ResetToIterIndexRange(50, 20));

            // Invalid: start < 0
            Assert.IsFalse(iter.ResetToIterIndexRange(-10, 50));
        }

        [TestMethod]
        public void RangedIteration_FullRange()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            // Full range is valid
            var success = iter.ResetToIterIndexRange(0, 100);

            Assert.IsTrue(success);
            Assert.AreEqual(0, iter.IterStart);
            Assert.AreEqual(100, iter.IterEnd);
        }

        // =========================================================================
        // Fix #2: Inner Stride Array Tests
        // =========================================================================

        [TestMethod]
        public unsafe void InnerStrides_SingleOperand()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);

            var innerStrides = iter.GetInnerStrideArray();

            // After coalescing contiguous array, inner stride should be 1
            Assert.AreEqual(1, innerStrides[0]);
        }

        [TestMethod]
        public unsafe void InnerStrides_MultipleOperands()
        {
            var a = np.arange(12).reshape(3, 4);
            var b = np.arange(4);  // Will broadcast

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            var innerStrides = iter.GetInnerStrideArray();

            // Contiguous array should have stride 1
            // Broadcast array may have stride 0 or 1 depending on axis
            Assert.IsTrue(innerStrides != null, "InnerStrides should not be null");
        }

        // =========================================================================
        // NumSharp Divergence: Unlimited Dimensions Tests
        // =========================================================================

        [TestMethod]
        public void UnlimitedDimensions_HighDimensionalArray()
        {
            // NUMSHARP DIVERGENCE: Unlike NumPy's NPY_MAXDIMS=64 limit,
            // NumSharp supports unlimited dimensions via dynamic allocation.
            // Practical limit is around 300,000 dimensions (stackalloc limit).
            //
            // This test verifies high-dimensional arrays work correctly.

            // Create a 20-dimensional array (well beyond typical use cases)
            var shape = new int[20];
            for (int i = 0; i < 20; i++)
                shape[i] = 2;

            var arr = np.ones(new Shape(shape));

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1048576, iter.IterSize);  // 2^20 = 1048576
            Assert.IsTrue(iter.NDim >= 1);  // May coalesce
        }

        [TestMethod]
        public void UnlimitedDimensions_MaxOperands()
        {
            // MaxOperands is still 8 (reasonable limit for multi-operand iteration)
            Assert.AreEqual(8, NpyIterState.MaxOperands);
        }

        // =========================================================================
        // C_INDEX and F_INDEX Tests (Flat Index Tracking)
        // =========================================================================

        [TestMethod]
        public void CIndex_TracksLinearPosition()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.C_INDEX | NpyIterGlobalFlags.MULTI_INDEX);

            Assert.IsTrue(iter.HasIndex);
            Assert.AreEqual(0, iter.GetIndex());

            // Move to position (1, 2) = element at linear index 6
            iter.GotoMultiIndex(new long[] { 1, 2 });
            Assert.AreEqual(6, iter.GetIndex());
        }

        [TestMethod]
        public void CIndex_AdvanceIncrementsIndex()
        {
            var arr = np.arange(10);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.C_INDEX);

            Assert.AreEqual(0, iter.GetIndex());

            // Advance a few times using GotoIterIndex (Advance is internal)
            iter.GotoIterIndex(5);
            Assert.AreEqual(5, iter.GetIndex());
        }

        [TestMethod]
        public void FIndex_TracksColumnMajorPosition()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.F_INDEX | NpyIterGlobalFlags.MULTI_INDEX);

            Assert.IsTrue(iter.HasIndex);
            Assert.AreEqual(0, iter.GetIndex());

            // F-order position (1, 2): column-major index is 1 + 2*3 = 7
            iter.GotoMultiIndex(new long[] { 1, 2 });
            Assert.AreEqual(7, iter.GetIndex());
        }

        [TestMethod]
        public void Index_ThrowsWithoutFlag()
        {
            var arr = np.arange(10);

            using var iter = NpyIterRef.New(arr);  // No C_INDEX/F_INDEX flag

            Assert.IsFalse(iter.HasIndex);

            // Should throw when trying to get index
            bool threwException = false;
            try
            {
                iter.GetIndex();
            }
            catch (InvalidOperationException)
            {
                threwException = true;
            }
            Assert.IsTrue(threwException);
        }

        [TestMethod]
        public void Index_ResetToZero()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.C_INDEX);

            iter.GotoIterIndex(50);
            Assert.AreEqual(50, iter.GetIndex());

            iter.Reset();
            Assert.AreEqual(0, iter.GetIndex());
        }

        // =========================================================================
        // GROWINNER Optimization Tests
        // =========================================================================

        [TestMethod]
        public void GrowInner_FlagSetsCorrectly()
        {
            var arr = np.arange(1000);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.GROWINNER);

            Assert.IsTrue(iter.HasGrowInner);
        }

        [TestMethod]
        public void GrowInner_WithBuffering()
        {
            var arr = np.arange(1000);

            using var iter = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { arr },
                flags: NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.GROWINNER,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                bufferSize: 256);

            Assert.IsTrue(iter.RequiresBuffering);
            Assert.IsTrue(iter.HasGrowInner);
        }

        // =========================================================================
        // iterShape Parameter Tests
        // =========================================================================

        [TestMethod]
        public void IterShape_ExplicitShape()
        {
            // When iterShape is specified, it overrides the broadcast shape
            var arr = np.arange(4);  // Shape (4,)

            using var iter = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { arr },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                iterShape: new long[] { 3, 4 });  // Explicit 2D iteration

            Assert.AreEqual(12, iter.IterSize);  // 3 * 4
        }

        [TestMethod]
        public void IterShape_IncompatibleThrows()
        {
            var arr = np.arange(5);  // Shape (5,)

            // iterShape (3, 4) requires inner dim of 4 or 1, not 5
            Assert.ThrowsException<IncorrectShapeException>(() =>
            {
                using var iter = NpyIterRef.AdvancedNew(
                    nop: 1,
                    op: new[] { arr },
                    flags: NpyIterGlobalFlags.None,
                    order: NPY_ORDER.NPY_KEEPORDER,
                    casting: NPY_CASTING.NPY_SAFE_CASTING,
                    opFlags: new[] { NpyIterPerOpFlags.READONLY },
                    iterShape: new long[] { 3, 4 });
            });
        }

        // =========================================================================
        // Buffer Reuse Tests
        // =========================================================================

        [TestMethod]
        public void BufferReuse_InvalidatedOnReset()
        {
            // Buffer reuse flags should be invalidated when iterator is reset
            var arr = np.arange(100);

            using var iter = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { arr },
                flags: NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                bufferSize: 32);

            // After Reset, buffers should be invalidated
            iter.Reset();
            // No direct way to check BUF_REUSABLE flag from outside,
            // but the reset should not throw
            Assert.AreEqual(0, iter.IterIndex);
        }

        [TestMethod]
        public void BufferReuse_InvalidatedOnGoto()
        {
            var arr = np.arange(100);

            using var iter = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { arr },
                flags: NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                bufferSize: 32);

            // GotoIterIndex should invalidate buffers
            iter.GotoIterIndex(50);
            Assert.AreEqual(50, iter.IterIndex);
        }
    }
}
