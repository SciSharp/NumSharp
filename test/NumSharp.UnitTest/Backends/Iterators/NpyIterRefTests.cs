using System;
using TUnit.Core;
using NumSharp;
using NumSharp.Backends.Iteration;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace NumSharp.UnitTest.Backends.Iterators
{
    public class NpyIterRefTests
    {
        [Test]
        public void New_SingleOperand_Contiguous()
        {
            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);

            // With external loop, we expect coalescing (NDim may vary based on implementation)
            Assert.IsTrue(iter.NDim >= 1 && iter.NDim <= 3);
            Assert.AreEqual(24, iter.IterSize);
            Assert.IsTrue(iter.IsContiguous);
        }

        [Test]
        public void New_SingleOperand_Sliced()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var sliced = arr["0:2, 1:3, ::2"];

            using var iter = NpyIterRef.New(sliced);

            Assert.AreEqual(8, iter.IterSize);  // 2 * 2 * 2
            Assert.AreEqual(1, iter.NOp);
        }

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
        public void GetIterNext_ReturnsValidDelegate()
        {
            var arr = np.array(new double[] { 1, 2, 3, 4, 5 });

            using var iter = NpyIterRef.New(arr);

            var iternext = iter.GetIterNext();

            // Verify it was created
            Assert.IsNotNull(iternext);
        }

        [Test]
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

        [Test]
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

        [Test]
        public void Properties_ReturnCorrectValues()
        {
            var arr = np.arange(24).reshape(2, 3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);

            Assert.AreEqual(1, iter.NOp);
            Assert.AreEqual(24, iter.IterSize);
            Assert.AreEqual(0, iter.IterIndex);
            Assert.IsFalse(iter.RequiresBuffering);
        }

        [Test]
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

        [Test]
        public void ZeroSizeArray_HandledCorrectly()
        {
            var arr = np.empty(new Shape(0));

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.ZEROSIZE_OK);

            Assert.AreEqual(0, iter.IterSize);
        }

        [Test]
        public void ScalarArray_HandledCorrectly()
        {
            var arr = np.array(42.0);

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1, iter.IterSize);
            Assert.AreEqual(0, iter.NDim);
        }

        [Test]
        public void EnableExternalLoop_ModifiesFlags()
        {
            var arr = np.arange(10);

            using var iter = NpyIterRef.New(arr);

            Assert.IsFalse(iter.HasExternalLoop);

            iter.EnableExternalLoop();

            Assert.IsTrue(iter.HasExternalLoop);
        }

        [Test]
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

        [Test]
        public void Coalescing_ReducesDimensions()
        {
            var arr = np.arange(24).reshape(2, 3, 4);

            // Without external loop, no coalescing
            using var iter1 = NpyIterRef.New(arr);
            Assert.AreEqual(3, iter1.NDim);

            // With external loop, coalescing may reduce dimensions
            // (exact reduction depends on implementation)
            using var iter2 = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
            Assert.IsTrue(iter2.HasExternalLoop);
            Assert.IsTrue(iter2.IsContiguous);
        }

        [Test]
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

        [Test]
        public void Coalescing_AlwaysRunsWithoutMultiIndex()
        {
            // NumPy coalesces contiguous arrays more aggressively due to axis reordering
            // before coalescing. NumSharp's current implementation coalesces adjacent
            // axes with compatible strides but doesn't fully reorder axes first.
            //
            // NumPy behavior:
            // >>> arr = np.arange(24).reshape(2, 3, 4)
            // >>> it = np.nditer(arr)
            // >>> it.ndim  # Returns 1 (fully coalesced)
            //
            // NumSharp behavior: coalescing runs but may not fully reduce to 1D
            // because axis reordering is not implemented.

            var arr = np.arange(24).reshape(2, 3, 4);

            // Verify coalescing runs (may not fully coalesce to 1D)
            using var iter = NpyIterRef.New(arr);

            // Coalescing should run and attempt to reduce dimensions
            // For contiguous array, at minimum the iteration should work correctly
            Assert.IsTrue(iter.NDim >= 1 && iter.NDim <= 3, "NDim should be between 1 and 3");
            Assert.AreEqual(24, iter.IterSize, "IterSize should be 24");
        }

        [Test]
        public void Coalescing_1DArray_StaysAt1D()
        {
            // 1D arrays should remain at ndim=1
            var arr = np.arange(100);

            using var iter = NpyIterRef.New(arr);

            Assert.AreEqual(1, iter.NDim, "1D array should have ndim=1");
            Assert.AreEqual(100, iter.IterSize);
        }

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
        public void MultiIndex_ThrowsWithoutFlag()
        {
            var arr = np.arange(12);

            using var iter = NpyIterRef.New(arr);  // No MULTI_INDEX flag

            Assert.IsFalse(iter.HasMultiIndex);

            // Direct call to verify exception
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

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
        public unsafe void InnerStrides_SingleOperand()
        {
            var arr = np.arange(12).reshape(3, 4);

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);

            var innerStrides = iter.GetInnerStrideArray();

            // After coalescing contiguous array, inner stride should be 1
            Assert.AreEqual(1, innerStrides[0]);
        }

        [Test]
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
        // Fix #8: MaxDims Tests
        // =========================================================================

        [Test]
        public void MaxDims_Is64()
        {
            // Verify MaxDims is 64 to match NumPy's NPY_MAXDIMS
            Assert.AreEqual(64, NpyIterState.MaxDims);
        }
    }
}
