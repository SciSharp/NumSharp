using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest suite for NpyIter_GetAxisStrideArray (nditer_api.c:1309).
    ///
    /// Semantics:
    /// - HASMULTIINDEX: returns strides for user-supplied axis in original-array coords.
    /// - No MULTI_INDEX: returns strides in Fortran order (fastest-changing axis first).
    ///
    /// Strides are byte strides (NumPy convention). Verified against NumPy 2.4.2:
    ///   a = np.arange(6).reshape(2,3).astype(np.int32)  # strides (12, 4)
    ///   b = np.arange(24).reshape(2,3,4).astype(np.int32) # strides (48, 16, 4)
    /// </summary>
    [TestClass]
    public class NpyIterAxisStrideArrayTests
    {
        [TestMethod]
        public unsafe void AxisStride_2D_MultiIndex_AxisZero_OuterStride()
        {
            // For np.arange(6).reshape(2,3) int32: strides = (12, 4).
            // Axis 0 (outer) stride = 12.
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[1];
            it.GetAxisStrideArray(0, strides);
            Assert.AreEqual(12L, strides[0]);
        }

        [TestMethod]
        public unsafe void AxisStride_2D_MultiIndex_AxisOne_InnerStride()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[1];
            it.GetAxisStrideArray(1, strides);
            Assert.AreEqual(4L, strides[0]);
        }

        [TestMethod]
        public unsafe void AxisStride_3D_MultiIndex_AllAxes()
        {
            // np.arange(24).reshape(2,3,4) int32: strides = (48, 16, 4)
            var a = np.arange(24).reshape(2, 3, 4).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[1];
            it.GetAxisStrideArray(0, strides);
            Assert.AreEqual(48L, strides[0]);

            it.GetAxisStrideArray(1, strides);
            Assert.AreEqual(16L, strides[0]);

            it.GetAxisStrideArray(2, strides);
            Assert.AreEqual(4L, strides[0]);
        }

        [TestMethod]
        public unsafe void AxisStride_2D_NoMultiIndex_Coalesces_AxisZeroInnermost()
        {
            // Without MULTI_INDEX, a contiguous 2D array coalesces to 1D.
            // (NumPy behavior: coalescing removes dims that iterate identically.)
            // After coalescing, NDim=1 and axis 0 stride = 4 (innermost of original).
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a);

            Assert.AreEqual(1, it.NDim);  // Coalesced

            Span<long> strides = stackalloc long[1];
            it.GetAxisStrideArray(0, strides);
            Assert.AreEqual(4L, strides[0]);
        }

        [TestMethod]
        public unsafe void AxisStride_2D_NonContig_NoMultiIndex_FortranOrder()
        {
            // Non-contiguous 2D: [:, ::2] won't coalesce.
            // np.arange(12).reshape(3,4).astype(int32)[:, ::2] has shape (3,2), strides (16, 8)
            var a = np.arange(12).reshape(3, 4).astype(np.int32)[":, ::2"];
            using var it = NpyIterRef.New(a);

            Assert.AreEqual(2, it.NDim);  // Does NOT coalesce (stride gap)

            Span<long> strides = stackalloc long[1];
            // Axis 0 in Fortran order = fastest-changing (innermost) = stride 8
            it.GetAxisStrideArray(0, strides);
            Assert.AreEqual(8L, strides[0]);

            // Axis 1 in Fortran order = outer = stride 16
            it.GetAxisStrideArray(1, strides);
            Assert.AreEqual(16L, strides[0]);
        }

        [TestMethod]
        public unsafe void AxisStride_MultiOperand_PerOperandStrides()
        {
            var x = np.arange(6).reshape(2, 3).astype(np.int32);     // strides (12, 4)
            var y = np.arange(6).reshape(2, 3).astype(np.int64);     // strides (24, 8)

            using var it = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { x, y },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Span<long> strides = stackalloc long[2];

            it.GetAxisStrideArray(0, strides);
            Assert.AreEqual(12L, strides[0]);
            Assert.AreEqual(24L, strides[1]);

            it.GetAxisStrideArray(1, strides);
            Assert.AreEqual(4L, strides[0]);
            Assert.AreEqual(8L, strides[1]);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]
        public void AxisStride_OutOfBounds_Throws()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[1];
            it.GetAxisStrideArray(5, strides);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]
        public void AxisStride_Negative_Throws()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[1];
            it.GetAxisStrideArray(-1, strides);
        }

        [TestMethod]
        public unsafe void AxisStride_NegStride_ReversedAxis_AbsoluteValue()
        {
            // a[::-1] K-order → NEGPERM set, stride flipped from -4 to +4
            var a = np.arange(5).astype(np.int32)["::-1"];
            using var it = NpyIterRef.New(a,
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER);

            Assert.IsTrue(it.HasNegPerm);

            Span<long> strides = stackalloc long[1];
            it.GetAxisStrideArray(0, strides);
            // After flip, stride is positive 4
            Assert.AreEqual(4L, strides[0]);
        }

        [TestMethod]
        public unsafe void AxisStride_Broadcast_StrideZero()
        {
            // Broadcast axis has stride 0 (no data advance)
            var a = np.arange(3).astype(np.int32);
            var b = np.arange(6).reshape(2, 3).astype(np.int32);

            using var it = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Span<long> strides = stackalloc long[2];

            // Axis 0: b strides by 12, a has no axis 0 → stride 0 (broadcast)
            it.GetAxisStrideArray(0, strides);
            Assert.AreEqual(0L, strides[0]);   // a is broadcast on axis 0
            Assert.AreEqual(12L, strides[1]);  // b

            // Axis 1: both stride 4
            it.GetAxisStrideArray(1, strides);
            Assert.AreEqual(4L, strides[0]);
            Assert.AreEqual(4L, strides[1]);
        }

        [TestMethod]
        public unsafe void AxisStride_1D_MultiIndex_SingleAxis()
        {
            var a = np.arange(10).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[1];
            it.GetAxisStrideArray(0, strides);
            Assert.AreEqual(4L, strides[0]);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void AxisStride_TooShortSpan_Throws()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[0];
            it.GetAxisStrideArray(0, strides);
        }
    }
}
