using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest suite for NpyIter_GetInnerFixedStrideArray (nditer_api.c:1357).
    ///
    /// Semantics:
    /// - Buffered: returns <see cref="NpyIterState.BufStrides"/> (per-operand buffer strides).
    /// - Non-buffered: returns the innermost-axis stride per operand.
    ///
    /// Stride values verified against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class NpyIterInnerFixedStrideArrayTests
    {
        [TestMethod]
        public unsafe void InnerFixed_1D_Int32_Contiguous_StrideIs4()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(a);

            Span<long> strides = stackalloc long[1];
            it.GetInnerFixedStrideArray(strides);
            Assert.AreEqual(4L, strides[0]);
        }

        [TestMethod]
        public unsafe void InnerFixed_1D_Int64_Contiguous_StrideIs8()
        {
            var a = np.arange(5).astype(np.int64);
            using var it = NpyIterRef.New(a);

            Span<long> strides = stackalloc long[1];
            it.GetInnerFixedStrideArray(strides);
            Assert.AreEqual(8L, strides[0]);
        }

        [TestMethod]
        public unsafe void InnerFixed_2D_Int32_InnermostIs4()
        {
            // np.arange(6).reshape(2,3) has strides (12, 4). Innermost = 4.
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a);

            Span<long> strides = stackalloc long[1];
            it.GetInnerFixedStrideArray(strides);
            Assert.AreEqual(4L, strides[0]);
        }

        [TestMethod]
        public unsafe void InnerFixed_1D_Strided_MatchesStride()
        {
            // a[::2] int32 has stride=8
            var a = np.arange(20).astype(np.int32)["::2"];
            using var it = NpyIterRef.New(a, order: NPY_ORDER.NPY_KEEPORDER);

            Span<long> strides = stackalloc long[1];
            it.GetInnerFixedStrideArray(strides);
            Assert.AreEqual(8L, strides[0]);
        }

        [TestMethod]
        public unsafe void InnerFixed_MultiOperand_PerOperandStrides()
        {
            var x = np.arange(5).astype(np.int32);      // stride 4
            var y = np.arange(5).astype(np.int64);       // stride 8

            using var it = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { x, y },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Span<long> strides = stackalloc long[2];
            it.GetInnerFixedStrideArray(strides);
            Assert.AreEqual(4L, strides[0]);
            Assert.AreEqual(8L, strides[1]);
        }

        [TestMethod]
        public unsafe void InnerFixed_Buffered_ReturnsBufStrides()
        {
            // With BUFFERED and cast, buffer stride = element size of target dtype
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { a },
                flags: NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                opDtypes: new[] { NPTypeCode.Double });

            Span<long> strides = stackalloc long[1];
            it.GetInnerFixedStrideArray(strides);
            // Buffer stride = dtypesize of target (double = 8)
            Assert.AreEqual(8L, strides[0]);
        }

        [TestMethod]
        public unsafe void InnerFixed_Broadcast_StrideIsZero()
        {
            // Broadcast axis has stride=0 (outer repeats, innermost varies)
            var a = np.arange(3).astype(np.int32);
            var b = np.arange(6).reshape(2, 3).astype(np.int32);

            using var it = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Span<long> strides = stackalloc long[2];
            it.GetInnerFixedStrideArray(strides);
            // Innermost axis (size 3): both a and b iterate along it with stride 4
            Assert.AreEqual(4L, strides[0]);
            Assert.AreEqual(4L, strides[1]);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void InnerFixed_TooShortSpan_Throws()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.MultiNew(
                nop: 1,
                op: new[] { a },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY });

            Span<long> strides = stackalloc long[0];
            it.GetInnerFixedStrideArray(strides);
        }

        [TestMethod]
        public unsafe void InnerFixed_NegStride_ReversedFlipped()
        {
            // a[::-1] int32 with K-order should flip negative stride
            // After flip: stride = 4 (was -4), memory iteration
            var a = np.arange(5).astype(np.int32)["::-1"];
            using var it = NpyIterRef.New(a, order: NPY_ORDER.NPY_KEEPORDER);

            Span<long> strides = stackalloc long[1];
            it.GetInnerFixedStrideArray(strides);
            // After flip, inner stride is positive 4
            Assert.AreEqual(4L, strides[0]);
        }
    }
}
