using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest suite for NpyIter_CreateCompatibleStrides (nditer_api.c:1058).
    ///
    /// Semantics: Builds contiguous strides matching the iterator's axis ordering.
    /// Use case: match the shape of an iterator while tacking on extra dimensions.
    ///
    /// Requires HASMULTIINDEX and no flipped axes.
    /// Expected values verified against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class NpyIterCreateCompatibleStridesTests
    {
        [TestMethod]
        public unsafe void CreateCompatibleStrides_1D_Int32_ItemSize4()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[1];
            Assert.IsTrue(it.CreateCompatibleStrides(4, strides));
            Assert.AreEqual(4L, strides[0]);
        }

        [TestMethod]
        public unsafe void CreateCompatibleStrides_2D_Int32_ReturnsContiguous()
        {
            // For (2,3) shape, C-order strides with itemsize=4: [12, 4]
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[2];
            Assert.IsTrue(it.CreateCompatibleStrides(4, strides));
            Assert.AreEqual(12L, strides[0]);
            Assert.AreEqual(4L, strides[1]);
        }

        [TestMethod]
        public unsafe void CreateCompatibleStrides_3D_Int64_ReturnsContiguous()
        {
            // For (2,3,4) int64 with itemsize=8: [96, 32, 8]
            var a = np.arange(24).reshape(2, 3, 4).astype(np.int64);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[3];
            Assert.IsTrue(it.CreateCompatibleStrides(8, strides));
            Assert.AreEqual(96L, strides[0]);
            Assert.AreEqual(32L, strides[1]);
            Assert.AreEqual(8L, strides[2]);
        }

        [TestMethod]
        public unsafe void CreateCompatibleStrides_ItemSize8_OnInt32_Compatible()
        {
            // Use case: tack on dimension. For (2,3) with itemsize=8 (e.g., 2 floats per elem):
            // Accumulator: idim=1 (inner=axis 1) → [_, 8], itemsize *= 3 = 24
            //              idim=0 (axis 0)       → [24, 8]
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[2];
            Assert.IsTrue(it.CreateCompatibleStrides(8, strides));
            Assert.AreEqual(24L, strides[0]);
            Assert.AreEqual(8L, strides[1]);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void CreateCompatibleStrides_WithoutMultiIndex_Throws()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a); // No MULTI_INDEX

            Span<long> strides = stackalloc long[2];
            it.CreateCompatibleStrides(4, strides);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void CreateCompatibleStrides_WithFlippedAxis_Throws()
        {
            // Reversed array under K-order triggers NEGPERM. Should fail.
            var a = np.arange(5).astype(np.int32)["::-1"];
            using var it = NpyIterRef.New(a,
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER);

            Assert.IsTrue(it.HasNegPerm);

            Span<long> strides = stackalloc long[1];
            it.CreateCompatibleStrides(4, strides);
        }

        [TestMethod]
        public unsafe void CreateCompatibleStrides_WithDontNegateStrides_Succeeds()
        {
            // With DONT_NEGATE_STRIDES flag, negative strides remain — no NEGPERM.
            // Should succeed.
            var a = np.arange(5).astype(np.int32)["::-1"];
            using var it = NpyIterRef.New(a,
                flags: NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.DONT_NEGATE_STRIDES,
                order: NPY_ORDER.NPY_KEEPORDER);

            Assert.IsFalse(it.HasNegPerm, "DONT_NEGATE_STRIDES should prevent flip");

            Span<long> strides = stackalloc long[1];
            Assert.IsTrue(it.CreateCompatibleStrides(4, strides));
            Assert.AreEqual(4L, strides[0]);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void CreateCompatibleStrides_TooShortSpan_Throws()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[1]; // Too short
            it.CreateCompatibleStrides(4, strides);
        }

        [TestMethod]
        public unsafe void CreateCompatibleStrides_ProducesUsableLayout()
        {
            // Strides from CreateCompatibleStrides are in BYTES (NumPy convention).
            // For shape (3,4) int32: byte strides should be (16, 4) — matching
            // a freshly-allocated C-contiguous array of same shape.
            var a = np.arange(12).reshape(3, 4).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            Span<long> strides = stackalloc long[2];
            it.CreateCompatibleStrides(4, strides);

            // Expected C-contiguous byte strides: shape=(3,4), elemsize=4 → (16, 4)
            Assert.AreEqual(16L, strides[0]);
            Assert.AreEqual(4L, strides[1]);
        }
    }
}
