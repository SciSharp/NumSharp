using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Tests for <see cref="NDIter.CreateCopyState"/> with emphasis on the
    /// "no broadcast needed" fast path added in S4 (skip np.broadcast_to when
    /// src and dst have identical dimensions).
    ///
    /// Each test exercises NDIter.Copy end-to-end and verifies the resulting
    /// data matches the equivalent NumPy operation. Correctness is the primary
    /// concern — the fast path must produce byte-identical results to the
    /// general path.
    /// </summary>
    [TestClass]
    public unsafe class NDIterCreateCopyStateTests
    {
        // =====================================================================
        // Section 1 — Fast path (shapes match exactly): same-dtype copies.
        // =====================================================================

        [TestMethod]
        public void FastPath_1D_SameShape_SameDtype()
        {
            var src = np.arange(1000).astype(NPTypeCode.Int32);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(1000), fillZeros: false);

            NDIter.Copy(dst, src);

            // dst should equal src element-by-element.
            for (int i = 0; i < 1000; i++)
                ((int)dst[i]).Should().Be((int)src[i]);
        }

        [TestMethod]
        public void FastPath_2D_SameShape_SameDtype()
        {
            var src = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(3, 4), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ((int)dst[i, j]).Should().Be((int)src[i, j]);
        }

        [TestMethod]
        public void FastPath_3D_SameShape_SameDtype()
        {
            var src = np.arange(24).reshape(2, 3, 4).astype(NPTypeCode.Double);
            var dst = new NDArray(NPTypeCode.Double, new Shape(2, 3, 4), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        ((double)dst[i, j, k]).Should().Be((double)src[i, j, k]);
        }

        [TestMethod]
        public void FastPath_5D_SameShape_SameDtype()
        {
            // Higher-rank test to stress the loop in ShapesMatchExactly.
            var src = np.arange(120).reshape(2, 3, 2, 5, 2).astype(NPTypeCode.Single);
            var dst = new NDArray(NPTypeCode.Single, new Shape(2, 3, 2, 5, 2), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 120; i++)
                ((float)dst.flat[i]).Should().Be((float)src.flat[i]);
        }

        [TestMethod]
        public void FastPath_Empty_SameShape_SameDtype()
        {
            // Empty arrays (size=0) — fast path should still work.
            var src = new NDArray(NPTypeCode.Int32, new Shape(0, 5), fillZeros: false);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(0, 5), fillZeros: false);

            // Should not throw.
            NDIter.Copy(dst, src);
            dst.size.Should().Be(0);
        }

        [TestMethod]
        public void FastPath_SingleElement_SameShape_SameDtype()
        {
            var src = np.array(new[] { 42 }).reshape(1);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(1), fillZeros: false);

            NDIter.Copy(dst, src);

            ((int)dst[0]).Should().Be(42);
        }

        [TestMethod]
        public void FastPath_AllDtypes_1D_SameShape()
        {
            // Cover every NumSharp dtype via the same-shape fast path.
            // Use small arrays (10 elements) to keep the test snappy.
            VerifySameDtypeCopy<byte>(NPTypeCode.Byte);
            VerifySameDtypeCopy<sbyte>(NPTypeCode.SByte);
            VerifySameDtypeCopy<short>(NPTypeCode.Int16);
            VerifySameDtypeCopy<ushort>(NPTypeCode.UInt16);
            VerifySameDtypeCopy<int>(NPTypeCode.Int32);
            VerifySameDtypeCopy<uint>(NPTypeCode.UInt32);
            VerifySameDtypeCopy<long>(NPTypeCode.Int64);
            VerifySameDtypeCopy<ulong>(NPTypeCode.UInt64);
            VerifySameDtypeCopy<float>(NPTypeCode.Single);
            VerifySameDtypeCopy<double>(NPTypeCode.Double);
            VerifySameDtypeCopy<bool>(NPTypeCode.Boolean);
            VerifySameDtypeCopy<char>(NPTypeCode.Char);
            VerifySameDtypeCopy<decimal>(NPTypeCode.Decimal);
            VerifySameDtypeCopy<Half>(NPTypeCode.Half);
            VerifySameDtypeCopy<System.Numerics.Complex>(NPTypeCode.Complex);
        }

        private static void VerifySameDtypeCopy<T>(NPTypeCode tc) where T : unmanaged
        {
            var src = np.arange(10).astype(tc);
            var dst = new NDArray(tc, new Shape(10), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 10; i++)
                dst[i].ToString().Should().Be(src[i].ToString(),
                    because: $"dtype={tc} index={i}");
        }

        // =====================================================================
        // Section 2 — Fast path (shapes match): cross-dtype copies.
        // =====================================================================

        [TestMethod]
        public void FastPath_1D_SameShape_Int32_To_Double()
        {
            var src = np.arange(100).astype(NPTypeCode.Int32);
            var dst = new NDArray(NPTypeCode.Double, new Shape(100), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 100; i++)
                ((double)dst[i]).Should().Be((double)i);
        }

        [TestMethod]
        public void FastPath_1D_SameShape_Float32_To_Double()
        {
            var src = np.arange(50).astype(NPTypeCode.Single);
            var dst = new NDArray(NPTypeCode.Double, new Shape(50), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 50; i++)
                ((double)dst[i]).Should().Be((double)i);
        }

        [TestMethod]
        public void FastPath_2D_SameShape_Int64_To_Int32_Narrowing()
        {
            var src = np.arange(20).reshape(4, 5).astype(NPTypeCode.Int64);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(4, 5), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 5; j++)
                    ((int)dst[i, j]).Should().Be((int)(long)src[i, j]);
        }

        // =====================================================================
        // Section 3 — Fast path with non-trivial layouts on src.
        // =====================================================================

        [TestMethod]
        public void FastPath_SameShape_CContig_Src_To_FContig_Dst()
        {
            // src is C-contig, dst is F-contig — same dims, different layouts.
            // The fast path should still apply (ShapesMatchExactly ignores strides)
            // and the copy should respect each side's strides.
            var src = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(new long[] { 3, 4 }, 'F'), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ((int)dst[i, j]).Should().Be((int)src[i, j]);
        }

        [TestMethod]
        public void FastPath_SameShape_SlicedSrc()
        {
            // src is a sliced view (offset != 0) but same dims as dst.
            var full = np.arange(20).astype(NPTypeCode.Int32);
            var src = full["5:15"];   // shape (10,) with offset 5
            var dst = new NDArray(NPTypeCode.Int32, new Shape(10), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 10; i++)
                ((int)dst[i]).Should().Be(i + 5);
        }

        [TestMethod]
        public void FastPath_SameShape_TransposedSrc()
        {
            // src.T has same dims as a freshly built (4,3) dst, but different strides.
            var src = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32).T; // shape (4,3)
            var dst = new NDArray(NPTypeCode.Int32, new Shape(4, 3), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    ((int)dst[i, j]).Should().Be((int)src[i, j]);
        }

        [TestMethod]
        public void FastPath_SameShape_NegativeStrideSrc()
        {
            // Reversed slice — same dims as dst but stride is negative.
            var full = np.arange(10).astype(NPTypeCode.Int32);
            var src = full["::-1"];   // shape (10,) reversed
            var dst = new NDArray(NPTypeCode.Int32, new Shape(10), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 10; i++)
                ((int)dst[i]).Should().Be(9 - i);
        }

        // =====================================================================
        // Section 4 — Slow path: broadcast required (shapes differ).
        //              These must still route through np.broadcast_to.
        // =====================================================================

        [TestMethod]
        public void SlowPath_Broadcast_Scalar_To_1D()
        {
            // src is a scalar (1-elem 1-D), dst is (N,). Must broadcast.
            var src = np.array(new[] { 7 });   // shape (1,)
            var dst = new NDArray(NPTypeCode.Int32, new Shape(10), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 10; i++)
                ((int)dst[i]).Should().Be(7);
        }

        [TestMethod]
        public void SlowPath_Broadcast_RowVector_To_2D()
        {
            // src=(1,4), dst=(3,4). Row repeated.
            var src = np.arange(4).reshape(1, 4).astype(NPTypeCode.Int32);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(3, 4), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ((int)dst[i, j]).Should().Be(j);
        }

        [TestMethod]
        public void SlowPath_Broadcast_ColVector_To_2D()
        {
            // src=(3,1), dst=(3,4). Column repeated.
            var src = np.arange(3).reshape(3, 1).astype(NPTypeCode.Int32);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(3, 4), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ((int)dst[i, j]).Should().Be(i);
        }

        [TestMethod]
        public void SlowPath_Broadcast_NDimMismatch()
        {
            // src=(4,), dst=(3,4). src promoted to (1,4), then stretched.
            var src = np.arange(4).astype(NPTypeCode.Int32);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(3, 4), fillZeros: false);

            NDIter.Copy(dst, src);

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ((int)dst[i, j]).Should().Be(j);
        }

        [TestMethod]
        public void SlowPath_Broadcast_Incompatible_Throws()
        {
            // src=(5,), dst=(3,). Cannot broadcast.
            var src = np.arange(5).astype(NPTypeCode.Int32);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(3), fillZeros: false);

            Action act = () => NDIter.Copy(dst, src);
            act.Should().Throw<IncorrectShapeException>();
        }

        // =====================================================================
        // Section 5 — np.concatenate end-to-end (which uses NDIter.Copy under
        //             the hood for the general path).  Verify NumPy parity.
        // =====================================================================

        [TestMethod]
        public void Concatenate_SameDtype_Contig_1D()
        {
            var a = np.arange(5).astype(NPTypeCode.Int32);
            var b = np.arange(5, 10).astype(NPTypeCode.Int32);

            var result = np.concatenate(new[] { a, b }, 0);

            result.shape.Should().BeEquivalentTo(new[] { 10 });
            for (int i = 0; i < 10; i++)
                ((int)result[i]).Should().Be(i);
        }

        [TestMethod]
        public void Concatenate_CrossDtype_Float32_Int32_To_Double()
        {
            var a = np.arange(3).astype(NPTypeCode.Single);
            var b = np.arange(3, 6).astype(NPTypeCode.Int32);

            var result = np.concatenate(new[] { a, b }, 0, dtype: NPTypeCode.Double);

            for (int i = 0; i < 6; i++)
                ((double)result[i]).Should().Be((double)i);
        }

        [TestMethod]
        public void Concatenate_2D_Axis1_SameDtype()
        {
            var a = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var b = np.arange(6, 10).reshape(2, 2).astype(NPTypeCode.Int32);

            var result = np.concatenate(new[] { a, b }, 1);

            result.shape.Should().BeEquivalentTo(new[] { 2, 5 });
            // Row 0: 0,1,2 | 6,7
            ((int)result[0, 0]).Should().Be(0);
            ((int)result[0, 4]).Should().Be(7);
            // Row 1: 3,4,5 | 8,9
            ((int)result[1, 0]).Should().Be(3);
            ((int)result[1, 4]).Should().Be(9);
        }

        // =====================================================================
        // Section 6 — Larger sizes: regression check on hot path.
        // =====================================================================

        [TestMethod]
        public void FastPath_Large_1M_Int32_SameShape()
        {
            var src = np.arange(1_000_000).astype(NPTypeCode.Int32);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(1_000_000), fillZeros: false);

            NDIter.Copy(dst, src);

            // Spot-check head, middle, tail.
            ((int)dst[0]).Should().Be(0);
            ((int)dst[500_000]).Should().Be(500_000);
            ((int)dst[999_999]).Should().Be(999_999);
        }

        [TestMethod]
        public void FastPath_Large_1M_Float32_To_Double_SameShape()
        {
            var src = np.arange(1_000_000).astype(NPTypeCode.Single);
            var dst = new NDArray(NPTypeCode.Double, new Shape(1_000_000), fillZeros: false);

            NDIter.Copy(dst, src);

            ((double)dst[0]).Should().Be(0.0);
            ((double)dst[500_000]).Should().Be(500_000.0);
            ((double)dst[999_999]).Should().Be(999_999.0);
        }

        // =====================================================================
        // Section 7 — Hash-collision / corner cases for ShapesMatchExactly.
        // =====================================================================

        [TestMethod]
        public unsafe void FastPath_DimZero_BothScalar()
        {
            // 0-D scalar arrays — same NDim=0 = match (ShapesMatchExactly returns
            // true via the "if (src.NDim == 0) return true;" early return).
            // Verify the single-element value transferred correctly. NDArray's
            // public indexers don't handle 0-D shapes uniformly, so read raw.
            var src = np.array(42);
            var dst = np.empty(new Shape(), NPTypeCode.Int32);

            src.ndim.Should().Be(0);
            dst.ndim.Should().Be(0);

            NDIter.Copy(dst, src);

            // Read the single int32 directly from unmanaged storage.
            int value = *(int*)dst.Storage.Address;
            value.Should().Be(42);
        }

        [TestMethod]
        public void SlowPath_NDimDiffers_NotMatch()
        {
            // src=(4,) (ndim=1) vs dst=(2,2) (ndim=2) — NDim differs → fast path skipped.
            // But because total elements match, broadcast_to can't make this work
            // (validation throws). Confirms NDim mismatch routes through broadcast_to.
            var src = np.arange(4).astype(NPTypeCode.Int32);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(2, 2), fillZeros: false);

            Action act = () => NDIter.Copy(dst, src);
            act.Should().Throw<IncorrectShapeException>(
                because: "src (4,) cannot broadcast to dst (2,2)");
        }

        [TestMethod]
        public void FastPath_SameDimsButDifferentSize_NotMatchOnZero()
        {
            // (0, 5) and (0, 5) — fast path applies, but size=0 so no copy occurs.
            // Verifies the size==0 short-circuit in NDIter.Copy.
            var src = new NDArray(NPTypeCode.Int32, new Shape(0, 5), fillZeros: false);
            var dst = new NDArray(NPTypeCode.Int32, new Shape(0, 5), fillZeros: false);

            NDIter.Copy(dst, src);  // must not throw

            dst.size.Should().Be(0);
        }
    }
}
