using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    /// Parity + correctness tests for the np.zeros allocation fast path
    /// (calloc / Windows VirtualAlloc demand-zero — see UnmanagedMemoryBlock.AllocateZeroed,
    /// SizeBucketedBufferPool.TakeZeroed, OsVirtualMemory).
    ///
    /// The optimization replaced an eager per-element fill with OS-backed zero
    /// pages. These tests pin the NumPy-equivalent behavior it must preserve:
    /// every dtype reads back as all-zeros, across the small (heap calloc),
    /// medium and large (VirtualAlloc demand-zero on Windows) size regimes,
    /// for contiguous, multi-dim, sliced and written-into arrays.
    /// </summary>
    [TestClass]
    public class NumPyZerosAllocationTests
    {
        // Sizes chosen to exercise each allocation regime:
        //  - 64    : tiny, heap calloc, below VirtualAlloc threshold
        //  - 257   : crosses the SIMD-tail boundary, still heap
        //  - 50_000: ~200KB-400KB, near/over the 128KB VirtualAlloc threshold
        //  - 2_000_000: multi-MB, squarely on the VirtualAlloc demand-zero path
        private static readonly int[] Sizes = { 1, 64, 257, 50_000, 2_000_000 };

        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
            NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64,
            NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
            NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
        };

        private static bool IsZero(object v, NPTypeCode c) => c switch
        {
            NPTypeCode.Boolean => (bool)v == false,
            NPTypeCode.Decimal => (decimal)v == 0m,
            NPTypeCode.Complex => (Complex)v == Complex.Zero,
            NPTypeCode.Half => (Half)v == (Half)0,
            NPTypeCode.Char => (char)v == '\0',
            _ => Convert.ToDouble(v) == 0d
        };

        [TestMethod]
        public void AllDtypes_AllSizes_AreZeroed()
        {
            // np.zeros(n, dtype) == all zeros, for every supported dtype and every
            // allocation regime. The whole point of the calloc/VirtualAlloc path is
            // that the all-zero bit pattern equals default(T) for all 15 dtypes.
            foreach (var c in AllDtypes)
            {
                foreach (var n in Sizes)
                {
                    using var z = np.zeros(new Shape(n), c);
                    Assert.AreEqual((long)n, z.size, $"size mismatch {c} n={n}");
                    Assert.AreEqual(c, z.typecode, $"dtype mismatch {c} n={n}");

                    // Spot-check head, tail, and an interior element (cheap for big n).
                    Assert.IsTrue(IsZero(z.GetAtIndex(0), c), $"{c} n={n} [0] not zero");
                    Assert.IsTrue(IsZero(z.GetAtIndex(n - 1), c), $"{c} n={n} [last] not zero");
                    Assert.IsTrue(IsZero(z.GetAtIndex(n / 2), c), $"{c} n={n} [mid] not zero");
                }
            }
        }

        [TestMethod]
        public void Large_FullScan_IsZeroed()
        {
            // Exhaustively scan a multi-MB array (VirtualAlloc demand-zero path on
            // Windows) to ensure EVERY element is zero, not just the spot-checks.
            using var z = np.zeros(new Shape(3_000_000), NPTypeCode.Double);
            var data = z.Data<double>();
            for (long i = 0; i < z.size; i++)
                if (data[i] != 0d)
                    Assert.Fail($"element {i} was {data[i]}, expected 0");
        }

        [TestMethod]
        public void Large_IsWriteable_AndCommitsCorrectly()
        {
            // A VirtualAlloc'd buffer must be writeable end-to-end: writing the
            // first and last pages must fault-in + commit correctly and read back.
            using var z = np.zeros(new Shape(2_000_000), NPTypeCode.Double);
            Assert.IsTrue(z.Shape.IsWriteable);

            z.SetData(1.5, 0);
            z.SetData(2.5, 1_999_999);
            z.SetData(3.5, 1_000_000);

            Assert.AreEqual(1.5, z.GetDouble(0));
            Assert.AreEqual(2.5, z.GetDouble(1_999_999));
            Assert.AreEqual(3.5, z.GetDouble(1_000_000));
            // Untouched neighbour still zero (demand-zero correctness).
            Assert.AreEqual(0d, z.GetDouble(1_000_001));
        }

        [TestMethod]
        public void OwnsData_NotAView()
        {
            // np.zeros owns its data (base is null), matching numpy's a.base is None.
            using var z = np.zeros(new Shape(50_000), NPTypeCode.Double);
            Assert.IsNull(z.@base, "np.zeros result must own its data (base == null)");
        }

        [TestMethod]
        public void TwoZeros_DoNotAlias()
        {
            // Distinct allocations must not share memory: writing one leaves the
            // other untouched. Guards against a pool/calloc handing out the same
            // pointer twice.
            using var a = np.zeros(new Shape(100_000), NPTypeCode.Double);
            using var b = np.zeros(new Shape(100_000), NPTypeCode.Double);
            a.SetData(42.0, 0);
            Assert.AreEqual(42.0, a.GetDouble(0));
            Assert.AreEqual(0d, b.GetDouble(0), "second zeros array aliased the first");
        }

        [TestMethod]
        public void ReuseAfterDispose_ReturnsZeroed()
        {
            // Dirty a buffer, dispose it (returns to the pool), then allocate the
            // same size again: it must still read back as zeros (calloc / re-zero),
            // never the previous contents.
            const int n = 20_000; // 160KB double -> may round-trip the pool
            for (int i = 0; i < 8; i++)
            {
                using var dirty = np.zeros(new Shape(n), NPTypeCode.Double);
                dirty.SetData(7.0, 0);
                dirty.SetData(9.0, n - 1);
            }
            using var fresh = np.zeros(new Shape(n), NPTypeCode.Double);
            Assert.AreEqual(0d, fresh.GetDouble(0));
            Assert.AreEqual(0d, fresh.GetDouble(n - 1));
            Assert.AreEqual(0d, fresh.GetDouble(n / 2));
        }

        [TestMethod]
        public void MultiDim_IsZeroed_AndShaped()
        {
            using var z = np.zeros(new Shape(64, 128), NPTypeCode.Single);
            CollectionAssert.AreEqual(new long[] { 64, 128 }, z.shape);
            Assert.AreEqual(64L * 128, z.size);
            var data = z.Data<float>();
            for (long i = 0; i < z.size; i++)
                Assert.AreEqual(0f, data[i]);
        }

        [TestMethod]
        public void EmptyShape_DoesNotCrash_AndIsEmpty()
        {
            // np.zeros((0,3)) -> size 0, no allocation work, no crash.
            using var z = np.zeros(new Shape(0, 3), NPTypeCode.Double);
            Assert.AreEqual(0L, z.size);
            CollectionAssert.AreEqual(new long[] { 0, 3 }, z.shape);
        }

        [TestMethod]
        public void DefaultDtype_IsDouble()
        {
            // np.zeros(shape) defaults to float64, like numpy.
            using var z = np.zeros(new Shape(8));
            Assert.AreEqual(typeof(double), z.dtype);
            Assert.AreEqual(NPTypeCode.Double, z.typecode);
        }

        [TestMethod]
        public void Overloads_AllProduceZeros()
        {
            // Every public np.zeros overload funnels through the zeroed-allocation
            // path and must produce zeros with the right dtype/shape.
            using var byInt = np.zeros(5);
            Assert.AreEqual(typeof(double), byInt.dtype);
            Assert.AreEqual(0d, byInt.GetDouble(0));
            Assert.AreEqual(0d, byInt.GetDouble(4));

            using var byIntArr = np.zeros(new int[] { 2, 3 });
            Assert.AreEqual(6L, byIntArr.size);
            Assert.AreEqual(0d, byIntArr.GetDouble(0));

            using var byLongArr = np.zeros(new long[] { 2, 3 });
            Assert.AreEqual(6L, byLongArr.size);

            using var byGeneric = np.zeros<int>(new int[] { 4 });
            Assert.AreEqual(typeof(int), byGeneric.dtype);
            Assert.AreEqual(0, byGeneric.GetInt32(0));

            using var byShapeType = np.zeros(new Shape(3), typeof(float));
            Assert.AreEqual(typeof(float), byShapeType.dtype);
            Assert.AreEqual(0f, byShapeType.GetSingle(2));
        }

        [TestMethod]
        public void HighRank_IsZeroed()
        {
            using var z = np.zeros(new Shape(2, 3, 4, 5, 6), NPTypeCode.Int32);
            Assert.AreEqual(2L * 3 * 4 * 5 * 6, z.size);
            var data = z.Data<int>();
            for (long i = 0; i < z.size; i++)
                Assert.AreEqual(0, data[i]);
        }

        [TestMethod]
        public void SlicedView_OfZeros_IsZeroed()
        {
            // A strided view over a zeros array reads zeros (no stale bytes leak
            // through offset/stride math).
            using var z = np.zeros(new Shape(1000), NPTypeCode.Double);
            var view = z["::2"];
            Assert.AreEqual(500L, view.size);
            Assert.AreEqual(0d, view.GetDouble(0));
            Assert.AreEqual(0d, view.GetDouble(499));
        }
    }
}
