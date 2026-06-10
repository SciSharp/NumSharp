using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Wave 1.4 hygiene gates — NumPy fill_axisdata stride invariant and its
    /// dependents. NumPy (nditer_constr.c:1594-1615) forces stride 0 for every
    /// operand on any iterator axis of size 1 and on broadcast-stretched dims;
    /// npyiter_coalesce_axes' trivial-branch and merge rule both rely on it.
    /// NumSharp used to copy raw operand strides instead, which broke three
    /// ways (all reproduced against NumPy 2.4.2 before the fix):
    ///
    ///  1. RemoveMultiIndex coalesced a size-1 axis carrying a nonzero stride
    ///     and the stale merge rule kept the wrong stride — a (1,4) view with
    ///     element strides (1,2) iterated [0,1,2,3] instead of [0,2,4,6].
    ///  2. The op_axes fill path used the raw array stride for an operand axis
    ///     of length 1 stretched to a larger iter dim — out-of-bounds reads.
    ///  3. FlipNegativeStrides converted element strides to byte offsets with
    ///     the BUFFER dtype size instead of the SOURCE dtype size (bug (b)
    ///     family) — K-order + buffered cast + negative strides read garbage.
    ///
    /// Also pins the NumPy write-broadcast validation (a stretched write dim is
    /// a reduction: REDUCE_OK + readable required) and the PARALLEL_SAFE
    /// extension-flag wiring.
    /// </summary>
    [TestClass]
    public class NpyIterSizeOneStrideTests
    {
        /// <summary>Shape (1,4) view over arange(8).reshape(4,2) with element strides (1,2).</summary>
        private static NDArray MakeSizeOneStridedView()
        {
            var baseArr = np.arange(8).reshape(4, 2);
            var v = baseArr[":, 0:1"].T;
            Assert.AreEqual(2, v.ndim, "view ndim");
            Assert.AreEqual(1L, (long)v.shape[0], "view dim 0");
            Assert.AreEqual(4L, (long)v.shape[1], "view dim 1");
            return v;
        }

        [TestMethod]
        public void RemoveMultiIndex_SizeOneAxisWithNonzeroStride_IteratesSourceOrder()
        {
            // NumPy: np.nditer(v, flags=['multi_index']); it.remove_multi_index()
            // -> [0, 2, 4, 6]. Before the fix the coalesce merge kept the size-1
            // axis' stride (1) and iterated [0, 1, 2, 3].
            var v = MakeSizeOneStridedView();

            using var iter = NpyIterRef.MultiNew(
                nop: 1,
                op: new[] { v },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY });

            iter.RemoveMultiIndex();

            var vals = new List<int>();
            do { vals.Add(iter.GetValue<int>(0)); } while (iter.Iternext());

            CollectionAssert.AreEqual(new[] { 0, 2, 4, 6 }, vals);
        }

        [TestMethod]
        public void PlainConstructor_SizeOneAxisWithNonzeroStride_IteratesSourceOrder()
        {
            // Control: the non-MULTI_INDEX constructor path on the same view.
            var v = MakeSizeOneStridedView();

            using var iter = NpyIterRef.MultiNew(
                nop: 1,
                op: new[] { v },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY });

            var vals = new List<int>();
            do { vals.Add(iter.GetValue<int>(0)); } while (iter.Iternext());

            CollectionAssert.AreEqual(new[] { 0, 2, 4, 6 }, vals);
        }

        [TestMethod]
        public void KOrderBufferedCast_NegativeStrides_IteratesMemoryOrder()
        {
            // NumPy: np.nditer(arange(10,int32)[::-1], ['buffered','external_loop'],
            // op_dtypes=[float64], casting='unsafe', order='K') -> [0..9].
            // FlipNegativeStrides used ElementSizes (8, the float64 buffer dtype)
            // instead of SrcElementSizes (4, int32) for the flip byte offset,
            // landing the base pointer past the array (garbage reads).
            var arr = np.arange(10).astype(np.int32)["::-1"];

            using var iter = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { arr },
                flags: NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_UNSAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                opDtypes: new[] { NPTypeCode.Double });

            var vals = new List<double>();
            do { vals.Add(iter.GetValue<double>(0)); } while (iter.Iternext());

            CollectionAssert.AreEqual(
                new[] { 0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, vals);
        }

        [TestMethod]
        public void OpAxes_OperandSizeOneDim_BroadcastsInsteadOfWalkingOutOfBounds()
        {
            // NumPy: np.nditer([arange(3.), array([10.])], op_axes=[[0],[0]])
            // -> (0,10) (1,10) (2,10). The op_axes fill used the raw array
            // stride for b's stretched size-1 axis and read past its buffer.
            var a3 = np.arange(3.0);
            var b1 = np.array(new double[] { 10.0 });

            using var iter = NpyIterRef.AdvancedNew(
                nop: 2,
                op: new[] { a3, b1 },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY },
                opAxes: new[] { new[] { 0 }, new[] { 0 } });

            var lhs = new List<double>();
            var rhs = new List<double>();
            do
            {
                lhs.Add(iter.GetValue<double>(0));
                rhs.Add(iter.GetValue<double>(1));
            } while (iter.Iternext());

            CollectionAssert.AreEqual(new[] { 0.0, 1, 2 }, lhs);
            CollectionAssert.AreEqual(new[] { 10.0, 10, 10 }, rhs);
        }

        [TestMethod]
        public void Coalesce_SizeOneAxes_FullyCoalesceToOneDim()
        {
            // With the stride-0 invariant in place, NumPy's STRICT trivial-branch
            // (shape==1 && stride==0) still fully coalesces contiguous arrays
            // that carry size-1 axes. NumPy external_loop chunking: (2,4,1),
            // (1,4) and (4,1) each produce a single chunk.
            foreach (var (shape, count) in new[]
                     {
                         (new Shape(2, 4, 1), 8),
                         (new Shape(1, 4), 4),
                         (new Shape(4, 1), 4),
                     })
            {
                var a = np.arange(count).reshape(shape);

                using var iter = NpyIterRef.MultiNew(
                    nop: 1,
                    op: new[] { a },
                    flags: NpyIterGlobalFlags.None,
                    order: NPY_ORDER.NPY_KEEPORDER,
                    casting: NPY_CASTING.NPY_SAFE_CASTING,
                    opFlags: new[] { NpyIterPerOpFlags.READONLY });

                Assert.AreEqual(1, iter.NDim, $"shape ({shape}) should coalesce to 1-D");

                var vals = new List<int>();
                do { vals.Add(iter.GetValue<int>(0)); } while (iter.Iternext());

                for (int i = 0; i < count; i++)
                    Assert.AreEqual(i, vals[i], $"shape ({shape}) element {i}");
            }
        }

        [TestMethod]
        public void WriteBroadcast_WithoutReduceOk_Throws()
        {
            // NumPy: np.nditer([arange(3.), zeros(1)], op_flags=[['readonly'],
            // ['readwrite']]) -> ValueError "output operand requires a reduction
            // along dimension 0, but the reduction is not enabled...". NumSharp
            // used to fill stride 0 silently and let the writes collide.
            var a = np.arange(3.0);
            var outArr = np.zeros(new Shape(1));

            var ex = Assert.ThrowsException<ArgumentException>(() =>
            {
                using var iter = NpyIterRef.MultiNew(
                    nop: 2,
                    op: new[] { a, outArr },
                    flags: NpyIterGlobalFlags.None,
                    order: NPY_ORDER.NPY_KEEPORDER,
                    casting: NPY_CASTING.NPY_SAFE_CASTING,
                    opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE });
            });

            StringAssert.Contains(ex.Message, "requires a reduction along dimension 0");
        }

        [TestMethod]
        public void WriteBroadcast_WriteOnlyWithReduceOk_Throws()
        {
            // NumPy: same construction with ['writeonly'] + reduce_ok ->
            // "output operand requires a reduction, but is flagged as
            // write-only, not read-write" (a reduction must read).
            var a = np.arange(3.0);
            var outArr = np.zeros(new Shape(1));

            var ex = Assert.ThrowsException<ArgumentException>(() =>
            {
                using var iter = NpyIterRef.MultiNew(
                    nop: 2,
                    op: new[] { a, outArr },
                    flags: NpyIterGlobalFlags.REDUCE_OK,
                    order: NPY_ORDER.NPY_KEEPORDER,
                    casting: NPY_CASTING.NPY_SAFE_CASTING,
                    opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            });

            StringAssert.Contains(ex.Message, "write-only, not read-write");
        }

        [TestMethod]
        public void WriteBroadcast_WithReduceOk_AccumulatesLikeNumPy()
        {
            // NumPy W7 probe: it = np.nditer([a, out], flags=['reduce_ok'],
            // op_flags=[['readonly'],['readwrite']]); y[...] = y + x per element
            // -> out == 0+1+2 == 3.0.
            var a = np.arange(3.0);
            var outArr = np.zeros(new Shape(1));

            using (var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, outArr },
                flags: NpyIterGlobalFlags.REDUCE_OK,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE }))
            {
                Assert.AreEqual(3L, iter.IterSize);

                do
                {
                    double x = iter.GetValue<double>(0);
                    double y = iter.GetValue<double>(1);
                    iter.SetValue(y + x, 1);
                } while (iter.Iternext());
            }

            Assert.AreEqual(3.0, outArr.GetValue<double>(0));
        }

        [TestMethod]
        public void GetAxisStrideArray_BufferedCast_ReturnsSourceByteStrides()
        {
            // Axis strides describe traversal of the SOURCE arrays (NumPy
            // NAD_STRIDES are array byte strides). Under a buffered cast the
            // multiplier must be the source element size — int32 contiguous
            // buffered as float64 has axis byte stride 4, not 8 (bug (b) family).
            var arr = np.arange(10).astype(np.int32);

            using var iter = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { arr },
                flags: NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_UNSAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                opDtypes: new[] { NPTypeCode.Double });

            Span<long> strides = stackalloc long[1];
            iter.GetAxisStrideArray(0, strides);
            Assert.AreEqual(4L, strides[0], "axis byte stride must use the int32 source element size");
        }

        [TestMethod]
        public void SizeOneAxis_StrideNormalizedToZero()
        {
            // The invariant itself, observed through GetAxisStrideArray with a
            // multi-index (no coalescing): the size-1 axis of a (1,4) float64
            // array reports byte stride 0 (NumPy fill_axisdata), the size-4
            // axis reports 8.
            var a = np.arange(4.0).reshape(1, 4);

            using var iter = NpyIterRef.MultiNew(
                nop: 1,
                op: new[] { a },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY });

            Span<long> strides = stackalloc long[1];
            iter.GetAxisStrideArray(0, strides);
            Assert.AreEqual(0L, strides[0], "size-1 axis must carry stride 0");

            iter.GetAxisStrideArray(1, strides);
            Assert.AreEqual(8L, strides[0], "size-4 axis keeps its true byte stride");
        }

        [TestMethod]
        public void NonContiguous_SizeOneAxes_AbsorbedFromInternalRepresentation()
        {
            // NumPy coalesces unconditionally after order resolution, absorbing
            // stride-0 size-1 axes. A trailing size-1 axis must not survive as
            // the innermost internal axis — it collapsed EXLOOP to one-element
            // inner loops ((N,1) strided views ran N kernel calls of count 1).
            var baseArr = np.arange(40).astype(np.float32).reshape(10, 4);
            var v = baseArr[":, 0:1"];  // (10,1) elem strides (4,1) — not contiguous

            using var iter = NpyIterRef.MultiNew(
                nop: 1,
                op: new[] { v },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_CORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(1, iter.NDim, "size-1 axis must be absorbed");

            var vals = new List<float>();
            do { vals.Add(iter.GetValue<float>(0)); } while (iter.Iternext());
            CollectionAssert.AreEqual(
                new[] { 0f, 4, 8, 12, 16, 20, 24, 28, 32, 36 }, vals);
        }

        [TestMethod]
        public void MultiIndex_SizeOneAxes_PreservedForIndexTracking()
        {
            // Index-tracking iterators must keep the original axis structure
            // (NumPy likewise skips coalescing under multi_index).
            var baseArr = np.arange(40).astype(np.float32).reshape(10, 4);
            var v = baseArr[":, 0:1"];

            using var iter = NpyIterRef.MultiNew(
                nop: 1,
                op: new[] { v },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY });

            Assert.AreEqual(2, iter.NDim, "multi-index tracking keeps original axes");
        }

        [TestMethod]
        public void ProductionBinary_TrailingSizeOneStridedView_CorrectValues()
        {
            // (N,1) strided view through np.add — the shape class that ran N
            // one-element inner loops before unit-axis absorption.
            var baseArr = np.arange(40).astype(np.float32).reshape(10, 4);
            var v = baseArr[":, 0:1"];
            var sum = v + v;
            for (int i = 0; i < 10; i++)
                Assert.AreEqual(8f * i, Convert.ToSingle(sum[i, 0].GetValue(0)), $"row {i}");
        }

        // =====================================================================
        // PARALLEL_SAFE wiring (NumSharp extension flag, consumed by Wave 6.2)
        // =====================================================================

        [TestMethod]
        public void ParallelSafe_ReadOnlyIterator_IsSet()
        {
            var a = np.arange(12.0).reshape(3, 4);
            var b = np.arange(4.0);

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, b },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY });

            Assert.IsTrue(iter.IsParallelSafe, "no WRITE operands -> parallel safe");
        }

        [TestMethod]
        public void ParallelSafe_SingleWriteWithCopyIfOverlap_IsSet()
        {
            var a = np.arange(8.0);
            var b = np.arange(8.0);
            var dst = np.zeros(new Shape(8));

            using var iter = NpyIterRef.MultiNew(
                nop: 3,
                op: new[] { a, b, dst },
                flags: NpyIterGlobalFlags.COPY_IF_OVERLAP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY,
                });

            Assert.IsTrue(iter.IsParallelSafe,
                "one WRITE operand with overlap resolved by COPY_IF_OVERLAP -> parallel safe");
        }

        [TestMethod]
        public void ParallelSafe_SingleWriteWithoutCopyIfOverlap_NotSet()
        {
            var a = np.arange(8.0);
            var dst = np.zeros(new Shape(8));

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, dst },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            Assert.IsFalse(iter.IsParallelSafe,
                "without COPY_IF_OVERLAP the write/input overlap is unverified");
        }

        [TestMethod]
        public void ParallelSafe_ReduceIterator_NotSet()
        {
            var a = np.arange(3.0);
            var outArr = np.zeros(new Shape(1));

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { a, outArr },
                flags: NpyIterGlobalFlags.REDUCE_OK | NpyIterGlobalFlags.COPY_IF_OVERLAP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE });

            Assert.IsFalse(iter.IsParallelSafe,
                "REDUCE accumulates across iterations on a shared slot");
        }
    }
}
