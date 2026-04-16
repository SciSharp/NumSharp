using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// TDD tests for bugs discovered during deep audit (2026-04-16).
    /// Each test was generated from actual NumPy 2.4.2 output.
    ///
    /// Bugs fixed by these tests:
    /// - Bug #1: Negative strides were always flipped (should only flip for K-order)
    /// - Bug #2: NO_BROADCAST flag was not enforced
    /// - Bug #3: F_INDEX returned C-order indices
    /// - Bug #4: ALLOCATE with null operand threw NullReferenceException
    /// - Bug #5,6,7: op_axes reductions produced wrong output or threw
    /// </summary>
    [TestClass]
    public class NpyIterParityFixTests
    {
        // =====================================================================
        // Bug #1: Negative stride flipping - should only flip for K-order
        //
        // NumPy source: nditer_constr.c:297-307
        //   if (!(itflags & NPY_ITFLAG_FORCEDORDER)) {
        //       if (!any_allocate && !(flags & NPY_ITER_DONT_NEGATE_STRIDES)) {
        //           npyiter_flip_negative_strides(iter);
        //       }
        //   }
        // NPY_ITFLAG_FORCEDORDER is set for C, F, and A orders.
        // Only K-order skips it.
        // =====================================================================

        [TestMethod]
        public void NegStride_1D_Reversed_COrder_IteratesLogical()
        {
            // NumPy 2.4.2:
            // arr = np.arange(5)[::-1]
            // list(np.nditer(arr, order='C')) == [4, 3, 2, 1, 0]
            var arr = np.arange(5)["::-1"];
            var expected = new[] { 4, 3, 2, 1, 0 };

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_CORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        [TestMethod]
        public void NegStride_1D_Reversed_FOrder_IteratesLogical()
        {
            var arr = np.arange(5)["::-1"];
            var expected = new[] { 4, 3, 2, 1, 0 };

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_FORTRANORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        [TestMethod]
        public void NegStride_1D_Reversed_AOrder_IteratesLogical()
        {
            var arr = np.arange(5)["::-1"];
            var expected = new[] { 4, 3, 2, 1, 0 };

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_ANYORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        [TestMethod]
        public void NegStride_1D_Reversed_KOrder_IteratesMemory()
        {
            // K-order should flip negative strides -> memory order
            var arr = np.arange(5)["::-1"];
            var expected = new[] { 0, 1, 2, 3, 4 };  // memory order

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_KEEPORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        [TestMethod]
        public void NegStride_2D_RowReversed_COrder_IteratesLogical()
        {
            // arr = np.arange(6).reshape(2,3)[::-1, :]
            // list(np.nditer(arr, order='C')) == [3, 4, 5, 0, 1, 2]
            var arr = np.arange(6).reshape(2, 3)["::-1, :"];
            var expected = new[] { 3, 4, 5, 0, 1, 2 };

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_CORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        [TestMethod]
        public void NegStride_2D_RowReversed_FOrder_IteratesLogical()
        {
            var arr = np.arange(6).reshape(2, 3)["::-1, :"];
            var expected = new[] { 3, 0, 4, 1, 5, 2 };

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_FORTRANORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        [TestMethod]
        public void NegStride_2D_ColReversed_COrder_IteratesLogical()
        {
            // arr = np.arange(6).reshape(2,3)[:, ::-1]
            // list(np.nditer(arr, order='C')) == [2, 1, 0, 5, 4, 3]
            var arr = np.arange(6).reshape(2, 3)[":, ::-1"];
            var expected = new[] { 2, 1, 0, 5, 4, 3 };

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_CORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        [TestMethod]
        public void NegStride_2D_ColReversed_FOrder_IteratesLogical()
        {
            var arr = np.arange(6).reshape(2, 3)[":, ::-1"];
            var expected = new[] { 2, 5, 1, 4, 0, 3 };

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_FORTRANORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        [TestMethod]
        public void NegStride_2D_BothReversed_COrder_IteratesLogical()
        {
            // arr = np.arange(6).reshape(2,3)[::-1, ::-1]
            // list(np.nditer(arr, order='C')) == [5, 4, 3, 2, 1, 0]
            var arr = np.arange(6).reshape(2, 3)["::-1, ::-1"];
            var expected = new[] { 5, 4, 3, 2, 1, 0 };

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_CORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        [TestMethod]
        public void NegStride_2D_BothReversed_FOrder_IteratesLogical()
        {
            var arr = np.arange(6).reshape(2, 3)["::-1, ::-1"];
            var expected = new[] { 5, 2, 4, 1, 3, 0 };

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_FORTRANORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        [TestMethod]
        public void NegStride_2D_BothReversed_AOrder_IteratesCOrder()
        {
            // A-order: When not all F-contiguous, behaves like C-order
            var arr = np.arange(6).reshape(2, 3)["::-1, ::-1"];
            var expected = new[] { 5, 4, 3, 2, 1, 0 };

            using var it = NpyIterRef.New(arr, order: NPY_ORDER.NPY_ANYORDER);
            var values = new List<int>();
            do { values.Add(it.GetValue<int>(0)); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, values.ToArray());
        }

        // =====================================================================
        // Bug #2: NO_BROADCAST flag enforcement
        //
        // NumPy behavior: ValueError with message about non-broadcastable operand
        // =====================================================================

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void NoBroadcast_ShapeMismatch_Throws()
        {
            // NumPy 2.4.2:
            // a = np.arange(3)         # shape (3,)
            // b = np.arange(6).reshape(2,3)  # shape (2,3)
            // np.nditer([a,b], op_flags=[['readonly','no_broadcast'],['readonly']])
            // -> ValueError: non-broadcastable operand with shape (3,) doesn't match the broadcast shape (2,3)
            var a = np.arange(3);
            var b = np.arange(6).reshape(2, 3);

            using var it = NpyIterRef.MultiNew(
                2, new[] { a, b },
                NpyIterGlobalFlags.None,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] {
                    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.NO_BROADCAST,
                    NpyIterPerOpFlags.READONLY
                });
        }

        [TestMethod]
        public void NoBroadcast_SameShape_Works()
        {
            // NO_BROADCAST with matching shapes should work fine
            var a = np.arange(6).reshape(2, 3);
            var b = np.arange(6).reshape(2, 3) * 10;

            using var it = NpyIterRef.MultiNew(
                2, new[] { a, b },
                NpyIterGlobalFlags.None,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] {
                    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.NO_BROADCAST,
                    NpyIterPerOpFlags.READONLY
                });

            // Should not throw
            Assert.AreEqual(6, it.IterSize);
        }

        // =====================================================================
        // Bug #3: F_INDEX returns F-order indices
        //
        // NumPy 2.4.2:
        // arr = np.arange(6).reshape(2,3)
        // F_INDEX iterates in C-order (memory) but reports F-order index
        // Expected: [0, 2, 4, 1, 3, 5]
        // =====================================================================

        [TestMethod]
        public void FIndex_2D_ReturnsFOrderIndices()
        {
            var arr = np.arange(6).reshape(2, 3);
            var expected = new long[] { 0, 2, 4, 1, 3, 5 };

            using var it = NpyIterRef.New(arr, NpyIterGlobalFlags.F_INDEX);
            var indices = new List<long>();
            do { indices.Add(it.GetIndex()); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, indices.ToArray());
        }

        [TestMethod]
        public void CIndex_2D_ReturnsCOrderIndices()
        {
            var arr = np.arange(6).reshape(2, 3);
            var expected = new long[] { 0, 1, 2, 3, 4, 5 };

            using var it = NpyIterRef.New(arr, NpyIterGlobalFlags.C_INDEX);
            var indices = new List<long>();
            do { indices.Add(it.GetIndex()); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, indices.ToArray());
        }

        [TestMethod]
        public void FIndex_3D_ReturnsFOrderIndices()
        {
            // arr = np.arange(24).reshape(2,3,4)
            // F-order strides: [1, 2, 6]
            // C-order iteration: multi_index (i,j,k) gives F_index = i*1 + j*2 + k*6
            var arr = np.arange(24).reshape(2, 3, 4);
            var expected = new List<long>();
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        expected.Add(i + j * 2 + k * 6);

            using var it = NpyIterRef.New(arr, NpyIterGlobalFlags.F_INDEX);
            var indices = new List<long>();
            do { indices.Add(it.GetIndex()); } while (it.Iternext());

            CollectionAssert.AreEqual(expected, indices);
        }

        // =====================================================================
        // Bug #4: ALLOCATE with null operand should allocate
        // =====================================================================

        [TestMethod]
        public void Allocate_NullOperand_CreatesOutput()
        {
            // NumPy:
            // a = np.arange(6).reshape(2,3)
            // it = np.nditer([a, None],
            //                op_flags=[['readonly'], ['writeonly','allocate']],
            //                op_dtypes=[None, np.float64])
            // it.operands[1] has shape (2,3), dtype float64
            // Note: np.arange(6) returns Int64 in NumSharp, so we use Empty dtype for op[0]
            //       (means "use the operand's own dtype").
            var a = np.arange(6).reshape(2, 3);
            NDArray[] ops = new NDArray[] { a, null };

            using var it = NpyIterRef.MultiNew(
                2, ops,
                NpyIterGlobalFlags.None,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_UNSAFE_CASTING,
                new[] {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.ALLOCATE
                },
                new[] { NPTypeCode.Empty, NPTypeCode.Double });

            var operands = it.GetOperandArray();
            Assert.IsNotNull(operands[1], "Output should be allocated");
            Assert.AreEqual(NPTypeCode.Double, operands[1].typecode);
            CollectionAssert.AreEqual(new long[] { 2, 3 }, operands[1].shape);
        }

        // =====================================================================
        // Bug #5-7: op_axes reductions must match NumPy
        // =====================================================================

        [TestMethod]
        public void OpAxes_Reduce_Axis0_2D_To_1D()
        {
            // NumPy:
            // a = np.arange(6).reshape(2,3)
            // out = np.zeros(3, dtype=np.int64)
            // it = np.nditer([a, out], flags=['reduce_ok'],
            //                op_flags=[['readonly'], ['readwrite']],
            //                op_axes=[[0,1], [-1,0]])
            // for x, y in it: y[...] = y + x
            // out == [3, 5, 7]  (column sums)
            var a = np.arange(6).reshape(2, 3);
            var outArr = np.zeros(new Shape(3), NPTypeCode.Int64);
            var ops = new NDArray[] { a, outArr };
            var opFlags = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE };
            var opAxes = new int[][] { new[] { 0, 1 }, new[] { -1, 0 } };

            using var it = NpyIterRef.AdvancedNew(
                2, ops, NpyIterGlobalFlags.REDUCE_OK,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                opFlags, null, opAxesNDim: 2, opAxes: opAxes);

            do {
                long x = it.GetValue<int>(0);
                long y = it.GetValue<long>(1);
                it.SetValue<long>(y + x, 1);
            } while (it.Iternext());

            var actual = outArr.ToArray<long>();
            CollectionAssert.AreEqual(new long[] { 3, 5, 7 }, actual);
        }

        [TestMethod]
        public void OpAxes_Reduce_Axis1_2D_To_1D()
        {
            // NumPy:
            // a = np.arange(6).reshape(2,3)
            // out = np.zeros(2, dtype=np.int64)
            // it = np.nditer([a, out], flags=['reduce_ok'],
            //                op_flags=[['readonly'], ['readwrite']],
            //                op_axes=[[0,1], [0,-1]])
            // for x, y in it: y[...] = y + x
            // out == [3, 12]  (row sums)
            var a = np.arange(6).reshape(2, 3);
            var outArr = np.zeros(new Shape(2), NPTypeCode.Int64);
            var ops = new NDArray[] { a, outArr };
            var opFlags = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE };
            var opAxes = new int[][] { new[] { 0, 1 }, new[] { 0, -1 } };

            using var it = NpyIterRef.AdvancedNew(
                2, ops, NpyIterGlobalFlags.REDUCE_OK,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                opFlags, null, opAxesNDim: 2, opAxes: opAxes);

            do {
                long x = it.GetValue<int>(0);
                long y = it.GetValue<long>(1);
                it.SetValue<long>(y + x, 1);
            } while (it.Iternext());

            var actual = outArr.ToArray<long>();
            CollectionAssert.AreEqual(new long[] { 3, 12 }, actual);
        }

        [TestMethod]
        public void OpAxes_FullReduce_2D_To_Scalar()
        {
            // NumPy:
            // a = np.arange(6).reshape(2,3)
            // out = np.zeros((), dtype=np.int64)
            // it = np.nditer([a, out], flags=['reduce_ok'],
            //                op_flags=[['readonly'], ['readwrite']],
            //                op_axes=[[0,1], [-1,-1]])
            // out == 15
            var a = np.arange(6).reshape(2, 3);
            var outArr = np.zeros(new Shape(), NPTypeCode.Int64);
            var ops = new NDArray[] { a, outArr };
            var opFlags = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE };
            var opAxes = new int[][] { new[] { 0, 1 }, new[] { -1, -1 } };

            using var it = NpyIterRef.AdvancedNew(
                2, ops, NpyIterGlobalFlags.REDUCE_OK,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                opFlags, null, opAxesNDim: 2, opAxes: opAxes);

            do {
                long x = it.GetValue<int>(0);
                long y = it.GetValue<long>(1);
                it.SetValue<long>(y + x, 1);
            } while (it.Iternext());

            Assert.AreEqual(15L, outArr.GetValue<long>());
        }

        [TestMethod]
        public void OpAxes_Reduce_Axis0_10x10()
        {
            // NumPy:
            // a = np.arange(100).reshape(10, 10)
            // out = np.zeros(10, dtype=np.int64)
            // it = np.nditer([a, out], flags=['reduce_ok'],
            //                op_flags=[['readonly'], ['readwrite']],
            //                op_axes=[[0,1], [-1,0]])
            // out == [450, 460, 470, 480, 490, 500, 510, 520, 530, 540]
            var a = np.arange(100).reshape(10, 10);
            var outArr = np.zeros(new Shape(10), NPTypeCode.Int64);
            var ops = new NDArray[] { a, outArr };
            var opFlags = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE };
            var opAxes = new int[][] { new[] { 0, 1 }, new[] { -1, 0 } };

            using var it = NpyIterRef.AdvancedNew(
                2, ops,
                NpyIterGlobalFlags.REDUCE_OK,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                opFlags, null, opAxesNDim: 2, opAxes: opAxes);

            do {
                long x = it.GetValue<int>(0);
                long y = it.GetValue<long>(1);
                it.SetValue<long>(y + x, 1);
            } while (it.Iternext());

            var actual = outArr.ToArray<long>();
            CollectionAssert.AreEqual(
                new long[] { 450, 460, 470, 480, 490, 500, 510, 520, 530, 540 },
                actual);
        }
    }
}
