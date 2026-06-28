using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     Parity tests for <c>np.broadcast(a, b).iters</c> — the per-operand flat iterators
    ///     (<see cref="NumSharp.Backends.Iteration.NDFlatIterator"/>) that replaced the removed
    ///     <c>NDIterator</c>. Every expected sequence below was produced by running NumPy 2.4.2.
    ///
    ///     Ground-truth invariant (verified against NumPy for contiguous, transposed/F-order,
    ///     reversed- and stepped-slice operands):
    ///     <code>
    ///     np.broadcast(a, b).iters[i] == np.broadcast_to(op_i, result_shape).ravel(order='C')
    ///     </code>
    ///     i.e. each iters[i] yields its operand stretched to the broadcast result shape, walked
    ///     in C-order of the RESULT (not the operand's own memory order).
    ///
    ///     np.broadcast accepts 0..64 operands like NumPy (see the N-operand tests), exposes the
    ///     live <c>index</c> cursor, is iterable (yields one per-operand value tuple per step), and
    ///     supports <c>reset()</c> — all matching NumPy (see the live-cursor tests).
    ///     Documented divergences (see the [Misaligned] tests):
    ///       - <c>iters</c> are re-enumerable and independent of the cursor, whereas NumPy's
    ///         flatiters are one-shot and share the broadcast's live cursor.
    ///       - no operand cap: NumSharp accepts any number of operands (matching its unlimited
    ///         NDIter), whereas NumPy caps at NPY_MAXARGS=64.
    /// </summary>
    [TestClass]
    public class NDBroadcastItersTests
    {
        #region Helpers

        /// <summary>Flatten iters[i] to a typed array (unboxes — fails loudly if the element dtype is wrong).</summary>
        private static T[] Iter<T>(np.Broadcast b, int i) => b.iters[i].Cast<T>().ToArray();

        /// <summary>Assert both iters sequences (int64 operands) against NumPy-derived expectations.</summary>
        private static void AssertIters(NDArray a, NDArray b, int[] shape, long[] e0, long[] e1)
        {
            var bc = np.broadcast(a, b);
            bc.shape.dimensions.Should().Equal(shape.Select(d => (long)d), "broadcast result shape must match NumPy");
            bc.ndim.Should().Be(shape.Length);
            bc.nd.Should().Be(shape.Length);
            bc.size.Should().Be(e0.Length);
            bc.numiter.Should().Be(2);
            bc.iters.Length.Should().Be(2);
            Iter<long>(bc, 0).Should().Equal(e0, "iters[0] must equal broadcast_to(a, shape).ravel(C)");
            Iter<long>(bc, 1).Should().Equal(e1, "iters[1] must equal broadcast_to(b, shape).ravel(C)");
        }

        /// <summary>
        ///     A dtype-T row [v0,v1,v2] broadcast against an int64 column [[0],[10]] -> shape (2,3).
        ///     iters[0] must cycle the row twice (in T); iters[1] must broadcast the column.
        /// </summary>
        private static void AssertCycle<T>(NDArray row, params T[] expectedRow)
        {
            var col = np.array(new long[,] { { 0 }, { 10 } }); // shape (2,1)
            var bc = np.broadcast(row, col);                   // -> (2,3)

            bc.shape.dimensions.Should().Equal(new long[] { 2, 3 });
            bc.size.Should().Be(6);
            bc.numiter.Should().Be(2);

            Iter<T>(bc, 0).Should().Equal(expectedRow.Concat(expectedRow).ToArray(),
                $"iters[0] for {typeof(T).Name} must cycle the row and stay typed as {typeof(T).Name}");
            Iter<long>(bc, 1).Should().Equal(new long[] { 0, 0, 0, 10, 10, 10 },
                "iters[1] (int64 column) must broadcast across the row axis");
        }

        /// <summary>Assert an N-operand broadcast's shape, numiter, and every int64 iters sequence.</summary>
        private static void AssertItersN(np.Broadcast bc, int[] shape, params long[][] expected)
        {
            bc.shape.dimensions.Should().Equal(shape.Select(d => (long)d), "broadcast result shape must match NumPy");
            bc.ndim.Should().Be(shape.Length);
            bc.numiter.Should().Be(expected.Length, "numiter == number of operands");
            bc.iters.Length.Should().Be(expected.Length);
            for (int i = 0; i < expected.Length; i++)
                Iter<long>(bc, i).Should().Equal(expected[i], $"iters[{i}] must equal broadcast_to(op_{i}, shape).ravel(C)");
        }

        #endregion

        // ================================================================
        // Canonical shape cases (exact NumPy 2.4.2 output)
        // ================================================================

        /// <summary>
        ///     np.broadcast([1,2,3], [[10],[20]]).iters
        ///       iters[0] -> 1,2,3,1,2,3
        ///       iters[1] -> 10,10,10,20,20,20
        /// </summary>
        [TestMethod]
        public void Iters_RowAndColumn_MatchNumPy()
        {
            AssertIters(
                np.array(new long[] { 1, 2, 3 }),
                np.array(new long[,] { { 10 }, { 20 } }),
                new[] { 2, 3 },
                new long[] { 1, 2, 3, 1, 2, 3 },
                new long[] { 10, 10, 10, 20, 20, 20 });
        }

        /// <summary>
        ///     np.broadcast(arange(6).reshape(2,3), arange(3)).iters
        ///       iters[0] -> 0,1,2,3,4,5
        ///       iters[1] -> 0,1,2,0,1,2
        /// </summary>
        [TestMethod]
        public void Iters_Matrix2x3_PlusVector_MatchNumPy()
        {
            AssertIters(
                np.arange(6).reshape(2, 3),
                np.arange(3),
                new[] { 2, 3 },
                new long[] { 0, 1, 2, 3, 4, 5 },
                new long[] { 0, 1, 2, 0, 1, 2 });
        }

        /// <summary>
        ///     Array + 0-d scalar: shape stays (3,); the scalar fills the whole stream.
        ///       np.broadcast([1,2,3], np.int64(7)).iters[1] -> 7,7,7
        /// </summary>
        [TestMethod]
        public void Iters_ArrayPlusScalar_MatchNumPy()
        {
            AssertIters(
                np.array(new long[] { 1, 2, 3 }),
                NDArray.Scalar(7L),
                new[] { 3 },
                new long[] { 1, 2, 3 },
                new long[] { 7, 7, 7 });
        }

        /// <summary>
        ///     KEY: a transposed (F-contiguous) operand must still iterate in C-order of the RESULT,
        ///     not its own memory order.
        ///       a = arange(6).reshape(2,3).T   # shape (3,2), F-contiguous
        ///       np.broadcast(a, arange(2)).iters[0] -> 0,3,1,4,2,5   (NOT 0,1,2,3,4,5)
        /// </summary>
        [TestMethod]
        public void Iters_TransposedOperand_FollowsResultCOrder()
        {
            AssertIters(
                np.arange(6).reshape(2, 3).T,
                np.arange(2),
                new[] { 3, 2 },
                new long[] { 0, 3, 1, 4, 2, 5 },
                new long[] { 0, 1, 0, 1, 0, 1 });
        }

        /// <summary>
        ///     Reversed-slice (negative-stride) operand broadcast against a column.
        ///       a = arange(3)[::-1]            # [2,1,0]
        ///       np.broadcast(a, [[0],[10],[20]]).iters[0] -> 2,1,0,2,1,0,2,1,0
        /// </summary>
        [TestMethod]
        public void Iters_ReversedSliceOperand_MatchNumPy()
        {
            AssertIters(
                np.arange(3)["::-1"],
                np.array(new long[,] { { 0 }, { 10 }, { 20 } }),
                new[] { 3, 3 },
                new long[] { 2, 1, 0, 2, 1, 0, 2, 1, 0 },
                new long[] { 0, 0, 0, 10, 10, 10, 20, 20, 20 });
        }

        /// <summary>
        ///     Stepped-slice (stride=2) operand.
        ///       a = arange(6)[::2]             # [0,2,4]
        ///       np.broadcast(a, [[0],[10]]).iters[0] -> 0,2,4,0,2,4
        /// </summary>
        [TestMethod]
        public void Iters_SteppedSliceOperand_MatchNumPy()
        {
            AssertIters(
                np.arange(6)["::2"],
                np.array(new long[,] { { 0 }, { 10 } }),
                new[] { 2, 3 },
                new long[] { 0, 2, 4, 0, 2, 4 },
                new long[] { 0, 0, 0, 10, 10, 10 });
        }

        /// <summary>
        ///     Offset-sliced operands (non-zero Shape.offset into the buffer) must resolve the
        ///     base address correctly while broadcasting.
        ///       arange(12).reshape(3,4)[:, 1:2]  # (3,1) at offset -> [[1],[5],[9]]
        ///       arange(12).reshape(3,4)[1:3, :]  # (2,4) at offset -> rows 1..2
        /// </summary>
        [TestMethod]
        public void Iters_OffsetSlicedOperands_MatchNumPy()
        {
            // Column slice -> (3,1) view, broadcast against (4,) -> (3,4)
            AssertIters(
                np.arange(12).reshape(3, 4)[":, 1:2"],
                np.arange(4),
                new[] { 3, 4 },
                new long[] { 1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9 },
                new long[] { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 });

            // Row slice (non-zero offset) -> (2,4) view, broadcast against (1,4) -> (2,4)
            AssertIters(
                np.arange(12).reshape(3, 4)["1:3, :"],
                np.arange(4).reshape(1, 4),
                new[] { 2, 4 },
                new long[] { 4, 5, 6, 7, 8, 9, 10, 11 },
                new long[] { 0, 1, 2, 3, 0, 1, 2, 3 });
        }

        /// <summary>
        ///     A spread of broadcast shape pairs (arange-filled) — every iters sequence is
        ///     verbatim NumPy 2.4.2 output.
        /// </summary>
        [TestMethod]
        public void Iters_BreadthShapes_MatchNumPy()
        {
            // (3,) & (2,1) -> (2,3)
            AssertIters(np.arange(3), np.arange(2).reshape(2, 1), new[] { 2, 3 },
                new long[] { 0, 1, 2, 0, 1, 2 }, new long[] { 0, 0, 0, 1, 1, 1 });

            // (4,) & (3,1) -> (3,4)
            AssertIters(np.arange(4), np.arange(3).reshape(3, 1), new[] { 3, 4 },
                new long[] { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 },
                new long[] { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2 });

            // (2,1,3) & (4,3) -> (2,4,3)
            AssertIters(np.arange(6).reshape(2, 1, 3), np.arange(12).reshape(4, 3), new[] { 2, 4, 3 },
                new long[] { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5 },
                new long[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });

            // (5,1) & (1,4) -> (5,4)
            AssertIters(np.arange(5).reshape(5, 1), np.arange(4).reshape(1, 4), new[] { 5, 4 },
                new long[] { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 },
                new long[] { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 });

            // (1,) & (5,) -> (5,)
            AssertIters(np.arange(1), np.arange(5), new[] { 5 },
                new long[] { 0, 0, 0, 0, 0 }, new long[] { 0, 1, 2, 3, 4 });

            // (2,3) & (2,3) -> (2,3) (no stretch)
            AssertIters(np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3), new[] { 2, 3 },
                new long[] { 0, 1, 2, 3, 4, 5 }, new long[] { 0, 1, 2, 3, 4, 5 });

            // (1,3) & (3,1) -> (3,3)
            AssertIters(np.arange(3).reshape(1, 3), np.arange(3).reshape(3, 1), new[] { 3, 3 },
                new long[] { 0, 1, 2, 0, 1, 2, 0, 1, 2 }, new long[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 });

            // (6,) & (1,) -> (6,)
            AssertIters(np.arange(6), np.arange(1), new[] { 6 },
                new long[] { 0, 1, 2, 3, 4, 5 }, new long[] { 0, 0, 0, 0, 0, 0 });
        }

        // ================================================================
        // Edge cases: high-dim, empty, 0-d
        // ================================================================

        /// <summary>
        ///     4-D broadcast (8,1,6,1) x (7,1,5) -> (8,7,6,5); size 1680, both iters full of 1.0.
        /// </summary>
        [TestMethod]
        public void Iters_HighDimensional_4D()
        {
            var bc = np.broadcast(np.ones(new Shape(8, 1, 6, 1)), np.ones(new Shape(7, 1, 5)));

            bc.shape.dimensions.Should().Equal(new long[] { 8, 7, 6, 5 });
            bc.ndim.Should().Be(4);
            bc.size.Should().Be(1680);

            var i0 = Iter<double>(bc, 0);
            var i1 = Iter<double>(bc, 1);
            i0.Should().HaveCount(1680).And.OnlyContain(x => x == 1.0);
            i1.Should().HaveCount(1680).And.OnlyContain(x => x == 1.0);
        }

        /// <summary>
        ///     Zero-size broadcast: (0,3) x (1,3) -> (0,3); size 0; both iters yield nothing.
        ///       np.broadcast(np.ones((0,3)), np.ones((1,3))).size == 0
        /// </summary>
        [TestMethod]
        public void Iters_Empty_ZeroSize()
        {
            var bc = np.broadcast(np.ones(new Shape(0, 3)), np.ones(new Shape(1, 3)));

            bc.shape.dimensions.Should().Equal(new long[] { 0, 3 });
            bc.size.Should().Be(0);
            bc.iters[0].size.Should().Be(0);
            Iter<double>(bc, 0).Should().BeEmpty();
            Iter<double>(bc, 1).Should().BeEmpty();
        }

        /// <summary>
        ///     0-d x 0-d -> 0-d; ndim 0, size 1; each iters yields exactly one element.
        ///       np.broadcast(np.int64(5), np.int64(7)).iters[0] -> [5], iters[1] -> [7]
        /// </summary>
        [TestMethod]
        public void Iters_ZeroDimensional_Scalars()
        {
            var bc = np.broadcast(NDArray.Scalar(5L), NDArray.Scalar(7L));

            bc.ndim.Should().Be(0);
            bc.size.Should().Be(1);
            Iter<long>(bc, 0).Should().Equal(new long[] { 5 });
            Iter<long>(bc, 1).Should().Equal(new long[] { 7 });
        }

        // ================================================================
        // All 15 dtypes — iters yield correctly-typed, correctly-cycled values
        // ================================================================

        [TestMethod] public void Iters_Dtype_Boolean() => AssertCycle<bool>(np.array(new bool[] { true, false, true }), true, false, true);
        [TestMethod] public void Iters_Dtype_Byte() => AssertCycle<byte>(np.array(new byte[] { 1, 2, 3 }), 1, 2, 3);
        [TestMethod] public void Iters_Dtype_SByte() => AssertCycle<sbyte>(np.array(new sbyte[] { -1, 2, -3 }), -1, 2, -3);
        [TestMethod] public void Iters_Dtype_Int16() => AssertCycle<short>(np.array(new short[] { 10, 20, 30 }), 10, 20, 30);
        [TestMethod] public void Iters_Dtype_UInt16() => AssertCycle<ushort>(np.array(new ushort[] { 10, 20, 30 }), 10, 20, 30);
        [TestMethod] public void Iters_Dtype_Int32() => AssertCycle<int>(np.array(new int[] { -100, 0, 100 }), -100, 0, 100);
        [TestMethod] public void Iters_Dtype_UInt32() => AssertCycle<uint>(np.array(new uint[] { 1, 2, 3 }), 1, 2, 3);
        [TestMethod] public void Iters_Dtype_Int64() => AssertCycle<long>(np.array(new long[] { 1, 2, 3 }), 1, 2, 3);
        [TestMethod] public void Iters_Dtype_UInt64() => AssertCycle<ulong>(np.array(new ulong[] { 1, 2, 3 }), 1, 2, 3);
        [TestMethod] public void Iters_Dtype_Char() => AssertCycle<char>(np.array(new char[] { 'a', 'b', 'c' }), 'a', 'b', 'c');
        [TestMethod] public void Iters_Dtype_Half() => AssertCycle<Half>(np.array(new Half[] { (Half)1, (Half)2, (Half)3 }), (Half)1, (Half)2, (Half)3);
        [TestMethod] public void Iters_Dtype_Single() => AssertCycle<float>(np.array(new float[] { 1.5f, 2.5f, 3.5f }), 1.5f, 2.5f, 3.5f);
        [TestMethod] public void Iters_Dtype_Double() => AssertCycle<double>(np.array(new double[] { 1.5, 2.5, 3.5 }), 1.5, 2.5, 3.5);
        [TestMethod] public void Iters_Dtype_Decimal() => AssertCycle<decimal>(np.array(new decimal[] { 1.5m, 2.5m, 3.5m }), 1.5m, 2.5m, 3.5m);
        [TestMethod] public void Iters_Dtype_Complex() => AssertCycle<Complex>(np.array(new Complex[] { new(1, 2), new(3, 4), new(5, 6) }), new(1, 2), new(3, 4), new(5, 6));

        // ================================================================
        // Object / iterator surface
        // ================================================================

        /// <summary>NDFlatIterator.size and the broadcast size agree; iters.Length == numiter == 2.</summary>
        [TestMethod]
        public void Iters_Size_And_Count()
        {
            var bc = np.broadcast(np.arange(3), np.ones(new Shape(2, 3)));

            bc.size.Should().Be(6);
            bc.numiter.Should().Be(2);
            bc.iters.Length.Should().Be(2);
            bc.iters[0].size.Should().Be(6);
            bc.iters[1].size.Should().Be(6);
        }

        /// <summary>shape / ndim / nd / size / numiter all match NumPy for a broadcasting pair.</summary>
        [TestMethod]
        public void Properties_MatchNumPy()
        {
            var bc = np.broadcast(np.ones(new Shape(2, 1)), np.ones(new Shape(1, 3)));

            bc.shape.dimensions.Should().Equal(new long[] { 2, 3 });
            bc.ndim.Should().Be(2);
            bc.nd.Should().Be(2);
            bc.size.Should().Be(6);
            bc.numiter.Should().Be(2);
        }

        // ================================================================
        // N operands (NumPy accepts 0..64; NumSharp now matches)
        // ================================================================

        /// <summary>
        ///     Single operand: np.broadcast(arange(3)) -> shape (3,), numiter 1, one iterator.
        /// </summary>
        [TestMethod]
        public void Iters_SingleOperand_MatchNumPy()
        {
            AssertItersN(np.broadcast(np.arange(3)), new[] { 3 },
                new long[] { 0, 1, 2 });
        }

        /// <summary>
        ///     Single transposed operand still iterates in C-order of its (own) shape.
        ///       np.broadcast(arange(6).reshape(2,3).T).iters[0] -> 0,3,1,4,2,5
        /// </summary>
        [TestMethod]
        public void Iters_SingleTransposedOperand_COrder()
        {
            AssertItersN(np.broadcast(np.arange(6).reshape(2, 3).T), new[] { 3, 2 },
                new long[] { 0, 3, 1, 4, 2, 5 });
        }

        /// <summary>
        ///     Three operands: np.broadcast(arange(3), [[0],[10]], 100) -> (2,3), numiter 3.
        ///       iters[0] -> 0,1,2,0,1,2 ; iters[1] -> 0,0,0,10,10,10 ; iters[2] -> 100 x6
        /// </summary>
        [TestMethod]
        public void Iters_ThreeOperands_MatchNumPy()
        {
            AssertItersN(
                np.broadcast(np.arange(3), np.array(new long[,] { { 0 }, { 10 } }), NDArray.Scalar(100L)),
                new[] { 2, 3 },
                new long[] { 0, 1, 2, 0, 1, 2 },
                new long[] { 0, 0, 0, 10, 10, 10 },
                new long[] { 100, 100, 100, 100, 100, 100 });
        }

        /// <summary>
        ///     Three operands with prepended dims (each operand stretches a different axis).
        ///       np.broadcast([[0],[1]], arange(3), [[100],[200]]) -> (2,3), numiter 3
        /// </summary>
        [TestMethod]
        public void Iters_ThreeOperands_PrependDims_MatchNumPy()
        {
            AssertItersN(
                np.broadcast(np.arange(2).reshape(2, 1), np.arange(3), np.array(new long[,] { { 100 }, { 200 } })),
                new[] { 2, 3 },
                new long[] { 0, 0, 0, 1, 1, 1 },
                new long[] { 0, 1, 2, 0, 1, 2 },
                new long[] { 100, 100, 100, 200, 200, 200 });
        }

        /// <summary>
        ///     Four operands of differing ndim broadcast to a common shape (NumPy docs example).
        ///       np.broadcast(ones(6,7), ones(5,6,1), ones(7), ones(5,1,7)) -> (5,6,7), numiter 4
        /// </summary>
        [TestMethod]
        public void Iters_FourOperands_Shape()
        {
            var bc = np.broadcast(
                np.ones(new Shape(6, 7)),
                np.ones(new Shape(5, 6, 1)),
                np.ones(new Shape(7)),
                np.ones(new Shape(5, 1, 7)));

            bc.shape.dimensions.Should().Equal(new long[] { 5, 6, 7 });
            bc.ndim.Should().Be(3);
            bc.size.Should().Be(210);
            bc.numiter.Should().Be(4);
            bc.iters.Length.Should().Be(4);
            foreach (var it in bc.iters)
            {
                it.size.Should().Be(210);
                it.Cast<double>().ToArray().Should().HaveCount(210).And.OnlyContain(x => x == 1.0);
            }
        }

        /// <summary>
        ///     Zero operands: np.broadcast() -> 0-d (shape ()), size 1, numiter 0, empty iters.
        /// </summary>
        [TestMethod]
        public void Iters_ZeroOperands_ScalarBroadcast()
        {
            var bc = np.broadcast();

            bc.ndim.Should().Be(0);
            bc.size.Should().Be(1);
            bc.numiter.Should().Be(0);
            bc.iters.Should().NotBeNull();
            bc.iters.Length.Should().Be(0);
        }

        /// <summary>
        ///     DIVERGENCE: NumPy caps np.broadcast at NPY_MAXARGS=64 (65+ -> ValueError). NumSharp's
        ///     NDIter allocates per-operand state dynamically and imposes no cap, so np.broadcast
        ///     accepts any number of operands.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Broadcast_ManyOperands_NoCap_MatchesNDIter()
        {
            // 65 operands — one past NumPy's NPY_MAXARGS=64 — is accepted, not rejected.
            var sixtyFive = Enumerable.Range(0, 65).Select(_ => np.ones(new Shape(1))).ToArray();
            var b65 = np.broadcast(sixtyFive);
            b65.numiter.Should().Be(65);
            b65.iters.Length.Should().Be(65);
            b65.shape.dimensions.Should().Equal(new long[] { 1 });

            // Thousands of operands (NDIter allocates per-operand state dynamically, so there is
            // no cap) — every operand still broadcasts to its own iters stream correctly.
            var many = Enumerable.Range(0, 1000).Select(i => np.array(new long[] { i })).ToArray();
            var bm = np.broadcast(many);
            bm.numiter.Should().Be(1000);
            bm.iters.Length.Should().Be(1000);
            bm.size.Should().Be(1);
            bm.iters[0].Cast<long>().Single().Should().Be(0L);
            bm.iters[999].Cast<long>().Single().Should().Be(999L);

            // N operands that genuinely broadcast: 1000 x (1,) against (4,) -> (4,). Shape
            // resolution and the live cursor both scale — each step yields a numiter-wide tuple.
            var stretch = Enumerable.Range(0, 1000).Select(_ => np.ones(new Shape(1)))
                                    .Append(np.arange(4)).ToArray();
            var bs = np.broadcast(stretch);
            bs.numiter.Should().Be(1001);
            bs.shape.dimensions.Should().Equal(new long[] { 4 });

            int steps = 0, width = -1;
            foreach (var v in bs)
            {
                if (width < 0) width = v.Length;
                steps++;
            }
            steps.Should().Be(4);
            width.Should().Be(1001, "each iteration tuple has one value per operand (numiter)");
            bs.index.Should().Be(4);
        }

        /// <summary>
        ///     Three-way incompatible shapes raise (mismatch surfaces while folding the operands).
        ///       np.broadcast(ones(3), ones(3), ones(4)) -> ValueError/shape mismatch
        /// </summary>
        [TestMethod]
        public void Broadcast_ThreeWayIncompatible_Throws()
        {
            new Action(() => np.broadcast(np.ones(new Shape(3)), np.ones(new Shape(3)), np.ones(new Shape(4))))
                .Should().Throw<IncorrectShapeException>("(3,) and (4,) cannot broadcast");
        }

        // ================================================================
        // Documented divergences from NumPy (captured, not "bugs")
        // ================================================================

        /// <summary>
        ///     DIVERGENCE: NumSharp's iters are re-enumerable (idiomatic IEnumerable) — iterating
        ///     the same NDFlatIterator twice yields the same sequence. NumPy's flatiter is a
        ///     one-shot C iterator that exhausts after the first pass (second pass is empty).
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Iters_AreReEnumerable_UnlikeNumPyOneShot()
        {
            var bc = np.broadcast(np.array(new long[] { 1, 2, 3 }), np.array(new long[,] { { 10 }, { 20 } }));
            var it = bc.iters[0];

            var first = it.Cast<long>().ToArray();
            var second = it.Cast<long>().ToArray();

            first.Should().Equal(new long[] { 1, 2, 3, 1, 2, 3 });
            second.Should().Equal(first,
                "NDFlatIterator is re-enumerable; NumPy's flatiter would yield an empty second pass");
        }

        /// <summary>
        ///     DIVERGENCE: reading <c>iters</c> does not move the broadcast's live cursor, and the
        ///     flatiters always start at 0 — whereas NumPy's flatiters share the cursor (so after two
        ///     next() calls NumPy's iters[0] would start at index 2 → [3,1,2,3]).
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Iters_AreIndependentOfCursor_UnlikeNumPy()
        {
            var bc = np.broadcast(np.array(new long[] { 1, 2, 3 }), np.array(new long[,] { { 10 }, { 20 } }));
            bc.MoveNext();
            bc.MoveNext();
            bc.index.Should().Be(2);

            bc.iters[0].Cast<long>().ToArray().Should().Equal(new long[] { 1, 2, 3, 1, 2, 3 },
                "NumSharp iters are independent of the cursor; NumPy's would start at index 2 -> [3,1,2,3]");
            bc.index.Should().Be(2, "reading iters must not move the broadcast cursor");
        }

        // ================================================================
        // Live cursor: index / iteration / reset (NumPy parity)
        // ================================================================

        /// <summary>broadcast.index starts at 0 (was previously a property that threw).</summary>
        [TestMethod]
        public void Index_StartsAtZero()
        {
            np.broadcast(np.arange(3), np.arange(3)).index.Should().Be(0);
        }

        /// <summary>
        ///     Iterating the broadcast yields one per-operand value tuple per step (C-order) and
        ///     advances index to size — np.broadcast([1,2,3],[[10],[20]]) -> (1,10),(2,10),(3,10),
        ///     (1,20),(2,20),(3,20).
        /// </summary>
        [TestMethod]
        public void Iteration_YieldsTuples_AndAdvancesIndexToSize()
        {
            var bc = np.broadcast(np.array(new long[] { 1, 2, 3 }), np.array(new long[,] { { 10 }, { 20 } }));

            var col0 = new List<long>();
            var col1 = new List<long>();
            foreach (var v in bc)
            {
                v.Length.Should().Be(2, "one value per operand (numiter)");
                col0.Add((long)v[0]);
                col1.Add((long)v[1]);
            }

            col0.Should().Equal(1L, 2, 3, 1, 2, 3);
            col1.Should().Equal(10L, 10, 10, 20, 20, 20);
            bc.index.Should().Be(6, "index == size once exhausted");
        }

        /// <summary>MoveNext advances index by one and exposes the step's values via Current.</summary>
        [TestMethod]
        public void MoveNext_AdvancesIndex_AndExposesCurrent()
        {
            var bc = np.broadcast(np.array(new long[] { 1, 2, 3 }), np.array(new long[,] { { 10 }, { 20 } }));

            bc.index.Should().Be(0);
            bc.MoveNext().Should().BeTrue();
            bc.index.Should().Be(1);
            bc.Current.Length.Should().Be(2);
            ((long)bc.Current[0]).Should().Be(1);
            ((long)bc.Current[1]).Should().Be(10);

            bc.MoveNext().Should().BeTrue();
            bc.index.Should().Be(2);
            ((long)bc.Current[0]).Should().Be(2);
            ((long)bc.Current[1]).Should().Be(10);
        }

        /// <summary>reset() rewinds the cursor to 0 and allows the broadcast to be iterated again.</summary>
        [TestMethod]
        public void Reset_RewindsCursor_AllowsReiteration()
        {
            var bc = np.broadcast(np.array(new long[] { 1, 2, 3 }), np.array(new long[,] { { 10 }, { 20 } }));

            foreach (var _ in bc) { }
            bc.index.Should().Be(6);
            bc.MoveNext().Should().BeFalse("cursor is exhausted");

            bc.reset();
            bc.index.Should().Be(0);
            bc.MoveNext().Should().BeTrue();
            ((long)bc.Current[0]).Should().Be(1);
            bc.index.Should().Be(1);
        }

        /// <summary>Single-operand iteration yields 1-tuples: np.broadcast([1,2,3]) -> (1,),(2,),(3,).</summary>
        [TestMethod]
        public void Iteration_SingleOperand_MatchNumPy()
        {
            var bc = np.broadcast(np.array(new long[] { 1, 2, 3 }));

            var col0 = new List<long>();
            foreach (var v in bc)
            {
                v.Length.Should().Be(1);
                col0.Add((long)v[0]);
            }
            col0.Should().Equal(1L, 2, 3);
            bc.index.Should().Be(3);
        }

        /// <summary>Three-operand iteration yields 3-tuples (NumPy parity).</summary>
        [TestMethod]
        public void Iteration_ThreeOperands_MatchNumPy()
        {
            var bc = np.broadcast(
                np.array(new long[] { 1, 2, 3 }),
                np.array(new long[,] { { 10 }, { 20 } }),
                NDArray.Scalar(100L));

            var c0 = new List<long>();
            var c1 = new List<long>();
            var c2 = new List<long>();
            foreach (var v in bc)
            {
                v.Length.Should().Be(3);
                c0.Add((long)v[0]);
                c1.Add((long)v[1]);
                c2.Add((long)v[2]);
            }
            c0.Should().Equal(1L, 2, 3, 1, 2, 3);
            c1.Should().Equal(10L, 10, 10, 20, 20, 20);
            c2.Should().Equal(100L, 100, 100, 100, 100, 100);
            bc.index.Should().Be(6);
        }

        /// <summary>
        ///     Zero operands iterate exactly once, yielding an empty tuple (NumPy: list(np.broadcast())
        ///     == [()] with index 1).
        /// </summary>
        [TestMethod]
        public void Iteration_ZeroOperands_OneEmptyTuple_MatchNumPy()
        {
            var bc = np.broadcast();

            var items = new List<object[]>();
            foreach (var v in bc)
                items.Add(v);

            items.Should().HaveCount(1, "NumPy: list(np.broadcast()) == [()]");
            items[0].Should().BeEmpty();
            bc.index.Should().Be(1);
        }

        /// <summary>The k-th iteration tuple's i-th value equals iters[i][k] (cursor and iters agree on values).</summary>
        [TestMethod]
        public void Iteration_TuplesAlignWithIters()
        {
            var bc = np.broadcast(np.arange(3), np.array(new long[,] { { 10 }, { 20 } }));

            var i0 = bc.iters[0].Cast<long>().ToArray();
            var i1 = bc.iters[1].Cast<long>().ToArray();

            int k = 0;
            foreach (var v in bc)
            {
                ((long)v[0]).Should().Be(i0[k]);
                ((long)v[1]).Should().Be(i1[k]);
                k++;
            }
            k.Should().Be(6);
        }
    }
}
