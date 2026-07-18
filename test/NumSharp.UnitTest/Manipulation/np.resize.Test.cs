using System;
using System.Numerics;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Battle tests for <c>np.resize</c> (function — repeats copies) and
    /// <c>ndarray.resize</c> (in-place method — zero fills). All expected values
    /// verified against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class ResizeTests
    {
        private static long[] L(NDArray a)
        {
            var r = new long[a.size];
            for (int i = 0; i < a.size; i++) r[i] = Convert.ToInt64(a.GetAtIndex(i));
            return r;
        }

        // ==================================================================
        // np.resize (function) — fills with REPEATED copies of a (C-order)
        // ==================================================================

        [TestMethod]
        public void Resize_Func_DocExample_2x3()
        {
            // np.resize([[0,1],[2,3]], (2,3)) -> [[0,1,2],[3,0,1]]
            var a = np.array(new int[,] { { 0, 1 }, { 2, 3 } });
            var r = np.resize(a, (2, 3));
            r.shape.Should().Equal(2L, 3L);
            L(r).Should().Equal(0, 1, 2, 3, 0, 1);
        }

        [TestMethod]
        public void Resize_Func_DocExample_1x4()
        {
            var a = np.array(new int[,] { { 0, 1 }, { 2, 3 } });
            var r = np.resize(a, (1, 4));
            r.shape.Should().Equal(1L, 4L);
            L(r).Should().Equal(0, 1, 2, 3);
        }

        [TestMethod]
        public void Resize_Func_DocExample_2x4_Repeats()
        {
            var a = np.array(new int[,] { { 0, 1 }, { 2, 3 } });
            var r = np.resize(a, (2, 4));
            r.shape.Should().Equal(2L, 4L);
            L(r).Should().Equal(0, 1, 2, 3, 0, 1, 2, 3);
        }

        [TestMethod]
        public void Resize_Func_IntShape_Truncates()
        {
            var a = np.array(new int[,] { { 0, 1 }, { 2, 3 } });
            var r = np.resize(a, 3);
            r.shape.Should().Equal(3L);
            L(r).Should().Equal(0, 1, 2);
        }

        [TestMethod]
        public void Resize_Func_Grow_Tiles()
        {
            // np.resize([1,2,3], (3,3)) -> rows all [1,2,3]
            var r = np.resize(np.array(new[] { 1, 2, 3 }), (3, 3));
            r.shape.Should().Equal(3L, 3L);
            L(r).Should().Equal(1, 2, 3, 1, 2, 3, 1, 2, 3);
        }

        [TestMethod]
        public void Resize_Func_NonMultiple_Tiles()
        {
            // np.resize(arange(7), (11,)) cycles: 0..6,0..3
            var r = np.resize(np.arange(7), 11);
            r.shape.Should().Equal(11L);
            L(r).Should().Equal(0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3);
        }

        [TestMethod]
        public void Resize_Func_ScalarInput_Repeats()
        {
            var r = np.resize(np.array(5), (2, 2));
            r.shape.Should().Equal(2L, 2L);
            L(r).Should().Equal(5, 5, 5, 5);
        }

        [TestMethod]
        public void Resize_Func_EmptySource_ZeroFills()
        {
            // np.resize([], (2,2)) -> zeros
            var r = np.resize(np.array(new int[] { }), (2, 2));
            r.shape.Should().Equal(2L, 2L);
            L(r).Should().Equal(0, 0, 0, 0);
        }

        [TestMethod]
        public void Resize_Func_NewSizeZero_ReturnsEmptyShaped()
        {
            var r = np.resize(np.array(new[] { 1, 2, 3 }), 0);
            r.shape.Should().Equal(0L);
            r.size.Should().Be(0);

            var r2 = np.resize(np.array(new[] { 1, 2, 3 }), (2, 0, 3));
            r2.shape.Should().Equal(2L, 0L, 3L);
            r2.size.Should().Be(0);
        }

        [TestMethod]
        public void Resize_Func_EmptyShape_ReturnsScalarFirstElement()
        {
            // np.resize(a, ()) -> array(0) (0-d, first element)
            var r = np.resize(np.arange(6), new Shape(new long[0]));
            r.ndim.Should().Be(0);
            Convert.ToInt64(r.GetAtIndex(0)).Should().Be(0);
        }

        [TestMethod]
        public void Resize_Func_PreservesDtype()
        {
            var r = np.resize(np.array(new float[] { 1.5f, 2.5f }), (2, 3));
            r.dtype.Should().Be(typeof(float));
            r.shape.Should().Equal(2L, 3L);
            Convert.ToDouble(r.GetAtIndex(0)).Should().Be(1.5);
            Convert.ToDouble(r.GetAtIndex(5)).Should().Be(2.5);
        }

        [TestMethod]
        public void Resize_Func_NegativeDimension_Throws()
        {
            Action act = () => np.resize(np.array(new[] { 1, 2, 3 }), (-1, 2));
            act.Should().Throw<Exception>().WithMessage("*non-negative*");
        }

        [TestMethod]
        public void Resize_Func_NullArray_Throws()
        {
            Action act = () => np.resize((NDArray)null, (2, 2));
            act.Should().Throw<ArgumentNullException>();
        }

        // ---- np.resize on non-contiguous inputs (ravel materializes C-order) ----

        [TestMethod]
        public void Resize_Func_Transposed_ReadsCOrder()
        {
            // arange(6).reshape(2,3).T ravels C-order to [0,3,1,4,2,5]
            var r = np.resize(np.arange(6).reshape(2, 3).T, (2, 4));
            r.shape.Should().Equal(2L, 4L);
            L(r).Should().Equal(0, 3, 1, 4, 2, 5, 0, 3);
        }

        [TestMethod]
        public void Resize_Func_Strided()
        {
            var r = np.resize(np.arange(20)["::2"], (2, 3));
            r.shape.Should().Equal(2L, 3L);
            L(r).Should().Equal(0, 2, 4, 6, 8, 10);
        }

        [TestMethod]
        public void Resize_Func_NegativeStride()
        {
            var r = np.resize(np.arange(6)["::-1"], (2, 4));
            r.shape.Should().Equal(2L, 4L);
            L(r).Should().Equal(5, 4, 3, 2, 1, 0, 5, 4);
        }

        [TestMethod]
        public void Resize_Func_Broadcast()
        {
            var r = np.resize(np.broadcast_to(np.arange(3), new Shape(2, 3)), (2, 5));
            r.shape.Should().Equal(2L, 5L);
            L(r).Should().Equal(0, 1, 2, 0, 1, 2, 0, 1, 2, 0);
        }

        [TestMethod]
        public void Resize_Func_DoesNotMutateSource()
        {
            var a = np.arange(4);
            var r = np.resize(a, (2, 4));
            a.shape.Should().Equal(4L);
            L(a).Should().Equal(0, 1, 2, 3);
        }

        // ==================================================================
        // ndarray.resize (in-place) — fills with ZEROS when growing
        // ==================================================================

        [TestMethod]
        public void Resize_Method_Grow_ZeroFills()
        {
            var a = np.array(new[] { 1, 2, 3 });
            a.resize(2, 3);
            a.shape.Should().Equal(2L, 3L);
            L(a).Should().Equal(1, 2, 3, 0, 0, 0);
        }

        [TestMethod]
        public void Resize_Method_Shrink_Truncates()
        {
            var a = np.arange(1, 7);
            a.resize(2, 2);
            a.shape.Should().Equal(2L, 2L);
            L(a).Should().Equal(1, 2, 3, 4);
        }

        [TestMethod]
        public void Resize_Method_SameSize_Reshapes()
        {
            var a = np.arange(6);
            a.resize(2, 3);
            a.shape.Should().Equal(2L, 3L);
            L(a).Should().Equal(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Resize_Method_Grow_3to4()
        {
            var a = np.array(new[] { 1, 2, 3 });
            a.resize(2, 2);
            a.shape.Should().Equal(2L, 2L);
            L(a).Should().Equal(1, 2, 3, 0);
        }

        [TestMethod]
        public void Resize_Method_ScalarInput_ZeroFills()
        {
            var a = np.array(5);
            a.resize(2, 2);
            a.shape.Should().Equal(2L, 2L);
            L(a).Should().Equal(5, 0, 0, 0);
        }

        [TestMethod]
        public void Resize_Method_To3D()
        {
            var a = np.array(new[] { 1, 2, 3 });
            a.resize(2, 2, 2);
            a.shape.Should().Equal(2L, 2L, 2L);
            L(a).Should().Equal(1, 2, 3, 0, 0, 0, 0, 0);
        }

        [TestMethod]
        public void Resize_Method_ToZero()
        {
            var a = np.array(new[] { 1, 2, 3 });
            a.resize(0);
            a.shape.Should().Equal(0L);
            a.size.Should().Be(0);
        }

        [TestMethod]
        public void Resize_Method_NoArgs_IsNoOp()
        {
            var a = np.array(new[] { 1, 2, 3 });
            a.resize();
            a.shape.Should().Equal(3L);
            L(a).Should().Equal(1, 2, 3);
        }

        [TestMethod]
        public void Resize_Method_GrowLarge_TailIsZero()
        {
            var a = np.arange(10);
            a.resize(100);
            a.size.Should().Be(100);
            for (int i = 0; i < 10; i++) Convert.ToInt64(a.GetAtIndex(i)).Should().Be(i);
            for (int i = 10; i < 100; i++) Convert.ToInt64(a.GetAtIndex(i)).Should().Be(0);
        }

        [TestMethod]
        public void Resize_Method_ReturnsNothing_MutatesInPlace()
        {
            var a = np.arange(3);
            var before = a; // same reference
            a.resize(5);
            ReferenceEquals(a, before).Should().BeTrue();
            a.size.Should().Be(5);
        }

        // ---- F-contiguous: resize operates on the raw memory buffer ----

        [TestMethod]
        public void Resize_Method_FContiguous_Grow_RelabelsMemory()
        {
            // asfortranarray([[0,1,2],[3,4,5]]) memory = [0,3,1,4,2,5]; grow to (3,3)
            // -> [0,3,1,4,2,5,0,0,0] read F-order -> [[0,4,0],[3,2,0],[1,5,0]]
            var a = np.asfortranarray(np.arange(6).reshape(2, 3));
            a.resize(3, 3);
            a.Shape.IsFContiguous.Should().BeTrue();
            // logical [[0,4,0],[3,2,0],[1,5,0]] read in C-order
            L(a).Should().Equal(0, 4, 0, 3, 2, 0, 1, 5, 0);
        }

        [TestMethod]
        public void Resize_Method_FContiguous_SameSize_RelabelsMemory()
        {
            // memory [0,3,1,4,2,5] relabeled to (3,2) F-order -> [[0,4],[3,2],[1,5]]
            var a = np.asfortranarray(np.arange(6).reshape(2, 3));
            a.resize(3, 2);
            a.Shape.IsFContiguous.Should().BeTrue();
            L(a).Should().Equal(0, 4, 3, 2, 1, 5);
        }

        // ==================================================================
        // ndarray.resize guards (all match NumPy 2.4.2 ValueError messages)
        // ==================================================================

        [TestMethod]
        public void Resize_Method_NonContiguous_Throws()
        {
            var nc = np.arange(12).reshape(3, 4)[":, ::2"]; // non-contiguous view
            Action act = () => nc.resize(3, 2);
            act.Should().Throw<IncorrectShapeException>().WithMessage("*single-segment*");
        }

        [TestMethod]
        public void Resize_Method_View_DoesNotOwnData_Throws()
        {
            var v = np.arange(6).reshape(2, 3); // reshape view
            Action act = () => v.resize(3, 3); // size change on non-owning view
            act.Should().Throw<IncorrectShapeException>().WithMessage("*does not own its data*");
        }

        [TestMethod]
        public void Resize_Method_Referenced_Throws()
        {
            var b = np.arange(6);
            var view = b["::"]; // b now referenced by view
            Action act = () => b.resize(3, 3);
            act.Should().Throw<IncorrectShapeException>().WithMessage("*references or is referenced*");
            GC.KeepAlive(view);
        }

        [TestMethod]
        public void Resize_Method_RefcheckFalse_Bypasses()
        {
            var b = np.arange(6);
            var view = b["::"];
            b.resize(new Shape(new long[] { 3, 3 }), refcheck: false);
            b.size.Should().Be(9);
            GC.KeepAlive(view);
        }

        [TestMethod]
        public void Resize_Method_NegativeDimension_Throws()
        {
            var b = np.arange(6);
            Action act = () => b.resize(-1, 2);
            act.Should().Throw<IncorrectShapeException>().WithMessage("*negative dimensions not allowed*");
        }

        [TestMethod]
        public void Resize_Method_ZeroThenNegative_NoError()
        {
            // NumPy stops at the first zero dim, so (0,-1) is accepted.
            var b = np.arange(6);
            b.resize(new Shape(new long[] { 0, -1 }));
            b.shape[0].Should().Be(0);
            b.shape[1].Should().Be(-1);
        }

        [TestMethod]
        public void Resize_Method_SameSizeView_AllowedNoOwnershipCheck()
        {
            // Same total size skips the ownership check — a view reshapes in place.
            var v = np.arange(6).reshape(2, 3);
            v.resize(3, 2);
            v.shape.Should().Equal(3L, 2L);
            L(v).Should().Equal(0, 1, 2, 3, 4, 5);
        }

        // ==================================================================
        // Dtype coverage — all 15 NumSharp types
        // ==================================================================

        [TestMethod]
        public void Resize_Func_AllDtypes_TileAndDtypePreserved()
        {
            AssertFuncDtype(np.array(new bool[] { true, false, true }), typeof(bool));
            AssertFuncDtype(np.array(new byte[] { 1, 2, 3 }), typeof(byte));
            AssertFuncDtype(np.array(new sbyte[] { 1, 2, 3 }), typeof(sbyte));
            AssertFuncDtype(np.array(new short[] { 1, 2, 3 }), typeof(short));
            AssertFuncDtype(np.array(new ushort[] { 1, 2, 3 }), typeof(ushort));
            AssertFuncDtype(np.array(new int[] { 1, 2, 3 }), typeof(int));
            AssertFuncDtype(np.array(new uint[] { 1, 2, 3 }), typeof(uint));
            AssertFuncDtype(np.array(new long[] { 1, 2, 3 }), typeof(long));
            AssertFuncDtype(np.array(new ulong[] { 1, 2, 3 }), typeof(ulong));
            AssertFuncDtype(np.array(new char[] { 'a', 'b', 'c' }), typeof(char));
            AssertFuncDtype(np.array(new Half[] { (Half)1, (Half)2, (Half)3 }), typeof(Half));
            AssertFuncDtype(np.array(new float[] { 1, 2, 3 }), typeof(float));
            AssertFuncDtype(np.array(new double[] { 1, 2, 3 }), typeof(double));
            AssertFuncDtype(np.array(new decimal[] { 1, 2, 3 }), typeof(decimal));
            AssertFuncDtype(np.array(new Complex[] { 1, 2, 3 }), typeof(Complex));
        }

        [TestMethod]
        public void Resize_Method_AllDtypes_GrowZeroFillAndDtypePreserved()
        {
            AssertMethodDtype(() => np.array(new bool[] { true, false, true }), typeof(bool));
            AssertMethodDtype(() => np.array(new byte[] { 1, 2, 3 }), typeof(byte));
            AssertMethodDtype(() => np.array(new sbyte[] { 1, 2, 3 }), typeof(sbyte));
            AssertMethodDtype(() => np.array(new short[] { 1, 2, 3 }), typeof(short));
            AssertMethodDtype(() => np.array(new ushort[] { 1, 2, 3 }), typeof(ushort));
            AssertMethodDtype(() => np.array(new int[] { 1, 2, 3 }), typeof(int));
            AssertMethodDtype(() => np.array(new uint[] { 1, 2, 3 }), typeof(uint));
            AssertMethodDtype(() => np.array(new long[] { 1, 2, 3 }), typeof(long));
            AssertMethodDtype(() => np.array(new ulong[] { 1, 2, 3 }), typeof(ulong));
            AssertMethodDtype(() => np.array(new char[] { 'a', 'b', 'c' }), typeof(char));
            AssertMethodDtype(() => np.array(new Half[] { (Half)1, (Half)2, (Half)3 }), typeof(Half));
            AssertMethodDtype(() => np.array(new float[] { 1, 2, 3 }), typeof(float));
            AssertMethodDtype(() => np.array(new double[] { 1, 2, 3 }), typeof(double));
            AssertMethodDtype(() => np.array(new decimal[] { 1, 2, 3 }), typeof(decimal));
            AssertMethodDtype(() => np.array(new Complex[] { 1, 2, 3 }), typeof(Complex));
        }

        private static void AssertFuncDtype(NDArray src, Type dtype)
        {
            var r = np.resize(src, (2, 4)); // tile 3 -> 8
            r.dtype.Should().Be(dtype);
            r.shape.Should().Equal(2L, 4L);
            // element i equals src[i % 3]
            for (int i = 0; i < 8; i++)
                r.GetAtIndex(i).Should().Be(src.GetAtIndex(i % 3));
        }

        private static void AssertMethodDtype(Func<NDArray> make, Type dtype)
        {
            var a = make();
            var first = a.GetAtIndex(0);
            a.resize(2, 4); // grow 3 -> 8, zero-filled tail
            a.dtype.Should().Be(dtype);
            a.shape.Should().Equal(2L, 4L);
            a.GetAtIndex(0).Should().Be(first);
        }
    }
}
