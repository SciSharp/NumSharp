using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using System;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class np_delete_Test : TestClass
    {
        // ---------------- scalar-int obj path ----------------

        [TestMethod]
        public void Int_Mid()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.delete(arr, 2);
            r.shape.Should().BeEquivalentTo(new long[] { 4 });
            r.array_equal(np.array(new long[] { 1, 2, 4, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Int_NegativeIndex()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, -1).array_equal(np.array(new long[] { 1, 2, 3, 4 })).Should().BeTrue();
            np.delete(arr, -5).array_equal(np.array(new long[] { 2, 3, 4, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Int_FirstAndLast()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, 0).array_equal(np.array(new long[] { 2, 3, 4, 5 })).Should().BeTrue();
            np.delete(arr, 4).array_equal(np.array(new long[] { 1, 2, 3, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void Int_OutOfBounds_Throws()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            Action act = () => np.delete(arr, 10);
            act.Should().Throw<IndexOutOfRangeException>();

            Action act2 = () => np.delete(arr, -10);
            act2.Should().Throw<IndexOutOfRangeException>();
        }

        // ---------------- slice obj path ----------------

        [TestMethod]
        public void Slice_StartStop()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, new Slice(1, 4)).array_equal(np.array(new long[] { 1, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Slice_StepGreater1()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, new Slice(null, null, 2)).array_equal(np.array(new long[] { 2, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void Slice_NegativeStep()
        {
            // slice(4, 0, -1) → indices [4, 3, 2, 1] = keep [0]
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, new Slice(4, 0, -1)).array_equal(np.array(new long[] { 1 })).Should().BeTrue();
        }

        [TestMethod]
        public void Slice_EmptyCovers_NoOp()
        {
            // slice(2, 2) covers nothing.
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, new Slice(2, 2)).array_equal(arr).Should().BeTrue();
        }

        // ---------------- array obj path ----------------

        [TestMethod]
        public void Long_Array()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, new long[] { 1, 3 }).array_equal(np.array(new long[] { 1, 3, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Int_Array_WithDuplicates()
        {
            // duplicates collapse — each index removed at most once.
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, new int[] { 1, 1, 2 }).array_equal(np.array(new long[] { 1, 4, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Long_Array_Empty_NoOp()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, new long[0]).array_equal(arr).Should().BeTrue();
        }

        [TestMethod]
        public void Long_Array_NegativeIndices()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, new long[] { -1, 0 }).array_equal(np.array(new long[] { 2, 3, 4 })).Should().BeTrue();
        }

        // ---------------- bool mask obj path ----------------

        [TestMethod]
        public void Bool_Mask()
        {
            // [T, F, T, F, T] → keep [F, T, F, T, F] → [2, 4]
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, new bool[] { true, false, true, false, true })
                .array_equal(np.array(new long[] { 2, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void Bool_Mask_AllTrue_ReturnsEmpty()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.delete(arr, new bool[] { true, true, true, true, true });
            r.size.Should().Be(0);
            r.shape.Should().BeEquivalentTo(new long[] { 0 });
        }

        [TestMethod]
        public void Bool_Mask_LengthMismatch_Throws()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            Action act = () => np.delete(arr, new bool[] { true, false });
            act.Should().Throw<ArgumentException>();
        }

        // ---------------- NDArray obj dispatch ----------------

        [TestMethod]
        public void NDArray_Obj_Int()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, np.array(new long[] { 1, 3 }))
                .array_equal(np.array(new long[] { 1, 3, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void NDArray_Obj_BoolMask()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, np.array(new bool[] { true, false, true, false, true }))
                .array_equal(np.array(new long[] { 2, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void NDArray_Obj_SingleElement_CollapsesToScalar()
        {
            // size==1 integer obj collapses to scalar-index path (NumPy parity).
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.delete(arr, np.array(new long[] { 2 }))
                .array_equal(np.array(new long[] { 1, 2, 4, 5 })).Should().BeTrue();
        }

        // ---------------- 2-D / axis paths ----------------

        [TestMethod]
        public void TwoD_Axis0()
        {
            // arr[1, :] removed.
            var arr2d = np.arange(12).reshape(3, 4);
            var r = np.delete(arr2d, 1, axis: 0);
            r.shape.Should().BeEquivalentTo(new long[] { 2, 4 });
            r.array_equal(np.array(new int[,] { { 0, 1, 2, 3 }, { 8, 9, 10, 11 } })).Should().BeTrue();
        }

        [TestMethod]
        public void TwoD_Axis1_MultiIndex()
        {
            var arr2d = np.arange(12).reshape(3, 4);
            var r = np.delete(arr2d, new long[] { 1, 3 }, axis: 1);
            r.shape.Should().BeEquivalentTo(new long[] { 3, 2 });
            r.array_equal(np.array(new int[,] { { 0, 2 }, { 4, 6 }, { 8, 10 } })).Should().BeTrue();
        }

        [TestMethod]
        public void TwoD_AxisNone_Flattens()
        {
            // axis=None ⇒ delete from ravel(arr).
            var arr2d = np.arange(12).reshape(3, 4);
            var r = np.delete(arr2d, new long[] { 1, 3, 5 });
            r.shape.Should().BeEquivalentTo(new long[] { 9 });
            r.array_equal(np.array(new int[] { 0, 2, 4, 6, 7, 8, 9, 10, 11 })).Should().BeTrue();
        }

        [TestMethod]
        public void TwoD_NegativeAxis()
        {
            var arr2d = np.arange(12).reshape(3, 4);
            np.delete(arr2d, 1, axis: -1)
                .array_equal(np.delete(arr2d, 1, axis: 1)).Should().BeTrue();
        }

        // ---------------- dtype coverage ----------------

        [TestMethod]
        public void Dtype_Double()
        {
            np.delete(np.array(new double[] { 1.0, 2.0, 3.0, 4.0 }), 1)
                .array_equal(np.array(new double[] { 1.0, 3.0, 4.0 })).Should().BeTrue();
        }

        [TestMethod]
        public void Dtype_Byte()
        {
            np.delete(np.array(new byte[] { 1, 2, 3, 4 }), 1)
                .array_equal(np.array(new byte[] { 1, 3, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void Dtype_Float()
        {
            np.delete(np.array(new float[] { 1f, 2f, 3f, 4f }), 1)
                .array_equal(np.array(new float[] { 1f, 3f, 4f })).Should().BeTrue();
        }

        // ---------------- NDArray.delete instance shim ----------------

        [TestMethod]
        public void Instance_Method()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            arr.delete(new long[] { 1, 3 })
                .array_equal(np.array(new long[] { 1, 3, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Instance_Method_Ints()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            arr.delete(new int[] { 1, 3 })
                .array_equal(np.array(new long[] { 1, 3, 5 })).Should().BeTrue();
        }

        // ---------------- result is a copy, not a view ----------------

        [TestMethod]
        public void Result_Is_Copy_Not_View()
        {
            // Mutating the result must not touch the original.
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.delete(arr, 2);
            r[0] = 999L;
            arr[0].GetValue(0).Should().Be(1L); // unchanged
        }
    }
}
