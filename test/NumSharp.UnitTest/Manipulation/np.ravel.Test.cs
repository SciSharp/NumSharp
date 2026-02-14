using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    public class np_ravel_Test
    {
        // ================================================================
        // VALUES: 1D arrays
        // ================================================================

        [Test]
        public void Ravel_1D_AlreadyFlat()
        {
            var a = np.array(new[] { 1, 2, 3, 4, 5 });
            var r = np.ravel(a);

            r.Should().BeShaped(5);
            r.Should().BeOfValues(1, 2, 3, 4, 5);
        }

        [Test]
        public void Ravel_1D_Instance()
        {
            var a = np.array(new[] { 10, 20, 30 });
            var r = a.ravel();

            r.Should().BeShaped(3);
            r.Should().BeOfValues(10, 20, 30);
        }

        // ================================================================
        // VALUES: 2D arrays
        // ================================================================

        [Test]
        public void Ravel_2D_COrder()
        {
            // NumPy: np.ravel([[1,2,3],[4,5,6]]) = [1,2,3,4,5,6]
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var r = np.ravel(a);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(1, 2, 3, 4, 5, 6);
        }

        [Test]
        public void Ravel_2D_Instance()
        {
            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var r = a.ravel();

            r.Should().BeShaped(4);
            r.Should().BeOfValues(1, 2, 3, 4);
        }

        [Test]
        public void Ravel_2D_SingleRow()
        {
            var a = np.array(new int[,] { { 1, 2, 3, 4, 5 } });
            var r = np.ravel(a);

            r.Should().BeShaped(5);
            r.Should().BeOfValues(1, 2, 3, 4, 5);
        }

        [Test]
        public void Ravel_2D_SingleColumn()
        {
            var a = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
            var r = np.ravel(a);

            r.Should().BeShaped(3);
            r.Should().BeOfValues(1, 2, 3);
        }

        // ================================================================
        // VALUES: 3D and 4D arrays
        // ================================================================

        [Test]
        public void Ravel_3D()
        {
            // NumPy: np.ravel(np.arange(24).reshape(2,3,4)) = [0,1,...,23]
            var a = np.arange(24).reshape(2, 3, 4);
            var r = np.ravel(a);

            r.Should().BeShaped(24);
            r.Should().BeOfValues(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
        }

        [Test]
        public void Ravel_4D()
        {
            var a = np.arange(24).reshape(new Shape(1, 2, 3, 4));
            var r = np.ravel(a);

            r.Should().BeShaped(24);
            r.ndim.Should().Be(1);
            r.GetInt32(0).Should().Be(0);
            r.GetInt32(23).Should().Be(23);
        }

        // ================================================================
        // VALUES: scalar and empty
        // ================================================================

        [Test]
        public void Ravel_Scalar()
        {
            // NumPy: np.ravel(np.array(42)) = [42] shape=(1,)
            var a = np.array(42);
            var r = np.ravel(a);

            r.ndim.Should().Be(1);
            r.size.Should().Be(1);
            r.GetInt32(0).Should().Be(42);
        }

        [Test]
        public void Ravel_Empty_1D()
        {
            // NumPy: np.ravel(np.array([])) = [] shape=(0,)
            var a = np.array(new double[0]);
            var r = np.ravel(a);

            r.ndim.Should().Be(1);
            r.size.Should().Be(0);
        }

        [Test]
        public void Ravel_SingleElement()
        {
            var a = np.array(new int[,] { { 42 } });
            var r = np.ravel(a);

            r.Should().BeShaped(1);
            r.Should().BeOfValues(42);
        }

        // ================================================================
        // VALUES: broadcast arrays
        // ================================================================

        [Test]
        public void Ravel_Broadcast_RowBroadcast()
        {
            // NumPy: np.ravel(np.broadcast_to([1,2,3], (3,3))) = [1,2,3,1,2,3,1,2,3]
            var a = np.broadcast_to(np.array(new[] { 1, 2, 3 }), new Shape(3, 3));
            var r = np.ravel(a);

            r.Should().BeShaped(9);
            r.Should().BeOfValues(1, 2, 3, 1, 2, 3, 1, 2, 3);
        }

        [Test]
        public void Ravel_Broadcast_ColumnBroadcast()
        {
            // NumPy: np.ravel(np.broadcast_to([[10],[20],[30]], (3,3)))
            //      = [10,10,10,20,20,20,30,30,30]
            var col = np.array(new int[,] { { 10 }, { 20 }, { 30 } });
            var a = np.broadcast_to(col, new Shape(3, 3));
            var r = np.ravel(a);

            r.Should().BeShaped(9);
            r.Should().BeOfValues(10, 10, 10, 20, 20, 20, 30, 30, 30);
        }

        [Test]
        public void Ravel_Broadcast_2x3_ColumnBroadcast()
        {
            // np.ravel(np.broadcast_to([[1],[2]], (2,3))) = [1,1,1,2,2,2]
            var col = np.array(new int[,] { { 1 }, { 2 } });
            var a = np.broadcast_to(col, new Shape(2, 3));
            var r = np.ravel(a);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(1, 1, 1, 2, 2, 2);
        }

        // ================================================================
        // VALUES: sliced arrays
        // ================================================================

        [Test]
        public void Ravel_Sliced_2D_ColumnSlice()
        {
            // NumPy: np.ravel(np.arange(12).reshape(3,4)[:,1:3]) = [1,2,5,6,9,10]
            var a = np.arange(12).reshape(3, 4)[":", "1:3"];
            var r = np.ravel(a);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(1, 2, 5, 6, 9, 10);
        }

        [Test]
        public void Ravel_Sliced_Step2()
        {
            // NumPy: np.ravel(np.arange(10)[::2]) = [0,2,4,6,8]
            var a = np.arange(10)["::2"];
            var r = np.ravel(a);

            r.Should().BeShaped(5);
            r.Should().BeOfValues(0, 2, 4, 6, 8);
        }

        [Test]
        public void Ravel_Sliced_Reversed()
        {
            // NumPy: np.ravel(np.arange(5)[::-1]) = [4,3,2,1,0]
            var a = np.arange(5)["::-1"];
            var r = np.ravel(a);

            r.Should().BeShaped(5);
            r.Should().BeOfValues(4, 3, 2, 1, 0);
        }

        // ================================================================
        // VALUES: transposed arrays
        // ================================================================

        [Test]
        public void Ravel_Transposed_2D()
        {
            // NumPy: np.ravel(np.array([[1,2,3],[4,5,6]]).T) = [1,4,2,5,3,6]
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var r = np.ravel(a.T);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(1, 4, 2, 5, 3, 6);
        }

        [Test]
        public void Ravel_Swapaxes_3D()
        {
            // NumPy: np.ravel(np.arange(12).reshape(2,3,2).swapaxes(1,2))
            //      = [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11]
            var a = np.arange(12).reshape(2, 3, 2);
            var s = np.swapaxes(a, 1, 2);
            var r = np.ravel(s);

            r.Should().BeShaped(12);
            r.Should().BeOfValues(0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11);
        }

        // ================================================================
        // VIEW SEMANTICS: contiguous array ravel is a view
        // ================================================================

        [Test]
        public void Ravel_Contiguous_IsView()
        {
            // NumPy: ravel of contiguous array returns view (shared memory)
            var a = np.array(new[] { 1, 2, 3, 4, 5 });
            var r = np.ravel(a);

            r.SetInt32(99, 0);
            a.GetInt32(0).Should().Be(99,
                "ravel of contiguous 1D array should return a view (shared memory). " +
                "NumPy: modifying ravel output modifies original.");
        }

        [Test]
        public void Ravel_Contiguous2D_IsView()
        {
            // NumPy: ravel of contiguous 2D array returns view
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var r = np.ravel(a);

            r.SetInt32(99, 0);
            a.GetInt32(0, 0).Should().Be(99,
                "ravel of contiguous 2D array should return a view. " +
                "NumPy: modifying ravel output modifies original.");
        }

        // ================================================================
        // VIEW SEMANTICS: non-contiguous ravel is a copy
        // ================================================================

        [Test]
        public void Ravel_StepSlice_IsCopy()
        {
            // NumPy: ravel of step-2 slice returns copy (not C-contiguous)
            var a = np.arange(10);
            var s = a["::2"];
            var r = np.ravel(s);

            r.SetInt32(99, 0);
            s.GetInt32(0).Should().Be(0,
                "ravel of step-2 slice should return a copy. " +
                "NumPy: step-2 slice is not C-contiguous.");
        }

        [Test]
        public void Ravel_Broadcast_IsCopy()
        {
            // NumPy: ravel of broadcast array returns copy
            var src = np.array(new[] { 1, 2, 3 });
            var bc = np.broadcast_to(src, new Shape(2, 3));
            var r = np.ravel(bc);

            r.SetInt32(99, 0);
            src.GetInt32(0).Should().Be(1,
                "ravel of broadcast array should return a copy. " +
                "NumPy: broadcast is not contiguous.");
        }

        [Test]
        public void Ravel_ColumnSlice_IsCopy()
        {
            // NumPy: ravel of column slice returns copy (not C-contiguous)
            var a = np.arange(12).reshape(3, 4);
            var s = a[":", "1:3"];
            var r = np.ravel(s);

            r.SetInt32(99, 0);
            s.GetInt32(0, 0).Should().Be(1,
                "ravel of column slice should return a copy. " +
                "NumPy: column slice is not C-contiguous.");
        }

        // ================================================================
        // VIEW SEMANTICS: bugs — ravel returns copy where NumPy returns view
        // These are upstream Shape.IsContiguous issues, not ravel bugs.
        // ================================================================

        [Test]
        [OpenBugs]
        public void Ravel_ContiguousSlice1D_ShouldBeView()
        {
            // NumPy: a[2:7] is c_contiguous=True, ravel returns a VIEW
            // NumSharp: a["2:7"] has IsSliced=True → IsContiguous=False → ravel copies
            var a = np.arange(10);
            var s = a["2:7"];  // [2,3,4,5,6] — contiguous in memory
            var r = np.ravel(s);

            r.SetInt32(99, 0);
            s.GetInt32(0).Should().Be(99,
                "NumPy: ravel of contiguous 1D slice (step=1) returns a view. " +
                "NumSharp: Shape.IsContiguous returns false for all slices, even " +
                "contiguous ones (step=1, no offset gaps). This causes ravel to " +
                "unnecessarily copy via CloneData.");
        }

        [Test]
        [OpenBugs]
        public void Ravel_ContiguousRowSlice2D_ShouldBeView()
        {
            // NumPy: c[1:3] (row slice) is c_contiguous=True, ravel returns VIEW
            // NumSharp: IsSliced=True → copies
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];  // rows 1-2, contiguous in memory
            var r = np.ravel(s);

            r.SetInt32(99, 0);
            s.GetInt32(0, 0).Should().Be(99,
                "NumPy: ravel of contiguous 2D row slice returns a view. " +
                "NumSharp: Shape.IsContiguous is false for all sliced shapes, " +
                "even when the slice is contiguous in memory.");
        }

        // ================================================================
        // FLATTEN: always returns a copy
        // ================================================================

        [Test]
        public void Flatten_Contiguous_IsCopy()
        {
            // NumPy: flatten always returns a copy, even for contiguous arrays
            var a = np.array(new[] { 1, 2, 3, 4, 5 });
            var f = a.flatten();

            f.SetInt32(99, 0);
            a.GetInt32(0).Should().Be(1,
                "flatten should always return a copy. " +
                "NumPy: modifying flatten output never modifies original.");
        }

        [Test]
        public void Flatten_2D_Values()
        {
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var f = a.flatten();

            f.Should().BeShaped(6);
            f.Should().BeOfValues(1, 2, 3, 4, 5, 6);
        }

        [Test]
        public void Flatten_Broadcast()
        {
            var bc = np.broadcast_to(np.array(new[] { 1, 2, 3 }), new Shape(2, 3));
            var f = bc.flatten();

            f.Should().BeShaped(6);
            f.Should().BeOfValues(1, 2, 3, 1, 2, 3);
        }

        [Test]
        public void Flatten_Broadcast_IsCopy()
        {
            var src = np.array(new[] { 1, 2, 3 });
            var bc = np.broadcast_to(src, new Shape(2, 3));
            var f = bc.flatten();

            f.SetInt32(99, 0);
            src.GetInt32(0).Should().Be(1,
                "flatten of broadcast array should not modify source");
        }

        // ================================================================
        // DTYPE PRESERVATION
        // ================================================================

        [Test]
        public void Ravel_PreservesDtype_Int32()
        {
            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(int));
        }

        [Test]
        public void Ravel_PreservesDtype_Int64()
        {
            var a = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(long));
        }

        [Test]
        public void Ravel_PreservesDtype_Float32()
        {
            var a = np.array(new float[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(float));
        }

        [Test]
        public void Ravel_PreservesDtype_Float64()
        {
            var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(double));
        }

        [Test]
        public void Ravel_PreservesDtype_Bool()
        {
            var a = np.array(new bool[,] { { true, false }, { false, true } });
            np.ravel(a).dtype.Should().Be(typeof(bool));
        }

        [Test]
        public void Ravel_PreservesDtype_Byte()
        {
            var a = np.array(new byte[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(byte));
        }

        [Test]
        public void Ravel_PreservesDtype_Int16()
        {
            var a = np.array(new short[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(short));
        }

        [Test]
        public void Ravel_PreservesDtype_UInt16()
        {
            var a = np.array(new ushort[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(ushort));
        }

        [Test]
        public void Ravel_PreservesDtype_UInt32()
        {
            var a = np.array(new uint[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(uint));
        }

        [Test]
        public void Ravel_PreservesDtype_UInt64()
        {
            var a = np.array(new ulong[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(ulong));
        }

        // ================================================================
        // NDIM AND SHAPE
        // ================================================================

        [Test]
        public void Ravel_AlwaysReturns1D()
        {
            np.ravel(np.arange(5)).ndim.Should().Be(1);
            np.ravel(np.arange(6).reshape(2, 3)).ndim.Should().Be(1);
            np.ravel(np.arange(24).reshape(2, 3, 4)).ndim.Should().Be(1);
        }

        [Test]
        public void Ravel_SizePreserved()
        {
            np.ravel(np.arange(5)).size.Should().Be(5);
            np.ravel(np.arange(6).reshape(2, 3)).size.Should().Be(6);
            np.ravel(np.arange(24).reshape(2, 3, 4)).size.Should().Be(24);
        }

        // ================================================================
        // ORIGINAL NOT MODIFIED (for copy cases)
        // ================================================================

        [Test]
        public void Ravel_Broadcast_OriginalNotModified()
        {
            var src = np.array(new[] { 1, 2, 3 });
            var bc = np.broadcast_to(src, new Shape(2, 3));
            var r = np.ravel(bc);

            // Modify ravel output
            r.SetInt32(99, 0);

            // Original source should not change
            src.Should().BeOfValues(1, 2, 3);
        }

        [Test]
        public void Ravel_StepSlice_OriginalNotModified()
        {
            var a = np.arange(10);
            var s = a["::2"];
            var r = np.ravel(s);

            r.SetInt32(99, 0);

            // Original should not change
            a.GetInt32(0).Should().Be(0);
        }

        // ================================================================
        // RAVEL vs FLATTEN EQUIVALENCE (values only)
        // ================================================================

        [Test]
        public void Ravel_EquivalentToFlatten_Values()
        {
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });

            var r = np.ravel(a);
            var f = a.flatten();

            np.array_equal(r, f).Should().BeTrue(
                "ravel and flatten should produce the same values for C-order");
        }

        [Test]
        public void Ravel_EquivalentToFlatten_Broadcast()
        {
            var bc = np.broadcast_to(np.array(new[] { 1, 2, 3 }), new Shape(2, 3));

            var r = np.ravel(bc);
            var f = bc.flatten();

            np.array_equal(r, f).Should().BeTrue();
        }

        // ================================================================
        // RAVEL of various dtypes — values
        // ================================================================

        [Test]
        public void Ravel_BoolValues()
        {
            var a = np.array(new bool[,] { { true, false }, { false, true } });
            var r = np.ravel(a);

            r.Should().BeShaped(4);
            r.Should().BeOfValues(true, false, false, true);
        }

        [Test]
        public void Ravel_DoubleValues()
        {
            var a = np.array(new double[,] { { 1.5, 2.5 }, { 3.5, 4.5 } });
            var r = np.ravel(a);

            r.Should().BeShaped(4);
            r.Should().BeOfValues(1.5, 2.5, 3.5, 4.5);
        }

        [Test]
        public void Ravel_FloatValues()
        {
            var a = np.array(new float[,] { { 1.5f, 2.5f }, { 3.5f, 4.5f } });
            var r = np.ravel(a);

            r.Should().BeShaped(4);
            r.Should().BeOfValues(1.5f, 2.5f, 3.5f, 4.5f);
        }

        [Test]
        public void Ravel_Int64Values()
        {
            var a = np.array(new long[,] { { 100, 200 }, { 300, 400 } });
            var r = np.ravel(a);

            r.Should().BeShaped(4);
            r.Should().BeOfValues(100L, 200L, 300L, 400L);
        }

        [Test]
        public void Ravel_ByteValues()
        {
            var a = np.array(new byte[,] { { 1, 2 }, { 3, 4 } });
            var r = np.ravel(a);

            r.Should().BeShaped(4);
            r.Should().BeOfValues((byte)1, (byte)2, (byte)3, (byte)4);
        }

        // ================================================================
        // LARGE ARRAY
        // ================================================================

        [Test]
        public void Ravel_LargeArray()
        {
            var a = np.arange(1000).reshape(10, 10, 10);
            var r = np.ravel(a);

            r.Should().BeShaped(1000);
            r.GetInt32(0).Should().Be(0);
            r.GetInt32(999).Should().Be(999);
        }
    }
}
