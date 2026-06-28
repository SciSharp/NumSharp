using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class np_ravel_Test
    {
        // ================================================================
        // VALUES: 1D arrays
        // ================================================================

        [TestMethod]
        public void Ravel_1D_AlreadyFlat()
        {
            var a = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.ravel(a);

            r.Should().BeShaped(5);
            r.Should().BeOfValues(1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Ravel_1D_Instance()
        {
            var a = np.array(new long[] { 10, 20, 30 });
            var r = a.ravel();

            r.Should().BeShaped(3);
            r.Should().BeOfValues(10, 20, 30);
        }

        // ================================================================
        // VALUES: 2D arrays
        // ================================================================

        [TestMethod]
        public void Ravel_2D_COrder()
        {
            // NumPy: np.ravel([[1,2,3],[4,5,6]]) = [1,2,3,4,5,6]
            var a = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var r = np.ravel(a);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void Ravel_2D_Instance()
        {
            var a = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
            var r = a.ravel();

            r.Should().BeShaped(4);
            r.Should().BeOfValues(1, 2, 3, 4);
        }

        [TestMethod]
        public void Ravel_2D_SingleRow()
        {
            var a = np.array(new long[,] { { 1, 2, 3, 4, 5 } });
            var r = np.ravel(a);

            r.Should().BeShaped(5);
            r.Should().BeOfValues(1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Ravel_2D_SingleColumn()
        {
            var a = np.array(new long[,] { { 1 }, { 2 }, { 3 } });
            var r = np.ravel(a);

            r.Should().BeShaped(3);
            r.Should().BeOfValues(1, 2, 3);
        }

        // ================================================================
        // VALUES: 3D and 4D arrays
        // ================================================================

        [TestMethod]
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

        [TestMethod]
        public void Ravel_4D()
        {
            var a = np.arange(24).reshape(new Shape(1, 2, 3, 4));
            var r = np.ravel(a);

            r.Should().BeShaped(24);
            r.ndim.Should().Be(1);
            r.GetInt64(0).Should().Be(0);
            r.GetInt64(23).Should().Be(23);
        }

        // ================================================================
        // VALUES: scalar and empty
        // ================================================================

        [TestMethod]
        public void Ravel_Scalar()
        {
            // NumPy: np.ravel(np.array(42)) = [42] shape=(1,)
            // np.array(42) creates int32 array
            var a = np.array(42);
            var r = np.ravel(a);

            r.ndim.Should().Be(1);
            r.size.Should().Be(1);
            r.GetInt32(0).Should().Be(42);
        }

        [TestMethod]
        public void Ravel_Empty_1D()
        {
            // NumPy: np.ravel(np.array([])) = [] shape=(0,)
            var a = np.array(new double[0]);
            var r = np.ravel(a);

            r.ndim.Should().Be(1);
            r.size.Should().Be(0);
        }

        [TestMethod]
        public void Ravel_SingleElement()
        {
            var a = np.array(new long[,] { { 42 } });
            var r = np.ravel(a);

            r.Should().BeShaped(1);
            r.Should().BeOfValues(42);
        }

        // ================================================================
        // VALUES: broadcast arrays
        // ================================================================

        [TestMethod]
        public void Ravel_Broadcast_RowBroadcast()
        {
            // NumPy: np.ravel(np.broadcast_to([1,2,3], (3,3))) = [1,2,3,1,2,3,1,2,3]
            var a = np.broadcast_to(np.array(new long[] { 1, 2, 3 }), new Shape(3, 3));
            var r = np.ravel(a);

            r.Should().BeShaped(9);
            r.Should().BeOfValues(1, 2, 3, 1, 2, 3, 1, 2, 3);
        }

        [TestMethod]
        public void Ravel_Broadcast_ColumnBroadcast()
        {
            // NumPy: np.ravel(np.broadcast_to([[10],[20],[30]], (3,3)))
            //      = [10,10,10,20,20,20,30,30,30]
            var col = np.array(new long[,] { { 10 }, { 20 }, { 30 } });
            var a = np.broadcast_to(col, new Shape(3, 3));
            var r = np.ravel(a);

            r.Should().BeShaped(9);
            r.Should().BeOfValues(10, 10, 10, 20, 20, 20, 30, 30, 30);
        }

        [TestMethod]
        public void Ravel_Broadcast_2x3_ColumnBroadcast()
        {
            // np.ravel(np.broadcast_to([[1],[2]], (2,3))) = [1,1,1,2,2,2]
            var col = np.array(new long[,] { { 1 }, { 2 } });
            var a = np.broadcast_to(col, new Shape(2, 3));
            var r = np.ravel(a);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(1, 1, 1, 2, 2, 2);
        }

        // ================================================================
        // VALUES: sliced arrays
        // ================================================================

        [TestMethod]
        public void Ravel_Sliced_2D_ColumnSlice()
        {
            // NumPy: np.ravel(np.arange(12).reshape(3,4)[:,1:3]) = [1,2,5,6,9,10]
            var a = np.arange(12).reshape(3, 4)[":", "1:3"];
            var r = np.ravel(a);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(1, 2, 5, 6, 9, 10);
        }

        [TestMethod]
        public void Ravel_Sliced_Step2()
        {
            // NumPy: np.ravel(np.arange(10)[::2]) = [0,2,4,6,8]
            var a = np.arange(10)["::2"];
            var r = np.ravel(a);

            r.Should().BeShaped(5);
            r.Should().BeOfValues(0, 2, 4, 6, 8);
        }

        [TestMethod]
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

        [TestMethod]
        public void Ravel_Transposed_2D()
        {
            // NumPy: np.ravel(np.array([[1,2,3],[4,5,6]]).T) = [1,4,2,5,3,6]
            var a = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var r = np.ravel(a.T);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(1, 4, 2, 5, 3, 6);
        }

        [TestMethod]
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

        [TestMethod]
        public void Ravel_Contiguous_IsView()
        {
            // NumPy: ravel of contiguous array returns view (shared memory)
            var a = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.ravel(a);

            r.SetInt64(99, 0);
            a.GetInt64(0).Should().Be(99,
                "ravel of contiguous 1D array should return a view (shared memory). " +
                "NumPy: modifying ravel output modifies original.");
        }

        [TestMethod]
        public void Ravel_Contiguous2D_IsView()
        {
            // NumPy: ravel of contiguous 2D array returns view
            var a = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var r = np.ravel(a);

            r.SetInt64(99, 0);
            a.GetInt64(0, 0).Should().Be(99,
                "ravel of contiguous 2D array should return a view. " +
                "NumPy: modifying ravel output modifies original.");
        }

        // ================================================================
        // VIEW SEMANTICS: non-contiguous ravel is a copy
        // ================================================================

        [TestMethod]
        public void Ravel_StepSlice_IsCopy()
        {
            // NumPy: ravel of step-2 slice returns copy (not C-contiguous)
            var a = np.arange(10);
            var s = a["::2"];
            var r = np.ravel(s);

            r.SetInt64(99, 0);
            s.GetInt64(0).Should().Be(0,
                "ravel of step-2 slice should return a copy. " +
                "NumPy: step-2 slice is not C-contiguous.");
        }

        [TestMethod]
        public void Ravel_Broadcast_IsCopy()
        {
            // NumPy: ravel of broadcast array returns copy
            var src = np.array(new long[] { 1, 2, 3 });
            var bc = np.broadcast_to(src, new Shape(2, 3));
            var r = np.ravel(bc);

            r.SetInt64(99, 0);
            src.GetInt64(0).Should().Be(1,
                "ravel of broadcast array should return a copy. " +
                "NumPy: broadcast is not contiguous.");
        }

        [TestMethod]
        public void Ravel_ColumnSlice_IsCopy()
        {
            // NumPy: ravel of column slice returns copy (not C-contiguous)
            var a = np.arange(12).reshape(3, 4);
            var s = a[":", "1:3"];
            var r = np.ravel(s);

            r.SetInt64(99, 0);
            s.GetInt64(0, 0).Should().Be(1,
                "ravel of column slice should return a copy. " +
                "NumPy: column slice is not C-contiguous.");
        }

        // ================================================================
        // VIEW SEMANTICS: bugs — ravel returns copy where NumPy returns view
        // These are upstream Shape.IsContiguous issues, not ravel bugs.
        // ================================================================

        [TestMethod]
        public void Ravel_ContiguousSlice1D_ShouldBeView()
        {
            // NumPy: a[2:7] is c_contiguous=True, ravel returns a VIEW
            // NumSharp: a["2:7"] has IsSliced=True → IsContiguous=False → ravel copies
            var a = np.arange(10);
            var s = a["2:7"];  // [2,3,4,5,6] — contiguous in memory
            var r = np.ravel(s);

            r.SetInt64(99, 0);
            s.GetInt64(0).Should().Be(99,
                "NumPy: ravel of contiguous 1D slice (step=1) returns a view. " +
                "NumSharp: Shape.IsContiguous returns false for all slices, even " +
                "contiguous ones (step=1, no offset gaps). This causes ravel to " +
                "unnecessarily copy via CloneData.");
        }

        [TestMethod]
        public void Ravel_ContiguousRowSlice2D_ShouldBeView()
        {
            // NumPy: c[1:3] (row slice) is c_contiguous=True, ravel returns VIEW
            // NumSharp: IsSliced=True → copies
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];  // rows 1-2, contiguous in memory
            var r = np.ravel(s);

            r.SetInt64(99, 0);
            s.GetInt64(0, 0).Should().Be(99,
                "NumPy: ravel of contiguous 2D row slice returns a view. " +
                "NumSharp: Shape.IsContiguous is false for all sliced shapes, " +
                "even when the slice is contiguous in memory.");
        }

        // ================================================================
        // FLATTEN: always returns a copy
        // ================================================================

        [TestMethod]
        public void Flatten_Contiguous_IsCopy()
        {
            // NumPy: flatten always returns a copy, even for contiguous arrays
            var a = np.array(new long[] { 1, 2, 3, 4, 5 });
            var f = a.flatten();

            f.SetInt64(99, 0);
            a.GetInt64(0).Should().Be(1,
                "flatten should always return a copy. " +
                "NumPy: modifying flatten output never modifies original.");
        }

        [TestMethod]
        public void Flatten_2D_Values()
        {
            var a = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var f = a.flatten();

            f.Should().BeShaped(6);
            f.Should().BeOfValues(1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void Flatten_Broadcast()
        {
            var bc = np.broadcast_to(np.array(new long[] { 1, 2, 3 }), new Shape(2, 3));
            var f = bc.flatten();

            f.Should().BeShaped(6);
            f.Should().BeOfValues(1, 2, 3, 1, 2, 3);
        }

        [TestMethod]
        public void Flatten_Broadcast_IsCopy()
        {
            var src = np.array(new long[] { 1, 2, 3 });
            var bc = np.broadcast_to(src, new Shape(2, 3));
            var f = bc.flatten();

            f.SetInt64(99, 0);
            src.GetInt64(0).Should().Be(1,
                "flatten of broadcast array should not modify source");
        }

        // ================================================================
        // DTYPE PRESERVATION
        // ================================================================

        [TestMethod]
        public void Ravel_PreservesDtype_Int32()
        {
            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void Ravel_PreservesDtype_Int64()
        {
            var a = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(long));
        }

        [TestMethod]
        public void Ravel_PreservesDtype_Float32()
        {
            var a = np.array(new float[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void Ravel_PreservesDtype_Float64()
        {
            var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void Ravel_PreservesDtype_Bool()
        {
            var a = np.array(new bool[,] { { true, false }, { false, true } });
            np.ravel(a).dtype.Should().Be(typeof(bool));
        }

        [TestMethod]
        public void Ravel_PreservesDtype_Byte()
        {
            var a = np.array(new byte[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(byte));
        }

        [TestMethod]
        public void Ravel_PreservesDtype_Int16()
        {
            var a = np.array(new short[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(short));
        }

        [TestMethod]
        public void Ravel_PreservesDtype_UInt16()
        {
            var a = np.array(new ushort[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(ushort));
        }

        [TestMethod]
        public void Ravel_PreservesDtype_UInt32()
        {
            var a = np.array(new uint[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(uint));
        }

        [TestMethod]
        public void Ravel_PreservesDtype_UInt64()
        {
            var a = np.array(new ulong[,] { { 1, 2 }, { 3, 4 } });
            np.ravel(a).dtype.Should().Be(typeof(ulong));
        }

        // ================================================================
        // NDIM AND SHAPE
        // ================================================================

        [TestMethod]
        public void Ravel_AlwaysReturns1D()
        {
            np.ravel(np.arange(5)).ndim.Should().Be(1);
            np.ravel(np.arange(6).reshape(2, 3)).ndim.Should().Be(1);
            np.ravel(np.arange(24).reshape(2, 3, 4)).ndim.Should().Be(1);
        }

        [TestMethod]
        public void Ravel_SizePreserved()
        {
            np.ravel(np.arange(5)).size.Should().Be(5);
            np.ravel(np.arange(6).reshape(2, 3)).size.Should().Be(6);
            np.ravel(np.arange(24).reshape(2, 3, 4)).size.Should().Be(24);
        }

        // ================================================================
        // ORIGINAL NOT MODIFIED (for copy cases)
        // ================================================================

        [TestMethod]
        public void Ravel_Broadcast_OriginalNotModified()
        {
            var src = np.array(new long[] { 1, 2, 3 });
            var bc = np.broadcast_to(src, new Shape(2, 3));
            var r = np.ravel(bc);

            // Modify ravel output
            r.SetInt64(99, 0);

            // Original source should not change
            src.Should().BeOfValues(1, 2, 3);
        }

        [TestMethod]
        public void Ravel_StepSlice_OriginalNotModified()
        {
            var a = np.arange(10);
            var s = a["::2"];
            var r = np.ravel(s);

            r.SetInt64(99, 0);

            // Original should not change
            a.GetInt64(0).Should().Be(0);
        }

        // ================================================================
        // RAVEL vs FLATTEN EQUIVALENCE (values only)
        // ================================================================

        [TestMethod]
        public void Ravel_EquivalentToFlatten_Values()
        {
            var a = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });

            var r = np.ravel(a);
            var f = a.flatten();

            np.array_equal(r, f).Should().BeTrue(
                "ravel and flatten should produce the same values for C-order");
        }

        [TestMethod]
        public void Ravel_EquivalentToFlatten_Broadcast()
        {
            var bc = np.broadcast_to(np.array(new long[] { 1, 2, 3 }), new Shape(2, 3));

            var r = np.ravel(bc);
            var f = bc.flatten();

            np.array_equal(r, f).Should().BeTrue();
        }

        // ================================================================
        // RAVEL of various dtypes — values
        // ================================================================

        [TestMethod]
        public void Ravel_BoolValues()
        {
            var a = np.array(new bool[,] { { true, false }, { false, true } });
            var r = np.ravel(a);

            r.Should().BeShaped(4);
            r.Should().BeOfValues(true, false, false, true);
        }

        [TestMethod]
        public void Ravel_DoubleValues()
        {
            var a = np.array(new double[,] { { 1.5, 2.5 }, { 3.5, 4.5 } });
            var r = np.ravel(a);

            r.Should().BeShaped(4);
            r.Should().BeOfValues(1.5, 2.5, 3.5, 4.5);
        }

        [TestMethod]
        public void Ravel_FloatValues()
        {
            var a = np.array(new float[,] { { 1.5f, 2.5f }, { 3.5f, 4.5f } });
            var r = np.ravel(a);

            r.Should().BeShaped(4);
            r.Should().BeOfValues(1.5f, 2.5f, 3.5f, 4.5f);
        }

        [TestMethod]
        public void Ravel_Int64Values()
        {
            var a = np.array(new long[,] { { 100, 200 }, { 300, 400 } });
            var r = np.ravel(a);

            r.Should().BeShaped(4);
            r.Should().BeOfValues(100L, 200L, 300L, 400L);
        }

        [TestMethod]
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

        [TestMethod]
        public void Ravel_LargeArray()
        {
            var a = np.arange(1000).reshape(10, 10, 10);
            var r = np.ravel(a);

            r.Should().BeShaped(1000);
            r.GetInt64(0).Should().Be(0);
            r.GetInt64(999).Should().Be(999);
        }

        // ================================================================
        // ORDER='F' — F-contiguous source returns a view (no copy)
        // ================================================================

        [TestMethod]
        public void Ravel_FOrder_FContig2D_IsView()
        {
            // NumPy: ravel(aF,'F') of F-contig source shares memory.
            //   aF = np.arange(12).reshape(3,4).copy('F')
            //   np.shares_memory(np.ravel(aF,'F'), aF) == True
            var aF = np.arange(12).reshape(3, 4).copy('F');
            aF.Shape.IsFContiguous.Should().BeTrue("test precondition");

            var r = np.ravel(aF, 'F');

            r.Should().BeShaped(12);
            r.ndim.Should().Be(1);

            r.SetAtIndex(999L, 0L);
            aF.GetAtIndex(0).Should().Be(999L,
                "ravel('F') of F-contig source must return a view sharing memory.");
        }

        [TestMethod]
        public void Ravel_FOrder_FContig2D_ValuesMatchColumnMajor()
        {
            // F-contig memory layout for arange(12).reshape(3,4) is
            //   columns: [0,4,8 | 1,5,9 | 2,6,10 | 3,7,11].
            // ravel('F') must read column-major and reproduce that sequence.
            var aF = np.arange(12).reshape(3, 4).copy('F');
            var r = np.ravel(aF, 'F');

            r.Should().BeOfValues(0L, 4L, 8L, 1L, 5L, 9L, 2L, 6L, 10L, 3L, 7L, 11L);
        }

        [TestMethod]
        public void Ravel_FOrder_FContig3D_IsView()
        {
            // 3D F-contig source: strides[0]==1 guarantees the linear memory walk
            // matches F-order traversal — ravel must return a view.
            var aF = np.arange(24).reshape(2, 3, 4).copy('F');
            aF.Shape.IsFContiguous.Should().BeTrue("test precondition");

            var r = np.ravel(aF, 'F');

            r.Should().BeShaped(24);
            r.SetAtIndex(777L, 5L);
            // The 5th element in F-order corresponds to memory[5] in F-contig storage.
            // Decompose F-flat-index 5 with dims (2,3,4) factors (1, 2, 6):
            //   5 / 6 = 0 (k=axis2), 5 - 0*6 = 5 → 5 / 2 = 2 (j=axis1), 5 - 2*2 = 1 (i=axis0)
            // So aF[1,2,0] should now be 777.
            aF.GetInt64(1, 2, 0).Should().Be(777L,
                "ravel('F') of F-contig 3D source must return a view sharing memory.");
        }

        [TestMethod]
        public void Ravel_FOrder_CContig_IsCopy()
        {
            // ravel('F') of a C-contig (NOT F-contig) source must copy:
            // memory walk gives C-order values, which differ from F-order.
            var aC = np.arange(12).reshape(3, 4);
            aC.Shape.IsContiguous.Should().BeTrue("test precondition");
            aC.Shape.IsFContiguous.Should().BeFalse("test precondition");

            var r = np.ravel(aC, 'F');

            // Values should be column-major read-out of the C-contig logical array:
            //   [0,4,8, 1,5,9, 2,6,10, 3,7,11]
            r.Should().BeOfValues(0L, 4L, 8L, 1L, 5L, 9L, 2L, 6L, 10L, 3L, 7L, 11L);

            // Writing to r must not propagate back to the C-contig source.
            r.SetAtIndex(999L, 0L);
            aC.GetInt64(0, 0).Should().Be(0L,
                "ravel('F') of C-contig source must materialize a fresh column-major copy.");
        }

        [TestMethod]
        public void Ravel_FOrder_Transpose2D_IsView()
        {
            // Transpose of C-contig 2D shares memory with swapped strides, yielding an
            // F-contig view. ravel('F') of that view should also be a view.
            var a = np.arange(12).reshape(3, 4);
            var aT = a.T; // (4,3), strides [1,4] — F-contig

            aT.Shape.IsFContiguous.Should().BeTrue("transpose of C-contig 2D should be F-contig");

            var r = np.ravel(aT, 'F');

            // F-order ravel of aT walks memory linearly. aT's underlying memory is still
            // the original C-contig buffer [0..11], so r values are [0..11].
            r.Should().BeOfValues(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L);

            // And it's a view back to the original storage.
            r.SetAtIndex(888L, 0L);
            a.GetInt64(0, 0).Should().Be(888L,
                "ravel('F') of an F-contig transpose-view should share memory with the underlying buffer.");
        }

        [TestMethod]
        public void Ravel_FOrder_KOrder_FContigSource_IsView()
        {
            // order='K' on an F-contig source resolves to 'F'; should hit the view path.
            var aF = np.arange(8).reshape(2, 4).copy('F');
            aF.Shape.IsFContiguous.Should().BeTrue("test precondition");

            var r = np.ravel(aF, 'K');

            r.Should().BeShaped(8);
            r.SetAtIndex(123L, 0L);
            aF.GetAtIndex(0).Should().Be(123L,
                "ravel('K') on an F-contig source must resolve to 'F' and return a view.");
        }

        [TestMethod]
        public void Ravel_FOrder_AOrder_FContigSource_IsView()
        {
            // order='A' on a strictly-F-contig source resolves to 'F'; should view.
            var aF = np.arange(6).reshape(2, 3).copy('F');
            aF.Shape.IsFContiguous.Should().BeTrue("test precondition");
            aF.Shape.IsContiguous.Should().BeFalse("strictly F-contig (not also C-contig)");

            var r = np.ravel(aF, 'A');

            r.Should().BeShaped(6);
            r.SetAtIndex(321L, 0L);
            aF.GetAtIndex(0).Should().Be(321L,
                "ravel('A') on strictly-F-contig source must resolve to 'F' and return a view.");
        }

        [TestMethod]
        public void Ravel_FOrder_FContig_DtypeFloat()
        {
            var aF = np.arange(6.0).reshape(2, 3).copy('F');
            aF.Shape.IsFContiguous.Should().BeTrue("test precondition");

            var r = np.ravel(aF, 'F');

            r.dtype.Should().Be(typeof(double));
            r.Should().BeShaped(6);
            // Memory layout F-contig: columns [0.0,3.0 | 1.0,4.0 | 2.0,5.0]
            r.Should().BeOfValues(0.0, 3.0, 1.0, 4.0, 2.0, 5.0);
        }

        [TestMethod]
        public void Ravel_FOrder_FContig_EquivalentToFlattenF_Values()
        {
            // ravel('F') and flatten('F') must produce the same values for any source.
            var aF = np.arange(12).reshape(3, 4).copy('F');

            var r = np.ravel(aF, 'F');
            var f = aF.flatten('F');

            np.array_equal(r, f).Should().BeTrue(
                "ravel('F') and flatten('F') must produce equal values regardless of copy/view choice.");
        }

        [TestMethod]
        public void Ravel_FOrder_FContig_PreservesSize()
        {
            np.ravel(np.arange(6).reshape(2, 3).copy('F'), 'F').size.Should().Be(6);
            np.ravel(np.arange(24).reshape(2, 3, 4).copy('F'), 'F').size.Should().Be(24);
            np.ravel(np.arange(120).reshape(2, 3, 4, 5).copy('F'), 'F').size.Should().Be(120);
        }

        [TestMethod]
        public void Ravel_FOrder_FContigColumnSlice_PreservesOffset_IsView()
        {
            // F-contig column slice has offset != 0 but remains F-contig:
            //   stride[1] == dim[0] * stride[0] still holds when slicing the second axis.
            // ravel('F') must preserve the offset and bufferSize so the view continues to
            // read from the correct buffer range.
            var aF = np.arange(20).reshape(4, 5).copy('F');
            var s = aF[":", "1:3"]; // (4,2), F-contig, offset=4
            s.Shape.IsFContiguous.Should().BeTrue("column slice of F-contig preserves F-contiguity");
            s.Shape.offset.Should().Be(4, "column 1 starts at memory offset 4 in F-contig (4,5)");

            var r = np.ravel(s, 'F');

            r.Should().BeShaped(8);
            // F-order traversal of s:
            //   s[0,0]=1, s[1,0]=6, s[2,0]=11, s[3,0]=16, s[0,1]=2, s[1,1]=7, s[2,1]=12, s[3,1]=17
            r.Should().BeOfValues(1L, 6L, 11L, 16L, 2L, 7L, 12L, 17L);

            // Write through r[0] and observe s[0,0] / aF[0,1] (all share memory[4]).
            r.SetInt64(999L, 0);
            s.GetInt64(0, 0).Should().Be(999L, "ravel('F') of F-contig column slice should be a view of the slice.");
            aF.GetInt64(0, 1).Should().Be(999L, "and therefore also a view back to the parent buffer.");
        }

        [TestMethod]
        public void Ravel_FOrder_FContig_BothCAndFContig_IsView()
        {
            // A (1, N) shape is both C-contig and F-contig. ravel('F') should still take
            // the view path; the 1-D Alias is also both C- and F-contig.
            var both = np.array(new long[,] { { 10, 20, 30, 40 } });
            both.Shape.IsContiguous.Should().BeTrue("test precondition");
            both.Shape.IsFContiguous.Should().BeTrue("test precondition");

            var r = np.ravel(both, 'F');

            r.Should().BeShaped(4);
            r.Should().BeOfValues(10L, 20L, 30L, 40L);
            r.SetInt64(777L, 0);
            both.GetInt64(0, 0).Should().Be(777L,
                "ravel('F') on shape (1,N) (both C & F contig) should return a view sharing memory.");
        }
    }
}
