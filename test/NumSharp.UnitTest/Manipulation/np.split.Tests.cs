using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Tests for np.split and np.array_split functions.
    /// Based on NumPy 2.4.2 behavior.
    /// </summary>
    [TestClass]
    public class np_split_Tests : TestClass
    {
        #region np.split - Equal Division

        [TestMethod]
        public void Split_1D_EqualParts()
        {
            // NumPy: np.split(np.arange(9), 3) -> [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
            var a = np.arange(9);
            var result = np.split(a, 3);

            result.Length.Should().Be(3);
            result[0].ToArray<long>().Should().ContainInOrder(0, 1, 2);
            result[1].ToArray<long>().Should().ContainInOrder(3, 4, 5);
            result[2].ToArray<long>().Should().ContainInOrder(6, 7, 8);
        }

        [TestMethod]
        public void Split_1D_UnequalDivision_ThrowsArgumentException()
        {
            // NumPy: np.split(np.arange(9), 4) raises ValueError
            var a = np.arange(9);

            Assert.ThrowsException<ArgumentException>(() => np.split(a, 4));
        }

        [TestMethod]
        public void Split_2D_Axis0()
        {
            // NumPy: np.split(np.arange(12).reshape(3, 4), 3, axis=0)
            // -> [array([[0,1,2,3]]), array([[4,5,6,7]]), array([[8,9,10,11]])]
            var d = np.arange(12).reshape(3, 4);
            var result = np.split(d, 3, axis: 0);

            result.Length.Should().Be(3);
            result[0].Should().BeShaped(1, 4);
            result[1].Should().BeShaped(1, 4);
            result[2].Should().BeShaped(1, 4);
        }

        [TestMethod]
        public void Split_2D_Axis1()
        {
            // NumPy: np.split(np.arange(12).reshape(3, 4), 2, axis=1)
            var d = np.arange(12).reshape(3, 4);
            var result = np.split(d, 2, axis: 1);

            result.Length.Should().Be(2);
            result[0].Should().BeShaped(3, 2);
            result[1].Should().BeShaped(3, 2);

            // First result: columns 0-1
            result[0]["0, :"].ToArray<long>().Should().ContainInOrder(0, 1);
            result[0]["1, :"].ToArray<long>().Should().ContainInOrder(4, 5);

            // Second result: columns 2-3
            result[1]["0, :"].ToArray<long>().Should().ContainInOrder(2, 3);
            result[1]["1, :"].ToArray<long>().Should().ContainInOrder(6, 7);
        }

        [TestMethod]
        public void Split_NegativeAxis()
        {
            // NumPy: np.split(d, 3, axis=-2) == np.split(d, 3, axis=0)
            var d = np.arange(12).reshape(3, 4);
            var r1 = np.split(d, 3, axis: -2);
            var r2 = np.split(d, 3, axis: 0);

            r1.Length.Should().Be(r2.Length);
            for (int i = 0; i < r1.Length; i++)
            {
                ((bool)np.array_equal(r1[i], r2[i])).Should().BeTrue();
            }
        }

        #endregion

        #region np.split - With Indices

        [TestMethod]
        public void Split_WithIndices()
        {
            // NumPy: np.split(a, [3, 5, 7]) -> a[:3], a[3:5], a[5:7], a[7:]
            var a = np.arange(9);
            var result = np.split(a, new[] { 3, 5, 7 });

            result.Length.Should().Be(4);
            result[0].ToArray<long>().Should().ContainInOrder(0, 1, 2);
            result[1].ToArray<long>().Should().ContainInOrder(3, 4);
            result[2].ToArray<long>().Should().ContainInOrder(5, 6);
            result[3].ToArray<long>().Should().ContainInOrder(7, 8);
        }

        [TestMethod]
        public void Split_WithIndices_EmptyAtStart()
        {
            // NumPy: np.split(a, [0, 3, 6]) -> [], a[:3], a[3:6], a[6:]
            var a = np.arange(9);
            var result = np.split(a, new[] { 0, 3, 6 });

            result.Length.Should().Be(4);
            result[0].size.Should().Be(0);
            result[1].size.Should().Be(3);
            result[2].size.Should().Be(3);
            result[3].size.Should().Be(3);
        }

        [TestMethod]
        public void Split_WithIndices_DuplicateIndices()
        {
            // NumPy: np.split(a, [3, 3, 6]) -> a[:3], [], a[3:6], a[6:]
            var a = np.arange(9);
            var result = np.split(a, new[] { 3, 3, 6 });

            result.Length.Should().Be(4);
            result[0].size.Should().Be(3);
            result[1].size.Should().Be(0);
            result[2].size.Should().Be(3);
            result[3].size.Should().Be(3);
        }

        [TestMethod]
        public void Split_WithIndices_EmptyIndicesArray()
        {
            // NumPy: np.split(a, []) -> [a]
            var a = np.arange(9);
            var result = np.split(a, new int[] { });

            result.Length.Should().Be(1);
            result[0].size.Should().Be(9);
        }

        #endregion

        #region np.array_split - Unequal Division

        [TestMethod]
        public void ArraySplit_UnequalDivision_9Into4()
        {
            // NumPy: np.array_split(np.arange(9), 4)
            // -> [array([0, 1, 2]), array([3, 4]), array([5, 6]), array([7, 8])]
            // 9 % 4 = 1 extra, so first 1 section gets size 3, rest get size 2
            var a = np.arange(9);
            var result = np.array_split(a, 4);

            result.Length.Should().Be(4);
            result[0].size.Should().Be(3); // 9//4 + 1 = 3
            result[1].size.Should().Be(2); // 9//4 = 2
            result[2].size.Should().Be(2);
            result[3].size.Should().Be(2);
        }

        [TestMethod]
        public void ArraySplit_UnequalDivision_10Into3()
        {
            // NumPy: np.array_split(np.arange(10), 3)
            // -> [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
            // 10 % 3 = 1 extra, so first 1 section gets size 4, rest get size 3
            var b = np.arange(10);
            var result = np.array_split(b, 3);

            result.Length.Should().Be(3);
            result[0].size.Should().Be(4); // 10//3 + 1 = 4
            result[1].size.Should().Be(3); // 10//3 = 3
            result[2].size.Should().Be(3);
        }

        [TestMethod]
        public void ArraySplit_UnequalDivision_8Into3()
        {
            // NumPy: np.array_split(np.arange(8), 3)
            // -> [array([0, 1, 2]), array([3, 4, 5]), array([6, 7])]
            // 8 % 3 = 2 extras, so first 2 sections get size 3, rest get size 2
            var c = np.arange(8);
            var result = np.array_split(c, 3);

            result.Length.Should().Be(3);
            result[0].size.Should().Be(3); // 8//3 + 1 = 3
            result[1].size.Should().Be(3); // 8//3 + 1 = 3
            result[2].size.Should().Be(2); // 8//3 = 2
        }

        [TestMethod]
        public void ArraySplit_2D_UnequalAxis0()
        {
            // NumPy: np.array_split(np.arange(12).reshape(3, 4), 2, axis=0)
            // -> [array([[0,1,2,3],[4,5,6,7]]), array([[8,9,10,11]])]
            var d = np.arange(12).reshape(3, 4);
            var result = np.array_split(d, 2, axis: 0);

            result.Length.Should().Be(2);
            result[0].Should().BeShaped(2, 4);
            result[1].Should().BeShaped(1, 4);
        }

        [TestMethod]
        public void ArraySplit_MoreSectionsThanElements()
        {
            // NumPy: np.array_split(np.array([1,2,3]), 5)
            // -> [array([1]), array([2]), array([3]), array([]), array([])]
            var small = np.array(new[] { 1, 2, 3 });
            var result = np.array_split(small, 5);

            result.Length.Should().Be(5);
            result[0].size.Should().Be(1);
            result[1].size.Should().Be(1);
            result[2].size.Should().Be(1);
            result[3].size.Should().Be(0);
            result[4].size.Should().Be(0);
        }

        [TestMethod]
        public void ArraySplit_ZeroSections_ThrowsArgumentException()
        {
            // NumPy: np.array_split(a, 0) raises ValueError
            var a = np.arange(9);

            Assert.ThrowsException<ArgumentException>(() => np.array_split(a, 0));
        }

        [TestMethod]
        public void ArraySplit_NegativeSections_ThrowsArgumentException()
        {
            // NumPy: np.array_split(a, -1) raises ValueError
            var a = np.arange(9);

            Assert.ThrowsException<ArgumentException>(() => np.array_split(a, -1));
        }

        #endregion

        #region Edge Cases

        [TestMethod]
        public void Split_SingleElement()
        {
            var single = np.array(new[] { 42 });
            var result = np.split(single, 1);

            result.Length.Should().Be(1);
            result[0].ToArray<int>().Should().ContainInOrder(42);
        }

        [TestMethod]
        public void Split_EmptyArray()
        {
            var empty = np.array(new double[0]);
            var result = np.split(empty, 1);

            result.Length.Should().Be(1);
            result[0].size.Should().Be(0);
        }

        [TestMethod]
        public void ArraySplit_EmptyArray_MultipleSections()
        {
            var empty = np.array(new double[0]);
            var result = np.array_split(empty, 3);

            result.Length.Should().Be(3);
            foreach (var r in result)
            {
                r.size.Should().Be(0);
            }
        }

        [TestMethod]
        public void Split_3D_Array()
        {
            // NumPy: np.split(np.arange(24).reshape(2, 3, 4), 2, axis=0)
            var e = np.arange(24).reshape(2, 3, 4);
            var result = np.split(e, 2, axis: 0);

            result.Length.Should().Be(2);
            result[0].Should().BeShaped(1, 3, 4);
            result[1].Should().BeShaped(1, 3, 4);
        }

        [TestMethod]
        public void Split_ReturnsViews()
        {
            // NumPy: split returns views, not copies
            var a = np.arange(9);
            var result = np.split(a, 3);

            // Modifying a view should affect the original
            result[0].SetInt64(999, 0);
            a.GetInt64(0).Should().Be(999);
        }

        [TestMethod]
        public void Split_DifferentDtypes()
        {
            // Test with float64
            var f = np.arange(6.0);
            var result = np.split(f, 3);

            result.Length.Should().Be(3);
            result[0].dtype.Should().Be(typeof(double));
            result[0].ToArray<double>().Should().ContainInOrder(0.0, 1.0);
        }

        #endregion

        #region NumPy Slice-Semantics On Indices

        [TestMethod]
        public void Split_Indices_NegativeIndexWrapsLikePythonSlice()
        {
            // NumPy: np.split(arange(5), [-1]) -> [arange(4), [4]]. -1 wraps to N-1.
            var a = np.arange(5);
            var result = np.split(a, new int[] { -1 });

            result.Length.Should().Be(2);
            result[0].ToArray<long>().Should().ContainInOrder(0, 1, 2, 3);
            result[1].ToArray<long>().Should().ContainInOrder(4);
        }

        [TestMethod]
        public void Split_Indices_UnsortedReturnsEmptyBetween()
        {
            // NumPy: np.split(arange(5), [3, 2]) -> [[0,1,2], [], [2,3,4]].
            // 3 > 2 means the middle slice [3:2] is empty (Python slice semantics).
            var a = np.arange(5);
            var result = np.split(a, new int[] { 3, 2 });

            result.Length.Should().Be(3);
            result[0].ToArray<long>().Should().ContainInOrder(0, 1, 2);
            result[1].size.Should().Be(0);
            result[2].ToArray<long>().Should().ContainInOrder(2, 3, 4);
        }

        [TestMethod]
        public void Split_Indices_OutOfBoundReturnsEmpty()
        {
            // NumPy: np.split(arange(5), [10]) -> [arange(5), []]. Beyond-N clamps to N.
            var a = np.arange(5);
            var result = np.split(a, new int[] { 10 });

            result.Length.Should().Be(2);
            result[0].ToArray<long>().Should().ContainInOrder(0, 1, 2, 3, 4);
            result[1].size.Should().Be(0);
        }

        [TestMethod]
        public void Split_Indices_RepeatedReturnsAllEmptyButOne()
        {
            // NumPy: np.split(arange(5), [3,3,3]) -> [[0,1,2], [], [], [3,4]].
            // Successive identical indices yield empty slices between them.
            var a = np.arange(5);
            var result = np.split(a, new int[] { 3, 3, 3 });

            result.Length.Should().Be(4);
            result[0].ToArray<long>().Should().ContainInOrder(0, 1, 2);
            result[1].size.Should().Be(0);
            result[2].size.Should().Be(0);
            result[3].ToArray<long>().Should().ContainInOrder(3, 4);
        }

        [TestMethod]
        public void Split_Indices_AtZeroReturnsLeadingEmpty()
        {
            // NumPy: np.split(arange(5), [0]) -> [[], arange(5)]. Cut at index 0
            // makes the first sub-array empty.
            var a = np.arange(5);
            var result = np.split(a, new int[] { 0 });

            result.Length.Should().Be(2);
            result[0].size.Should().Be(0);
            result[1].ToArray<long>().Should().ContainInOrder(0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void Split_Indices_AtNReturnsTrailingEmpty()
        {
            // NumPy: np.split(arange(5), [5]) -> [arange(5), []].
            var a = np.arange(5);
            var result = np.split(a, new int[] { 5 });

            result.Length.Should().Be(2);
            result[0].ToArray<long>().Should().ContainInOrder(0, 1, 2, 3, 4);
            result[1].size.Should().Be(0);
        }

        [TestMethod]
        public void Split_Indices_LongOverloadEquivalentToInt()
        {
            // long[] and int[] overloads must produce identical results.
            var a = np.arange(8);
            var fromInts = np.split(a, new int[] { 3, 5 });
            var fromLongs = np.split(a, new long[] { 3L, 5L });

            fromInts.Length.Should().Be(fromLongs.Length);
            for (int i = 0; i < fromInts.Length; i++)
                fromInts[i].ToArray<long>().Should().Equal(fromLongs[i].ToArray<long>());
        }

        #endregion

        #region Argument Validation

        [TestMethod]
        public void Split_ZeroSections_ThrowsArgumentException()
        {
            // Pre-fix: this threw DivideByZeroException because we did N % 0 before
            // validating sections > 0. Now both split() and array_split() throw
            // ArgumentException consistently.
            var a = np.arange(9);
            Assert.ThrowsException<ArgumentException>(() => np.split(a, 0));
        }

        [TestMethod]
        public void Split_NegativeSections_ThrowsArgumentException()
        {
            var a = np.arange(9);
            Assert.ThrowsException<ArgumentException>(() => np.split(a, -1));
        }

        [TestMethod]
        public void Split_ZeroDimensional_ThrowsArgumentOutOfRange()
        {
            // NumPy: np.split(np.array(5.0), 1) raises IndexError because there is
            // no axis 0 on a 0-d array. We surface this as ArgumentOutOfRangeException
            // via NormalizeSplitAxis.
            var scalar = np.array(5.0);
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.split(scalar, 1));
        }

        [TestMethod]
        public void Split_AxisOutOfBounds_ThrowsArgumentOutOfRange()
        {
            var a = np.arange(12).reshape(3, 4);
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.split(a, 2, axis: 5));
        }

        [TestMethod]
        public void Split_NullArray_ThrowsArgumentNullException()
        {
            Assert.ThrowsException<ArgumentNullException>(() => np.split((NDArray)null, 2));
            Assert.ThrowsException<ArgumentNullException>(() => np.split((NDArray)null, new int[] { 1 }));
            Assert.ThrowsException<ArgumentNullException>(() => np.array_split((NDArray)null, 2));
            Assert.ThrowsException<ArgumentNullException>(() => np.array_split((NDArray)null, new int[] { 1 }));
        }

        #endregion

        #region View Semantics

        [TestMethod]
        public void Split_StridedInput_ProducesViews()
        {
            // sub-arrays of a strided slice are still views into the original
            // buffer; mutating one must affect the parent.
            var a = np.arange(20).reshape(4, 5);
            var stripe = a[":, ::2"]; // (4, 3) view, non-contig
            stripe.Shape.IsContiguous.Should().BeFalse();

            var parts = np.split(stripe, 2, axis: 0);
            parts.Length.Should().Be(2);
            parts[0].Should().BeShaped(2, 3);

            // Mutate through the sub-view; original should see the change.
            parts[0].SetInt64(999, 0, 0);
            a.GetInt64(0, 0).Should().Be(999);
        }

        [TestMethod]
        public void Split_ContigInput_OffsetsAdvanceCorrectly()
        {
            // Each sub-array should start at start*axisStride into the buffer.
            // For 2-D (3,4) C-contig array, axis=1 with 2 sections: stride[1]=1,
            // so sub[1].offset == 2 elements past sub[0].offset.
            var a = np.arange(12).reshape(3, 4);
            var parts = np.split(a, 2, axis: 1);

            parts.Length.Should().Be(2);
            // Sub 0: rows of (a[:, 0:2]) = first 2 cols.
            parts[0].GetInt64(0, 0).Should().Be(0);
            parts[0].GetInt64(0, 1).Should().Be(1);
            parts[0].GetInt64(2, 0).Should().Be(8);
            parts[0].GetInt64(2, 1).Should().Be(9);
            // Sub 1: (a[:, 2:4]).
            parts[1].GetInt64(0, 0).Should().Be(2);
            parts[1].GetInt64(0, 1).Should().Be(3);
            parts[1].GetInt64(2, 0).Should().Be(10);
            parts[1].GetInt64(2, 1).Should().Be(11);
        }

        [TestMethod]
        public void Split_AllPartsShareStorage()
        {
            // All sub-arrays should alias the same buffer; their per-part start
            // is encoded in shape.offset (element offset), not in Storage.Address
            // (which is the raw buffer base, identical across aliases).
            var a = np.arange(20.0);
            var parts = np.split(a, 4);
            unsafe
            {
                byte* basePtr = (byte*)a.Storage.Address;
                for (int i = 0; i < parts.Length; i++)
                {
                    byte* partPtr = (byte*)parts[i].Storage.Address;
                    ((long)partPtr).Should().Be((long)basePtr, $"part {i} aliases the same base buffer");
                    parts[i].Shape.offset.Should().Be(i * 5L, $"part {i} starts at element {i * 5}");
                }
            }
        }

        #endregion

        #region Dtype Coverage

        [TestMethod]
        public void Split_AllDtypes_PreservesDtypeAndContents()
        {
            // Split is a view-only operation — dtype must be preserved across
            // every dtype NumSharp supports.
            var types = new (System.Type, System.Func<int, object>)[]
            {
                (typeof(bool),    i => (i & 1) == 0),
                (typeof(byte),    i => (byte)i),
                (typeof(sbyte),   i => (sbyte)i),
                (typeof(short),   i => (short)i),
                (typeof(ushort),  i => (ushort)i),
                (typeof(int),     i => i),
                (typeof(uint),    i => (uint)i),
                (typeof(long),    i => (long)i),
                (typeof(ulong),   i => (ulong)i),
                (typeof(char),    i => (char)('a' + i)),
                (typeof(float),   i => (float)i),
                (typeof(double),  i => (double)i),
                (typeof(decimal), i => (decimal)i),
                (typeof(System.Numerics.Complex), i => new System.Numerics.Complex(i, 0)),
            };

            foreach (var (t, gen) in types)
            {
                var arr = System.Array.CreateInstance(t, 6);
                for (int i = 0; i < 6; i++)
                    arr.SetValue(System.Convert.ChangeType(gen(i), t, System.Globalization.CultureInfo.InvariantCulture), i);
                var nd = np.array(arr);

                var parts = np.split(nd, 3);
                parts.Length.Should().Be(3, $"dtype={t.Name}");
                parts[0].dtype.Should().Be(t, $"dtype preservation for {t.Name}");
                parts[0].size.Should().Be(2, $"shape preservation for {t.Name}");
                parts[1].size.Should().Be(2, $"shape preservation for {t.Name}");
                parts[2].size.Should().Be(2, $"shape preservation for {t.Name}");
            }
        }

        #endregion

        #region hsplit / vsplit / dsplit Sanity

        [TestMethod]
        public void Hsplit_1D_SplitsAxis0()
        {
            // hsplit on 1-D should behave like split with axis=0 (NumPy doc).
            var a = np.arange(6);
            var parts = np.hsplit(a, 2);
            parts.Length.Should().Be(2);
            parts[0].ToArray<long>().Should().ContainInOrder(0, 1, 2);
            parts[1].ToArray<long>().Should().ContainInOrder(3, 4, 5);
        }

        [TestMethod]
        public void Hsplit_2D_SplitsAxis1()
        {
            // 2-D hsplit -> axis=1 (columns).
            var a = np.arange(16).reshape(4, 4);
            var parts = np.hsplit(a, 2);
            parts.Length.Should().Be(2);
            parts[0].Should().BeShaped(4, 2);
            parts[1].Should().BeShaped(4, 2);
        }

        [TestMethod]
        public void Vsplit_RequiresAtLeast2D()
        {
            var a = np.arange(6);
            Assert.ThrowsException<ArgumentException>(() => np.vsplit(a, 2));
        }

        [TestMethod]
        public void Dsplit_RequiresAtLeast3D()
        {
            var a = np.arange(16).reshape(4, 4);
            Assert.ThrowsException<ArgumentException>(() => np.dsplit(a, 2));
        }

        [TestMethod]
        public void Dsplit_3D()
        {
            var a = np.arange(8).reshape(2, 2, 2);
            var parts = np.dsplit(a, 2);
            parts.Length.Should().Be(2);
            parts[0].Should().BeShaped(2, 2, 1);
            parts[0].GetInt64(0, 0, 0).Should().Be(0);
            parts[0].GetInt64(1, 1, 0).Should().Be(6);
            parts[1].GetInt64(0, 0, 0).Should().Be(1);
            parts[1].GetInt64(1, 1, 0).Should().Be(7);
        }

        [TestMethod]
        public void Hsplit_ZeroDimensional_Throws()
        {
            var scalar = np.array(5.0);
            Assert.ThrowsException<ArgumentException>(() => np.hsplit(scalar, 1));
        }

        #endregion

        #region Flag Inheritance (NumPy parity)

        [TestMethod]
        public void Split_ReadOnlyDiagonalView_SubsAreReadOnly()
        {
            // np.diagonal returns a read-only view (WRITEABLE flag cleared per
            // NumPy contract). Sub-arrays of that view must also be read-only.
            var m = np.arange(16).reshape(4, 4);
            var diag = np.diagonal(m);
            diag.Shape.IsWriteable.Should().BeFalse();

            var parts = np.split(diag, 2);
            foreach (var p in parts)
                p.Shape.IsWriteable.Should().BeFalse("read-only parents must yield read-only sub-arrays");
        }

        [TestMethod]
        public void Split_BroadcastedInput_SubsRetainBroadcastAndAreReadOnly()
        {
            // np.broadcast_to(shape (1,3) -> (4,3)) produces a stride-0 view that's
            // broadcasted along axis 0. Each sub-array of a split must keep the
            // BROADCASTED flag and stay non-writeable.
            var b = np.array(new double[] { 1, 2, 3 }).reshape(1, 3);
            var bcast = np.broadcast_to(b, new Shape(4, 3));
            bcast.Shape.IsBroadcasted.Should().BeTrue();
            bcast.Shape.IsWriteable.Should().BeFalse();

            var parts = np.split(bcast, 2, axis: 0);
            foreach (var p in parts)
            {
                p.Shape.IsBroadcasted.Should().BeTrue();
                p.Shape.IsWriteable.Should().BeFalse();
                p.Should().BeShaped(2, 3);
            }
        }

        [TestMethod]
        public void Split_FContiguousInput_AxisShrinksLoseFContig()
        {
            // F-contig (4,3) split on axis 0 shrinks the leading axis. For F-contig
            // the invariant `strides[i] = dims[i-1]*strides[i-1]` breaks at i=1
            // when dims[0] changes — sub is no longer F-contig (matches NumPy).
            var t = np.arange(12).reshape(3, 4).T;
            t.Shape.IsFContiguous.Should().BeTrue();

            var parts = np.split(t, 2, axis: 0);
            parts[0].Shape.IsFContiguous.Should().BeFalse(
                "shrinking axis 0 of an F-contig array breaks F-contig (NumPy parity)");
        }

        [TestMethod]
        public void Split_CContiguousInput_LeadingAxisPreservesCContig()
        {
            // C-contig array split on axis 0 keeps C-contig — the i=0 invariant
            // is the only one involving dim[0], and it's strictly satisfied for
            // the sub when strides remain unchanged.
            var a = np.arange(12).reshape(3, 4);
            a.Shape.IsContiguous.Should().BeTrue();

            var parts = np.split(a, 3, axis: 0);
            foreach (var p in parts)
                p.Shape.IsContiguous.Should().BeTrue(
                    "C-contig preserved when splitting along leading axis");
        }

        [TestMethod]
        public void Split_CContiguousInput_NonLeadingAxisLosesCContig()
        {
            // C-contig (3,4) split on axis 1 breaks the i=0 invariant
            // (strides[0]==4 but new dim[1]==2 → expected strides[0]==2).
            var a = np.arange(12).reshape(3, 4);
            a.Shape.IsContiguous.Should().BeTrue();

            var parts = np.split(a, 2, axis: 1);
            parts[0].Shape.IsContiguous.Should().BeFalse(
                "C-contig lost when splitting along a non-leading axis");
        }

        [TestMethod]
        public void Split_NegativeStride_ReversedInput()
        {
            // Reversed view ([::-1]) has stride=-1 and offset=N-1.
            // split should walk forward through the reversed elements.
            var n = np.arange(10)["::-1"];
            var parts = np.split(n, 5);

            parts.Length.Should().Be(5);
            parts[0].ToArray<long>().Should().ContainInOrder(9, 8);
            parts[1].ToArray<long>().Should().ContainInOrder(7, 6);
            parts[4].ToArray<long>().Should().ContainInOrder(1, 0);
        }

        [TestMethod]
        public void Split_FiveDimensional()
        {
            // High-rank: 5-D (2,3,4,5,1) split on axis 2 (size 4) into 2 pieces.
            // Verifies axis-removal logic works beyond 3-D.
            var x5 = np.arange(120).reshape(2, 3, 4, 5, 1);
            var parts = np.split(x5, 2, axis: 2);

            parts.Length.Should().Be(2);
            parts[0].Should().BeShaped(2, 3, 2, 5, 1);
            parts[1].Should().BeShaped(2, 3, 2, 5, 1);
        }

        [TestMethod]
        public void Split_EmptyInputOnEmptyAxis_PreservesShape()
        {
            // (0,3) array split into 1 → one (0,3) sub-array. NumPy parity.
            var e = np.zeros(new Shape(0, 3));
            var parts = np.split(e, 1);
            parts.Length.Should().Be(1);
            parts[0].Should().BeShaped(0, 3);
        }

        [TestMethod]
        public void Split_EmptyInput_NonEmptyAxisSubsAreEmpty()
        {
            // (5,0) array split into 5 pieces along axis 0 → all empty.
            var e = np.zeros(new Shape(5, 0));
            var parts = np.split(e, 5);
            parts.Length.Should().Be(5);
            foreach (var p in parts)
            {
                p.Should().BeShaped(1, 0);
                p.size.Should().Be(0);
            }
        }

        #endregion
    }
}
