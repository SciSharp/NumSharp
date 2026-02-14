using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.View
{
    /// <summary>
    ///     Tests for Shape.IsContiguous and contiguous-slice view semantics.
    ///
    ///     Option 2 fix: contiguous step-1 slices should produce clean arrays
    ///     (no ViewInfo, IsContiguous=true) by using InternalArray.Slice(offset, count)
    ///     instead of creating ViewInfo-based aliases.
    ///
    ///     Tests are split into:
    ///       - Regression: values must be correct before AND after the fix
    ///       - Fix validation (Option2Fix): semantics that change with the fix
    ///
    ///     PYTHON VERIFICATION (NumPy 2.4.2):
    ///       All expected values and view/copy semantics verified against NumPy.
    /// </summary>
    public class ShapeIsContiguousTest
    {
        // ================================================================
        //  REGRESSION: Values must be correct before AND after the fix
        //  These tests ensure the fix doesn't break anything.
        // ================================================================

        #region Regression: 1D slices — values

        [Test]
        public void Slice1D_Step1_Values()
        {
            // np.arange(10)[2:7] = [2, 3, 4, 5, 6]
            var a = np.arange(10);
            var s = a["2:7"];
            s.shape.Should().BeEquivalentTo(new[] { 5 });
            s.GetInt32(0).Should().Be(2);
            s.GetInt32(1).Should().Be(3);
            s.GetInt32(2).Should().Be(4);
            s.GetInt32(3).Should().Be(5);
            s.GetInt32(4).Should().Be(6);
        }

        [Test]
        public void Slice1D_Step2_Values()
        {
            // np.arange(10)[::2] = [0, 2, 4, 6, 8]
            var a = np.arange(10);
            var s = a["::2"];
            s.shape.Should().BeEquivalentTo(new[] { 5 });
            s.GetInt32(0).Should().Be(0);
            s.GetInt32(1).Should().Be(2);
            s.GetInt32(2).Should().Be(4);
            s.GetInt32(3).Should().Be(6);
            s.GetInt32(4).Should().Be(8);
        }

        [Test]
        public void Slice1D_Reversed_Values()
        {
            // np.arange(5)[::-1] = [4, 3, 2, 1, 0]
            var a = np.arange(5);
            var s = a["::-1"];
            s.shape.Should().BeEquivalentTo(new[] { 5 });
            s.GetInt32(0).Should().Be(4);
            s.GetInt32(1).Should().Be(3);
            s.GetInt32(2).Should().Be(2);
            s.GetInt32(3).Should().Be(1);
            s.GetInt32(4).Should().Be(0);
        }

        [Test]
        public void Slice1D_SingleElement_Values()
        {
            // np.arange(10)[3:4] = [3]
            var a = np.arange(10);
            var s = a["3:4"];
            s.shape.Should().BeEquivalentTo(new[] { 1 });
            s.GetInt32(0).Should().Be(3);
        }

        #endregion

        #region Regression: 2D slices — values

        [Test]
        public void Slice2D_RowSlice_Values()
        {
            // np.arange(12).reshape(3,4)[1:3] = [[4,5,6,7],[8,9,10,11]]
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];
            s.shape.Should().BeEquivalentTo(new[] { 2, 4 });
            s.GetInt32(0, 0).Should().Be(4);
            s.GetInt32(0, 1).Should().Be(5);
            s.GetInt32(0, 2).Should().Be(6);
            s.GetInt32(0, 3).Should().Be(7);
            s.GetInt32(1, 0).Should().Be(8);
            s.GetInt32(1, 1).Should().Be(9);
            s.GetInt32(1, 2).Should().Be(10);
            s.GetInt32(1, 3).Should().Be(11);
        }

        [Test]
        public void Slice2D_ColumnSlice_Values()
        {
            // np.arange(12).reshape(3,4)[:,1:3] = [[1,2],[5,6],[9,10]]
            var a = np.arange(12).reshape(3, 4);
            var s = a[":,1:3"];
            s.shape.Should().BeEquivalentTo(new[] { 3, 2 });
            s.GetInt32(0, 0).Should().Be(1);
            s.GetInt32(0, 1).Should().Be(2);
            s.GetInt32(1, 0).Should().Be(5);
            s.GetInt32(1, 1).Should().Be(6);
            s.GetInt32(2, 0).Should().Be(9);
            s.GetInt32(2, 1).Should().Be(10);
        }

        [Test]
        public void Slice2D_SingleRow_Values()
        {
            // np.arange(12).reshape(3,4)[1:2] = [[4,5,6,7]]
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:2"];
            s.shape.Should().BeEquivalentTo(new[] { 1, 4 });
            s.GetInt32(0, 0).Should().Be(4);
            s.GetInt32(0, 1).Should().Be(5);
            s.GetInt32(0, 2).Should().Be(6);
            s.GetInt32(0, 3).Should().Be(7);
        }

        [Test]
        public void Slice2D_SingleRowPartialCol_Values()
        {
            // np.arange(12).reshape(3,4)[1:2,1:3] = [[5,6]]
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:2,1:3"];
            s.shape.Should().BeEquivalentTo(new[] { 1, 2 });
            s.GetInt32(0, 0).Should().Be(5);
            s.GetInt32(0, 1).Should().Be(6);
        }

        #endregion

        #region Regression: 3D slices — values

        [Test]
        public void Slice3D_RowSlice_Values()
        {
            // np.arange(24).reshape(2,3,4)[0:1] — shape (1,3,4), values 0..11
            var a = np.arange(24).reshape(2, 3, 4);
            var s = a["0:1"];
            s.shape.Should().BeEquivalentTo(new[] { 1, 3, 4 });
            for (int i = 0; i < 12; i++)
                s.GetAtIndex(i).Should().Be(i);
        }

        [Test]
        public void Slice3D_SingleRowPartialCol_Values()
        {
            // np.arange(24).reshape(2,3,4)[0:1,1:3,:] = [[4..11]]
            var a = np.arange(24).reshape(2, 3, 4);
            var s = a["0:1,1:3,:"];
            s.shape.Should().BeEquivalentTo(new[] { 1, 2, 4 });
            s.GetInt32(0, 0, 0).Should().Be(4);
            s.GetInt32(0, 0, 1).Should().Be(5);
            s.GetInt32(0, 0, 2).Should().Be(6);
            s.GetInt32(0, 0, 3).Should().Be(7);
            s.GetInt32(0, 1, 0).Should().Be(8);
            s.GetInt32(0, 1, 1).Should().Be(9);
            s.GetInt32(0, 1, 2).Should().Be(10);
            s.GetInt32(0, 1, 3).Should().Be(11);
        }

        [Test]
        public void Slice3D_MiddleDim_Values()
        {
            // np.arange(24).reshape(2,3,4)[:,1:2,:] — shape (2,1,4)
            // Row 0: [4,5,6,7], Row 1: [16,17,18,19]
            var a = np.arange(24).reshape(2, 3, 4);
            var s = a[":,1:2,:"];
            s.shape.Should().BeEquivalentTo(new[] { 2, 1, 4 });
            s.GetInt32(0, 0, 0).Should().Be(4);
            s.GetInt32(0, 0, 3).Should().Be(7);
            s.GetInt32(1, 0, 0).Should().Be(16);
            s.GetInt32(1, 0, 3).Should().Be(19);
        }

        #endregion

        #region Regression: Slice-of-slice — values

        [Test]
        public void SliceOfContiguousSlice_Values()
        {
            // np.arange(10)[2:8][1:4] = [3, 4, 5]
            var a = np.arange(10);
            var s1 = a["2:8"]; // [2,3,4,5,6,7]
            var s2 = s1["1:4"]; // [3,4,5]
            s2.shape.Should().BeEquivalentTo(new[] { 3 });
            s2.GetInt32(0).Should().Be(3);
            s2.GetInt32(1).Should().Be(4);
            s2.GetInt32(2).Should().Be(5);
        }

        [Test]
        public void SliceOfSteppedSlice_SingleElement_Values()
        {
            // np.arange(10)[::2][0:1] = [0]
            var a = np.arange(10);
            var stepped = a["::2"]; // [0,2,4,6,8]
            var s = stepped["0:1"];
            s.shape.Should().BeEquivalentTo(new[] { 1 });
            s.GetInt32(0).Should().Be(0);
        }

        [Test]
        public void SliceOfSteppedSlice_Range_Values()
        {
            // np.arange(10)[::2][1:3] = [2, 4]
            var a = np.arange(10);
            var stepped = a["::2"]; // [0,2,4,6,8]
            var s = stepped["1:3"];
            s.shape.Should().BeEquivalentTo(new[] { 2 });
            s.GetInt32(0).Should().Be(2);
            s.GetInt32(1).Should().Be(4);
        }

        #endregion

        #region Regression: Ravel values (must be correct regardless of view/copy)

        [Test]
        public void Ravel_ContiguousSlice1D_Values()
        {
            // np.ravel(np.arange(10)[2:7]) = [2,3,4,5,6]
            var a = np.arange(10);
            var s = a["2:7"];
            var r = np.ravel(s);
            r.shape.Should().BeEquivalentTo(new[] { 5 });
            r.GetInt32(0).Should().Be(2);
            r.GetInt32(1).Should().Be(3);
            r.GetInt32(2).Should().Be(4);
            r.GetInt32(3).Should().Be(5);
            r.GetInt32(4).Should().Be(6);
        }

        [Test]
        public void Ravel_ContiguousRowSlice2D_Values()
        {
            // np.ravel(np.arange(12).reshape(3,4)[1:3]) = [4,5,6,7,8,9,10,11]
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];
            var r = np.ravel(s);
            r.shape.Should().BeEquivalentTo(new[] { 8 });
            for (int i = 0; i < 8; i++)
                r.GetInt32(i).Should().Be(4 + i);
        }

        [Test]
        public void Ravel_ColumnSlice2D_Values()
        {
            // np.ravel(np.arange(12).reshape(3,4)[:,1:3]) = [1,2,5,6,9,10]
            var a = np.arange(12).reshape(3, 4);
            var s = a[":,1:3"];
            var r = np.ravel(s);
            r.shape.Should().BeEquivalentTo(new[] { 6 });
            r.GetInt32(0).Should().Be(1);
            r.GetInt32(1).Should().Be(2);
            r.GetInt32(2).Should().Be(5);
            r.GetInt32(3).Should().Be(6);
            r.GetInt32(4).Should().Be(9);
            r.GetInt32(5).Should().Be(10);
        }

        [Test]
        public void Ravel_SteppedSlice_Values()
        {
            // np.ravel(np.arange(10)[::2]) = [0,2,4,6,8]
            var a = np.arange(10);
            var s = a["::2"];
            var r = np.ravel(s);
            r.shape.Should().BeEquivalentTo(new[] { 5 });
            r.GetInt32(0).Should().Be(0);
            r.GetInt32(1).Should().Be(2);
            r.GetInt32(2).Should().Be(4);
            r.GetInt32(3).Should().Be(6);
            r.GetInt32(4).Should().Be(8);
        }

        #endregion

        #region Regression: Iterator correctness through slices

        [Test]
        public void Iterator_ContiguousSlice1D()
        {
            // Iterating through a contiguous slice must produce correct values
            var a = np.arange(10);
            var s = a["3:7"];
            var expected = new[] { 3, 4, 5, 6 };
            for (int i = 0; i < expected.Length; i++)
                s.GetAtIndex(i).Should().Be(expected[i], $"index {i}");
        }

        [Test]
        public void Iterator_ContiguousRowSlice2D()
        {
            var a = np.arange(20).reshape(4, 5);
            var s = a["1:3"]; // rows 1-2
            s.shape.Should().BeEquivalentTo(new[] { 2, 5 });
            for (int i = 0; i < 10; i++)
                s.GetAtIndex(i).Should().Be(5 + i, $"index {i}");
        }

        [Test]
        public void Iterator_NonContiguousColumnSlice2D()
        {
            var a = np.arange(12).reshape(3, 4);
            var s = a[":,1:3"];
            var expected = new[] { 1, 2, 5, 6, 9, 10 };
            for (int i = 0; i < expected.Length; i++)
                s.GetAtIndex(i).Should().Be(expected[i], $"index {i}");
        }

        [Test]
        public void Iterator_SteppedSlice()
        {
            var a = np.arange(10);
            var s = a["::3"]; // [0, 3, 6, 9]
            var expected = new[] { 0, 3, 6, 9 };
            s.size.Should().Be(4);
            for (int i = 0; i < expected.Length; i++)
                s.GetAtIndex(i).Should().Be(expected[i], $"index {i}");
        }

        #endregion

        #region Regression: np.roll on sliced arrays (uses ravel internally)

        [Test]
        public void Roll_OnContiguousSlice()
        {
            // np.roll(np.arange(10)[2:7], 2) = [5, 6, 2, 3, 4]
            var a = np.arange(10);
            var s = a["2:7"]; // [2,3,4,5,6]
            var r = np.roll(s, 2);
            r.shape.Should().BeEquivalentTo(new[] { 5 });
            r.GetInt32(0).Should().Be(5);
            r.GetInt32(1).Should().Be(6);
            r.GetInt32(2).Should().Be(2);
            r.GetInt32(3).Should().Be(3);
            r.GetInt32(4).Should().Be(4);
        }

        [Test]
        public void Roll_OnRowSlice2D()
        {
            // np.roll(np.arange(12).reshape(3,4)[1:3], 1, axis=1)
            // [[4,5,6,7],[8,9,10,11]] rolled axis=1 by 1 = [[7,4,5,6],[11,8,9,10]]
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];
            var r = np.roll(s, 1, 1);
            r.GetInt32(0, 0).Should().Be(7);
            r.GetInt32(0, 1).Should().Be(4);
            r.GetInt32(0, 2).Should().Be(5);
            r.GetInt32(0, 3).Should().Be(6);
            r.GetInt32(1, 0).Should().Be(11);
            r.GetInt32(1, 1).Should().Be(8);
            r.GetInt32(1, 2).Should().Be(9);
            r.GetInt32(1, 3).Should().Be(10);
        }

        #endregion

        // ================================================================
        //  FIX VALIDATION: Tests that fail BEFORE and pass AFTER Option 2
        //  These assert the correct NumPy behavior for view/copy semantics
        //  and IsContiguous. Tagged Option2Fix for filtering.
        // ================================================================

        #region Fix validation: IsContiguous should be true for contiguous slices

        [Test]
                public void IsContiguous_Step1Slice1D()
        {
            // NumPy: a[2:7].flags['C_CONTIGUOUS'] = True
            var a = np.arange(10);
            var s = a["2:7"];
            s.Shape.IsContiguous.Should().BeTrue(
                "step-1 slice [2:7] is contiguous in memory");
        }

        [Test]
                public void IsContiguous_RowSlice2D()
        {
            // NumPy: a[1:3].flags['C_CONTIGUOUS'] = True
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];
            s.Shape.IsContiguous.Should().BeTrue(
                "row slice [1:3] of C-contiguous 2D array is contiguous");
        }

        [Test]
                public void IsContiguous_SingleRow2D()
        {
            // NumPy: a[1:2].flags['C_CONTIGUOUS'] = True
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:2"];
            s.Shape.IsContiguous.Should().BeTrue(
                "single row slice [1:2] is contiguous");
        }

        [Test]
                public void IsContiguous_SingleRowPartialCol2D()
        {
            // NumPy: a[1:2,1:3].flags['C_CONTIGUOUS'] = True
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:2,1:3"];
            s.Shape.IsContiguous.Should().BeTrue(
                "single row with partial columns [1:2,1:3] is contiguous");
        }

        [Test]
                public void IsContiguous_SingleElement1D()
        {
            // NumPy: a[3:4].flags['C_CONTIGUOUS'] = True
            var a = np.arange(10);
            var s = a["3:4"];
            s.Shape.IsContiguous.Should().BeTrue(
                "single element slice [3:4] is contiguous");
        }

        [Test]
                public void IsContiguous_3D_RowSlice()
        {
            // NumPy: a[0:1].flags['C_CONTIGUOUS'] = True
            var a = np.arange(24).reshape(2, 3, 4);
            var s = a["0:1"];
            s.Shape.IsContiguous.Should().BeTrue(
                "3D row slice [0:1] is contiguous");
        }

        [Test]
                public void IsContiguous_3D_SingleRowPartialCol()
        {
            // NumPy: a[0:1,1:3,:].flags['C_CONTIGUOUS'] = True
            var a = np.arange(24).reshape(2, 3, 4);
            var s = a["0:1,1:3,:"];
            s.Shape.IsContiguous.Should().BeTrue(
                "3D single-row partial-col [0:1,1:3,:] is contiguous");
        }

        [Test]
                public void IsContiguous_SliceOfContiguousSlice()
        {
            // np.arange(10)[2:8][1:4] — merged step-1, contiguous
            var a = np.arange(10);
            var s = a["2:8"]["1:4"];
            s.Shape.IsContiguous.Should().BeTrue(
                "slice of a contiguous slice should be contiguous");
        }

        [Test]
                public void IsContiguous_SliceOfSteppedSlice_SingleElement()
        {
            // np.arange(10)[::2][0:1] — merged step=2 but count=1 (contiguous)
            var a = np.arange(10);
            var s = a["::2"]["0:1"];
            s.Shape.IsContiguous.Should().BeTrue(
                "single element from stepped slice is contiguous (count=1)");
        }

        #endregion

        #region Fix validation: IsContiguous should be false for non-contiguous slices

        [Test]
        public void IsContiguous_Step2Slice_False()
        {
            // NumPy: a[::2].flags['C_CONTIGUOUS'] = False
            var a = np.arange(10);
            var s = a["::2"];
            s.Shape.IsContiguous.Should().BeFalse(
                "step-2 slice has gaps in memory");
        }

        [Test]
        public void IsContiguous_ReversedSlice_False()
        {
            // NumPy: a[::-1].flags['C_CONTIGUOUS'] = False
            var a = np.arange(5);
            var s = a["::-1"];
            s.Shape.IsContiguous.Should().BeFalse(
                "reversed slice is not contiguous");
        }

        [Test]
        public void IsContiguous_ColumnSlice_False()
        {
            // NumPy: a[:,1:3].flags['C_CONTIGUOUS'] = False
            var a = np.arange(12).reshape(3, 4);
            var s = a[":,1:3"];
            s.Shape.IsContiguous.Should().BeFalse(
                "column slice has gaps between rows");
        }

        [Test]
        public void IsContiguous_3D_MiddleDim_False()
        {
            // NumPy: a[:,1:2,:].flags['C_CONTIGUOUS'] = False
            var a = np.arange(24).reshape(2, 3, 4);
            var s = a[":,1:2,:"];
            s.Shape.IsContiguous.Should().BeFalse(
                "middle-dim slice of 3D is not contiguous when outer dim > 1");
        }

        [Test]
        public void IsContiguous_MultiRowPartialCol_False()
        {
            // NumPy: a[0:2,1:3].flags['C_CONTIGUOUS'] = False
            var a = np.arange(12).reshape(3, 4);
            var s = a["0:2,1:3"];
            s.Shape.IsContiguous.Should().BeFalse(
                "multiple rows with partial columns is not contiguous");
        }

        [Test]
        public void IsContiguous_SliceOfSteppedSlice_Range_False()
        {
            // np.arange(10)[::2][1:3] — merged step=2, count=2 (not contiguous)
            var a = np.arange(10);
            var s = a["::2"]["1:3"];
            s.Shape.IsContiguous.Should().BeFalse(
                "range from stepped slice inherits step>1");
        }

        [Test]
        public void IsContiguous_Broadcast_False()
        {
            // Broadcast is never contiguous
            var a = np.broadcast_to(np.array(new[] { 1, 2, 3 }), new Shape(3, 3));
            a.Shape.IsContiguous.Should().BeFalse(
                "broadcast arrays have stride=0 dims");
        }

        #endregion

        #region Fix validation: Contiguous slices should share memory (view semantics)

        [Test]
                public void ViewSemantics_Step1Slice1D_MutationPropagates()
        {
            // NumPy: s = a[2:7]; s[0] = 999; a[2] == 999
            var a = np.arange(10);
            var s = a["2:7"];
            s.SetInt32(999, 0);
            a.GetInt32(2).Should().Be(999,
                "contiguous slice should share memory with original");
        }

        [Test]
                public void ViewSemantics_RowSlice2D_MutationPropagates()
        {
            // NumPy: s = a[1:3]; s[0,0] = 999; a[1,0] == 999
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];
            s.SetInt32(999, 0, 0);
            a.GetInt32(1, 0).Should().Be(999,
                "contiguous row slice should share memory with original");
        }

        [Test]
                public void ViewSemantics_SingleRowPartialCol_MutationPropagates()
        {
            // NumPy: s = a[1:2,1:3]; s[0,0] = 999; a[1,1] == 999
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:2,1:3"];
            s.SetInt32(999, 0, 0);
            a.GetInt32(1, 1).Should().Be(999,
                "contiguous single-row partial-col slice should share memory");
        }

        [Test]
                public void ViewSemantics_SliceOfContiguousSlice_MutationPropagates()
        {
            // NumPy: s = a[2:8][1:4]; s[0] = 999; a[3] == 999
            var a = np.arange(10);
            var s = a["2:8"]["1:4"];
            s.SetInt32(999, 0);
            a.GetInt32(3).Should().Be(999,
                "slice of contiguous slice should share memory with root");
        }

        #endregion

        #region Fix validation: Non-contiguous slices should NOT share memory

        [Test]
        public void ViewSemantics_SteppedSlice_MutationDoesNotPropagate()
        {
            // NumPy: s = np.ravel(a[::2]); s is a copy
            var a = np.arange(10);
            var s = a["::2"];
            var r = np.ravel(s);
            r.SetInt32(999, 0);
            a.GetInt32(0).Should().Be(0,
                "ravel of stepped slice should be a copy");
        }

        [Test]
        public void ViewSemantics_ColumnSlice_RavelIsCopy()
        {
            var a = np.arange(12).reshape(3, 4);
            var s = a[":,1:3"];
            var r = np.ravel(s);
            r.SetInt32(999, 0);
            a.GetInt32(0, 1).Should().Be(1,
                "ravel of column slice should be a copy");
        }

        #endregion

        #region Fix validation: Ravel of contiguous slice should be a view

        [Test]
                public void Ravel_ContiguousSlice1D_IsView()
        {
            // NumPy: r = np.ravel(a[2:7]); r is a view
            var a = np.arange(10);
            var s = a["2:7"];
            var r = np.ravel(s);
            r.SetInt32(999, 0);
            a.GetInt32(2).Should().Be(999,
                "ravel of contiguous slice should return a view");
        }

        [Test]
                public void Ravel_ContiguousRowSlice2D_IsView()
        {
            // NumPy: r = np.ravel(a[1:3]); r is a view
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];
            var r = np.ravel(s);
            r.SetInt32(999, 0);
            a.GetInt32(1, 0).Should().Be(999,
                "ravel of contiguous row slice should return a view");
        }

        #endregion

        #region Fix validation: Downstream consumers — copyto, unique

        [Test]
                public void Copyto_ContiguousSlice_FastPath()
        {
            // np.copyto with contiguous slices should use fast memcpy path
            var src = np.arange(10)["2:7"]; // [2,3,4,5,6]
            var dst = np.zeros(new Shape(5), np.int32);
            np.copyto(dst, src);
            for (int i = 0; i < 5; i++)
                dst.GetInt32(i).Should().Be(2 + i);
        }

        #endregion

        #region Fix validation: Multiple dtypes through contiguous slice path

        [Test]
                public void ContiguousSlice_Float64_Values()
        {
            var a = np.arange(10.0); // float64
            var s = a["2:7"];
            s.Shape.IsContiguous.Should().BeTrue();
            s.GetDouble(0).Should().Be(2.0);
            s.GetDouble(4).Should().Be(6.0);
        }

        [Test]
                public void ContiguousSlice_Float32_Values()
        {
            var a = np.arange(10).astype(np.float32);
            var s = a["2:7"];
            s.Shape.IsContiguous.Should().BeTrue();
            s.GetSingle(0).Should().Be(2.0f);
            s.GetSingle(4).Should().Be(6.0f);
        }

        [Test]
                public void ContiguousSlice_Byte_Values()
        {
            var a = np.arange(10).astype(np.uint8);
            var s = a["2:7"];
            s.Shape.IsContiguous.Should().BeTrue();
            s.GetByte(0).Should().Be(2);
            s.GetByte(4).Should().Be(6);
        }

        [Test]
                public void ContiguousSlice_Int64_Values()
        {
            var a = np.arange(10).astype(np.int64);
            var s = a["2:7"];
            s.Shape.IsContiguous.Should().BeTrue();
            s.GetInt64(0).Should().Be(2);
            s.GetInt64(4).Should().Be(6);
        }

        #endregion

        #region Edge cases

        [Test]
        public void EmptySlice_Values()
        {
            // np.arange(10)[5:5] = []
            var a = np.arange(10);
            var s = a["5:5"];
            s.size.Should().Be(0);
        }

        [Test]
                public void FullSlice_IsContiguous()
        {
            // np.arange(10)[:] should be contiguous (it's all elements)
            var a = np.arange(10);
            var s = a[":"];
            s.Shape.IsContiguous.Should().BeTrue(
                "full slice [:] is the same as the original");
            s.size.Should().Be(10);
        }

        [Test]
                public void ContiguousSlice_ThenReshape_Values()
        {
            // np.arange(12)[2:10].reshape(2,4) should work and be contiguous
            var a = np.arange(12);
            var s = a["2:10"]; // [2,3,4,5,6,7,8,9]
            var r = s.reshape(2, 4);
            r.shape.Should().BeEquivalentTo(new[] { 2, 4 });
            r.GetInt32(0, 0).Should().Be(2);
            r.GetInt32(0, 3).Should().Be(5);
            r.GetInt32(1, 0).Should().Be(6);
            r.GetInt32(1, 3).Should().Be(9);
        }

        #endregion
    }
}
