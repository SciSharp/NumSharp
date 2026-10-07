using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The text calls NumPy 2.4.2 computes on text but answers from the SHAPE: a truth test, count, index search, sort,
    ///     arg-reduction, unique, set routine or equality test over an empty array, over lines of one element, or over a pair
    ///     the comparison operators have no loop for. NumSharp answers them now; the same calls over text NumPy reads stay
    ///     deferred (<see cref="StringComputationNotSupportedException"/>). Every expected value was probed against NumPy
    ///     2.4.2; the string oracle's <c>strshape</c> section replays the full matrix, these tests pin one representative of
    ///     each rule.
    /// </summary>
    [TestClass]
    public class StringShapeDecidedTests
    {
        private static NDArray E() => np.zeros(new Shape(0L), "U2");
        private static NDArray E2() => np.zeros(new Shape(0L, 3L), "U2");
        private static NDArray E3() => np.zeros(new Shape(3L, 0L), "U2");
        private static NDArray U() => np.array(new[] { "ab", "c", "de" });
        private static NDArray U1() => np.array(new[] { "b" });
        private static NDArray Col() => np.array(new[,] { { "b" }, { "a" } });   // lines of one element along axis 1
        private static NDArray Row() => np.array(new[,] { { "a", "b", "a" } });  // one slab along axis 0
        private static NDArray Bytes(params string[] s) => np.array(s.Select(x => (object)x.Select(c => (byte)c).ToArray()).ToArray(), dtype: "S");

        // ---------------------------------------------------------------- truth tests and counts

        [TestMethod]
        public void TruthTests_OfAnEmptyArray_AreTheIdentity()
        {
            // np.any(E) -> False, np.all(E) -> True, np.any(E2, axis=0) -> [False, False, False],
            // np.all(E2, axis=0, keepdims=True) -> [[True, True, True]], np.count_nonzero(E2, axis=0) -> [0, 0, 0]
            np.any(E()).Should().BeFalse();
            np.all(E()).Should().BeTrue();
            np.any(E2(), 0).ToArray<bool>().Should().Equal(false, false, false);
            var kept = np.all(E2(), (int?)0, keepdims: true);
            kept.shape.Should().Equal(new long[] { 1, 3 });
            kept.ToArray<bool>().Should().Equal(true, true, true);
            var counts = np.count_nonzero(E2(), 0);
            counts.dtype.Should().Be(np.int64);
            counts.ToArray<long>().Should().Equal(0L, 0L, 0L);
        }

        [TestMethod]
        public void TruthTest_OfAnEmptyArray_WritesTheIdentityIntoOut()
        {
            // o = np.ones(3, bool); np.any(E2, axis=0, out=o) -> o is [False, False, False]
            var o = np.ones(new Shape(3L), NPTypeCode.Boolean);
            var r = np.any(E2(), (int?)0, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.ToArray<bool>().Should().Equal(false, false, false);
        }

        [TestMethod]
        public void TruthTests_OfText_StillDefer()
        {
            Action any = () => np.any(U());
            any.Should().Throw<StringComputationNotSupportedException>().WithMessage("any is not supported*");
            Action count = () => np.count_nonzero(U(), 0);
            count.Should().Throw<StringComputationNotSupportedException>();
        }

        [TestMethod]
        public void TruthTest_OfAnEmptyArray_ChecksTheAxisFirst()
        {
            Action act = () => np.any(E2(), 5);
            act.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 2*");
        }

        // ---------------------------------------------------------------- index searches

        [TestMethod]
        public void IndexSearches_OfAnEmptyArray_AreEmptyIndices()
        {
            // np.nonzero(E2) -> (array([], int64), array([], int64)); np.argwhere(E2).shape -> (0, 2); np.flatnonzero(E2) -> []
            var nz = np.nonzero(E2());
            nz.Length.Should().Be(2);
            nz.All(x => x.size == 0 && x.dtype == np.int64).Should().BeTrue();
            np.argwhere(E2()).shape.Should().Equal(new long[] { 0, 2 });
            np.flatnonzero(E2()).size.Should().Be(0);
        }

        [TestMethod]
        public void Nonzero_OfA0dStringArray_IsNumPysValueError()
        {
            // NumPy refuses nonzero on a 0-d array before it reads the element; the text form has no arr[cond] hint.
            Action act = () => np.nonzero(np.array("a"));
            act.Should().Throw<ValueError>()
               .WithMessage("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.");
        }

        // ---------------------------------------------------------------- arg-reductions, sorts, partitions

        [TestMethod]
        public void ArgReductions_OverLinesOfOne_AreZero()
        {
            // np.argmax(col, axis=1) -> [0, 0]; np.argmin(['b']) -> 0; np.nanargmax(col, axis=1) -> [0, 0]
            np.argmax(Col(), 1).ToArray<long>().Should().Equal(0L, 0L);
            np.argmin(U1()).Should().Be(0L);
            np.nanargmax(Col(), (int?)1).ToArray<long>().Should().Equal(0L, 0L);
            // An empty result compares nothing: np.argmax(E2, axis=1) is an empty int64 array.
            var empty = np.argmax(E2(), 1);
            empty.dtype.Should().Be(np.int64);
            empty.shape.Should().Equal(new long[] { 0 });
        }

        [TestMethod]
        public void ArgReductions_OverAnEmptyLine_AndOverText()
        {
            Action emptyLine = () => np.argmax(E2(), 0);
            emptyLine.Should().Throw<ValueError>().WithMessage("attempt to get argmax of an empty sequence");
            Action text = () => np.argmax(U(), 0);
            text.Should().Throw<StringComputationNotSupportedException>();
        }

        [TestMethod]
        public void Sorts_OverLinesOfOne_KeepTheElements()
        {
            // np.sort(col, axis=1) -> [['b'], ['a']]; np.sort(np.array('ab'), axis=None) -> ['ab']; np.argsort(col, axis=1)
            // -> [[0], [0]]; np.argsort(np.array('ab')) -> [0] (a 0-d input ravels first)
            var sorted = np.sort(Col(), 1);
            sorted.dtype.str.Should().Be("<U1");
            sorted.item(0, 0).Should().Be("b");
            sorted.item(1, 0).Should().Be("a");
            np.sort(np.array("ab"), null).item(0).Should().Be("ab");
            np.argsort(Col(), 1).ToArray<long>().Should().Equal(0L, 0L);
            var arg0d = np.argsort(np.array("ab"));
            arg0d.shape.Should().Equal(new long[] { 1 });
            arg0d.GetInt64(0).Should().Be(0L);
        }

        [TestMethod]
        public void Sorts_RefuseA0dAxis_AndDeferText()
        {
            Action zeroD = () => np.sort(np.array("ab"));
            zeroD.Should().Throw<AxisError>().WithMessage("axis -1 is out of bounds for array of dimension 0*");
            Action text = () => np.sort(U());
            text.Should().Throw<StringComputationNotSupportedException>();
        }

        [TestMethod]
        public void Partitions_OverLinesOfOne_CheckKthAndKeepTheElements()
        {
            // np.partition(col, 0, axis=1) -> [['b'], ['a']]; kth=1 -> ValueError kth(=1) out of bounds (1);
            // np.argpartition(np.array('ab'), 0) -> [0]
            var p = np.partition(Col(), 0, 1);
            p.item(0, 0).Should().Be("b");
            Action kth = () => np.partition(Col(), 1, 1);
            kth.Should().Throw<ValueError>().WithMessage("kth(=1) out of bounds (1)");
            np.argpartition(np.array("ab"), 0).ToArray<long>().Should().Equal(0L);
        }

        // ---------------------------------------------------------------- unique

        [TestMethod]
        public void Unique_OfOneElement_IsTheElementWithItsIndices()
        {
            // np.unique(['b'], True, True, True) -> (['b'], [0], [0], [1])
            var r = (NDArray[])np.unique(U1(), true, true, true);
            r[0].item(0).Should().Be("b");
            r[1].ToArray<long>().Should().Equal(0L);
            r[2].ToArray<long>().Should().Equal(0L);
            r[3].ToArray<long>().Should().Equal(1L);
        }

        [TestMethod]
        public void Unique_OfZeroWidthSlabs_IsOneSlab()
        {
            // np.unique(np.zeros((3, 0), 'U2'), True, True, True, axis=0) -> (shape (1, 0), [0], [0, 0, 0], [3])
            var r = (NDArray[])np.unique(E3(), true, true, true, 0);
            r[0].shape.Should().Equal(new long[] { 1, 0 });
            r[0].dtype.str.Should().Be("<U2");
            r[1].ToArray<long>().Should().Equal(0L);
            r[2].ToArray<long>().Should().Equal(0L, 0L, 0L);
            r[3].ToArray<long>().Should().Equal(3L);
        }

        [TestMethod]
        public void Unique_OfOneSlab_IsTheInputAsAView()
        {
            // np.unique(row, axis=0) is the row itself, built (like NumPy's reshape_uniq) as a moveaxis view.
            var r = (NDArray)np.unique(Row(), axis: 0);
            r.shape.Should().Equal(new long[] { 1, 3 });
            r.item(0, 1).Should().Be("b");
            r.flags.owndata.Should().BeFalse();
        }

        [TestMethod]
        public void Unique_OfAnEmpty2DArray_InverseHasTheInputShape()
        {
            // np.unique(np.zeros((0, 3), 'U2'), return_inverse=True)[1].shape -> (0, 3)
            var r = (NDArray[])np.unique(E2(), false, true);
            r[0].shape.Should().Equal(new long[] { 0 });
            r[1].shape.Should().Equal(new long[] { 0, 3 });
        }

        [TestMethod]
        public void Unique_OfText_Defers()
        {
            Action values = () => np.unique(U());
            values.Should().Throw<StringComputationNotSupportedException>().WithMessage("unique is not supported*");
            Action api = () => np.unique_values(U());
            api.Should().Throw<StringComputationNotSupportedException>().WithMessage("unique_values is not supported*");
        }

        // ---------------------------------------------------------------- isin and the set routines

        [TestMethod]
        public void Isin_DecidedBySizesAndDtypes()
        {
            // np.isin(E, U) -> []; np.isin(U, E) -> [F, F, F] (invert: [T, T, T]); np.isin(U, [1.5, 2.0]) -> [F, F, F]
            // (equal has no loop for str and float64, so the loop path's comparisons are all False)
            np.isin(E(), U()).size.Should().Be(0);
            np.isin(U(), E()).ToArray<bool>().Should().Equal(false, false, false);
            np.isin(U(), E(), invert: true).ToArray<bool>().Should().Equal(true, true, true);
            np.isin(U(), np.array(new[] { 1.5, 2.0 })).ToArray<bool>().Should().Equal(false, false, false);
            np.isin(U(), Bytes("x", "y")).ToArray<bool>().Should().Equal(false, false, false);
            // NumPy reshapes _isin's answer: the result is a view.
            np.isin(U(), E()).flags.owndata.Should().BeFalse();
        }

        [TestMethod]
        public void Isin_ThatComparesText_Defers_AndTableRefuses()
        {
            Action same = () => np.isin(U(), U());
            same.Should().Throw<StringComputationNotSupportedException>();
            // 20 numbers take the sort path, which formats them as text: np.isin(['1'], np.arange(20)) is [True].
            Action sortPath = () => np.isin(U(), np.arange(20));
            sortPath.Should().Throw<StringComputationNotSupportedException>();
            Action table = () => np.isin(U(), U(), kind: "table");
            table.Should().Throw<ValueError>().WithMessage("The 'table' method is only supported for boolean or integer arrays.*");
        }

        [TestMethod]
        public void Isin_OfAnEmptyStrArray_AgainstBytes_DependsOnTheBytes()
        {
            // ASCII bytes decode without fail: an empty answer. A non-ASCII byte fails NumPy's decode after it sorts the
            // bytes (its error names the first failing element of the sorted text): deferred.
            np.isin(E(), Bytes("x", "y")).size.Should().Be(0);
            var nonAscii = np.array(new object[] { new byte[] { 0xff }, new byte[] { (byte)'a' } }, dtype: "S");
            Action act = () => np.isin(E(), nonAscii);
            act.Should().Throw<StringComputationNotSupportedException>();
        }

        [TestMethod]
        public void SetRoutines_DecidedByShape()
        {
            // np.union1d(E, ['b']) -> ['b'] (<U2); np.intersect1d(E, U) -> [] (<U2);
            // np.intersect1d(U, np.zeros(0), return_indices=True) -> ([] <U32, [], []); np.setxor1d(E, ['b']) -> ['b'];
            // np.setdiff1d(E, U) -> [] (<U2); np.setdiff1d(['b'], [1.5, 2.0]) -> ['b']; np.setdiff1d([0, 1, 2], U) -> [0, 1, 2]
            var union = np.union1d(E(), U1());
            union.dtype.str.Should().Be("<U2");
            union.item(0).Should().Be("b");
            var inter = np.intersect1d(E(), U());
            inter.dtype.str.Should().Be("<U2");
            inter.size.Should().Be(0);
            var withIdx = np.intersect1d(U(), np.zeros(new Shape(0L)), return_indices: true);
            withIdx[0].dtype.str.Should().Be("<U32");
            withIdx[1].dtype.Should().Be(np.int64);
            np.setxor1d(E(), U1()).item(0).Should().Be("b");
            np.setdiff1d(E(), U()).dtype.str.Should().Be("<U2");
            np.setdiff1d(U1(), np.array(new[] { 1.5, 2.0 })).item(0).Should().Be("b");
            np.setdiff1d(np.arange(3), U()).ToArray<long>().Should().Equal(0L, 1L, 2L);
        }

        [TestMethod]
        public void SetRoutines_ThatSortText_Defer()
        {
            Action union = () => np.union1d(U1(), U1());
            union.Should().Throw<StringComputationNotSupportedException>().WithMessage("union1d is not supported*");
            Action xor = () => np.setxor1d(U(), E());
            xor.Should().Throw<StringComputationNotSupportedException>().WithMessage("setxor1d is not supported*");
        }

        // ---------------------------------------------------------------- equality

        [TestMethod]
        public void Equality_DecidedByShapeAndDtype()
        {
            np.array_equal(E(), E()).Should().BeTrue();
            np.array_equal(E(), np.zeros(new Shape(0L), "S2")).Should().BeTrue();   // no equal loop: all-False of nothing
            np.array_equal(U(), np.zeros(new Shape(3L))).Should().BeFalse();       // no equal loop
            np.array_equal(U(), U1()).Should().BeFalse();                          // shape mismatch
            np.array_equiv(E(), U1()).Should().BeTrue();                           // an empty broadcast
            np.array_equiv(U(), np.zeros(new Shape(3L))).Should().BeFalse();
            var u = U();
            np.array_equal(u, u, true).Should().BeTrue();                          // a1 is a2
        }

        [TestMethod]
        public void Equality_ThatComparesText_Defers_AndEqualNanMeetsIsnan()
        {
            Action text = () => np.array_equal(U(), U());
            text.Should().Throw<StringComputationNotSupportedException>();
            Action nan = () => np.array_equal(U(), U(), true);
            nan.Should().Throw<TypeError>().WithMessage("ufunc 'isnan' not supported for the input types*");
        }

        // ---------------------------------------------------------------- lexsort, sort_complex, trim_zeros, diff

        [TestMethod]
        public void Lexsort_SortComplex_TrimZeros_Diff()
        {
            // np.lexsort([col]) sorts each line of one element: [[0], [0]], the key's shape.
            var lex = np.lexsort(new[] { Col() });
            lex.shape.Should().Equal(new long[] { 2, 1 });
            lex.ToArray<long>().Should().Equal(0L, 0L);
            var c = np.sort_complex(E());
            c.dtype.Should().Be(np.complex128);
            c.size.Should().Be(0);
            Action zeroD = () => np.sort_complex(np.array("ab"));
            zeroD.Should().Throw<AxisError>();
            // np.trim_zeros(np.zeros((0, 3), 'U2')).shape -> (0, 0); a 0-d input is returned as it is.
            np.trim_zeros(E2()).shape.Should().Equal(new long[] { 0, 0 });
            var zero = np.array("ab");
            ReferenceEquals(np.trim_zeros(zero), zero).Should().BeTrue();
            // np.diff of text refuses whatever the size (subtract has no string loop); n=0 returns the input itself.
            Action diff = () => np.diff(E());
            diff.Should().Throw<TypeError>()
                .WithMessage("ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U2'), dtype('<U2')) -> None");
            var u = U();
            ReferenceEquals(np.diff(u, 0), u).Should().BeTrue();
        }

        // ---------------------------------------------------------------- binary ufuncs over an empty broadcast

        [TestMethod]
        public void BinaryUfuncs_OverAnEmptyBroadcast_ReturnTheEmptyResult()
        {
            // np.equal(E, E) -> bool []; np.add(E2, np.zeros(3, 'U3')) -> <U5 (0, 3); np.logical_and(E, np.zeros(0)) -> bool []
            var eq = np.equal(E(), E());
            eq.dtype.Should().Be(np.@bool);
            eq.size.Should().Be(0);
            var sum = np.add(E2(), np.zeros(new Shape(3L), "U3"));
            sum.dtype.str.Should().Be("<U5");
            sum.shape.Should().Equal(new long[] { 0, 3 });
            np.logical_and(E(), np.zeros(new Shape(0L))).dtype.Should().Be(np.@bool);
            // Two classes have no loop at any size.
            Action mixed = () => np.equal(E(), np.zeros(new Shape(0L), "S2"));
            mixed.Should().Throw<TypeError>().WithMessage("ufunc 'equal' did not contain a loop with signature matching types*");
        }
    }
}
