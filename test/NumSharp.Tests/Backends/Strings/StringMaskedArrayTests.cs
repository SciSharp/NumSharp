using System;
using System.Linq;
using System.Text;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     numpy.ma over the fixed-width string dtypes: which calls NumPy answers from the dtype or the shape (its own
    ///     errors, empty results, the fill value), and which read the text (deferred with
    ///     <see cref="StringComputationNotSupportedException"/> until string computation lands). Every expected value and
    ///     message was probed against NumPy 2.4.2; the string oracle's <c>strma</c> section replays the full matrix over
    ///     every layout, these tests pin one representative of each rule.
    /// </summary>
    [TestClass]
    public class StringMaskedArrayTests
    {
        /// <summary>A masked <c>&lt;U2</c> array with its first element masked.</summary>
        private static NDMaskedArray MU() => np.ma.array(np.array(new[] { "ab", "c", "de" }), mask: new[] { true, false, false });

        /// <summary>A masked <c>|S2</c> array with its first element masked.</summary>
        private static NDMaskedArray MS() => np.ma.array(Bytes("ab", "c", "de"), mask: new[] { true, false, false });

        /// <summary>An empty <c>&lt;U1</c> masked array with an (empty) mask array.</summary>
        private static NDMaskedArray Empty() => np.ma.array(np.zeros(new Shape(0L), "U1"), mask: np.zeros(new Shape(0L), np.@bool));

        private static NDArray Floats() => np.array(new[] { 1.5, 2.0, 3.0 });

        private static NDArray Bytes(params string[] s) =>
            np.array(s.Select(x => (object)Encoding.ASCII.GetBytes(x)).ToArray(), dtype: "S");

        private static string Text(object fill) => fill is byte[] b ? Encoding.ASCII.GetString(b) : (string)fill;

        // ---------------------------------------------------------------- reductions: the add / multiply loops refuse

        [TestMethod]
        public void Reductions_OfText_RaiseTheLoopsTypeError()
        {
            // ma.sum / mean / var / std / anom fill and reduce with add: NumPy's add has no string reduce loop.
            const string reduce = "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U2'), dtype('<U2'), dtype('<U4'))";
            foreach (var call in new Action[] { () => np.ma.sum(MU()), () => np.ma.mean(MU()), () => np.ma.var(MU()), () => np.ma.std(MU()), () => np.ma.anom(MU()) })
                call.Should().Throw<TypeError>().WithMessage(reduce);
            Action cumsum = () => np.ma.cumsum(MU());
            cumsum.Should().Throw<TypeError>()
                  .WithMessage("the resolved dtypes are not compatible with add.accumulate. Resolved (dtype('<U2'), dtype('<U2'), dtype('<U4'))");
            Action prod = () => np.ma.prod(MU());
            prod.Should().Throw<TypeError>()
                .WithMessage("ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U2'), dtype('<U2')) -> None");
            Action cumprod = () => np.ma.cumprod(MS());
            cumprod.Should().Throw<TypeError>()
                   .WithMessage("ufunc 'multiply' did not contain a loop with signature matching types (dtype('S2'), dtype('S2')) -> None");
            // The dtype decides: an empty array and a 2-D bytes array along an axis refuse the same way.
            Action empty = () => np.ma.sum(np.ma.array(np.zeros(new Shape(0L), "U2")));
            empty.Should().Throw<TypeError>().WithMessage(reduce);
            Action axis = () => np.ma.sum(np.ma.array(Bytes("a", "b").reshape(1, 2), mask: new[,] { { true, false } }), 1);
            axis.Should().Throw<TypeError>()
                .WithMessage("the resolved dtypes are not compatible with add.reduce. Resolved (dtype('S1'), dtype('S1'), dtype('S2'))");
        }

        [TestMethod]
        public void Median_OfText_ChecksTheAxisThenRaisesTheAddLoopsTypeError()
        {
            Action median = () => np.ma.median(MU());
            median.Should().Throw<TypeError>()
                  .WithMessage("the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U2'), dtype('<U2'), dtype('<U4'))");
            Action badAxis = () => np.ma.median(MU(), 2);
            badAxis.Should().Throw<AxisError>().WithMessage("axis 2 is out of bounds for array of dimension 1*");
            // Fully masked bytes still reach np.mean of the filled data.
            Action allMasked = () => np.ma.median(np.ma.array(Bytes("ab", "c"), mask: true));
            allMasked.Should().Throw<TypeError>()
                     .WithMessage("the resolved dtypes are not compatible with add.reduce. Resolved (dtype('S2'), dtype('S2'), dtype('S4'))");
        }

        [TestMethod]
        public void Trace_OfText_SumsWithoutAMask_AndParsesWithOne()
        {
            var t2 = np.array(new[,] { { "1", "2" }, { "3", "4" } });
            // nomask: ndarray.trace adds the diagonal -> the add loop's TypeError.
            Action plain = () => np.ma.trace(np.ma.array(t2));
            plain.Should().Throw<TypeError>()
                 .WithMessage("the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U1'), dtype('<U1'), dtype('<U2'))");
            // A mask: D.astype(float64) PARSES the text ('1' -> 1.0, 'a' -> ValueError) — the next phase.
            Action masked = () => np.ma.trace(np.ma.array(t2, mask: new[,] { { false, false }, { false, true } }));
            masked.Should().Throw<StringComputationNotSupportedException>();
            // A 1-D operand has no diagonal.
            Action oneD = () => np.ma.trace(MU());
            oneD.Should().Throw<Exception>().WithMessage("diag requires an array of at least two dimensions*");
        }

        // ---------------------------------------------------------------- products, extrema, clip, allclose

        [TestMethod]
        public void Products_OfText_RaiseNumPysErrorsInNumPysOrder()
        {
            // ma.inner reshapes a 0-d operand to (1,) BEFORE np.inner checks the lengths.
            Action inner = () => np.ma.inner(MU(), "ab");
            inner.Should().Throw<Exception>().WithMessage("shapes (3,) and (1,) not aligned: 3 (dim 0) != 1 (dim 0)*");
            // Matching lengths reach the result allocation, which the string type number cannot size — empty or not.
            Action innerText = () => np.ma.inner(MU(), MU());
            innerText.Should().Throw<ValueError>().WithMessage("data type must provide an itemsize");
            Action innerEmpty = () => np.ma.inner(Empty(), Empty());
            innerEmpty.Should().Throw<ValueError>().WithMessage("data type must provide an itemsize");
            // ma.dot of an array and a str multiplies (no string loop); ma.outer calls np.multiply.outer directly.
            Action dot = () => np.ma.dot(MU(), "ab");
            dot.Should().Throw<TypeError>()
               .WithMessage("ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U2'), dtype('<U2')) -> None");
            const string outerText = "The 'out' kwarg is necessary when using the string multiply ufunc directly. " +
                                     "Use numpy.strings.multiply to multiply strings without specifying 'out'.";
            Action outer = () => np.ma.outer(MU(), 1);
            outer.Should().Throw<TypeError>().WithMessage(outerText);
            Action outerEmpty = () => np.ma.outer(Empty(), 1);
            outerEmpty.Should().Throw<TypeError>().WithMessage(outerText);
        }

        [TestMethod]
        public void Extrema_OfTextAndNumbers_NameTheirComparison()
        {
            // ma.minimum(a, b) = where(less(a, b), a, b): the comparison refuses the pair, with the operands in order.
            Action min = () => np.ma.minimum(MU(), Floats());
            min.Should().Throw<TypeError>().WithMessage(
                "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.StrDType'>, " +
                "<class 'numpy.dtypes.Float64DType'>) -> None");
            Action max = () => np.ma.maximum(Floats(), MU());
            max.Should().Throw<TypeError>().WithMessage(
                "ufunc 'greater' did not contain a loop with signature matching types (<class 'numpy.dtypes.Float64DType'>, " +
                "<class 'numpy.dtypes.StrDType'>) -> None");
            // ma.masked_values compares with `equal` first.
            Action values = () => np.ma.masked_values(MU(), Floats());
            values.Should().Throw<TypeError>().WithMessage(
                "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.StrDType'>, " +
                "<class 'numpy.dtypes.Float64DType'>) -> None");
        }

        [TestMethod]
        public void Clip_OfText_NamesPythonScalarBoundsByTheirDefaultDescriptor()
        {
            // A Python int bound is int64 in the no-loop text, a Python float float64 — whatever C# width carried it.
            var u = np.array(new[] { "ab" });
            Action ints = () => np.clip(u, 0, 1);
            ints.Should().Throw<TypeError>().WithMessage(
                "ufunc 'clip' did not contain a loop with signature matching types (dtype('<U2'), dtype('int64'), dtype('int64')) -> None");
            Action floats = () => np.clip(u, 0.5, 1.5);
            floats.Should().Throw<TypeError>().WithMessage(
                "ufunc 'clip' did not contain a loop with signature matching types (dtype('<U2'), dtype('float64'), dtype('float64')) -> None");
            Action mixed = () => np.clip(u, 0, 1.5);
            mixed.Should().Throw<TypeError>().WithMessage(
                "ufunc 'clip' did not contain a loop with signature matching types (dtype('<U2'), dtype('int64'), dtype('float64')) -> None");
            Action masked = () => np.ma.clip(MU(), 0, 1);
            masked.Should().Throw<TypeError>().WithMessage(
                "ufunc 'clip' did not contain a loop with signature matching types (dtype('<U2'), dtype('int64'), dtype('int64')) -> None");
        }

        [TestMethod]
        public void Allclose_OfText_FailsWhereNumPysStatementsFail()
        {
            // allclose(text, numbers): `isinf(masked_array(x))` has no string loop.
            Action textFirst = () => np.ma.allclose(MU(), Floats());
            textFirst.Should().Throw<TypeError>().WithMessage(
                "ufunc 'isinf' not supported for the input types, and the inputs could not be safely coerced to any supported " +
                "types according to the casting rule ''safe''");
            // allclose(numbers, text): `y = masked_array(b, dtype=result_type(b, 1.0))` cannot promote str with a float.
            Action textSecond = () => np.ma.allclose(Floats(), MU());
            textSecond.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.StrDType'> could not be promoted by <class 'numpy.dtypes._PyFloatDType'>. " +
                "This means that no common DType exists for the given inputs. For example they cannot be stored in a single " +
                "array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.StrDType'>, " +
                "<class 'numpy.dtypes._PyFloatDType'>)");
        }

        // ---------------------------------------------------------------- shape-decided answers and deferrals

        [TestMethod]
        public void Nonzero_OfText_RefusesA0dOperand_AndAnswersAnEmptyOne()
        {
            Action zeroD = () => np.ma.nonzero(np.ma.array(np.array("ab"), mask: true));
            zeroD.Should().Throw<ValueError>()
                 .WithMessage("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.");
            var empty = np.ma.nonzero(Empty());
            empty.Length.Should().Be(1);
            empty[0].size.Should().Be(0);
            empty[0].dtype.Should().Be(np.int64);
            // Non-empty text: the truth values are the strings' lengths — read the text.
            Action text = () => np.ma.nonzero(np.ma.array(np.array(new[] { "a", "" }), mask: new[] { false, false }));
            text.Should().Throw<StringComputationNotSupportedException>();
        }

        [TestMethod]
        public void LogicalReductionsAndCount_OfText()
        {
            // ma.alltrue(empty text) -> True (nothing read); ma.sometrue of an empty (0, 2) slice -> [False, False],
            // masked [True, True] (logical_and.reduce of an empty mask column is True).
            np.ma.getdata(np.ma.alltrue(np.ma.array(np.zeros(new Shape(0L), "U1")))).GetBoolean().Should().BeTrue();
            var some = np.ma.sometrue(np.ma.array(np.zeros(new Shape(0L, 2L), "U1"), mask: np.zeros(new Shape(0L, 2L), np.@bool)));
            np.ma.getdata(some).ToArray<bool>().Should().Equal(false, false);
            np.ma.getmaskarray(some).ToArray<bool>().Should().Equal(true, true);
            // Non-empty text: the truth of '' is read.
            Action alltrue = () => np.ma.alltrue(np.ma.array(np.array(new[] { "a", "" }), mask: new[] { true, false }));
            alltrue.Should().Throw<StringComputationNotSupportedException>();
            // ma.count reads only the mask: [[a, b]] with [0, 0] masked, axis 0 -> [0, 1].
            np.ma.count(np.ma.array(np.array(new[,] { { "a", "b" } }), mask: new[,] { { true, false } }), 0)
              .ToArray<long>().Should().Equal(0L, 1L);
        }

        [TestMethod]
        public void Isin_OfAnEmptyTextArray_IsAnEmptyUnmaskedResult()
        {
            // ma.isin(empty <U1, 1.5): unique + concatenate + sort over nothing -> an empty bool array, nomask.
            var r = np.ma.isin(Empty(), 1.5);
            r.dtype.Should().Be(np.@bool);
            r.shape.Should().Equal(new long[] { 0 });
            ReferenceEquals(np.ma.getmask(r), np.ma.nomask).Should().BeTrue();
            Action text = () => np.ma.isin(MU(), "ab");
            text.Should().Throw<StringComputationNotSupportedException>();
        }

        [TestMethod]
        public void Compress_WithATextCondition_Defers()
        {
            // The condition's truth values are the strings' lengths: ma.compress(['a', ''], [1, 2]) -> [1] reads them.
            Action act = () => np.ma.compress(np.array(new[] { "a", "" }), np.array(new long[] { 1, 2 }));
            act.Should().Throw<StringComputationNotSupportedException>();
        }

        [TestMethod]
        public void Vander_OfText_IsDecidedByTheColumns()
        {
            // Text promotes with int64 to <U21. No column (or no row with one column): the empty allocation.
            var noColumn = np.vander(np.array(new[] { "a", "b" }), 0);
            noColumn.dtype.str.Should().Be("<U21");
            noColumn.shape.Should().Equal(new long[] { 2, 0 });
            np.vander(np.zeros(new Shape(0L), "U1"), 1).shape.Should().Equal(new long[] { 0, 1 });
            np.vander(Bytes("a"), 0).dtype.str.Should().Be("|S21");
            // One column writes the formatted '1' — the next phase; two columns multiply — no string loop, rows or not.
            Action one = () => np.vander(np.array(new[] { "a" }), 1);
            one.Should().Throw<StringComputationNotSupportedException>();
            const string multiply = "ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U21'), dtype('<U21')) -> None";
            Action two = () => np.vander(np.array(new[] { "a" }), 2);
            two.Should().Throw<TypeError>().WithMessage(multiply);
            Action twoEmpty = () => np.vander(np.zeros(new Shape(0L), "U1"), 2);
            twoEmpty.Should().Throw<TypeError>().WithMessage(multiply);
            // ma.vander of an empty masked text array is the plain empty matrix (no mask is written into nothing).
            var ma = np.ma.vander(Empty());
            ma.dtype.str.Should().Be("<U21");
            ma.shape.Should().Equal(new long[] { 0, 0 });
        }

        // ---------------------------------------------------------------- the text fill value

        [TestMethod]
        public void TextFillValue_IsMaterializedAtItsOwnWidth()
        {
            // The default is 'N/A' stored at ITS width (<U3 / |S3), whatever the array's width.
            Text(MU().fill_value).Should().Be("N/A");
            Text(MS().fill_value).Should().Be("N/A");
            // Once materialized, a new value is written INTO the stored <U3: 'hello' -> 'hel'.
            var u5 = np.ma.array(np.array(new[] { "abcde" }), mask: new[] { true });
            _ = u5.fill_value;
            u5.fill_value = "hello";
            Text(u5.fill_value).Should().Be("hel");
            // A fill stored at the array's width truncates a reset to the default: 'zz' then None -> 'N/'.
            var u2 = np.ma.array(np.array(new[] { "ab" }), mask: new[] { true }, fill_value: "zz");
            u2.fill_value = null;
            Text(u2.fill_value).Should().Be("N/");
            // Materialized first, the reset keeps 'N/A'.
            var u2b = np.ma.array(np.array(new[] { "ab" }), mask: new[] { true });
            _ = u2b.fill_value;
            u2b.fill_value = null;
            Text(u2b.fill_value).Should().Be("N/A");
            // A new value on a fresh array is checked into the array's dtype: |S2 b'wxyz' -> b'wx', <U1 'xyz' -> 'x'.
            var s2 = np.ma.array(Bytes("ab"), mask: new[] { true });
            s2.fill_value = Encoding.ASCII.GetBytes("wxyz");
            Text(s2.fill_value).Should().Be("wx");
            var u1 = np.ma.array(np.array(new[] { "a" }), mask: new[] { true });
            u1.fill_value = "xyz";
            Text(u1.fill_value).Should().Be("x");
        }

        [TestMethod]
        public void TextFillValue_IsConvertedOnlyWhenItWasMaterialized()
        {
            // __array_finalize__ converts an inherited (materialized) fill to the new dtype: <U1 -> 'N'.
            var read = np.ma.array(np.array(new[] { "a", "b" }), mask: new[] { true, false });
            _ = read.fill_value;
            Text(np.ma.masked_where(new[] { true, false }, read).fill_value).Should().Be("N");
            // Never read, there is nothing to convert: the new array materializes its own 'N/A'.
            var unread = np.ma.array(np.array(new[] { "a", "b" }), mask: new[] { true, false });
            Text(np.ma.masked_where(new[] { true, false }, unread).fill_value).Should().Be("N/A");
        }

        [TestMethod]
        public void Filled_OfText_WritesTextFillsAndDefersFormattedNumbers()
        {
            var x = np.ma.array(np.array(new[] { "ab", "c" }), mask: new[] { true, false });
            NDArray.AsStringArray(x.filled()).Should().Equal("N/", "c");
            NDArray.AsStringArray(x.filled("zz")).Should().Equal("zz", "c");
            // A number written into text is formatted ('5') — the next phase.
            Action number = () => x.filled(5);
            number.Should().Throw<StringComputationNotSupportedException>();
            // Nothing masked: `filled` returns the data itself before any fill is converted.
            var clean = np.ma.array(np.array(new[] { "ab" }), mask: new[] { false });
            var view = clean.filled(5);
            NDArray.AsStringArray(view).Should().Equal("ab");
            view.flags.owndata.Should().BeFalse();
        }

        [TestMethod]
        public void Unique_OfAFullyMaskedTextArray_KeepsTheFirstMaskedElement()
        {
            // ma.unique(ma.array(['c', 'a', 'b'], mask=True)) -> data ['c'], mask [True], fill 'N' (the sort materialized
            // 'N/A', the result's dtype <U1 converts it).
            var r = np.ma.unique(np.ma.array(np.array(new[] { "c", "a", "b" }), mask: true));
            NDArray.AsStringArray(np.ma.getdata(r)).Should().Equal("c");
            np.ma.getmaskarray(r).ToArray<bool>().Should().Equal(true);
            Text(r.fill_value).Should().Be("N");
            // Masked and unmasked text: where the masked slot sorts is a string comparison with the fill.
            Action mixed = () => np.ma.unique(MU());
            mixed.Should().Throw<StringComputationNotSupportedException>();
        }
    }
}
