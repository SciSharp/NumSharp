using System;
using System.Linq;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    ///     np.evaluate over text, phases E1 and E2 (docs/plans/evaluate-text.md): S, U and StringDType operands — arrays of
    ///     any layout, 0-d arrays and a C# <c>string</c> as NumPy's <c>str</c> — in the six comparisons, the truth readers
    ///     (the logical nodes, <c>Where</c>'s condition, <c>Any</c> / <c>All</c> / <c>CountNonzero</c>), <c>IsNaN</c>, a
    ///     <c>Cast</c> to a number and the numpy.strings factories (<c>StrLen</c>, the predicates, the searches), anywhere in
    ///     a tree; their numeric results feed every numeric node, reduction and keyword of the fused remainder. Each runs as
    ///     NumPy's own statement (the engine's string functions), so its values, its StringDType NA rules and its errors are
    ///     NumPy's; every other node with a text operand is refused, naming the node (docs/plans/evaluate-text.md lists the
    ///     phase that lifts each refusal). Every expected value and message was probed against numpy==2.4.2.
    /// </summary>
    [TestClass]
    public class NDEvaluateTextTests
    {
        /// <summary>The probe's texts: empty, a trailing and an embedded NUL, non-ASCII, astral, widths 0 to 3.</summary>
        private static readonly string[] Texts = { "", "a", "a\0", "ab", "b", "é", "\U0001F600", "a\0b", "zz" };

        /// <summary>The probe's <c>U</c> operand: <c>np.array(Texts)</c>, a <c>&lt;U3</c> (U keeps no trailing NUL).</summary>
        private static NDArray U() => np.array(Texts);

        /// <summary>The probe's <c>U</c> operand reversed, as a fresh C-contiguous array.</summary>
        private static NDArray UR() => U()["::-1"].copy();

        /// <summary>The probe's <c>StringDType</c> operand, built from the texts themselves (a NUL is kept as a character).</summary>
        private static NDArray T() => np.array(Texts.Cast<object>().ToArray(), np.dtypes.StringDType());

        /// <summary>The probe's <c>StringDType</c> operand reversed, as a fresh array.</summary>
        private static NDArray TR() => T()["::-1"].copy();

        /// <summary>The probe's <c>|S3</c> operand: each text's UTF-8 bytes cut to three (<c>v.encode('utf-8')[:3]</c>).</summary>
        private static NDArray S() => np.array(Texts.Select(t => (object)System.Text.Encoding.UTF8.GetBytes(t).Take(3).ToArray()).ToArray(),
            np.dtype("S3"));

        /// <summary>The probe's <c>S</c> operand reversed, as a fresh array.</summary>
        private static NDArray SR() => S()["::-1"].copy();

        /// <summary>The probe's numeric column: <c>np.arange(9) * 7 % 10</c> = 0, 7, 4, 1, 8, 5, 2, 9, 6 (int64).</summary>
        private static NDArray Age() => np.array(new long[] { 0, 7, 4, 1, 8, 5, 2, 9, 6 });

        /// <summary>The flags of a bool array in C order.</summary>
        /// <param name="a">A bool array.</param>
        /// <returns>The flags.</returns>
        private static bool[] B(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => (bool)a.Storage.GetAtIndex(i)).ToArray();

        /// <summary>Parses a probe row (<c>"FFTTF"</c>) into flags.</summary>
        /// <param name="row">One character per element, <c>T</c> or <c>F</c>.</param>
        /// <returns>The flags.</returns>
        private static bool[] Row(string row) => row.Select(c => c == 'T').ToArray();

        /// <summary>The comparison factory of an op name.</summary>
        /// <param name="op">One of the six comparison names.</param>
        /// <returns>The factory.</returns>
        private static Func<NDExpr, NDExpr, NDExpr> Cmp(string op) => op switch
        {
            "equal" => NDExpr.Equal,
            "not_equal" => NDExpr.NotEqual,
            "less" => NDExpr.Less,
            "less_equal" => NDExpr.LessEqual,
            "greater" => NDExpr.Greater,
            _ => NDExpr.GreaterEqual,
        };

        /// <summary>
        ///     The six comparisons over every pair NumPy has a loop for — S with S, U with U, StringDType with StringDType and
        ///     StringDType with U either way — each element pair a different case (a trailing NUL, an embedded NUL, a code
        ///     point above U+FFFF, the empty text, different widths), against the probe's rows.
        /// </summary>
        /// <param name="op">The comparison.</param>
        /// <param name="expected">NumPy's row (identical for the five pairs: the texts compare alike in every class here).</param>
        [TestMethod]
        [DataRow("equal", "FFFFTFFFF")]
        [DataRow("not_equal", "TTTTFTTTT")]
        [DataRow("less", "TTTTFFFFF")]
        [DataRow("less_equal", "TTTTTFFFF")]
        [DataRow("greater", "FFFFFTTTT")]
        [DataRow("greater_equal", "FFFFTTTTT")]
        public void SixComparisons_EveryClassPair_AreNumPys(string op, string expected)
        {
            var f = Cmp(op);
            B(np.evaluate(f(U(), UR()))).Should().Equal(Row(expected), "U with U");
            B(np.evaluate(f(S(), SR()))).Should().Equal(Row(expected), "S with S");
            B(np.evaluate(f(T(), TR()))).Should().Equal(Row(expected), "StringDType with StringDType");
            B(np.evaluate(f(T(), UR()))).Should().Equal(Row(expected), "StringDType with U");
            B(np.evaluate(f(U(), TR()))).Should().Equal(Row(expected), "U with StringDType");
        }

        /// <summary>
        ///     A C# string is NumPy's <c>str</c> on either side of a comparison (a 0-d <c>&lt;U1</c> here): <c>'a'</c> equals
        ///     <c>'a\0'</c> in a U array (the trailing NUL is padding), and the operators with the string on the left keep its
        ///     place (<c>"a" &lt; u</c> is <c>np.less('a', u)</c>).
        /// </summary>
        /// <param name="op">The comparison.</param>
        /// <param name="right">NumPy's <c>np.{op}(u, 'a')</c>.</param>
        /// <param name="left">NumPy's <c>np.{op}('a', u)</c>.</param>
        [TestMethod]
        [DataRow("equal", "FTTFFFFFF", "FTTFFFFFF")]
        [DataRow("not_equal", "TFFTTTTTT", "TFFTTTTTT")]
        [DataRow("less", "TFFFFFFFF", "FFFTTTTTT")]
        [DataRow("less_equal", "TTTFFFFFF", "FTTTTTTTT")]
        [DataRow("greater", "FFFTTTTTT", "TFFFFFFFF")]
        [DataRow("greater_equal", "FTTTTTTTT", "TTTFFFFFF")]
        public void PythonStrLiteral_EitherSide_IsNumPysStr(string op, string right, string left)
        {
            var f = Cmp(op);
            B(np.evaluate(f(U(), "a"))).Should().Equal(Row(right), "u OP 'a'");
            B(np.evaluate(f("a", U()))).Should().Equal(Row(left), "'a' OP u");
            if (op is "less" or "less_equal" or "greater" or "greater_equal")
            {
                Func<NDExpr, NDExpr, NDExpr> oper = op switch
                {
                    "less" => (x, y) => x < y,
                    "less_equal" => (x, y) => x <= y,
                    "greater" => (x, y) => x > y,
                    _ => (x, y) => x >= y,
                };
                B(np.evaluate(oper((NDExpr)U(), "a"))).Should().Equal(Row(right), "the operator, string on the right");
                B(np.evaluate(oper("a", (NDExpr)U()))).Should().Equal(Row(left), "the operator, string on the left");
            }
        }

        /// <summary>
        ///     The comparison's bool result is one more input of the fused numeric remainder: logical and bitwise
        ///     combinations with numeric comparisons, a Where that selects numbers by text, and the arithmetic on top.
        /// </summary>
        [TestMethod]
        public void ComparisonResults_FeedTheFusedRemainder()
        {
            var u = U();
            var age = Age();
            B(np.evaluate(NDExpr.Equal(u, "a") | ((NDExpr)age > 5))).Should().Equal(Row("FTTFTFFTT"));
            var where = np.evaluate(NDExpr.Where(NDExpr.Less(u, "b"), (NDExpr)age * 2, age));
            where.typecode.Should().Be(NPTypeCode.Int64);
            where.ToArray<long>().Should().Equal(0, 14, 8, 2, 8, 5, 2, 18, 6);
            var mixed = np.evaluate(NDExpr.LogicalAnd(NDExpr.GreaterEqual(u, "a"), NDExpr.NotEqual(u, "zz")) & ((NDExpr)age < 9));
            B(mixed).Should().Equal(Row("FTTTTTTFF"));
        }

        /// <summary>
        ///     Every reduction over a comparison of text, flat and per axis, with NumPy's result dtypes: counts and sums are
        ///     int64, the mean float64, Any / All bool (keepdims keeps the axis).
        /// </summary>
        [TestMethod]
        public void Reductions_OverTextComparisons_AreNumPys()
        {
            var u = U();
            var count = np.evaluate(NDExpr.CountNonzero(NDExpr.GreaterEqual(u, "a")));
            count.typecode.Should().Be(NPTypeCode.Int64);
            ((long)count).Should().Be(8);
            var sum = np.evaluate(NDExpr.Sum(NDExpr.NotEqual(u, TR())));
            sum.typecode.Should().Be(NPTypeCode.Int64);
            ((long)sum).Should().Be(8);
            ((double)np.evaluate(NDExpr.Mean(NDExpr.Greater(u, "a")))).Should().Be(0.6666666666666666);

            var u2 = u.reshape(3, 3);
            B(np.evaluate(NDExpr.Any(NDExpr.Equal(u2, "zz"), 1))).Should().Equal(false, false, true);
            var all = np.evaluate(NDExpr.All(NDExpr.NotEqual(u2, ""), 0, keepdims: true));
            all.shape.Should().Equal(1L, 3L);
            B(all).Should().Equal(false, true, true);
            var perColumn = np.evaluate(NDExpr.Sum(NDExpr.Less(u2, "b"), 0));
            perColumn.typecode.Should().Be(NPTypeCode.Int64);
            perColumn.ToArray<long>().Should().Equal(2, 2, 1);
        }

        /// <summary>
        ///     The operand layouts the comparison guard walks (an F-ordered operand keeps an F result, as NumPy's ufunc
        ///     allocates it), transposed, reversed and stepped views, a column against a row, and an empty operand.
        /// </summary>
        [TestMethod]
        public void Layouts_AreNumPys()
        {
            var u2 = U().reshape(3, 3);
            var f = np.asfortranarray(u2);
            var fr = np.evaluate(NDExpr.Equal(f, "a"));
            fr.Shape.IsFContiguous.Should().BeTrue("NumPy's np.equal of an F operand is F-ordered");
            B(fr).Should().Equal(Row("FTTFFFFFF"));
            B(np.evaluate(NDExpr.Less(u2.T, "b"))).Should().Equal(Row("TTFTFTTFF"));
            B(np.evaluate(NDExpr.Greater(U()["::-2"], "a"))).Should().Equal(Row("TTTFF"));
            B(np.evaluate(NDExpr.Equal(U()["1::3"], U()["2::3"]))).Should().Equal(Row("TFF"));
            var outer = np.evaluate(NDExpr.Equal(U()[":3"].reshape(3, 1), U()["3:6"].reshape(1, 3)));
            outer.shape.Should().Equal(3L, 3L);
            B(outer).Should().OnlyContain(x => !x);
            var empty = np.evaluate(NDExpr.Equal(np.zeros(new Shape(0, 3), np.dtype("U2")), "a"));
            empty.typecode.Should().Be(NPTypeCode.Boolean);
            empty.shape.Should().Equal(0L, 3L);
            var zeroD = np.evaluate(NDExpr.Equal((NDExpr)"a", "a"));
            zeroD.ndim.Should().Be(0);
            ((bool)zeroD).Should().BeTrue();
        }

        /// <summary>
        ///     StringDType's NA rules follow NumPy's loop, which reads them from its FIRST descriptor: a NaN-like NA makes
        ///     every comparison with it False (<c>!=</c> too) on the left, while <c>'a' OP t</c> promotes the str to a
        ///     default StringDType that reads the NA as <c>''</c>; a None-like NA equals <c>''</c> and refuses an ordering.
        /// </summary>
        [TestMethod]
        public void StringDTypeNA_IsNumPysFirstDescriptorRule()
        {
            using var tn = np.array(new object[] { "a", double.NaN, "b" }, np.dtypes.StringDType(na_object: double.NaN));
            B(np.evaluate(NDExpr.Equal(tn, "a"))).Should().Equal(true, false, false);
            B(np.evaluate(NDExpr.NotEqual(tn, "a"))).Should().Equal(false, false, true);
            B(np.evaluate(NDExpr.NotEqual("a", tn))).Should().Equal(false, true, true);
            B(np.evaluate(NDExpr.Less(tn, "a"))).Should().Equal(false, false, false);
            B(np.evaluate(NDExpr.Greater("a", tn))).Should().Equal(false, true, false);
            B(np.evaluate(NDExpr.GreaterEqual(tn, tn))).Should().Equal(true, false, true);
            ((long)np.evaluate(NDExpr.CountNonzero(NDExpr.NotEqual(tn, "b")))).Should().Be(1);

            using var tnone = np.array(new object[] { "a", null, "b" }, np.dtypes.StringDType(na_object: null));
            B(np.evaluate(NDExpr.Equal(tnone, ""))).Should().Equal(false, true, false);
            B(np.evaluate(NDExpr.NotEqual(tnone, tnone))).Should().Equal(false, false, false);
            new Action(() => np.evaluate(NDExpr.Less(tnone, "b")))
                .Should().Throw<ValueError>().WithMessage("'less' not supported for null values that are not nan-like or strings.");
            new Action(() => np.evaluate(NDExpr.GreaterEqual(tnone, "b") | true))
                .Should().Throw<ValueError>().WithMessage("'greater_equal' not supported for null values that are not nan-like or strings.");
        }

        /// <summary>
        ///     The keywords apply to the fused remainder exactly as for a numeric tree: <c>out=</c> receives the bools through
        ///     its cast and is returned itself, <c>where=</c> writes only the selected elements, <c>dtype=</c> casts the result,
        ///     <c>order=</c> lays a fresh result out.
        /// </summary>
        [TestMethod]
        public void Keywords_ApplyToTheRemainder()
        {
            var u = U();
            var o = np.zeros(new Shape(9), NPTypeCode.Int32);
            np.evaluate(NDExpr.Equal(u, "a"), @out: o).Should().BeSameAs(o);
            o.ToArray<int>().Should().Equal(0, 1, 1, 0, 0, 0, 0, 0, 0);

            var kept = np.ones(new Shape(9), NPTypeCode.Boolean);
            var mask = np.array(new[] { true, true, false, false, true, true, false, false, true });
            np.evaluate(NDExpr.Equal(u, "zz"), @out: kept, where: mask);
            B(kept).Should().Equal(false, false, true, true, false, false, true, true, true);

            var f64 = np.evaluate(NDExpr.Equal(u, "a"), dtype: np.float64);
            f64.typecode.Should().Be(NPTypeCode.Double);
            f64.ToArray<double>().Should().Equal(0, 1, 1, 0, 0, 0, 0, 0, 0);

            var fresh = np.evaluate(NDExpr.Less(u.reshape(3, 3), "b"), order: 'F');
            fresh.Shape.IsFContiguous.Should().BeTrue();
            B(fresh.copy()).Should().Equal(Row("TTTTFFFTF"), "'a\\0b' < 'b' at its first unit");
        }

        /// <summary>
        ///     Every comparison NumPy refuses raises NumPy's own error when the call runs: bytes against str either way, text
        ///     against a Python int / float / bool, against a numeric array or a computed number, and operands that do not
        ///     broadcast; two StringDType NA objects with no common value raise NumPy's TypeError.
        /// </summary>
        [TestMethod]
        public void Errors_AreNumPys()
        {
            var u = U();
            const string noLoop = "ufunc 'equal' did not contain a loop with signature matching types ";
            new Action(() => np.evaluate(NDExpr.Equal(S(), "a")))
                .Should().Throw<TypeError>().WithMessage(noLoop + "(<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StrDType'>) -> None");
            var bytes0d = np.array(new object[] { new byte[] { 0x61 } }, np.dtype("S1")).reshape(Shape.NewScalar());
            new Action(() => np.evaluate(NDExpr.Equal(T(), bytes0d)))
                .Should().Throw<TypeError>().WithMessage(noLoop + "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None");
            new Action(() => np.evaluate(NDExpr.Equal(u, 1)))
                .Should().Throw<TypeError>().WithMessage(noLoop + "(<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes._PyLongDType'>) -> None");
            new Action(() => np.evaluate(NDExpr.Equal(u, 1.5)))
                .Should().Throw<TypeError>().WithMessage(noLoop + "(<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes._PyFloatDType'>) -> None");
            new Action(() => np.evaluate(NDExpr.Equal(u, true)))
                .Should().Throw<TypeError>().WithMessage(noLoop + "(<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes.BoolDType'>) -> None");
            new Action(() => np.evaluate(NDExpr.Equal(u, Age().astype(NPTypeCode.Int32))))
                .Should().Throw<TypeError>().WithMessage(noLoop + "(<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes.Int32DType'>) -> None");
            new Action(() => np.evaluate(NDExpr.Equal(u, (NDExpr)Age() * 2)))
                .Should().Throw<TypeError>().WithMessage(noLoop + "(<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes.Int64DType'>) -> None");
            new Action(() => np.evaluate(NDExpr.Equal(u, u[":2"])))
                .Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (9,) (2,) ");
            using var tn = np.array(new object[] { "a" }, np.dtypes.StringDType(na_object: double.NaN));
            using var ts = np.array(new object[] { "a" }, np.dtypes.StringDType(na_object: "zz"));
            new Action(() => np.evaluate(NDExpr.Equal(tn, ts)))
                .Should().Throw<TypeError>().WithMessage("Cannot find a compatible null string value for null strings 'nan' and ''zz''");
        }

        /// <summary>
        ///     The factories' truth test is not a comparison: <c>CountNonzero</c> / <c>Any</c> / <c>All</c> of text read the
        ///     text's truth as NumPy does — <c>np.any</c> / <c>np.all</c> and an axis or keepdims count through the bool cast,
        ///     a plain <c>np.count_nonzero</c> through the dtype's nonzero function (they differ for a StringDType whose NA
        ///     object is the empty text) — over every axis form, a 0-d operand with axis 0 / -1 included; a user's own
        ///     <c>NotEqual(text, 0)</c> stays NumPy's no-loop comparison.
        /// </summary>
        [TestMethod]
        public void TruthTests_ReadTheTruthValue()
        {
            var u = U();
            ((long)np.evaluate(NDExpr.CountNonzero(u))).Should().Be(8);
            ((long)np.evaluate(NDExpr.CountNonzero(S()))).Should().Be(8);
            ((long)np.evaluate(NDExpr.CountNonzero(T()))).Should().Be(8);
            ((bool)np.evaluate(NDExpr.Any(u, 0))).Should().BeTrue();
            var allKeep = np.evaluate(NDExpr.All(u, keepdims: true));
            allKeep.shape.Should().Equal(1L);
            allKeep.ToArray<bool>().Should().Equal(false);

            // StringDType with an empty-text NA object: np.count_nonzero(te) reads the nonzero function (the NA does not
            // count), np.count_nonzero(te, axis=None, keepdims=True) and the axis form read the cast (it does), as do any / all.
            using var te = np.array(new object[] { "a", "", "b" }, np.dtypes.StringDType(na_object: ""));
            ((long)np.evaluate(NDExpr.CountNonzero(te))).Should().Be(2);
            var countKeep = np.evaluate(NDExpr.CountNonzero(te, keepdims: true));
            countKeep.shape.Should().Equal(1L);
            countKeep.ToArray<long>().Should().Equal(3L);
            ((long)np.evaluate(NDExpr.CountNonzero(te, 0))).Should().Be(3);
            ((bool)np.evaluate(NDExpr.Any(te))).Should().BeTrue();
            ((bool)np.evaluate(NDExpr.All(te))).Should().BeTrue();

            var u2 = U().reshape(3, 3);
            np.evaluate(NDExpr.CountNonzero(u2, 1)).ToArray<long>().Should().Equal(2L, 3L, 3L);
            var anyKeep = np.evaluate(NDExpr.Any(u2, 0, true));
            anyKeep.shape.Should().Equal(1L, 3L);
            anyKeep.ToArray<bool>().Should().Equal(true, true, true);
            np.evaluate(NDExpr.All(S().reshape(3, 3), 1)).ToArray<bool>().Should().Equal(Row("FTT"));
            // NumPy lets axis 0 / -1 through for a 0-d operand
            ((long)np.evaluate(NDExpr.CountNonzero(np.array(new[] { "q" }).reshape(Shape.NewScalar()), 0))).Should().Be(1);
            ((bool)np.evaluate(NDExpr.Any(np.array(new[] { "" }).reshape(Shape.NewScalar()), -1))).Should().BeFalse();

            new Action(() => np.evaluate(NDExpr.Sum(NDExpr.NotEqual(u, 0))))
                .Should().Throw<TypeError>()
                .WithMessage("ufunc 'not_equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes._PyLongDType'>) -> None");
            // numbers are unaffected: the flagged test is the same kernel
            ((long)np.evaluate(NDExpr.CountNonzero(Age()))).Should().Be(8);
        }

        /// <summary>
        ///     <c>LogicalAnd</c> / <c>LogicalOr</c> / <c>LogicalXor</c> read every text operand through its bool cast, for any
        ///     mix of S, U, StringDType, bools and numeric sub-trees, the StringDType NA by its descriptor (a <c>None</c>-like NA
        ///     is False, a NaN-like one True), and the result keeps the layout NumPy's own logical ufunc gives it: F for a
        ///     transposed operand, C for a broadcast one (NumPy casts inside its iterator, so the broadcast operand abstains).
        /// </summary>
        [TestMethod]
        public void LogicalNodes_ReadTheTruthValue()
        {
            var u = U();
            var age = Age();
            np.evaluate(NDExpr.LogicalAnd(u, (NDExpr)age > 3)).ToArray<bool>().Should().Equal(Row("FTTFTTFTT"));
            np.evaluate(NDExpr.LogicalOr(S(), false)).ToArray<bool>().Should().Equal(Row("FTTTTTTTT"));
            np.evaluate(NDExpr.LogicalXor(T(), UR())).ToArray<bool>().Should().Equal(Row("TFFFFFFFT"));
            ((long)np.evaluate(NDExpr.CountNonzero(NDExpr.LogicalAnd(u, (NDExpr)age > 3) | NDExpr.Equal(age, 0)))).Should().Be(7);

            var transposed = np.evaluate(NDExpr.LogicalAnd(u.reshape(3, 3).T, true));
            transposed.ToArray<bool>().Should().Equal(Row("FTTTTTTTT"));
            transposed.Shape.IsFContiguous.Should().BeTrue();
            transposed.Shape.IsContiguous.Should().BeFalse();

            var bc = np.broadcast_to(np.array(new[] { "a", "", "zz" }).reshape(1, 3), new Shape(3, 3));
            var mixed = np.evaluate(NDExpr.LogicalAnd(bc, (NDExpr)age.reshape(3, 3) > 3));
            mixed.ToArray<bool>().Should().Equal(Row("FFTFFTFFT"));
            mixed.Shape.IsContiguous.Should().BeTrue();
            var alone = np.evaluate(NDExpr.LogicalAnd(bc, true));
            alone.ToArray<bool>().Should().Equal(Row("TFTTFTTFT"));
            alone.Shape.IsContiguous.Should().BeTrue("the broadcast operand abstains, as in NumPy (its dense cast would be F)");

            using var tn = np.array(new object[] { "a", "", null, "b" }, np.dtypes.StringDType(na_object: null));
            using var tf = np.array(new object[] { "a", "", double.NaN, "b" }, np.dtypes.StringDType(na_object: double.NaN));
            np.evaluate(NDExpr.LogicalAnd(tn, true)).ToArray<bool>().Should().Equal(Row("TFFT"));
            np.evaluate(NDExpr.LogicalAnd(tf, true)).ToArray<bool>().Should().Equal(Row("TFTT"));
        }

        /// <summary>
        ///     A text <c>Where</c> condition is read through its bool cast, as <c>np.where</c> reads it, the branches typed and
        ///     fused as over numbers; a broadcast text condition leaves NumPy's C layout.
        /// </summary>
        [TestMethod]
        public void WhereCondition_ReadsTheTruthValue()
        {
            var age = Age();
            np.evaluate(NDExpr.Where(U(), age, -(NDExpr)age)).ToArray<long>().Should().Equal(0L, 7, 4, 1, 8, 5, 2, 9, 6);
            var f = np.evaluate(NDExpr.Where(S(), 1.5, 0));
            f.typecode.Should().Be(NPTypeCode.Double);
            f.ToArray<double>().Should().Equal(0.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5);
            using var tn = np.array(new object[] { "a", "", null, "b" }, np.dtypes.StringDType(na_object: null));
            using var tf = np.array(new object[] { "a", "", double.NaN, "b" }, np.dtypes.StringDType(na_object: double.NaN));
            np.evaluate(NDExpr.Where(tn, 1, 0)).ToArray<long>().Should().Equal(1L, 0, 0, 1);
            np.evaluate(NDExpr.Where(tf, 1, 0)).ToArray<long>().Should().Equal(1L, 0, 1, 1);

            var bc = np.broadcast_to(np.array(new[] { "a", "", "zz" }).reshape(1, 3), new Shape(3, 3));
            var picked = np.evaluate(NDExpr.Where(bc, age.reshape(3, 3), -1));
            picked.ToArray<long>().Should().Equal(0L, -1, 4, 1, -1, 5, 2, -1, 6);
            picked.Shape.IsContiguous.Should().BeTrue();
        }

        /// <summary>
        ///     <c>IsNaN</c> of text is NumPy's <c>np.isnan</c>: True for a StringDType NA whose NA object is NaN-like, False for
        ///     every other element (a <c>None</c>-like NA included), NumPy's refusal for S / U.
        /// </summary>
        [TestMethod]
        public void IsNaN_ReadsStringDTypesNaNLikeNA()
        {
            using var tn = np.array(new object[] { "a", "", null, "b" }, np.dtypes.StringDType(na_object: null));
            using var tf = np.array(new object[] { "a", "", double.NaN, "b" }, np.dtypes.StringDType(na_object: double.NaN));
            np.evaluate(NDExpr.IsNaN(tf)).ToArray<bool>().Should().Equal(Row("FFTF"));
            np.evaluate(NDExpr.IsNaN(tn)).ToArray<bool>().Should().Equal(Row("FFFF"));
            np.evaluate(NDExpr.IsNaN(T()) | ((NDExpr)Age() > 5)).ToArray<bool>().Should().Equal(Row("FTFFTFFTT"));
            new Action(() => np.evaluate(NDExpr.IsNaN(U())))
                .Should().Throw<TypeError>()
                .WithMessage("ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''");
        }

        /// <summary>
        ///     The numpy.strings factories are NumPy's own functions over the operands: <c>StrLen</c>, the nine predicates and
        ///     the searches with every <c>start</c> / <c>end</c> form — a literal, a negative one, a constant-only expression, an
        ///     array, a computed sub-tree — and their results feed the fused remainder.
        /// </summary>
        [TestMethod]
        public void StringsFunctions_AreNumPysStatements()
        {
            var u = U();
            var t = T();
            var age = Age();
            np.evaluate(NDExpr.StrLen(u)).ToArray<long>().Should().Equal(0L, 1, 1, 2, 1, 1, 1, 3, 2);
            np.evaluate(NDExpr.StrLen(S())).ToArray<long>().Should().Equal(0L, 1, 1, 2, 1, 2, 3, 3, 2);
            np.evaluate(NDExpr.StrLen(t)).ToArray<long>().Should().Equal(0L, 1, 1, 2, 1, 1, 1, 3, 2);
            ((long)np.evaluate(NDExpr.Sum(NDExpr.StrLen(t)))).Should().Be(12);

            np.evaluate(NDExpr.IsAlpha(u)).ToArray<bool>().Should().Equal(Row("FTTTTTFFT"));
            np.evaluate(NDExpr.IsDigit(np.array(new object[] { "1", "a", "", "²", "١" }, np.dtypes.StringDType())))
                .ToArray<bool>().Should().Equal(Row("TFFTT"));
            np.evaluate(NDExpr.IsNumeric(np.array(new[] { "1", "½", "a", "" }))).ToArray<bool>().Should().Equal(Row("TTFF"));
            np.evaluate(NDExpr.IsUpper(np.array(new[] { "A", "a", "AB1", "1", "" }))).ToArray<bool>().Should().Equal(Row("TFTFF"));
            np.evaluate(NDExpr.IsTitle(np.array(new[] { "Ab Cd", "ab", "AB", "A1b", "" }))).ToArray<bool>().Should().Equal(Row("TFFFF"));
            np.evaluate(NDExpr.IsSpace(np.array(new[] { " ", "\t\n", "a ", "" }).astype(np.dtype("S2"))))
                .ToArray<bool>().Should().Equal(Row("TTFF"));
            np.evaluate(NDExpr.IsAlnum(np.array(new object[] { "a1", "a 1", "", "é" }, np.dtypes.StringDType())))
                .ToArray<bool>().Should().Equal(Row("TFFT"));
            np.evaluate(NDExpr.IsLower(np.array(new object[] { "ab", "aB", "1", "a1" }, np.dtypes.StringDType())))
                .ToArray<bool>().Should().Equal(Row("TFFT"));
            np.evaluate(NDExpr.IsDecimal(np.array(new[] { "12", "½", "" }))).ToArray<bool>().Should().Equal(Row("TFF"));

            np.evaluate(NDExpr.Find(u, "a")).ToArray<long>().Should().Equal(-1L, 0, 0, 0, -1, -1, -1, 0, -1);
            np.evaluate(NDExpr.Find(u, "a", 1)).ToArray<long>().Should().Equal(-1L, -1, -1, -1, -1, -1, -1, -1, -1);
            np.evaluate(NDExpr.Find(u, "b", -2)).ToArray<long>().Should().Equal(-1L, -1, -1, 1, 0, -1, -1, 2, -1);
            np.evaluate(NDExpr.Find(u, "a", -(NDExpr)3)).ToArray<long>().Should().Equal(-1L, 0, 0, 0, -1, -1, -1, 0, -1);
            np.evaluate(NDExpr.Find(u, "a", (NDExpr)age - 5)).ToArray<long>().Should().Equal(-1L, -1, 0, 0, -1, -1, -1, -1, -1);
            np.evaluate(NDExpr.Find(u, "a", 0, 2)).ToArray<long>().Should().Equal(-1L, 0, 0, 0, -1, -1, -1, 0, -1);
            np.evaluate(NDExpr.RFind(t, "a")).ToArray<long>().Should().Equal(-1L, 0, 0, 0, -1, -1, -1, 0, -1);
            np.evaluate(NDExpr.Count(u, "")).ToArray<long>().Should().Equal(1L, 2, 2, 3, 2, 2, 2, 4, 3);
            np.evaluate(NDExpr.Count(t, "a", 1)).ToArray<long>().Should().Equal(0L, 0, 0, 0, 0, 0, 0, 0, 0);
            np.evaluate(NDExpr.Index(u["3:4"], "b")).ToArray<long>().Should().Equal(1L);
            np.evaluate(NDExpr.RIndex(t["3:5"], "b")).ToArray<long>().Should().Equal(1L, 0);
            var bytesA = np.array(new[] { "a" }).astype(np.dtype("S1")).reshape(Shape.NewScalar());
            np.evaluate(NDExpr.StartsWith(S(), bytesA)).ToArray<bool>().Should().Equal(Row("FTTTFFFTF"));
            np.evaluate(NDExpr.StartsWith(u, "a", 1)).ToArray<bool>().Should().Equal(Row("FFFFFFFFF"));
            np.evaluate(NDExpr.EndsWith(t, "b")).ToArray<bool>().Should().Equal(Row("FFFTTFFTF"));
            np.evaluate(NDExpr.EndsWith(u, "z", -1)).ToArray<bool>().Should().Equal(Row("FFFFFFFFT"));

            ((long)np.evaluate(NDExpr.CountNonzero(NDExpr.StartsWith(u, "a") & ((NDExpr)age > 3)))).Should().Be(3);
            np.evaluate(NDExpr.Where(NDExpr.IsAlpha(u), age, -(NDExpr)age)).ToArray<long>().Should().Equal(0L, 7, 4, 1, 8, 5, -2, -9, 6);
            using var tf = np.array(new object[] { "a", "", double.NaN, "b" }, np.dtypes.StringDType(na_object: double.NaN));
            np.evaluate(NDExpr.StartsWith(tf, "a")).ToArray<bool>().Should().Equal(Row("TFFF"));
        }

        /// <summary>
        ///     The numpy.strings factories raise NumPy's own errors per call: no loop for the operand classes (an <c>S</c> array
        ///     with a <c>str</c> substring, a number, <c>isdecimal</c> over bytes), a float start, a missing match of
        ///     <c>index</c>, and the StringDType NA rules of each function.
        /// </summary>
        [TestMethod]
        public void StringsFunctions_RaiseNumPysErrors()
        {
            var u = U();
            new Action(() => np.evaluate(NDExpr.Find(S(), "a"))).Should().Throw<TypeError>().WithMessage(
                "ufunc 'find' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes._PyLongDType'>, <class 'numpy.dtypes._PyLongDType'>) -> None");
            new Action(() => np.evaluate(NDExpr.Find(u, "a", 1.5))).Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'find' input 2 from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
            new Action(() => np.evaluate(NDExpr.StrLen(Age()))).Should().Throw<TypeError>().WithMessage(
                "ufunc 'str_len' did not contain a loop with signature matching types <class 'numpy.dtypes.Int64DType'> -> None");
            new Action(() => np.evaluate(NDExpr.IsDecimal(S()))).Should().Throw<TypeError>().WithMessage(
                "ufunc 'isdecimal' did not contain a loop with signature matching types <class 'numpy.dtypes.BytesDType'> -> None");
            new Action(() => np.evaluate(NDExpr.Index(u, "z"))).Should().Throw<ValueError>().WithMessage("substring not found");

            using var tn = np.array(new object[] { "a", "", null, "b" }, np.dtypes.StringDType(na_object: null));
            using var tf = np.array(new object[] { "a", "", double.NaN, "b" }, np.dtypes.StringDType(na_object: double.NaN));
            new Action(() => np.evaluate(NDExpr.StrLen(tn))).Should().Throw<ValueError>().WithMessage("The length of a null string is undefined");
            new Action(() => np.evaluate(NDExpr.IsAlpha(tn))).Should().Throw<ValueError>()
                .WithMessage("Cannot use the isalpha function with a null that is not a nan-like value");
            new Action(() => np.evaluate(NDExpr.Find(tf, "a"))).Should().Throw<ValueError>()
                .WithMessage("'find' not supported for null values that are not strings.");
            new Action(() => np.evaluate(NDExpr.StartsWith(tn, "a"))).Should().Throw<ValueError>()
                .WithMessage("'startswith' not supported for null values that are not nan-like or strings.");
            new Action(() => NDExpr.Find(u, null)).Should().Throw<ArgumentNullException>().WithParameterName("sub");
            new Action(() => NDExpr.StartsWith(u, null)).Should().Throw<ArgumentNullException>().WithParameterName("prefix");
        }

        /// <summary>
        ///     A <c>Cast</c> of text to a number is NumPy's parse (<c>x.astype(dtype)</c>), its result fused into the rest:
        ///     integers and floats from S / U / StringDType, the bool cast's truth, and NumPy's errors for a text that does not
        ///     parse, an integer out of the target's range and a StringDType NA.
        /// </summary>
        [TestMethod]
        public void CastToNumbers_IsNumPysParse()
        {
            var nums = np.array(new[] { "1", "2", "-3", "300" });
            np.evaluate(NDExpr.Cast(nums, NPTypeCode.Int64) * 2).ToArray<long>().Should().Equal(2L, 4, -6, 600);
            np.evaluate(NDExpr.Cast(nums, NPTypeCode.Double) + 0.5).ToArray<double>().Should().Equal(1.5, 2.5, -2.5, 300.5);
            np.evaluate(NDExpr.Cast(U(), NPTypeCode.Boolean)).ToArray<bool>().Should().Equal(Row("FTTTTTTTT"));
            np.evaluate(NDExpr.Cast(np.array(new[] { "12", "-4" }).astype(np.dtype("S2")), NPTypeCode.Int64))
                .ToArray<long>().Should().Equal(12L, -4);
            np.evaluate(NDExpr.Cast(np.array(new object[] { "1.5", "-2e3", "inf" }, np.dtypes.StringDType()), NPTypeCode.Single))
                .ToArray<float>().Should().Equal(1.5f, -2000f, float.PositiveInfinity);

            new Action(() => np.evaluate(NDExpr.Cast(nums, NPTypeCode.SByte)))
                .Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");
            new Action(() => np.evaluate(NDExpr.Cast(nums, NPTypeCode.Byte)))
                .Should().Throw<OverflowException>().WithMessage("Python integer -3 out of bounds for uint8");
            new Action(() => np.evaluate(NDExpr.Cast(U(), NPTypeCode.Int64)))
                .Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('')");
            using var tf = np.array(new object[] { "a", "", double.NaN, "b" }, np.dtypes.StringDType(na_object: double.NaN));
            new Action(() => np.evaluate(NDExpr.Cast(tf, NPTypeCode.Double)))
                .Should().Throw<ValueError>().WithMessage("could not convert string to float: 'a'");
        }

        /// <summary>
        ///     The stages take the call's keywords through the fused remainder (<c>out=</c>, <c>where=</c>, <c>dtype=</c>) and
        ///     the explicit handles: a <c>Compile()</c>d tree over a strings function and the positional form, with the
        ///     function's int64 result fused with a numeric operand.
        /// </summary>
        [TestMethod]
        public void StringsStages_TakeTheKeywordsAndTheHandles()
        {
            var u = U();
            var age = Age();
            var o = np.zeros(new Shape(9), NPTypeCode.Int64);
            var returned = np.evaluate(NDExpr.StrLen(u) + 1, @out: o);
            ReferenceEquals(returned, o).Should().BeTrue();
            o.ToArray<long>().Should().Equal(1L, 2, 2, 3, 2, 2, 2, 4, 3);

            var masked = np.full(new Shape(9), -7L);
            var mask = np.array(Row("TFTFTFTFT"));
            np.evaluate(NDExpr.StrLen(u), @out: masked, where: mask);
            masked.ToArray<long>().Should().Equal(0L, -7, 1, -7, 1, -7, 1, -7, 2);

            var widened = np.evaluate(NDExpr.StrLen(u), dtype: np.float64);
            widened.typecode.Should().Be(NPTypeCode.Double);
            widened.ToArray<double>().Should().Equal(0.0, 1, 1, 2, 1, 1, 1, 3, 2);

            var handle = NDExpr.Find(u, "a").Compile();
            handle.IsReduction.Should().BeFalse();
            handle.ResultType.Should().Be(NPTypeCode.Int64);
            handle.Evaluate().ToArray<long>().Should().Equal(-1L, 0, 0, 0, -1, -1, -1, 0, -1);
            NDExpr.CountNonzero(NDExpr.IsAlpha(u)).Compile().IsReduction.Should().BeTrue();

            np.evaluate(NDExpr.StrLen(NDExpr.Input(0)) * NDExpr.Input(1), new[] { u, age })
                .ToArray<long>().Should().Equal(0L, 7, 4, 2, 8, 5, 2, 27, 12);
            var positional = NDExpr.StrLen(NDExpr.Input(0)).Compile(NPTypeCode.String);
            positional.Evaluate(new[] { T() }).ToArray<long>().Should().Equal(0L, 1, 1, 2, 1, 1, 1, 3, 2);
        }

        /// <summary>
        ///     A strings function's text arguments are inputs like a comparison's: trees that differ only in their literal
        ///     substrings share one program.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // reads the process-wide NDExpr program cache, which another test may clear when it fills
        public void StringsArguments_AreInputs_OneProgramForEveryText()
        {
            var u = U();
            var first = NDExpr.StartsWith(u, "a").GetProgram(null);
            foreach (var prefix in new[] { "zz", "", "a\0b", "much longer than any element" })
                Assert.AreSame(first, NDExpr.StartsWith(u, prefix).GetProgram(null), $"'{prefix}': one program serves every prefix");
            np.evaluate(NDExpr.StartsWith(u, "zz")).ToArray<bool>().Should().Equal(Row("FFFFFFFFT"));
        }

        /// <summary>The elements of a text result in C order: a <c>U</c> / StringDType element as its string (an NA as its NA
        ///     object: null, or NaN), an <c>S</c> element as its bytes read as Latin-1 (one char per byte).</summary>
        /// <param name="a">A text array.</param>
        /// <returns>The elements.</returns>
        private static object[] Txt(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => a.Storage.GetAtIndex(i) switch
        {
            byte[] bytes => System.Text.Encoding.Latin1.GetString(bytes),
            var v => v,
        }).ToArray();

        /// <summary>Asserts that an evaluation raises NumPy's error: the house exception type and NumPy's exact text.</summary>
        /// <typeparam name="TException">The exception type NumSharp raises for NumPy's (a <see cref="TypeError"/> for NumPy's
        ///     TypeError and its UFuncTypeError / DTypePromotionError, a <see cref="ValueError"/> for its ValueError).</typeparam>
        /// <param name="tree">Builds the tree (inside the assertion, so a factory refusal is caught too).</param>
        /// <param name="message">NumPy's text.</param>
        private static void RaisesNumPys<TException>(Func<NDExpr> tree, string message) where TException : Exception
            => new Action(() => np.evaluate(tree())).Should().Throw<TException>().Which.Message.Should().Be(message);

        /// <summary>
        ///     Phase E3, the text results of the binary ufuncs: np.add concatenates two strings of one class (U with U to the summed
        ///     width, S with S, StringDType with StringDType, U or a Python str — a StringDType element keeps a trailing NUL, a U
        ///     element does not), np.multiply repeats StringDType text by an integer from either side, np.maximum /
        ///     np.minimum compare StringDType (a NaN-like NA wins), and an <c>S</c> / <c>U</c> repeat exists only into an
        ///     <c>out=</c>, NumPy's own rule.
        /// </summary>
        [TestMethod]
        public void TextResults_ConcatenationRepeatAndMaxMin_AreNumPys()
        {
            var u = U();
            var t = T();
            var age = Age();

            var uu = np.evaluate(NDExpr.Add(u, UR()));
            uu.dtype.ToString(true).Should().Be("dtype('<U6')");
            Txt(uu).Should().Equal("zz", "aa\0b", "a\U0001F600", "abé", "bb", "éab", "\U0001F600a", "a\0ba", "zz");
            var tu = np.evaluate(NDExpr.Add(t, UR()));
            tu.dtype.ToString(true).Should().Be("StringDType()");
            Txt(tu).Should().Equal("zz", "aa\0b", "a\0\U0001F600", "abé", "bb", "éab", "\U0001F600a", "a\0ba", "zz");
            var bang = np.evaluate(NDExpr.Add(u, "!"));
            bang.dtype.ToString(true).Should().Be("dtype('<U4')");
            Txt(bang).Should().Equal("!", "a!", "a!", "ab!", "b!", "é!", "\U0001F600!", "a\0b!", "zz!");
            var ss = np.evaluate(NDExpr.Add(S(), SR()));
            ss.dtype.ToString(true).Should().Be("dtype('S6')");
            Txt(ss).Should().Equal("zz", "aa\0b", "að\u009f\u0098", "abÃ©", "bb", "Ã©ab", "ð\u009f\u0098a",
                "a\0ba", "zz");

            Txt(np.evaluate(NDExpr.Multiply(t, (NDExpr)age % 3)))
                .Should().Equal("", "a", "a\0", "ab", "bb", "éé", "\U0001F600\U0001F600", "", "");
            Txt(np.evaluate(NDExpr.Multiply(2, t)))
                .Should().Equal("", "aa", "a\0a\0", "abab", "bb", "éé", "\U0001F600\U0001F600", "a\0ba\0b", "zzzz");
            RaisesNumPys<TypeError>(() => NDExpr.Multiply(u, 2),
                "The 'out' kwarg is necessary when using the string multiply ufunc directly. Use numpy.strings.multiply to " +
                "multiply strings without specifying 'out'.");
            var into = np.empty(new Shape(9), np.dtype("U6"));
            np.evaluate(NDExpr.Multiply(u, 2), @out: into).Should().BeSameAs(into);
            Txt(into).Should().Equal("", "aa", "aa", "abab", "bb", "éé", "\U0001F600\U0001F600", "a\0ba\0b", "zzzz");

            Txt(np.evaluate(NDExpr.Maximum(t, TR())))
                .Should().Equal("zz", "a\0b", "\U0001F600", "é", "b", "é", "\U0001F600", "a\0b", "zz");
            Txt(np.evaluate(NDExpr.Minimum(t, TR()))).Should().Equal("", "a", "a\0", "ab", "b", "ab", "a\0", "a", "");
            var nanNa = np.dtypes.StringDType(na_object: double.NaN);
            using var tn = np.array(new object[] { "a", double.NaN, "b" }, nanNa);
            using var other = np.array(new object[] { "b", "a", "a" }, nanNa);
            var nanMax = Txt(np.evaluate(NDExpr.Maximum(tn, other)));
            nanMax[0].Should().Be("b");
            double.IsNaN((double)nanMax[1]).Should().BeTrue("a NaN-like NA wins np.maximum");
            nanMax[2].Should().Be("b");
        }

        /// <summary>
        ///     Phase E3, <c>Where</c> with text branches is NumPy's <c>np.where</c>: the branches' NEP 50 promotion (U with a
        ///     Python str keeps the wider U, U with int64 is <c>&lt;U21</c>, with float64 <c>&lt;U32</c>, U with StringDType is
        ///     StringDType), a text condition read through its truth inside the call, and NumPy's DTypePromotionError for a
        ///     Python int beside U, StringDType beside int64 and S beside StringDType.
        /// </summary>
        [TestMethod]
        public void Where_WithTextBranches_FollowsNumPysPromotion()
        {
            var u = U();
            var age = Age();
            var x = np.evaluate(NDExpr.Where((NDExpr)age > 3, u, "x"));
            x.dtype.ToString(true).Should().Be("dtype('<U3')");
            Txt(x).Should().Equal("x", "a", "a", "x", "b", "é", "x", "a\0b", "zz");
            var ints = np.evaluate(NDExpr.Where((NDExpr)age > 3, u, age));
            ints.dtype.ToString(true).Should().Be("dtype('<U21')");
            Txt(ints).Should().Equal("0", "a", "a", "1", "b", "é", "2", "a\0b", "zz");
            var floats = np.evaluate(NDExpr.Where((NDExpr)age > 3, u, (NDExpr)age * 1.5));
            floats.dtype.ToString(true).Should().Be("dtype('<U32')");
            Txt(floats).Should().Equal("0.0", "a", "a", "1.5", "b", "é", "3.0", "a\0b", "zz");
            var mixed = np.evaluate(NDExpr.Where((NDExpr)age > 3, u, TR()));
            mixed.dtype.ToString(true).Should().Be("StringDType()");
            Txt(mixed).Should().Equal("zz", "a", "a", "é", "b", "é", "a\0", "a\0b", "zz");
            Txt(np.evaluate(NDExpr.Where(u, T(), "-")))
                .Should().Equal("-", "a", "a\0", "ab", "b", "é", "\U0001F600", "a\0b", "zz");

            const string noCommon = "This means that no common DType exists for the given inputs. For example they cannot be " +
                "stored in a single array unless the dtype is `object`. The full list of DTypes is: ";
            new Action(() => np.evaluate(NDExpr.Where((NDExpr)age > 3, u, 5))).Should().Throw<DTypePromotionError>()
                .Which.Message.Should().StartWith("The DType <class 'numpy.dtypes.StrDType'> could not be promoted by " +
                                                  "<class 'numpy.dtypes._PyLongDType'>. " + noCommon);
            new Action(() => np.evaluate(NDExpr.Where((NDExpr)age > 3, T(), age))).Should().Throw<DTypePromotionError>()
                .Which.Message.Should().StartWith("The DType <class 'numpy.dtypes.StringDType'> could not be promoted by " +
                                                  "<class 'numpy.dtypes.Int64DType'>. " + noCommon);
            new Action(() => np.evaluate(NDExpr.Where((NDExpr)age > 3, S(), T()))).Should().Throw<DTypePromotionError>()
                .Which.Message.Should().StartWith("The DType <class 'numpy.dtypes.BytesDType'> could not be promoted by " +
                                                  "<class 'numpy.dtypes.StringDType'>. " + noCommon);
        }

        /// <summary>
        ///     Phase E3, a cast with text on either side is NumPy's <c>astype</c>: numbers formatted and cut to a width
        ///     (<c>10.5 → '10.'</c> in <c>U3</c>), bools as <c>'True'</c> / <c>'False'</c>, text recast between classes and
        ///     widths (a StringDType NaN-like NA as <c>'nan'</c>), NumPy's ascii refusal of a non-ASCII character into <c>S</c>;
        ///     the storage lane <see cref="NPTypeCode.String"/> names no dtype and is refused when the node is built.
        /// </summary>
        [TestMethod]
        public void Casts_ToAndFromText_AreNumPysAstype()
        {
            var age = Age();
            var u1 = np.evaluate(NDExpr.Cast(age, "U1"));
            u1.dtype.ToString(true).Should().Be("dtype('<U1')");
            Txt(u1).Should().Equal("0", "7", "4", "1", "8", "5", "2", "9", "6");
            Txt(np.evaluate(NDExpr.Cast((NDExpr)age * 1.5, "U3")))
                .Should().Equal("0.0", "10.", "6.0", "1.5", "12.", "7.5", "3.0", "13.", "9.0");
            var flags = np.evaluate(NDExpr.Cast((NDExpr)age > 3, "S"));
            flags.dtype.ToString(true).Should().Be("dtype('S5')");
            Txt(flags).Should().Equal("False", "True", "True", "False", "True", "True", "False", "True", "True");
            Txt(np.evaluate(NDExpr.Cast(T(), "U2"))).Should().Equal("", "a", "a", "ab", "b", "é", "\U0001F600", "a", "zz");
            var asT = np.evaluate(NDExpr.Cast(U(), np.dtypes.StringDType()));
            asT.dtype.ToString(true).Should().Be("StringDType()");
            Txt(asT).Should().Equal("", "a", "a", "ab", "b", "é", "\U0001F600", "a\0b", "zz");
            using var tn = np.array(new object[] { "a", double.NaN, "b" }, np.dtypes.StringDType(na_object: double.NaN));
            Txt(np.evaluate(NDExpr.Cast(tn, "U3"))).Should().Equal("a", "nan", "b");
            new Action(() => np.evaluate(NDExpr.Cast(U(), "S"))).Should().Throw<UnicodeEncodeError>()
                .Which.Message.Should().Be("'ascii' codec can't encode character '\\xe9' in position 0: ordinal not in range(128)");
            new Action(() => NDExpr.Cast(U(), NPTypeCode.String)).Should().Throw<NotSupportedException>()
                .WithMessage("*asked for a dtype by storage lane (NPTypeCode.String)*");
        }

        /// <summary>
        ///     Phase E3, the root reductions of text are NumPy's: np.sum / np.max / np.min over StringDType (a flat one over a 1-D
        ///     operand, per axis with keepdims, and NumPy's refusal of a flat one over several axes), np.argmax / np.argmin over
        ///     U / S / StringDType (a NaN-like NA ranks last), np.mean of a NaN-like NA's sum (float64 NaN), and every other
        ///     answer NumPy's own text — <c>NanSum</c> / <c>NanProd</c> included, which are np.sum / np.prod of text in NumPy
        ///     (its NaN replacement leaves text as it is), not their numeric <c>Where(IsNaN(x), fill, x)</c> rewrite.
        /// </summary>
        [TestMethod]
        public void TextReductions_AreNumPys()
        {
            var t = T();
            var t33 = T().reshape(3, 3);
            var sum = np.evaluate(NDExpr.Sum(t));
            sum.ndim.Should().Be(0);
            Txt(sum).Should().Equal("aa\0abbé\U0001F600a\0bzz");
            RaisesNumPys<ValueError>(() => NDExpr.Sum(t33),
                "reduction operation 'add' is not reorderable, so at most one axis may be specified");
            Txt(np.evaluate(NDExpr.Sum(t33, 0))).Should().Equal("ab\U0001F600", "aba\0b", "a\0ézz");
            var rows = np.evaluate(NDExpr.Sum(t33, 1, true));
            rows.shape.Should().Equal(3, 1);
            Txt(rows).Should().Equal("aa\0", "abbé", "\U0001F600a\0bzz");
            Txt(np.evaluate(NDExpr.Max(t))).Should().Equal("\U0001F600");
            Txt(np.evaluate(NDExpr.Min(t33, 1))).Should().Equal("", "ab", "a\0b");
            Txt(np.evaluate(NDExpr.NanSum(t))).Should().Equal("aa\0abbé\U0001F600a\0bzz");

            ((long)np.evaluate(NDExpr.ArgMax(U()))).Should().Be(6);
            ((long)np.evaluate(NDExpr.ArgMin(U()))).Should().Be(0);
            ((long)np.evaluate(NDExpr.ArgMax(S()))).Should().Be(6);
            np.evaluate(NDExpr.ArgMax(t33, 0)).ToArray<long>().Should().Equal(2, 1, 1);
            var argmin = np.evaluate(NDExpr.ArgMin(U().reshape(3, 3), 1, true));
            argmin.shape.Should().Equal(3, 1);
            argmin.ToArray<long>().Should().Equal(0, 0, 1);
            using var tn = np.array(new object[] { "a", double.NaN, "b" }, np.dtypes.StringDType(na_object: double.NaN));
            ((long)np.evaluate(NDExpr.ArgMax(tn))).Should().Be(1, "a NaN-like NA ranks last");
            ((long)np.evaluate(NDExpr.ArgMin(tn))).Should().Be(0);
            var mean = np.evaluate(NDExpr.Mean(tn));
            mean.dtype.ToString(true).Should().Be("dtype('float64')");
            double.IsNaN((double)mean).Should().BeTrue();
            double.IsNaN((double)np.evaluate(NDExpr.NanMean(tn))).Should().BeTrue();

            RaisesNumPys<TypeError>(() => NDExpr.Mean(t), "ufunc 'divide' not supported for the input types, and the inputs " +
                "could not be safely coerced to any supported types according to the casting rule ''safe''");
            const string addReduce = "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U3'), dtype('<U3'), dtype('<U6'))";
            RaisesNumPys<TypeError>(() => NDExpr.Sum(U()), addReduce);
            RaisesNumPys<TypeError>(() => NDExpr.NanSum(U()), addReduce);
            RaisesNumPys<TypeError>(() => NDExpr.Std(U()), addReduce);
            RaisesNumPys<TypeError>(() => NDExpr.NanProd(t),
                "ufunc 'multiply' did not contain a loop with signature matching types (None, <class 'numpy.dtypes.StringDType'>) -> None");
            RaisesNumPys<TypeError>(() => NDExpr.Prod(U()),
                "ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U3'), dtype('<U3')) -> None");
            RaisesNumPys<TypeError>(() => NDExpr.Var(t), "ufunc 'divide' did not contain a loop with signature matching types " +
                "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.Int64DType'>) -> None");
            RaisesNumPys<TypeError>(() => NDExpr.Ptp(t),
                "ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U1'), dtype('<U1')) -> None");
            RaisesNumPys<TypeError>(() => NDExpr.NanMin(U()),
                "ufunc 'fmin' did not contain a loop with signature matching types (dtype('<U3'), dtype('<U3')) -> None");
            RaisesNumPys<TypeError>(() => NDExpr.NanMax(t),
                "ufunc 'fmax' did not contain a loop with signature matching types (None, <class 'numpy.dtypes.StringDType'>) -> None");
            new Action(() => np.evaluate(NDExpr.Average(t, np.ones(new Shape(9))))).Should().Throw<DTypePromotionError>()
                .Which.Message.Should().StartWith("The DType <class 'numpy.dtypes.StringDType'> could not be promoted by " +
                                                  "<class 'numpy.dtypes.Float64DType'>.");
            using var tnone = np.array(new object[] { "a", null, "b" }, np.dtypes.StringDType(na_object: null));
            RaisesNumPys<ValueError>(() => NDExpr.Sum(tnone), "Cannot add null that is not a nan-like value");
            RaisesNumPys<ValueError>(() => NDExpr.Max(tnone), "Cannot compare null that is not a nan-like value");
            RaisesNumPys<ValueError>(() => NDExpr.ArgMax(np.zeros(new Shape(0), np.dtype("U2"))), "attempt to get argmax of an empty sequence");
            RaisesNumPys<ValueError>(() => NDExpr.Max(np.empty(new Shape(0), np.dtypes.StringDType())),
                "zero-size array to reduction operation maximum which has no identity");

            // A reduction's keywords: dtype= is refused as over numbers; out= takes the flat result by the reduction's rank rule.
            new Action(() => np.evaluate(NDExpr.Sum(t), dtype: np.dtypes.StringDType()))
                .Should().Throw<NotSupportedException>().WithMessage("np.evaluate: dtype= and order= are not supported on a reduction tree*");
            var index = np.zeros(Shape.NewScalar(), np.int64);
            np.evaluate(NDExpr.ArgMax(U()), @out: index).Should().BeSameAs(index);
            ((long)index).Should().Be(6);
            new Action(() => np.evaluate(NDExpr.ArgMax(U()), @out: np.zeros(new Shape(3), np.int64))).Should().Throw<ArgumentException>()
                .WithMessage("output parameter for reduction operation argmax has the wrong number of dimensions: Found 1 but expected 0");
        }

        /// <summary>
        ///     Phase E3, every other node with a text operand raises NumPy's own error when the call runs — the legacy resolver's
        ///     "not supported for the input types" for S / U, the no-loop text naming StringDType's class or the operand
        ///     descriptors, np.clip's naming all three operands, np.round's from its multiply step, np.angle's from its arctan2
        ///     — while np.real / np.imag of text compute. Only a delegate refuses text in its own words.
        /// </summary>
        [TestMethod]
        public void OtherTextNodes_RaiseNumPysOwnErrors()
        {
            var u = U();
            var t = T();
            const string coerced = " not supported for the input types, and the inputs could not be safely coerced to any " +
                                   "supported types according to the casting rule ''safe''";
            RaisesNumPys<TypeError>(() => NDExpr.LogicalNot(u), "ufunc 'logical_not'" + coerced);
            RaisesNumPys<TypeError>(() => NDExpr.LogicalNot(t),
                "ufunc 'logical_not' did not contain a loop with signature matching types <class 'numpy.dtypes.StringDType'> -> None");
            RaisesNumPys<TypeError>(() => NDExpr.IsInf(t),
                "ufunc 'isinf' did not contain a loop with signature matching types <class 'numpy.dtypes.StringDType'> -> None");
            RaisesNumPys<TypeError>(() => NDExpr.Sqrt(u), "ufunc 'sqrt'" + coerced);
            RaisesNumPys<TypeError>(() => NDExpr.Negate(t),
                "ufunc 'negative' did not contain a loop with signature matching types <class 'numpy.dtypes.StringDType'> -> None");
            RaisesNumPys<TypeError>(() => NDExpr.Abs(S()),
                "ufunc 'absolute' did not contain a loop with signature matching types <class 'numpy.dtypes.BytesDType'> -> None");
            RaisesNumPys<TypeError>(() => NDExpr.IsPosInf(S()), "ufunc 'isinf'" + coerced);
            RaisesNumPys<TypeError>(() => NDExpr.Round(u), "ufunc 'rint'" + coerced);
            RaisesNumPys<TypeError>(() => NDExpr.Round(u, 2),
                "ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U3'), dtype('float64')) -> None");
            RaisesNumPys<TypeError>(() => NDExpr.Angle(t), "ufunc 'arctan2' did not contain a loop with signature matching types " +
                "(<class 'numpy.dtypes._PyLongDType'>, <class 'numpy.dtypes.StringDType'>) -> None");
            RaisesNumPys<TypeError>(() => (NDExpr)u - u,
                "ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U3'), dtype('<U3')) -> None");
            RaisesNumPys<TypeError>(() => (NDExpr)t / t, "ufunc 'divide' did not contain a loop with signature matching types " +
                "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.StringDType'>) -> None");
            RaisesNumPys<TypeError>(() => NDExpr.Power(u, 2), "ufunc 'power'" + coerced);
            RaisesNumPys<TypeError>(() => (NDExpr)u % u, "ufunc 'remainder'" + coerced);
            RaisesNumPys<TypeError>(() => NDExpr.BitwiseAnd(u, u), "ufunc 'bitwise_and'" + coerced);
            RaisesNumPys<TypeError>(() => NDExpr.FMax(t, t), "ufunc 'fmax' did not contain a loop with signature matching types " +
                "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.StringDType'>) -> None");
            RaisesNumPys<TypeError>(() => NDExpr.Clip(t, "a", "b"), "ufunc 'clip' did not contain a loop with signature matching " +
                "types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes.StrDType'>) -> None");
            RaisesNumPys<TypeError>(() => NDExpr.Clip(u, "a", "b"),
                "ufunc 'clip' did not contain a loop with signature matching types (dtype('<U3'), dtype('<U1'), dtype('<U1')) -> None");
            // NumPy's statement order: the refused negative runs before the comparison that would read it.
            RaisesNumPys<TypeError>(() => NDExpr.Equal(NDExpr.Negate(u), "a"), "ufunc 'negative' did not contain a loop with " +
                "signature matching types <class 'numpy.dtypes.StrDType'> -> None");

            Txt(np.evaluate(NDExpr.Real(u))).Should().Equal("", "a", "a", "ab", "b", "é", "\U0001F600", "a\0b", "zz");
            var imag = np.evaluate(NDExpr.Imag(t));
            imag.dtype.ToString(true).Should().Be("StringDType()");
            imag.Shape.IsWriteable.Should().BeTrue("np.evaluate answers with an array of its own");
            Txt(imag).Should().Equal("", "", "", "", "", "", "", "", "");
            new Action(() => np.evaluate(NDExpr.Call((Func<double, double>)System.Math.Abs, u)))
                .Should().Throw<StringComputationNotSupportedException>().WithMessage("*NDExpr.Call delegate*");
        }

        /// <summary>
        ///     Phase E3, text results compose with every other node: a comparison, a numpy.strings function, a reduction, a
        ///     logical node, a cast and a where read a stage's text result, as NumPy's next statement reads its temporary.
        /// </summary>
        [TestMethod]
        public void TextResults_FeedTheRestOfTheTree()
        {
            var u = U();
            var age = Age();
            B(np.evaluate(NDExpr.Equal(NDExpr.Add(u, "!"), "a!"))).Should().Equal(Row("FTTFFFFFF"));
            np.evaluate(NDExpr.StrLen(NDExpr.Add(T(), T()))).ToArray<long>().Should().Equal(0, 2, 3, 4, 2, 2, 2, 6, 4);
            ((long)np.evaluate(NDExpr.ArgMax(NDExpr.Add(u, UR())))).Should().Be(6);
            using var digits = np.array(new[] { "1", "22", "3" });
            np.evaluate(NDExpr.Cast(NDExpr.Add(digits, "0"), NPTypeCode.Int64) * 2).ToArray<long>().Should().Equal(20, 440, 60);
            B(np.evaluate(NDExpr.LogicalAnd(NDExpr.Add(u, u), (NDExpr)age > 3))).Should().Equal(Row("FTTFTTFTT"));
            var picked = np.evaluate(NDExpr.Where((NDExpr)age > 3, NDExpr.Add(u, "+"), u));
            picked.dtype.ToString(true).Should().Be("dtype('<U4')");
            Txt(picked).Should().Equal("", "a+", "a+", "ab", "b+", "é+", "\U0001F600", "a\0b+", "zz+");
        }

        /// <summary>
        ///     Phase E3, a text result takes the call's keywords: a root ufunc writes <c>out=</c> itself (NumPy's cut to the
        ///     out's width, its same_kind refusal), <c>where=</c> writes only the selected elements, <c>dtype=</c> is NumPy's
        ///     astype, <c>order=</c> lays a fresh result out, a numeric result is formatted into a text <c>out=</c> /
        ///     <c>dtype=</c>, and <c>casting='unsafe'</c> parses text into a numeric <c>out=</c>.
        /// </summary>
        [TestMethod]
        public void TextResults_TakeTheKeywords()
        {
            var u = U();
            var age = Age();
            var cut = np.empty(new Shape(9), np.dtype("U2"));
            np.evaluate(NDExpr.Add(u, u), @out: cut).Should().BeSameAs(cut);
            Txt(cut).Should().Equal("", "aa", "aa", "ab", "bb", "éé", "\U0001F600\U0001F600", "a", "zz");
            new Action(() => np.evaluate(NDExpr.Add(u, u), @out: np.empty(new Shape(9), np.int64))).Should().Throw<ArgumentException>()
                .Which.Message.Should().Be("Cannot cast ufunc 'add' output from dtype('<U6') to dtype('int64') with casting rule 'same_kind'");
            var masked = np.full(new Shape(9), "?", np.dtype("U4"));
            np.evaluate(NDExpr.Add(u, "-"), @out: masked, where: (NDArray)(age > 3));
            Txt(masked).Should().Equal("?", "a-", "a-", "?", "b-", "é-", "?", "a\0b-", "zz-");

            Txt(np.evaluate(NDExpr.Equal(u, "a"), @out: np.empty(new Shape(9), np.dtype("U5"))))
                .Should().Equal("False", "True", "True", "False", "False", "False", "False", "False", "False");
            Txt(np.evaluate((NDExpr)age * 3, @out: np.empty(new Shape(9), np.dtype("U1"))))
                .Should().Equal("0", "2", "1", "3", "2", "1", "6", "2", "1");
            var formatted = np.evaluate((NDExpr)age * 3, dtype: np.dtype("U1"));
            formatted.dtype.ToString(true).Should().Be("dtype('<U1')");
            Txt(formatted).Should().Equal("0", "2", "1", "3", "2", "1", "6", "2", "1");
            var recast = np.evaluate(NDExpr.Where((NDExpr)age > 3, u, "x"), dtype: np.dtypes.StringDType());
            recast.dtype.ToString(true).Should().Be("StringDType()");

            var grid = np.evaluate(NDExpr.Add(U().reshape(3, 3).T, "-"));
            grid.Shape.IsFContiguous.Should().BeTrue("np.add of a transpose keeps NumPy's K order");
            var c = np.evaluate(NDExpr.Add(U().reshape(3, 3).T, "-"), order: 'C');
            c.Shape.IsContiguous.Should().BeTrue();
            Txt(c).Should().Equal(Txt(grid));

            using var digits = np.array(new[] { "1", "22", "3" });
            var parsed = np.empty(new Shape(3), np.int64);
            new Action(() => np.evaluate(NDExpr.Add(digits, "0"), @out: parsed)).Should().Throw<ArgumentException>()
                .WithMessage("Cannot cast ufunc 'add' output from dtype('<U3') to dtype('int64') with casting rule 'same_kind'");
            np.evaluate(NDExpr.Add(digits, "0"), @out: parsed, casting: "unsafe");
            parsed.ToArray<long>().Should().Equal(10, 220, 30);
        }

        /// <summary>
        ///     Phase E3, a bare text leaf as the whole expression is a copy of it (np.evaluate answers with an array of its own),
        ///     a C# string alone a 0-d <c>&lt;U{len}</c>; a text program serves every width of its operands and keeps a text
        ///     cast's descriptor and NanSum's NaN-family mark in its identity.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // reads the process-wide NDExpr program cache, which another test may clear when it fills
        public void BareTextRoot_AndTextProgramIdentity()
        {
            var u = U();
            var copy = np.evaluate((NDExpr)u);
            copy.Should().NotBeSameAs(u);
            copy.dtype.ToString(true).Should().Be("dtype('<U3')");
            Txt(copy).Should().Equal(Txt(u));
            var literal = np.evaluate((NDExpr)"abc");
            literal.ndim.Should().Be(0);
            Txt(literal).Should().Equal("abc");

            using var narrow = np.array(new[] { "a", "b" });
            using var wide = np.array(new[] { "long text", "x" });
            Assert.AreSame(NDExpr.Add(NDExpr.Input(0), NDExpr.Input(1)).GetProgram(new[] { narrow, wide }),
                NDExpr.Add(NDExpr.Input(0), NDExpr.Input(1)).GetProgram(new[] { wide, narrow }), "one program for every width");
            Assert.AreNotSame(NDExpr.Cast(u, "U5").GetProgram(null), NDExpr.Cast(u, "U3").GetProgram(null), "the descriptor is identity");
            var t = T();
            Assert.AreNotSame(NDExpr.NanSum(t).GetProgram(null),
                NDExpr.Sum(NDExpr.Where(NDExpr.IsNaN(t), 0, t)).GetProgram(null), "NanSum is np.nansum, not its rewrite");
        }

        /// <summary>
        ///     A text value is an input, not IL: trees that differ only in their literal texts (of any width) share one program,
        ///     a hoisted 0-d text array is read per call, and the explicit handles and the positional form take text operands.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // reads the process-wide NDExpr program cache, which another test may clear when it fills
        public void TextValues_AreInputs_OneProgramForEveryText()
        {
            var u = U();
            var first = NDExpr.Equal(u, "a").GetProgram(null);
            foreach (var text in new[] { "zz", "", "a\0b", "\U0001F600", "much longer than any element" })
            {
                var root = NDExpr.Equal(u, text);
                Assert.AreSame(first, root.GetProgram(null), $"'{text}': one program serves every literal text");
                B(np.evaluate(root)).Should().Equal(Texts.Select(x => x.TrimEnd('\0') == text.TrimEnd('\0')).ToArray());
            }

            NDArray key = "zz";
            var hoisted = NDExpr.Equal(u, key);
            B(np.evaluate(hoisted)).Should().Equal(Row("FFFFFFFFT"));
            np.copyto(key, "b");   // the 0-d text array changes in place; the same root reads the new value
            B(np.evaluate(hoisted)).Should().Equal(Row("FFFFTFFFF"));

            var handle = NDExpr.Equal(u, "ab").Compile();
            handle.IsReduction.Should().BeFalse();
            handle.ResultType.Should().Be(NPTypeCode.Boolean);
            B(handle.Evaluate()).Should().Equal(Row("FFFTFFFFF"));
            NDExpr.CountNonzero(NDExpr.Equal(u, "ab")).Compile().IsReduction.Should().BeTrue();

            var positional = NDExpr.Less(NDExpr.Input(0), NDExpr.Input(1)).Compile(NPTypeCode.String, NPTypeCode.String);
            B(positional.Evaluate(new[] { u, (NDArray)"b" })).Should().Equal(Row("TTTTFFFTF"));
            B(positional.Evaluate(new[] { T(), TR() })).Should().Equal(Row("TTTTFFFFF"), "another text class, the same handle");
            B(np.evaluate(NDExpr.Greater(NDExpr.Input(0), NDExpr.Input(1)), new[] { u, (NDArray)"a" })).Should().Equal(Row("FFFTTTTTT"));
        }
    }
}
