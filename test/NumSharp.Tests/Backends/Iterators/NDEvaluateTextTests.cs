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

        /// <summary>
        ///     Every other node with a text operand is refused when the program is built, naming the node, so no text is read and
        ///     no result is half computed (the plan's phase E3 lifts these refusals: text results, text reductions and NumPy's
        ///     own error for every remaining node, <c>LogicalNot</c> of text included).
        /// </summary>
        [TestMethod]
        public void OtherTextNodes_AreRefusedByName()
        {
            var u = U();
            var t = T();
            var age = Age();
            void Refused(Func<NDExpr> tree, string pattern)
                => new Action(() => np.evaluate(tree())).Should().Throw<StringComputationNotSupportedException>().WithMessage(pattern);

            Refused(() => NDExpr.LogicalNot(u), "*NDExpr.LogicalNot over text*");
            Refused(() => NDExpr.IsInf(t), "*NDExpr.IsInf over text*");
            Refused(() => NDExpr.Cast(u, NPTypeCode.String), "*NDExpr.Cast to a text dtype over text*");
            Refused(() => (NDExpr)u + u, "*NDExpr.Add over text*");
            Refused(() => (NDExpr)t * 2, "*NDExpr.Multiply over text*");
            Refused(() => NDExpr.Maximum(t, t), "*NDExpr.Maximum over text*");
            Refused(() => NDExpr.Where((NDExpr)age > 3, u, "x"), "*NDExpr.Where with text branches over text*");
            Refused(() => NDExpr.Sqrt(u), "*NDExpr.Sqrt over text*");
            Refused(() => NDExpr.ArgMax(u), "*NDExpr.ArgMax over text*");
            Refused(() => NDExpr.Sum(t), "*NDExpr.Sum over text*");
            Refused(() => (NDExpr)u, "*a text result (a bare text leaf as the expression)*");
            new Action(() => np.evaluate(NDExpr.Call((Func<double, double>)System.Math.Abs, u)))
                .Should().Throw<StringComputationNotSupportedException>().WithMessage("*NDExpr.Call delegate*");

            var textOut = np.zeros(new Shape(9), np.dtype("U5"));
            new Action(() => np.evaluate(NDExpr.Equal(u, "a"), @out: textOut))
                .Should().Throw<StringComputationNotSupportedException>().WithMessage("*a text result (a text out=)*");
            new Action(() => np.evaluate(NDExpr.Equal(u, "a"), dtype: np.dtype("U5")))
                .Should().Throw<StringComputationNotSupportedException>().WithMessage("*a text result (a text dtype=)*");
            new Action(() => np.evaluate(NDExpr.Sum(NDExpr.Equal(u, "a")), dtype: np.float64))
                .Should().Throw<NotSupportedException>().WithMessage("np.evaluate: dtype= and order= are not supported on a reduction tree*");
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
