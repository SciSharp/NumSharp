using System;
using System.Collections;
using System.Linq;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The NumPy surface over NEP 55 <c>StringDType</c> operands (strings phase 1, P1.7f): the functions that answer text
    ///     through refusals, parses and promotions give NumPy 2.4.2's answers for a <c>StringDType</c> operand too — the
    ///     dispatcher's class-named refusals, the legacy products' descriptor lookup and OBJECT fallback, the casts of an
    ///     <c>out=</c> descriptor, the order of NumPy's checks. Every expected value was probed against NumPy 2.4.2; the oracle
    ///     section <c>tsurface</c> replays the whole surface, these pin the individual rules.
    /// </summary>
    [TestClass]
    public class StringDTypeSurfaceTests
    {
        /// <summary>The <c>StringDType()</c> descriptor (no NA object, coercing).</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A 1-D <c>StringDType</c> array of the given texts.</summary>
        /// <param name="values">The texts.</param>
        /// <returns>The array.</returns>
        private static NDArray Ts(params string[] values) => np.array(values, T);

        /// <summary><c>['ab', 'c', 'de']</c> as <c>StringDType</c>.</summary>
        private static NDArray T3 => Ts("ab", "c", "de");

        /// <summary><c>[['1', '2'], ['3', '4']]</c> as <c>StringDType</c>.</summary>
        private static NDArray T22 => Ts("1", "2", "3", "4").reshape(2, 2);

        /// <summary>The 0-d <c>StringDType</c> array <c>'x'</c>.</summary>
        private static NDArray T0 => Ts("x").reshape(Array.Empty<long>());

        /// <summary><c>[1.5, 2.0, 3.0]</c> (float64).</summary>
        private static NDArray F3 => np.array(new[] { 1.5, 2.0, 3.0 });

        /// <summary>NumPy's refusal of a <c>StringDType</c> operand by its legacy products (two lines, its typo included).</summary>
        private const string LegacyOnly =
            "This function currently only supports native NumPy dtypes and old-style user dtypes, but the dtype was " +
            "StringDType().\n(The function may need to be updated to support arbitraryuser dtypes.)";

        /// <summary>The dispatcher's refusal text for a ufunc and its operand classes.</summary>
        /// <param name="ufunc">The ufunc name.</param>
        /// <param name="classes">The DType class names (<c>StringDType</c>, <c>Float64DType</c>, <c>_PyLongDType</c>, ...).</param>
        /// <returns>The text: one class prints bare, several as a tuple; the output is <c>None</c>.</returns>
        private static string NoLoop(string ufunc, params string[] classes)
        {
            var reprs = classes.Select(c => $"<class 'numpy.dtypes.{c}'>").ToArray();
            string ins = reprs.Length == 1 ? reprs[0] : "(" + string.Join(", ", reprs) + ")";
            return $"ufunc '{ufunc}' did not contain a loop with signature matching types {ins} -> None";
        }

        /// <summary>The elements of a text array in C order (NumPy's <c>tolist()</c>).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The texts.</returns>
        private static string[] Texts(NDArray a)
        {
            object list = a.ndim == 0 ? a.tolist() : a.flatten().tolist();
            return list is IList items ? items.Cast<object>().Select(x => (string)x).ToArray() : new[] { (string)list };
        }

        // ------------------------------------------------------------------------------------------------ products

        [TestMethod]
        public void Vdot_StringDTypeFirst_IsTheLegacyOnlyRefusal()
        {
            new Action(() => np.vdot(T3, F3)).Should().Throw<TypeError>().WithMessage(LegacyOnly);
        }

        [TestMethod]
        public void Vdot_NumberBeforeStringDType_IsAnObjectProduct()
        {
            // promote(number, StringDType) fails, and NumPy's legacy discovery goes on with OBJECT arrays: the first product
            // conj(a[0]) * b[0] decides by the Python types.
            new Action(() => np.vdot(F3, T3)).Should().Throw<TypeError>()
                .WithMessage("can't multiply sequence by non-int of type 'float'");
            new Action(() => np.vdot(np.array(new[] { "a", "b", "c" }).astype(np.dtype("S1")), T3))
                .Should().Throw<AttributeError>().WithMessage("'bytes' object has no attribute 'conjugate'");
            // The flattening reshapes come first: b is reshaped to a's length.
            new Action(() => np.vdot(np.array(new[] { 1.0, 2.0 }), T3)).Should().Throw<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 3 into shape (2,)");
            // An integer repeats the text: NumPy returns a Python str ('abde' here), which NumSharp cannot hold.
            new Action(() => np.vdot(np.array(new[] { true, false, true }), T3)).Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void Kron_StringDTypeByIntegers_Repeats_ByOtherNumbers_Refuses()
        {
            Texts(np.kron(T3, np.array(new long[] { 1, 2, 3 })))
                .Should().Equal("ab", "abab", "ababab", "c", "cc", "ccc", "de", "dede", "dedede");
            var square = np.kron(T22, np.array(new long[] { 1, 0, 2, 1 }).reshape(2, 2));
            square.shape.Should().Equal(4, 4);
            Texts(square).Should().Equal("1", "", "2", "", "11", "1", "22", "2", "3", "", "4", "", "33", "3", "44", "4");
            new Action(() => np.kron(T3, F3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("multiply", "StringDType", "Float64DType"));
        }

        [TestMethod]
        public void Cross_StringDType_PromotesFirst()
        {
            new Action(() => np.cross(T3, F3)).Should().Throw<DTypePromotionError>().WithMessage(
                "The DTypes <class 'numpy.dtypes.StringDType'> and <class 'numpy.dtypes.Float64DType'> do not have a common " +
                "DType. For example they cannot be stored in a single array unless the dtype is `object`.");
            new Action(() => np.cross(T3, T3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("multiply", "StringDType", "StringDType"));
        }

        [TestMethod]
        public void Gufuncs_NameTheOperandClasses()
        {
            new Action(() => np.matmul(T22, T22)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("matmul", "StringDType", "StringDType"));
            new Action(() => np.vecdot(T3, F3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("vecdot", "StringDType", "Float64DType"));
        }

        [TestMethod]
        public void Einsum_StringDType_IsRefused()
        {
            new Action(() => np.einsum("i,i", T3, T3)).Should().Throw<TypeError>().WithMessage("invalid data type for einsum");
            new Action(() => np.einsum("i,i", new[] { T3, T3 }, dtype: np.dtype("S"))).Should().Throw<TypeError>()
                .WithMessage("cannot cast dtype StringDType() to <class 'numpy.dtypes.BytesDType'>.");
        }

        // ---------------------------------------------------------------------------------------------- ufunc refusals

        [TestMethod]
        public void TwoOutputAndUnaryUfuncs_NameTheClass()
        {
            new Action(() => np.divmod(T3, T3)).Should().Throw<TypeError>().WithMessage(
                "ufunc 'divmod' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, " +
                "<class 'numpy.dtypes.StringDType'>) -> (None, None)");
            new Action(() => np.modf(T3)).Should().Throw<TypeError>().WithMessage(
                "ufunc 'modf' did not contain a loop with signature matching types <class 'numpy.dtypes.StringDType'> -> (None, None)");
            new Action(() => np.invert(T3)).Should().Throw<TypeError>().WithMessage(NoLoop("invert", "StringDType"));
        }

        [TestMethod]
        public void StringDTypeOut_OfAFixedWidthLoop_IsCastByDescriptor()
        {
            var s6 = np.array(new[] { "ab", "c", "de", "f", "gh", "i" }).astype(np.dtype("S3"));
            // resolve_descriptors casts the out= descriptor to the loop's output DType: BytesDType cannot be sized from it.
            new Action(() => np.add(s6, s6, @out: np.zeros(new Shape(6), T))).Should().Throw<TypeError>()
                .WithMessage("cannot cast dtype StringDType() to <class 'numpy.dtypes.BytesDType'>.");
            // A bool loop's output takes it: the formatted bools are written.
            var flipped = s6["::-1"];
            Texts(np.less(s6, flipped, @out: np.zeros(new Shape(6), T)))
                .Should().Equal("True", "True", "True", "False", "False", "False");
            new Action(() => np.add(np.arange(6.0), np.arange(6.0), @out: np.zeros(new Shape(6), T), dtype: np.dtype("U")))
                .Should().Throw<TypeError>().WithMessage("cannot cast dtype StringDType() to <class 'numpy.dtypes.StrDType'>.");
        }

        // ---------------------------------------------------------------------------------------------- linear algebra

        [TestMethod]
        public void Linalg_StringDType()
        {
            new Action(() => np.linalg.pinv(T22)).Should().Throw<TypeError>().WithMessage("cannot conjugate non-numeric dtype");
            new Action(() => np.linalg.eigvals(T22)).Should().Throw<TypeError>().WithMessage(NoLoop("isfinite", "StringDType"));
            // The gufunc's input cast (the house ArgumentException of NumPy's UFuncTypeError, its text verbatim).
            new Action(() => np.linalg.cholesky(T22)).Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'cholesky_lo' input from StringDType() to dtype('float64') with casting rule 'same_kind'");
            // norm parses text into float64 (x.astype(float)), then the norm of the numbers.
            np.linalg.norm(T22).GetDouble(0).Should().Be(5.477225575051661);
            new Action(() => np.linalg.norm(T3)).Should().Throw<ValueError>().WithMessage("could not convert string to float: 'ab'");
            // matrix_rank below rank 2 is int(not all(A == 0)): StringDType == 0 is all False.
            np.linalg.matrix_rank(T3).GetInt64(0).Should().Be(1);
            np.linalg.matrix_rank(Ts()).GetInt64(0).Should().Be(0);
        }

        // ---------------------------------------------------------------------------------------------- math functions

        [TestMethod]
        public void Calculus_StringDType_FollowsNumPysOrder()
        {
            new Action(() => np.gradient(T3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("subtract", "StringDType", "StringDType"));
            // diff's rank check comes before its subtract.
            new Action(() => np.unwrap(T0)).Should().Throw<ValueError>()
                .WithMessage("diff requires input that is at least one dimensional");
            new Action(() => np.unwrap(T3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("subtract", "StringDType", "StringDType"));
            // trapezoid: the Python float dx times the concatenation; slice1[axis] of a 0-d y fails first.
            new Action(() => np.trapezoid(T3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("multiply", "_PyFloatDType", "StringDType"));
            new Action(() => np.trapezoid(T3, F3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("multiply", "Float64DType", "StringDType"));
            new Action(() => np.trapezoid(F3, T3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("subtract", "StringDType", "StringDType"));
            new Action(() => np.trapezoid(T0)).Should().Throw<IndexError>().WithMessage("list assignment index out of range");
            Texts(np.trim_zeros(Ts("", "a", "", "b", ""))).Should().Equal("a", "", "b");
        }

        [TestMethod]
        public void Transcendentals_StringDType()
        {
            new Action(() => np.sinc(T3)).Should().Throw<TypeError>().WithMessage(NoLoop("multiply", "_PyFloatDType", "StringDType"));
            var i0 = np.i0(Ts("1", "2", "4"));
            i0.GetDouble(0).Should().Be(1.2660658777520082);
            i0.GetDouble(2).Should().Be(11.30192195213633);
            new Action(() => np.i0(T3)).Should().Throw<ValueError>().WithMessage("could not convert string to float: 'ab'");
            new Action(() => np.emath.sqrt(T3)).Should().Throw<TypeError>().WithMessage(NoLoop("less", "StringDType", "_PyLongDType"));
            new Action(() => np.emath.arccos(T3)).Should().Throw<TypeError>().WithMessage(NoLoop("absolute", "StringDType"));
            new Action(() => np.interp(T3, F3, F3)).Should().Throw<InvalidCastException>().WithMessage(
                "Cannot cast array data from StringDType() to dtype('float64') according to the rule 'safe'");
            new Action(() => np.interp(T0, F3, F3)).Should().Throw<InvalidCastException>().WithMessage(
                "Cannot cast scalar from StringDType() to dtype('float64') according to the rule 'safe'");
        }

        // ---------------------------------------------------------------------------------------------- polynomials

        [TestMethod]
        public void Polynomials_StringDType()
        {
            // A 1-D StringDType p iterates as Python strs: the add meets the legacy resolver with the str's <U descriptor;
            // an N-D p yields StringDType rows, named by class.
            new Action(() => np.polyval(T3, F3)).Should().Throw<TypeError>().WithMessage(
                "ufunc 'add' did not contain a loop with signature matching types (dtype('float64'), dtype('<U2')) -> None");
            new Action(() => np.polyval(T22, F3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("add", "Float64DType", "StringDType"));
            new Action(() => np.polyval(F3, T3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("multiply", "StringDType", "StringDType"));
            // poly1d of all-empty text is the text '0' (NumPy's array([0], dtype=...)); polymul then convolves.
            using (var p = new poly1d(Ts("", "")))
                Texts(p.coeffs).Should().Equal("0");
            new Action(() => np.polymul(T3, F3)).Should().Throw<TypeError>().WithMessage(LegacyOnly);
            new Action(() => np.polymul(Ts(), F3)).Should().Throw<TypeError>()
                .WithMessage("can't multiply sequence by non-int of type 'float'");
            new Action(() => np.polyint(T3)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("divide", "StringDType", "Int64DType"));
            // m == 0 returns p before any division; len() of a 0-d p refuses before it (a number too).
            Texts(np.polyint(T3, 0)).Should().Equal("ab", "c", "de");
            new Action(() => np.polyint(T0)).Should().Throw<TypeError>().WithMessage("len() of unsized object");
            new Action(() => np.polyint(np.array(5.0))).Should().Throw<TypeError>().WithMessage("len() of unsized object");
        }

        // ---------------------------------------------------------------------------------------------- promotion

        [TestMethod]
        public void ResultType_PromotesTheDtypeArgumentsFirst()
        {
            const string tail = "This means that no common DType exists for the given inputs. For example they cannot be stored " +
                                "in a single array unless the dtype is `object`. The full list of DTypes is: ";
            // np.result_type(t, np.float64): the dtype argument is promoted before the array, in either call order.
            new Action(() => np.result_type(T3, np.float64)).Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.Float64DType'> could not be promoted by <class 'numpy.dtypes.StringDType'>. " + tail +
                "(<class 'numpy.dtypes.Float64DType'>, <class 'numpy.dtypes.StringDType'>)");
            // Two arrays keep their order.
            new Action(() => np.result_type(T3, F3)).Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes.Float64DType'>. " + tail +
                "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.Float64DType'>)");
            // np.cov's result_type(m, np.float64) names the same.
            new Action(() => np.cov(T3)).Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.Float64DType'> could not be promoted by <class 'numpy.dtypes.StringDType'>. " + tail +
                "(<class 'numpy.dtypes.Float64DType'>, <class 'numpy.dtypes.StringDType'>)");
        }

        [TestMethod]
        public void Cov_StringDType_WithRequests()
        {
            new Action(() => np.cov(T3, dtype: np.dtype("U"))).Should().Throw<TypeError>()
                .WithMessage("cannot cast dtype StringDType() to <class 'numpy.dtypes.StrDType'>.");
            new Action(() => np.cov(T3, dtype: np.dtype("U3"))).Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U3'), dtype('<U3'), dtype('<U6'))");
            np.cov(Ts("1", "2", "4"), dtype: np.float64).GetDouble(0).Should().Be(2.333333333333333);
            np.cov(np.zeros(new Shape(0, 2), T), dtype: np.float64).shape.Should().Equal(0, 0);
        }

        /// <summary>
        ///     <c>np.cov(..., dtype=StringDType())</c> is a text request like <c>U</c> and <c>S</c>: <c>X = array(m, ndmin=2,
        ///     dtype=dtype)</c> makes StringDType text of any operand (numbers formatted), and the weighted or plain mean
        ///     refuses it — <c>a.mean</c>'s add.reduce concatenates, so its division by the intp count has no loop (no
        ///     observations: add.reduce has no identity); with weights, <c>result_type(a.dtype, w.dtype)</c> has no common
        ///     DType. A text operand used to recurse (the request was read as numeric, its parse gave text again) until the
        ///     stack overflowed and ended the process; a numeric one raised NumSharp's storage-lane NotSupportedException,
        ///     even for no variables, where NumPy returns its empty (0, 0) float64. Probed against <c>numpy==2.4.2</c>.
        /// </summary>
        [TestMethod]
        public void Cov_StringDTypeRequest_RunsTheTextStatements()
        {
            using var numbers = np.array(new[] { 3.0, -1.0, 2.0, 0.0, 5.0, -4.0, 1.0, 7.0 }).reshape(2, 4);
            string divide = NoLoop("divide", "StringDType", "Int64DType");
            new Action(() => np.cov(numbers, dtype: T)).Should().Throw<TypeError>().WithMessage(divide);
            new Action(() => np.cov(numbers.astype(np.dtype("U5")), dtype: T)).Should().Throw<TypeError>().WithMessage(divide);
            new Action(() => np.cov(Ts("3", "-1", "2", "0").reshape(2, 2), dtype: T)).Should().Throw<TypeError>()
                .WithMessage(divide);
            new Action(() => np.cov(numbers, numbers, dtype: T)).Should().Throw<TypeError>().WithMessage(divide);
            new Action(() => np.corrcoef(numbers, dtype: T)).Should().Throw<TypeError>().WithMessage(divide);
            // No observations: add.reduce of StringDType has no identity.
            new Action(() => np.cov(np.zeros(new Shape(3, 0)), dtype: T)).Should().Throw<ValueError>()
                .WithMessage("zero-size array to reduction operation add which has no identity");
            // Weights: the weighted mean's promotion of the text with the float64 weights.
            new Action(() => np.cov(numbers, aweights: np.ones(new Shape(4)), dtype: T)).Should().Throw<DTypePromotionError>()
                .WithMessage("The DType <class 'numpy.dtypes.StringDType'> could not be promoted by " +
                             "<class 'numpy.dtypes.Float64DType'>. This means that no common DType exists for the given inputs. " +
                             "For example they cannot be stored in a single array unless the dtype is `object`. The full list " +
                             "of DTypes is: (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.Float64DType'>)");
            // The weight checks come first.
            new Action(() => np.cov(numbers, fweights: np.array(new[] { 1.5, 1.0, 1.0, 1.0 }), dtype: T))
                .Should().Throw<TypeError>().WithMessage("fweights must be integer");
            // No variables: NumPy's empty (0, 0) float64, a reshape view.
            using (var none = np.cov(np.zeros(new Shape(0, 3)), dtype: T))
            {
                none.shape.Should().Equal(0, 0);
                none.dtype.Should().Be(np.float64);
            }
        }

        [TestMethod]
        public void Average_ChecksTheWeightsShapesBeforePromoting()
        {
            new Action(() => np.average(T22, (int?)null, np.array(new[] { 1.0, 2.0 }))).Should().Throw<TypeError>()
                .WithMessage("Axis must be specified when shapes of a and weights differ.");
            new Action(() => np.average(T22, 1, np.array(new[] { 1.0, 2.0, 3.0 }))).Should().Throw<ValueError>()
                .WithMessage("Shape of weights must be consistent with shape of a along specified axis.");
        }

        // ---------------------------------------------------------------------------------------------- copyto

        [TestMethod]
        public void Copyto_FromStringDType_ConvertsOnlyWhatItCopies()
        {
            using (var empty = np.zeros(new Shape(0)))
                np.copyto(empty, Ts("x"), "unsafe");
            new Action(() => np.copyto(np.zeros(new Shape(3)), Ts("x", "y"), "unsafe")).Should().Throw<ValueError>()
                .WithMessage("could not broadcast input array from shape (2,) into shape (3,)");
            using (var none = np.zeros(new Shape(3)))
            {
                np.copyto(none, Ts("x"), "unsafe", np.array(new[] { false, false, false }));
                none.GetDouble(0).Should().Be(0.0);
            }
            // A 0-d source into one element converts only for a selected element; into several it converts up front.
            using (var one = np.zeros(new Shape(1)))
                np.copyto(one, T0, "unsafe", np.array(new[] { false }));
            new Action(() => np.copyto(np.zeros(new Shape(2)), T0, "unsafe", np.array(new[] { false, false })))
                .Should().Throw<ValueError>().WithMessage("could not convert string to float: 'x'");
            // The unselected 'x' is never parsed.
            using (var some = np.zeros(new Shape(3)))
            {
                np.copyto(some, Ts("1", "x", "3"), "unsafe", np.array(new[] { true, false, true }));
                some.GetDouble(0).Should().Be(1.0);
                some.GetDouble(1).Should().Be(0.0);
                some.GetDouble(2).Should().Be(3.0);
            }
            new Action(() => np.copyto(np.zeros(new Shape(3)), Ts("1", "x", "3"), "unsafe", np.array(new[] { true, true, true })))
                .Should().Throw<ValueError>().WithMessage("could not convert string to float: 'x'");
            new Action(() => np.copyto(np.zeros(new Shape(0)), T0, "unsafe", np.array(new[] { false, false })))
                .Should().Throw<ValueError>().WithMessage("could not broadcast where mask from shape (2,) into shape (0,)");
            // A broadcast source under a 2-D mask: only the selected row's selected elements are read.
            using (var grid = np.zeros(new Shape(2, 3)))
            {
                var mask = np.array(new[] { true, false, true, false, false, false }).reshape(2, 3);
                np.copyto(grid, Ts("1", "x", "3"), "unsafe", mask);
                grid.GetDouble(0, 2).Should().Be(3.0);
                grid.GetDouble(1, 0).Should().Be(0.0);
            }
        }

        // ---------------------------------------------------------------------------------------------- elements as Python strs

        [TestMethod]
        public void StringDTypeElements_AreReportedAsPythonStrs()
        {
            // NumPy hands out a StringDType element as a str (a U element is an np.str_).
            new Action(() => np.insert(F3, Ts("1"), NDArray.Scalar(5.0))).Should().Throw<TypeError>()
                .WithMessage("'<' not supported between instances of 'str' and 'int'");
            new Action(() => np.insert(F3, Ts("1", "2", "4"), NDArray.Scalar(5.0))).Should().Throw<TypeError>()
                .WithMessage(NoLoop("less", "StringDType", "_PyLongDType"));
            new Action(() => np.histogram2d(F3, F3, Ts("1", "2"))).Should().Throw<TypeError>()
                .WithMessage("'<' not supported between instances of 'str' and 'int'");
            new Action(() => np.random.SeedSequence(Ts("1a"))).Should().Throw<ValueError>()
                .WithMessage("invalid literal for int() with base 10: '1a'");
        }

        // ---------------------------------------------------------------------------------------------- text values packed into numbers

        /// <summary>
        ///     <c>initial=</c> and <c>ndarray.fill</c> pack a 0-d text array into the loop / array dtype the way NumPy's
        ///     <c>PyArray_Pack</c> does: the descriptor cast, which PARSES the text (int() for an integer, float() for a float,
        ///     the truth of a non-empty string for bool), with the parse's own error texts.
        /// </summary>
        [TestMethod]
        public void InitialAndFill_ParseATextArray()
        {
            np.max(np.zeros(3), initial: np.array("4")).GetDouble().Should().Be(4.0);
            new Action(() => np.max(np.zeros(3), initial: np.array("x"))).Should().Throw<ValueError>()
                .WithMessage("could not convert string to float: np.str_('x')");
            np.max(np.zeros(3, np.int64), initial: Ts("4").reshape(Array.Empty<long>())).GetInt64().Should().Be(4);
            new Action(() => np.max(np.zeros(3, np.int64), initial: Ts("4.5").reshape(Array.Empty<long>()))).Should()
                .Throw<ValueError>().WithMessage("invalid literal for int() with base 10: '4.5'");
            np.max(np.zeros(3, np.@bool), initial: Ts("x").reshape(Array.Empty<long>())).GetBoolean().Should().BeTrue();
            np.max(np.zeros(3, np.@bool), initial: Ts("").reshape(Array.Empty<long>())).GetBoolean().Should().BeFalse();
            new Action(() => np.max(np.zeros(3, np.int8), initial: Ts("300").reshape(Array.Empty<long>()))).Should()
                .Throw<OverflowException>().WithMessage("Integer 300 is out of bounds for byte");

            using var f3 = np.zeros(3);
            f3.fill(np.array("4"));
            f3.ToArray<double>().Should().Equal(4.0, 4.0, 4.0);
            using var i3 = np.zeros(3, np.int64);
            i3.fill(Ts("4").reshape(Array.Empty<long>()));
            i3.ToArray<long>().Should().Equal(4L, 4L, 4L);
            new Action(() => i3.fill(Ts("x").reshape(Array.Empty<long>()))).Should().Throw<ValueError>()
                .WithMessage("invalid literal for int() with base 10: 'x'");
        }

        /// <summary>
        ///     <c>np.full</c> / <c>np.full_like</c> are NumPy's <c>empty</c> + <c>copyto(casting='unsafe')</c>: an ARRAY fill
        ///     value broadcasts into the result (it is not a scalar to repeat), text parsed into a numeric result and numbers
        ///     formatted into a text one; a 0-d text value is one parsed value; a fill that does not broadcast is copyto's error.
        /// </summary>
        [TestMethod]
        public void FullAndFullLike_BroadcastAnArrayFill()
        {
            np.full_like(np.zeros(3), new[] { 1, 2, 3 }).ToArray<double>().Should().Equal(1.0, 2.0, 3.0);
            np.full_like(np.zeros(3), Ts("1", "2", "3")).ToArray<double>().Should().Equal(1.0, 2.0, 3.0);
            np.full_like(np.zeros(3), Ts("4").reshape(Array.Empty<long>())).ToArray<double>().Should().Equal(4.0, 4.0, 4.0);
            new Action(() => np.full_like(np.zeros(3), Ts("x").reshape(Array.Empty<long>()))).Should().Throw<ValueError>()
                .WithMessage("could not convert string to float: 'x'");
            np.full_like(np.zeros(3, np.int64), Ts("1", "2", "3")).ToArray<long>().Should().Equal(1L, 2L, 3L);
            using (var grid = np.full_like(np.zeros(new Shape(2, 3)), new[] { 1, 2, 3 }))
            {
                grid.shape.Should().Equal(2L, 3L);
                grid.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 1.0, 2.0, 3.0);
            }
            new Action(() => np.full_like(np.zeros(3), new[] { 1, 2 })).Should().Throw<ValueError>()
                .WithMessage("could not broadcast input array from shape (2,) into shape (3,)");
            Texts(np.full_like(Ts("a", "b", "c"), new[] { 1, 2, 3 })).Should().Equal("1", "2", "3");
            Texts(np.full_like(Ts("a", "b", "c"), np.array(new[] { 1.5, 2, 3 }))).Should().Equal("1.5", "2.0", "3.0");
            Texts(np.full(new Shape(3), Ts("1", "2", "3"))).Should().Equal("1", "2", "3");
            np.full(new Shape(3), Ts("1", "2", "3"), np.float64).ToArray<double>().Should().Equal(1.0, 2.0, 3.0);
            np.full(new Shape(3), Ts("4").reshape(Array.Empty<long>()), np.float64).ToArray<double>().Should().Equal(4.0, 4.0, 4.0);
            new Action(() => np.full(new Shape(3), Ts("x").reshape(Array.Empty<long>()), np.float64)).Should()
                .Throw<ValueError>().WithMessage("could not convert string to float: 'x'");
        }

        // ---------------------------------------------------------------------------------------------- quantile with text

        /// <summary>
        ///     The quantile family over an empty q and text: NumPy's statements reach the text at different points —
        ///     quantile's lerp multiplies by an empty U q (the string multiply needs out=), a <c>StringDType</c> q meets
        ///     <c>floor</c>; nanquantile of <c>StringDType</c> data counts its NaNs (the empty-slice sum has no identity), of
        ///     U data asks <c>isnan</c>, which U has no loop for.
        /// </summary>
        [TestMethod]
        public void Quantile_EmptyTextQ_FollowsNumPysStatements()
        {
            using var qe = np.array(Array.Empty<string>()).astype(np.dtype("U1"));
            using var qeT = Ts();
            const string OutKwarg = "The 'out' kwarg is necessary when using the string multiply ufunc directly. Use numpy.strings.multiply to multiply strings without specifying 'out'.";
            new Action(() => np.quantile(Ts("a", "b"), qe)).Should().Throw<TypeError>().WithMessage(OutKwarg);
            new Action(() => np.nanquantile(Ts("a", "b"), qe)).Should().Throw<TypeError>().WithMessage(OutKwarg);
            new Action(() => np.quantile(Ts("a", "b"), qeT)).Should().Throw<TypeError>().WithMessage(NoLoop("floor", "StringDType"));
            new Action(() => np.nanquantile(T22, qeT, 0)).Should().Throw<TypeError>().WithMessage(NoLoop("floor", "StringDType"));
            new Action(() => np.nanquantile(Ts(), qe)).Should().Throw<ValueError>()
                .WithMessage("zero-size array to reduction operation add which has no identity");
            new Action(() => np.nanquantile(np.array(new[] { "a", "b" }), qe)).Should().Throw<TypeError>().WithMessage(
                "ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''");
            new Action(() => np.quantile(np.array(new[] { 1.0, 2.0 }), qeT)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("floor", "StringDType"));
            np.nanquantile(np.array(Array.Empty<double>()), qe).GetDouble().Should().Be(double.NaN);
        }

        // ---------------------------------------------------------------------------------------------- np.ndarray over StringDType cells

        /// <summary>
        ///     <c>np.ndarray(shape, StringDType, buffer=a)</c> lays a view over another <c>StringDType</c> array's 16-byte cells
        ///     (NumPy's buffer export of a StringDType array is its raw cells): byte offsets and strides in cells, write-through,
        ///     another StringDType descriptor reading the same cells; without a buffer the array starts with empty strings.
        ///     The documented divergences refuse what NumPy can only read as garbage: a non-StringDType dtype over the cells, a
        ///     StringDType over raw bytes, an offset that splits a cell.
        /// </summary>
        [TestMethod]
        public void NdarrayOverStringDTypeCells()
        {
            var t3 = T3;
            using (var view = np.ndarray(new Shape(3), T, t3))
            {
                Texts(view).Should().Equal("ab", "c", "de");
                view.flags.owndata.Should().BeFalse();
                view[0] = Ts("zz").reshape(Array.Empty<long>());
                Texts(t3).Should().Equal("zz", "c", "de");
            }
            Texts(np.ndarray(new Shape(2), T, t3, offset: 16)).Should().Equal("c", "de");
            Texts(np.ndarray(new Shape(2), T, t3, strides: new long[] { 32 })).Should().Equal("zz", "de");
            Texts(np.ndarray(new Shape(3), T, t3, offset: 32, strides: new long[] { -16 })).Should().Equal("de", "c", "zz");
            Texts(np.ndarray(new Shape(3), T, t3, strides: new long[] { 0 })).Should().Equal("zz", "zz", "zz");
            np.ndarray(new Shape(0), T, t3, offset: 8).size.Should().Be(0);
            Texts(np.ndarray(new Shape(-1), T, t3)).Should().Equal("zz", "c", "de");
            Texts(np.ndarray(new Shape(3), np.dtypes.StringDType(na_object: null), t3)).Should().Equal("zz", "c", "de");

            Texts(np.ndarray(new Shape(3), T)).Should().Equal("", "", "");
            Texts(np.ndarray(new Shape(2, 3), T, order: 'F')).Should().Equal("", "", "", "", "", "");
            new Action(() => np.ndarray(new Shape(2), T, strides: new long[] { 32 })).Should().Throw<ValueError>()
                .WithMessage("strides is incompatible with shape of requested array and size of buffer");

            new Action(() => np.ndarray(new Shape(3), np.float64, t3)).Should().Throw<NotSupportedException>();
            new Action(() => np.ndarray(new Shape(1), T, new byte[16])).Should().Throw<NotSupportedException>();
            new Action(() => np.ndarray(new Shape(1), T, t3, offset: 8)).Should().Throw<NotSupportedException>();
        }

        // ---------------------------------------------------------------------------------------------- fields and fancy indexing

        /// <summary>
        ///     <c>getfield</c> / <c>setfield</c> over a <c>StringDType</c> array follow NumPy's <c>_getfield_is_safe</c>: cells
        ///     holding references may only be read as the whole element in their own descriptor (a view), anything else is
        ///     <c>Cannot get/set field of an object array</c>.
        /// </summary>
        [TestMethod]
        public void GetfieldSetfield_OfStringDType_AreTheWholeElementOnly()
        {
            var t3 = T3;
            using (var same = t3.getfield(t3.dtype))
                Texts(same).Should().Equal("ab", "c", "de");
            new Action(() => t3.getfield(np.float64)).Should().Throw<TypeError>().WithMessage("Cannot get/set field of an object array");
            new Action(() => t3.getfield(t3.dtype, 8)).Should().Throw<TypeError>().WithMessage("Cannot get/set field of an object array");
            new Action(() => np.zeros(3).getfield(T)).Should().Throw<TypeError>().WithMessage("Cannot get/set field of an object array");
            t3.setfield(Ts("q", "r", "s"), t3.dtype);
            Texts(t3).Should().Equal("q", "r", "s");
            new Action(() => t3.setfield(1.0, np.float64)).Should().Throw<TypeError>().WithMessage("Cannot get/set field of an object array");
        }

        /// <summary>
        ///     Fancy indexing of a <c>StringDType</c> array — a gather into a given out=, a scatter, the typed index-array
        ///     indexer — moves cells through the StringDType copy (each value re-packed into the destination's arena).
        /// </summary>
        [TestMethod]
        public void FancyIndexing_OfStringDType_MovesTheTexts()
        {
            var t3 = T3;
            using (var idx = np.array(new long[] { 2, 0 }))
            using (var gathered = t3[idx])
                Texts(gathered).Should().Equal("de", "ab");
            using (var idx = np.array(new long[] { 1 }))
            {
                t3[idx] = Ts("long text beyond the short-string size");
                Texts(t3).Should().Equal("ab", "long text beyond the short-string size", "de");
            }
            var t22 = T22;
            using (var rows = np.array(new long[] { 1, 1 }))
            using (var cols = np.array(new long[] { 0, 1 }))
                Texts(t22[rows, cols]).Should().Equal("3", "4");
        }

        /// <summary>
        ///     A Python scalar copied into text: a number into a <c>StringDType</c> array is its str() per selected element
        ///     (StringDType holds references, so NumPy converts per element); a Python int past uint64 is an object scalar,
        ///     refused under every rule but 'unsafe', which stores its str().
        /// </summary>
        [TestMethod]
        public void Copyto_PythonScalarIntoText()
        {
            var t3 = T3;
            np.copyto(t3, 7);
            Texts(t3).Should().Equal("7", "7", "7");
            var t3b = T3;
            np.copyto(t3b, 2.5, @where: np.array(new[] { true, false, true }));
            Texts(t3b).Should().Equal("2.5", "c", "2.5");
            var big = System.Numerics.BigInteger.Pow(2, 70);
            new Action(() => np.copyto(T3, big)).Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast scalar from dtype('O') to StringDType() according to the rule 'same_kind'");
            var t3c = T3;
            np.copyto(t3c, big, casting: "unsafe");
            Texts(t3c).Should().Equal("1180591620717411303424", "1180591620717411303424", "1180591620717411303424");
        }
    }
}
