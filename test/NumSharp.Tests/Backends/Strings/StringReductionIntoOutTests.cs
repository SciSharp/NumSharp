using System;
using NumSharp.Backends;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Reductions, accumulations and statistics into a TEXT <c>out=</c>, and of text into any <c>out=</c> (strings phase 2,
    ///     P2.7): NumPy resolves the reduction's loop on <c>(out, input)</c>, so the out decides whether a text loop computes —
    ///     the <c>StringDType</c> concatenation of a <c>U</c> input into a <c>StringDType</c> out, the string repeat of
    ///     integer counts into a text out — or which refusal NumPy raises. Every expected value and text below was probed
    ///     against NumPy 2.4.2; the string oracle replays the full matrix, these tests pin one representative of each rule.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The statistics compute part of their answer before they refuse: <c>np.median</c> sums the middle elements INTO the
    ///     text out, and <c>np.percentile</c>'s lerp adds into it, before the division / the subtract has no text loop — so
    ///     the out holds that partial result when the exception is thrown, as NumPy's does.
    ///     </para>
    ///     <para>
    ///     Exception types follow the house map: NumPy's <c>UFuncTypeError</c> / <c>TypeError</c> is <see cref="TypeError"/>
    ///     (the legacy resolver's <c>not supported for the input types</c> text included), a broadcast <c>ValueError</c> is
    ///     <see cref="IncorrectShapeException"/>.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class StringReductionIntoOutTests
    {
        private static DType T => np.dtypes.StringDType();

        private static NDArray U(params string[] s) => np.array(s);

        private static NDArray TArr(params string[] s) => np.array(s).astype(T);

        private static NDArray I(params long[] v) => np.array(v);

        /// <summary>The elements of a text array in C order, each as the <see cref="string"/> NumSharp reads it.</summary>
        /// <param name="a">A <c>U</c> or <c>StringDType</c> array.</param>
        /// <returns>The texts.</returns>
        private static string[] Texts(NDArray a)
        {
            var texts = new string[a.size];
            for (long i = 0; i < a.size; i++)
                texts[i] = (string)a.GetAtIndex(i);
            return texts;
        }

        // ---------------------------------------------------------------- accumulations

        [TestMethod]
        public void Cumsum_UIntoStringDTypeOut_Concatenates()
        {
            // np.cumsum(np.array(['c', 'a', 'b']), out=T) -> ['c', 'ca', 'cab'] (the StringDType add promoter pairs
            // StringDType with StrDType; the U input is cast to the default StringDType()).
            var o = np.full(new Shape(3), "", T);
            var r = np.cumsum(U("c", "a", "b"), null, null, o);
            ReferenceEquals(r, o).Should().BeTrue();
            Texts(o).Should().Equal("c", "ca", "cab");

            var on = np.full(new Shape(3), "", T);
            np.nancumsum(U("c", "a", "b"), null, null, on);
            Texts(on).Should().Equal("c", "ca", "cab");
        }

        [TestMethod]
        public void Cumsum_UIntoStringDTypeOutWithNA_IsTheUniformDescriptorRefusal()
        {
            // The cast's StringDType() has no NA object, so add.accumulate's enforce_uniform_args refuses an out that has one.
            var o = np.full(new Shape(2), "", np.dtypes.StringDType(na_object: null));
            Action act = () => np.cumsum(U("a", "b"), null, null, o);
            act.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.accumulate. Resolved (StringDType(na_object=None), " +
                "StringDType(), StringDType(na_object=None))");
        }

        [TestMethod]
        public void Cumsum_UIntoStringDTypeOut_TakesTheIteratorPathForItsShapeErrors()
        {
            // The U input's descriptor differs from the loop's, so PyUFunc_Accumulate builds its iterator: a wrong-sized out is
            // the iterator's remapped-shapes error, not the direct path's "provided out is the wrong size".
            var o = np.full(new Shape(3), "", T);
            Action act = () => np.cumsum(U("a", "b"), null, null, o);
            // NumPy's ValueError for a broadcast failure is the house IncorrectShapeException.
            act.Should().Throw<IncorrectShapeException>().WithMessage(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (3,)->(3,) (2,)->(2,) ");
        }

        [TestMethod]
        public void Cumprod_StringDTypeIntoIntegerOut_NamesTwoStringDTypeClasses()
        {
            // The repeat loop (int64, StringDType) -> StringDType is found but is not reduce-compatible, so NumPy fixes the
            // first slot to StringDType and retries (StringDType, StringDType), which has no loop.
            Action act = () => np.cumprod(TArr("ab", "c"), null, null, np.zeros(new Shape(2), np.int8));
            act.Should().Throw<TypeError>().WithMessage(
                "ufunc 'multiply' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, " +
                "<class 'numpy.dtypes.StringDType'>) -> None");
        }

        [TestMethod]
        public void Cumprod_CountsIntoStringDTypeOut_NamesTheCountDescriptor()
        {
            // StringDType registers its repeat loops for int64 and uint64: a uint64 count keeps its own descriptor, every
            // other width is promoted to int64.
            Action u64 = () => np.cumprod(np.array(new ulong[] { 2, 3 }), null, null, np.full(new Shape(2), "x", T));
            u64.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with multiply.accumulate. Resolved (StringDType(), dtype('uint64'), " +
                "StringDType())");
            Action i8 = () => np.cumprod(np.array(new sbyte[] { 2, 3 }), null, null, np.full(new Shape(2), "x", T));
            i8.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with multiply.accumulate. Resolved (StringDType(), dtype('int64'), " +
                "StringDType())");
        }

        // ---------------------------------------------------------------- np.trace

        [TestMethod]
        public void Trace_UIntoStringDTypeOut_Concatenates()
        {
            var a = np.array(new[,] { { "c", "a" }, { "a", "b" } });
            var o = np.array("", T);
            np.trace(a, 0, 0, 1, null, o);
            ((string)o.GetAtIndex(0)).Should().Be("cb");
            // An out with an NA object computes too (a reduction has no uniform-descriptor check).
            var on = np.array("", np.dtypes.StringDType(na_object: null));
            np.trace(np.array(new[,] { { "a", "b" }, { "c", "d" } }), 0, 0, 1, null, on);
            ((string)on.GetAtIndex(0)).Should().Be("ad");
        }

        [TestMethod]
        public void Trace_ResolvesTheLoopOnTheOutAndTheDiagonal()
        {
            var a = np.array(new[,] { { "c", "a" }, { "a", "b" } });
            Action fixedOut = () => np.trace(a, 0, 0, 1, null, np.zeros(Shape.Scalar, "U6"));
            fixedOut.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U6'), dtype('<U1'), dtype('<U7'))");
            Action numbers = () => np.trace(np.eye(2, dtype: np.int64), 0, 0, 1, null, np.zeros(Shape.Scalar, "U6"));
            numbers.Should().Throw<TypeError>().WithMessage(
                "ufunc 'add' did not contain a loop with signature matching types (dtype('<U6'), dtype('int64')) -> None");
        }

        // ---------------------------------------------------------------- numpy.matrix methods

        [TestMethod]
        public void MatrixSum_TextRoutes_ComputeIntoTheOut()
        {
            // matrix.sum(axis, out=o) is add.reduce(self, axis, out=o, keepdims=True).
            var mu = (NDMatrix)np.matrix(U("c", "a", "b").reshape(1, 3));
            var ot = (NDMatrix)np.matrix(np.full(new Shape(1, 1), "", T));
            mu.sum(1, null, ot);
            ((string)ot.GetAtIndex(0)).Should().Be("cab");

            var mt = (NDMatrix)np.matrix(TArr("c", "a", "b").reshape(1, 3));
            var ou = (NDMatrix)np.matrix(np.zeros(new Shape(1, 1), "U6"));
            mt.sum(1, null, ou);
            ((string)ou.GetAtIndex(0)).Should().Be("cab");
        }

        [TestMethod]
        public void MatrixProd_CountsIntoTextOut_RepeatTheFirstCount()
        {
            // np.matrix([[3, 1, 2]]).prod(axis=1, out=U6) is '33': the first count's str() repeated by the later counts.
            var m = (NDMatrix)np.matrix(I(3, 1, 2).reshape(1, 3));
            var ou = (NDMatrix)np.matrix(np.zeros(new Shape(1, 1), "U6"));
            m.prod(1, null, ou);
            ((string)ou.GetAtIndex(0)).Should().Be("33");
            var ot = (NDMatrix)np.matrix(np.full(new Shape(1, 1), "", T));
            m.prod(1, null, ot);
            ((string)ot.GetAtIndex(0)).Should().Be("33");
        }

        [TestMethod]
        public void MatrixMeanVar_FloatIntoTextOut_IsTheAddRefusal()
        {
            // A float matrix keeps its own loop (no implicit dtype), so the sum into a text out has no loop.
            var m = (NDMatrix)np.matrix(np.array(new[] { 3.0, 1.0, 2.0 }).reshape(1, 3));
            const string text = "ufunc 'add' did not contain a loop with signature matching types (dtype('<U6'), dtype('float64')) -> None";
            Action mean = () => m.mean(1, null, (NDMatrix)np.matrix(np.zeros(new Shape(1, 1), "U6")));
            mean.Should().Throw<TypeError>().WithMessage(text);
            Action var = () => m.var(1, null, (NDMatrix)np.matrix(np.zeros(new Shape(1, 1), "U6")));
            var.Should().Throw<TypeError>().WithMessage(text);
        }

        [TestMethod]
        public void MatrixMean_UIntoStringDTypeOut_ConcatenatesThenTheDivisionRefuses()
        {
            var m = (NDMatrix)np.matrix(U("c", "a", "b").reshape(1, 3));
            var o = (NDMatrix)np.matrix(np.full(new Shape(1, 1), "", T));
            Action act = () => m.mean(1, null, o);
            act.Should().Throw<TypeError>().WithMessage(
                "ufunc 'divide' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, " +
                "<class 'numpy.dtypes.Int64DType'>) -> None");
            ((string)o.GetAtIndex(0)).Should().Be("cab");
        }

        // ---------------------------------------------------------------- statistics

        [TestMethod]
        public void Median_IntegersIntoTextOut_SumTheMiddleIntoItThenRefuse()
        {
            // _mean's implicit float64 loop sums the middle (2 + 3) into the out, then true_divide has no text loop.
            var o = np.zeros(Shape.Scalar, "U6");
            Action act = () => np.median(I(4, 1, 3, 2), (int?)null, o);
            act.Should().Throw<TypeError>().WithMessage(
                "ufunc 'divide' not supported for the input types, and the inputs could not be safely coerced to any supported " +
                "types according to the casting rule ''safe''");
            ((string)o.GetAtIndex(0)).Should().Be("5.0");
        }

        [TestMethod]
        public void Median_UIntoStringDTypeOut_ConcatenatesTheMiddleThenRefuses()
        {
            var o = np.array("z", T);
            Action act = () => np.median(U("d", "a", "c", "b"), (int?)null, o);
            act.Should().Throw<TypeError>().WithMessage(
                "ufunc 'divide' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, " +
                "<class 'numpy.dtypes.Int64DType'>) -> None");
            ((string)o.GetAtIndex(0)).Should().Be("bc");
        }

        [TestMethod]
        public void Percentile_NumbersIntoTextOut_IsTheLerpSubtractRefusal()
        {
            // _lerp's add writes the out, then subtract(..., dtype=type(out.dtype)) names the out's class and finds no loop.
            var o = np.zeros(Shape.Scalar, "U6");
            Action act = () => np.percentile(I(3, 1, 2), 50.0, (int?)null, o);
            act.Should().Throw<TypeError>().WithMessage(
                "ufunc 'subtract' did not contain a loop with signature matching types (<class 'numpy.dtypes.Int64DType'>, " +
                "<class 'numpy.dtypes.Float64DType'>) -> <class 'numpy.dtypes.StrDType'>");
            ((string)o.GetAtIndex(0)).Should().Be("2.0");
        }
    }
}
