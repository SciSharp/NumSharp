using System;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.cumsum / np.cumprod refuse every out whose dimensions are not the accumulation's, as NumPy 2.5's
    ///     <c>PyUFunc_Accumulate</c> does (gh-32709): after its iterator accepts the operands — or on its direct 1-D path —
    ///     it compares the out's dimensions with the input's BEFORE its empty-input / empty-out exits, so an out with MORE
    ///     trailing axes, an out an extent-1 input broadcasts against, an empty input or out of other dimensions, a 0-d out
    ///     and an out aliasing the input under another shape all raise <c>provided out is the wrong size for the
    ///     accumulation.</c> and are left untouched. (NumPy 2.4 wrote the cells its iterator walked of such an out and
    ///     returned an empty input's out untouched — parity QA-54, which this file pinned until the 2.5.4 bump.) Every out
    ///     starts filled with -1, so a write shows; every expectation is NumPy 2.5.4's (scratchpad probe_accum_outs.py).
    /// </summary>
    [TestClass]
    public class CumsumCumprodMisfitOutsTests
    {
        /// <summary>NumPy's refusal of an out whose dimensions are not the accumulation's.</summary>
        private const string WrongSize = "provided out is the wrong size for the accumulation.";

        /// <summary>An out of <paramref name="shape"/> and <paramref name="dtype"/> filled with -1.</summary>
        /// <param name="dtype">The out's dtype.</param>
        /// <param name="shape">The out's shape (empty for a 0-d out).</param>
        /// <returns>The out.</returns>
        private static NDArray Full(DType dtype, params long[] shape) => np.full(new Shape(shape, copy: false), -1, dtype);

        /// <summary>The out's elements in C order, as doubles.</summary>
        /// <param name="o">The out.</param>
        /// <returns>The values.</returns>
        private static double[] Vals(NDArray o)
        {
            if (o.size == 0)
                return Array.Empty<double>();
            using var widened = o.astype(np.float64);
            using var flat = widened.flatten();
            return flat.ToArray<double>();
        }

        /// <summary>The accumulation by name.</summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        /// <param name="x">The input.</param>
        /// <param name="axis">The axis (null for the flat accumulation).</param>
        /// <param name="o">The out.</param>
        /// <returns>The call's return (NumPy returns the out itself).</returns>
        private static NDArray Call(string name, NDArray x, int? axis, NDArray o)
            => name == "cumsum" ? np.cumsum(x, axis, @out: o) : np.cumprod(x, axis, @out: o);

        /// <summary>
        ///     Asserts that the accumulation refuses <paramref name="o"/> with NumPy's wrong-size text and leaves every one of
        ///     its cells -1.
        /// </summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        /// <param name="x">The input.</param>
        /// <param name="axis">The axis (null for the flat accumulation).</param>
        /// <param name="o">The out (filled with -1).</param>
        /// <param name="because">The case, for the failure message.</param>
        private static void Refused(string name, NDArray x, int? axis, NDArray o, string because)
        {
            new Action(() => Call(name, x, axis, o)).Should().Throw<IncorrectShapeException>(because).WithMessage(WrongSize);
            foreach (var v in Vals(o))
                v.Should().Be(-1.0, because + ": the refused out is never written");
        }

        /// <summary>
        ///     The iterator path: an out of more dimensions, an extent-1 input broadcast against a longer out, and an input of
        ///     extent 1 along the accumulated axis — all accepted by NumPy's iterator, all refused by the dimension check that
        ///     follows it.
        /// </summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        [TestMethod]
        [DataRow("cumsum")]
        [DataRow("cumprod")]
        public void IteratorPath_OutsOfOtherDimensions_AreTheWrongSize(string name)
        {
            using var a = np.arange(12.0).reshape(3, 4);
            using (var o = Full(np.float64, 3, 4, 1))
                Refused(name, a, 0, o, "an out with a trailing axis");
            using (var o = Full(np.float64, 3, 4, 2))
                Refused(name, a, 1, o, "an out with a longer trailing axis");

            using var ones = np.ones(new Shape(3, 1));
            using var col = ones * 2;
            using (var o = Full(np.float64, 3, 4))
                Refused(name, col, 0, o, "a column broadcast against the out");
            using (var o = Full(np.float64, 3, 4))
                Refused(name, col, 1, o, "a column of extent 1 along the axis");

            using var row = np.arange(1.0, 5.0).reshape(1, 4);
            using (var o = Full(np.float64, 3, 4))
                Refused(name, row, 0, o, "a row of extent 1 along the axis");
            using (var o = Full(np.float64, 3, 4))
                Refused(name, row, 1, o, "a row broadcast against the out");

            using var ints = np.arange(1, 7).astype(np.int32).reshape(2, 3);
            using (var o = Full(np.int64, 2, 3, 1))
                Refused(name, ints, 1, o, "an int64 out with a trailing axis");
        }

        /// <summary>
        ///     One-dimensional inputs: a float32 out of more dimensions (the iterator path NumPy takes for a cast out), an
        ///     extent-1 input into a longer out, the flat accumulation into a 2-D out, and a 0-d out (which crashed NumPy 2.4).
        /// </summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        [TestMethod]
        [DataRow("cumsum")]
        [DataRow("cumprod")]
        public void OneDimensionalInputs_IntoOutsOfOtherDimensions_AreTheWrongSize(string name)
        {
            using var v = np.arange(1.0, 4.0);
            using (var o = Full(np.float32, 3, 1))
                Refused(name, v, 0, o, "a (3, 1) float32 out");
            using var five = np.array(new[] { 5.0 });
            using (var o = Full(np.float32, 3))
                Refused(name, five, 0, o, "a one-element input into a longer out");
            using (var o = Full(np.float32, 3, 2))
                Refused(name, v, null, o, "the flat accumulation into a 2-D out");
            using (var o = Full(np.float64))
                Refused(name, v, 0, o, "a 0-d out");
        }

        /// <summary>
        ///     The dimension check runs before NumPy's empty exits: an empty input or an empty out of other dimensions is the
        ///     wrong size too (NumPy 2.4 returned the out untouched).
        /// </summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        [TestMethod]
        [DataRow("cumsum")]
        [DataRow("cumprod")]
        public void EmptyInputOrOut_OfOtherDimensions_AreTheWrongSize(string name)
        {
            using var empty = np.zeros(new Shape(0));
            using (var o = Full(np.float64, 5))
                Refused(name, empty, null, o, "an empty input into a (5,) out");
            using var three = np.arange(3.0);
            using (var o = Full(np.float64, 0))
                Refused(name, three, 0, o, "a (3,) input into an empty out");
            using var emptyRows = np.zeros(new Shape(0, 4));
            using (var o = Full(np.float64, 0, 4, 2))
                Refused(name, emptyRows, 0, o, "a (0, 4) input into a (0, 4, 2) out");
        }

        /// <summary>
        ///     An out of the accumulation's dimensions is written and returned; one that aliases the input under other
        ///     dimensions (a reshaped view of its memory) is the wrong size, its cells keeping the input's values; and a
        ///     read-only out of other dimensions reports read-only first (NumPy's order).
        /// </summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        [TestMethod]
        [DataRow("cumsum")]
        [DataRow("cumprod")]
        public void FittingOut_Written_AliasAndReadOnly_Refused(string name)
        {
            bool sum = name == "cumsum";
            using (var x = np.arange(1.0, 4.0))
            using (var o = Full(np.float64, 3))
            {
                ReferenceEquals(Call(name, x, 0, o), o).Should().BeTrue("NumPy returns the out");
                Vals(o).Should().Equal(sum ? new[] { 1.0, 3, 6 } : new[] { 1.0, 2, 6 });
            }

            using var buf = np.arange(1.0, 9.0);
            using var head = buf[":3"];
            using var alias = head.reshape(3, 1);
            new Action(() => Call(name, head, 0, alias)).Should().Throw<IncorrectShapeException>().WithMessage(WrongSize);
            Vals(alias).Should().Equal(1.0, 2, 3);

            using var three = np.arange(3.0);
            using var readOnly = Full(np.float64, 4);
            readOnly.setflags(write: false);
            new Action(() => Call(name, three, 0, readOnly)).Should().Throw<ValueError>().WithMessage("output array is read-only");
        }
    }
}
