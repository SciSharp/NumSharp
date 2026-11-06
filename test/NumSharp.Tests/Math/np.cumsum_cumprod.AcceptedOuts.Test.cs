using System;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     Parity QA-54: np.cumsum / np.cumprod store into every out NumPy's <c>PyUFunc_Accumulate</c> accepts, where its
    ///     iterator writes. An out with MORE trailing axes keeps them at index 0; along the accumulated axis only the
    ///     input's extent is written (an extent-1 input writes the first cell); an extent-1 input broadcasts along the
    ///     other axes; an empty input or an empty out returns the out untouched; an out sharing the input's memory reads
    ///     the input first. Every out starts filled with -1, so a cell NumPy leaves untouched shows; every expected list
    ///     is NumPy 2.4.2's raveled out (scratchpad probe p174_numpy.py).
    /// </summary>
    [TestClass]
    public class CumsumCumprodAcceptedOutsTests
    {
        /// <summary>An out of <paramref name="shape"/> and <paramref name="dtype"/> filled with -1.</summary>
        /// <param name="dtype">The out's dtype.</param>
        /// <param name="shape">The out's shape (empty for a 0-d out).</param>
        /// <returns>The out.</returns>
        private static NDArray Full(DType dtype, params long[] shape) => np.full(Shape.Adopt(shape), -1, dtype);

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

        /// <summary>The two accumulations, each with its NumPy results for one case, keyed by the call's name.</summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        /// <param name="x">The input.</param>
        /// <param name="axis">The axis (null for the flat accumulation).</param>
        /// <param name="o">The out.</param>
        /// <returns>The call's return (NumPy returns the out itself).</returns>
        private static NDArray Call(string name, NDArray x, int? axis, NDArray o)
            => name == "cumsum" ? np.cumsum(x, axis, @out: o) : np.cumprod(x, axis, @out: o);

        /// <summary>
        ///     The iterator path: an out of more dimensions (trailing axes at index 0), an extent-1 input broadcast against
        ///     a longer out, and an input of extent 1 along the accumulated axis writing only the first cell.
        /// </summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        /// <exception cref="AssertFailedException">An out differs from NumPy's.</exception>
        [TestMethod]
        [DataRow("cumsum")]
        [DataRow("cumprod")]
        public void IteratorPath_StoresWhereNumPysIteratorWrites(string name)
        {
            bool sum = name == "cumsum";
            using var a = np.arange(12.0).reshape(3, 4);

            using (var o = Full(np.float64, 3, 4, 1))
            {
                ReferenceEquals(Call(name, a, 0, o), o).Should().BeTrue("NumPy returns the out");
                Vals(o).Should().Equal(sum
                    ? new[] { 0.0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21 }
                    : new[] { 0.0, 1, 2, 3, 0, 5, 12, 21, 0, 45, 120, 231 });
            }
            using (var o = Full(np.float64, 3, 4, 2))
            {
                Call(name, a, 1, o);
                Vals(o).Should().Equal(sum
                    ? new[] { 0.0, -1, 1, -1, 3, -1, 6, -1, 4, -1, 9, -1, 15, -1, 22, -1, 8, -1, 17, -1, 27, -1, 38, -1 }
                    : new[] { 0.0, -1, 0, -1, 0, -1, 0, -1, 4, -1, 20, -1, 120, -1, 840, -1, 8, -1, 72, -1, 720, -1, 7920, -1 });
            }

            using var ones = np.ones(new Shape(3, 1));
            using var col = ones * 2;
            using (var o = Full(np.float64, 3, 4))
            {
                Call(name, col, 0, o);
                Vals(o).Should().Equal(sum
                    ? new[] { 2.0, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6 }
                    : new[] { 2.0, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8 });
            }
            using (var o = Full(np.float64, 3, 4))
            {
                Call(name, col, 1, o);
                Vals(o).Should().Equal(2.0, -1, -1, -1, 2, -1, -1, -1, 2, -1, -1, -1);
            }

            using var row = np.arange(1.0, 5.0).reshape(1, 4);
            using (var o = Full(np.float64, 3, 4))
            {
                Call(name, row, 0, o);
                Vals(o).Should().Equal(1.0, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1);
            }
            using (var o = Full(np.float64, 3, 4))
            {
                Call(name, row, 1, o);
                Vals(o).Should().Equal(sum
                    ? new[] { 1.0, 3, 6, 10, 1, 3, 6, 10, 1, 3, 6, 10 }
                    : new[] { 1.0, 2, 6, 24, 1, 2, 6, 24, 1, 2, 6, 24 });
            }

            using var ints = np.arange(1, 7).astype(np.int32).reshape(2, 3);
            using (var o = Full(np.int64, 2, 3, 1))
            {
                Call(name, ints, 1, o);
                Vals(o).Should().Equal(sum ? new[] { 1.0, 3, 6, 4, 9, 15 } : new[] { 1.0, 2, 6, 4, 20, 120 });
            }
        }

        /// <summary>
        ///     One-dimensional inputs: a float32 out of more dimensions (the iterator path NumPy takes for a cast out), an
        ///     extent-1 input into a longer out (only the first cell written), the flat accumulation into a 2-D out.
        /// </summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        /// <exception cref="AssertFailedException">An out differs from NumPy's.</exception>
        [TestMethod]
        [DataRow("cumsum")]
        [DataRow("cumprod")]
        public void OneDimensionalInputs_IntoWiderOuts(string name)
        {
            bool sum = name == "cumsum";
            using var v = np.arange(1.0, 4.0);
            using (var o = Full(np.float32, 3, 1))
            {
                Call(name, v, 0, o);
                Vals(o).Should().Equal(sum ? new[] { 1.0, 3, 6 } : new[] { 1.0, 2, 6 });
            }
            using var five = np.array(new[] { 5.0 });
            using (var o = Full(np.float32, 3))
            {
                Call(name, five, 0, o);
                Vals(o).Should().Equal(5.0, -1, -1);
            }
            using (var o = Full(np.float32, 3, 2))
            {
                Call(name, v, null, o);
                Vals(o).Should().Equal(sum ? new[] { 1.0, -1, 3, -1, 6, -1 } : new[] { 1.0, -1, 2, -1, 6, -1 });
            }
        }

        /// <summary>
        ///     NumPy's early exits: an empty input or an empty out returns the out untouched, before any shape is compared.
        /// </summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        /// <exception cref="AssertFailedException">An empty case raised or wrote.</exception>
        [TestMethod]
        [DataRow("cumsum")]
        [DataRow("cumprod")]
        public void EmptyInputOrOut_ReturnsTheOutUntouched(string name)
        {
            using var empty = np.zeros(new Shape(0));
            using (var o = Full(np.float64, 5))
            {
                ReferenceEquals(Call(name, empty, null, o), o).Should().BeTrue();
                Vals(o).Should().Equal(-1.0, -1, -1, -1, -1);
            }
            using var three = np.arange(3.0);
            using (var o = Full(np.float64, 0))
                ReferenceEquals(Call(name, three, 0, o), o).Should().BeTrue();
            using var emptyRows = np.zeros(new Shape(0, 4));
            using (var o = Full(np.float64, 0, 4, 2))
                ReferenceEquals(Call(name, emptyRows, 0, o), o).Should().BeTrue();
        }

        /// <summary>
        ///     An out that is a reshaped view of the input's own memory: NumPy's overlap check sends it through the
        ///     iterator, which reads the input before the write, so the result is the plain accumulation.
        /// </summary>
        /// <param name="name">"cumsum" or "cumprod".</param>
        /// <exception cref="AssertFailedException">The aliased out differs from NumPy's.</exception>
        [TestMethod]
        [DataRow("cumsum")]
        [DataRow("cumprod")]
        public void AnOutAliasingTheInput_ReadsTheInputFirst(string name)
        {
            using var buf = np.arange(1.0, 9.0);
            using var head = buf[":3"];
            using var o = head.reshape(3, 1);
            Call(name, head, 0, o);
            Vals(o).Should().Equal(name == "cumsum" ? new[] { 1.0, 3, 6 } : new[] { 1.0, 2, 6 });
        }
    }
}
