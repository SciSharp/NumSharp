using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #550, "[NEP27 NEP51] Scalar and Zero-Rank Array Handling"
    ///     (https://github.com/SciSharp/NumSharp/issues/550): zero-rank arrays behave as NumPy's do, down to the error texts
    ///     of <c>len()</c>, <c>item()</c>, <c>take</c> and indexing. Expected values and texts are NumPy 2.4.2's. NEP 51
    ///     (the repr of NumPy's scalar objects) has no C# counterpart: NumSharp's scalars are CLR primitives.
    /// </summary>
    [TestClass]
    public class Issue550
    {
        /// <summary>
        ///     The answer's example: a 0-d array has shape (), ndim 0 and size 1, <c>x["..."]</c> is a 0-d view of the same
        ///     memory (NumPy's <c>x[...]</c>), <c>x[np.newaxis]</c> is <c>array([5])</c>, and the repr is <c>array(5)</c>.
        /// </summary>
        [TestMethod]
        public void ZeroRankArray_AsInTheAnswer()
        {
            var x = NDArray.Scalar(5L);

            x.shape.Should().BeEmpty();
            x.ndim.Should().Be(0);
            x.size.Should().Be(1);

            var view = x["..."];
            view.ndim.Should().Be(0);
            np.shares_memory(view, x).Should().BeTrue("x[...] is a view, not a copy");

            x[np.newaxis].Should().BeShaped(1).And.BeOfValues(5L);
            np.array_repr(x).Should().Be("array(5)");
            x.ToString().Should().Be("5");
        }

        /// <summary>
        ///     <c>item()</c> returns the value, <c>take(x, 0)</c> reads it through the flattened view, and <c>flat</c> / <c>ravel</c>
        ///     are one-element vectors, as in NumPy.
        /// </summary>
        [TestMethod]
        public void ZeroRankArray_ItemTakeFlat()
        {
            var x = NDArray.Scalar(5L);

            Convert.ToInt64(x.item()).Should().Be(5);
            np.take(x, 0).GetInt64().Should().Be(5);
            x.flat.Should().BeShaped(1);
            x.ravel().Should().BeShaped(1);
        }

        /// <summary>
        ///     The 0-d refusals carry NumPy's exact texts, as the answer says: <c>len()</c> "len() of unsized object"
        ///     (TypeError), <c>take(x, 1)</c> "index 1 is out of bounds for axis 0 with size 1" and <c>x[0]</c> "too many
        ///     indices for array: array is 0-dimensional, but 1 were indexed" (IndexError), and <c>item(1)</c> "index 1 is out
        ///     of bounds for size 1". The item text is checked on any exception type: its type is pinned separately.
        /// </summary>
        [TestMethod]
        public void ZeroRankArray_Refusals_UseNumPysTexts()
        {
            var x = NDArray.Scalar(5L);

            new Action(() => x.__len__()).Should().Throw<TypeError>().WithMessage("len() of unsized object");
            new Action(() => np.take(x, 1)).Should().Throw<IndexError>().WithMessage("index 1 is out of bounds for axis 0 with size 1");
            new Action(() => { var _ = x[0]; }).Should().Throw<IndexError>()
                .WithMessage("too many indices for array: array is 0-dimensional, but 1 were indexed");
            new Action(() => x.item(1)).Should().Throw<Exception>().WithMessage("index 1 is out of bounds for size 1");
        }

        /// <summary>
        ///     <c>item(1)</c> of a 0-d array raises <c>IndexError</c> in NumPy 2.4.2; NumSharp raises the right text as a
        ///     <c>System.IndexOutOfRangeException</c>, which a <c>catch (IndexError)</c> ported from Python misses (found while
        ///     turning the #550 answer into tests). Fails until <c>item</c> throws NumSharp's <c>IndexError</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "https://github.com/SciSharp/NumSharp/issues/550")]
        public void ZeroRankArray_ItemOutOfRange_RaisesIndexError()
        {
            var x = NDArray.Scalar(5L);

            new Action(() => x.item(1)).Should().Throw<IndexError>().WithMessage("index 1 is out of bounds for size 1");
        }
    }
}
