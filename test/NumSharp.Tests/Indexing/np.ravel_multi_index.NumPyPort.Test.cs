using System;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     np.ravel_multi_index against NumPy 2.4.2's <c>arr_ravel_multi_index</c> (<c>compiled_base.c</c>): what the
    ///     snapshot replay of its 29,106 calls found. Every expectation is NumPy's output, the same on Windows and Linux.
    /// </summary>
    [TestClass]
    public class np_ravel_multi_index_numpy_port_tests
    {
        /// <summary>
        ///     A null coordinate array is NumPy's <c>None</c>: <c>astype_anyint</c> makes it a 0-d object array and refuses it
        ///     like a float array, <c>TypeError: only int indices permitted</c>, in the list's order - after the count check
        ///     and the checks of the mode, the order and the dims. It leaked a NullReferenceException.
        /// </summary>
        [TestMethod]
        public void NullCoordinateArray_IsNumPysNone_OnlyIntIndicesPermitted()
        {
            // np.ravel_multi_index((np.arange(3), None), (4, 4)) -> TypeError: only int indices permitted
            new Action(() => np.ravel_multi_index(new NDArray[] { np.arange(3), null }, new[] { 4, 4 }))
                .Should().Throw<TypeError>().WithMessage("only int indices permitted");
            // np.ravel_multi_index((None, np.array([1.5])), (4, 4)) / ((np.array([1.5]), None), (4, 4)): the same text
            new Action(() => np.ravel_multi_index(new NDArray[] { null, np.array(new[] { 1.5 }) }, new[] { 4, 4 }))
                .Should().Throw<TypeError>().WithMessage("only int indices permitted");
            new Action(() => np.ravel_multi_index(new NDArray[] { np.array(new[] { 1.5 }), null }, new[] { 4, 4 }))
                .Should().Throw<TypeError>().WithMessage("only int indices permitted");
            // the per-axis modes overload: np.ravel_multi_index((np.arange(2), None), (4, 4), mode=('wrap', 'clip'))
            new Action(() => np.ravel_multi_index(new NDArray[] { np.arange(2), null }, new[] { 4, 4 }, new[] { "wrap", "clip" }))
                .Should().Throw<TypeError>().WithMessage("only int indices permitted");
            // before the zero-dimension check: np.ravel_multi_index((None, np.arange(2)), (0, 4))
            new Action(() => np.ravel_multi_index(new NDArray[] { null, np.arange(2) }, new[] { 0, 4 }))
                .Should().Throw<TypeError>().WithMessage("only int indices permitted");
        }

        /// <summary>
        ///     The checks NumPy makes before it converts the coordinates still come first with a null entry: the count, the
        ///     mode, the order and the dims product.
        /// </summary>
        [TestMethod]
        public void NullCoordinateArray_TheEarlierChecksComeFirst()
        {
            // np.ravel_multi_index((None,), (4, 4)) -> ValueError: parameter multi_index must be a sequence of length 2
            new Action(() => np.ravel_multi_index(new NDArray[] { null }, new[] { 4, 4 }))
                .Should().Throw<ValueError>().WithMessage("parameter multi_index must be a sequence of length 2");
            // mode='bogus' -> ValueError: clipmode must be one of 'clip', 'raise', or 'wrap' (got 'bogus')
            new Action(() => np.ravel_multi_index(new NDArray[] { np.arange(3), null }, new[] { 4, 4 }, "bogus"))
                .Should().Throw<ValueError>().WithMessage("clipmode must be one of 'clip', 'raise', or 'wrap' (got 'bogus')");
            // order='A' -> ValueError: only 'C' or 'F' order is permitted
            new Action(() => np.ravel_multi_index(new NDArray[] { np.arange(3), null }, new[] { 4, 4 }, "raise", 'A'))
                .Should().Throw<ValueError>().WithMessage("only 'C' or 'F' order is permitted");
            // dims (2**31-1, 2**31-1, 3) -> ValueError: invalid dims: ... (the count is 2 of 3, checked later)
            new Action(() => np.ravel_multi_index(new NDArray[] { np.arange(2), null }, new[] { int.MaxValue, int.MaxValue, 3 }))
                .Should().Throw<ValueError>()
                .WithMessage("invalid dims: array size defined by dims is larger than the maximum possible size.");
        }
    }
}
