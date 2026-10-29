using System;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Tests.Utilities;

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

        /// <summary>
        ///     <see cref="TensorEngine.RavelMultiIndex"/> takes NumPy's mode argument as it arrives
        ///     (<c>PyArray_ConvertClipmodeSequence</c> over a Python object): a list or tuple of strings - any non-string
        ///     sequence here - gives each axis its mode, a None entry raises on its axis, the length is checked before the
        ///     entries, and an object that is neither a string, None nor a sequence of them is <c>TypeError: clipmode not
        ///     understood</c>, even for an empty dims. The public overloads pass only a string or a string[]; this calls the
        ///     engine member.
        /// </summary>
        [TestMethod]
        public void EngineMember_TakesNumPysModeObject()
        {
            var engine = BackendFactory.GetEngine();
            NDArray[] mi = { np.array(new long[] { 0, 3, 5 }), np.array(new long[] { 1, -1, 4 }) };
            var dims = new[] { 4, 4 };
            // np.ravel_multi_index(mi, (4, 4), mode=['wrap', 'clip']) -> array([ 1, 12,  7])
            engine.RavelMultiIndex(mi, dims, new List<string> { "wrap", "clip" }).Should().BeOfValues(1L, 12L, 7L);
            // mode=['wrap', None] -> ValueError: invalid entry in coordinates array (None raises on axis 1: the -1)
            new Action(() => engine.RavelMultiIndex(mi, dims, new object[] { "wrap", null }))
                .Should().Throw<ValueError>().WithMessage("invalid entry in coordinates array");
            // mode=3.5 / mode=['wrap', 3.5] / mode=object() -> TypeError: clipmode not understood
            new Action(() => engine.RavelMultiIndex(mi, dims, 3.5))
                .Should().Throw<TypeError>().WithMessage("clipmode not understood");
            new Action(() => engine.RavelMultiIndex(mi, dims, new object[] { "wrap", 3.5 }))
                .Should().Throw<TypeError>().WithMessage("clipmode not understood");
            new Action(() => engine.RavelMultiIndex(mi, dims, new object()))
                .Should().Throw<TypeError>().WithMessage("clipmode not understood");
            // mode=[3.5] -> ValueError: list of clipmodes has wrong length (1 instead of 2) (the length first)
            new Action(() => engine.RavelMultiIndex(mi, dims, new object[] { 3.5 }))
                .Should().Throw<ValueError>().WithMessage("list of clipmodes has wrong length (1 instead of 2)");
            // np.ravel_multi_index((), (), mode=3.5) -> TypeError: clipmode not understood (a single mode is checked
            // even when there is no axis)
            new Action(() => engine.RavelMultiIndex(Array.Empty<NDArray>(), Array.Empty<int>(), 3.5))
                .Should().Throw<TypeError>().WithMessage("clipmode not understood");
        }
    }
}
