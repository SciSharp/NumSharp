using System;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs found while closing Gym.NET's NumSharp gaps (Farama's Gymnasium ported to C#, 2026-11-10): each test
    ///     asserts NumPy 2.4.2's behavior and fails until NumSharp gives it.
    /// </summary>
    /// <remarks>
    ///     The gaps themselves (an empty bare <c>AssertionError()</c>, the character dtype, <c>NotImplementedError</c> /
    ///     <c>ImportError</c> with causes, <c>np.linspace</c> over array bounds) are fixed and tested where they live; this
    ///     file holds what the fixes showed and leaves to the owner's open decision OD2 (the array_like parameter type,
    ///     <c>docs/plans/API_DIFF_GAP.md</c>).
    /// </remarks>
    [TestClass]
    public class OpenBugsGymNetTests : TestClass
    {
        // ============================================================================
        //  BUG: np.asarray(C# array, dtype) casts the array instead of storing each item through the dtype.
        //
        //  NumPy's np.asarray(list, dtype) is np.array(list, dtype, copy=False): the coercion of a sequence, every item
        //  stored by the requested dtype's setitem (PyArray_Pack). NumSharp's np.asarray has no array_like overload, so a
        //  C# array binds asarray(NDArray a, DType dtype) through the implicit Array -> NDArray conversion: the array is
        //  built with its OWN discovered dtype first and then cast. The character dtype then truncates instead of
        //  splitting (Gymnasium's np.asarray(desc, dtype="c")), an object[] list is refused outright, and a parse error
        //  quotes np.str_('x') (the cast's text) instead of the Python str 'x'. np.array(C# array, dtype) is correct today.
        //  The fix waits on OD2: an array_like (Array / object) parameter that reaches NDArray.array's coercion.
        // ============================================================================

        /// <summary>
        ///     <c>np.asarray(list, dtype)</c> stores every item through the dtype, as <c>np.array(list, dtype)</c> does:
        ///     the character dtype splits texts into a (2, 4) <c>|S1</c> map, a Python int out of an int8's range raises
        ///     NumPy's OverflowError, and a text that is no int quotes the Python str.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception while the cast route is taken.</exception>
        [TestMethod, OpenBugs(IssueUrl = "consumer-gymnet-asarray-list-dtype")]
        public void AsArrayOfAList_StoresThroughTheDtype()
        {
            // np.asarray(["SFFF", "FHFH"], dtype="c").shape == (2, 4); today (2,) |S1 b'SF' (the cast keeps one character).
            using (var map = np.asarray(new[] { "SFFF", "FHFH" }, np.dtype("c")))
                map.shape.Should().Equal(new long[] { 2, 4 }, "NumPy splits each text into its characters");

            // np.asarray([300], dtype=np.int8): OverflowError: Python integer 300 out of bounds for int8 (today: NotSupportedException).
            Action overflow = () => np.asarray(new object[] { 300 }, np.int8);
            overflow.Should().Throw<OverflowError>().WithMessage("Python integer 300 out of bounds for int8");

            // np.asarray(['12', 'x'], dtype=np.int64): the setitem's text quotes the Python str (today: np.str_('x')).
            Action parse = () => np.asarray(new[] { "12", "x" }, np.int64);
            parse.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: 'x'");
        }
    }
}
