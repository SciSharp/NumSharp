namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #341, "NDArray string problem" (https://github.com/SciSharp/NumSharp/issues/341):
    ///     <c>new NDArray(typeof(string), shape)</c> threw NotSupportedException in 2019; since 0.80.0 NumPy's text dtypes
    ///     exist and every line of the answer's example runs. Expected reprs are NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class Issue341
    {
        /// <summary>
        ///     The issue's own constructor call gives an empty-string <c>&lt;U1</c> array, as <c>np.zeros(3, dtype=str)</c>
        ///     does in NumPy.
        /// </summary>
        [TestMethod]
        public void NewNDArrayOfTypeString_IsAnEmptyU1Array()
        {
            var a = new NDArray(typeof(string), new Shape(3));

            a.dtype.str.Should().Be("<U1");
            a.shape.Should().Equal(3L);
            np.array_repr(a).Should().Be("array(['', '', ''], dtype='<U1')");
        }

        /// <summary>
        ///     <c>np.array(string[])</c> is a real fixed-width <c>U</c> array sized to its longest string, and
        ///     <c>np.strings.upper</c> computes on it, as the answer shows.
        /// </summary>
        [TestMethod]
        public void StringArray_AndNpStrings_AsInTheAnswer()
        {
            var b = np.array(new[] { "hello", "world!" });

            np.array_repr(b).Should().Be("array(['hello', 'world!'], dtype='<U6')");
            np.array_repr(np.strings.upper(b)).Should().Be("array(['HELLO', 'WORLD!'], dtype='<U6')");
        }
    }
}
