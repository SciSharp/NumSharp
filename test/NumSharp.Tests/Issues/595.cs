using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #595, "[Performance] ClipNDArray with @out does redundant copy"
    ///     (https://github.com/SciSharp/NumSharp/issues/595): <c>np.maximum</c> / <c>np.minimum</c> / <c>np.fmax</c> /
    ///     <c>np.fmin</c> write <c>out=</c> in their single kernel pass, and <c>np.clip</c> writes straight into <c>out</c>.
    /// </summary>
    /// <remarks>
    ///     The pass count itself is not observable from a test, but its consequence is: the old route first copied the array
    ///     into <c>out</c> (<c>copyto(out, lhs)</c>) and then clipped in place, so an <c>out</c> that is the OTHER operand
    ///     (the second array, or a bound) was overwritten before it was read. NumPy computes such a call as if the inputs
    ///     were read first; each test passes such an aliased <c>out</c> and expects NumPy 2.4.2's values, and that the call
    ///     returns <c>out</c> itself.
    /// </remarks>
    [TestClass]
    public class Issue595
    {
        /// <summary>
        ///     <c>np.maximum(a, b, out: b)</c> and <c>np.minimum(a, b, out: b)</c> read <c>b</c> before writing it
        ///     (the copy-first route gave <c>max(a, a)</c> = <c>a</c>), and return <c>out</c> itself.
        /// </summary>
        [TestMethod]
        public void MaximumMinimum_OutIsTheSecondOperand()
        {
            var a = np.array(new long[] { 1, 5, 9 });

            var b = np.array(new long[] { 4, 4, 4 });
            var r = np.maximum(a, b, @out: b);
            ReferenceEquals(r, b).Should().BeTrue("a given out is returned itself");
            b.Should().BeOfValues(4L, 5L, 9L);

            b = np.array(new long[] { 4, 4, 4 });
            np.minimum(a, b, @out: b);
            b.Should().BeOfValues(1L, 4L, 4L);
        }

        /// <summary>
        ///     <c>np.fmax</c> / <c>np.fmin</c> with <c>out</c> as the second operand skip NaN from either side, as NumPy's do.
        /// </summary>
        [TestMethod]
        public void FmaxFmin_OutIsTheSecondOperand()
        {
            var a = np.array(new[] { 1.0, double.NaN, 9.0 });

            var b = np.array(new[] { 4.0, 4.0, double.NaN });
            np.fmax(a, b, @out: b);
            b.Should().BeOfValues(4.0, 4.0, 9.0);

            b = np.array(new[] { 4.0, 4.0, double.NaN });
            np.fmin(a, b, @out: b);
            b.Should().BeOfValues(1.0, 4.0, 9.0);
        }

        /// <summary>
        ///     <c>np.clip(a, lo, 6, out: lo)</c> reads the bound before overwriting it (<c>[2, 5, 6]</c>; the copy-first route
        ///     gave <c>[1, 5, 6]</c>), an in-place clip works, and both return <c>out</c> itself.
        /// </summary>
        [TestMethod]
        public void Clip_OutIsABound_OrTheArrayItself()
        {
            var a = np.array(new long[] { 1, 5, 9 });

            var lo = np.array(new long[] { 2, 2, 2 });
            var r = np.clip(a, lo, 6, @out: lo);
            ReferenceEquals(r, lo).Should().BeTrue("a given out is returned itself");
            lo.Should().BeOfValues(2L, 5L, 6L);

            var x = np.array(new long[] { 1, 5, 9 });
            r = np.clip(x, 2, 6, @out: x);
            ReferenceEquals(r, x).Should().BeTrue();
            x.Should().BeOfValues(2L, 5L, 6L);
        }
    }
}
