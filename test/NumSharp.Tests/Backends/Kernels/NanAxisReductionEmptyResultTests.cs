using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     <c>np.nansum</c> / <c>np.nanprod</c> along one axis of a float array with a zero extent: the result has NumPy's
    ///     shape, dtype and identity values, and an EMPTY result reads nothing from the input.
    /// </summary>
    /// <remarks>
    ///     <para><b>The defect this pins.</b> The NaN axis driver (<c>DefaultEngine.ExecuteNanAxisReduction</c>, and its
    ///     scalar fallback that serves float16) forced one output for an empty result
    ///     (<c>outputSize = result.size &gt; 0 ? result.size : 1</c>). The kernel then read a whole axis from an input
    ///     holding no element and wrote one element into a result holding none; with a zero extent left of another output
    ///     axis (<c>(2, 0, 3)</c> along axis 2) its output-coordinate decomposition divided by zero instead, where NumPy
    ///     returns an empty <c>(2, 0)</c> array. The read ended the oracle's test host once, through the sum_identity
    ///     tier's <c>np.nansum</c> of a <c>(0, 3)</c> array along axis 1, which the leak sweep replays.</para>
    ///     <para><b>What a failure looks like.</b> <see cref="ZeroExtent_NanSumAndNanProd_MatchNumPy"/> fails with the
    ///     <see cref="DivideByZeroException"/>; <see cref="EmptyResult_OfAFencedInput_ReadsNoElement"/> ends the test host
    ///     ("Test host process crashed") with <c>NanReduceContiguousAxis</c> (float32 / float64) or the scalar fallback
    ///     (float16) in the stack. Expected shapes and values are NumPy 2.4.2's.</para>
    /// </remarks>
    [TestClass]
    public unsafe class NanAxisReductionEmptyResultTests : TestClass
    {
        /// <summary>The float dtypes np.nansum / np.nanprod reduce through the NaN axis driver.</summary>
        private static readonly NPTypeCode[] FloatTypes = { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Half };

        /// <summary>
        ///     Every input shape and axis with a zero extent, with NumPy's result shape without and with keepdims. An
        ///     empty <c>Result</c> means the reduced axis is not the only zero extent: nothing is computed. A non-empty one
        ///     means the reduced axis is the empty one: every output is the identity.
        /// </summary>
        private static readonly (long[] Shape, int Axis, long[] Result, long[] KeptResult)[] Cases =
        {
            // Empty results. (2, 0, 3) / (3, 0, 0) along axis 2 and (2, 3, 0, 4) along axes 1 and 3 threw
            // DivideByZeroException; the others read the input and wrote the result out of bounds.
            (new long[] { 0, 5 }, 1, new long[] { 0 }, new long[] { 0, 1 }),
            (new long[] { 0, 5 }, -1, new long[] { 0 }, new long[] { 0, 1 }),
            (new long[] { 5, 0 }, 0, new long[] { 0 }, new long[] { 1, 0 }),
            (new long[] { 0, 0 }, 1, new long[] { 0 }, new long[] { 0, 1 }),
            (new long[] { 2, 0, 3 }, 2, new long[] { 2, 0 }, new long[] { 2, 0, 1 }),
            (new long[] { 2, 0, 3 }, 0, new long[] { 0, 3 }, new long[] { 1, 0, 3 }),
            (new long[] { 3, 0, 0 }, 2, new long[] { 3, 0 }, new long[] { 3, 0, 1 }),
            (new long[] { 0, 7, 3 }, 1, new long[] { 0, 3 }, new long[] { 0, 1, 3 }),
            (new long[] { 0, 7, 3 }, 2, new long[] { 0, 7 }, new long[] { 0, 7, 1 }),
            (new long[] { 2, 3, 0, 4 }, 3, new long[] { 2, 3, 0 }, new long[] { 2, 3, 0, 1 }),
            (new long[] { 2, 3, 0, 4 }, 1, new long[] { 2, 0, 4 }, new long[] { 2, 1, 0, 4 }),
            // The reduced axis is the empty one: the identity per output (these were right before the fix).
            (new long[] { 0, 5 }, 0, new long[] { 5 }, new long[] { 1, 5 }),
            (new long[] { 5, 0 }, 1, new long[] { 5 }, new long[] { 5, 1 }),
            (new long[] { 2, 0, 3 }, 1, new long[] { 2, 3 }, new long[] { 2, 1, 3 }),
            (new long[] { 0, 7, 3 }, 0, new long[] { 7, 3 }, new long[] { 1, 7, 3 }),
            (new long[] { 2, 3, 0, 4 }, 2, new long[] { 2, 3, 4 }, new long[] { 2, 3, 1, 4 }),
        };

        /// <summary>Runs <c>np.nansum</c> or <c>np.nanprod</c>.</summary>
        /// <param name="product">True for np.nanprod (identity 1), false for np.nansum (identity 0).</param>
        /// <param name="a">The input.</param>
        /// <param name="axis">The reduced axis.</param>
        /// <param name="keepdims">NumPy's keepdims.</param>
        /// <returns>The reduction's result.</returns>
        private static NDArray Reduce(bool product, NDArray a, int axis, bool keepdims)
            => product ? np.nanprod(a, axis, keepdims) : np.nansum(a, axis, keepdims);

        /// <summary>
        ///     Checks a result against NumPy's: the input's dtype, the expected shape, and the identity in every element.
        /// </summary>
        /// <param name="what">Label for the failure message.</param>
        /// <param name="r">The result.</param>
        /// <param name="dtype">The input's dtype (NumPy keeps it).</param>
        /// <param name="shape">NumPy's result shape.</param>
        /// <param name="identity">The reduction's identity, which every element of a non-empty result holds.</param>
        private static void AssertNumPyResult(string what, NDArray r, NPTypeCode dtype, long[] shape, double identity)
        {
            r.typecode.Should().Be(dtype, what);
            r.shape.Should().Equal(shape, what);
            using var values = r.astype(np.float64).flatten();
            for (long k = 0; k < values.size; k++)
                if (values.GetAtIndex<double>(k) != identity)
                    Assert.Fail($"{what}: element {k} is {values.GetAtIndex<double>(k)}, NumPy's identity is {identity}");
        }

        /// <summary>
        ///     Every zero-extent case, both reductions, every float dtype, with and without keepdims: NumPy's shape, dtype
        ///     and identity values. On the parent of the fix the cases with a zero extent left of another output axis threw
        ///     <see cref="DivideByZeroException"/>.
        /// </summary>
        [TestMethod]
        public void ZeroExtent_NanSumAndNanProd_MatchNumPy()
        {
            foreach (var dtype in FloatTypes)
            foreach (bool product in new[] { false, true })
            foreach (var (shape, axis, result, keptResult) in Cases)
            foreach (bool keepdims in new[] { false, true })
            {
                string what = $"{(product ? "nanprod" : "nansum")} {dtype} ({string.Join(", ", shape)}) axis {axis} keepdims {keepdims}";
                using var a = np.zeros(new Shape(shape), dtype);
                using var r = Reduce(product, a, axis, keepdims);
                AssertNumPyResult(what, r, dtype, keepdims ? keptResult : result, product ? 1 : 0);
            }
        }

        /// <summary>
        ///     The same cases over an input whose data pointer is the first byte of an inaccessible page: a reduction that
        ///     reads one element of an input holding none ends the test host. Windows only (the fence is a reserved page);
        ///     Inconclusive elsewhere.
        /// </summary>
        [TestMethod]
        public void EmptyResult_OfAFencedInput_ReadsNoElement()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");

            using var fence = new FencedMemory(pages: 1);
            // A block of 0 bytes against the high fence starts AT the fence: its first element would be inaccessible.
            byte* atFence = fence.Place(0, atEnd: true);
            foreach (var dtype in FloatTypes)
            foreach (bool product in new[] { false, true })
            foreach (var (shape, axis, result, keptResult) in Cases)
            foreach (bool keepdims in new[] { false, true })
            {
                string what = $"fenced {(product ? "nanprod" : "nansum")} {dtype} ({string.Join(", ", shape)}) axis {axis} keepdims {keepdims}";
                // The empty array is a view of the fence (frombuffer copies nothing, reshape of a contiguous array is a
                // view); the address check keeps the test from silently degrading to a copy on the heap.
                using var flat = np.frombuffer((IntPtr)atFence, 0, dtype);
                using var a = flat.reshape(new Shape(shape));
                ((IntPtr)a.Storage.Address).Should().Be((IntPtr)atFence, what + ": the input must sit at the fence");
                using var r = Reduce(product, a, axis, keepdims);
                AssertNumPyResult(what, r, dtype, keepdims ? keptResult : result, product ? 1 : 0);
            }
        }
    }
}
