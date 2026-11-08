using System;
using System.Linq;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Indexing;

/// <summary>
///     np.trace's float results against NumPy 2.4.2, bit for bit. NumPy's trace is <c>PyArray_Trace</c>: the diagonal VIEW's
///     <c>add.reduce</c> along its last axis, so the loop form NumPy's reduction iterator picks decides the last bits — one
///     reduce-form call per output (<c>+0 + pairwise_sum(diagonal)</c>) when the diagonal is the iterator's innermost axis
///     (every 2-D trace, a 3-D trace whose remaining axis has the larger |stride|), else the elementwise add repeated along
///     the diagonal (a sequential sum from +0; float16 rounded to float16 after every step). The fast path
///     (<c>DefaultEngine.TryTraceFast</c>) reproduces both forms for float16 / float32 / float64 / complex128 (parity QA-47,
///     2026-11-05). Expected values are NumPy's raw bits from the same RandomState draws.
/// </summary>
[TestClass]
public class TraceNumPyLoopFormsTests
{
    /// <summary>The raw bits of a float64 (or complex128, two words per element) result, C order.</summary>
    /// <param name="a">The result.</param>
    /// <returns>One uint64 per float64 word.</returns>
    private static ulong[] Bits64(NDArray a) => np.ascontiguousarray(a).view(np.uint64).ToArray<ulong>();

    /// <summary>The raw bits of a float32 result, C order.</summary>
    /// <param name="a">The result.</param>
    /// <returns>One uint32 per element.</returns>
    private static uint[] Bits32(NDArray a) => np.ascontiguousarray(a).view(np.uint32).ToArray<uint>();

    /// <summary>The raw bits of a float16 result, C order.</summary>
    /// <param name="a">The result.</param>
    /// <returns>One uint16 per element.</returns>
    private static ushort[] Bits16(NDArray a) => np.ascontiguousarray(a).view(np.uint16).ToArray<ushort>();

    /// <summary>
    ///     The parity-QA reproduction: an 8 x 8 float64 trace is the pairwise sum (one leaf of eight accumulators), which the
    ///     old sequential walk missed in the last bit (<c>469.5169613223648</c>), and equals NumSharp's own exact
    ///     <c>np.sum(np.diagonal(m))</c>, as NumPy's does.
    /// </summary>
    [TestMethod]
    public void Float64_2D_IsThePairwiseSum()
    {
        var rs = np.random.RandomState(7);
        using var m = rs.rand(8, 8) * 1000.0 - 500.0;

        using var trace = np.trace(m);
        using var viaSum = np.sum(np.diagonal(m));

        Bits64(trace).Should().Equal(0x407D5845793C4DAAUL);   // 469.51696132236486
        Bits64(viaSum).Should().Equal(Bits64(trace));
        trace.dtype.Should().Be(typeof(double));
        trace.ndim.Should().Be(0);
    }

    /// <summary>A 1000-element float32 diagonal: the pairwise recursion above one 128-element leaf, in float32.</summary>
    [TestMethod]
    public void Float32_2D_1000_IsThePairwiseSum()
    {
        var rs = np.random.RandomState(7);
        using var m = (rs.rand(1000, 1000) * 1000.0 - 500.0).astype(np.float32);

        using var trace = np.trace(m);

        Bits32(trace).Should().Equal(0x46113686u);           // 9293.631
        trace.dtype.Should().Be(typeof(float));
    }

    /// <summary>A 129-element complex128 diagonal: CDOUBLE_pairwise_sum (four accumulators per part), +0 per part.</summary>
    [TestMethod]
    public void Complex128_2D_IsThePairwiseSum()
    {
        var rs = np.random.RandomState(11);
        using var re = rs.rand(129, 129) * 1000.0 - 500.0;
        using var im = rs.rand(129, 129) * 1000.0 - 500.0;
        var r = re.ToArray<double>();
        var i = im.ToArray<double>();
        using var c = np.array(Enumerable.Range(0, r.Length).Select(k => new Complex(r[k], i[k])).ToArray()).reshape(129, 129);

        using var trace = np.trace(c);

        Bits64(trace).Should().Equal(0x407ACA1F2833F3E8UL, 0x4092F449DAE0EF09UL);
        trace.dtype.Should().Be(typeof(Complex));
    }

    /// <summary>
    ///     A 200-element float16 diagonal: HALF_pairwise_sum in float32, added to the +0 identity and narrowed to float16 once
    ///     (not a double accumulator rounded at the end).
    /// </summary>
    [TestMethod]
    public void Float16_2D_IsTheFloat32PairwiseSumNarrowedOnce()
    {
        var rs = np.random.RandomState(19);
        using var h = (rs.rand(200, 200) * 20.0 - 10.0).astype(np.float16);

        using var trace = np.trace(h);

        Bits16(trace).Should().Equal((ushort)0xD60A);          // -96.6
        trace.dtype.Should().Be(typeof(Half));
    }

    /// <summary>
    ///     A batched trace over the two LAST axes of a C-ordered (3, 16, 16) stack: the diagonal view is (3, 16) with strides
    ///     (2048, 136) bytes, so the diagonal is innermost and every output is one pairwise call (parity QA-47's <c>rp19</c>).
    /// </summary>
    [TestMethod]
    public void Float64_3D_DiagonalInnermost_IsOnePairwiseCallPerOutput()
    {
        var rs = np.random.RandomState(23);
        using var b = rs.rand(3, 16, 16) * 1000.0 - 500.0;

        using var trace = np.trace(b, 0, 1, 2);

        Bits64(trace).Should().Equal(0xC050CF8378D870C0UL, 0xC084C29AD7FEC536UL, 0x4087E10D924F169EUL);
    }

    /// <summary>
    ///     A batched trace over the two FIRST axes of a C-ordered (16, 16, 3) array: the diagonal view is (3, 16) with strides
    ///     (8, 408) bytes, so NumPy's iterator walks the remaining axis innermost and each diagonal step is an elementwise add
    ///     into all three outputs — a sequential sum from +0 (NumPy's bits equal that model), not the pairwise sum.
    /// </summary>
    [TestMethod]
    public void Float64_3D_RemainingAxisInnermost_IsTheSequentialSum()
    {
        var rs = np.random.RandomState(29);
        using var s = rs.rand(16, 16, 3) * 1000.0 - 500.0;

        using var trace = np.trace(s);

        Bits64(trace).Should().Equal(0x4092BF3C2B3A58A5UL, 0xC079DAC879980C8AUL, 0xC09B04CB5D54B44CUL);
    }

    /// <summary>
    ///     The float16 elementwise form: HALF_add narrows the float32 sum to float16 after EVERY diagonal step, so a double
    ///     accumulator rounded once (the old kernel) differs.
    /// </summary>
    [TestMethod]
    public void Float16_3D_RemainingAxisInnermost_RoundsEveryStep()
    {
        var rs = np.random.RandomState(31);
        using var s = (rs.rand(40, 40, 3) * 20.0 - 10.0).astype(np.float16);

        using var trace = np.trace(s);

        Bits16(trace).Should().Equal((ushort)0xD0C0, (ushort)0x4C01, (ushort)0x54F1);
    }

    /// <summary>
    ///     A reversed view (negative diagonal stride, which NumPy's reduction keeps: DONT_NEGATE_STRIDES) is summed in
    ///     logical order, offset diagonal included.
    /// </summary>
    [TestMethod]
    public void Float64_ReversedView_SumsInLogicalOrder()
    {
        var rs = np.random.RandomState(37);
        using var m = rs.rand(130, 130) * 1000.0 - 500.0;
        var rv = m["::-1, ::-1"];

        using var trace = np.trace(rv);
        using var shifted = np.trace(rv, 3);

        Bits64(trace).Should().Equal(0x40AA98AE4C055B5BUL);
        Bits64(shifted).Should().Equal(0xC0B049A5D7F83DA9UL);
    }

    /// <summary>
    ///     Broadcast operands: a row broadcast to (150, 150) has a diagonal stride of one item, and a scalar broadcast to
    ///     (300, 300) a diagonal stride of 0 — NumPy pairwise-sums the repeated element 300 times (one call), it does not
    ///     multiply.
    /// </summary>
    [TestMethod]
    public void Broadcast_Operands_ArePairwiseSummedInPlace()
    {
        var rs = np.random.RandomState(41);
        using var row = rs.rand(1, 150) * 1000.0 - 500.0;
        var bc = np.broadcast_to(row, new Shape(150, 150));
        var scalar = np.broadcast_to(NDArray.Scalar(0.1f), new Shape(300, 300));

        using var t1 = np.trace(bc);
        using var t2 = np.trace(scalar);

        Bits64(t1).Should().Equal(0xC0BACE55014E62E8UL);
        Bits32(t2).Should().Equal(0x41F00002u);
    }

    /// <summary>
    ///     The add identity starts every output: a diagonal of -0.0 sums to +0.0, and an empty diagonal (offset past the
    ///     matrix) is +0.0, in both loop forms.
    /// </summary>
    [TestMethod]
    public void Identity_NegativeZerosAndEmptyDiagonal_ArePositiveZero()
    {
        using var nz = np.full(new Shape(5, 5), -0.0);
        using var nz3 = np.full(new Shape(5, 5, 3), -0.0f);

        Bits64(np.trace(nz)).Should().Equal(0UL);
        Bits64(np.trace(nz, 7)).Should().Equal(0UL);
        Bits32(np.trace(nz3)).Should().Equal(0u, 0u, 0u);       // elementwise form (remaining axis innermost)
        Bits32(np.trace(nz3, 0, 1, 2)).Should().Equal(0u, 0u, 0u, 0u, 0u);   // reduce form
    }

    /// <summary>
    ///     The elementwise form's NaN operand order: <c>out = out + x</c> keeps the second operand's NaN when both are NaN, so
    ///     the LATER diagonal element's payload survives; a lone signalling NaN comes back quiet with its payload.
    /// </summary>
    [TestMethod]
    public void Float32_ElementwiseForm_TwoNaNs_KeepTheLaterElementsPayload()
    {
        // C index i*4 + j*2 + k: [0,0,0] = 0x7FC0A5A5, [0,0,1] = 0x7FA00001 (signalling), [1,1,0] = 0xFFC00001.
        using var bits = np.array(new uint[] { 0x7FC0A5A5u, 0x7FA00001u, 0, 0, 0, 0, 0xFFC00001u, 0 }).reshape(2, 2, 2);
        var sl = bits.view(np.float32);

        using var trace = np.trace(sl);

        Bits32(trace).Should().Equal(0xFFC00001u, 0x7FE00001u);
    }

    /// <summary>
    ///     NumPy's identity <c>np.trace(m) == np.sum(np.diagonal(m))</c> on every inexact dtype, square and rectangular, below
    ///     and above one pairwise leaf: NumSharp's flat np.sum is NumPy's add.reduce, so the two must agree bit for bit.
    /// </summary>
    [TestMethod]
    public void TwoD_EqualsTheSumOfTheDiagonal_EveryInexactDtype()
    {
        var rs = np.random.RandomState(43);
        foreach (var (rows, cols) in new[] { (7, 7), (8, 9), (129, 130), (257, 128) })
        foreach (var dtype in new[] { np.float64, np.float32, np.float16 })
        {
            using var m = (rs.rand(rows, cols) * 200.0 - 100.0).astype(dtype);
            foreach (int offset in new[] { 0, 1, -2 })
            {
                using var trace = np.trace(m, offset);
                using var viaSum = np.sum(np.diagonal(m, offset));
                Bits64(trace.astype(np.float64)).Should().Equal(Bits64(viaSum.astype(np.float64)),
                    $"{dtype.name} ({rows}, {cols}) offset {offset}");
            }
        }
    }
}
