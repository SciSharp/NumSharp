using System;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends.Kernels;

/// <summary>
/// The trivial loop for a binary ufunc with a provided <c>out=</c> (<c>DefaultEngine.TryTrivialBinaryOpInto</c>):
/// NumPy's ufunc runs <c>try_trivial_single_output_loop</c> for a provided output too, and so does NumSharp — the
/// DirectIL kernel the fresh result uses, written straight into <c>out</c>, with no iterator.
/// </summary>
/// <remarks>
/// <para>
/// Every case runs the same call twice into identical outputs — once with the trivial loop disabled
/// (<see cref="DefaultEngine.DisableTrivialBinaryOut"/>, the iterator route every out= call took before) and once
/// enabled — and asserts the two outputs are byte-identical and that the trivial loop ran exactly when the operands
/// are trivially iterable (<see cref="DefaultEngine.TrivialBinaryOutRuns"/>), because a silent decline would pass every
/// value check. The iterator route is the oracle-verified one (out_where, nan_order and clip_ufunc tiers), so equal
/// bytes carry NumPy parity over.
/// </para>
/// <para>
/// The cases that must DECLINE are pinned too: a partial overlap between an input and the output (a single forward
/// pass would read already-written elements; the iterator copies the input first, NumPy's COPY_IF_OVERLAP), a size-1
/// input inside the output's memory, mixed dtypes, a broadcast array input, a layout mismatch, and a size-1 input of
/// higher rank than the output (NumPy raises; the iterator route raises it).
/// </para>
/// </remarks>
[TestClass]
public class TrivialBinaryOutTests
{
    /// <summary>
    /// Runs <paramref name="call"/> into an output built by <paramref name="makeOut"/> through the iterator route and
    /// through the trivial loop, and asserts both outputs hold the same bytes, the call returned its output, and the
    /// trivial loop ran exactly when <paramref name="expectTrivial"/> says so.
    /// </summary>
    /// <param name="call">The ufunc call; receives the output and returns what the ufunc returned.</param>
    /// <param name="makeOut">Builds a fresh output with the case's layout and initial contents.</param>
    /// <param name="expectTrivial">Whether the trivial loop must serve the call.</param>
    /// <param name="what">A label for failure messages.</param>
    private static void AssertRoutesAgree(Func<NDArray, NDArray> call, Func<NDArray> makeOut, bool expectTrivial, string what)
    {
        var viaIterator = makeOut();
        DefaultEngine.DisableTrivialBinaryOut = true;
        try
        {
            call(viaIterator);
        }
        finally
        {
            DefaultEngine.DisableTrivialBinaryOut = false;
        }

        var viaTrivial = makeOut();
        int before = DefaultEngine.TrivialBinaryOutRuns;
        var returned = call(viaTrivial);
        Assert.AreSame(viaTrivial, returned, $"{what}: the ufunc returns the out it was given");
        Assert.AreEqual(before + (expectTrivial ? 1 : 0), DefaultEngine.TrivialBinaryOutRuns,
            $"{what}: the trivial loop must {(expectTrivial ? "" : "NOT ")}serve the call");
        CollectionAssert.AreEqual(viaIterator.tobytes(), viaTrivial.tobytes(), $"{what}: bytes vs the iterator route");
    }

    /// <summary>Builds a C-contiguous (3, 5) array of <paramref name="dtype"/> holding small, op-safe values.</summary>
    /// <param name="dtype">The element dtype.</param>
    /// <param name="seed">Shifts the values so two operands differ.</param>
    /// <returns>A fresh array; for floats it carries a <c>-0.0</c> and a fractional value.</returns>
    private static NDArray Block(NPTypeCode dtype, int seed)
    {
        var v = new double[15];
        for (int i = 0; i < v.Length; i++)
            v[i] = ((i * 7 + seed * 3) % 11) + 1 + (dtype is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double ? 0.25 : 0);
        if (dtype is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double)
            v[seed % 15] = -0.0;
        return np.array(v).astype(dtype).reshape(3, 5);
    }

    /// <summary>
    /// Same-shape operands of every dtype the DirectIL kernels serve, C-contiguous and F-contiguous (all three operands
    /// in one order), through add / subtract / multiply / maximum: the trivial loop serves each and writes the iterator
    /// route's bytes.
    /// </summary>
    [TestMethod]
    public void SameShape_COrF_EveryDtype_MatchesTheIteratorRoute()
    {
        var dtypes = new[]
        {
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex, NPTypeCode.Decimal,
            NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
            NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Boolean,
        };
        foreach (var dt in dtypes)
        {
            var a = Block(dt, 1);
            var b = Block(dt, 2);
            var aF = np.asfortranarray(a);
            var bF = np.asfortranarray(b);
            Func<NDArray, NDArray, NDArray, NDArray>[] ops = dt == NPTypeCode.Boolean
                ? new Func<NDArray, NDArray, NDArray, NDArray>[] { (x, y, o) => np.add(x, y, o), (x, y, o) => np.multiply(x, y, o) }
                : dt == NPTypeCode.Complex
                    ? new Func<NDArray, NDArray, NDArray, NDArray>[] { (x, y, o) => np.add(x, y, o), (x, y, o) => np.subtract(x, y, o), (x, y, o) => np.multiply(x, y, o) }
                    : new Func<NDArray, NDArray, NDArray, NDArray>[] { (x, y, o) => np.add(x, y, o), (x, y, o) => np.subtract(x, y, o), (x, y, o) => np.multiply(x, y, o), (x, y, o) => np.maximum(x, y, o) };
            for (int k = 0; k < ops.Length; k++)
            {
                var op = ops[k];
                AssertRoutesAgree(o => op(a, b, o), () => np.zeros(new Shape(3, 5), dt), true, $"{dt} op{k} C");
                AssertRoutesAgree(o => op(aF, bF, o), () => np.asfortranarray(np.zeros(new Shape(3, 5), dt)), true, $"{dt} op{k} F");
                // A C out against F inputs is not one linear walk: the iterator route — except float16 add / subtract /
                // multiply (ops 0-2 here), whose strided widen-compute-narrow kernel reads any input layout into a C or F
                // out (plan api-parameter-parity X15; the iterator's float16 body is scalar).
                bool halfArith = dt == NPTypeCode.Half && k < 3;
                AssertRoutesAgree(o => op(aF, bF, o), () => np.zeros(new Shape(3, 5), dt), halfArith, $"{dt} op{k} F inputs, C out");
            }
        }
    }

    /// <summary>
    /// A size-1 input broadcast against an array (0-d, <c>(1,)</c>, <c>(1,1)</c>, on either side) is the trivial loop's
    /// scalar-broadcast path; a size-1 input of HIGHER rank than the output is not trivial and raises NumPy's error from
    /// the iterator route.
    /// </summary>
    [TestMethod]
    public void SizeOneInput_EitherSide_IsTheScalarBroadcastPath()
    {
        var x = np.arange(24.0).reshape(4, 6);
        foreach (var s in new[] { NDArray.Scalar(2.5), np.array(new[] { 2.5 }), np.array(new[] { 2.5 }).reshape(1, 1) })
        {
            AssertRoutesAgree(o => np.subtract(s, x, o), () => np.zeros(new Shape(4, 6)), true, $"scalar {s.Shape} - x");
            AssertRoutesAgree(o => np.subtract(x, s, o), () => np.zeros(new Shape(4, 6)), true, $"x - scalar {s.Shape}");
        }

        int before = DefaultEngine.TrivialBinaryOutRuns;
        var deep = np.array(new[] { 2.5 }).reshape(1, 1, 1);
        var error = Assert.ThrowsException<ArgumentException>(() => np.add(deep, x, np.zeros(new Shape(4, 6))),
            "a (1,1,1) input broadcasts the result to (1,4,6), which a (4,6) out cannot hold (NumPy's ValueError)");
        StringAssert.StartsWith(error.Message, "non-broadcastable output operand with shape (4,6)", "NumPy's text");
        Assert.AreEqual(before, DefaultEngine.TrivialBinaryOutRuns, "the higher-rank size-1 input is not trivial");
    }

    /// <summary>
    /// NaN operands keep NumPy's operand-order rule through the trivial loop: a float add / multiply with a NaN 0-d
    /// operand over 40 elements (past the AVX2 broadcast loop's two-vector blocks) and two NaN arrays with distinct
    /// payloads give the iterator route's bits and the fresh result's bits.
    /// </summary>
    [TestMethod]
    public void NaNOperands_KeepTheIteratorRoutesBits()
    {
        var nanA = new double[40];
        var nanB = new double[40];
        for (int i = 0; i < 40; i++)
        {
            nanA[i] = BitConverter.Int64BitsToDouble(0x7ff8000000000000L | (i + 1));
            nanB[i] = BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000000UL) | (i + 100));
        }
        var a = np.array(nanA);
        var b = np.array(nanB);
        var scalarNaN = NDArray.Scalar(BitConverter.Int64BitsToDouble(0x7ff80000000000aaL));
        var plain = np.arange(40.0);

        AssertRoutesAgree(o => np.add(a, b, o), () => np.zeros(new Shape(40)), true, "nan + nan");
        AssertRoutesAgree(o => np.multiply(a, b, o), () => np.zeros(new Shape(40)), true, "nan * nan");
        AssertRoutesAgree(o => np.add(scalarNaN, plain, o), () => np.zeros(new Shape(40)), true, "0-d nan + x");
        AssertRoutesAgree(o => np.multiply(a, scalarNaN, o), () => np.zeros(new Shape(40)), true, "nan * 0-d nan");

        var a32 = a.astype(np.float32);
        var s32 = NDArray.Scalar(BitConverter.Int32BitsToSingle(0x7fc000aa));
        AssertRoutesAgree(o => np.add(s32, a32, o), () => np.zeros(new Shape(40), np.float32), true, "f32 0-d nan + nan");

        // The trivial loop is the fresh result's kernel: same bits as np.add without out.
        var fresh = np.add(scalarNaN, plain);
        var into = np.zeros(new Shape(40));
        np.add(scalarNaN, plain, into);
        CollectionAssert.AreEqual(fresh.tobytes(), into.tobytes(), "out= and the fresh result are one kernel");
    }

    /// <summary>
    /// float16 add / subtract / multiply / divide into an out take the SIMD widen-compute-narrow kernels (the iterator
    /// route has only a per-element scalar body for float16) and write the fresh result's bits — the route that made the
    /// float16 length-1-axis sum 7x faster.
    /// </summary>
    [TestMethod]
    public void Float16Arithmetic_WritesTheFreshResultsBits()
    {
        var h = np.arange(1000).astype(np.float16) * NDArray.Scalar((Half)0.37);
        var g = np.arange(1000).astype(np.float16)["::-1"].copy() + NDArray.Scalar((Half)1);
        var zero = NDArray.Scalar(Half.NegativeZero);
        foreach (var (name, op) in new (string, Func<NDArray, NDArray, NDArray, NDArray>)[]
                 {
                     ("add", (x, y, o) => np.add(x, y, o)), ("subtract", (x, y, o) => np.subtract(x, y, o)),
                     ("multiply", (x, y, o) => np.multiply(x, y, o)), ("divide", (x, y, o) => np.divide(x, y, o)),
                 })
        {
            AssertRoutesAgree(o => op(h, g, o), () => np.zeros(new Shape(1000), np.float16), true, $"f16 {name}");
            AssertRoutesAgree(o => op(zero, h, o), () => np.zeros(new Shape(1000), np.float16), true, $"f16 -0 {name} x");
        }
    }

    /// <summary>
    /// An out that IS an input (the in-place <c>np.add(x, 1, out=x)</c>) is trivial and correct; an out that partly
    /// overlaps an input is not — a single forward pass would read elements it already wrote — and the iterator route's
    /// copy gives NumPy's values (each output from the ORIGINAL input). A size-1 input inside the output's memory
    /// declines too.
    /// </summary>
    [TestMethod]
    public void Overlap_InPlaceIsTrivial_PartialOverlapIsNot()
    {
        var x = np.arange(10.0);
        int before = DefaultEngine.TrivialBinaryOutRuns;
        np.add(x, NDArray.Scalar(1.0), x);
        Assert.AreEqual(before + 1, DefaultEngine.TrivialBinaryOutRuns, "in place is trivial");
        CollectionAssert.AreEqual(new[] { 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, x.ToArray<double>(), "in-place values");

        // out = x[1:], input x[:-1]: NumPy computes from the original x (COPY_IF_OVERLAP): [0, 0, 2, 4, ..., 16].
        var y = np.arange(10.0);
        before = DefaultEngine.TrivialBinaryOutRuns;
        np.multiply(y["0:9"], NDArray.Scalar(2.0), y["1:10"]);
        Assert.AreEqual(before, DefaultEngine.TrivialBinaryOutRuns, "a partial overlap is not trivial");
        CollectionAssert.AreEqual(new[] { 0.0, 0, 2, 4, 6, 8, 10, 12, 14, 16 }, y.ToArray<double>(), "partial-overlap values");

        // A size-1 input that is an element of the out.
        var z = np.arange(1.0, 6.0);
        before = DefaultEngine.TrivialBinaryOutRuns;
        np.multiply(z["0:1"], z, z);
        Assert.AreEqual(before, DefaultEngine.TrivialBinaryOutRuns, "a size-1 input inside the out is not trivial");
        CollectionAssert.AreEqual(new[] { 1.0, 2, 3, 4, 5 }, z.ToArray<double>(), "scalar-overlap values (z[0] is 1)");
    }

    /// <summary>
    /// A contiguous slice of a larger array as the out (a view whose first element is not the buffer's) receives the
    /// values at its own place and leaves the rest of the buffer untouched; a strided out, a broadcast array input and
    /// mixed dtypes take the iterator route.
    /// </summary>
    [TestMethod]
    public void OffsetOut_And_TheDeclines()
    {
        var big = np.full(new Shape(12), -1.0);
        var view = big["3:8"];
        var a = np.arange(5.0);
        int before = DefaultEngine.TrivialBinaryOutRuns;
        np.add(a, a, view);
        Assert.AreEqual(before + 1, DefaultEngine.TrivialBinaryOutRuns, "a contiguous slice out is trivial");
        CollectionAssert.AreEqual(new[] { -1.0, -1, -1, 0, 2, 4, 6, 8, -1, -1, -1, -1 }, big.ToArray<double>(), "slice placement");

        AssertRoutesAgree(o => np.add(a, a, o), () => np.zeros(new Shape(10))["::2"], false, "strided out");
        var row = np.arange(6.0).reshape(1, 6);
        var block = np.arange(24.0).reshape(4, 6);
        AssertRoutesAgree(o => np.add(row, block, o), () => np.zeros(new Shape(4, 6)), false, "broadcast row input");
        var ints = np.arange(5);
        AssertRoutesAgree(o => np.add(ints, a, o), () => np.zeros(new Shape(5)), false, "int32 + float64 into float64");
        AssertRoutesAgree(o => np.add(a, a, o), () => np.zeros(new Shape(5), np.float32), false, "float64 loop into a float32 out (cast)");
    }
}
