using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Reproductions (then regression pins) for the <see cref="Generator"/> divergences found by the
    ///     2026-09-25 gap audit against NumPy 2.4.2. Every expected value and message below was produced by
    ///     running the same call on NumPy 2.4.2 (<c>numpy.random.default_rng(42)</c> unless stated), so a
    ///     failure here means NumSharp answers differently from NumPy — not that a test drifted.
    /// </summary>
    /// <remarks>
    ///     The groups mirror the audit's findings: bounded-integer range checks, the <c>size=0</c> and
    ///     <c>size=()</c> shortcuts, the sign-bit rule behind NumPy's <c>CONS_NON_NEGATIVE</c> constraint,
    ///     the <c>out=</c> contract of the distribution (<c>cont</c>) path, <c>choice</c>'s result shapes and
    ///     validation, the <c>AxisError</c> raised by the shuffling family, the memory-order walk of
    ///     <c>permuted</c>, and the per-bit-generator lock that makes a shared generator thread-safe.
    /// </remarks>
    [TestClass]
    public class GeneratorAuditFixesTest
    {
        /// <summary>The generator every case draws from: NumPy's <c>np.random.default_rng(42)</c>.</summary>
        /// <returns>A freshly seeded PCG64 generator.</returns>
        private static Generator R() => np.random.default_rng(42);

        /// <summary>Reads an integer-typed array of any rank in logical C order as <see cref="long"/> values.</summary>
        /// <param name="a">The array to read.</param>
        /// <returns>The values, widened to <see cref="long"/>.</returns>
        private static long[] L(NDArray a)
        {
            var flat = a.flat;
            var r = new long[flat.size];
            for (long i = 0; i < flat.size; i++) r[i] = Convert.ToInt64(flat.GetAtIndex(i));
            return r;
        }

        /// <summary>Reads a floating-point array of any rank in logical C order as <see cref="double"/> values.</summary>
        /// <param name="a">The array to read.</param>
        /// <returns>The values, widened to <see cref="double"/>.</returns>
        private static double[] D(NDArray a)
        {
            var flat = a.flat;
            var r = new double[flat.size];
            for (long i = 0; i < flat.size; i++) r[i] = Convert.ToDouble(flat.GetAtIndex(i));
            return r;
        }

        // =====================================================================================
        //  integers: NumPy's scalar path checks the INCLUSIVE high against the dtype's INCLUSIVE max
        // =====================================================================================

        /// <summary>
        ///     A <c>high</c> one past the dtype's exclusive limit must raise NumPy's range error. Before the
        ///     fix NumSharp compared the inclusive high with the exclusive limit, accepted the call, masked the
        ///     range to zero and silently returned a constant array (e.g. all <c>-128</c> for int8).
        /// </summary>
        /// <param name="dtype">NumPy dtype name.</param>
        /// <param name="low">Inclusive low.</param>
        /// <param name="high">Exclusive high — one past the largest legal value.</param>
        [TestMethod]
        [DataRow("bool", 0L, 3L)]
        [DataRow("int8", -128L, 129L)]
        [DataRow("uint8", 0L, 257L)]
        [DataRow("int16", -32768L, 32769L)]
        [DataRow("uint16", 0L, 65537L)]
        [DataRow("int32", -2147483648L, 2147483649L)]
        [DataRow("uint32", 0L, 4294967297L)]
        public void Integers_HighPastDtypeMax_Throws(string dtype, long low, long high)
        {
            Action act = () => R().integers(low, high, new Shape(6), np.dtype(dtype));
            act.Should().Throw<ValueError>().WithMessage($"high is out of bounds for {dtype}");
        }

        /// <summary>
        ///     With <c>endpoint=True</c> the same limit applies to the (already inclusive) high: <c>[0, 256]</c>
        ///     is out of range for uint8 in NumPy.
        /// </summary>
        /// <param name="dtype">NumPy dtype name.</param>
        /// <param name="low">Inclusive low.</param>
        /// <param name="high">Inclusive high — one past the largest legal value.</param>
        [TestMethod]
        [DataRow("bool", 0L, 2L)]
        [DataRow("int8", -128L, 128L)]
        [DataRow("uint8", 0L, 256L)]
        [DataRow("int16", -32768L, 32768L)]
        [DataRow("uint16", 0L, 65536L)]
        [DataRow("int32", -2147483648L, 2147483648L)]
        [DataRow("uint32", 0L, 4294967296L)]
        public void Integers_Endpoint_HighPastDtypeMax_Throws(string dtype, long low, long high)
        {
            Action act = () => R().integers(low, high, new Shape(6), np.dtype(dtype), endpoint: true);
            act.Should().Throw<ValueError>().WithMessage($"high is out of bounds for {dtype}");
        }

        /// <summary>
        ///     The full legal range of every narrow dtype (exclusive high = max + 1) keeps working and stays
        ///     byte-identical to NumPy after the bound fix.
        /// </summary>
        [TestMethod]
        public void Integers_FullNarrowRange_StillValid_ByteExact()
        {
            L(R().integers(0, 2, new Shape(6), np.bool_)).Should().Equal(0, 0, 0, 1, 0, 0);
            L(R().integers(-128, 128, new Shape(6), np.int8)).Should().Equal(8, -90, 89, -106, 77, 123);
            L(R().integers(0, 256, new Shape(6), np.uint8)).Should().Equal(136, 38, 217, 22, 205, 251);
            L(R().integers(-32768, 32768, new Shape(6), np.int16)).Should().Equal(-22904, -26919, 31693, 17953, 32705, 10129);
            L(R().integers(0, 65536, new Shape(6), np.uint16)).Should().Equal(9864, 5849, 64461, 50721, 65473, 42897);
            L(R().integers(-2147483648L, 2147483648L, new Shape(6), np.int32))
                .Should().Equal(-1764153720, 1176632269, 663879617, -262515103, -287697372, 1540166338);
            L(R().integers(0, 4294967296L, new Shape(6), np.uint32))
                .Should().Equal(383329928, 3324115917, 2811363265, 1884968545, 1859786276, 3687649986);
        }

        /// <summary>
        ///     NumPy returns an empty array for a zero-size request BEFORE it validates the bounds, so
        ///     <c>integers(10, 5, size=0)</c> is an empty int64 array, not <c>ValueError: low &gt;= high</c> —
        ///     and it consumes nothing from the stream.
        /// </summary>
        [TestMethod]
        public void Integers_ZeroSize_ReturnsEmptyBeforeBoundsCheck()
        {
            var e = R().integers(10, 5, new Shape(0));
            e.shape.Should().Equal(0L);
            e.dtype.Should().Be(np.int64);

            var e2 = R().integers(0, 1000, new Shape(2, 0), np.uint8);
            e2.shape.Should().Equal(2L, 0L);
            e2.dtype.Should().Be(np.uint8);

            var g = R();
            g.integers(10, 5, new Shape(0));
            L(g.integers(0, 100, new Shape(3))).Should().Equal(8, 77, 65);
        }

        /// <summary>
        ///     NumPy accepts the exclusive high <c>2**63</c> for int64 (the whole non-negative range). The
        ///     unsigned overload is the C# spelling for a high above <see cref="long.MaxValue"/>; before the
        ///     fix it rejected every non-uint64 dtype there.
        /// </summary>
        [TestMethod]
        public void Integers_Int64_ExclusiveHighTwoPow63_ByteExact()
        {
            const ulong twoPow63 = 9223372036854775808UL;
            L(R().integers(0UL, twoPow63, new Shape(3), np.int64))
                .Should().Equal(7138484576005690180, 4047939128787533792, 7919168045412322066);
            L(R().integers(twoPow63, null, new Shape(3), np.int64))
                .Should().Equal(7138484576005690180, 4047939128787533792, 7919168045412322066);
            var s = R().integers(4611686018427387904UL, twoPow63, dtype: np.int64);
            s.ndim.Should().Be(0);
            Convert.ToInt64(s.GetAtIndex(0)).Should().Be(8180928306430232994);

            Action over = () => R().integers(0UL, twoPow63 + 1, new Shape(3), np.int64);
            over.Should().Throw<ValueError>().WithMessage("high is out of bounds for int64");
        }

        /// <summary>
        ///     A non-native byte order is rejected: <c>integers</c> raises NumPy's <c>ValueError</c>, the float
        ///     fills raise <c>TypeError</c> naming the full dtype repr. An explicitly little-endian (native)
        ///     dtype is still accepted.
        /// </summary>
        [TestMethod]
        public void NonNativeByteOrder_Dtype_IsRejected()
        {
            Action i = () => R().integers(5, null, new Shape(3), np.dtype(">i8"));
            i.Should().Throw<ValueError>().WithMessage(
                "Providing a dtype with a non-native byteorder is not supported. If you require platform-independent byteorder, call byteswap when required.");

            L(R().integers(5, null, new Shape(3), np.dtype("<i8"))).Should().Equal(0, 3, 3);

            Action r = () => R().random(new Shape(3), np.dtype(">f8"));
            r.Should().Throw<TypeError>().WithMessage("Unsupported dtype dtype('>f8') for random");
            Action n = () => R().standard_normal(new Shape(3), np.dtype(">f8"));
            n.Should().Throw<TypeError>().WithMessage("Unsupported dtype dtype('>f8') for standard_normal");
            Action x = () => R().standard_exponential(new Shape(3), np.dtype(">f8"));
            x.Should().Throw<TypeError>().WithMessage("Unsupported dtype dtype('>f8') for standard_exponential");
            Action g = () => R().standard_gamma(2.0, new Shape(3), np.dtype(">f8"));
            g.Should().Throw<TypeError>().WithMessage("Unsupported dtype dtype('>f8') for standard_gamma");
        }

        // =====================================================================================
        //  CONS_NON_NEGATIVE tests the sign bit: -0.0 is "< 0", a NaN of either sign is not
        // =====================================================================================

        /// <summary>
        ///     NumPy's <c>CONS_NON_NEGATIVE</c> constraint is <c>not isnan(v) and signbit(v)</c>, so a
        ///     negative zero scale/shape/range is rejected. NumSharp tested <c>v &lt; 0</c>, which is false
        ///     for <c>-0.0</c>, and drew a value instead.
        /// </summary>
        [TestMethod]
        public void NegativeZero_Parameters_AreRejected()
        {
            ((Action)(() => R().normal(0.0, -0.0))).Should().Throw<ValueError>().WithMessage("scale < 0");
            ((Action)(() => R().normal(0.0, -0.0, new Shape(3)))).Should().Throw<ValueError>().WithMessage("scale < 0");
            ((Action)(() => R().exponential(-0.0))).Should().Throw<ValueError>().WithMessage("scale < 0");
            ((Action)(() => R().gamma(-0.0))).Should().Throw<ValueError>().WithMessage("shape < 0");
            ((Action)(() => R().gamma(1.0, -0.0))).Should().Throw<ValueError>().WithMessage("scale < 0");
            ((Action)(() => R().standard_gamma(-0.0))).Should().Throw<ValueError>().WithMessage("shape < 0");
            ((Action)(() => R().standard_gamma(-0.0, dtype: np.float32))).Should().Throw<ValueError>().WithMessage("shape < 0");
            ((Action)(() => R().uniform(0.0, -0.0))).Should().Throw<ValueError>().WithMessage("high - low < 0");
        }

        /// <summary>
        ///     A NaN parameter — even one whose sign bit is set — passes the constraint (NumPy excludes NaN
        ///     before looking at the sign) and propagates into the draw.
        /// </summary>
        [TestMethod]
        public void NegativeNaN_Scale_PassesTheConstraint()
        {
            double negNaN = BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000000UL));
            var v = R().normal(0.0, negNaN);
            double.IsNaN(Convert.ToDouble(v.GetAtIndex(0))).Should().BeTrue();
        }

        // =====================================================================================
        //  standard_gamma's out= rides NumPy's cont()/cont_f(): C-contiguous only, validated first
        // =====================================================================================

        /// <summary>
        ///     <c>standard_gamma</c> goes through NumPy's <c>cont</c>/<c>cont_f</c>, whose <c>check_output</c>
        ///     demands a C-contiguous <c>out</c> (the plain fills accept F). Both dtypes must reject an
        ///     F-contiguous out with the "C-contiguous" message.
        /// </summary>
        [TestMethod]
        public void StandardGamma_Out_FContiguous_IsRejected()
        {
            Action f64 = () => R().standard_gamma(2.0, @out: np.empty(new Shape(2, 3), np.float64).T);
            f64.Should().Throw<ValueError>().WithMessage(
                "Supplied output array must be C-contiguous, writable, aligned, and in machine byte-order.");
            Action f32 = () => R().standard_gamma(2.0, dtype: np.float32, @out: np.empty(new Shape(2, 3), np.float32).T);
            f32.Should().Throw<ValueError>().WithMessage(
                "Supplied output array must be C-contiguous, writable, aligned, and in machine byte-order.");
        }

        /// <summary>
        ///     NumPy validates <c>out</c> before the shape constraint: a negative shape with a wrong-dtype out
        ///     reports the out error; with a valid out it reports <c>shape &lt; 0</c>.
        /// </summary>
        [TestMethod]
        public void StandardGamma_OutIsValidatedBeforeShape()
        {
            Action wrong = () => R().standard_gamma(-1.0, @out: np.empty(new Shape(3), np.float32));
            wrong.Should().Throw<TypeError>().WithMessage("Supplied output array has the wrong type. Expected float64, got float32");
            Action ok = () => R().standard_gamma(-1.0, @out: np.empty(new Shape(3), np.float64));
            ok.Should().Throw<ValueError>().WithMessage("shape < 0");
        }

        /// <summary>A C-contiguous <c>out</c> still fills byte-identically to NumPy.</summary>
        [TestMethod]
        public void StandardGamma_Out_CContiguous_ByteExact()
        {
            var o = np.empty(new Shape(2, 3), np.float64);
            var r = R().standard_gamma(2.0, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            D(o).Should().Equal(2.0918172704999494, 2.8353455897858653, 1.8372155803071377,
                1.6450704225011006, 3.0792553295549006, 1.7533735410905071);
        }

        // =====================================================================================
        //  size=() is a 0-d ARRAY request (keeps float32); size=None is the scalar request
        // =====================================================================================

        /// <summary>
        ///     NumPy widens a float32 draw to a Python float only for <c>size=None</c>; <c>size=()</c> asks for
        ///     a 0-d array and keeps float32. NumSharp treated <c>Shape.Scalar</c> like "no size" and returned
        ///     float64.
        /// </summary>
        [TestMethod]
        public void EmptyTupleSize_Float32_Stays0dFloat32()
        {
            var a = R().random(Shape.Scalar, np.float32);
            a.ndim.Should().Be(0);
            a.dtype.Should().Be(np.float32);
            ((double)(float)a.GetAtIndex(0)).Should().Be(0.08925092220306396);

            var n = R().standard_normal(Shape.Scalar, np.float32);
            n.dtype.Should().Be(np.float32);
            ((double)(float)n.GetAtIndex(0)).Should().Be(0.14190717041492462);

            var e = R().standard_exponential(Shape.Scalar, np.float32);
            e.dtype.Should().Be(np.float32);
            ((double)(float)e.GetAtIndex(0)).Should().Be(0.08859460800886154);

            var ei = R().standard_exponential(Shape.Scalar, np.float32, "inv");
            ei.dtype.Should().Be(np.float32);
            ((double)(float)ei.GetAtIndex(0)).Should().Be(0.09348785877227783);

            var g = R().standard_gamma(2.5, Shape.Scalar, np.float32);
            g.dtype.Should().Be(np.float32);
            ((double)(float)g.GetAtIndex(0)).Should().Be(2.3823325634002686);

            // size=None (the default Shape) still widens to float64, as NumPy's float_fill does.
            R().random(dtype: np.float32).dtype.Should().Be(np.float64);
        }

        // =====================================================================================
        //  choice: result shapes, population validation, p tolerance/type checks
        // =====================================================================================

        /// <summary>
        ///     <c>choice(a)</c> with <c>size=None</c> is <c>a.take(scalar_index, axis)</c> in NumPy: a single
        ///     element for 1-D <c>a</c> (a 0-d array here) and one sub-array for N-D <c>a</c>. NumSharp took a
        ///     1-element index and returned <c>(1,)</c> / <c>(1, cols)</c>.
        /// </summary>
        [TestMethod]
        public void Choice_SizeNone_ReturnsScalarOrSubArray()
        {
            var one = R().choice(np.array(new long[] { 10, 20, 30 }));
            one.ndim.Should().Be(0);
            Convert.ToInt64(one.GetAtIndex(0)).Should().Be(10);

            var row = R().choice(np.arange(6).reshape(3, 2));
            row.shape.Should().Equal(2L);
            L(row).Should().Equal(0, 1);

            var col = R().choice(np.arange(6).reshape(3, 2), axis: 1);
            col.shape.Should().Equal(3L);
            L(col).Should().Equal(0, 2, 4);

            var noRep = R().choice(np.array(new long[] { 10, 20, 30 }), replace: false);
            noRep.ndim.Should().Be(0);
            Convert.ToInt64(noRep.GetAtIndex(0)).Should().Be(10);

            var weighted = R().choice(np.array(new long[] { 10, 20, 30 }), p: np.array(new[] { 0.2, 0.3, 0.5 }));
            weighted.ndim.Should().Be(0);
            Convert.ToInt64(weighted.GetAtIndex(0)).Should().Be(30);
        }

        /// <summary>
        ///     <c>size=()</c> asks for a 0-d array: a 0-d element for 1-D <c>a</c> and the sub-array for N-D
        ///     <c>a</c> (NumPy's <c>a.take(0-d index)</c>).
        /// </summary>
        [TestMethod]
        public void Choice_SizeEmptyTuple_Shapes()
        {
            var one = R().choice(np.array(new long[] { 10, 20, 30 }), Shape.Scalar);
            one.ndim.Should().Be(0);
            Convert.ToInt64(one.GetAtIndex(0)).Should().Be(10);

            var row = R().choice(np.arange(6).reshape(3, 2), Shape.Scalar);
            row.shape.Should().Equal(2L);
            L(row).Should().Equal(0, 1);
        }

        /// <summary>
        ///     A 0-d population must be an integer: NumPy's <c>operator.index(a.item())</c> fails for a float
        ///     or complex scalar and raises <c>ValueError</c>. NumSharp rounded 3.5 to a population of 4.
        ///     A 0-d bool is an integer (population 1), as in Python.
        /// </summary>
        [TestMethod]
        public void Choice_ZeroDimPopulation_MustBeInteger()
        {
            const string msg = "a must be a sequence or an integer, not <class 'numpy.ndarray'>";
            ((Action)(() => R().choice(NDArray.Scalar(3.5), new Shape(2)))).Should().Throw<ValueError>().WithMessage(msg);
            ((Action)(() => R().choice(NDArray.Scalar(new System.Numerics.Complex(3, 0)), new Shape(2))))
                .Should().Throw<ValueError>().WithMessage(msg);

            var b = R().choice(NDArray.Scalar(true));
            Convert.ToInt64(b.GetAtIndex(0)).Should().Be(0);
            L(R().choice(NDArray.Scalar(4L), new Shape(3))).Should().Equal(0, 3, 2);
        }

        /// <summary>
        ///     NumPy widens the sum-to-one tolerance to <c>sqrt(eps)</c> of a floating <c>p</c>'s own dtype
        ///     (float32: ~3.45e-4, float16: 0.03125), so single/half-precision probabilities are accepted.
        ///     A float64 <c>p</c> with the same error is still rejected.
        /// </summary>
        [TestMethod]
        public void Choice_P_ToleranceFollowsPDtype()
        {
            L(R().choice(4, new Shape(3), p: np.array(new[] { 0.25f, 0.25f, 0.25f, 0.25002f }))).Should().Equal(3, 1, 3);
            L(R().choice(3, new Shape(3), p: np.array(new[] { (Half)0.3, (Half)0.3, (Half)0.41 }))).Should().Equal(2, 1, 2);

            Action f64 = () => R().choice(4, new Shape(3), p: np.array(new[] { 0.25, 0.25, 0.25, 0.25002 }));
            f64.Should().Throw<ValueError>().WithMessage(
                "Probabilities do not sum to 1. See Notes section of docstring for more information.");
        }

        /// <summary>
        ///     NumPy evaluates <c>len(p)</c> first, so a 0-d <c>p</c> is a <c>TypeError</c>, and it reads the
        ///     population with <c>a.shape[axis]</c>, so an out-of-range axis is an <c>IndexError</c>.
        /// </summary>
        [TestMethod]
        public void Choice_ErrorTypes_MatchNumPy()
        {
            ((Action)(() => R().choice(3, p: NDArray.Scalar(1.0)))).Should().Throw<TypeError>().WithMessage("len() of unsized object");
            ((Action)(() => R().choice(np.arange(5), new Shape(2), axis: 1))).Should().Throw<IndexError>().WithMessage("tuple index out of range");
            ((Action)(() => R().choice(np.arange(5), new Shape(2), axis: -2))).Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        // =====================================================================================
        //  shuffle / permutation / permuted raise AxisError (NumPy's normalize_axis_index)
        // =====================================================================================

        /// <summary>
        ///     The shuffling family normalizes its axis with NumPy's <c>normalize_axis_index</c>, which raises
        ///     <c>AxisError</c> (NumSharp threw a bare <see cref="ArgumentException"/> with the same text).
        /// </summary>
        [TestMethod]
        public void ShuffleFamily_BadAxis_RaisesAxisError()
        {
            ((Action)(() => R().shuffle(np.arange(6).reshape(2, 3), 2))).Should().Throw<AxisError>()
                .WithMessage("axis 2 is out of bounds for array of dimension 2*");
            ((Action)(() => R().shuffle(np.arange(6).reshape(2, 3), -3))).Should().Throw<AxisError>()
                .WithMessage("axis -3 is out of bounds for array of dimension 2*");
            ((Action)(() => R().permutation(NDArray.Scalar(5L)))).Should().Throw<AxisError>()
                .WithMessage("axis 0 is out of bounds for array of dimension 0*");
            ((Action)(() => R().permutation(np.arange(6).reshape(2, 3), 2))).Should().Throw<AxisError>()
                .WithMessage("axis 2 is out of bounds for array of dimension 2*");
            ((Action)(() => R().permuted(np.arange(6).reshape(2, 3), 5))).Should().Throw<AxisError>()
                .WithMessage("axis 5 is out of bounds for array of dimension 2*");
            ((Action)(() => R().permuted(NDArray.Scalar(5L), 0))).Should().Throw<AxisError>()
                .WithMessage("axis 0 is out of bounds for array of dimension 0*");
        }

        // =====================================================================================
        //  permuted(axis=None) shuffles the output in MEMORY order; out= casts with casting='safe'
        // =====================================================================================

        /// <summary>
        ///     NumPy copies <c>x</c> with <c>order='K'</c> and shuffles <c>out.ravel(order='A')</c> — the
        ///     memory order of a C- or F-contiguous output — so an F-ordered input (or output) permutes a
        ///     different sequence than its C-order flattening. NumSharp always shuffled C order.
        /// </summary>
        [TestMethod]
        public void Permuted_AxisNone_ShufflesMemoryOrder()
        {
            var xF = np.asfortranarray(np.arange(12).reshape(3, 4));
            var r = R().permuted(xF);
            r.Shape.IsFContiguous.Should().BeTrue();
            L(r).Should().Equal(0, 3, 9, 7, 6, 11, 8, 4, 2, 1, 5, 10);

            var outC = np.zeros(new Shape(3, 4), np.int64);
            L(R().permuted(xF, null, outC)).Should().Equal(0, 7, 6, 9, 11, 3, 5, 2, 4, 10, 1, 8);

            var outF = np.asfortranarray(np.zeros(new Shape(3, 4), np.int64));
            L(R().permuted(np.arange(12).reshape(3, 4), null, outF)).Should().Equal(0, 3, 9, 7, 6, 11, 8, 4, 2, 1, 5, 10);
        }

        /// <summary>
        ///     A non-contiguous <c>out</c> is shuffled through a C-contiguous write-back copy (C order), and a
        ///     non-contiguous <c>x</c> is copied with <c>order='K'</c> first, so its stride order decides.
        /// </summary>
        [TestMethod]
        public void Permuted_AxisNone_NonContiguousLayouts()
        {
            var baseArr = np.zeros(new Shape(3, 8), np.int64);
            var outStrided = baseArr[":, ::2"];
            R().permuted(np.arange(12).reshape(3, 4), null, outStrided);
            L(baseArr).Should().Equal(0, 0, 7, 0, 6, 0, 9, 0, 11, 0, 3, 0, 5, 0, 2, 0, 4, 0, 10, 0, 1, 0, 8, 0);

            var cOrdered = R().permuted(np.arange(24).reshape(4, 6)[":, ::2"]);
            cOrdered.Shape.IsContiguous.Should().BeTrue();
            L(cOrdered).Should().Equal(0, 14, 12, 18, 22, 6, 10, 4, 8, 20, 2, 16);

            var fOrdered = R().permuted(np.arange(24).reshape(6, 4).T["::2"]);
            fOrdered.Shape.IsFContiguous.Should().BeTrue();
            L(fOrdered).Should().Equal(0, 12, 22, 10, 8, 2, 14, 18, 6, 4, 20, 16);
        }

        /// <summary>
        ///     <c>permuted</c> fills <c>out</c> with <c>np.copyto(out, x, casting='safe')</c>: a narrowing or
        ///     float-to-int out is a <c>TypeError</c>; a safe widening (int64 → float64) is accepted.
        /// </summary>
        [TestMethod]
        public void Permuted_Out_UsesSafeCasting()
        {
            Action narrow = () => R().permuted(np.arange(6).astype(np.int32), null, np.empty(new Shape(6), np.int16));
            narrow.Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('int32') to dtype('int16') according to the rule 'safe'");
            Action f2i = () => R().permuted(np.arange(6).astype(np.float64), null, np.empty(new Shape(6), np.int64));
            f2i.Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'");

            var o = np.empty(new Shape(6), np.float64);
            D(R().permuted(np.arange(6), null, o)).Should().Equal(3.0, 2.0, 5.0, 4.0, 1.0, 0.0);
        }

        // =====================================================================================
        //  Thread safety: every draw holds the bit generator's lock (NumPy BitGenerator.lock)
        // =====================================================================================

        /// <summary>
        ///     NumPy serializes every fill on <c>BitGenerator.lock</c>, so threads sharing one generator jointly
        ///     consume exactly the sequential stream. Without the lock the 128-bit PCG64 state tears and most
        ///     values fall outside the stream (measured: 72–99% before the fix).
        /// </summary>
        [TestMethod]
        public void SharedGenerator_ConcurrentDraws_ConsumeTheSequentialStream()
        {
            const int threads = 4, per = 50_000;
            var reference = L(np.random.default_rng(7).integers(0, 1L << 62, new Shape(threads * per)));
            Array.Sort(reference);

            var shared = np.random.default_rng(7);
            var parts = new long[threads][];
            Parallel.For(0, threads, t => parts[t] = L(shared.integers(0, 1L << 62, new Shape(per))));

            var union = parts.SelectMany(p => p).ToArray();
            Array.Sort(union);
            union.Should().Equal(reference);
        }

        /// <summary>
        ///     Scalar draws are individually atomic too: many threads calling <c>random()</c> on one generator
        ///     produce exactly the values of the sequential stream (as a multiset).
        /// </summary>
        [TestMethod]
        public void SharedGenerator_ConcurrentScalarDraws_ConsumeTheSequentialStream()
        {
            const int total = 20_000;
            var reference = D(np.random.default_rng(11).random(new Shape(total)));
            Array.Sort(reference);

            var shared = np.random.default_rng(11);
            var got = new double[total];
            Parallel.For(0, total, i => got[i] = Convert.ToDouble(shared.random().GetAtIndex(0)));
            Array.Sort(got);
            got.Should().Equal(reference);
        }

        /// <summary>
        ///     Two generators over the SAME bit generator share its lock (NumPy's <c>Generator.lock</c> is the
        ///     bit generator's), so their concurrent draws also partition one sequential stream.
        /// </summary>
        [TestMethod]
        public void TwoGeneratorsOverOneBitGenerator_ShareTheLock()
        {
            const int per = 50_000;
            var reference = L(np.random.default_rng(3).integers(0, 1L << 62, new Shape(2 * per)));
            Array.Sort(reference);

            var bg = new PCG64(3);
            var g1 = new Generator(bg);
            var g2 = new Generator(bg);
            long[] a = null, b = null;
            Parallel.Invoke(() => a = L(g1.integers(0, 1L << 62, new Shape(per))),
                            () => b = L(g2.integers(0, 1L << 62, new Shape(per))));
            var union = a.Concat(b).ToArray();
            Array.Sort(union);
            union.Should().Equal(reference);
        }

        // =====================================================================================
        //  SeedSequence.generate_state validates n_words like np.zeros
        // =====================================================================================

        /// <summary>
        ///     <c>generate_state(-1)</c> raises NumPy's <c>negative dimensions are not allowed</c> (the
        ///     <c>np.zeros(n_words)</c> it allocates); <c>generate_state(0)</c> is an empty result.
        /// </summary>
        [TestMethod]
        public void SeedSequence_GenerateState_NegativeWords_IsValueError()
        {
            ((Action)(() => new SeedSequence(0).generate_state(-1))).Should().Throw<ValueError>()
                .WithMessage("negative dimensions are not allowed");
            new SeedSequence(0).generate_state(0).size.Should().Be(0);
        }
    }
}
