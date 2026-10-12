using System;
using System.Linq;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — the APIs the coverage inventory listed as available but unproven
    // =====================================================================
    //
    // Found 2026-10-08 by the surface tier (test/oracle/gen_surface_oracle.py, replayed by
    // NumSharp.Tests.Oracle/Fuzz/FuzzCorpusTests.Surface.cs): NumPy 2.4.2 contracts for the np.linalg Array-API forms,
    // the histogram family, the memory predicates, the text formatters, the ndarray members, the finfo / iinfo / dtype /
    // flags / poly1d attributes, the np.broadcast / flatiter / np.nditer members, the print and buffer options and the
    // cumulative_* aliases. Every divergence is carved into NumSharp.Tests.Oracle/Fuzz/corpus/gaps/surface_gaps.jsonl
    // under a gap id (SU1-SU42, the GAPS table of the generator) and replayed by Surface_KnownGaps. This file reproduces
    // the library defects among them through PUBLIC calls, one test per gap (two for SU42):
    //
    //   * Silent wrong values: np.cumsum / np.cumprod with dtype=complex128 over integers (SU38), np.cumsum with
    //     dtype=bool (SU39), np.finfo's float32 / float64 minexp, float32 resolution and tiny (SU12-SU14), np.nditer's K
    //     order over an F operand and a broadcast one (SU25) and after remove_multi_index (SU31), np.broadcast.iters
    //     after a step (SU20), the float formatters' inf / nan text (SU6) and rounding carries (SU18, SU19, SU9).
    //   * np.nditer's write-backs (the second pass): a readwrite updateifcopy operand of another dtype is refused without
    //     buffering, where NumPy iterates a copy that close() writes back (SU41); the iteration protocol publishes the
    //     element and THEN advances, where NumPy's __next__ advances on entry, so after k next() calls the cursor is one
    //     element ahead and a buffered readwrite loop loses its write into the last element of each buffer (SU42).
    //   * Crashes where NumPy computes: np.apply_along_axis over a stride-0 view at an offset (SU5), np.nditer.remove_axis
    //     under a non-C iteration order (SU30).
    //   * Refusals NumPy makes and NumSharp does not (or the reverse), and error texts: SU1, SU2, SU4, SU8, SU10, SU11,
    //     SU15, SU16, SU22, SU28, SU29, SU32-SU35, SU40.
    //   * Reporting: np.nditer.shape's axis order (SU26), index tracking blocking coalescing (SU27), get_printoptions'
    //     missing keys (SU21), edgeitems=0 summarization (SU23), dtype.newbyteorder's repr (SU17), the datetime
    //     dtype.type (SU24), complex histogram edges (SU3).
    //
    // Not reproduced here: SU7 (the float formatters take only a double — an API shape, not a defect of a call) and
    // SU36 / SU37 (numpy.cumulative_sum / cumulative_prod do not exist in NumSharp; the inventory aliases them to
    // np.cumsum / np.cumprod, whose own contract differs). Two pooled-buffer leaks the sweep found are pinned in the
    // oracle's KnownEscapes (np.MemoryView.tobytes of a non-contiguous array; np.array_equiv with an empty operand).
    //
    // Outside NumPy's contract, found by the harness itself: NDArray.SetAtIndex<T>(T, long) checks T against the dtype
    // only with a Debug.Assert, so in Release a double written into a float32 array lands as 8 bytes over the next two
    // elements (and past the buffer's end for the last one). The last test pins it; it is Inconclusive in Debug.
    //
    // FIXED 2026-10-12, when branch strings was rebased onto journey4 (its np.nditer work: NumPy's descriptors,
    // construction checks and iteration protocol; its axis normalization): SU1, SU25-SU34, SU40, SU41 and SU42 match
    // NumPy, and SU4 for the 'auto' / 'fd' estimators. Their pins stay below as regression tests, untagged, each summary
    // opening with "FIXED"; SU4's open part (the other estimators) has a pin of its own, and the generator's rules for the
    // fixed ids are dropped. The np.array_equiv leak named above went with the comparison operators' empty-left early
    // return (the sweep gates it at zero now).
    //
    // Every expected value below is NumPy 2.4.2's answer for the same call. EXPECTED (when fixed): each test passes;
    // remove its [OpenBugs], move it next to its API's tests, and drop the gap's rule from the generator's _gap_of so the
    // cases return to the gate.
    // =====================================================================
    public partial class OpenBugs
    {
        // ----- np.linalg Array-API forms -----

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.linalg.cross(a, b, axis=5)</c> on (2, 3) operands: NumPy raises IndexError <c>tuple index out of
        ///     range</c> (its <c>a.shape[axis]</c> lookup); NumSharp raises AxisError with its own text (gap SU1).
        /// </summary>
        [TestMethod]
        public void Surface_LinalgCross_AxisOutOfRange_IsNumPysIndexError()
        {
            using var a = np.arange(6.0).reshape(2, 3);
            Action act = () => np.linalg.cross(a, a, 5);

            act.Should().Throw<Exception>().Which.Message.Should().Be("tuple index out of range");
        }

        /// <summary>
        ///     <c>np.linalg.cross</c> of (4, 3) and (5, 3): NumPy raises broadcast_shapes' ValueError (gap SU2).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU2")]
        public void Surface_LinalgCross_NonBroadcastable_IsNumPysShapeMismatchText()
        {
            using var a = np.ones(new Shape(4, 3));
            using var b = np.ones(new Shape(5, 3));
            Action act = () => np.linalg.cross(a, b);

            act.Should().Throw<Exception>().Which.Message.Should().Be(
                "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape " +
                "(4,) and arg 1 with shape (5,).");
        }

        // ----- histogram -----

        /// <summary>
        ///     <c>np.histogram([1+1j, 2, 3], bins=2)</c>: NumPy's edges are complex128 <c>[1+1j, 2+0.5j, 3]</c>
        ///     (bin_type = result_type(first, last, a)); NumSharp's are float64 (gap SU3).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU3")]
        public void Surface_Histogram_ComplexSample_KeepsComplexEdges()
        {
            using var a = np.array(new Complex[] { new(1, 1), 2, 3 });
            var r = np.histogram(a, 2);

            r.bin_edges.typecode.Should().Be(NPTypeCode.Complex);
            r.bin_edges.ToArray<Complex>().Should().Equal(new Complex[] { new(1, 1), new(2, 0.5), 3 });
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.histogram(complex, bins="auto")</c>: NumPy raises TypeError <c>a must be an array of real numbers</c>;
        ///     NumSharp returns a result (gap SU4).
        /// </summary>
        [TestMethod]
        public void Surface_Histogram_ComplexSampleWithEstimator_Refuses()
        {
            using var a = np.array(new Complex[] { new(1, 1), 2 });
            Action act = () => np.histogram(a, "auto");

            act.Should().Throw<Exception>().Which.Message.Should().Be("a must be an array of real numbers");
        }

        /// <summary>
        ///     <c>np.histogram([1+1j, 2, 3], bins="sturges")</c>: NumPy raises TypeError <c>ufunc 'ceil' not supported for
        ///     the input types, ...</c> (the bin count's ceil has no complex loop); NumSharp returns a result. The open part
        ///     of gap SU4 since 'auto' / 'fd' were fixed: 'doane', 'scott', 'stone', 'rice', 'sturges' and 'sqrt' still
        ///     compute.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU4")]
        public void Surface_Histogram_ComplexSampleWithSturges_Refuses()
        {
            using var a = np.array(new Complex[] { new(1, 1), 2, 3 });
            Action act = () => np.histogram(a, "sturges");

            act.Should().Throw<Exception>().Which.Message.Should().StartWith("ufunc 'ceil' not supported for the input types");
        }

        // ----- apply_along_axis -----

        /// <summary>
        ///     <c>np.apply_along_axis(np.sum, 1, np.broadcast_to(np.split(arange(4.), [3])[1], (3, 4)))</c>: the one-element
        ///     split child keeps Shape.offset 3, so the broadcast is every stride 0 at offset 3. NumPy returns
        ///     <c>[12, 12, 12]</c>; NumSharp throws ArgumentOutOfRangeException (Parameter 'length') (gap SU5). A 0-d
        ///     index (<c>buf["3"]</c>) re-seats the address instead and does not reproduce it.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU5")]
        public void Surface_ApplyAlongAxis_StrideZeroViewAtAnOffset_Computes()
        {
            using var buf = np.arange(4.0);
            var pieces = np.split(buf, new[] { 3 });
            using var v = np.broadcast_to(pieces[1], new Shape(3, 4));
            using var r = np.apply_along_axis(row => np.sum(row), 1, v);

            r.ToArray<double>().Should().Equal(new[] { 12.0, 12.0, 12.0 });
        }

        // ----- the float formatters -----

        /// <summary>
        ///     <c>format_float_positional(inf)</c>, <c>(nan)</c>, <c>format_float_scientific(-inf)</c> and
        ///     <c>format_float_positional(inf, sign=True)</c>: NumPy prints <c>inf</c>, <c>nan</c>, <c>-inf</c>,
        ///     <c>+inf</c>; NumSharp prints .NET's <c>Infinity.</c> / <c>NaN.</c> / <c>I.nfinitye+07</c> (gap SU6).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU6")]
        public void Surface_FormatFloat_NonFinite_PrintsInfAndNan()
        {
            np.format_float_positional(double.PositiveInfinity).Should().Be("inf");
            np.format_float_positional(double.NaN).Should().Be("nan");
            np.format_float_scientific(double.NegativeInfinity).Should().Be("-inf");
            np.format_float_positional(double.PositiveInfinity, sign: true).Should().Be("+inf");
        }

        /// <summary>
        ///     <c>format_float_positional(1.5, precision=-1)</c>: NumPy raises ValueError <c>precision must be &gt;= 0</c>;
        ///     NumSharp formats (gap SU8).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU8")]
        public void Surface_FormatFloat_NegativePrecision_Refuses()
        {
            Action act = () => np.format_float_positional(1.5, precision: -1);

            act.Should().Throw<Exception>().Which.Message.Should().StartWith("precision must be >= 0");
        }

        /// <summary>
        ///     <c>format_float_positional(123456789.125, precision=3, fractional=False)</c>: precision counts significant
        ///     digits, so NumPy prints <c>123000000.</c>; NumSharp ignores fractional (gap SU9).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU9")]
        public void Surface_FormatFloatPositional_FractionalFalse_CountsSignificantDigits()
        {
            np.format_float_positional(123456789.125, precision: 3, fractional: false).Should().Be("123000000.");
        }

        /// <summary>
        ///     <c>format_float_positional(1.99999, precision=2)</c> and <c>format_float_scientific(0.0995,
        ///     precision=1)</c>: the cutoff carries through 9s and NumPy's Dragon4 drops the carried digits (<c>2.</c>,
        ///     <c>1.e-01</c>); NumSharp keeps them (<c>2.00</c>, <c>1.0e-01</c>) (gap SU18).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU18")]
        public void Surface_FormatFloat_CarryAtTheCutoff_DropsTheCarriedZeros()
        {
            np.format_float_positional(1.99999, precision: 2).Should().Be("2.");
            np.format_float_scientific(0.0995, precision: 1).Should().Be("1.e-01");
        }

        /// <summary>
        ///     <c>format_float_scientific(0.1001, precision=2, trim='-')</c>: NumPy keeps the point when the cutoff left
        ///     only zero fraction digits (<c>1.e-01</c>); NumSharp drops it (<c>1e-01</c>) (gap SU19).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU19")]
        public void Surface_FormatFloatScientific_TrimDash_KeepsThePointAfterZeroDigits()
        {
            np.format_float_scientific(0.1001, precision: 2, trim: '-').Should().Be("1.e-01");
        }

        // ----- ndarray.setfield -----

        /// <summary>
        ///     <c>np.arange(3, dtype=int32).setfield(-1, np.uint8, 1)</c>: NumPy raises OverflowError <c>Python integer -1
        ///     out of bounds for uint8</c>; NumSharp writes 255 (gap SU10).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU10")]
        public void Surface_Setfield_PythonIntOutOfFieldRange_Refuses()
        {
            using var a = np.arange(3).astype(np.int32);
            Action act = () => a.setfield(-1, np.uint8, 1);

            act.Should().Throw<OverflowException>().Which.Message.Should().Be("Python integer -1 out of bounds for uint8");
        }

        // ----- finfo / iinfo -----

        /// <summary>
        ///     <c>np.finfo(np.int32)</c>: NumPy's text names the scalar class, <c>data type &lt;class 'numpy.int32'&gt; not
        ///     inexact</c>; NumSharp's says <c>data type 'int32' not inexact</c> (gap SU11).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU11")]
        public void Surface_Finfo_NonInexactDtype_IsNumPysText()
        {
            Action act = () => np.finfo(np.int32);

            act.Should().Throw<Exception>().Which.Message.Should().StartWith("data type <class 'numpy.int32'> not inexact");
        }

        /// <summary>
        ///     <c>np.finfo(float32).minexp</c> is -126 and <c>np.finfo(float64).minexp</c> -1022 in NumPy; NumSharp's are one
        ///     higher (gap SU12).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU12")]
        public void Surface_Finfo_Minexp_IsTheSmallestNormalExponent()
        {
            np.finfo(np.float32).minexp.Should().Be(-126);
            np.finfo(np.float64).minexp.Should().Be(-1022);
        }

        /// <summary>
        ///     <c>np.finfo(float32).resolution</c> is <c>np.float32(1e-06)</c>, 9.999999974752427e-07 as a double, and
        ///     <c>np.finfo(float16).resolution</c> 0.0010004043579101562; NumSharp returns the decimal 1e-06 / 0.001 (gap
        ///     SU13).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU13")]
        public void Surface_Finfo_Resolution_IsRoundedToTheDtype()
        {
            np.finfo(np.float32).resolution.Should().Be(9.999999974752427e-07);
            np.finfo(np.float16).resolution.Should().Be(0.0010004043579101562);
        }

        /// <summary>
        ///     <c>np.finfo(float32).tiny</c> (and smallest_normal) is exactly 2**-126 = 1.1754943508222875e-38 in NumPy;
        ///     NumSharp's is the double literal 1.17549435e-38 (gap SU14).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU14")]
        public void Surface_Finfo_Float32Tiny_IsExactlyTwoToTheMinus126()
        {
            np.finfo(np.float32).tiny.Should().Be(System.Math.Pow(2, -126));
            np.finfo(np.float32).smallest_normal.Should().Be(System.Math.Pow(2, -126));
        }

        /// <summary>
        ///     <c>np.iinfo(np.bool_)</c>: NumPy raises ValueError <c>Invalid integer data type 'b'.</c>; NumSharp returns
        ///     an iinfo (gap SU15).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU15")]
        public void Surface_Iinfo_Bool_Refuses()
        {
            Action act = () => np.iinfo(np.@bool);

            act.Should().Throw<Exception>().Which.Message.Should().StartWith("Invalid integer data type 'b'.");
        }

        /// <summary>
        ///     <c>np.iinfo(np.float64)</c>: NumPy's text names the kind character and ends with a period, <c>Invalid integer
        ///     data type 'f'.</c>; NumSharp's names the dtype (gap SU16).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU16")]
        public void Surface_Iinfo_Float_IsNumPysText()
        {
            Action act = () => np.iinfo(np.float64);

            act.Should().Throw<Exception>().Which.Message.Should().StartWith("Invalid integer data type 'f'.");
        }

        // ----- dtype -----

        /// <summary>
        ///     <c>repr(np.dtype('&gt;i4').newbyteorder())</c> is <c>dtype('&lt;i4')</c> in NumPy (the swapped descriptor keeps
        ///     an explicit '&lt;'); NumSharp returns the canonical native descriptor, <c>dtype('int32')</c> (gap SU17).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU17")]
        public void Surface_DtypeNewbyteorder_KeepsTheExplicitByteOrder()
        {
            np.dtype(">i4").newbyteorder().ToString(true).Should().Be("dtype('<i4')");
        }

        /// <summary>
        ///     <c>np.dtype('M8[ms]').type</c> is <c>numpy.datetime64</c>; NumSharp's is null (no CLR element type before
        ///     Stage C) (gap SU24).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU24")]
        public void Surface_DtypeType_OfADatetimeDescriptor_IsNotNull()
        {
            np.dtype("M8[ms]").type.Should().NotBeNull();
        }

        /// <summary>
        ///     <c>np.dtype('cdouble')</c> is complex128 in NumPy; NumSharp raises <c>data type 'cdouble' not understood</c>
        ///     (gap SU35).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU35")]
        public void Surface_Dtype_CdoubleName_IsComplex128()
        {
            np.dtype("cdouble").typecode.Should().Be(NPTypeCode.Complex);
        }

        // ----- np.broadcast -----

        /// <summary>
        ///     <c>b = np.broadcast(arange(3), arange(2)[:, None])</c>, two steps taken: NumPy's iters share b's cursor and
        ///     yield the remaining <c>[2, 0, 1, 2]</c> / <c>[0, 1, 1, 1]</c>; NumSharp's start over with all six (gap SU20).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU20")]
        public void Surface_BroadcastIters_ShareTheBroadcastCursor()
        {
            using var a = np.arange(3);
            using var col = np.arange(2).reshape(2, 1);
            using var b = np.broadcast(a, col);
            b.MoveNext();
            b.MoveNext();

            b.iters[0].Select(Convert.ToInt64).Should().Equal(2L, 0L, 1L, 2L);
            b.iters[1].Select(Convert.ToInt64).Should().Equal(0L, 1L, 1L, 1L);
        }

        // ----- print options -----

        /// <summary>
        ///     <c>np.get_printoptions()</c> holds NumPy's 'formatter' and 'override_repr' keys (None by default); NumSharp's
        ///     dictionary lacks both (gap SU21).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU21")]
        public void Surface_GetPrintoptions_HasFormatterAndOverrideRepr()
        {
            var d = np.get_printoptions();

            d.Keys.OrderBy(k => k, StringComparer.Ordinal).Should().Equal(
                "edgeitems", "floatmode", "formatter", "infstr", "legacy", "linewidth", "nanstr", "override_repr",
                "precision", "sign", "suppress", "threshold");
        }

        /// <summary>
        ///     <c>np.set_printoptions(precision=-1)</c> then <c>np.array2string(np.arange(3))</c>: NumPy stores the value
        ///     and prints <c>[0 1 2]</c> (it raises only when a float is formatted); NumSharp refuses at once (gap SU22).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU22")]
        public void Surface_SetPrintoptions_NegativePrecision_IsStoredUntilAFloatPrints()
        {
            using var scope = np.printoptions();   // restores the previous options however the test ends
            using var a = np.arange(3);
            np.set_printoptions(precision: -1);

            np.array2string(a).Should().Be("[0 1 2]");
        }

        /// <summary>
        ///     <c>np.array2string([1/3, 1e-9, nan, inf, -2.5, 1e5], threshold=3, edgeitems=0)</c>: NumPy's
        ///     <c>_leading_trailing(a, 0)</c> takes <c>a[-0:]</c> (the whole array), so the format is scientific:
        ///     <c>[...  1.00000000e+05]</c>; NumSharp formats from the last element alone (gap SU23).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU23")]
        public void Surface_Array2string_EdgeitemsZero_FormatsFromTheWholeArray()
        {
            using var a = np.array(new[] { 1.0 / 3, 1e-9, double.NaN, double.PositiveInfinity, -2.5, 1e5 });

            np.array2string(a, threshold: 3, edgeitems: 0).Should().Be("[...  1.00000000e+05]");
        }

        // ----- np.nditer -----

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.nditer([x, col])</c> with x an F-ordered (2, 3) <c>arange(6)</c> and col a (2, 1) <c>[0, 1]</c>: the
        ///     broadcast col abstains from the K-order vote, so NumPy walks F order — (0, 0), (3, 1), (1, 0), (4, 1);
        ///     NumSharp walks C order (gap SU25).
        /// </summary>
        [TestMethod]
        public void Surface_Nditer_KOrder_FOperandWithABroadcastOne_WalksF()
        {
            using var c = np.arange(6).astype(np.int32).reshape(2, 3);
            using var x = np.asfortranarray(c);
            using var col = np.arange(2).astype(np.int16).reshape(2, 1);
            using var it = np.nditer(new[] { x, col });
            var seen = new (long, long)[4];
            for (int i = 0; i < 4; i++)
            {
                seen[i] = (Convert.ToInt64(it.value[0].GetAtIndex(0)), Convert.ToInt64(it.value[1].GetAtIndex(0)));
                it.iternext();
            }

            seen.Should().Equal((0L, 0L), (3L, 1L), (1L, 0L), (4L, 1L));
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.nditer(np.broadcast_to(arange(3), (2, 3))).shape</c> is <c>(3, 2)</c> in NumPy (NpyIter_GetShape lists
        ///     the axes innermost first); NumSharp lists them outermost first, <c>(2, 3)</c> (gap SU26).
        /// </summary>
        [TestMethod]
        public void Surface_NditerShape_WithoutMultiIndex_ListsTheInnermostAxisFirst()
        {
            using var row = np.arange(3);
            using var v = np.broadcast_to(row, new Shape(2, 3));
            using var it = np.nditer(v);

            it.shape.Should().Equal(3L, 2L);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.nditer(arange(6).reshape(2, 3), flags=['c_index'])</c> coalesces to a 1-D <c>(6,)</c> iterator in NumPy
        ///     (the C index stays linear); NumSharp keeps two axes (gap SU27).
        /// </summary>
        [TestMethod]
        public void Surface_Nditer_CIndex_StillCoalesces()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, new[] { "c_index" });

            it.ndim.Should().Be(1);
            it.shape.Should().Equal(6L);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: Reading <c>iterindex</c> after the last <c>iternext()</c>: NumPy raises ValueError <c>Iterator is past the
        ///     end</c>; NumSharp returns the size (gap SU28).
        /// </summary>
        [TestMethod]
        public void Surface_Nditer_PastTheEnd_Refuses()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a);
            for (int i = 0; i < 3; i++)
                it.iternext();
            Action act = () => _ = it.iterindex;

            act.Should().Throw<Exception>().Which.Message.Should().StartWith("Iterator is past the end");
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: After <c>close()</c>: NumPy's <c>iternext()</c> returns False and <c>ndim</c> raises <c>Iterator is
        ///     invalid</c>; NumSharp throws <c>Iterator is closed</c> for both (gap SU29).
        /// </summary>
        [TestMethod]
        public void Surface_Nditer_AfterClose_IternextIsFalse()
        {
            using var a = np.arange(3);
            var it = np.nditer(a);
            it.close();

            it.iternext().Should().BeFalse();
            Action ndim = () => _ = it.ndim;
            ndim.Should().Throw<Exception>().Which.Message.Should().StartWith("Iterator is invalid");
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.nditer(arange(6).reshape(2, 3), flags=['multi_index'], order='F').remove_axis(0)</c>: NumPy leaves a
        ///     1-D <c>(3,)</c> iterator; NumSharp throws IndexOutOfRangeException (gap SU30).
        /// </summary>
        [TestMethod]
        public void Surface_NditerRemoveAxis_UnderFOrder_Works()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, new[] { "multi_index" }, null, null, 'F');
            it.remove_axis(0);

            it.shape.Should().Equal(3L);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.nditer([arange(6).reshape(2, 3), arange(3)], flags=['multi_index'])</c> then
        ///     <c>remove_multi_index()</c>: NumPy keeps its C walk, 0, 1, 2, 3; NumSharp re-sorts the axes by stride and
        ///     moves the broadcast row's stride-0 axis innermost, walking 0, 3, 1, 4 (gap SU31).
        /// </summary>
        [TestMethod]
        public void Surface_NditerRemoveMultiIndex_KeepsTheIterationOrder()
        {
            using var a = np.arange(6).astype(np.int32).reshape(2, 3);
            using var row = np.arange(3).astype(np.int32);
            using var it = np.nditer(new[] { a, row }, new[] { "multi_index" });
            it.remove_multi_index();
            var seen = new long[4];
            for (int i = 0; i < 4; i++)
            {
                seen[i] = Convert.ToInt64(it.value[0].GetAtIndex(0));
                it.iternext();
            }

            seen.Should().Equal(0L, 1L, 2L, 3L);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.nditer(arange(4.), flags=['buffered', 'delay_bufalloc'], op_dtypes=['float32'],
        ///     casting='same_kind').value</c> before reset(): NumPy raises ValueError; NumSharp returns a value (gap SU32).
        /// </summary>
        [TestMethod]
        public void Surface_Nditer_DelayedBufferValueBeforeReset_Refuses()
        {
            using var a = np.arange(4.0);
            using var it = np.nditer(a, new[] { "buffered", "delay_bufalloc" }, null, new DType[] { np.float32 }, 'K',
                                     "same_kind");
            Action act = () => _ = it.value;

            act.Should().Throw<Exception>().Which.Message.Should().StartWith(
                "Iterator construction used delayed buffer allocation, and no reset has been done yet");
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>remove_axis</c> without <c>multi_index</c>: NumPy raises RuntimeError <c>Iterator RemoveAxis may only be
        ///     called if a multi-index is being tracked</c>; NumSharp accepts it (gap SU33).
        /// </summary>
        [TestMethod]
        public void Surface_NditerRemoveAxis_WithoutMultiIndex_Refuses()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a);
            Action act = () => it.remove_axis(0);

            act.Should().Throw<Exception>().Which.Message.Should().StartWith(
                "Iterator RemoveAxis may only be called if a multi-index is being tracked");
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>enable_external_loop()</c> while a multi-index is tracked: NumPy raises ValueError <c>Iterator flag
        ///     EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked</c>; NumSharp accepts it (gap SU34).
        /// </summary>
        [TestMethod]
        public void Surface_NditerEnableExternalLoop_WithMultiIndex_Refuses()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, new[] { "multi_index" });
            Action act = () => it.enable_external_loop();

            act.Should().Throw<Exception>().Which.Message.Should().StartWith(
                "Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked");
        }

        // ----- np.cumsum / np.cumprod -----

        /// <summary>
        ///     <c>np.cumsum([-3, 4, 1, 2] (int32), dtype=complex128)</c> is <c>[-3, 1, 2, 4]</c> and np.cumprod of it
        ///     <c>[-3, -12, -12, -24]</c>; NumSharp returns the int32 input's raw bytes read as complex values (gap SU38).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU38")]
        public void Surface_Cumsum_IntegerInputWithComplexDtype_CastsFirst()
        {
            using var a = np.array(new[] { -3, 4, 1, 2 });
            using var sum = np.cumsum(a, null, np.complex128);
            using var prod = np.cumprod(a, null, np.complex128);

            sum.ToArray<Complex>().Should().Equal(new Complex[] { -3, 1, 2, 4 });
            prod.ToArray<Complex>().Should().Equal(new Complex[] { -3, -12, -12, -24 });
        }

        /// <summary>
        ///     <c>np.cumsum([1, 1, 0, 2] (int32), dtype=bool)</c> is all True (NumPy's bool add is logical OR); NumSharp
        ///     stores the running counts 1, 2, 2, 3 into the bool result (gap SU39).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SU39")]
        public void Surface_Cumsum_BoolDtype_IsLogicalOr()
        {
            using var a = np.array(new[] { 1, 1, 0, 2 });
            using var r = np.cumsum(a, null, np.@bool);
            using var bytes = r.view(np.uint8);   // the raw bytes: a bool result holds 0 or 1 in every one

            bytes.ToArray<byte>().Should().Equal(new byte[] { 1, 1, 1, 1 });
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.cumprod(np.arange(3), axis=1)</c>: NumPy raises AxisError <c>axis 1 is out of bounds for array of
        ///     dimension 1</c>; NumSharp raises ArgumentOutOfRangeException with .NET's generic text (np.cumsum matches)
        ///     (gap SU40).
        /// </summary>
        [TestMethod]
        public void Surface_Cumprod_AxisOutOfRange_IsNumPysAxisError()
        {
            using var a = np.arange(3);
            Action act = () => np.cumprod(a, 1);

            act.Should().Throw<Exception>().Which.Message.Should().StartWith(
                "axis 1 is out of bounds for array of dimension 1");
        }

        // ----- np.nditer's write-backs (the second pass, 2026-10-08) -----

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.nditer(a_f32, op_flags=[['readwrite', 'updateifcopy']], op_dtypes=['float64'], casting='same_kind')</c>
        ///     without buffering: NumPy iterates a float64 COPY and close() writes it back (a = [1, 3, 5, 7, 9, 11] after
        ///     <c>x[...] = x * 2 + 1</c>); NumSharp refuses the cast ('Casting between different dtypes requires the BUFFERED
        ///     flag') (gap SU41).
        /// </summary>
        [TestMethod]
        public void Surface_Nditer_UpdateIfCopy_WritesBackAtClose()
        {
            using var a = np.arange(6).astype(np.float32);
            var it = np.nditer(a, op_flags: new[] { "readwrite", "updateifcopy" }, op_dtypes: new DType[] { np.float64 },
                               casting: "same_kind");
            foreach (var v in it)
                v[0].SetAtIndex(v[0].GetAtIndex<double>(0) * 2 + 1, 0);
            it.close();

            a.ToArray<float>().Should().Equal(1f, 3f, 5f, 7f, 9f, 11f);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: A buffered readwrite loop, <c>for x in np.nditer(a_f32, flags=['buffered'], op_flags=[['readwrite']],
        ///     op_dtypes=['float64'], casting='same_kind'): x[...] = x * 2 + 1</c>: NumPy's <c>__next__</c> advances ON ENTRY,
        ///     so every write lands before its buffer is flushed (a = [1, 3, 5, 7, 9, 11]); NumSharp publishes the element
        ///     and THEN advances, so the last element's buffer is flushed before the loop body writes it (a[5] stays 5)
        ///     (gap SU42).
        /// </summary>
        [TestMethod]
        public void Surface_Nditer_BufferedReadWriteLoop_KeepsTheLastWrite()
        {
            using var a = np.arange(6).astype(np.float32);
            using (var it = np.nditer(a, flags: new[] { "buffered" }, op_flags: new[] { "readwrite" },
                                      op_dtypes: new DType[] { np.float64 }, casting: "same_kind"))
            {
                foreach (var v in it)
                    v[0].SetAtIndex(v[0].GetAtIndex<double>(0) * 2 + 1, 0);
            }

            a.ToArray<float>().Should().Equal(1f, 3f, 5f, 7f, 9f, 11f);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: One step of the iteration protocol (NumPy's <c>next(it)</c>, C#'s MoveNext): NumPy's cursor is still at the
        ///     element it returned (iterindex 0); NumSharp's has already advanced (iterindex 1) (gap SU42).
        /// </summary>
        [TestMethod]
        public void Surface_Nditer_AfterOneNext_IterindexIsTheReturnedElement()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a);
            var en = it.GetEnumerator();
            en.MoveNext();

            it.iterindex.Should().Be(0);
        }

        /// <summary>
        ///     <c>NDArray.SetAtIndex&lt;T&gt;(T, long)</c> with T other than the array's element type: the type check is a
        ///     <c>Debug.Assert</c> only, so in Release <c>a.SetAtIndex(7.5, 1)</c> on a float32 array writes the double's 8
        ///     bytes at byte offset 8 — elements 2 and 3 — and leaves element 1 unchanged (for the last element the write
        ///     runs past the buffer's end). NumPy's <c>a.flat[1] = 7.5</c> casts the value into element 1. Found while building
        ///     the surface tier's write-back adapter; EXPECTED (when fixed): element 1 holds 7.5 (or the call refuses), and
        ///     the neighbours are untouched.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "surface-SetAtIndex")]
        public void Surface_SetAtIndexGeneric_WrongElementType_DoesNotWriteIntoTheNeighbours()
        {
#if DEBUG
            // The Debug.Assert fires in a Debug build (and may end the test host): the defect is the Release behaviour.
            Assert.Inconclusive("Release-only defect: in Debug the type mismatch trips SetAtIndex<T>'s Debug.Assert.");
#else
            using var a = np.zeros(new Shape(4), np.float32);
            try
            {
                a.SetAtIndex(7.5, 1);
            }
            catch (Exception)
            {
                // A refusal is an acceptable fix; the neighbours must still be untouched (checked below).
            }

            var v = a.ToArray<float>();
            v[2].Should().Be(0f);
            v[3].Should().Be(0f);
#endif
        }
    }
}
