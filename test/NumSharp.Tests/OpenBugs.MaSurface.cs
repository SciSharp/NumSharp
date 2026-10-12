using System;
using System.Linq;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — the numpy.ma functions the coverage inventory listed as available but unproven
    // =====================================================================
    //
    // Found 2026-10-08 by the numpy.ma surface tier (test/oracle/gen_ma_surface_oracle.py, replayed by
    // NumSharp.Tests.Oracle/Fuzz/FuzzCorpusTests.MaSurface.cs): NumPy 2.4.2 contracts for the creation wrappers, the joins,
    // the fill-value helpers, the predicates and queries, ids, the hard-mask switches, masked_object, putmask, convolve /
    // correlate / cov / corrcoef, the edge finders, the compressors, ndenumerate and apply_along_axis / apply_over_axes.
    // Every divergence is carved into NumSharp.Tests.Oracle/Fuzz/corpus/gaps/ma_surface_gaps.jsonl under a gap id
    // (MS1-MS37, the GAPS table of the generator) and replayed by MaSurface_KnownGaps. This file reproduces the library
    // defects among them through PUBLIC calls, one test per gap:
    //
    //   * Silent wrong values: putmask / a hard-masked setitem cast unsafely (MS11, MS31); set_fill_value stores the raw
    //     value and common_fill_value compares raw values (MS12, MS13); masked_object never sets fill_value (MS7);
    //     zeros_like / ones_like / empty_like drop the mask (MS3); resize to 0-d keeps the mask (MS4); cov / corrcoef
    //     with a 1-D x and y under rowvar=False (MS37); corrcoef's masked-slot data (MS34); ids of a strided or reversed
    //     view (MS36); is_mask of a masked bool array (MS17); the bool extremum fill values (MS14).
    //   * Crashes or refusals where NumPy computes: corrcoef of one variable (MS33), convolve of a 0-d operand (MS35),
    //     frombuffer of a buffer whose length is not a multiple of the item size (MS22), notmasked_edges of a 0-d array
    //     with an axis (MS26), minimum_fill_value of a dtype (MS15).
    //   * Refusals NumPy makes and NumSharp does not, and error types / texts: MS6, MS8, MS21, MS23, MS24, MS27, MS28, MS30,
    //     MS32.
    //   * Reporting: the result FORM — an all-False mask dropped to nomask (MS1, MS2), masked_object's kept all-False mask
    //     (MS9); flatnotmasked_edges of an empty array (MS25).
    //
    // Not reproduced here (an API shape, not a defect of a call): MS5 (apply_over_axes returns a masked array where NumPy
    // returns a plain ndarray), MS10 (masked_object has no shrink=), MS16 (default_fill_value takes a DType only), MS18
    // (ma.size has no axis=), MS19 (ma.ndenumerate has no compressed=), MS20 (arange dtype= and a lone float stop, the
    // creation wrappers' order=, indices' sparse=), MS29 (compress_nd takes an int axis only).
    //
    // FIXED 2026-10-12, when branch strings was rebased onto journey4 (its numpy.ma work: the fill value stored as NumPy
    // stores it, masked_object's statements and shrink=, default_fill_value of any argument, putmask's same_kind cast, the
    // edge finders of an empty array, compress_nd's axis check, the products' 0-d / 2-D handling): MS3, MS6-MS14, MS16,
    // MS25, MS28 and MS35 match NumPy, and MS2 for column_stack / dstack / diagflat. Their pins below stay where they are
    // as regression tests, untagged, each summary opening with "FIXED"; MS2's open part (hsplit) has a pin of its own,
    // and the generator's rules for the fixed ids are dropped (MS35's remaining 0-d convolve cases are MS1's class).
    //
    // Every expected value below is NumPy 2.4.2's answer for the same call. EXPECTED (when fixed): each test passes;
    // remove its [OpenBugs], move it next to its API's tests, and drop the gap's rule from the generator's _gap_of so the
    // cases return to the gate.
    // =====================================================================
    public partial class OpenBugs
    {
        // ----- the result form: NumPy keeps an all-False mask array -----

        /// <summary>
        ///     <c>ma.convolve([1., 2., 3.], [1., 1.])</c>: NumPy's result is <c>masked_array(data, mask=mask)</c> over the
        ///     computed all-False mask, so <c>getmask</c> is a (4,) bool array; NumSharp drops it to nomask (gap MS1).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS1")]
        public void MaSurface_Convolve_KeepsAllFalseMaskArray()
        {
            var r = np.ma.convolve(np.ma.array(np.array(new[] { 1.0, 2.0, 3.0 })), np.ma.array(np.array(new[] { 1.0, 1.0 })));

            var m = np.ma.getmask(r);
            m.ndim.Should().Be(1);
            m.size.Should().Be(4);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.diagflat([1., 2.])</c> over a nomask operand: NumPy's _fromnxfunction builds the mask with getmaskarray,
        ///     so <c>getmask</c> is a (2, 2) all-False array; NumSharp keeps nomask (gap MS2).
        /// </summary>
        [TestMethod]
        public void MaSurface_Diagflat_NomaskOperand_GetsFullMask()
        {
            var r = np.ma.diagflat(np.ma.array(np.array(new[] { 1.0, 2.0 })));

            var m = np.ma.getmask(r);
            m.shape.Should().Equal(2L, 2L);
        }

        /// <summary>
        ///     <c>ma.hsplit(ma.array(np.arange(6.).reshape(2, 3)), 3)</c> over a nomask operand: NumPy's _fromnxfunction
        ///     builds each part's mask with getmaskarray, so every part's <c>getmask</c> is a (2, 1) all-False array;
        ///     NumSharp's parts keep nomask. The open part of gap MS2 since column_stack / dstack / diagflat were fixed.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS2")]
        public void MaSurface_Hsplit_NomaskOperand_PartsGetFullMasks()
        {
            var parts = np.ma.hsplit(np.ma.array(np.arange(6.0).reshape(2, 3)), 3);

            parts.Should().HaveCount(3);
            foreach (var part in parts)
            {
                var m = np.ma.getmask(part);
                ReferenceEquals(m, np.ma.nomask).Should().BeFalse();
                m.shape.Should().Equal(2L, 1L);
            }
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.zeros_like(ma.array([1., 2., 3.], mask=[0, 1, 0]))</c>: NumPy keeps a copy of the prototype's mask
        ///     (<c>[False, True, False]</c>); NumSharp's result has no mask (gap MS3).
        /// </summary>
        [TestMethod]
        public void MaSurface_ZerosLike_KeepsThePrototypesMask()
        {
            var a = np.ma.array(np.array(new[] { 1.0, 2.0, 3.0 }), np.array(new[] { false, true, false }));

            var r = np.ma.zeros_like(a);

            np.ma.getmaskarray(r).ToArray<bool>().Should().Equal(false, true, false);
        }

        /// <summary>
        ///     <c>ma.resize(ma.array([1.], mask=[True]), ())</c>: NumPy sets the mask only when the result has dimensions, so
        ///     the 0-d result is NOT masked; NumSharp resizes the mask too and returns a masked 0-d (gap MS4).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS4")]
        public void MaSurface_ResizeToZeroD_DropsTheMask()
        {
            var a = np.ma.array(np.array(new[] { 1.0 }), np.array(new[] { true }));

            var r = np.ma.resize(a, new Shape(Array.Empty<long>()));

            np.ma.is_masked(r).Should().BeFalse();
        }

        // ----- refusals and error texts -----

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.apply_along_axis(ma.max, 0, ma.array(uint8 [[1, 2], [3, 4]]))</c>: NumPy sets the result's fill value to
        ///     the default 999999, which does not fit uint8, and raises TypeError (a NumPy quirk NumSharp does not copy;
        ///     gap MS6).
        /// </summary>
        [TestMethod]
        public void MaSurface_ApplyAlongAxis_NarrowIntResult_RaisesNumPysFillValueTypeError()
        {
            var a = np.ma.array(np.array(new byte[,] { { 1, 2 }, { 3, 4 } }));
            Action act = () => np.ma.apply_along_axis(x => np.ma.max(x), 0, a);

            act.Should().Throw<TypeError>().WithMessage("Cannot convert fill_value 999999 to dtype uint8");
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.masked_object([1., 2., 1.], 1.).fill_value</c>: NumPy sets fill_value=value (1.0); NumSharp keeps the
        ///     default 1e20 (gap MS7).
        /// </summary>
        [TestMethod]
        public void MaSurface_MaskedObject_SetsTheFillValue()
        {
            var r = np.ma.masked_object(np.array(new[] { 1.0, 2.0, 1.0 }), 1.0);

            Convert.ToDouble(r.fill_value).Should().Be(1.0);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.masked_object(int8 [1, 2], 12345)</c>: NumPy raises TypeError (the fill value 12345 does not fit int8);
        ///     NumSharp computes (gap MS8).
        /// </summary>
        [TestMethod]
        public void MaSurface_MaskedObject_ValueOutOfRange_RaisesTypeError()
        {
            Action act = () => np.ma.masked_object(np.array(new sbyte[] { 1, 2 }), 12345);

            act.Should().Throw<TypeError>().WithMessage("Cannot convert fill_value 12345 to dtype int8");
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.masked_object([1., 2.], 5.)</c> over a plain array: nothing matches and NumPy's
        ///     <c>make_mask(shrink=True)</c> returns nomask; NumSharp keeps an all-False mask array (gap MS9).
        /// </summary>
        [TestMethod]
        public void MaSurface_MaskedObject_NoMatch_ShrinksToNomask()
        {
            var r = np.ma.masked_object(np.array(new[] { 1.0, 2.0 }), 5.0);

            ReferenceEquals(np.ma.getmask(r), np.ma.nomask).Should().BeTrue();
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this
        ///     test pins it — NumPy's text under np.copyto's house exception type for a cast refusal
        ///     (<see cref="InvalidCastException"/>, NumPy's TypeError; see np.copyto's divergences).
        ///     Before the fix: <c>ma.putmask(ma.array([1, 2, 3]), [True, False, False], 2.5)</c>: NumPy's
        ///     np.copyto(where=mask) casts same_kind and raises TypeError; NumSharp truncated 2.5 into the int64 array
        ///     (gap MS11).
        /// </summary>
        [TestMethod]
        public void MaSurface_Putmask_FloatIntoInt_RaisesSameKindTypeError()
        {
            var a = np.ma.array(np.array(new long[] { 1, 2, 3 }));
            Action act = () => np.ma.putmask(a, np.array(new[] { true, false, false }), 2.5);

            act.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast scalar from dtype('float64') to dtype('int64') according to the rule 'same_kind'");
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.set_fill_value(ma.array([1, 2, 3]), 1.5)</c>: NumPy converts the fill value to the dtype (int64 1);
        ///     NumSharp stores 1.5 (gap MS12).
        /// </summary>
        [TestMethod]
        public void MaSurface_SetFillValue_ConvertsToTheDtype()
        {
            var a = np.ma.array(np.array(new long[] { 1, 2, 3 }));

            np.ma.set_fill_value(a, 1.5);

            Convert.ToDouble(a.fill_value).Should().Be(1.0);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: Two float64 masked arrays with fill values set to 3 and 3.0: NumPy's common_fill_value compares the converted
        ///     fill values and returns 3.0; NumSharp compares the raw long 3 with the double 3.0 and returns null (gap MS13).
        /// </summary>
        [TestMethod]
        public void MaSurface_CommonFillValue_ComparesConvertedValues()
        {
            var a = np.ma.array(np.array(new[] { 1.0, 2.0 }));
            var b = np.ma.array(np.array(new[] { 0.0, 1.0 }));
            np.ma.set_fill_value(a, 3);
            np.ma.set_fill_value(b, 3.0);

            var common = np.ma.common_fill_value(a, b);

            common.Should().NotBeNull();
            Convert.ToDouble(common).Should().Be(3.0);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.maximum_fill_value(ma.array([True, False]))</c>: NumPy's extremum table holds the Python int 0 for bool
        ///     (so a bool array plus the fill promotes to int64); NumSharp returns the bool False (gap MS14).
        /// </summary>
        [TestMethod]
        public void MaSurface_MaximumFillValue_OfBool_IsTheIntZero()
        {
            var f = np.ma.maximum_fill_value(np.ma.array(np.array(new[] { true, false })));

            f.Should().NotBeOfType<bool>();
            Convert.ToInt64(f).Should().Be(0);
        }

        /// <summary>
        ///     <c>ma.minimum_fill_value(np.dtype('float64'))</c>: NumPy reads the dtype and returns inf; NumSharp throws
        ///     NotSupportedException (gap MS15).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS15")]
        public void MaSurface_MinimumFillValue_OfADtype_IsInfinity()
        {
            var f = np.ma.minimum_fill_value(np.dtype("float64"));

            Convert.ToDouble(f).Should().Be(double.PositiveInfinity);
        }

        /// <summary>
        ///     <c>ma.is_mask(ma.array([True, False]))</c>: NumPy's is_mask reads the dtype (bool) and returns True; NumSharp
        ///     accepts only a plain bool NDArray (gap MS17).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS17")]
        public void MaSurface_IsMask_OfAMaskedBoolArray_IsTrue()
        {
            np.ma.is_mask(np.ma.array(np.array(new[] { true, false }))).Should().BeTrue();
        }

        /// <summary>
        ///     <c>ma.arange(0, 10, 0)</c>: NumPy raises ZeroDivisionError; NumSharp raises ArgumentException("step can't be
        ///     0") (gap MS21).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS21")]
        public void MaSurface_Arange_ZeroStep_RaisesDivideByZero()
        {
            Action act = () => np.ma.arange(0, 10, 0);

            act.Should().Throw<DivideByZeroException>();
        }

        /// <summary>
        ///     <c>ma.frombuffer(bytes(24), dtype=complex128, count=0)</c>: NumPy checks the selected range and returns an
        ///     empty complex128 array; NumSharp rejects the whole 24-byte buffer as not a multiple of 16 (gap MS22).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS22")]
        public void MaSurface_Frombuffer_CountZero_OfANonMultipleBuffer_IsEmpty()
        {
            var r = np.ma.frombuffer(new byte[24], np.complex128, 0);

            r.size.Should().Be(0);
        }

        /// <summary>
        ///     <c>ma.nonzero</c> of a 0-d masked array: NumPy raises ValueError; NumSharp returns index arrays (gap MS23).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS23")]
        public void MaSurface_Nonzero_OfZeroD_RaisesValueError()
        {
            var a = np.ma.array(NDArray.Scalar(1.0));
            Action act = () => np.ma.nonzero(a);

            act.Should().Throw<ValueError>()
                .WithMessage("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.");
        }

        /// <summary>
        ///     <c>ma.clump_masked</c> of a 0-d array with a mask: NumPy's _ezclump indexes <c>mask[1:]</c> and raises
        ///     IndexError; NumSharp returns a list (gap MS24).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS24")]
        public void MaSurface_ClumpMasked_OfZeroDWithAMask_RaisesIndexError()
        {
            var a = np.ma.array(NDArray.Scalar(1.0), NDArray.Scalar(true));
            Action act = () => np.ma.clump_masked(a);

            act.Should().Throw<Exception>().Which.Message.Should()
                .Be("too many indices for array: array is 0-dimensional, but 1 were indexed");
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.flatnotmasked_edges</c> of an empty array: NumPy returns <c>array([0, -1])</c>
        ///     (<c>[0, a.size - 1]</c>); NumSharp returns null (gap MS25).
        /// </summary>
        [TestMethod]
        public void MaSurface_FlatnotmaskedEdges_OfEmpty_IsZeroMinusOne()
        {
            var e = np.ma.flatnotmasked_edges(np.ma.array(np.zeros(new Shape(0))));

            e.ToArray<long>().Should().Equal(0L, -1L);
        }

        /// <summary>
        ///     <c>ma.notmasked_edges(0-d, axis=0)</c>: NumPy returns <c>[(), ()]</c> (no dimension, no edge array);
        ///     NumSharp raises AxisError (gap MS26).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS26")]
        public void MaSurface_NotmaskedEdges_OfZeroDWithAnAxis_IsTwoEmptyTuples()
        {
            var r = np.ma.notmasked_edges(np.ma.array(NDArray.Scalar(1.0)), 0);

            r.Should().BeOfType<NDArray[][]>().Which.Select(t => t.Length).Should().Equal(0, 0);
        }

        /// <summary>
        ///     <c>ma.notmasked_contiguous(zeros((3, 4)), axis=2)</c>: NumPy raises IndexError <c>list assignment index out of
        ///     range</c> (its <c>idx[axis] = slice(None)</c>); NumSharp raises AxisError (gap MS27).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS27")]
        public void MaSurface_NotmaskedContiguous_AxisOutOfRange_IsNumPysIndexError()
        {
            var a = np.ma.array(np.zeros(new Shape(3, 4)));
            Action act = () => np.ma.notmasked_contiguous(a, 2);

            act.Should().Throw<Exception>().Which.Message.Should().Be("list assignment index out of range");
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.compress_nd(ma.array(arange(6.)), axis=3)</c>: NumPy normalizes the axis first and raises AxisError even
        ///     without a mask; NumSharp returns the data (gap MS28).
        /// </summary>
        [TestMethod]
        public void MaSurface_CompressNd_AxisOutOfRange_RaisesAxisError()
        {
            var a = np.ma.array(np.arange(6.0));
            Action act = () => np.ma.compress_nd(a, 3);

            act.Should().Throw<AxisError>().WithMessage("axis 3 is out of bounds for array of dimension 1*");
        }

        /// <summary>
        ///     <c>ma.hsplit(ma.array(arange(12.).reshape(3, 4)), [1, 3])</c>: NumPy's _fromnxfunction_single wraps the parts
        ///     (of widths 1, 2, 1) in one masked_array and raises the inhomogeneous-shape ValueError; NumSharp returns the
        ///     parts (gap MS30).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS30")]
        public void MaSurface_Hsplit_UnequalParts_RaisesNumPysInhomogeneousValueError()
        {
            var a = np.ma.array(np.arange(12.0).reshape(3, 4));
            Action act = () => np.ma.hsplit(a, new[] { 1, 3 });

            act.Should().Throw<ValueError>().WithMessage(
                "setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 " +
                "dimensions. The detected shape was (3, 3) + inhomogeneous part.*");
        }

        /// <summary>
        ///     A hard-masked bool array, <c>a[0:2] = 99</c>: NumPy's hard path is np.copyto(..., where=~mask) with
        ///     casting='same_kind' and raises TypeError; NumSharp writes True (gap MS31).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS31")]
        public void MaSurface_HardMaskSetitem_IntIntoBool_RaisesSameKindTypeError()
        {
            var a = np.ma.array(np.array(new[] { true, false, true }), np.array(new[] { false, true, false }));
            a.harden_mask();
            Action act = () => a["0:2"] = 99;

            act.Should().Throw<TypeError>()
                .WithMessage("Cannot cast scalar from dtype('int64') to dtype('bool') according to the rule 'same_kind'");
        }

        /// <summary>
        ///     A hard-masked (2, 3) array with an all-False mask, <c>a[1:3] = ma.array([5., 6.], mask=[1, 0])</c>: NumPy's
        ///     hard path raises <c>operands could not be broadcast together with shapes (1,3) (2,) </c>; NumSharp raises the
        ///     soft path's text (gap MS32).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS32")]
        public void MaSurface_HardMaskSetitem_BroadcastError_IsNumPysText()
        {
            var a = np.ma.array(np.arange(6.0).reshape(2, 3), np.zeros(new Shape(2, 3), np.@bool));
            a.harden_mask();
            var v = np.ma.array(np.array(new[] { 5.0, 6.0 }), np.array(new[] { true, false }));
            Action act = () => a["1:3"] = v;

            act.Should().Throw<Exception>().Which.Message.Should()
                .Be("operands could not be broadcast together with shapes (1,3) (2,) ");
        }

        // ----- products -----

        /// <summary>
        ///     <c>ma.corrcoef([1., 2., 3.])</c> (one variable): NumPy's covariance squeezes to 0-d, diagonal raises, and
        ///     corrcoef returns the masked singleton; NumSharp throws ArgumentException (gap MS33).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS33")]
        public void MaSurface_Corrcoef_OfOneVariable_IsTheMaskedSingleton()
        {
            var r = np.ma.corrcoef(np.ma.array(np.array(new[] { 1.0, 2.0, 3.0 })));

            ReferenceEquals(r, np.ma.masked).Should().BeTrue();
        }

        /// <summary>
        ///     <c>ma.corrcoef([[1, 2], [3, 5]] masked [[0, 1], [0, 1]], rowvar=False)</c>: NumPy's in-place
        ///     <c>corr /= outer(std, std)</c> divides a masked slot by 1, keeping cov's -0.0 there; NumSharp writes +0.0
        ///     (gap MS34).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS34")]
        public void MaSurface_Corrcoef_MaskedSlotKeepsCovsData()
        {
            var x = np.ma.array(np.array(new[,] { { 1.0, 2.0 }, { 3.0, 5.0 } }),
                                np.array(new[,] { { false, true }, { false, true } }));

            var r = np.ma.corrcoef(x, rowvar: false);

            BitConverter.DoubleToInt64Bits(r.data.GetDouble(0, 1)).Should().Be(unchecked((long)0x8000000000000000UL));
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>ma.convolve(ma.array(2.), [1., 2., 3.])</c>: NumPy reads the 0-d operand as one element and returns
        ///     <c>[2., 4., 6.]</c>; NumSharp throws IncorrectShapeException (gap MS35).
        /// </summary>
        [TestMethod]
        public void MaSurface_Convolve_OfAZeroDOperand_ReadsOneElement()
        {
            var r = np.ma.convolve(np.ma.array(NDArray.Scalar(2.0)), np.ma.array(np.array(new[] { 1.0, 2.0, 3.0 })));

            r.data.ToArray<double>().Should().Equal(2.0, 4.0, 6.0);
        }

        /// <summary>
        ///     <c>ma.ids(a[::-1])[0] - ma.ids(a)[0]</c> for an 8-element float64 array: NumPy's ids is the address of the
        ///     first element (56 bytes past the array's); NumSharp reports the buffer start (gap MS36).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS36")]
        public void MaSurface_Ids_OfAReversedView_IsTheFirstElementsAddress()
        {
            var a = np.ma.array(np.arange(8.0));
            var v = (NDMaskedArray)a["::-1"];

            (np.ma.ids(v).data - np.ma.ids(a).data).Should().Be(56);
        }

        /// <summary>
        ///     <c>ma.cov([1., 2., 3., 4.], [2., 1., 4., 3.], rowvar=False)</c>: NumPy forces rowvar=True for a one-row x
        ///     BEFORE concatenating y, so the covariance is 2x2; NumSharp decides after the concatenation (one variable of 8
        ///     observations, a 0-d result) (gap MS37).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "ma-surface-MS37")]
        public void MaSurface_Cov_OneDimensionalXAndY_RowvarFalse_IsTwoByTwo()
        {
            var r = np.ma.cov(np.ma.array(np.array(new[] { 1.0, 2.0, 3.0, 4.0 })),
                              np.ma.array(np.array(new[] { 2.0, 1.0, 4.0, 3.0 })), rowvar: false);

            r.shape.Should().Equal(2L, 2L);
        }
    }
}
