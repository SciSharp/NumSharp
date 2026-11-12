using System;
using System.Linq;
using System.Numerics;
using System.Reflection;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 5 (2026-11-12, docs/plans/API_PARITY_DIFF.md): NumPy's TUPLE axis for np.sum, np.prod, np.mean,
    ///     np.var, np.std, np.max / np.amax, np.min / np.amin, the nan family (nansum, nanprod, nanmean, nanvar, nanstd,
    ///     nanmax, nanmin) and their ndarray / numpy.matrix methods; ONE axis signature for np.argmax / np.argmin and the
    ///     ndarray methods (a tuple is NumPy's TypeError); ONE np.roll signature with an array shift and an axis tuple; and
    ///     <c>size=None</c> as <c>Shape? size = null</c> on the Generator samplers. Every expected value, bit pattern or text
    ///     is NumPy 2.4.2's answer for the same call (probed 2026-11-12: batch5_numpy_evidence.py, batch5_edges.py,
    ///     batch5_matrix_tuple.py, batch5_bits.py).
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch5Tests
    {
        /// <summary><c>np.arange(24.0).reshape(2, 3, 4)</c>, the evidence scripts' <c>a</c> / <c>c3</c>.</summary>
        private static NDArray A => np.arange(24.0).reshape(2, 3, 4);

        /// <summary><see cref="A"/> with a NaN at <c>[0, 0, 0]</c>, the evidence's <c>an</c>.</summary>
        private static NDArray AN
        {
            get
            {
                var an = A;
                an[0, 0, 0] = double.NaN;
                return an;
            }
        }

        /// <summary><see cref="A"/> with its whole <c>[:, 0, :]</c> slab NaN, batch5_edges.py's nan-family <c>an</c>.</summary>
        private static NDArray AN2
        {
            get
            {
                var an = A;
                an[":, 0, :"] = double.NaN;
                return an;
            }
        }

        /// <summary>The int8 2 x 3 array <c>[[1, -2, 3], [4, 5, -6]]</c>, the evidence's <c>i8</c>.</summary>
        private static NDArray I8 => np.array(new sbyte[,] { { 1, -2, 3 }, { 4, 5, -6 } });

        /// <summary>The 2 x 3 matrix <c>[[1, 5, 2], [4, 0, 6]]</c>, batch5_numpy_evidence.py's <c>m</c>.</summary>
        private static NDArray M => np.matrix(np.array(new double[,] { { 1, 5, 2 }, { 4, 0, 6 } }));

        /// <summary>The 2 x 3 matrix <c>[[1, nan, 2], [4, 0, 6]]</c>, batch5_matrix_tuple.py's <c>m</c>.</summary>
        private static NDArray MN => np.matrix(np.array(new double[,] { { 1, double.NaN, 2 }, { 4, 0, 6 } }));

        /// <summary><c>default_rng(7).standard_normal((3, 50, 7))</c> as float32, batch5_bits.py's <c>r</c>: a C-ordered
        ///     operand whose float sums over <c>(0, 2)</c> depend on the call schedule (34 of 50 outputs differ from two
        ///     chained single-axis sums).</summary>
        private static NDArray R32 => np.random.default_rng(7).standard_normal(new Shape(3, 50, 7)).astype(np.float32);

        /// <summary>The float64 draws of <see cref="R32"/> before the cast, batch5_bits.py's <c>d</c>.</summary>
        private static NDArray R64 => np.random.default_rng(7).standard_normal(new Shape(3, 50, 7));

        /// <summary>The raw bits of the first <paramref name="count"/> elements of a float32 array, in C order.</summary>
        /// <param name="r">The array.</param>
        /// <param name="count">How many elements.</param>
        /// <returns>The bit patterns.</returns>
        private static uint[] Bits32(NDArray r, int count)
            => r.ToArray<float>().Take(count).Select(BitConverter.SingleToUInt32Bits).ToArray();

        /// <summary>The raw bits of the first <paramref name="count"/> elements of a float64 array, in C order.</summary>
        /// <param name="r">The array.</param>
        /// <param name="count">How many elements.</param>
        /// <returns>The bit patterns.</returns>
        private static ulong[] Bits64(NDArray r, int count)
            => r.ToArray<double>().Take(count).Select(BitConverter.DoubleToUInt64Bits).ToArray();

        /// <summary>A result's strides in BYTES, NumPy's <c>.strides</c>.</summary>
        /// <param name="r">The array.</param>
        /// <returns>The strides times the item size.</returns>
        private static long[] ByteStrides(NDArray r) => r.Shape.strides.Select(s => s * r.dtypesize).ToArray();

        // ------------------------------------------------------------------ fixes 1-2: sum / prod

        /// <summary>np.sum / ndarray.sum over a tuple: one pass over the set, keepdims keeping each reduced axis, the empty
        ///     tuple as identity + every element, NumPy's repeat and bounds errors.</summary>
        [TestMethod]
        public void Sum_TupleAxis_ReducesTheSetTogether()
        {
            np.sum(A, (0, 2)).ToArray<double>().Should().Equal(60.0, 92.0, 124.0);
            np.sum(A, (0, 2), keepdims: true).shape.Should().Equal(1, 3, 1);
            A.sum((0, 2)).ToArray<double>().Should().Equal(60.0, 92.0, 124.0);
            np.sum(A, new NDAxis())["0, 1"].ToArray<double>().Should().Equal(4.0, 5.0, 6.0, 7.0);
            Action dup = () => np.sum(A, (0, 0));
            dup.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            Action bounds = () => np.sum(A, (0, 9));
            bounds.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 3*");

            // NEP 50's accumulator and the identity start: int8 sums in int64, and 0 + -0.0 is +0.0.
            var i8 = np.sum(I8, new NDAxis());
            i8.typecode.Should().Be(NPTypeCode.Int64);
            i8.ToArray<long>().Should().Equal(1L, -2L, 3L, 4L, 5L, -6L);
            Bits64(np.sum(np.full(new Shape(2, 2), -0.0), new NDAxis()), 4).Should().AllBeEquivalentTo(0UL);

            // The 0-d allowance is the integer form's only: () reduces nothing, (0,) is out of bounds.
            np.sum(np.array(5.0), new NDAxis()).GetDouble(0).Should().Be(5.0);
            Action zeroD = () => np.sum(np.array(5.0), new NDAxis(new[] { 0 }));
            zeroD.Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");
        }

        /// <summary>The set's bits are NumPy's iterator schedule (axis sort, reduce form per inner call), not two chained
        ///     single-axis sums, which differ in 34 of the 50 float32 outputs.</summary>
        [TestMethod]
        public void Sum_TupleAxis_IsNumPysCallSchedule()
        {
            Bits32(np.sum(R32, (0, 2)), 6).Should().Equal(0xbdf14ce0u, 0x40a53ec4u, 0xc141198fu, 0xc033f437u, 0xc1321d24u, 0xc000a882u);
            Bits64(np.sum(R64, (0, 2)), 4).Should().Equal(0xbfbe299a39d254c0UL, 0x4014a7d87c004a70UL, 0xc0282331ef945532UL,
                0xc0067e86ed1c05beUL);
        }

        /// <summary>The result is laid out as NumPy's iterator allocates it: an F input gives F strides, the empty tuple
        ///     keeps the input's memory order, a transposed input its axis order.</summary>
        [TestMethod]
        public void Sum_TupleAxis_KeepsNumPysLayout()
        {
            var f3 = np.asfortranarray(A);
            ByteStrides(np.sum(f3, new NDAxis())).Should().Equal(8, 16, 48);
            ByteStrides(np.sum(f3, (0, 2), keepdims: true)).Should().Equal(8, 8, 24);
            var t3 = A.transpose(2, 0, 1);
            var along1 = np.sum(t3, new NDAxis(new[] { 1 }));
            ByteStrides(along1).Should().Equal(8, 32);
            along1.ToArray<double>().Should().Equal(12.0, 20.0, 28.0, 14.0, 22.0, 30.0, 16.0, 24.0, 32.0, 18.0, 26.0, 34.0);
            ByteStrides(np.max(np.asfortranarray(np.ones(new Shape(4, 3, 2, 5))), (1, 3))).Should().Equal(8, 32);
            ByteStrides(np.mean(A, new NDAxis())).Should().Equal(96, 32, 8);
        }

        /// <summary>np.prod / ndarray.prod over a tuple, the empty tuple's 1 * x keeping -0.0, a float32 dtype= loop.</summary>
        [TestMethod]
        public void Prod_TupleAxis_ReducesTheSetTogether()
        {
            np.prod(A, (0, 2)).ToArray<double>().Should().Equal(0.0, 78140160.0, 1683158400.0);
            A.prod((0, 2)).ToArray<double>().Should().Equal(0.0, 78140160.0, 1683158400.0);
            var p = np.prod(I8, (0, 1));
            p.typecode.Should().Be(NPTypeCode.Int64);
            p.GetInt64(0).Should().Be(720L);
            Bits64(np.prod(np.full(new Shape(2, 2), -0.0), new NDAxis()), 4).Should().AllBeEquivalentTo(0x8000000000000000UL);
            Bits32(np.prod(A + 1, (0, 2), np.float32), 3).Should().Equal(0x497ff000u, 0x4d3a4cf8u, 0x4f349554u);
            Action dup = () => np.prod(A, (0, 0));
            dup.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
        }

        /// <summary>sum / prod / max with initial= and where= over a set: the keyword statements over the whole set.</summary>
        [TestMethod]
        public void SumProdMax_TupleAxis_TakeTheKeywords()
        {
            np.sum(A, (0, 2), initial: 10.0, @where: A > 3).ToArray<double>().Should().Equal(64.0, 102.0, 134.0);
            np.max(A, (0, 2), initial: -1.0, @where: A > 3).ToArray<double>().Should().Equal(15.0, 19.0, 23.0);
            var o = np.empty(new Shape(3));
            ReferenceEquals(np.max(A, (0, 2), o), o).Should().BeTrue();
            o.ToArray<double>().Should().Equal(15.0, 19.0, 23.0);
            np.sum(A > 5, (0, 2)).ToArray<long>().Should().Equal(4L, 6L, 8L);
            np.prod(A > 5, (0, 2)).ToArray<long>().Should().Equal(0L, 0L, 1L);
            np.max(A > 5, (0, 2)).ToArray<bool>().Should().Equal(true, true, true);
            np.sum(A.astype(np.uint8), (0, 2)).typecode.Should().Be(NPTypeCode.UInt64);
            np.sum((A * 20).astype(np.uint8), (0, 2), np.uint8).ToArray<byte>().Should().Equal((byte)176, (byte)48, (byte)176);
        }

        /// <summary>The checks run in NumPy's order: the out / where converters, then each tuple item's bounds, then the
        ///     repeat - np.mean's count normalizes the axes first, its sum checks the out before the repeat, and np.var's
        ///     first sum (the mean's, without the out) raises the repeat before any out check.</summary>
        [TestMethod]
        public void TupleAxis_ChecksRunInNumPysOrder()
        {
            var ro = np.empty(new Shape(3));
            ro.Shape.IsWriteable.Should().BeTrue();
            var readOnly = np.broadcast_to(np.zeros(new Shape(1)), new Shape(3));   // a broadcast view is read-only

            Action sumOut = () => np.sum(A, (0, 0), @out: readOnly);
            sumOut.Should().Throw<ValueError>().WithMessage("output array is read-only");
            Action sumBounds = () => np.sum(A, (0, 9), @out: np.empty(new Shape(7)));
            sumBounds.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 3*");
            Action sumDup = () => np.sum(A, (0, 0), @out: np.empty(new Shape(7)));
            sumDup.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            Action sumWhere = () => np.sum(A, (0, 0), @where: np.ones(new Shape(4), np.int64));
            sumWhere.Should().Throw<ArgumentException>()
                .WithMessage("Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
            Action maxOut = () => np.max(A, (0, 0), readOnly);
            maxOut.Should().Throw<ValueError>().WithMessage("output array is read-only");

            Action meanOut = () => np.mean(A, (0, 0), @out: readOnly);
            meanOut.Should().Throw<ValueError>().WithMessage("output array is read-only");
            Action meanBounds = () => np.mean(A, new NDAxis(9, 0, 0), @out: readOnly);
            meanBounds.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 3*");
            Action varOut = () => np.var(A, (0, 0), @out: readOnly);
            varOut.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            Action nansumOut = () => np.nansum(A, (0, 0), @out: readOnly);
            nansumOut.Should().Throw<ValueError>().WithMessage("output array is read-only");
            Action nanmeanOut = () => np.nanmean(A, (0, 0), @out: readOnly);
            nanmeanOut.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            Action nanvarDtype = () => np.nanvar(A, (0, 9), np.int64);
            nanvarDtype.Should().Throw<TypeError>().WithMessage("If a is inexact, then dtype must be inexact");
        }

        // ------------------------------------------------------------------ fixes 3-5: mean / var / std

        /// <summary>np.mean over a set: _mean's count (the product of the reduced extents) and sum; where= counts per
        ///     output; dtype= and out= as _mean hands them on; an empty set of elements is 0 / 0.</summary>
        [TestMethod]
        public void Mean_TupleAxis_IsMeansStatements()
        {
            np.mean(A, (0, 2)).ToArray<double>().Should().Equal(7.5, 11.5, 15.5);
            A.mean((0, 2)).ToArray<double>().Should().Equal(7.5, 11.5, 15.5);
            np.mean(A, (0, 2), keepdims: true).shape.Should().Equal(1, 3, 1);
            np.mean(A, (0, 2), @where: A > 3).ToArray<double>().Should().Equal(13.5, 11.5, 15.5);
            np.mean(A, (0, 2), @where: np.array(new[] { true, false, true, true })).ToArray<double>()
                .Should().Equal(7.666666666666667, 11.666666666666666, 15.666666666666666);
            var f32 = np.mean(A, (0, 2), np.float32);
            f32.typecode.Should().Be(NPTypeCode.Single);
            f32.ToArray<float>().Should().Equal(7.5f, 11.5f, 15.5f);
            var intOut = np.empty(new Shape(3), np.int64);
            np.mean(A, (0, 2), @out: intOut).ToArray<long>().Should().Equal(7L, 11L, 15L);
            np.mean(A.astype(np.int64), (0, 2), np.int8).ToArray<sbyte>().Should().Equal((sbyte)7, (sbyte)11, (sbyte)15);
            np.mean(A > 5, (0, 2)).ToArray<double>().Should().Equal(0.5, 0.75, 1.0);
            np.mean(I8, new NDAxis()).ToArray<double>().Should().Equal(1.0, -2.0, 3.0, 4.0, 5.0, -6.0);
            var f16 = np.mean(I8.astype(np.float16), (0, 1));
            f16.typecode.Should().Be(NPTypeCode.Half);
            BitConverter.HalfToUInt16Bits(f16.GetAtIndex<Half>(0)).Should().Be((ushort)0x3aab);
            np.mean(np.zeros(new Shape(0, 3, 2)), (0, 2)).ToArray<double>().Should().OnlyContain(v => double.IsNaN(v));
            Bits32(np.mean(R32, (0, 2)), 6).Should().Equal(0xbbb7d90cu, 0x3e7bcd5bu, 0xbf131faau, 0xbe091b98u, 0xbf07b4aeu, 0xbdc40cf7u);
            Action zeroD = () => np.mean(np.array(5.0), new NDAxis(new[] { 0 }));
            zeroD.Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");
        }

        /// <summary>np.var / np.std over a set: _var's statements (the count, the keepdims mean, the squared deviations, their
        ///     sum), ddof, where= and mean= included; the empty tuple's deviation from each element's own mean.</summary>
        [TestMethod]
        public void VarStd_TupleAxis_IsVarsStatements()
        {
            np.var(A, (0, 2)).ToArray<double>().Should().Equal(37.25, 37.25, 37.25);
            A.var((0, 2)).ToArray<double>().Should().Equal(37.25, 37.25, 37.25);
            np.std(A, (0, 2)).ToArray<double>().Should().Equal(6.103277807866851, 6.103277807866851, 6.103277807866851);
            A.std((0, 2)).ToArray<double>().Should().Equal(6.103277807866851, 6.103277807866851, 6.103277807866851);
            np.std(A, (0, 2), ddof: 1).ToArray<double>().Should().Equal(6.524678426668135, 6.524678426668135, 6.524678426668135);
            np.var(A, new NDAxis(), ddof: 1).ToArray<double>().Should().OnlyContain(v => double.IsNaN(v));
            np.var(A, (0, 2), mean: np.mean(A, (0, 2), keepdims: true)).ToArray<double>().Should().Equal(37.25, 37.25, 37.25);
            np.std(A, (0, 2), @where: A > 3).ToArray<double>().Should().Equal(1.118033988749895, 6.103277807866851, 6.103277807866851);
            np.var(A, (0, 2), ddof: 1, @where: np.array(new[] { true, false, true, true })).ToArray<double>()
                .Should().Equal(45.06666666666667, 45.06666666666667, 45.06666666666667);
            var o = np.empty(new Shape(3));
            ReferenceEquals(np.std(A, (0, 2), @out: o), o).Should().BeTrue();
            o.ToArray<double>().Should().Equal(6.103277807866851, 6.103277807866851, 6.103277807866851);
            np.var(A + new Complex(0, 1) * A["::-1"], (0, 2)).ToArray<double>().Should().Equal(74.5, 74.5, 74.5);
            np.var(A > 5, (0, 2)).ToArray<double>().Should().Equal(0.25, 0.1875, 0.0);
            np.var(np.zeros(new Shape(0, 3, 2)), (0, 2)).ToArray<double>().Should().OnlyContain(v => double.IsNaN(v));
            Bits32(np.var(R32, (0, 2)), 6).Should().Equal(0x3eeef3d8u, 0x3f36fae7u, 0x3f65d60bu, 0x3fb61681u, 0x3f322a12u, 0x3ebf3834u);
            Bits64(np.std(R64, (1, 2)), 3).Should().Equal(0x3fed90f42300a0fdUL, 0x3fee105eccb5e25dUL, 0x3feef9ac5ab07b1aUL);
            Action dup = () => np.std(A, (0, 0));
            dup.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
        }

        // ------------------------------------------------------------------ fixes 6-7: max / min

        /// <summary>np.max / np.amax / np.min / np.amin and the ndarray methods over a set; the empty tuple reduces nothing
        ///     (an empty result needs no identity); an empty reduced axis without initial= is NumPy's ValueError.</summary>
        [TestMethod]
        public void MaxMin_TupleAxis_ReduceTheSetTogether()
        {
            np.max(A, (0, 2)).ToArray<double>().Should().Equal(15.0, 19.0, 23.0);
            np.amax(A, (0, 2)).ToArray<double>().Should().Equal(15.0, 19.0, 23.0);
            A.max((0, 2)).ToArray<double>().Should().Equal(15.0, 19.0, 23.0);
            np.min(A, (0, 2)).ToArray<double>().Should().Equal(0.0, 4.0, 8.0);
            np.amin(A, (0, 2), keepdims: true).shape.Should().Equal(1, 3, 1);
            A.min((0, 2)).ToArray<double>().Should().Equal(0.0, 4.0, 8.0);
            np.max(A, new NDAxis())["0, 1"].ToArray<double>().Should().Equal(4.0, 5.0, 6.0, 7.0);
            Action dup = () => np.max(A, (0, 0));
            dup.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");

            var z = np.zeros(new Shape(0, 3, 2));
            np.max(z, new NDAxis()).shape.Should().Equal(0, 3, 2);
            np.max(z, (1, 2)).shape.Should().Equal(0);
            Action empty = () => np.max(z, (0, 2));
            empty.Should().Throw<ValueError>().WithMessage("zero-size array to reduction operation maximum which has no identity");
            np.max(z, (0, 2), initial: 1.0).ToArray<double>().Should().Equal(1.0, 1.0, 1.0);
            np.max(np.array(5.0), new NDAxis()).GetDouble(0).Should().Be(5.0);
            Bits32(np.max(np.asfortranarray(R32), (0, 2)), 4).Should().Equal(0x3ff64c4au, 0x3fef78dbu, 0x3f5ac9a3u, 0x40107ea8u);
        }

        // ------------------------------------------------------------------ fixes 8-11: the nan family

        /// <summary>nansum / nanprod: _replace_nan, then np.sum / np.prod over the set (an all-NaN slab gives the identity).</summary>
        [TestMethod]
        public void NanSumProd_TupleAxis_AreSumProdOfTheReplacedCopy()
        {
            np.nansum(AN, (0, 2)).ToArray<double>().Should().Equal(60.0, 92.0, 124.0);
            np.nanprod(AN, (0, 2)).ToArray<double>().Should().Equal(196560.0, 78140160.0, 1683158400.0);
            np.nansum(AN, (0, 2), keepdims: true).shape.Should().Equal(1, 3, 1);
            np.nansum(AN2, (0, 2)).ToArray<double>().Should().Equal(0.0, 92.0, 124.0);
            np.nanprod(AN2, (0, 2)).ToArray<double>().Should().Equal(1.0, 78140160.0, 1683158400.0);
            Action dup = () => np.nanprod(AN, (0, 0));
            dup.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
        }

        /// <summary>nanmean / nanvar / nanstd over a set: the non-NaN counts, the sums, the divisions (an all-NaN slab is
        ///     NaN), where= and ddof included.</summary>
        [TestMethod]
        public void NanMeanVarStd_TupleAxis_AreNumPysStatements()
        {
            np.nanmean(AN, (0, 2)).ToArray<double>().Should().Equal(8.571428571428571, 11.5, 15.5);
            np.nanvar(AN, (0, 2)).ToArray<double>().Should().Equal(33.38775510204082, 37.25, 37.25);
            np.nanstd(AN, (0, 2)).ToArray<double>().Should().Equal(5.778213833187625, 6.103277807866851, 6.103277807866851);
            np.nanvar(AN, (0, 2), keepdims: true).shape.Should().Equal(1, 3, 1);

            var mean2 = np.nanmean(AN2, (0, 2)).ToArray<double>();
            double.IsNaN(mean2[0]).Should().BeTrue();
            mean2.Skip(1).Should().Equal(11.5, 15.5);
            var std2 = np.nanstd(AN2, (0, 2), ddof: 1).ToArray<double>();
            double.IsNaN(std2[0]).Should().BeTrue();
            std2.Skip(1).Should().Equal(6.524678426668135, 6.524678426668135);
            var masked = np.nanmean(AN2, (0, 2), @where: AN2 > 3).ToArray<double>();
            double.IsNaN(masked[0]).Should().BeTrue();
            masked.Skip(1).Should().Equal(11.5, 15.5);

            np.nanvar(I8, (0, 1)).GetDouble(0).Should().Be(14.472222222222223);
            np.nanvar(A.astype(np.float32), (0, 2)).ToArray<float>().Should().Equal(37.25f, 37.25f, 37.25f);
            np.nanmean(A.astype(np.float16), (0, 2)).ToArray<Half>().Select(BitConverter.HalfToUInt16Bits)
                .Should().Equal((ushort)0x4780, (ushort)0x49c0, (ushort)0x4bc0);
            Bits64(np.nanvar(R64, (0, 1)), 4).Should().Equal(0x3fef18f15076cf4bUL, 0x3fe8047b44ee47fbUL, 0x3ff0b69202b0f01aUL,
                0x3fe88da044862c7cUL);
            Action bounds = () => np.nanmean(AN, (0, 9));
            bounds.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 3*");
            Action dup = () => np.nanvar(AN, (0, 0));
            dup.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
        }

        /// <summary>nanmax / nanmin over a set: fmax.reduce / fmin.reduce of the whole set (an all-NaN slab is NaN), with
        ///     out= / initial= / where=, for integers (fmax of integers is max) and complex.</summary>
        [TestMethod]
        public void NanMaxMin_TupleAxis_AreFmaxFminReduce()
        {
            np.nanmax(AN, (0, 2)).ToArray<double>().Should().Equal(15.0, 19.0, 23.0);
            np.nanmin(AN, (0, 2)).ToArray<double>().Should().Equal(1.0, 4.0, 8.0);
            var max2 = np.nanmax(AN2, (0, 2)).ToArray<double>();
            double.IsNaN(max2[0]).Should().BeTrue();
            max2.Skip(1).Should().Equal(19.0, 23.0);
            var min2 = np.nanmin(AN2, (0, 2), keepdims: true);
            min2.shape.Should().Equal(1, 3, 1);
            ByteStrides(min2).Should().Equal(24, 8, 8);
            np.nanmax(AN2, (0, 2), initial: -1.0, @where: AN2 > 3).ToArray<double>().Should().Equal(-1.0, 19.0, 23.0);
            var i8 = np.nanmax(I8, (0, 1));
            i8.typecode.Should().Be(NPTypeCode.SByte);
            i8.GetAtIndex<sbyte>(0).Should().Be((sbyte)5);
            np.nanmax(np.array(new Complex[,] { { new(1, 1), new(double.NaN, 0) }, { new(3, 0), new(0, 2) } }), (0, 1))
                .GetAtIndex<Complex>(0).Should().Be(new Complex(3, 0));
            np.nanmin(A.astype(np.float16), (0, 2)).ToArray<Half>().Select(BitConverter.HalfToUInt16Bits)
                .Should().Equal((ushort)0x0, (ushort)0x4400, (ushort)0x4800);
            Action wrongOut = () => np.nanmax(AN, (0, 2), np.empty(new Shape(5)));
            wrongOut.Should().Throw<IncorrectShapeException>()
                .WithMessage("operands could not be broadcast together with remapped shapes [original->remapped]: (5,)->(5,newaxis) (2,3,4) *");
            Action bounds = () => np.nanmax(AN, (0, 9));
            bounds.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 3*");
        }

        // ------------------------------------------------------------------ numpy.matrix operands

        /// <summary>A matrix keeps its matrix answer for a tuple: the matrix methods' keepdims result, which _collapse
        ///     hands back for any axis but None - the (1, 1) matrix for (0, 1), the whole matrix for (), a row or a column
        ///     for a one-axis tuple - for the functions that call the matrix method (sum, max, var) and for those whose matrix
        ///     intermediate is collapsed (the nan family).</summary>
        [TestMethod]
        public void MatrixOperand_TupleAxis_KeepsTheMatrixShape()
        {
            void Matrix(NDArray r, long rows, long cols, params double[] values)
            {
                (r is NDMatrix).Should().BeTrue();
                r.shape.Should().Equal(rows, cols);
                var got = r.ToArray<double>();
                got.Length.Should().Be(values.Length);
                for (int i = 0; i < values.Length; i++)
                    (double.IsNaN(values[i]) ? double.IsNaN(got[i]) : got[i] == values[i]).Should().BeTrue($"element {i}");
            }

            Matrix(np.sum(M, (0, 1)), 1, 1, 18.0);
            Matrix(np.prod(M, (0, 1)), 1, 1, 0.0);
            Matrix(np.mean(M, (0, 1)), 1, 1, 3.0);
            Matrix(np.var(M, (0, 1)), 1, 1, 4.666666666666667);
            Matrix(np.std(M, (0, 1)), 1, 1, 2.160246899469287);
            Matrix(np.max(M, (0, 1)), 1, 1, 6.0);
            Matrix(np.amin(M, (0, 1)), 1, 1, 0.0);
            // The method of the static type: a matrix variable binds matrix.sum (an NDArray-typed one ndarray.sum).
            Matrix(((NDMatrix)M).sum((0, 1)), 1, 1, 18.0);

            double nan = double.NaN;
            Matrix(np.nansum(MN, (0, 1)), 1, 1, 13.0);
            Matrix(np.nanprod(MN, (0, 1)), 1, 1, 0.0);
            Matrix(np.nanmean(MN, (0, 1)), 1, 1, 2.6);
            Matrix(np.nanvar(MN, (0, 1)), 1, 1, 4.64);
            Matrix(np.nanstd(MN, (0, 1)), 1, 1, 2.1540659228538015);
            Matrix(np.nanmax(MN, (0, 1)), 1, 1, 6.0);
            Matrix(np.nanmin(MN, (0, 1)), 1, 1, 0.0);

            Matrix(np.sum(MN, new NDAxis()), 2, 3, 1.0, nan, 2.0, 4.0, 0.0, 6.0);
            Matrix(np.nansum(MN, new NDAxis()), 2, 3, 1.0, 0.0, 2.0, 4.0, 0.0, 6.0);
            Matrix(np.nanprod(MN, new NDAxis()), 2, 3, 1.0, 1.0, 2.0, 4.0, 0.0, 6.0);
            Matrix(np.nanvar(MN, new NDAxis()), 2, 3, 0.0, nan, 0.0, 0.0, 0.0, 0.0);
            Matrix(np.nanmax(MN, new NDAxis()), 2, 3, 1.0, nan, 2.0, 4.0, 0.0, 6.0);

            Matrix(np.nansum(MN, new NDAxis(new[] { 0 })), 1, 3, 5.0, 0.0, 8.0);
            Matrix(np.nansum(MN, new NDAxis(new[] { 1 })), 2, 1, 3.0, 10.0);
            Matrix(np.nanmean(MN, new NDAxis(new[] { -1 })), 2, 1, 1.5, 3.3333333333333335);
            Matrix(np.nanmax(MN, new NDAxis(new[] { 1 })), 2, 1, 2.0, 6.0);
            Matrix(np.sum(MN, new NDAxis(new[] { 1 })), 2, 1, nan, 10.0);
        }

        /// <summary>The quantile family's matrix answer for a full tuple, and np.median's along a one-axis tuple, are NumPy's
        ///     kept-dims matrices (the collapse reads the tuple instead of None). The rows NumPy's _quantile makes for an
        ///     axis-1 percentile stay a pin (OpenBugs.ApiParity).</summary>
        [TestMethod]
        public void MatrixOperand_QuantileFamily_FullTuple_IsTheOneByOneMatrix()
        {
            var m = np.matrix(np.array(new double[,] { { 1, 5, 2 }, { 4, 0, 6 } }));
            void Matrix(NDArray r, long rows, long cols, params double[] values)
            {
                (r is NDMatrix).Should().BeTrue();
                r.shape.Should().Equal(rows, cols);
                r.ToArray<double>().Should().Equal(values);
            }

            Matrix(np.median(m, new[] { 1 }), 2, 1, 2.0, 4.0);
            Matrix(np.median(m, (0, 1)), 1, 1, 3.0);
            Matrix(np.percentile(m, 50, (0, 1)), 1, 1, 3.0);
        }

        // ------------------------------------------------------------------ text operands

        /// <summary>Text under a set: StringDType's add / maximum / minimum are not reorderable (two axes or more is
        ///     NumPy's ValueError), the empty tuple reduces nothing (a copy, or initial= combined with each element), and every
        ///     refusal is the loop's, whatever the axes.</summary>
        [TestMethod]
        public void TextOperand_TupleAxis_FollowsTheLoops()
        {
            var S = np.dtypes.StringDType();
            var v2 = np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }).astype(S);
            var v3 = np.array(new[,,] { { { "a", "b" }, { "c", "d" } }, { { "e", "f" }, { "g", "h" } } }).astype(S);
            var u2 = np.array(new[,] { { "1", "2", "3" }, { "4", "5", "6" } });
            var mask = np.array(new[,] { { true, false, true }, { false, true, true } });
            string Text(NDArray r) => r.ToString();

            Text(np.sum(v2, new NDAxis())).Should().Be("[['a' 'b' 'c']\n ['d' 'e' 'f']]");
            Text(np.sum(v2, new NDAxis(), initial: "x")).Should().Be("[['xa' 'xb' 'xc']\n ['xd' 'xe' 'xf']]");
            Text(np.sum(v2, new NDAxis(), initial: "x", @where: mask)).Should().Be("[['xa' 'x' 'xc']\n ['x' 'xe' 'xf']]");
            Action noIdentity = () => np.sum(v2, new NDAxis(), @where: mask);
            noIdentity.Should().Throw<ValueError>()
                .WithMessage("reduction operation 'add' does not have an identity, so to use a where mask one has to specify 'initial'");
            Text(np.max(v2, new NDAxis(), initial: "b")).Should().Be("[['b' 'b' 'c']\n ['d' 'e' 'f']]");
            Text(np.max(v2, new NDAxis(), initial: "b", @where: mask)).Should().Be("[['b' 'b' 'c']\n ['b' 'e' 'f']]");
            Text(np.nansum(v2, new NDAxis())).Should().Be("[['a' 'b' 'c']\n ['d' 'e' 'f']]");

            Action sum2 = () => np.sum(v3, (0, 1));
            sum2.Should().Throw<ValueError>()
                .WithMessage("reduction operation 'add' is not reorderable, so at most one axis may be specified");
            Action max2 = () => np.max(v3, (0, 2));
            max2.Should().Throw<ValueError>()
                .WithMessage("reduction operation 'maximum' is not reorderable, so at most one axis may be specified");
            Action min2 = () => np.min(v3, (1, 2));
            min2.Should().Throw<ValueError>()
                .WithMessage("reduction operation 'minimum' is not reorderable, so at most one axis may be specified");
            Action mean2 = () => np.mean(v2, (0, 1));
            mean2.Should().Throw<ValueError>()
                .WithMessage("reduction operation 'add' is not reorderable, so at most one axis may be specified");
            Action meanNone = () => np.mean(v2, new NDAxis());
            meanNone.Should().Throw<TypeError>().WithMessage(
                "ufunc 'divide' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.Int64DType'>) -> None");
            Action nanmax = () => np.nanmax(v2, new NDAxis());
            nanmax.Should().Throw<TypeError>().WithMessage(
                "ufunc 'fmax' did not contain a loop with signature matching types (None, <class 'numpy.dtypes.StringDType'>) -> None");
            Action sumU = () => np.sum(u2, (0, 1));
            sumU.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U1'), dtype('<U1'), dtype('<U2'))");
            Action maxU = () => np.max(u2, new NDAxis());
            maxU.Should().Throw<TypeError>()
                .WithMessage("ufunc 'maximum' did not contain a loop with signature matching types (dtype('<U1'), dtype('<U1')) -> None");
            np.prod(u2, (0, 1), np.float64).GetDouble(0).Should().Be(720.0);
        }

        // ------------------------------------------------------------------ fix 13: argmax / argmin

        /// <summary>np.argmax / np.argmin and the ndarray methods keep ONE axis signature with NumPy's axis type: a tuple of
        ///     any length is NumPy's TypeError (a matrix's too), and a C# literal as the third (second) positional argument
        ///     is out=, NumPy's "output must be an array".</summary>
        [TestMethod]
        public void ArgMaxMin_OneAxisSignature_RefusesATuple()
        {
            const string tuple = "'tuple' object cannot be interpreted as an integer";
            Action pair = () => np.argmax(A, (0, 2));
            pair.Should().Throw<TypeError>().WithMessage(tuple);
            Action one = () => np.argmin(A, new NDAxis(new[] { 0 }));
            one.Should().Throw<TypeError>().WithMessage(tuple);
            Action none = () => np.argmax(A, new NDAxis());
            none.Should().Throw<TypeError>().WithMessage(tuple);
            Action method = () => A.argmax((0, 2));
            method.Should().Throw<TypeError>().WithMessage(tuple);
            Action matrix = () => np.argmax(M, (0, 1));
            matrix.Should().Throw<TypeError>().WithMessage(tuple);
            Action literalOut = () => np.argmax(A, 0, true);
            literalOut.Should().Throw<TypeError>().WithMessage("output must be an array");
            Action methodLiteralOut = () => A.argmin(0, true);
            methodLiteralOut.Should().Throw<TypeError>().WithMessage("output must be an array");

            np.argmax(A, 0, keepdims: true).shape.Should().Equal(1, 3, 4);
            np.argmax(A).Should().Be(23L);
            A.argmin(1).ToArray<long>().Should().OnlyContain(v => v == 0L);

            static string[] Forms(Type owner, string name, BindingFlags flags)
                => owner.GetMethods(flags).Where(m => m.Name == name).WithoutKeptIntegerForms()
                    .Select(m => string.Join(", ", m.GetParameters().Select(p => p.ParameterType.Name + " " + p.Name))).OrderBy(s => s).ToArray();
            Forms(typeof(np), "argmax", BindingFlags.Public | BindingFlags.Static)
                .Should().Equal("NDArray a", "NDArray a, Nullable`1 axis, NDArray out, Boolean keepdims");
            Forms(typeof(NDArray), "argmin", BindingFlags.Public | BindingFlags.Instance)
                .Should().Equal("", "Nullable`1 axis, NDArray out, Boolean keepdims");
        }

        // ------------------------------------------------------------------ fix 12: roll

        /// <summary>np.roll's one signature: an array shift broadcast against the axis tuple, every element read as Python's
        ///     int(), repeated axes adding up, NumPy's errors.</summary>
        [TestMethod]
        public void Roll_OneSignature_IsNumPysStatements()
        {
            var b = np.arange(6L).reshape(2, 3);
            np.roll(b, new[] { 1, 2 }, (0, 1)).ToArray<long>().Should().Equal(4L, 5L, 3L, 1L, 2L, 0L);
            np.roll(b, 1, (0, 1)).ToArray<long>().Should().Equal(5L, 3L, 4L, 2L, 0L, 1L);
            np.roll(b, new[] { 1, 2 }, 1).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            np.roll(b, new[] { 1, 2 }).ToArray<long>().Should().Equal(3L, 4L, 5L, 0L, 1L, 2L);
            np.roll(b, 1.7, 1).ToArray<long>().Should().Equal(2L, 0L, 1L, 5L, 3L, 4L);
            np.roll(b, -1.7, 1).ToArray<long>().Should().Equal(1L, 2L, 0L, 4L, 5L, 3L);
            np.roll(b, true, 1).ToArray<long>().Should().Equal(2L, 0L, 1L, 5L, 3L, 4L);
            np.roll(b, "1", 1).ToArray<long>().Should().Equal(2L, 0L, 1L, 5L, 3L, 4L);
            np.roll(b, new int[0], 1).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            np.roll(b, 1, new NDAxis()).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            np.roll(b, 1, (0, 0)).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            // Python ints: no overflow, Python's modulo.
            np.roll(b, np.array(new[] { 9223372036854775808UL }), 1).ToArray<long>().Should().Equal(1L, 2L, 0L, 4L, 5L, 3L);
            np.roll(b, ulong.MaxValue, 1).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            np.roll(b, new[] { 1L << 62, 1L << 62, 1L << 62 }, new NDAxis(1, 1, 1)).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            np.roll(b, 1e300, 1).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            // The layouts: an axis roll keeps the operand's memory order, the flat roll is C.
            var f = np.asfortranarray(b);
            ByteStrides(np.roll(f, 1, 0)).Should().Equal(8, 16);
            np.roll(f, 1, 0).ToArray<long>().Should().Equal(3L, 4L, 5L, 0L, 1L, 2L);
            ByteStrides(np.roll(f, 1)).Should().Equal(24, 8);
            np.roll(np.array(5L), 1).GetInt64(0).Should().Be(5L);
            var mat = np.roll(np.matrix(np.array(new long[,] { { 1, 2 }, { 3, 4 } })), new[] { 1, 1 }, (0, 1));
            (mat is NDMatrix).Should().BeTrue();
            mat.ToArray<long>().Should().Equal(4L, 3L, 2L, 1L);

            Action mismatch = () => np.roll(b, new[] { 1, 2, 3 }, (0, 1));
            mismatch.Should().Throw<IncorrectShapeException>().WithMessage(
                "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (3,) and arg 1 with shape (2,).");
            Action twoD = () => np.roll(b, new int[,] { { 1, 2 } }, (0, 1));
            twoD.Should().Throw<ValueError>().WithMessage("'shift' and 'axis' should be scalars or 1D sequences");
            Action nan = () => np.roll(b, double.NaN, 1);
            nan.Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
            Action inf = () => np.roll(b, double.PositiveInfinity, 1);
            inf.Should().Throw<OverflowError>().WithMessage("cannot convert float infinity to integer");
            Action literal = () => np.roll(b, np.array(new[] { "1.5" }), 1);
            literal.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('1.5')");
            Action bounds = () => np.roll(b, 1, (0, 5));
            bounds.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 2*");
            Action zeroD = () => np.roll(np.array(5L), 1, new NDAxis());
            zeroD.Should().Throw<ValueError>().WithMessage("not enough values to unpack (expected 2, got 0)");

            typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == "roll").WithoutKeptIntegerForms()
                .Select(m => string.Join(", ", m.GetParameters().Select(p => p.ParameterType.Name + " " + p.Name)))
                .Should().Equal("NDArray a, NDArray shift, Nullable`1 axis");
        }

        // ------------------------------------------------------------------ fix 15: Generator size=None

        /// <summary>The Generator samplers take <c>size</c> as <c>Shape?</c>: null is NumPy's <c>size=None</c> (one draw,
        ///     the house 0-d array for NumPy's scalar), <c>()</c> a 0-d request, a shape the array; every public sampler
        ///     with a <c>size</c> declares it so, defaulting to null.</summary>
        [TestMethod]
        public void Generator_SizeNone_IsNull()
        {
            var normal = np.random.default_rng(0).normal(size: null);
            normal.ndim.Should().Be(0);
            normal.GetDouble(0).Should().Be(0.1257302210933933);
            np.random.default_rng(0).integers(5, size: null).GetInt64(0).Should().Be(4L);
            np.random.default_rng(0).normal(size: Shape.Scalar).ndim.Should().Be(0);
            np.random.default_rng(0).normal(size: new Shape(2, 3)).shape.Should().Equal(2, 3);

            var wrong = typeof(Generator).GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .SelectMany(m => m.GetParameters().Where(p => p.Name == "size").Select(p => (m, p)))
                .Where(x => x.p.ParameterType != typeof(Shape?) || !x.p.HasDefaultValue || x.p.DefaultValue is not null)
                .Select(x => x.m.Name + "(" + x.p.ParameterType.Name + ")").ToArray();
            wrong.Should().BeEmpty("NumPy's Generator samplers take size=None");
        }

        // ------------------------------------------------------------------ the signatures

        /// <summary>Every reduction of the batch has ONE public form per name whose axis is <c>NDAxis?</c>, on np, ndarray
        ///     and numpy.matrix (whose [NumPyMethod] forms must match np.*'s, or the woven matrix arm drops).</summary>
        [TestMethod]
        public void Reductions_AxisIsNumPysAxisType()
        {
            string[] names =
            {
                "sum", "prod", "mean", "var", "std", "max", "amax", "min", "amin", "nansum", "nanprod", "nanmean", "nanvar",
                "nanstd", "nanmax", "nanmin",
            };
            foreach (var name in names)
            {
                var forms = typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => m.Name == name && m.GetParameters().Any(p => p.Name == "axis")).WithoutKeptIntegerForms();
                forms.Should().HaveCount(1, $"np.{name} has one axis-taking form");
                forms[0].GetParameters().Single(p => p.Name == "axis").ParameterType.Should().Be(typeof(NDAxis?), $"np.{name}");
            }

            foreach (var name in new[] { "sum", "prod", "mean", "var", "std", "max", "min", "argmax", "argmin" })
            {
                typeof(NDArray).GetMethods(BindingFlags.Public | BindingFlags.Instance)
                    .Where(m => m.Name == name && m.DeclaringType == typeof(NDArray) && m.GetParameters().Any(p => p.Name == "axis"))
                    .WithoutKeptIntegerForms()
                    .Select(m => m.GetParameters().Single(p => p.Name == "axis").ParameterType)
                    .Should().Equal(new[] { typeof(NDAxis?) }, $"ndarray.{name}");
                typeof(NDMatrix).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                    .Where(m => m.Name == name && m.GetCustomAttributes().Any(a => a.GetType().Name == "NumPyMethodAttribute"))
                    .Select(m => m.GetParameters().Single(p => p.Name == "axis").ParameterType)
                    .Should().Equal(new[] { typeof(NDAxis?) }, $"matrix.{name}");
            }
        }
    }
}
