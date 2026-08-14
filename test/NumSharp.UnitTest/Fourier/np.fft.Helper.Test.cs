using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     np.fft helper routines — fully computed (no engine seam). Every expectation probed against
    ///     NumPy 2.4.2 (fftfreq / rfftfreq / fftshift / ifftshift): values, dtype, the error ORDER,
    ///     the Array-API <c>device</c> param, non-contiguous layouts, and axis validation.
    /// </summary>
    [TestClass]
    public class NpFftHelperTest : TestClass
    {
        // Extract a result's values in logical C-order regardless of the result's memory layout.
        private static double[] Vals(NDArray a) => a.flatten().astype(NPTypeCode.Double).ToArray<double>();

        // ---- fftfreq / rfftfreq (values) ----

        [TestMethod]
        public void FftFreq_Even()
        {
            // np.fft.fftfreq(8) == [0, .125, .25, .375, -.5, -.375, -.25, -.125]
            np.fft.fftfreq(8).Data<double>().Should().BeEquivalentTo(
                new[] { 0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125 });
        }

        [TestMethod]
        public void FftFreq_WithSpacing()
        {
            // np.fft.fftfreq(10, 0.1) == [0,1,2,3,4,-5,-4,-3,-2,-1]
            np.fft.fftfreq(10, 0.1).Data<double>().Should().BeEquivalentTo(
                new[] { 0.0, 1, 2, 3, 4, -5, -4, -3, -2, -1 });
        }

        [TestMethod]
        public void FftFreq_Odd_Length()
        {
            var r = np.fft.fftfreq(9);
            r.size.Should().Be(9);
            var d = r.Data<double>();
            d[0].Should().Be(0.0);
            d[4].Should().BeApproximately(4.0 / 9.0, 1e-15);
            d[5].Should().BeApproximately(-4.0 / 9.0, 1e-15);
        }

        [TestMethod]
        public void FftFreq_Prime7_Odd()
        {
            // np.fft.fftfreq(7) — the [0..(n-1)/2, -(n-1)/2..-1]/(d*n) odd layout
            np.fft.fftfreq(7).Data<double>().Should().BeEquivalentTo(new[]
            {
                0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855,
                -0.42857142857142855, -0.2857142857142857, -0.14285714285714285
            });
        }

        [TestMethod]
        public void FftFreq_N1_And_N2()
        {
            np.fft.fftfreq(1).Data<double>().Should().BeEquivalentTo(new[] { 0.0 });
            np.fft.fftfreq(2).Data<double>().Should().BeEquivalentTo(new[] { 0.0, -0.5 });
        }

        [TestMethod]
        public void FftFreq_Result_Is_Float64()
        {
            np.fft.fftfreq(8).dtype.Should().Be(typeof(double));
            np.fft.rfftfreq(8).dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void FftFreq_D_Arbitrary_Scalar()
        {
            // d is any scalar: int-valued, fractional, negative, NaN — all just scale 1/(n*d).
            np.fft.fftfreq(4, 2.0).Data<double>().Should().BeEquivalentTo(new[] { 0.0, 0.125, -0.25, -0.125 });
            np.fft.fftfreq(4, 0.5).Data<double>().Should().BeEquivalentTo(new[] { 0.0, 0.5, -1.0, -0.5 });
            np.fft.fftfreq(8, -0.1).Data<double>().Should().BeEquivalentTo(
                new[] { -0.0, -1.25, -2.5, -3.75, 5.0, 3.75, 2.5, 1.25 });
            foreach (var v in np.fft.fftfreq(4, double.NaN).Data<double>())
                double.IsNaN(v).Should().BeTrue();
        }

        [TestMethod]
        public void RfftFreq_Even()
        {
            // np.fft.rfftfreq(8) == [0, .125, .25, .375, .5]   (length n//2+1)
            np.fft.rfftfreq(8).Data<double>().Should().BeEquivalentTo(
                new[] { 0.0, 0.125, 0.25, 0.375, 0.5 });
        }

        [TestMethod]
        public void RfftFreq_Odd_Length()
        {
            var r = np.fft.rfftfreq(9);
            r.size.Should().Be(5);   // 9//2 + 1
            r.Data<double>()[4].Should().BeApproximately(4.0 / 9.0, 1e-15);
        }

        [TestMethod]
        public void RfftFreq_Prime7()
        {
            // np.fft.rfftfreq(7) == [0, 1/7, 2/7, 3/7]  (length 7//2+1 == 4)
            np.fft.rfftfreq(7).Data<double>().Should().BeEquivalentTo(new[]
            {
                0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855
            });
        }

        [TestMethod]
        public void RfftFreq_WithSpacing()
        {
            np.fft.rfftfreq(8, 0.1).Data<double>().Should().BeEquivalentTo(
                new[] { 0.0, 1.25, 2.5, 3.75, 5.0 });
        }

        [TestMethod]
        public void FftFreq_AnyIntegerWidth_Computes()
        {
            // NumPy accepts any integer type (int / np.int64...). In C# a long (or uint/short) must
            // compute — only a genuine floating n is rejected as "n should be an integer".
            var expected = new[] { 0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125 };
            np.fft.fftfreq(8L).Data<double>().Should().BeEquivalentTo(expected);
            np.fft.fftfreq((uint)8).Data<double>().Should().BeEquivalentTo(expected);
            np.fft.fftfreq((short)8).Data<double>().Should().BeEquivalentTo(expected);
            np.fft.rfftfreq(8L).Data<double>().Should().BeEquivalentTo(
                new[] { 0.0, 0.125, 0.25, 0.375, 0.5 });
            // long preserves the error contract
            ((Action)(() => { var _ = np.fft.fftfreq(0L); }))
                .Should().Throw<DivideByZeroException>();
            np.fft.rfftfreq(-5L).size.Should().Be(0);
        }

        // ---- fftfreq / rfftfreq error parity (type + ORDER, probed vs NumPy 2.4.2) ----

        [TestMethod]
        public void FftFreq_NonInteger_Raises()
        {
            // isinstance(n, int) check is FIRST — fires before spacing/device.
            Action act = () => { var _ = np.fft.fftfreq(2.5); };
            act.Should().Throw<ValueError>().WithMessage("n should be an integer");
        }

        [TestMethod]
        public void RfftFreq_NonInteger_Raises()
        {
            Action act = () => { var _ = np.fft.rfftfreq(3.5); };
            act.Should().Throw<ValueError>().WithMessage("n should be an integer");
        }

        [TestMethod]
        public void FftFreq_Negative_Raises()
        {
            // val=1/(n*d) is non-zero for n=-3,d=1, so empty(n) raises negative-dimensions.
            Action act = () => { var _ = np.fft.fftfreq(-3); };
            act.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
        }

        [TestMethod]
        public void FftFreq_ZeroN_IsDivisionByZero()
        {
            // val = 1.0/(0*d) -> NumPy ZeroDivisionError("float division by zero"); .NET analog.
            Action act = () => { var _ = np.fft.fftfreq(0); };
            act.Should().Throw<DivideByZeroException>().WithMessage("float division by zero");
        }

        [TestMethod]
        public void FftFreq_ZeroSpacing_IsDivisionByZero()
        {
            // d==0 -> 1.0/(n*0) division by zero for ANY n>0 (.NET's float 1/0 is +inf, so it is
            // checked explicitly to match NumPy which raises).
            Action act = () => { var _ = np.fft.fftfreq(8, 0.0); };
            act.Should().Throw<DivideByZeroException>().WithMessage("float division by zero");
        }

        [TestMethod]
        public void FftFreq_NegativeN_ZeroSpacing_DivisionFiresBeforeNegDims()
        {
            // ORDER: val=1/(n*d) computed BEFORE empty(n). n*d==0 (d=0), so ZeroDivision wins over
            // negative-dimensions even though n<0.
            Action act = () => { var _ = np.fft.fftfreq(-5, 0.0); };
            act.Should().Throw<DivideByZeroException>().WithMessage("float division by zero");
        }

        [TestMethod]
        public void RfftFreq_Negative_ReturnsEmpty_NotError()
        {
            // rfftfreq has NO empty(n) call: arange(0, n//2+1) is empty for n<0 -> empty float64.
            np.fft.rfftfreq(-5).size.Should().Be(0);
            np.fft.rfftfreq(-5).dtype.Should().Be(typeof(double));
            np.fft.rfftfreq(-1).size.Should().Be(0);   // floor-div edge: -1//2+1 == 0
            np.fft.rfftfreq(-2).size.Should().Be(0);
            np.fft.rfftfreq(-3).size.Should().Be(0);
        }

        [TestMethod]
        public void RfftFreq_ZeroSpacing_IsDivisionByZero()
        {
            Action a1 = () => { var _ = np.fft.rfftfreq(8, 0.0); };
            a1.Should().Throw<DivideByZeroException>().WithMessage("float division by zero");
            Action a2 = () => { var _ = np.fft.rfftfreq(-5, 0.0); };   // val first -> ZeroDivision
            a2.Should().Throw<DivideByZeroException>().WithMessage("float division by zero");
        }

        // ---- device (Array-API interop param, NumPy 2.0) ----

        [TestMethod]
        public void FftFreq_Device_Cpu_Ok()
        {
            np.fft.fftfreq(8, device: "cpu").Data<double>().Should().BeEquivalentTo(
                new[] { 0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125 });
            np.fft.rfftfreq(8, device: "cpu").Data<double>().Should().BeEquivalentTo(
                new[] { 0.0, 0.125, 0.25, 0.375, 0.5 });
        }

        [TestMethod]
        public void FftFreq_Device_NonCpu_Raises()
        {
            Action act = () => { var _ = np.fft.fftfreq(8, device: "gpu"); };
            act.Should().Throw<ValueError>().WithMessage("Device not understood*gpu");
            Action ract = () => { var _ = np.fft.rfftfreq(8, device: "cuda"); };
            ract.Should().Throw<ValueError>().WithMessage("Device not understood*cuda");
        }

        [TestMethod]
        public void FftFreq_Device_OrderVsOtherErrors()
        {
            // zero-division precedes device; device precedes negative-dims; integer-check precedes all.
            ((Action)(() => { var _ = np.fft.fftfreq(0, device: "gpu"); }))
                .Should().Throw<DivideByZeroException>();
            ((Action)(() => { var _ = np.fft.fftfreq(-5, device: "gpu"); }))
                .Should().Throw<ValueError>().WithMessage("Device not understood*");
            ((Action)(() => { var _ = np.fft.fftfreq(3.5, device: "gpu"); }))
                .Should().Throw<ValueError>().WithMessage("n should be an integer");
        }

        // ---- fftshift / ifftshift (values) ----

        [TestMethod]
        public void FftShift_1D_Even()
        {
            // np.fft.fftshift(arange(10)) == [5,6,7,8,9,0,1,2,3,4]
            np.fft.fftshift(np.arange(10)).Data<long>().Should().BeEquivalentTo(
                new long[] { 5, 6, 7, 8, 9, 0, 1, 2, 3, 4 });
        }

        [TestMethod]
        public void FftShift_1D_Odd()
        {
            // np.fft.fftshift(arange(9)) == [5,6,7,8,0,1,2,3,4]
            np.fft.fftshift(np.arange(9)).Data<long>().Should().BeEquivalentTo(
                new long[] { 5, 6, 7, 8, 0, 1, 2, 3, 4 });
        }

        [TestMethod]
        public void IfftShift_1D_Odd_DiffersByOne()
        {
            // np.fft.ifftshift(arange(9)) == [4,5,6,7,8,0,1,2,3]  (fftshift gives [5,6,7,8,0,1,2,3,4])
            np.fft.ifftshift(np.arange(9)).Data<long>().Should().BeEquivalentTo(
                new long[] { 4, 5, 6, 7, 8, 0, 1, 2, 3 });
        }

        [TestMethod]
        public void FftShift_IfftShift_Odd5_OneSampleApart()
        {
            np.fft.fftshift(np.arange(5)).Data<long>().Should().BeEquivalentTo(new long[] { 3, 4, 0, 1, 2 });
            np.fft.ifftshift(np.arange(5)).Data<long>().Should().BeEquivalentTo(new long[] { 2, 3, 4, 0, 1 });
        }

        [TestMethod]
        public void FftShift_2D_AllAxes()
        {
            // np.fft.fftshift(arange(6).reshape(2,3)) == [[5,3,4],[2,0,1]]
            var m = np.arange(6).reshape(2, 3);
            np.fft.fftshift(m).Data<long>().Should().BeEquivalentTo(
                new long[] { 5, 3, 4, 2, 0, 1 });
        }

        [TestMethod]
        public void FftShift_2D_SingleAxis_Int()
        {
            // np.fft.fftshift(m, axes=1) == [[2,0,1],[5,3,4]]
            var m = np.arange(6).reshape(2, 3);
            np.fft.fftshift(m, 1).Data<long>().Should().BeEquivalentTo(
                new long[] { 2, 0, 1, 5, 3, 4 });
        }

        [TestMethod]
        public void FftShift_2D_AxesList()
        {
            // np.fft.fftshift(m, axes=(0,)) == [[3,4,5],[0,1,2]]
            var m = np.arange(6).reshape(2, 3);
            np.fft.fftshift(m, new[] { 0 }).Data<long>().Should().BeEquivalentTo(
                new long[] { 3, 4, 5, 0, 1, 2 });
        }

        [TestMethod]
        public void FftShift_KeywordAxes()
        {
            // NumPy's parameter is `axes` for BOTH the int and tuple forms — keyword must bind.
            var m = np.arange(6).reshape(2, 3);
            np.fft.fftshift(m, axes: 1).Data<long>().Should().BeEquivalentTo(
                new long[] { 2, 0, 1, 5, 3, 4 });
            np.fft.ifftshift(m, axes: new[] { 0 }).Data<long>().Should().BeEquivalentTo(
                new long[] { 3, 4, 5, 0, 1, 2 });
        }

        [TestMethod]
        public void FftShift_3D_AllAxes_And_AxesTuple()
        {
            var b3 = np.arange(24).reshape(2, 3, 4);
            Vals(np.fft.fftshift(b3)).Should().BeEquivalentTo(new double[]
            {
                22, 23, 20, 21, 14, 15, 12, 13, 18, 19, 16, 17,
                10, 11, 8, 9, 2, 3, 0, 1, 6, 7, 4, 5
            }, o => o.WithStrictOrdering());
            Vals(np.fft.fftshift(b3, new[] { 0, 2 })).Should().BeEquivalentTo(new double[]
            {
                14, 15, 12, 13, 18, 19, 16, 17, 22, 23, 20, 21,
                2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9
            }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void FftShift_IfftShift_RoundTrip()
        {
            var m = np.arange(6).reshape(2, 3);
            np.fft.ifftshift(np.fft.fftshift(m)).Data<long>().Should().BeEquivalentTo(
                new long[] { 0, 1, 2, 3, 4, 5 });
        }

        // ---- fftshift memory-layout DOD (non-contiguous inputs, read via strides) ----

        [TestMethod]
        public void FftShift_Transposed()
        {
            var b = np.arange(12).reshape(3, 4);
            Vals(np.fft.fftshift(b.T)).Should().BeEquivalentTo(
                new double[] { 10, 2, 6, 11, 3, 7, 8, 0, 4, 9, 1, 5 }, o => o.WithStrictOrdering());
            Vals(np.fft.fftshift(b.T, 1)).Should().BeEquivalentTo(
                new double[] { 8, 0, 4, 9, 1, 5, 10, 2, 6, 11, 3, 7 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void FftShift_NegativeStride_Reversed()
        {
            var b = np.arange(12).reshape(3, 4);
            Vals(np.fft.fftshift(b["::-1"])).Should().BeEquivalentTo(
                new double[] { 2, 3, 0, 1, 10, 11, 8, 9, 6, 7, 4, 5 }, o => o.WithStrictOrdering());
            Vals(np.fft.ifftshift(b["::-1"], 0)).Should().BeEquivalentTo(
                new double[] { 4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void FftShift_Strided_And_Sliced()
        {
            var b = np.arange(12).reshape(3, 4);
            Vals(np.fft.fftshift(b[":, ::2"])).Should().BeEquivalentTo(
                new double[] { 10, 8, 2, 0, 6, 4 }, o => o.WithStrictOrdering());
            Vals(np.fft.fftshift(b["1:3"])).Should().BeEquivalentTo(
                new double[] { 10, 11, 8, 9, 6, 7, 4, 5 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void IfftShift_Transposed()
        {
            var b = np.arange(12).reshape(3, 4);
            Vals(np.fft.ifftshift(b.T)).Should().BeEquivalentTo(
                new double[] { 6, 10, 2, 7, 11, 3, 4, 8, 0, 5, 9, 1 }, o => o.WithStrictOrdering());
        }

        // ---- fftshift axis validation (IndexError, NOT AxisError — NumPy indexes the shape tuple) ----

        [TestMethod]
        public void FftShift_AxisOutOfRange_IndexError()
        {
            var m = np.arange(6).reshape(2, 3);
            ((Action)(() => np.fft.fftshift(m, 5))).Should().Throw<IndexError>()
                .WithMessage("tuple index out of range");
            ((Action)(() => np.fft.fftshift(m, 2))).Should().Throw<IndexError>()
                .WithMessage("tuple index out of range");
            ((Action)(() => np.fft.fftshift(m, -3))).Should().Throw<IndexError>()
                .WithMessage("tuple index out of range");
            ((Action)(() => np.fft.fftshift(m, new[] { 5 }))).Should().Throw<IndexError>()
                .WithMessage("tuple index out of range");
            ((Action)(() => np.fft.ifftshift(m, 5))).Should().Throw<IndexError>()
                .WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void FftShift_NegativeAxis_Normalizes()
        {
            // axes=-1 on a 2-D array addresses the last axis (shape[-1]).
            var m = np.arange(6).reshape(2, 3);
            np.fft.fftshift(m, -1).Data<long>().Should().BeEquivalentTo(new long[] { 2, 0, 1, 5, 3, 4 });
        }

        // ---- fftshift edge cases ----

        [TestMethod]
        public void FftShift_Empty_And_SingleElement()
        {
            np.fft.fftshift(np.zeros(new int[] { 0, 3 })).size.Should().Be(0);
            np.fft.fftshift(np.array(new long[] { 5 })).Data<long>().Should().BeEquivalentTo(new long[] { 5 });
        }

        [TestMethod]
        public void FftShift_EmptyAxes_ReturnsFreshCopy()
        {
            // NumPy's roll always returns a new array; fftshift(m, ()) must not alias the input.
            var src = np.arange(6).reshape(2, 3);
            var r = np.fft.fftshift(src, new int[0]);
            r.Data<long>().Should().BeEquivalentTo(new long[] { 0, 1, 2, 3, 4, 5 });
            r[0, 0] = 99;
            ((long)src[0, 0]).Should().Be(0);   // original untouched
        }

        [TestMethod]
        public void FftShift_PreservesDtype()
        {
            np.fft.fftshift(np.arange(10)).dtype.Should().Be(typeof(long));
            np.fft.fftshift(np.arange(10).astype(NPTypeCode.Double)).dtype.Should().Be(typeof(double));
            np.fft.fftshift(np.arange(10).astype(NPTypeCode.Single)).dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        [Misaligned]
        public void FftShift_ZeroD_NoOp_Divergence()
        {
            // NumPy's fftshift(0-d) with axes=None hits an internal roll-unpack error
            // (ValueError "not enough values to unpack (expected 2, got 0)"). NumSharp treats a
            // scalar shift as a no-op (returns a copy). An EXPLICIT axis on a 0-d still raises
            // IndexError in both, matching NumPy.
            var z = np.array(5.0);
            var r = np.fft.fftshift(z);
            r.ndim.Should().Be(0);
            ((double)r).Should().Be(5.0);
            ((Action)(() => np.fft.fftshift(z, 0))).Should().Throw<IndexError>()
                .WithMessage("tuple index out of range");
        }
    }
}
