using System;
using System.Linq;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.logspace / np.geomspace — the log-scale members of the linspace family. Every expectation is real
    ///     NumPy 2.4.2 output. The real (float64) paths are BIT-EXACT vs NumPy (verified by a 4,352-case differential
    ///     sweep); the complex geomspace paths are ACCURATE within the documented ≤3-ULP complex-unary envelope
    ///     (asserted with a relative tolerance, not bit-for-bit).
    /// </summary>
    [TestClass]
    public class np_logspace_geomspace_tests
    {
        // Bit-exact float64 equality (the same discipline the differential sweep uses).
        private static void ExactF64(NDArray a, params double[] expected)
        {
            a.typecode.Should().Be(NPTypeCode.Double);
            Assert.IsTrue(Enumerable.SequenceEqual(a.ToArray<double>(), expected),
                $"expected [{string.Join(", ", expected)}] got [{string.Join(", ", a.ToArray<double>())}]");
        }

        // Complex allclose (relative), since NumSharp's complex log10/power match NumPy only to ≤3 ULP.
        private static void AllcloseC128(NDArray a, Complex[] expected, double rtol = 1e-12)
        {
            a.typecode.Should().Be(NPTypeCode.Complex);
            var got = a.ToArray<Complex>();
            got.Length.Should().Be(expected.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                double denom = Complex.Abs(expected[i]);
                if (denom == 0) denom = 1;
                double rel = Complex.Abs(expected[i] - got[i]) / denom;
                Assert.IsTrue(rel <= rtol, $"[{i}] expected {expected[i]} got {got[i]} rel={rel:e3}");
            }
        }

        // ========================================================== logspace

        [TestMethod]
        public void Logspace_Basic()
        {
            ExactF64(np.logspace(2.0, 3.0, 4), 100.0, 215.44346900318845, 464.15888336127773, 1000.0);
            ExactF64(np.logspace(2.0, 3.0, 4, endpoint: false), 100.0, 177.82794100389228, 316.22776601683796, 562.341325190349);
        }

        [TestMethod]
        public void Logspace_Base()
        {
            ExactF64(np.logspace(2.0, 3.0, 4, @base: 2.0), 4.0, 5.039684199579493, 6.3496042078727974, 8.0);
            // Negative base with integer exponents is real (odd/even powers alternate sign), matching NumPy.
            ExactF64(np.logspace(1.0, 3.0, 3, @base: -2.0), -2.0, 4.0, -8.0);
        }

        [TestMethod]
        public void Logspace_NumEdgeCases()
        {
            np.logspace(2.0, 3.0, 0).Should().BeShaped(0).And.BeOfType(NPTypeCode.Double);
            ExactF64(np.logspace(2.0, 3.0, 1), 100.0);                    // == [base**start]
            ExactF64(np.logspace(2.0, 3.0, 1, endpoint: false), 100.0);   // endpoint ignored for num==1
        }

        [TestMethod]
        public void Logspace_Dtype_TruncatesTowardZero()
        {
            // logspace uses power(...).astype(dtype): float→int TRUNCATES (NOT floor, unlike linspace).
            np.logspace(2.0, 3.0, 4, dtype: np.int64).Should().BeOfType(NPTypeCode.Int64)
                .And.BeOfValues(100L, 215L, 464L, 1000L);
            np.logspace(2.0, 3.0, 4, @base: 2.0, dtype: np.int64).Should().BeOfType(NPTypeCode.Int64)
                .And.BeOfValues(4L, 5L, 6L, 8L);
        }

        [TestMethod]
        public void Logspace_Dtype_Float32()
        {
            var a = np.logspace(2.0, 3.0, 4, dtype: np.float32);
            a.Should().BeOfType(NPTypeCode.Single);
            Assert.IsTrue(Enumerable.SequenceEqual(a.ToArray<float>(), new[] { 100f, 215.44347f, 464.15887f, 1000f }));
        }

        [TestMethod]
        public void Logspace_Dtype_Complex_IsRealPlusZeroImag()
        {
            // NumPy computes float64 then casts to complex128 (real + 0j) — the real parts equal the float64 logspace.
            AllcloseC128(np.logspace(2.0, 3.0, 4, dtype: np.complex128), new[]
            {
                new Complex(100.0, 0), new Complex(215.44346900318845, 0),
                new Complex(464.15888336127773, 0), new Complex(1000.0, 0)
            }, rtol: 0.0); // exact
        }

        [TestMethod]
        public void Logspace_Axis_NoOp_And_Error()
        {
            // Scalar inputs → 1-D output; axis 0 and -1 are no-ops, everything else is an AxisError.
            ExactF64(np.logspace(1.0, 3.0, 3, axis: -1), 10.0, 100.0, 1000.0);
            // AxisError forces .NET's paramName suffix "(Parameter 'axis')", so match with wildcards (house convention).
            new Action(() => np.logspace(1.0, 3.0, 4, axis: 5))
                .Should().Throw<AxisError>().WithMessage("*destination: axis 5 is out of bounds for array of dimension 1*");
        }

        // ========================================================== geomspace (real)

        [TestMethod]
        public void Geomspace_Basic()
        {
            ExactF64(np.geomspace(1.0, 1000.0, 4), 1.0, 10.0, 100.0, 1000.0);
            ExactF64(np.geomspace(1.0, 1000.0, 3, endpoint: false), 1.0, 10.0, 100.0);
            ExactF64(np.geomspace(1.0, 1000.0, 4, endpoint: false),
                1.0, 5.623413251903491, 31.622776601683793, 177.82794100389228);
        }

        [TestMethod]
        public void Geomspace_NotAlwaysExactIntegers()
        {
            // NumPy's documented example: geomspace(1,256,9) is NOT exact powers of two.
            ExactF64(np.geomspace(1.0, 256.0, 9),
                1.0, 2.0, 4.0, 7.999999999999999, 16.0, 32.00000000000001, 63.999999999999986, 127.99999999999999, 256.0);
            // astype(int) truncates the near-misses down.
            np.geomspace(1.0, 256.0, 9, dtype: np.int64).Should().BeOfType(NPTypeCode.Int64)
                .And.BeOfValues(1L, 2L, 4L, 7L, 16L, 32L, 63L, 127L, 256L);
        }

        [TestMethod]
        public void Geomspace_DecreasingAndNegative()
        {
            ExactF64(np.geomspace(1000.0, 1.0, 4), 1000.0, 100.0, 10.0, 1.0);
            ExactF64(np.geomspace(-1000.0, -1.0, 4), -1000.0, -100.0, -10.0, -1.0);
            ExactF64(np.geomspace(-1.0, -1000.0, 4), -1.0, -10.0, -100.0, -1000.0);
            np.geomspace(-1.0, -256.0, 9, dtype: np.int64).Should().BeOfType(NPTypeCode.Int64)
                .And.BeOfValues(-1L, -2L, -4L, -7L, -16L, -32L, -63L, -127L, -256L);
        }

        [TestMethod]
        public void Geomspace_MixedSign_YieldsNaNInterior()
        {
            // Opposite-sign real endpoints: rotating start onto +real makes stop negative, so log10 → NaN interior,
            // while the endpoints are overwritten. NumPy: geomspace(-1,1,4) == [-1, nan, nan, 1].
            var a = np.geomspace(-1.0, 1.0, 4).ToArray<double>();
            a[0].Should().Be(-1.0);
            Assert.IsTrue(double.IsNaN(a[1]) && double.IsNaN(a[2]));
            a[3].Should().Be(1.0);

            var b = np.geomspace(1.0, -1.0, 4).ToArray<double>();
            b[0].Should().Be(1.0);
            Assert.IsTrue(double.IsNaN(b[1]) && double.IsNaN(b[2]));
            b[3].Should().Be(-1.0);
        }

        [TestMethod]
        public void Geomspace_NumEdgeCases()
        {
            np.geomspace(1.0, 1000.0, 0).Should().BeShaped(0).And.BeOfType(NPTypeCode.Double);
            ExactF64(np.geomspace(2.0, 8.0, 1), 2.0);                    // == [start]
            ExactF64(np.geomspace(2.0, 8.0, 1, endpoint: false), 2.0);   // endpoint ignored for num==1
            ExactF64(np.geomspace(2.0, 8.0, 2, endpoint: false), 2.0, 3.999999999999999);
        }

        [TestMethod]
        public void Geomspace_Dtype_Float32_Preserves_Clean_Powers()
        {
            // dt stays float64 (int64+float32 → float64), so [1,2,4,8] is computed exactly then cast to float32.
            var a = np.geomspace(1.0, 8.0, 4, dtype: np.float32);
            a.Should().BeOfType(NPTypeCode.Single);
            Assert.IsTrue(Enumerable.SequenceEqual(a.ToArray<float>(), new[] { 1f, 2f, 4f, 8f }));
        }

        [TestMethod]
        public void Geomspace_Axis_NoOp_And_Error()
        {
            ExactF64(np.geomspace(1.0, 8.0, 4, axis: -1), 1.0, 2.0, 4.0, 8.0);
            new Action(() => np.geomspace(1.0, 8.0, 4, axis: 2))
                .Should().Throw<AxisError>().WithMessage("*destination: axis 2 is out of bounds for array of dimension 1*");
        }

        // ========================================================== geomspace (complex)

        [TestMethod]
        public void Geomspace_Complex_ImaginaryLine()
        {
            // A straight line along the imaginary axis.
            AllcloseC128(np.geomspace(new Complex(0, 1), new Complex(0, 1000), 4), new[]
            {
                new Complex(0, 1), new Complex(0, 10), new Complex(0, 100), new Complex(0, 1000)
            });
        }

        [TestMethod]
        public void Geomspace_Complex_Circle()
        {
            // geomspace(-1+0j, 1+0j, 5): a half-circle through i.
            AllcloseC128(np.geomspace(new Complex(-1, 0), new Complex(1, 0), 5), new[]
            {
                new Complex(-1.0, 1.2246467991473532e-16),
                new Complex(-0.7071067811865476, 0.7071067811865476),
                new Complex(6.123233995736766e-17, 1.0),
                new Complex(0.7071067811865475, 0.7071067811865476),
                new Complex(1.0, 0.0)
            });
        }

        [TestMethod]
        public void Geomspace_Complex_Dtype_On_Real_Inputs()
        {
            // dtype=complex switches the computation into the complex128 domain; the interior differs slightly from
            // the real path (NumPy's [1] here is 1.9999999999999998, not the real path's exact 2.0).
            AllcloseC128(np.geomspace(1.0, 8.0, 4, dtype: np.complex128), new[]
            {
                new Complex(1.0, 0), new Complex(1.9999999999999998, 0),
                new Complex(3.999999999999999, 0), new Complex(8.0, 0)
            }, rtol: 1e-12);
        }

        // ========================================================== errors & precedence

        [TestMethod]
        public void Errors_SampleCount_Negative()
        {
            new Action(() => np.logspace(1.0, 3.0, -1))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -1, must be non-negative.");
            new Action(() => np.geomspace(1.0, 8.0, -1))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -1, must be non-negative.");
        }

        [TestMethod]
        public void Errors_Geomspace_Zero()
        {
            new Action(() => np.geomspace(0.0, 1.0, 4))
                .Should().Throw<ValueError>().WithMessage("Geometric sequence cannot include zero");
            new Action(() => np.geomspace(1.0, 0.0, 4))
                .Should().Throw<ValueError>().WithMessage("Geometric sequence cannot include zero");
        }

        [TestMethod]
        public void Errors_Geomspace_Precedence_ZeroBeatsNum()
        {
            // NumPy checks the zero endpoint BEFORE the sample count (and both before the axis).
            new Action(() => np.geomspace(0.0, 1.0, -1))
                .Should().Throw<ValueError>().WithMessage("Geometric sequence cannot include zero");
        }
    }
}
