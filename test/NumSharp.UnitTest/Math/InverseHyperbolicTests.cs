using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Maths
{
    /// <summary>
    /// np.arcsinh / np.arccosh / np.arctanh (+ the Array-API aliases asinh/acosh/atanh).
    /// All expected values verified against NumPy 2.4.2.
    ///
    /// Real float32/float64 are BYTE-IDENTICAL to NumPy (Math.Asinh/Acosh/Atanh call the same MSVC
    /// ucrtbase CRT as npy_asinh/acosh/atanh — verified 0-diff over 4521 adversarial inputs at both
    /// widths). Complex matches NumPy within the documented 3-ULP complex-unary envelope, derived from
    /// the byte-exact Asin/Acos/Atan(=Catanh) ports via the exact msun involution I*conj(.).
    /// </summary>
    [TestClass]
    public class InverseHyperbolicTests
    {
        private static void AssertComplex(Complex actual, double re, double im, double tol = 1e-12)
        {
            actual.Real.Should().BeApproximately(re, tol);
            actual.Imaginary.Should().BeApproximately(im, tol);
        }

        // ---------------------------------------------------------------- real float64

        [TestMethod]
        public void Real_Arcsinh_Float64()
        {
            var x = np.array(new[] { 0.0, 1.0, 2.0, 0.5, -0.5, 3.0 });
            var r = np.arcsinh(x);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(0.0);
            r.GetAtIndex<double>(1).Should().Be(0.881373587019543);
            r.GetAtIndex<double>(2).Should().Be(1.4436354751788103);
            r.GetAtIndex<double>(3).Should().Be(0.48121182505960347);
            r.GetAtIndex<double>(4).Should().Be(-0.48121182505960347);
            r.GetAtIndex<double>(5).Should().Be(1.8184464592320668);
        }

        [TestMethod]
        public void Real_Arccosh_Float64_DomainAndEdges()
        {
            var x = np.array(new[] { 1.0, 2.0, 3.0 });
            var r = np.arccosh(x);
            r.GetAtIndex<double>(0).Should().Be(0.0);                 // arccosh(1) == 0
            r.GetAtIndex<double>(1).Should().Be(1.3169578969248166);
            r.GetAtIndex<double>(2).Should().Be(1.762747174039086);

            // x < 1 is out of domain -> NaN (matches NumPy)
            double.IsNaN(np.arccosh(np.array(new[] { 0.5 })).GetAtIndex<double>(0)).Should().BeTrue();
            double.IsNaN(np.arccosh(np.array(new[] { 0.0 })).GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Real_Arctanh_Float64_DomainAndEdges()
        {
            var x = np.array(new[] { 0.0, 0.5, -0.5 });
            var r = np.arctanh(x);
            r.GetAtIndex<double>(0).Should().Be(0.0);
            r.GetAtIndex<double>(1).Should().Be(0.5493061443340549);
            r.GetAtIndex<double>(2).Should().Be(-0.5493061443340549);

            // |x| == 1 -> +-inf ; |x| > 1 -> NaN (matches NumPy)
            np.arctanh(np.array(new[] { 1.0 })).GetAtIndex<double>(0).Should().Be(double.PositiveInfinity);
            np.arctanh(np.array(new[] { -1.0 })).GetAtIndex<double>(0).Should().Be(double.NegativeInfinity);
            double.IsNaN(np.arctanh(np.array(new[] { 2.0 })).GetAtIndex<double>(0)).Should().BeTrue();
        }

        // ---------------------------------------------------------------- float32 byte-exact spot check

        [TestMethod]
        public void Real_Float32_ByteExact()
        {
            // Bit-for-bit with NumPy 2.4.2 (same CRT). Assert exact float32 values.
            np.arcsinh(np.array(new[] { 1.0f })).GetAtIndex<float>(0).Should().Be(0.8813736f);
            np.arccosh(np.array(new[] { 2.0f })).GetAtIndex<float>(0).Should().Be(1.316958f);
            np.arctanh(np.array(new[] { 0.5f })).GetAtIndex<float>(0).Should().Be(0.54930615f);
        }

        // ---------------------------------------------------------------- float16 BYTE-EXACT (native Half.Asinh, = NumPy astype 'e'->'f')

        [TestMethod]
        public void Float16_ByteExact()
        {
            // NumPy 2.4.2 float16 loop = (Half)asinhf((float)h); NumSharp uses native Half.Asinh
            // (also float32-internal) -> byte-identical. Bit patterns from np.arcsinh(f16).view(uint16).
            ushort Bits(NDArray r, int i) => BitConverter.HalfToUInt16Bits(r.GetAtIndex<Half>(i));

            var rs = np.arcsinh(np.array(new[] { (Half)0.0f, (Half)1.0f, (Half)2.0f }));
            rs.typecode.Should().Be(NPTypeCode.Half);
            Bits(rs, 0).Should().Be(0x0000);
            Bits(rs, 1).Should().Be(0x3b0d);
            Bits(rs, 2).Should().Be(0x3dc6);

            var rc = np.arccosh(np.array(new[] { (Half)1.0f, (Half)2.0f, (Half)4.0f }));
            Bits(rc, 0).Should().Be(0x0000);
            Bits(rc, 1).Should().Be(0x3d45);
            Bits(rc, 2).Should().Be(0x4020);

            var rt = np.arctanh(np.array(new[] { (Half)0.0f, (Half)0.5f, (Half)(-0.5f) }));
            Bits(rt, 0).Should().Be(0x0000);
            Bits(rt, 1).Should().Be(0x3865);
            Bits(rt, 2).Should().Be(0xb865);
        }

        // ---------------------------------------------------------------- overload resolution (all forms unambiguous)

        [TestMethod]
        public void Overloads_AllFormsResolve()
        {
            var x = np.array(new[] { 0.5, 1.0, 2.0 });
            // (x) ; (x, out) ; (x, out, where) ; (x, out, where, dtype) ; (x, dtype:) ; (x, NPTypeCode) ; (x, Type)
            np.arcsinh(x).typecode.Should().Be(NPTypeCode.Double);
            np.arcsinh(x, np.zeros(3, NPTypeCode.Double)).typecode.Should().Be(NPTypeCode.Double);
            np.arcsinh(x, np.zeros(3, NPTypeCode.Double), np.array(new[] { true, false, true })).typecode.Should().Be(NPTypeCode.Double);
            np.arcsinh(x, dtype: NPTypeCode.Single).typecode.Should().Be(NPTypeCode.Single);
            np.arcsinh(x, NPTypeCode.Single).typecode.Should().Be(NPTypeCode.Single);   // positional NPTypeCode overload
            np.arcsinh(x, typeof(float)).typecode.Should().Be(NPTypeCode.Single);        // positional Type overload
            // aliases resolve the same 3 arg-shapes
            np.asinh(x, NPTypeCode.Single).typecode.Should().Be(NPTypeCode.Single);
            np.acosh(np.array(new[] { 2.0 }), typeof(float)).typecode.Should().Be(NPTypeCode.Single);
            np.atanh(x, dtype: NPTypeCode.Single).typecode.Should().Be(NPTypeCode.Single);
        }

        // ---------------------------------------------------------------- dtype tiers (NEP50 float promotion)

        [TestMethod]
        public void Dtype_FloatPromotionTiers()
        {
            // NumPy: bool/int8/uint8 -> float16 ; int16/uint16 -> float32 ; int32+ -> float64
            np.arcsinh(np.array(new sbyte[] { 0, 1 })).typecode.Should().Be(NPTypeCode.Half);
            np.arcsinh(np.array(new byte[] { 0, 1 })).typecode.Should().Be(NPTypeCode.Half);
            np.arcsinh(np.array(new[] { true, false })).typecode.Should().Be(NPTypeCode.Half);
            np.arcsinh(np.array(new short[] { 0, 1 })).typecode.Should().Be(NPTypeCode.Single);
            np.arcsinh(np.array(new ushort[] { 0, 1 })).typecode.Should().Be(NPTypeCode.Single);
            np.arcsinh(np.array(new[] { 0, 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.arccosh(np.array(new long[] { 1, 2 })).typecode.Should().Be(NPTypeCode.Double);
            np.arctanh(np.array(new short[] { 0 })).typecode.Should().Be(NPTypeCode.Single);
        }

        // ---------------------------------------------------------------- complex128

        [TestMethod]
        public void Complex_Arcsinh()
        {
            // NumPy: np.arcsinh([1+2j, 3-4j])
            var z = np.array(new Complex[] { new(1, 2), new(3, -4) });
            var r = np.arcsinh(z);
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertComplex(r.GetAtIndex<Complex>(0), 1.4693517443681852, 1.0634400235777521);
            AssertComplex(r.GetAtIndex<Complex>(1), 2.2999140408792695, -0.9176168533514787);
        }

        [TestMethod]
        public void Complex_Arccosh()
        {
            // NumPy: np.arccosh([1+2j, 0.5+0.5j])
            var z = np.array(new Complex[] { new(1, 2), new(0.5, 0.5) });
            var r = np.arccosh(z);
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertComplex(r.GetAtIndex<Complex>(0), 1.5285709194809982, 1.1437177404024204);
            AssertComplex(r.GetAtIndex<Complex>(1), 0.5306375309525179, 1.118517879643706);
        }

        [TestMethod]
        public void Complex_Arctanh()
        {
            // NumPy: np.arctanh([1+2j, 3-4j])
            var z = np.array(new Complex[] { new(1, 2), new(3, -4) });
            var r = np.arctanh(z);
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertComplex(r.GetAtIndex<Complex>(0), 0.17328679513998632, 1.1780972450961724);
            AssertComplex(r.GetAtIndex<Complex>(1), 0.1175009073114339, -1.4099210495965755);
        }

        [TestMethod]
        public void Complex_SpecialValues()
        {
            // NumPy 2.4.2:
            //   arcsinh(inf+1j) = (inf, 0)
            //   arctanh(0+2j)   = (0, 1.1071487177940904)   [imaginary-axis cut |y|>1]
            //   arccosh(0+nan j)= (nan, nan)
            var asinh = np.arcsinh(np.array(new Complex[] { new(double.PositiveInfinity, 1.0) })).GetAtIndex<Complex>(0);
            asinh.Real.Should().Be(double.PositiveInfinity);
            asinh.Imaginary.Should().Be(0.0);

            var atanh = np.arctanh(np.array(new Complex[] { new(0.0, 2.0) })).GetAtIndex<Complex>(0);
            AssertComplex(atanh, 0.0, 1.1071487177940904);

            var acosh = np.arccosh(np.array(new Complex[] { new(0.0, double.NaN) })).GetAtIndex<Complex>(0);
            double.IsNaN(acosh.Real).Should().BeTrue();
            double.IsNaN(acosh.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void Complex_Arcsinh_SignedZeroBranchCut()
        {
            // On the imaginary-axis cut, arcsinh's real-part sign follows the input imaginary sign:
            //   NumPy: arcsinh(2+0j) = ( 1.4436354751788103,  0.0)
            //          arcsinh(2-0j) = ( 1.4436354751788103, -0.0)
            var pos = np.arcsinh(np.array(new Complex[] { new(2, 0.0) })).GetAtIndex<Complex>(0);
            AssertComplex(pos, 1.4436354751788103, 0.0);
            var neg = np.arcsinh(np.array(new Complex[] { new(2, -0.0) })).GetAtIndex<Complex>(0);
            neg.Real.Should().Be(1.4436354751788103);
            neg.Imaginary.Should().Be(0.0);
            double.IsNegative(neg.Imaginary).Should().BeTrue("NumPy arcsinh(2-0j).imag is -0.0");
        }

        // ---------------------------------------------------------------- Array-API aliases

        [TestMethod]
        public void Aliases_MatchPrimary()
        {
            // np.asinh / np.acosh / np.atanh are the NumPy 2.0 Array-API aliases (same ufunc).
            var xr = np.array(new[] { 0.3, 0.7, 1.5, 2.5 });
            np.array_equal(np.asinh(xr), np.arcsinh(xr)).Should().BeTrue();
            np.array_equal(np.acosh(np.array(new[] { 1.5, 2.5, 4.0 })), np.arccosh(np.array(new[] { 1.5, 2.5, 4.0 }))).Should().BeTrue();
            np.array_equal(np.atanh(np.array(new[] { -0.9, 0.0, 0.9 })), np.arctanh(np.array(new[] { -0.9, 0.0, 0.9 }))).Should().BeTrue();

            var zc = np.array(new Complex[] { new(1, 2), new(-3, 0.5) });
            np.array_equal(np.asinh(zc), np.arcsinh(zc)).Should().BeTrue();
            np.array_equal(np.acosh(zc), np.arccosh(zc)).Should().BeTrue();
            np.array_equal(np.atanh(zc), np.arctanh(zc)).Should().BeTrue();
        }

        // ---------------------------------------------------------------- ufunc out= / where= / dtype=

        [TestMethod]
        public void Ufunc_Out_ReturnsSameInstanceAndWrites()
        {
            var x = np.array(new[] { 0.0, 1.0, 2.0 });
            var dst = np.zeros(3, NPTypeCode.Double);
            var r = np.arcsinh(x, @out: dst);
            ReferenceEquals(r, dst).Should().BeTrue("ufunc out= returns the out array as-is");
            dst.GetAtIndex<double>(1).Should().Be(0.881373587019543);
        }

        [TestMethod]
        public void Ufunc_Where_LeavesMaskedSlotsUntouched()
        {
            var x = np.array(new[] { 1.0, 2.0, 3.0, 4.0 });
            var dst = np.full(4, -1.0, NPTypeCode.Double);
            var mask = np.array(new[] { true, false, true, false });
            np.arctanh(np.array(new[] { 0.5, 0.5, 0.25, 0.25 }), @out: dst, where: mask);
            dst.GetAtIndex<double>(0).Should().Be(0.5493061443340549);   // computed
            dst.GetAtIndex<double>(1).Should().Be(-1.0);                 // masked off, kept prior
            dst.GetAtIndex<double>(2).Should().Be(0.25541281188299536);  // computed
            dst.GetAtIndex<double>(3).Should().Be(-1.0);                 // masked off, kept prior
        }

        [TestMethod]
        public void Ufunc_Dtype_SelectsLoopPrecision()
        {
            // dtype= selects the LOOP (computation precision), like NumPy.
            var x = np.array(new[] { 2.0 });
            np.arccosh(x, dtype: NPTypeCode.Single).typecode.Should().Be(NPTypeCode.Single);
            np.arccosh(x, dtype: NPTypeCode.Single).GetAtIndex<float>(0).Should().Be(1.316958f);

            // An integer dtype= is a no-loop error (float-only ufunc), matching NumPy.
            Action bad = () => np.arcsinh(x, dtype: NPTypeCode.Int32);
            bad.Should().Throw<Exception>();
        }

        // ---------------------------------------------------------------- non-contiguous layouts

        [TestMethod]
        public void NonContiguous_ReversedAndStrided()
        {
            // reversed view
            var x = np.array(new[] { 0.0, 0.5, 1.0, 1.5, 2.0 });
            var rev = np.arcsinh(x["::-1"]);
            rev.GetAtIndex<double>(0).Should().Be(np.arcsinh(np.array(new[] { 2.0 })).GetAtIndex<double>(0));
            rev.GetAtIndex<double>(4).Should().Be(0.0);

            // 2-D transposed
            var m = np.arange(6).reshape(2, 3).astype(NPTypeCode.Double);
            var t = m.T;                              // (3,2), F-order view
            var rt = np.arcsinh(t);
            rt.shape.Should().Equal(3, 2);
            rt.GetData(new[] { 0, 0 }).GetAtIndex<double>(0).Should().Be(np.arcsinh(np.array(new[] { 0.0 })).GetAtIndex<double>(0));
            rt.GetData(new[] { 2, 1 }).GetAtIndex<double>(0).Should().Be(np.arcsinh(np.array(new[] { 5.0 })).GetAtIndex<double>(0));
        }

        // ---------------------------------------------------------------- np.evaluate fusion (NumSharp extension)

        [TestMethod]
        public void Evaluate_FusesInverseHyperbolic()
        {
            var a = np.array(new[] { 0.0, 1.0, 2.0, 0.5 });
            np.evaluate(NDExpr.Asinh((NDExpr)a)).GetAtIndex<double>(1).Should().Be(np.arcsinh(a).GetAtIndex<double>(1));
            np.evaluate(NDExpr.Atanh((NDExpr)a)).GetAtIndex<double>(3).Should().Be(np.arctanh(a).GetAtIndex<double>(3));

            var b = np.array(new[] { 1.0, 2.0, 3.0 });
            np.evaluate(NDExpr.Acosh((NDExpr)b)).GetAtIndex<double>(1).Should().Be(np.arccosh(b).GetAtIndex<double>(1));
        }
    }
}
