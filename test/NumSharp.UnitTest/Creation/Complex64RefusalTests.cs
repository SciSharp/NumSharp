using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     NumSharp only supports <c>complex128</c> (System.Numerics.Complex =
    ///     2 × float64). Attempts to access <c>complex64</c> through any API
    ///     must throw <see cref="NotSupportedException"/> — we do not silently
    ///     widen to complex128, because that would mask user intent and hide
    ///     precision-loss expectations.
    /// </summary>
    [TestClass]
    public class Complex64RefusalTests
    {
        [TestMethod]
        public void NpComplex64_Direct_Access_Throws()
        {
            Action act = () => { var _ = np.complex64; };
            act.Should().Throw<NotSupportedException>()
                .WithMessage("*complex64*")
                .WithMessage("*complex128*");
        }

        [TestMethod]
        public void NpCsingle_Direct_Access_Throws()
        {
            // NumPy: np.csingle is alias for complex64. NumSharp: throws like complex64.
            Action act = () => { var _ = np.csingle; };
            act.Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void NpComplex128_Direct_Access_Works() =>
            np.complex128.Should().Be(typeof(Complex));

        [TestMethod]
        public void NpCdouble_Direct_Access_Works() =>
            // NumPy: np.cdouble is alias for complex128.
            np.cdouble.Should().Be(typeof(Complex));

        [TestMethod]
        public void NpClongdouble_Direct_Access_Works() =>
            // NumPy: np.clongdouble is long-double complex; NumSharp collapses to complex128.
            np.clongdouble.Should().Be(typeof(Complex));

        [TestMethod]
        public void NpComplex_Direct_Access_Works() =>
            np.complex_.Should().Be(typeof(Complex));

        [TestMethod]
        public void Dtype_String_complex_Works_ReturnsComplex128()
        {
            // NumPy: np.dtype("complex") returns complex128 (NumPy 2.x default complex).
            np.dtype("complex").typecode.Should().Be(NPTypeCode.Complex);
            np.dtype("complex").itemsize.Should().Be(16);
        }

        [TestMethod]
        public void Dtype_String_complex64_Throws()
        {
            Action act = () => np.dtype("complex64");
            act.Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void Dtype_String_c8_Throws()
        {
            Action act = () => np.dtype("c8");
            act.Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void Dtype_String_F_Throws()
        {
            Action act = () => np.dtype("F");
            act.Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void Dtype_String_complex128_Works()
        {
            np.dtype("complex128").typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Dtype_String_D_Works()
        {
            np.dtype("D").typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Dtype_String_c16_Works()
        {
            np.dtype("c16").typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Dtype_String_complex_Works()
        {
            np.dtype("complex").typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Dtype_String_G_LongDoubleComplex_CollapsesToComplex128()
        {
            // 'G' is long-double complex in NumPy, which NumSharp collapses to Complex (128-bit).
            // NOT the same as complex64 — users explicitly asking for extended precision get the
            // best available (complex128), not 'complex64' (which is a narrower type).
            np.dtype("G").typecode.Should().Be(NPTypeCode.Complex);
        }
    }
}
