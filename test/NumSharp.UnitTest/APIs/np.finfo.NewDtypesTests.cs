using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.APIs
{
    /// <summary>
    ///     Full NumPy 2.x parity tests for <c>np.finfo</c> on the new dtypes
    ///     (<see cref="Half"/> / float16 and <see cref="Complex"/> / complex128).
    ///
    ///     Every expectation cross-checked against
    ///     <c>python -c "import numpy as np; i = np.finfo(np.float16); print(i.bits, repr(i.eps), ...)"</c>.
    /// </summary>
    [TestClass]
    public class NpFInfoNewDtypesTests
    {
        // ---------------------------------------------------------------------
        // finfo(Half) / finfo(float16) / finfo("float16") / finfo("half")
        // ---------------------------------------------------------------------

        [TestMethod]
        public void FInfo_Half_Bits()
            => np.finfo(NPTypeCode.Half).bits.Should().Be(16);

        [TestMethod]
        public void FInfo_Half_Eps()
        {
            // NumPy: np.finfo(np.float16).eps == 0.000977 (2^-10)
            var eps = np.finfo(NPTypeCode.Half).eps;
            eps.Should().Be(0.0009765625);  // exact 2^-10
        }

        [TestMethod]
        public void FInfo_Half_EpsNeg()
        {
            // NumPy: 0.0004883 (2^-11)
            np.finfo(NPTypeCode.Half).epsneg.Should().Be(0.00048828125);
        }

        [TestMethod]
        public void FInfo_Half_Max()
            => np.finfo(NPTypeCode.Half).max.Should().Be(65504.0);

        [TestMethod]
        public void FInfo_Half_Min()
            => np.finfo(NPTypeCode.Half).min.Should().Be(-65504.0);

        [TestMethod]
        public void FInfo_Half_SmallestNormal()
        {
            // NumPy: 6.104e-05 (2^-14)
            np.finfo(NPTypeCode.Half).smallest_normal.Should().Be(6.103515625e-05);
        }

        [TestMethod]
        public void FInfo_Half_SmallestSubnormal()
        {
            // NumPy: 6e-08 (2^-24); (double)Half.Epsilon == 5.960464477539063e-08
            np.finfo(NPTypeCode.Half).smallest_subnormal.Should().Be(5.960464477539063e-08);
        }

        [TestMethod]
        public void FInfo_Half_Tiny_Equals_SmallestNormal()
        {
            var i = np.finfo(NPTypeCode.Half);
            i.tiny.Should().Be(i.smallest_normal);
        }

        [TestMethod]
        public void FInfo_Half_Precision()
            => np.finfo(NPTypeCode.Half).precision.Should().Be(3);

        [TestMethod]
        public void FInfo_Half_Resolution()
            => np.finfo(NPTypeCode.Half).resolution.Should().Be(1e-3);

        [TestMethod]
        public void FInfo_Half_MaxExp()
            => np.finfo(NPTypeCode.Half).maxexp.Should().Be(16);

        [TestMethod]
        public void FInfo_Half_MinExp()
            => np.finfo(NPTypeCode.Half).minexp.Should().Be(-14);

        [TestMethod]
        public void FInfo_Half_Dtype()
            => np.finfo(NPTypeCode.Half).dtype.Should().Be(NPTypeCode.Half);

        [TestMethod]
        public void FInfo_Half_From_Type()
            => np.finfo(typeof(Half)).bits.Should().Be(16);

        [TestMethod]
        public void FInfo_Half_From_Generic()
            => np.finfo<Half>().bits.Should().Be(16);

        [TestMethod]
        public void FInfo_Half_From_Array()
        {
            var arr = np.array(new Half[] { (Half)1.0, (Half)2.0 });
            np.finfo(arr).bits.Should().Be(16);
        }

        [TestMethod]
        public void FInfo_Half_From_String_float16()
            => np.finfo("float16").bits.Should().Be(16);

        [TestMethod]
        public void FInfo_Half_From_String_half()
            => np.finfo("half").bits.Should().Be(16);

        [TestMethod]
        public void FInfo_Half_From_String_e()
            => np.finfo("e").bits.Should().Be(16);

        [TestMethod]
        public void FInfo_Half_From_String_f2()
            => np.finfo("f2").bits.Should().Be(16);

        // ---------------------------------------------------------------------
        // finfo(Complex) — NumPy reports the underlying float precision
        // ---------------------------------------------------------------------

        [TestMethod]
        public void FInfo_Complex_Bits_ReportsUnderlyingFloat64()
        {
            // NumPy: np.finfo(np.complex128).bits == 64, NOT 128.
            // NumSharp's Complex = System.Numerics.Complex = 2 × float64.
            np.finfo(NPTypeCode.Complex).bits.Should().Be(64);
        }

        [TestMethod]
        public void FInfo_Complex_Eps_MatchesFloat64()
        {
            np.finfo(NPTypeCode.Complex).eps.Should().Be(np.finfo(NPTypeCode.Double).eps);
        }

        [TestMethod]
        public void FInfo_Complex_Max_MatchesFloat64()
        {
            np.finfo(NPTypeCode.Complex).max.Should().Be(np.finfo(NPTypeCode.Double).max);
        }

        [TestMethod]
        public void FInfo_Complex_Min_MatchesFloat64()
        {
            np.finfo(NPTypeCode.Complex).min.Should().Be(np.finfo(NPTypeCode.Double).min);
        }

        [TestMethod]
        public void FInfo_Complex_Precision_MatchesFloat64()
        {
            np.finfo(NPTypeCode.Complex).precision.Should().Be(15);
        }

        [TestMethod]
        public void FInfo_Complex_Resolution_MatchesFloat64()
        {
            np.finfo(NPTypeCode.Complex).resolution.Should().Be(1e-15);
        }

        [TestMethod]
        public void FInfo_Complex_MaxExp()
            => np.finfo(NPTypeCode.Complex).maxexp.Should().Be(1024);

        [TestMethod]
        public void FInfo_Complex_MinExp()
            => np.finfo(NPTypeCode.Complex).minexp.Should().Be(-1021);

        [TestMethod]
        public void FInfo_Complex_SmallestNormal_MatchesFloat64()
        {
            np.finfo(NPTypeCode.Complex).smallest_normal.Should().Be(2.2250738585072014e-308);
        }

        [TestMethod]
        public void FInfo_Complex_SmallestSubnormal_MatchesFloat64()
        {
            np.finfo(NPTypeCode.Complex).smallest_subnormal.Should().Be(double.Epsilon);
        }

        [TestMethod]
        public void FInfo_Complex_Dtype_ReportsUnderlyingFloat()
        {
            // NumPy parity: np.finfo(np.complex128).dtype == np.float64
            np.finfo(NPTypeCode.Complex).dtype.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void FInfo_Complex_From_Type()
            => np.finfo(typeof(Complex)).bits.Should().Be(64);

        [TestMethod]
        public void FInfo_Complex_From_Generic()
            => np.finfo<Complex>().bits.Should().Be(64);

        [TestMethod]
        public void FInfo_Complex_From_Array()
        {
            var arr = np.array(new Complex[] { new Complex(1, 2) });
            np.finfo(arr).bits.Should().Be(64);
        }

        [TestMethod]
        public void FInfo_Complex_From_String_complex128()
            => np.finfo("complex128").bits.Should().Be(64);

        [TestMethod]
        public void FInfo_Complex_From_String_complex()
            => np.finfo("complex").bits.Should().Be(64);

        [TestMethod]
        public void FInfo_Complex_From_String_D()
            => np.finfo("D").bits.Should().Be(64);

        [TestMethod]
        public void FInfo_Complex_From_String_c16()
            => np.finfo("c16").bits.Should().Be(64);

        [TestMethod]
        public void FInfo_Complex_From_String_complex64_Throws()
        {
            // NumSharp rejects complex64 outright — users must use complex128 / 'D' / 'c16' / 'complex'.
            Action act = () => np.finfo("complex64");
            act.Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void FInfo_Complex_From_String_c8_Throws()
        {
            Action act = () => np.finfo("c8");
            act.Should().Throw<NotSupportedException>();
        }

        // ---------------------------------------------------------------------
        // Integer types STILL throw "not inexact"
        // ---------------------------------------------------------------------

        [TestMethod]
        public void FInfo_SByte_Throws()
        {
            Action act = () => np.finfo(NPTypeCode.SByte);
            act.Should().Throw<ArgumentException>().WithMessage("*not inexact*");
        }

        [TestMethod]
        public void FInfo_Int32_Throws()
        {
            Action act = () => np.finfo(NPTypeCode.Int32);
            act.Should().Throw<ArgumentException>().WithMessage("*not inexact*");
        }

        [TestMethod]
        public void FInfo_Boolean_Throws()
        {
            Action act = () => np.finfo(NPTypeCode.Boolean);
            act.Should().Throw<ArgumentException>();
        }
    }
}
