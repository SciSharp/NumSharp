using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    ///     Complex → non-Complex scalar cast must throw <see cref="TypeError"/>.
    ///     Aligns with Python's <c>int(complex)</c>/<c>float(complex)</c> TypeError semantics;
    ///     NumPy 2.x emits a ComplexWarning then silently drops imaginary, but NumSharp
    ///     has no warning mechanism and treats it as a hard error.
    ///
    ///     The rule applies regardless of whether the imaginary part is zero —
    ///     NumPy also throws for <c>int(np.complex128(3+0j))</c>.
    ///
    ///     The rule does NOT apply to:
    ///     <list type="bullet">
    ///       <item>Complex → Complex (identity, always OK)</item>
    ///       <item>Any non-Complex → Complex (widening, always OK)</item>
    ///       <item><c>nd.astype(Complex)</c> from any type (array-level cast, separate path)</item>
    ///     </list>
    /// </summary>
    [TestClass]
    public class ComplexToRealTypeErrorTests
    {
        private static NDArray ComplexScalar(double real, double imag) =>
            NDArray.Scalar<Complex>(new Complex(real, imag));

        private static void AssertTypeError(Action act, string targetType)
        {
            act.Should().Throw<TypeError>()
                .WithMessage($"*can't convert complex to {targetType}*");
        }

        // ---------------------------------------------------------------------
        // Complex → each real type throws
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Complex_To_Bool_Throws()  => AssertTypeError(() => { var _ = (bool)ComplexScalar(1, 2); }, "bool");

        [TestMethod]
        public void Complex_To_SByte_Throws() => AssertTypeError(() => { var _ = (sbyte)ComplexScalar(1, 2); }, "sbyte");

        [TestMethod]
        public void Complex_To_Byte_Throws()  => AssertTypeError(() => { var _ = (byte)ComplexScalar(1, 2); }, "byte");

        [TestMethod]
        public void Complex_To_Short_Throws() => AssertTypeError(() => { var _ = (short)ComplexScalar(1, 2); }, "short");

        [TestMethod]
        public void Complex_To_UShort_Throws() => AssertTypeError(() => { var _ = (ushort)ComplexScalar(1, 2); }, "ushort");

        [TestMethod]
        public void Complex_To_Int_Throws() => AssertTypeError(() => { var _ = (int)ComplexScalar(1, 2); }, "int");

        [TestMethod]
        public void Complex_To_UInt_Throws() => AssertTypeError(() => { var _ = (uint)ComplexScalar(1, 2); }, "uint");

        [TestMethod]
        public void Complex_To_Long_Throws() => AssertTypeError(() => { var _ = (long)ComplexScalar(1, 2); }, "long");

        [TestMethod]
        public void Complex_To_ULong_Throws() => AssertTypeError(() => { var _ = (ulong)ComplexScalar(1, 2); }, "ulong");

        [TestMethod]
        public void Complex_To_Char_Throws() => AssertTypeError(() => { var _ = (char)ComplexScalar(1, 2); }, "char");

        [TestMethod]
        public void Complex_To_Float_Throws() => AssertTypeError(() => { var _ = (float)ComplexScalar(1, 2); }, "float");

        [TestMethod]
        public void Complex_To_Double_Throws() => AssertTypeError(() => { var _ = (double)ComplexScalar(1, 2); }, "double");

        [TestMethod]
        public void Complex_To_Half_Throws() => AssertTypeError(() => { var _ = (Half)ComplexScalar(1, 2); }, "half");

        [TestMethod]
        public void Complex_To_Decimal_Throws() => AssertTypeError(() => { var _ = (decimal)ComplexScalar(1, 2); }, "decimal");

        // ---------------------------------------------------------------------
        // Zero-imaginary still throws (matches NumPy: "int(np.complex128(3+0j))" throws)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Complex_ZeroImag_To_Int_StillThrows()
        {
            Action act = () => { var _ = (int)ComplexScalar(3, 0); };
            act.Should().Throw<TypeError>();
        }

        [TestMethod]
        public void Complex_ZeroImag_To_Double_StillThrows()
        {
            Action act = () => { var _ = (double)ComplexScalar(3, 0); };
            act.Should().Throw<TypeError>();
        }

        [TestMethod]
        public void Complex_Zero_To_Bool_StillThrows()
        {
            // (bool)(0+0j) would be False in NumPy (warning), but we throw.
            Action act = () => { var _ = (bool)ComplexScalar(0, 0); };
            act.Should().Throw<TypeError>();
        }

        // ---------------------------------------------------------------------
        // Complex → Complex (identity) works
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Complex_To_Complex_Works()
        {
            var c = new Complex(3, 4);
            ((Complex)ComplexScalar(3, 4)).Should().Be(c);
        }

        // ---------------------------------------------------------------------
        // Real → Complex (widening) still works
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Int_To_Complex_Works()
        {
            var result = (Complex)NDArray.Scalar<int>(42);
            result.Real.Should().Be(42);
            result.Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void Half_To_Complex_Works()
        {
            var result = (Complex)NDArray.Scalar<Half>((Half)2.5);
            result.Real.Should().Be(2.5);
            result.Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void Double_To_Complex_Works()
        {
            var result = (Complex)NDArray.Scalar<double>(3.14);
            result.Real.Should().Be(3.14);
            result.Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void SByte_To_Complex_Works()
        {
            var result = (Complex)NDArray.Scalar<sbyte>(-42);
            result.Real.Should().Be(-42);
            result.Imaginary.Should().Be(0);
        }

        // ---------------------------------------------------------------------
        // Shape guard still fires before the type guard
        // ---------------------------------------------------------------------

        [TestMethod]
        public void OneD_Complex_To_Int_Throws_IncorrectShape_First()
        {
            // ndim != 0 check runs before complex-source check. For 1-d Complex, we want
            // IncorrectShapeException, not TypeError (shape is the more fundamental violation).
            var arr = np.array(new Complex[] { new Complex(1, 2) });
            Action act = () => { var _ = (int)arr; };
            act.Should().Throw<IncorrectShapeException>();
        }
    }
}
