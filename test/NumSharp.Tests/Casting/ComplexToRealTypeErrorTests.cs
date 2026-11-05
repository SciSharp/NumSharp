using System;
using System.Numerics;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     Complex → integer / float scalar casts throw <see cref="TypeError"/> with CPython's texts, as NumPy 2.4.2's
    ///     <c>int(arr)</c> / <c>float(arr)</c> do for a complex 0-d array (<c>int() argument must be a string, a bytes-like
    ///     object or a real number, not 'complex'</c>; <c>float() argument must be a string or a real number, not
    ///     'complex'</c>); NumSharp has no ComplexWarning, so the silent imaginary drop of the array-level cast stays an
    ///     error here.
    ///
    ///     The rule applies regardless of whether the imaginary part is zero —
    ///     NumPy also throws for <c>int(np.array(3+0j))</c>.
    ///
    ///     The rule does NOT apply to:
    ///     <list type="bullet">
    ///       <item>Complex → bool: NumPy's <c>bool(arr)</c> is the element's truth (either part nonzero; parity QA-42 —
    ///             it used to throw)</item>
    ///       <item>Complex → Complex (identity, always OK)</item>
    ///       <item>Any non-Complex → Complex (widening, always OK)</item>
    ///       <item><c>nd.astype(Complex)</c> from any type (array-level cast, separate path)</item>
    ///     </list>
    /// </summary>
    [TestClass]
    public class ComplexToRealTypeErrorTests
    {
        /// <summary>CPython's <c>int()</c> refusal of a complex, which NumPy's <c>int(arr)</c> raises.</summary>
        private const string IntText = "int() argument must be a string, a bytes-like object or a real number, not 'complex'";
        /// <summary>CPython's <c>float()</c> refusal of a complex, which NumPy's <c>float(arr)</c> raises.</summary>
        private const string FloatText = "float() argument must be a string or a real number, not 'complex'";

        private static NDArray ComplexScalar(double real, double imag) =>
            NDArray.Scalar<Complex>(new Complex(real, imag));

        /// <summary>Asserts that a conversion raises <see cref="TypeError"/> with exactly <paramref name="text"/>.</summary>
        /// <param name="act">The conversion.</param>
        /// <param name="text">The expected message (<see cref="IntText"/> or <see cref="FloatText"/>).</param>
        private static void AssertTypeError(Action act, string text)
        {
            act.Should().Throw<TypeError>().WithMessage(text);
        }

        // ---------------------------------------------------------------------
        // Complex → each integer / float type throws (CPython's int() / float() texts)
        // ---------------------------------------------------------------------

        /// <summary>NumPy 2.4.2: <c>bool(np.array(1+2j))</c> and <c>bool(np.array(1j))</c> are True — the element's truth,
        ///     not a refusal.</summary>
        [TestMethod]
        public void Complex_To_Bool_IsTheTruth()
        {
            ((bool)ComplexScalar(1, 2)).Should().BeTrue();
            ((bool)ComplexScalar(0, 1)).Should().BeTrue("an imaginary part alone is nonzero");
        }

        /// <summary><c>(sbyte)</c> of a complex is CPython's <c>int()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_SByte_Throws() => AssertTypeError(() => { var _ = (sbyte)ComplexScalar(1, 2); }, IntText);

        /// <summary><c>(byte)</c> of a complex is CPython's <c>int()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_Byte_Throws()  => AssertTypeError(() => { var _ = (byte)ComplexScalar(1, 2); }, IntText);

        /// <summary><c>(short)</c> of a complex is CPython's <c>int()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_Short_Throws() => AssertTypeError(() => { var _ = (short)ComplexScalar(1, 2); }, IntText);

        /// <summary><c>(ushort)</c> of a complex is CPython's <c>int()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_UShort_Throws() => AssertTypeError(() => { var _ = (ushort)ComplexScalar(1, 2); }, IntText);

        /// <summary><c>(int)</c> of a complex is CPython's <c>int()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_Int_Throws() => AssertTypeError(() => { var _ = (int)ComplexScalar(1, 2); }, IntText);

        /// <summary><c>(uint)</c> of a complex is CPython's <c>int()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_UInt_Throws() => AssertTypeError(() => { var _ = (uint)ComplexScalar(1, 2); }, IntText);

        /// <summary><c>(long)</c> of a complex is CPython's <c>int()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_Long_Throws() => AssertTypeError(() => { var _ = (long)ComplexScalar(1, 2); }, IntText);

        /// <summary><c>(ulong)</c> of a complex is CPython's <c>int()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_ULong_Throws() => AssertTypeError(() => { var _ = (ulong)ComplexScalar(1, 2); }, IntText);

        /// <summary><c>(char)</c> of a complex is CPython's <c>int()</c> TypeError (char is an integer lane).</summary>
        [TestMethod]
        public void Complex_To_Char_Throws() => AssertTypeError(() => { var _ = (char)ComplexScalar(1, 2); }, IntText);

        /// <summary><c>(float)</c> of a complex is CPython's <c>float()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_Float_Throws() => AssertTypeError(() => { var _ = (float)ComplexScalar(1, 2); }, FloatText);

        /// <summary><c>(double)</c> of a complex is CPython's <c>float()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_Double_Throws() => AssertTypeError(() => { var _ = (double)ComplexScalar(1, 2); }, FloatText);

        /// <summary><c>(Half)</c> of a complex is CPython's <c>float()</c> TypeError.</summary>
        [TestMethod]
        public void Complex_To_Half_Throws() => AssertTypeError(() => { var _ = (Half)ComplexScalar(1, 2); }, FloatText);

        /// <summary><c>(decimal)</c> of a complex is CPython's <c>float()</c> TypeError (decimal is a float lane).</summary>
        [TestMethod]
        public void Complex_To_Decimal_Throws() => AssertTypeError(() => { var _ = (decimal)ComplexScalar(1, 2); }, FloatText);

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

        /// <summary>NumPy 2.4.2: <c>bool(np.array(0j))</c> is False (<c>-0j</c> too), <c>bool(np.array(complex(nan,
        ///     0)))</c> is True — NaN != 0 counts as nonzero.</summary>
        [TestMethod]
        public void Complex_Zero_To_Bool_IsFalse_AndANaNPartIsTrue()
        {
            ((bool)ComplexScalar(0, 0)).Should().BeFalse();
            ((bool)ComplexScalar(-0.0, -0.0)).Should().BeFalse();
            ((bool)ComplexScalar(double.NaN, 0)).Should().BeTrue();
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

        /// <summary>
        ///     The rank check runs before the complex-source check, as NumPy's <c>int()</c> does: a 1-D complex array is refused
        ///     for its rank (<c>only 0-dimensional arrays can be converted to Python scalars</c>), not with <c>int()</c>'s complex text (parity QA-14 made the rank refusal NumPy's
        ///     TypeError; it was an IncorrectShapeException).
        /// </summary>
        [TestMethod]
        public void OneD_Complex_To_Int_Throws_TheRankTypeError_First()
        {
            var arr = np.array(new Complex[] { new Complex(1, 2) });
            Action act = () => { var _ = (int)arr; };
            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }
    }
}
