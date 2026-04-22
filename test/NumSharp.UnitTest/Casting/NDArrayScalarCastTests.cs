using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    ///     Coverage for the scalar ↔ NDArray cast operators on
    ///     <see cref="NDArray"/>. Ensures NumPy 2.x parity:
    ///     <list type="bullet">
    ///       <item>scalar → NDArray is implicit (always a 0-d array).</item>
    ///       <item>NDArray → scalar is explicit and requires <c>ndim == 0</c>;
    ///             even single-element 1-d/2-d arrays throw (matches NumPy 2.x
    ///             "only 0-dimensional arrays can be converted to Python scalars").</item>
    ///     </list>
    ///     Focused on <see cref="sbyte"/>, <see cref="Half"/>, and <see cref="Complex"/>
    ///     which were missing cast operators before this branch.
    /// </summary>
    [TestClass]
    public class NDArrayScalarCastTests
    {
        // ---------------------------------------------------------------------
        // Scalar → NDArray (implicit)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Implicit_SByte_To_NDArray()
        {
            NDArray a = (sbyte)42;
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.ndim.Should().Be(0);
            a.size.Should().Be(1);
            ((sbyte)a).Should().Be((sbyte)42);
        }

        [TestMethod]
        public void Implicit_Half_To_NDArray()
        {
            NDArray a = (Half)3.5;
            a.typecode.Should().Be(NPTypeCode.Half);
            a.ndim.Should().Be(0);
            ((Half)a).Should().Be((Half)3.5);
        }

        [TestMethod]
        public void Implicit_Complex_To_NDArray()
        {
            NDArray a = new Complex(1, 2);
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.ndim.Should().Be(0);
            ((Complex)a).Should().Be(new Complex(1, 2));
        }

        // ---------------------------------------------------------------------
        // NDArray → scalar (explicit, 0-d only)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Explicit_SByte_From_ZeroD() =>
            ((sbyte)NDArray.Scalar<sbyte>(42)).Should().Be((sbyte)42);

        [TestMethod]
        public void Explicit_Half_From_ZeroD() =>
            ((Half)NDArray.Scalar<Half>((Half)2.5)).Should().Be((Half)2.5);

        [TestMethod]
        public void Explicit_Complex_From_ZeroD() =>
            ((Complex)NDArray.Scalar<Complex>(new Complex(7, 3))).Should().Be(new Complex(7, 3));

        // ---------------------------------------------------------------------
        // Boundary values
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Boundary_SByte_MaxValue() =>
            ((sbyte)NDArray.Scalar<sbyte>(sbyte.MaxValue)).Should().Be(sbyte.MaxValue);

        [TestMethod]
        public void Boundary_SByte_MinValue() =>
            ((sbyte)NDArray.Scalar<sbyte>(sbyte.MinValue)).Should().Be(sbyte.MinValue);

        [TestMethod]
        public void Boundary_Half_NaN_Preserved() =>
            Half.IsNaN((Half)NDArray.Scalar<Half>(Half.NaN)).Should().BeTrue();

        [TestMethod]
        public void Boundary_Half_PosInf_Preserved() =>
            Half.IsPositiveInfinity((Half)NDArray.Scalar<Half>(Half.PositiveInfinity)).Should().BeTrue();

        [TestMethod]
        public void Boundary_Half_NegInf_Preserved() =>
            Half.IsNegativeInfinity((Half)NDArray.Scalar<Half>(Half.NegativeInfinity)).Should().BeTrue();

        [TestMethod]
        public void Boundary_Half_MaxValue_Preserved() =>
            ((Half)NDArray.Scalar<Half>(Half.MaxValue)).Should().Be(Half.MaxValue);

        [TestMethod]
        public void Boundary_Half_MinValue_Preserved() =>
            ((Half)NDArray.Scalar<Half>(Half.MinValue)).Should().Be(Half.MinValue);

        [TestMethod]
        public void Boundary_Complex_Zero() =>
            ((Complex)NDArray.Scalar<Complex>(Complex.Zero)).Should().Be(Complex.Zero);

        [TestMethod]
        public void Boundary_Complex_One() =>
            ((Complex)NDArray.Scalar<Complex>(Complex.One)).Should().Be(Complex.One);

        [TestMethod]
        public void Boundary_Complex_ImaginaryOne() =>
            ((Complex)NDArray.Scalar<Complex>(Complex.ImaginaryOne)).Should().Be(Complex.ImaginaryOne);

        [TestMethod]
        public void Boundary_Complex_Negative() =>
            ((Complex)NDArray.Scalar<Complex>(new Complex(-3, -4))).Should().Be(new Complex(-3, -4));

        // ---------------------------------------------------------------------
        // Cross-type conversion via Converts.ChangeType
        // ---------------------------------------------------------------------

        [TestMethod]
        public void CrossType_Int32_To_Half() =>
            ((Half)NDArray.Scalar<int>(42)).Should().Be((Half)42);

        [TestMethod]
        public void CrossType_Double_To_Half()
        {
            var result = (Half)NDArray.Scalar<double>(3.14);
            // Half has ~3 sig-digit precision near 3.14: expect 3.140625
            Math.Abs((float)result - 3.14f).Should().BeLessThan(0.01f);
        }

        [TestMethod]
        public void CrossType_Int32_To_Complex()
        {
            var result = (Complex)NDArray.Scalar<int>(42);
            result.Real.Should().Be(42);
            result.Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void CrossType_Half_To_SByte() =>
            ((sbyte)NDArray.Scalar<Half>((Half)7.5)).Should().Be((sbyte)7);

        [TestMethod]
        public void CrossType_Complex_To_Half_Throws_TypeError()
        {
            // NumPy 2.4.2: float(complex) / int(complex) throws TypeError (Python semantics).
            // NumSharp treats this the same way for scalar casts — NumSharp has no warning
            // system, so we reject rather than silently discarding imaginary.
            // Use np.real(nd) to get the real component explicitly if that's intended.
            var nd = NDArray.Scalar<Complex>(new Complex(3.5, 1.7));
            Action act = () => { var _ = (Half)nd; };
            act.Should().Throw<TypeError>().WithMessage("*can't convert complex to*");
        }

        [TestMethod]
        public void CrossType_Complex_To_Half_Throws_Even_IfImaginaryZero()
        {
            // NumPy's rule applies even when imaginary == 0: int(np.complex128(3+0j)) throws.
            var nd = NDArray.Scalar<Complex>(new Complex(3.5, 0));
            Action act = () => { var _ = (Half)nd; };
            act.Should().Throw<TypeError>();
        }

        [TestMethod]
        public void CrossType_Complex_To_Int_Throws_TypeError()
        {
            var nd = NDArray.Scalar<Complex>(new Complex(3, 4));
            Action act = () => { var _ = (int)nd; };
            act.Should().Throw<TypeError>().WithMessage("*can't convert complex to int*");
        }

        [TestMethod]
        public void CrossType_Complex_To_Double_Throws_TypeError()
        {
            var nd = NDArray.Scalar<Complex>(new Complex(3, 4));
            Action act = () => { var _ = (double)nd; };
            act.Should().Throw<TypeError>();
        }

        [TestMethod]
        public void CrossType_Complex_To_SByte_Throws_TypeError()
        {
            var nd = NDArray.Scalar<Complex>(new Complex(5, 2));
            Action act = () => { var _ = (sbyte)nd; };
            act.Should().Throw<TypeError>();
        }

        [TestMethod]
        public void CrossType_Complex_To_Bool_Throws_TypeError()
        {
            var nd = NDArray.Scalar<Complex>(new Complex(1, 0));
            Action act = () => { var _ = (bool)nd; };
            act.Should().Throw<TypeError>();
        }

        [TestMethod]
        public void CrossType_Complex_To_Complex_Works()
        {
            // Complex → Complex is still allowed (no conversion needed).
            var nd = NDArray.Scalar<Complex>(new Complex(3, 4));
            ((Complex)nd).Should().Be(new Complex(3, 4));
        }

        [TestMethod]
        public void CrossType_SByte_To_Complex_ImagIsZero()
        {
            var nd = NDArray.Scalar<sbyte>(-42);
            var result = (Complex)nd;
            result.Real.Should().Be(-42);
            result.Imaginary.Should().Be(0);
        }

        // ---------------------------------------------------------------------
        // ndim != 0 must throw (NumPy 2.x strict)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void OneD_NDArray_Cast_To_SByte_Throws()
        {
            var arr = np.array(new sbyte[] { 1, 2, 3 });
            Action act = () => { var _ = (sbyte)arr; };
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void OneD_NDArray_Cast_To_Half_Throws()
        {
            var arr = np.array(new Half[] { (Half)1.0, (Half)2.0 });
            Action act = () => { var _ = (Half)arr; };
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void OneD_NDArray_Cast_To_Complex_Throws()
        {
            var arr = np.array(new Complex[] { new Complex(1, 2), new Complex(3, 4) });
            Action act = () => { var _ = (Complex)arr; };
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void OneD_SingleElement_Still_Throws_SByte()
        {
            // NumPy 2.x: np.array([42], dtype=int8) -> int(x) raises TypeError
            var arr = np.array(new sbyte[] { 42 });
            Action act = () => { var _ = (sbyte)arr; };
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void OneD_SingleElement_Still_Throws_Half()
        {
            var arr = np.array(new Half[] { (Half)3.5 });
            Action act = () => { var _ = (Half)arr; };
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void OneD_SingleElement_Still_Throws_Complex()
        {
            var arr = np.array(new Complex[] { new Complex(1, 2) });
            Action act = () => { var _ = (Complex)arr; };
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void TwoD_OneByOne_Still_Throws_SByte()
        {
            // NumPy 2.x: np.array([[42]], dtype=int8) -> int(x) raises TypeError
            var arr = np.array(new sbyte[] { 42 }).reshape(1, 1);
            Action act = () => { var _ = (sbyte)arr; };
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void TwoD_OneByOne_Still_Throws_Half()
        {
            var arr = np.array(new Half[] { (Half)3.5 }).reshape(1, 1);
            Action act = () => { var _ = (Half)arr; };
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void TwoD_OneByOne_Still_Throws_Complex()
        {
            var arr = np.array(new Complex[] { new Complex(1, 2) }).reshape(1, 1);
            Action act = () => { var _ = (Complex)arr; };
            act.Should().Throw<IncorrectShapeException>();
        }

        // ---------------------------------------------------------------------
        // Round-trip via indexing (arr[i] returns 0-d NDArray)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Indexing_SByte_RoundTrip()
        {
            var arr = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            ((sbyte)arr[0]).Should().Be((sbyte)(-128));
            ((sbyte)arr[1]).Should().Be((sbyte)(-1));
            ((sbyte)arr[2]).Should().Be((sbyte)0);
            ((sbyte)arr[3]).Should().Be((sbyte)1);
            ((sbyte)arr[4]).Should().Be((sbyte)127);
        }

        [TestMethod]
        public void Indexing_Half_RoundTrip()
        {
            var arr = np.array(new Half[] { Half.MinValue, (Half)(-1.5), Half.Zero, (Half)1.5, Half.MaxValue });
            ((Half)arr[0]).Should().Be(Half.MinValue);
            ((Half)arr[1]).Should().Be((Half)(-1.5));
            ((Half)arr[2]).Should().Be(Half.Zero);
            ((Half)arr[3]).Should().Be((Half)1.5);
            ((Half)arr[4]).Should().Be(Half.MaxValue);
        }

        [TestMethod]
        public void Indexing_Complex_RoundTrip()
        {
            var arr = np.array(new Complex[] { new Complex(1, 2), new Complex(-3, -4), Complex.Zero, Complex.One, Complex.ImaginaryOne });
            ((Complex)arr[0]).Should().Be(new Complex(1, 2));
            ((Complex)arr[1]).Should().Be(new Complex(-3, -4));
            ((Complex)arr[2]).Should().Be(Complex.Zero);
            ((Complex)arr[3]).Should().Be(Complex.One);
            ((Complex)arr[4]).Should().Be(Complex.ImaginaryOne);
        }

        [TestMethod]
        public void Indexing_TwoD_SByte_RoundTrip()
        {
            var arr = np.array(new sbyte[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
            ((sbyte)arr[1, 2]).Should().Be((sbyte)6);
            ((sbyte)arr[0, 0]).Should().Be((sbyte)1);
        }

        [TestMethod]
        public void Indexing_TwoD_Half_RoundTrip()
        {
            var arr = np.array(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 }).reshape(2, 2);
            ((Half)arr[0, 1]).Should().Be((Half)2);
            ((Half)arr[1, 0]).Should().Be((Half)3);
        }

        // ---------------------------------------------------------------------
        // Implicit scalar conversions compose with operations
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Implicit_SByte_Used_In_Arithmetic()
        {
            NDArray a = (sbyte)10;
            NDArray b = (sbyte)5;
            // sbyte + sbyte → int8 (NumPy: same dtype preserved for same-kind)
            var sum = a + b;
            sum.typecode.Should().Be(NPTypeCode.SByte);
            ((sbyte)sum).Should().Be((sbyte)15);
        }

        [TestMethod]
        public void Implicit_Half_Used_In_Arithmetic()
        {
            NDArray a = (Half)10;
            NDArray b = (Half)5;
            var sum = a + b;
            sum.typecode.Should().Be(NPTypeCode.Half);
            ((Half)sum).Should().Be((Half)15);
        }

        [TestMethod]
        public void Implicit_Complex_Used_In_Arithmetic()
        {
            NDArray a = new Complex(3, 4);
            NDArray b = new Complex(1, 2);
            var sum = a + b;
            sum.typecode.Should().Be(NPTypeCode.Complex);
            ((Complex)sum).Should().Be(new Complex(4, 6));
        }
    }
}
