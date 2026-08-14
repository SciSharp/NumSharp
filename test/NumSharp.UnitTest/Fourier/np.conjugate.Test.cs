using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     np.conjugate / np.conj — the FFT prerequisite (used by hfft/ihfft). Values/dtypes probed
    ///     against NumPy 2.4.2. Real dtypes are an identity with the dtype PRESERVED — with ONE NumPy
    ///     quirk: <c>conjugate</c> has no bool loop, so a bool input resolves to the int8 loop and yields
    ///     <c>int8</c> (values 0/1), NOT bool. Complex flips the imaginary sign (signed-zero exact).
    ///     Mirrors NumPy's ufunc surface <c>conjugate(x, /, out=None, *, where=True, dtype=None)</c>.
    /// </summary>
    [TestClass]
    public class NpConjugateTest : TestClass
    {
        // =====================================================================================
        // Complex — imaginary-sign flip (incl. signed zero)
        // =====================================================================================

        [TestMethod]
        public void Conjugate_Complex_FlipsImaginarySign()
        {
            // np.conjugate([1+2j, 3-4j, -5+0j]) == [1-2j, 3+4j, -5-0j]
            var a = new NDArray(new[] { new Complex(1, 2), new Complex(3, -4), new Complex(-5, 0) });
            var r = np.conjugate(a);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.Data<Complex>().Should().BeEquivalentTo(
                new[] { new Complex(1, -2), new Complex(3, 4), new Complex(-5, 0) });
        }

        [TestMethod]
        public void Conjugate_Complex_SignedZero_ImaginaryIsNegativeZero()
        {
            // conj(-5+0j) == -5-0j : the imaginary part is NEGATIVE zero (bit 0x8000000000000000).
            var r = np.conjugate(new NDArray(new[] { new Complex(-5, 0.0) }));
            var im = r.Data<Complex>()[0].Imaginary;
            im.Should().Be(0.0);
            double.IsNegative(im).Should().BeTrue("conj of +0.0 imaginary is -0.0");
        }

        [TestMethod]
        public void Conj_Alias_MatchesConjugate()
        {
            var a = new NDArray(new[] { new Complex(2, 7), new Complex(-1, -1) });
            np.conj(a).Data<Complex>().Should().BeEquivalentTo(
                new[] { new Complex(2, -7), new Complex(-1, 1) });
        }

        // =====================================================================================
        // Real dtypes — identity, dtype PRESERVED (except bool -> int8, the NumPy quirk)
        // =====================================================================================

        [TestMethod]
        public void Conjugate_Bool_PromotesToInt8_MatchesNumpy()
        {
            // NumPy: np.conjugate(np.array([True,False,True])) -> dtype int8, values [1,0,1].
            // conjugate has NO bool loop, so bool resolves to the int8 loop.
            var a = new NDArray(new[] { true, false, true });
            var r = np.conjugate(a);
            r.typecode.Should().Be(NPTypeCode.SByte);   // int8, NOT bool
            r.Data<sbyte>().Should().BeEquivalentTo(new sbyte[] { 1, 0, 1 });

            np.conj(a).typecode.Should().Be(NPTypeCode.SByte);
        }

        [TestMethod]
        public void Conjugate_Int32_IsIdentity_PreservesDtype()
        {
            var a = new NDArray(new[] { 1, -2, 3 });
            var r = np.conjugate(a);
            r.typecode.Should().Be(NPTypeCode.Int32);
            r.Data<int>().Should().BeEquivalentTo(new[] { 1, -2, 3 });
        }

        [TestMethod]
        public void Conjugate_Double_IsIdentity()
        {
            var a = new NDArray(new[] { 1.5, -2.5, 0.0 });
            var r = np.conjugate(a);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.Data<double>().Should().BeEquivalentTo(new[] { 1.5, -2.5, 0.0 });
        }

        [DataTestMethod]
        // every real dtype OTHER than bool preserves its dtype through conjugate (identity).
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        public void Conjugate_RealDtype_IsIdentity_PreservesDtype(NPTypeCode dt)
        {
            var a = new NDArray(new[] { 1, 2, 3 }).astype(dt);
            var r = np.conjugate(a);
            r.typecode.Should().Be(dt);
            // values unchanged: compare through a common double view.
            r.astype(np.float64).Data<double>().Should().BeEquivalentTo(new[] { 1.0, 2.0, 3.0 });
        }

        // =====================================================================================
        // out=
        // =====================================================================================

        [TestMethod]
        public void Conjugate_Out_WritesIntoProvidedArrayAndReturnsIt()
        {
            var a = new NDArray(new[] { new Complex(1, 2), new Complex(3, 4) });
            var outp = new NDArray(np.complex128, 2);
            var r = np.conjugate(a, outp);
            ReferenceEquals(r, outp).Should().BeTrue();
            outp.Data<Complex>().Should().BeEquivalentTo(
                new[] { new Complex(1, -2), new Complex(3, -4) });
        }

        [TestMethod]
        public void Conjugate_Out_InPlace_Works()
        {
            // out= aliasing the input (conjugate in place) is safe (element-wise, no cross-dependency).
            var a = new NDArray(new[] { new Complex(1, 2), new Complex(3, -4) });
            var r = np.conjugate(a, a);
            ReferenceEquals(r, a).Should().BeTrue();
            a.Data<Complex>().Should().BeEquivalentTo(new[] { new Complex(1, -2), new Complex(3, 4) });
        }

        // =====================================================================================
        // where= (NumPy ufunc mask — masked-off slots keep the prior out contents)
        // =====================================================================================

        [TestMethod]
        public void Conjugate_Where_OnlyWritesMaskTrue()
        {
            // np.conjugate([1+2j,3+4j,5+6j], out=[9+9j]*3, where=[T,F,T]) == [1-2j, 9+9j, 5-6j]
            var x = new NDArray(new[] { new Complex(1, 2), new Complex(3, 4), new Complex(5, 6) });
            var o = new NDArray(new[] { new Complex(9, 9), new Complex(9, 9), new Complex(9, 9) });
            var r = np.conjugate(x, @out: o, where: new NDArray(new[] { true, false, true }));
            ReferenceEquals(r, o).Should().BeTrue();
            r.Data<Complex>().Should().BeEquivalentTo(
                new[] { new Complex(1, -2), new Complex(9, 9), new Complex(5, -6) });
        }

        // =====================================================================================
        // dtype= (loop selection)
        // =====================================================================================

        [TestMethod]
        public void Conjugate_Dtype_SelectsLoop()
        {
            // conjugate(int32, dtype=Double) -> float64 identity (NumPy parity).
            var r = np.conjugate(new NDArray(new[] { 1, 2, 3 }), dtype: NPTypeCode.Double);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.Data<double>().Should().BeEquivalentTo(new[] { 1.0, 2.0, 3.0 });
        }

        [TestMethod]
        public void Conjugate_Bool_Dtype_OverridesInt8Promotion()
        {
            // An explicit dtype= selects the loop, so conjugate(bool, dtype=Double) -> float64 (not int8).
            var r = np.conjugate(new NDArray(new[] { true, false, true }), dtype: NPTypeCode.Double);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.Data<double>().Should().BeEquivalentTo(new[] { 1.0, 0.0, 1.0 });
        }

        // =====================================================================================
        // memory layouts — reads through strides in logical (C) order
        // =====================================================================================

        [TestMethod]
        public void Conjugate_TransposedComplexView_ReadsInLogicalOrder()
        {
            // np.conjugate([[1+1j,2-2j],[3+3j,4-4j]].T) == [[1-1j,3-3j],[2+2j,4+4j]]
            var a = new NDArray(new[]
            {
                new Complex(1, 1), new Complex(2, -2),
                new Complex(3, 3), new Complex(4, -4)
            }).reshape(2, 2);
            var r = np.conjugate(a.T);
            np.ascontiguousarray(r).Data<Complex>().Should().BeEquivalentTo(new[]
            {
                new Complex(1, -1), new Complex(3, -3),
                new Complex(2, 2), new Complex(4, 4)
            });
        }

        [TestMethod]
        public void Conjugate_ReversedView_MatchesNumpy()
        {
            // np.conjugate([1+1j,2-2j,3+3j,4-4j][::-1]) == [4+4j, 3-3j, 2+2j, 1-1j]
            var a = new NDArray(new[]
            {
                new Complex(1, 1), new Complex(2, -2), new Complex(3, 3), new Complex(4, -4)
            })["::-1"];
            var r = np.conjugate(a);
            np.ascontiguousarray(r).Data<Complex>().Should().BeEquivalentTo(new[]
            {
                new Complex(4, 4), new Complex(3, -3), new Complex(2, 2), new Complex(1, -1)
            });
        }

        [TestMethod]
        public void Conjugate_BroadcastView_MatchesNumpy()
        {
            // np.conjugate(broadcast_to([1+1j,2+2j], (3,2))) -> each row conjugated; shape kept.
            var a = np.broadcast_to(new NDArray(new[] { new Complex(1, 1), new Complex(2, 2) }), new Shape(3, 2));
            var r = np.conjugate(a);
            r.shape.Should().BeEquivalentTo(new[] { 3, 2 });
            np.ascontiguousarray(r).Data<Complex>().Should().BeEquivalentTo(new[]
            {
                new Complex(1, -1), new Complex(2, -2),
                new Complex(1, -1), new Complex(2, -2),
                new Complex(1, -1), new Complex(2, -2)
            });
        }

        // =====================================================================================
        // 0-d / empty
        // =====================================================================================

        [TestMethod]
        public void Conjugate_ZeroD_Complex()
        {
            var r = np.conjugate(NDArray.Scalar(new Complex(2, 3)));
            r.ndim.Should().Be(0);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(2, -3));
        }

        [TestMethod]
        public void Conjugate_Empty_ReturnsEmptySameDtype()
        {
            var a = new NDArray(np.complex128, 0);
            var r = np.conjugate(a);
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Conjugate_EmptyBool_ReturnsEmptyInt8()
        {
            // even for an empty array the bool->int8 dtype resolution holds.
            var r = np.conjugate(new NDArray(NPTypeCode.Boolean, new Shape(0), false));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.SByte);
        }
    }
}
