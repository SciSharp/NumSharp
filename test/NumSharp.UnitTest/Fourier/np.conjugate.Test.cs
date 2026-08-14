using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     np.conjugate / np.conj — the FFT prerequisite. Values probed against NumPy 2.4.2.
    ///     Real dtypes are an identity (values + dtype preserved); Complex flips the imaginary sign.
    /// </summary>
    [TestClass]
    public class NpConjugateTest : TestClass
    {
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
        public void Conj_Alias_MatchesConjugate()
        {
            var a = new NDArray(new[] { new Complex(2, 7), new Complex(-1, -1) });
            np.conj(a).Data<Complex>().Should().BeEquivalentTo(
                new[] { new Complex(2, -7), new Complex(-1, 1) });
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

        [TestMethod]
        public void Conjugate_Bool_IsIdentity_PreservesDtype()
        {
            var a = new NDArray(new[] { true, false, true });
            var r = np.conjugate(a);
            r.typecode.Should().Be(NPTypeCode.Boolean);
            r.Data<bool>().Should().BeEquivalentTo(new[] { true, false, true });
        }

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
        public void Conjugate_TransposedComplexView_ReadsInLogicalOrder()
        {
            // Non-contiguous (transposed) source must conjugate through its own strides.
            // np.conjugate([[1+1j,2-2j],[3+3j,4-4j]].T) == [[1-1j,3-3j],[2+2j,4+4j]]
            var a = new NDArray(new[]
            {
                new Complex(1, 1), new Complex(2, -2),
                new Complex(3, 3), new Complex(4, -4)
            }).reshape(2, 2);
            var r = np.conjugate(a.T);
            r.Data<Complex>().Should().BeEquivalentTo(new[]
            {
                new Complex(1, -1), new Complex(3, -3),
                new Complex(2, 2), new Complex(4, 4)
            });
        }

        [TestMethod]
        public void Conjugate_Empty_ReturnsEmptySameDtype()
        {
            var a = new NDArray(np.complex128, 0);
            var r = np.conjugate(a);
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Complex);
        }
    }
}
