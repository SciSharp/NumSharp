using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    ///     <c>np.einsum</c>'s CONTRACTION — the values, against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The contraction is a composition over the matrix products (a port of NumPy's
    ///     <c>bmm_einsum</c>), so every product-shaped case reduces to <see cref="np.matmul"/> and
    ///     runs through OpenBLAS when a backend is referenced — byte-identical to NumPy for
    ///     float32/float64/complex128, and byte-exact everywhere for integer/boolean (modular /
    ///     logical reduction is order-independent). Each expected array below was produced by running
    ///     the same expression through NumPy 2.4.2.
    ///     </para>
    ///     <para>
    ///     A pure float summation done OUTSIDE a product (e.g. <c>ij-&gt;i</c>, a diagonal-then-sum,
    ///     or the accumulation across three or more operands) can differ from NumPy in the last ULP
    ///     because its order follows <see cref="np.sum"/> and a left-to-right fold; the integer cases
    ///     below pin those shapes exactly, and the float ones stay allclose.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class EinsumContractionTests
    {
        private static NDArray A23 => np.arange(6).reshape(2, 3);
        private static NDArray A32 => np.arange(6).reshape(3, 2);
        private static NDArray Sq3 => np.arange(9).reshape(3, 3);

        #region core contractions

        [TestMethod]
        public void Matmul()
            => np.einsum("ij,jk->ik", A23, A32).Should().BeOfValues(10, 13, 28, 40).And.BeShaped(2, 2);

        [TestMethod]
        public void Transpose()
            => np.einsum("ij->ji", A23).Should().BeOfValues(0, 3, 1, 4, 2, 5).And.BeShaped(3, 2);

        [TestMethod]
        public void Diagonal()
            => np.einsum("ii->i", Sq3).Should().BeOfValues(0, 4, 8).And.BeShaped(3);

        [TestMethod]
        public void TraceExplicit()
            => np.einsum("ii->", Sq3).Should().BeOfValues(12).And.BeShaped();

        [TestMethod]
        public void TraceImplicit()
            => np.einsum("ii", Sq3).Should().BeOfValues(12).And.BeShaped();

        [TestMethod]
        public void RowSum()
            => np.einsum("ij->i", A23).Should().BeOfValues(3, 12).And.BeShaped(2);

        [TestMethod]
        public void TotalSum()
            => np.einsum("ij->", A23).Should().BeOfValues(15).And.BeShaped();

        [TestMethod]
        public void Outer()
            => np.einsum("i,j->ij", np.arange(3), np.arange(4))
                .Should().BeOfValues(0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6).And.BeShaped(3, 4);

        [TestMethod]
        public void Inner()
            => np.einsum("i,i->", np.arange(4), np.arange(4)).Should().BeOfValues(14).And.BeShaped();

        [TestMethod]
        public void MatVec()
            => np.einsum("ij,j->i", A23, np.arange(3)).Should().BeOfValues(5, 14).And.BeShaped(2);

        [TestMethod]
        public void Bilinear_ContractsTheLeadingAxis()
            => np.einsum("ij,ik->jk", np.arange(6).reshape(3, 2), np.arange(12).reshape(3, 4))
                .Should().BeOfValues(40, 46, 52, 58, 52, 61, 70, 79).And.BeShaped(2, 4);

        #endregion

        #region batched, tensor, multi-operand

        [TestMethod]
        public void BatchedMatmul()
            => np.einsum("bij,bjk->bik", np.arange(12).reshape(2, 2, 3), np.arange(12).reshape(2, 3, 2))
                .Should().BeOfValues(10, 13, 28, 40, 172, 193, 244, 274).And.BeShaped(2, 2, 2);

        [TestMethod]
        public void TensorContraction_OverTwoAxes()
            => np.einsum("ijk,jkl->il", np.arange(24).reshape(2, 3, 4), np.arange(60).reshape(3, 4, 5))
                .Should().BeOfValues(2530, 2596, 2662, 2728, 2794, 6490, 6700, 6910, 7120, 7330)
                .And.BeShaped(2, 5);

        [TestMethod]
        public void ThreeOperandChain()
            => np.einsum("ij,jk,kl->il", A23, A32, A23)
                .Should().BeOfValues(39, 62, 85, 120, 188, 256).And.BeShaped(2, 3);

        [TestMethod]
        public void Kron_FourIndexOuter()
            => np.einsum("ij,kl->ikjl", np.arange(4).reshape(2, 2), np.arange(4).reshape(2, 2))
                .Should().BeOfValues(0, 0, 0, 1, 0, 0, 2, 3, 0, 2, 0, 3, 4, 6, 6, 9).And.BeShaped(2, 2, 2, 2);

        #endregion

        #region ellipsis

        [TestMethod]
        public void Ellipsis_BatchedMatmul()
            => np.einsum("...ij,...jk->...ik", np.arange(12).reshape(2, 2, 3), np.arange(12).reshape(2, 3, 2))
                .Should().BeOfValues(10, 13, 28, 40, 172, 193, 244, 274).And.BeShaped(2, 2, 2);

        [TestMethod]
        public void Ellipsis_ReduceTrailingAxis()
            => np.einsum("...i->...", np.arange(24).reshape(2, 3, 4))
                .Should().BeOfValues(6, 22, 38, 54, 70, 86).And.BeShaped(2, 3);

        [TestMethod]
        public void Ellipsis_BatchedDiagonal()
            => np.einsum("...ii->...i", np.arange(18).reshape(2, 3, 3))
                .Should().BeOfValues(0, 4, 8, 9, 13, 17).And.BeShaped(2, 3);

        #endregion

        #region diagonals, hadamard, scalar

        [TestMethod]
        public void BatchTrace()
            => np.einsum("bii->b", np.arange(18).reshape(2, 3, 3)).Should().BeOfValues(12, 39).And.BeShaped(2);

        [TestMethod]
        public void DiagonalThenSum()
            => np.einsum("iij->j", np.arange(36).reshape(3, 3, 4)).Should().BeOfValues(48, 51, 54, 57).And.BeShaped(4);

        [TestMethod]
        public void Hadamard()
            => np.einsum("ij,ij->ij", A23, A23).Should().BeOfValues(0, 1, 4, 9, 16, 25).And.BeShaped(2, 3);

        [TestMethod]
        public void Frobenius()
            => np.einsum("ij,ij->", A23, A23).Should().BeOfValues(55).And.BeShaped();

        [TestMethod]
        public void ScalarMultiplication()
            => np.einsum(",ij->ij", NDArray.Scalar(3), A23).Should().BeOfValues(0, 3, 6, 9, 12, 15).And.BeShaped(2, 3);

        #endregion

        #region implicit output ordering (ASCII order — UPPER before lower)

        [TestMethod]
        public void ImplicitOutput_IsTheOnceUsedLabelsInASCIIOrder()
        {
            // No "->": labels used once, ASCII-ordered. "ji" reorders to "ij" -> a transpose.
            np.einsum("ji", A23).Should().BeOfValues(0, 3, 1, 4, 2, 5).And.BeShaped(3, 2);
            // Upper-case sorts before lower-case, so "zA" transposes and "Az" does not.
            np.einsum("zA", A23).Should().BeOfValues(0, 3, 1, 4, 2, 5).And.BeShaped(3, 2);
            np.einsum("Az", A23).Should().BeOfValues(0, 1, 2, 3, 4, 5).And.BeShaped(2, 3);
        }

        #endregion

        #region dtype

        [TestMethod]
        public void Reduction_PreservesTheOperandDtype_UnlikeNpSum()
        {
            // einsum accumulates in the operand's dtype: int32 stays int32, where np.sum would widen
            // to int64. This is the NEP50 subtlety that a naive np.sum-based reduction gets wrong.
            var r = np.einsum("ij->i", A23.astype(NPTypeCode.Int32));
            r.Should().BeOfValues(3, 12);
            r.dtype.Should().Be(typeof(int));

            np.sum(A23.astype(NPTypeCode.Int32), 1).dtype.Should().Be(typeof(long));
        }

        [TestMethod]
        public void Matmul_PreservesInt8_AndPromotesAcrossDtypes()
        {
            var i8 = np.einsum("ij,jk->ik", A23.astype(NPTypeCode.SByte), A32.astype(NPTypeCode.SByte));
            i8.Should().BeOfValues(10, 13, 28, 40);
            i8.dtype.Should().Be(typeof(sbyte));

            // result_type(int32, float64) = float64.
            var promoted = np.einsum("ij,jk->ik", A23.astype(NPTypeCode.Int32), A32.astype(NPTypeCode.Double));
            promoted.Should().BeOfValues(10.0, 13.0, 28.0, 40.0);
            promoted.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void Bool_ContractsLogically_AndStaysBool()
        {
            // NumPy: bool @ bool is a logical matrix product (OR of ANDs), and stays bool.
            var r = np.einsum("ij,jk->ik",
                np.array(new[,] {{true, false}, {true, true}}),
                np.array(new[,] {{true, true}, {false, true}}));
            r.dtype.Should().Be(typeof(bool));
            r.astype(NPTypeCode.Int64).Should().BeOfValues(1, 1, 1, 1).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void EveryNumericDtype_ContractsInThatDtype()
        {
            // The DOD dtype sweep — a plain matmul over all real numeric dtypes preserves the dtype
            // and gives NumPy's values. arange(6).reshape(2,3) @ arange(6).reshape(3,2) = [10,13,28,40].
            foreach (var tc in new[]
            {
                NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
                NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Half, NPTypeCode.Single,
                NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Char
            })
            {
                var r = np.einsum("ij,jk->ik", A23.astype(tc), A32.astype(tc));
                r.typecode.Should().Be(tc, $"einsum must preserve {tc}");
                r.astype(NPTypeCode.Int64).Should().BeOfValues(10, 13, 28, 40).And.BeShaped(2, 2);
            }
        }

        #endregion

        #region complex (unconjugated — einsum never conjugates)

        [TestMethod]
        public void Complex_InnerIsUnconjugated()
        {
            // (1+2j)^2 + (3-1j)^2 = (-3+4j) + (8-6j) = 5-2j. einsum does NOT conjugate (unlike vdot).
            var z = np.array(new Complex[] {new(1, 2), new(3, -1)});
            ((Complex)np.ravel(np.einsum("i,i->", z, z)).GetValue(0)).Should().Be(new Complex(5, -2));
        }

        [TestMethod]
        public void Complex_Matmul()
        {
            var a = np.array(new Complex[] {new(1, 2), new(3, 4)}).reshape(1, 2);
            var b = np.array(new Complex[] {new(1, 1), new(1, 1)}).reshape(2, 1);
            var r = np.einsum("ij,jk->ik", a, b);
            r.dtype.Should().Be(typeof(Complex));
            ((Complex)np.ravel(r).GetValue(0)).Should().Be(new Complex(-2, 10));
        }

        #endregion

        #region keywords: out=, dtype=, sublists

        [TestMethod]
        public void Out_ReceivesTheResult_AndIsReturned()
        {
            var dst = np.zeros(new Shape(2, 2), NPTypeCode.Int64);
            var returned = np.einsum("ij,jk->ik", new[] {A23, A32}, @out: dst);
            returned.Should().BeSameAs(dst);
            dst.Should().BeOfValues(10, 13, 28, 40);
        }

        [TestMethod]
        public void Dtype_ForcesTheAccumulationDtype()
        {
            // dtype= widens the accumulation; int32 -> int64 is a safe cast, so it is allowed.
            var r = np.einsum("ij->i", new[] {A23.astype(NPTypeCode.Int32)}, dtype: NPTypeCode.Int64);
            r.dtype.Should().Be(typeof(long));
            r.Should().BeOfValues(3, 12);
        }

        [TestMethod]
        public void Dtype_RejectsAnUnsafeCastUnderTheDefaultRule()
        {
            // int32 -> float32 is not a 'safe' cast, exactly as NumPy's einsum loop rejects it.
            new Action(() => np.einsum("ij->i", new[] {A23.astype(NPTypeCode.Int32)}, dtype: NPTypeCode.Single))
                .Should().Throw<TypeError>();
        }

        [TestMethod]
        public void SublistSpelling_Computes()
        {
            // np.einsum(a, [0,1], b, [1,2], [0,2]) is matmul.
            np.einsum(A23, new[] {0, 1}, A32, new[] {1, 2}, new[] {0, 2})
                .Should().BeOfValues(10, 13, 28, 40).And.BeShaped(2, 2);
        }

        #endregion
    }
}
