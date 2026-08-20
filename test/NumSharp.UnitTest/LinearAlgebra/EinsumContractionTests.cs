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

        #region the view path (single operand, no out=, nothing summed)

        [TestMethod]
        public void ViewPath_DiagonalIsAWriteableView_ThatWritesThrough()
        {
            // NumPy's documented idiom: np.einsum('ii->i', a)[:] = 1 sets a's diagonal. The view is
            // WRITEABLE (np.diagonal's own contract is read-only — einsum restores the flag).
            var a = np.arange(9.0).reshape(3, 3);
            var d = np.einsum("ii->i", a);
            d.Shape.IsWriteable.Should().BeTrue();

            d.SetValue(99.0, 1);
            a.GetValue<double>(1, 1).Should().Be(99.0, "the view shares a's storage");
        }

        [TestMethod]
        public void ViewPath_TransposeAndIdentity_AreDistinctViewsThatShareStorage()
        {
            var a = np.arange(6.0).reshape(2, 3);

            var t = np.einsum("ij->ji", a);
            t.Shape.IsWriteable.Should().BeTrue();
            t.SetValue(50.0, 2, 0);
            a.GetValue<double>(0, 2).Should().Be(50.0);

            // einsum('ij->ij', a) is a VIEW of a, never a itself — NumPy returns a distinct object,
            // and handing back the operand would let result.resize() mutate it.
            var r = np.einsum("ij->ij", a);
            ReferenceEquals(r, a).Should().BeFalse();
            r.SetValue(70.0, 1, 1);
            a.GetValue<double>(1, 1).Should().Be(70.0);
        }

        [TestMethod]
        public void ViewPath_BroadcastInputStaysReadOnly()
        {
            // Writeable IFF the operand is: a broadcast view is read-only, and so is its diagonal.
            var bc = np.broadcast_to(np.arange(3.0), new Shape(3, 3));
            np.einsum("ii->i", bc).Shape.IsWriteable.Should().BeFalse();
        }

        [TestMethod]
        public void ViewPath_IgnoresOrderAndDtypeRequests()
        {
            // Probed on 2.4.2: the view attempt wins over BOTH keywords — order='F'/'C' still
            // returns the same non-contiguous view, and dtype=int64 on an int32 operand returns the
            // int32 view rather than a widened copy.
            var a = np.arange(9.0).reshape(3, 3);
            var d = np.einsum("ii->i", new[] {a}, order: 'F');
            d.Shape.IsContiguous.Should().BeFalse("the diagonal stride is n+1; NumPy does not copy it for order=");
            d.SetValue(42.0, 0);
            a.GetValue<double>(0, 0).Should().Be(42.0);

            var i32 = np.arange(9).astype(NPTypeCode.Int32).reshape(3, 3);
            np.einsum("ii->i", new[] {i32}, dtype: NPTypeCode.Int64).typecode.Should().Be(NPTypeCode.Int32);
        }

        #endregion

        #region casting= gates every cast, with NumPy's iterator texts

        [TestMethod]
        public void CastingNo_RejectsThePromotionCast_Verbatim()
        {
            new Action(() => np.einsum("i,i->i",
                    new[] {np.arange(3).astype(NPTypeCode.Int32), np.arange(3.0)}, casting: "no"))
                .Should().Throw<TypeError>().WithMessage(
                    "Iterator operand 0 dtype could not be cast from dtype('int32') to dtype('float64') " +
                    "according to the rule 'no'");

            // Same dtype everywhere needs no cast, so casting='no' computes.
            np.einsum("i,i->i", new[] {np.arange(3.0), np.arange(3.0)}, casting: "no")
                .Should().BeOfValues(0, 1, 4);
        }

        [TestMethod]
        public void OutCast_IsRejectedUnderTheRule_Verbatim()
        {
            // NumPy names out as operand <nop> in this message (1 input -> operand 1).
            new Action(() => np.einsum("ij->i",
                    new[] {np.arange(6.0).reshape(2, 3)}, @out: np.zeros(new Shape(2), NPTypeCode.Int32)))
                .Should().Throw<TypeError>().WithMessage(
                    "Iterator requested dtype could not be cast from dtype('float64') to dtype('int32'), " +
                    "the operand 1 dtype, according to the rule 'safe'");
        }

        #endregion

        #region order= resolves the computed result's layout

        [TestMethod]
        public void Order_AllFortranInputs_ComeBackFortran_EvenAtTheDefaultK()
        {
            // Probed: c_einsum's iterator picks F for all-F inputs whatever 'A'/'K' says.
            var fa = np.asfortranarray(np.arange(6.0).reshape(2, 3));
            var fb = np.asfortranarray(np.arange(6.0).reshape(3, 2));

            foreach (char order in new[] {'K', 'A', 'F'})
            {
                var r = np.einsum("ij,jk->ik", new[] {fa, fb}, order: order);
                r.Shape.IsFContiguous.Should().BeTrue($"order '{order}' with all-F inputs is F in NumPy");
                r.Should().BeOfValues(10, 13, 28, 40);
            }

            np.einsum("ij,jk->ik", new[] {fa, fb}, order: 'C').Shape.IsContiguous.Should().BeTrue();

            // Mixed C,F stays C at 'K' and 'A'.
            var ca = np.arange(6.0).reshape(2, 3);
            np.einsum("ij,jk->ik", new[] {ca, fb}, order: 'K').Shape.IsContiguous.Should().BeTrue();
            np.einsum("ij,jk->ik", new[] {ca, fb}, order: 'A').Shape.IsContiguous.Should().BeTrue();
        }

        #endregion

        #region empties and degenerate extents

        [TestMethod]
        public void EmptyContractionAxis_GivesZeros_AndEmptyOperandsGiveEmptyResults()
        {
            // (2,0)@(0,3): every entry is an empty sum -> exactly 0, matching NumPy.
            np.einsum("ij,jk->ik", np.ones(new Shape(2, 0)), np.ones(new Shape(0, 3)))
                .Should().BeOfValues(0, 0, 0, 0, 0, 0).And.BeShaped(2, 3);

            np.einsum("ij->i", np.ones(new Shape(2, 0))).Should().BeOfValues(0, 0).And.BeShaped(2);
            np.einsum("ij->i", np.ones(new Shape(0, 3))).Should().BeShaped(0);
            np.einsum("i->", np.ones(new Shape(0))).Should().BeOfValues(0).And.BeShaped();
            np.einsum("ii->i", np.zeros(new Shape(0, 0))).Should().BeShaped(0);
        }

        #endregion

        #region layouts (the full matrix is differential-verified; these pin one of each kind)

        [TestMethod]
        public void NonContiguousOperands_ContractCorrectly()
        {
            // Transposed lhs: arange(6).reshape(3,2).T @ arange(6).reshape(3,2).
            var at = np.transpose(np.arange(6.0).reshape(3, 2));
            np.einsum("ij,jk->ik", at, np.arange(6.0).reshape(3, 2))
                .Should().BeOfValues(20, 26, 26, 35).And.BeShaped(2, 2);

            // Negative-stride reduce: rows reversed.
            np.einsum("ij->i", np.arange(6.0).reshape(2, 3)["::-1"])
                .Should().BeOfValues(12, 3).And.BeShaped(2);

            // F-order diagonal.
            np.einsum("ii->i", np.asfortranarray(np.arange(9.0).reshape(3, 3)))
                .Should().BeOfValues(0, 4, 8).And.BeShaped(3);
        }

        [TestMethod]
        public void StridedOut_WritesThroughItsBase()
        {
            // out= may be a strided view; the result lands through its strides (probed layout).
            var @base = np.zeros(new Shape(4, 4));
            var target = @base["::2, ::2"];
            var r = np.einsum("ij,jk->ik",
                new[] {np.arange(4.0).reshape(2, 2), np.arange(4.0).reshape(2, 2)}, @out: target);
            ReferenceEquals(r, target).Should().BeTrue();
            @base.Should().BeOfValues(2, 0, 3, 0, 0, 0, 0, 0, 6, 0, 11, 0, 0, 0, 0, 0);
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
        public void Dtype_RejectsAnUnsafeCastUnderTheDefaultRule_Verbatim()
        {
            // int32 -> float32 is not a 'safe' cast, exactly as NumPy's einsum loop rejects it —
            // note this expression SUMS ('ij->i'), so the view path does not swallow the dtype.
            new Action(() => np.einsum("ij->i", new[] {A23.astype(NPTypeCode.Int32)}, dtype: NPTypeCode.Single))
                .Should().Throw<TypeError>().WithMessage(
                    "Iterator operand 0 dtype could not be cast from dtype('int32') to dtype('float32') " +
                    "according to the rule 'safe'");
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
