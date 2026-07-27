using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    ///     The CBLAS product family NumSharp gained alongside <c>dot</c>/<c>matmul</c>:
    ///     <c>inner</c>, <c>vdot</c>, <c>vecdot</c>, <c>matvec</c>, <c>vecmat</c> and
    ///     <c>tensordot</c>.
    /// </summary>
    /// <remarks>
    ///     Every expectation was probed against NumPy 2.4.2 and cross-checked by replaying the same
    ///     56-case matrix through both libraries; the values and shapes here are NumPy's output, not
    ///     hand-derived. These functions have a managed implementation and do NOT need a backend —
    ///     see <see cref="LinAlgEngineSeamTests"/> for the ones that do.
    /// </remarks>
    [TestClass]
    public class BlasProductApiTests
    {
        private static NDArray V3 => np.arange(3.0);
        private static NDArray M23 => np.arange(6.0).reshape(2, 3);

        #region inner

        [TestMethod]
        public void Inner_TwoVectors_IsTheScalarSumProduct()
        {
            // np.inner([0,1,2], [1,2,3]) -> 8.0
            np.inner(V3, V3 + 1.0).Should().BeOfValues(8).And.BeShaped();
        }

        [TestMethod]
        public void Inner_ContractsTheLastAxisOfBOTHOperands()
        {
            // np.inner((2,3), (2,3)) -> (2,2); dot would have refused this pair entirely.
            np.inner(M23, M23).Should().BeOfValues(5, 14, 14, 50).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Inner_ResultShapeIsBothOperandsMinusTheirLastAxis()
        {
            np.inner(np.arange(24.0).reshape(2, 3, 4), np.ones(new Shape(5, 4)))
                .Should().BeShaped(2, 3, 5);
        }

        [TestMethod]
        public void Inner_ZeroDimensionalOperand_DegeneratesToMultiply()
        {
            np.inner(NDArray.Scalar(2.0), V3).Should().BeOfValues(0, 2, 4).And.BeShaped(3);
        }

        [TestMethod]
        public void Inner_DoesNotConjugate_UnlikeVdot()
        {
            // np.inner([1+2j, 3+4j], [1+1j, 1+1j]) -> (-2+10j); np.vdot of the same pair is (10-2j).
            var product = np.inner(
                np.array(new Complex[] {new(1, 2), new(3, 4)}),
                np.array(new Complex[] {new(1, 1), new(1, 1)}));

            ((Complex)product.GetValue(0)).Should().Be(new Complex(-2, 10));
        }

        [TestMethod]
        public void Inner_UnalignedOperands_ReportBsShapeALREADYTRANSPOSED()
        {
            // NumPy implements inner as matrixproduct(a, swapaxes(b, -1, -2)), so the message
            // describes the SWAPPED b: np.inner(ones((2,3)), ones((3,2))) reports "(2,3) and (2,3)".
            new Action(() => np.inner(M23, np.arange(6.0).reshape(3, 2)))
                .Should().Throw<Exception>()
                .WithMessage("shapes (2,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)");
        }

        #endregion

        #region vdot

        [TestMethod]
        public void Vdot_FlattensBothOperands_SoUnequalShapesOfEqualSizeAgree()
        {
            // np.vdot((2,3), (3,2)) -> 55.0 : neither a matrix product nor an error.
            np.vdot(M23, np.arange(6.0).reshape(3, 2)).Should().BeOfValues(55).And.BeShaped();
        }

        [TestMethod]
        public void Vdot_ConjugatesTheFirstOperand()
        {
            var product = np.vdot(
                np.array(new Complex[] {new(1, 2), new(3, 4)}),
                np.array(new Complex[] {new(1, 1), new(1, 1)}));

            ((Complex)product.GetValue(0)).Should().Be(new Complex(10, -2));
        }

        [TestMethod]
        public void Vdot_LengthMismatch_IsAReshapeError_NotALengthOne()
        {
            // NumPy's array_vdot flattens both operands through ONE reused dims buffer holding -1.
            // _fix_unknown_dimension writes the resolved length back into it, so the second flatten
            // asks for (a.size,) rather than (-1,) — and the "vectors have different lengths" check
            // that follows in the C source can never be reached.
            new Action(() => np.vdot(V3, np.arange(2.0)))
                .Should().Throw<Exception>()
                .WithMessage("cannot reshape array of size 2 into shape (3,)");
        }

        [TestMethod]
        public void Vdot_TheReshapeQuirkAlsoGovernsZeroDimensionalAndEmptyOperands()
        {
            new Action(() => np.vdot(NDArray.Scalar(1.0), np.arange(2.0)))
                .Should().Throw<Exception>().WithMessage("cannot reshape array of size 2 into shape (1,)");

            new Action(() => np.vdot(np.zeros(new Shape(0)), np.ones(new Shape(3))))
                .Should().Throw<Exception>().WithMessage("cannot reshape array of size 3 into shape (0,)");

            // Two empties agree on length, so this one succeeds — at the additive identity.
            np.vdot(np.zeros(new Shape(0)), np.zeros(new Shape(0))).Should().BeOfValues(0);
        }

        #endregion

        #region vecdot / matvec / vecmat

        [TestMethod]
        public void Vecdot_ReducesInTheLOOPDtype_NotNEP50sWiderAccumulator()
        {
            // np.vecdot's registered loops are 'ii->i', so an int32 pair stays int32 — where
            // np.sum(int32) would promote to int64.
            var product = np.vecdot(np.arange(3).astype(np.int32), np.arange(3).astype(np.int32));
            product.dtype.Should().Be(typeof(int));
            product.Should().BeOfValues(5);
        }

        [TestMethod]
        public void Vecdot_ConjugatesTheFirstOperand_LikeVdot()
        {
            var product = np.vecdot(
                np.array(new Complex[] {new(1, 2), new(3, 4)}),
                np.array(new Complex[] {new(1, 1), new(1, 1)}));

            ((Complex)product.GetValue(0)).Should().Be(new Complex(10, -2));
        }

        [TestMethod]
        public void Vecdot_AxisNamesTheSharedCoreAxisInEachOperand()
        {
            // np.vecdot((2,3), (2,), axis=0) -> (3,)
            np.vecdot(M23, np.arange(2.0), axis: 0).Should().BeOfValues(3, 4, 5).And.BeShaped(3);
        }

        [TestMethod]
        public void Vecdot_KeepdimsRestoresTheContractedAxisWhereItWasTaken()
        {
            np.vecdot(M23, V3, keepdims: true).Should().BeShaped(2, 1);
            np.vecdot(M23, np.arange(2.0), axis: 0, keepdims: true).Should().BeShaped(1, 3);
        }

        [TestMethod]
        public void Vecdot_DtypeSelectsTheLoop()
        {
            np.vecdot(M23, V3, dtype: NPTypeCode.Single).dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void Matvec_IsTheMatrixVectorProduct_AndBroadcastsLeadingAxes()
        {
            np.matvec(M23, V3).Should().BeOfValues(5, 14).And.BeShaped(2);
            np.matvec(np.arange(24.0).reshape(2, 3, 4), np.ones(new Shape(4)))
                .Should().BeOfValues(6, 22, 38, 54, 70, 86).And.BeShaped(2, 3);
        }

        [TestMethod]
        public void Vecmat_ConjugatesTheVector_MakingItMatvecsAdjoint()
        {
            np.vecmat(np.arange(2.0), M23).Should().BeOfValues(3, 4, 5).And.BeShaped(3);

            var adjoint = np.vecmat(
                np.array(new Complex[] {new(1, 2), new(3, 4)}),
                np.eye(2).astype(np.complex128));

            ((Complex)adjoint.GetValue(0)).Should().Be(new Complex(1, -2));
            ((Complex)adjoint.GetValue(1)).Should().Be(new Complex(3, -4));
        }

        [TestMethod]
        public void Gufuncs_RejectTooFewDimensions_WithNumPysVerbatimText()
        {
            new Action(() => np.vecdot(NDArray.Scalar(1.0), V3))
                .Should().Throw<ValueError>().WithMessage(
                    "vecdot: Input operand 0 does not have enough dimensions " +
                    "(has 0, gufunc core with signature (n),(n)->() requires 1)");

            new Action(() => np.matvec(np.ones(new Shape(4)), np.ones(new Shape(4))))
                .Should().Throw<ValueError>().WithMessage(
                    "matvec: Input operand 0 does not have enough dimensions " +
                    "(has 1, gufunc core with signature (m,n),(n)->(m) requires 2)");

            new Action(() => np.vecmat(V3, V3))
                .Should().Throw<ValueError>().WithMessage(
                    "vecmat: Input operand 1 does not have enough dimensions " +
                    "(has 1, gufunc core with signature (n),(n,m)->(m) requires 2)");
        }

        [TestMethod]
        public void Gufuncs_RejectMismatchedCoreDimensions_WithNumPysVerbatimText()
        {
            // Core dimensions match EXACTLY — they never broadcast. NumPy reports the offending
            // operand's size first and the size an earlier operand already fixed second.
            new Action(() => np.vecdot(np.arange(2.0), V3))
                .Should().Throw<ValueError>().WithMessage(
                    "vecdot: Input operand 1 has a mismatch in its core dimension 0, " +
                    "with gufunc signature (n),(n)->() (size 3 is different from 2)");

            new Action(() => np.matvec(M23, np.arange(2.0)))
                .Should().Throw<ValueError>().WithMessage(
                    "matvec: Input operand 1 has a mismatch in its core dimension 0, " +
                    "with gufunc signature (m,n),(n)->(m) (size 2 is different from 3)");

            new Action(() => np.vecmat(np.arange(2.0), np.ones(new Shape(3, 4))))
                .Should().Throw<ValueError>().WithMessage(
                    "vecmat: Input operand 1 has a mismatch in its core dimension 0, " +
                    "with gufunc signature (n),(n,m)->(m) (size 3 is different from 2)");
        }

        #endregion

        #region tensordot

        [TestMethod]
        public void Tensordot_DefaultContractsTwoAxes()
        {
            np.tensordot(np.arange(24.0).reshape(2, 3, 4), np.arange(24.0).reshape(3, 4, 2))
                .Should().BeOfValues(1012, 1078, 2596, 2806).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Tensordot_OneAxisIsTheOrdinaryMatrixProduct()
        {
            np.tensordot(np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(3, 2), 1)
                .Should().BeOfValues(10, 13, 28, 40).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Tensordot_ZeroAndNegativeCountsBothGiveTheOuterProduct()
        {
            // NumPy forms range(-axes, 0), which is empty for any axes <= 0 — so a negative count
            // contracts nothing rather than counting from the other end.
            var a = np.arange(24.0).reshape(2, 3, 4);
            var b = np.arange(24.0).reshape(4, 3, 2);

            np.tensordot(a, b, 0).Should().BeShaped(2, 3, 4, 4, 3, 2);
            np.tensordot(a, b, -1).Should().BeShaped(2, 3, 4, 4, 3, 2);
        }

        [TestMethod]
        public void Tensordot_ExplicitAxisListsPairElementwise()
        {
            np.tensordot(np.arange(24.0).reshape(2, 3, 4), np.arange(24.0).reshape(4, 3, 2),
                    new[] {1, 2}, new[] {1, 0})
                .Should().BeOfValues(880, 946, 2464, 2674).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Tensordot_SingleAxisPairTuple()
        {
            np.tensordot(np.arange(24.0).reshape(2, 3, 4), np.arange(24.0).reshape(4, 3, 2), (0, 2))
                .Should().BeShaped(3, 4, 4, 3);
        }

        [TestMethod]
        public void Tensordot_NegativeAxisIndicesNormalize()
        {
            np.tensordot(np.arange(24.0).reshape(2, 3, 4), np.arange(24.0).reshape(4, 3, 2),
                    new[] {-1}, new[] {0})
                .Should().BeShaped(2, 3, 3, 2);
        }

        [TestMethod]
        public void Tensordot_EveryDisagreementIsTheSameOneLineMessage()
        {
            var a = np.arange(24.0).reshape(2, 3, 4);
            var b = np.arange(24.0).reshape(4, 3, 2);

            // Mismatched extents, too many axes and unequal list lengths all land here.
            new Action(() => np.tensordot(a, b)).Should().Throw<ValueError>().WithMessage("shape-mismatch for sum");
            new Action(() => np.tensordot(a, b, 3)).Should().Throw<ValueError>().WithMessage("shape-mismatch for sum");
            new Action(() => np.tensordot(a, b, new[] {0}, new[] {0}))
                .Should().Throw<ValueError>().WithMessage("shape-mismatch for sum");
            new Action(() => np.tensordot(a, b, new[] {0, 1}, new[] {0}))
                .Should().Throw<ValueError>().WithMessage("shape-mismatch for sum");
        }

        #endregion
    }
}
