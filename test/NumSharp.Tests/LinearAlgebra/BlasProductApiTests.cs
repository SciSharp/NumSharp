using System;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.LinearAlgebra
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
        public void Vecdot_KeepdimsComposesWithAxes_JustAsNumPy()
        {
            var a = np.arange(6.0).reshape(2, 3);

            // Under keepdims the output gains a core dim, so a PROVIDED output entry must be length 1
            // — but it may still be OMITTED, in which case the kept axis lands at the END.
            np.vecdot(a, a, axes: new[] {new[] {1}, new[] {1}}, keepdims: true)
                .Should().BeOfValues(5, 50).And.BeShaped(2, 1);
            np.vecdot(a, a, axes: new[] {new[] {1}, new[] {1}, new[] {1}}, keepdims: true)
                .Should().BeOfValues(5, 50).And.BeShaped(2, 1);
            np.vecdot(a, a, axes: new[] {new[] {1}, new[] {1}, new[] {0}}, keepdims: true)
                .Should().BeOfValues(5, 50).And.BeShaped(1, 2);
            np.vecdot(a, a, axes: new[] {new[] {0}, new[] {0}}, keepdims: true)
                .Should().BeOfValues(9, 17, 29).And.BeShaped(3, 1);

            // The output axis is placed in the OUTPUT and can be negative; the values are unchanged.
            var a3 = np.arange(24.0).reshape(2, 4, 3);
            np.vecdot(a3, a3, axes: new[] {new[] {2}, new[] {2}, new[] {0}}, keepdims: true).Should().BeShaped(1, 2, 4);
            np.vecdot(a3, a3, axes: new[] {new[] {2}, new[] {2}, new[] {1}}, keepdims: true).Should().BeShaped(2, 1, 4);
            np.vecdot(a3, a3, axes: new[] {new[] {1}, new[] {1}, new[] {2}}, keepdims: true)
                .Should().BeOfValues(126, 166, 214, 1134, 1270, 1414).And.BeShaped(2, 3, 1);
            np.vecdot(a3, a3, axes: new[] {new[] {2}, new[] {2}, new[] {-1}}, keepdims: true).Should().BeShaped(2, 4, 1);

            // An explicit output entry of the WRONG length for the effective (keepdims-adjusted) rank
            // is an AxisError, and an out-of-range output axis is one too.
            new Action(() => np.vecdot(a, a, axes: new[] {new[] {1}, new[] {1}, new int[0]}, keepdims: true))
                .Should().Throw<AxisError>().WithMessage(
                    "*vecdot: operand 2 has 1 core dimensions, but 0 dimensions are specified by axes tuple.*");
            new Action(() => np.vecdot(a, a, axes: new[] {new[] {1}, new[] {1}, new[] {1}}))
                .Should().Throw<AxisError>().WithMessage(
                    "*vecdot: operand 2 has 0 core dimensions, but 1 dimensions are specified by axes tuple.*");
            new Action(() => np.vecdot(a, a, axes: new[] {new[] {1}, new[] {1}, new[] {2}}, keepdims: true))
                .Should().Throw<AxisError>().WithMessage("axis 2 is out of bounds for array of dimension 2*");
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
        public void Axes_NamesTheCoreAxesPerOperand_AndIsTheGeneralFormOfAxis()
        {
            var m = M23;
            int[] none = new int[0];

            // vecdot's output has no core axes, so its entry may be omitted.
            np.vecdot(m, V3, axes: new[] {new[] {-1}, new[] {-1}, none})
                .Should().BeOfValues(5, 14).And.BeShaped(2);
            np.vecdot(m, V3, axes: new[] {new[] {-1}, new[] {-1}})
                .Should().BeOfValues(5, 14).And.BeShaped(2);
            np.vecdot(m, np.arange(2.0), axes: new[] {new[] {0}, new[] {0}, none})
                .Should().BeOfValues(3, 4, 5).And.BeShaped(3);

            // matvec and vecmat DO have an output core axis, and axes says where it goes.
            np.matvec(m, V3, axes: new[] {new[] {-2, -1}, new[] {-1}, new[] {-1}})
                .Should().BeOfValues(5, 14).And.BeShaped(2);
            np.matvec(m, V3, axes: new[] {new[] {0, 1}, new[] {0}, new[] {0}})
                .Should().BeOfValues(5, 14).And.BeShaped(2);

            // Moving the output core axis to the front transposes the result.
            np.matvec(np.arange(24.0).reshape(2, 3, 4), np.ones(new Shape(4)),
                    axes: new[] {new[] {-2, -1}, new[] {-1}, new[] {0}})
                .Should().BeOfValues(6, 54, 22, 70, 38, 86).And.BeShaped(3, 2);

            np.vecmat(np.arange(2.0), m, axes: new[] {new[] {-1}, new[] {-2, -1}, new[] {-1}})
                .Should().BeOfValues(3, 4, 5).And.BeShaped(3);
        }

        [TestMethod]
        public void Axes_AndAxis_AreMutuallyExclusive()
        {
            new Action(() => np.vecdot(M23, V3, axes: new[] {new[] {-1}, new[] {-1}, new int[0]}, axis: -1))
                .Should().Throw<TypeError>().WithMessage("cannot specify both 'axis' and 'axes'");
        }

        [TestMethod]
        public void Axes_WrongNumberOfEntries_ReportsNumPysVerbatimText()
        {
            // Only vecdot may omit the output entry, so a 2-entry list is an error for matvec.
            new Action(() => np.matvec(M23, V3, axes: new[] {new[] {-2, -1}, new[] {-1}}))
                .Should().Throw<ValueError>().WithMessage(
                    "axes should be a list with an entry for all 3 inputs and outputs; " +
                    "entries for outputs can only be omitted if none of them has core axes.");

            new Action(() => np.matvec(M23, V3,
                    axes: new[] {new[] {-2, -1}, new[] {-1}, new[] {-1}, new[] {0}}))
                .Should().Throw<ValueError>().WithMessage(
                    "axes should be a list with an entry for all 3 inputs and outputs; " +
                    "entries for outputs can only be omitted if none of them has core axes.");
        }

        [TestMethod]
        public void Axes_EntryOfTheWrongArity_ReportsTheCoreDimensionCounts()
        {
            new Action(() => np.vecdot(M23, V3, axes: new[] {new[] {-1, -2}, new[] {-1}, new int[0]}))
                .Should().Throw<AxisError>().WithMessage(
                    "*vecdot: operand 0 has 1 core dimensions, but 2 dimensions are specified by axes tuple.*");

            new Action(() => np.vecdot(M23, V3, axes: new[] {new[] {-1}, new[] {-1}, new[] {0}}))
                .Should().Throw<AxisError>().WithMessage(
                    "*vecdot: operand 2 has 0 core dimensions, but 1 dimensions are specified by axes tuple.*");
        }

        [TestMethod]
        public void MatvecAndVecmat_OfferAxisAndKeepdims_OnlyToRejectThem()
        {
            // NumPy's signature carries both, and both always raise because these signatures have
            // two DISTINCT core dimensions. Offering them keeps a ported call failing the same way
            // with the same message rather than failing to compile.
            new Action(() => np.matvec(M23, V3, axis: 0)).Should().Throw<TypeError>().WithMessage(
                "matvec: axis can only be used with a single shared core dimension, " +
                "not with the 2 distinct ones implied by signature (m,n),(n)->(m).");

            new Action(() => np.matvec(M23, V3, keepdims: true)).Should().Throw<TypeError>().WithMessage(
                "matvec does not support keepdims: its signature (m,n),(n)->(m) requires input 1 to " +
                "have 1 core dimensions, but keepdims can only be used when all inputs have the same " +
                "number of core dimensions and all outputs have no core dimensions.");

            new Action(() => np.vecmat(np.arange(2.0), M23, axis: 0)).Should().Throw<TypeError>().WithMessage(
                "vecmat: axis can only be used with a single shared core dimension, " +
                "not with the 2 distinct ones implied by signature (n),(n,m)->(m).");

            // vecmat names input 1's core rank as 2 where matvec names 1 — the counts are per-signature.
            new Action(() => np.vecmat(np.arange(2.0), M23, keepdims: true)).Should().Throw<TypeError>().WithMessage(
                "vecmat does not support keepdims: its signature (n),(n,m)->(m) requires input 1 to " +
                "have 2 core dimensions, but keepdims can only be used when all inputs have the same " +
                "number of core dimensions and all outputs have no core dimensions.");
        }

        [TestMethod]
        public void ProductFamily_AgreesWithNumPyAcrossEveryMemoryLayout()
        {
            // The DOD's layout axis: the SAME (2,3) values [[0,1,2],[3,4,5]] reached six different
            // ways must give one answer. Each variant is built so its contents are identical and
            // only its strides differ, so a stride bug in any one path shows up as a mismatch.
            var variants = new[]
            {
                np.arange(6.0).reshape(2, 3),                                             // C
                np.asfortranarray(np.arange(6.0).reshape(2, 3)),                          // F
                np.array(new[] {0.0, 3, 1, 4, 2, 5}).reshape(3, 2).T,                     // transposed
                np.array(new[] {0.0, 9, 1, 9, 2, 9, 3, 9, 4, 9, 5, 9}).reshape(2, 6)[":, ::2"], // stepped
                np.array(new[] {2.0, 1, 0, 5, 4, 3}).reshape(2, 3)[":, ::-1"],            // negative stride
                np.arange(-3.0, 6.0).reshape(3, 3)["1:"]                                  // offset view
            };

            // Guard the guard: if a variant does not actually hold [[0,1,2],[3,4,5]] then the
            // products below would agree for the wrong reason.
            foreach (var m in variants)
                m.Should().BeOfValues(0, 1, 2, 3, 4, 5).And.BeShaped(2, 3);

            foreach (var m in variants)
            {
                np.inner(m, V3).Should().BeOfValues(5, 14);
                np.vdot(m, m).Should().BeOfValues(55);
                np.vecdot(m, V3).Should().BeOfValues(5, 14);
                np.matvec(m, V3).Should().BeOfValues(5, 14);
                np.vecmat(np.arange(2.0), m).Should().BeOfValues(3, 4, 5);
            }

            // A broadcast (stride-0) operand is read-only and must still contract correctly.
            var broadcast = np.broadcast_to(V3, new Shape(2, 3));
            np.vecdot(broadcast, V3).Should().BeOfValues(5, 5);
            np.matvec(broadcast, V3).Should().BeOfValues(5, 5);
        }

        [TestMethod]
        public void ProductFamily_PreservesTheLoopDtypeAcrossTheDtypeAxis()
        {
            // np.vecdot's loops are 'ii->i' at every integer width, so nothing promotes.
            foreach (var code in new[]
                     {
                         NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int32,
                         NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Single, NPTypeCode.Double
                     })
            {
                var m = np.arange(6).reshape(2, 3).astype(code);
                var w = np.arange(3).astype(code);

                np.vecdot(m, w).typecode.Should().Be(code, $"vecdot must keep {code}");
                np.matvec(m, w).typecode.Should().Be(code, $"matvec must keep {code}");
                np.vdot(np.arange(6).astype(code), np.arange(6).astype(code)).typecode.Should().Be(code);
            }

            // bool is its own loop ('??->?') and stays bool rather than counting.
            var b = np.arange(6).reshape(2, 3).astype(NPTypeCode.Boolean);
            np.vecdot(b, np.arange(3).astype(NPTypeCode.Boolean)).typecode.Should().Be(NPTypeCode.Boolean);
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
        public void Tensordot_RejectionsMatchNumPy_ShapeMismatchDuplicateAndIndexError()
        {
            var a = np.arange(24.0).reshape(2, 3, 4);
            var b = np.arange(24.0).reshape(4, 3, 2);

            // Mismatched extents and unequal list lengths are "shape-mismatch for sum". `axes=3` here
            // is IN range (both operands are 3-D), so it too is an extent mismatch, not an index error.
            new Action(() => np.tensordot(a, b)).Should().Throw<ValueError>().WithMessage("shape-mismatch for sum");
            new Action(() => np.tensordot(a, b, 3)).Should().Throw<ValueError>().WithMessage("shape-mismatch for sum");
            new Action(() => np.tensordot(a, b, new[] {0}, new[] {0}))
                .Should().Throw<ValueError>().WithMessage("shape-mismatch for sum");
            new Action(() => np.tensordot(a, b, new[] {0, 1}, new[] {0}))
                .Should().Throw<ValueError>().WithMessage("shape-mismatch for sum");

            // A repeated RAW axis value is rejected first, ahead of any shape or range check.
            new Action(() => np.tensordot(a, b, new[] {0, 0}, new[] {0, 1}))
                .Should().Throw<ValueError>().WithMessage("duplicate axes are not allowed in tensordot");
            new Action(() => np.tensordot(a, b, new[] {1, 2}, new[] {0, 0}))
                .Should().Throw<ValueError>().WithMessage("duplicate axes are not allowed in tensordot");

            var m = np.ones(new Shape(3, 4));
            var n = np.ones(new Shape(4, 3));

            // Duplicate wins even when the repeated axis is also out of range.
            new Action(() => np.tensordot(m, n, new[] {5, 5}, new[] {0, 1}))
                .Should().Throw<ValueError>().WithMessage("duplicate axes are not allowed in tensordot");

            // An out-of-range contraction axis is an IndexError — NumPy indexes the shape TUPLE with
            // the raw axis — whether the int count exceeds ndim or a listed axis is out of bounds.
            new Action(() => np.tensordot(m, n, 3)).Should().Throw<IndexError>().WithMessage("tuple index out of range");
            new Action(() => np.tensordot(m, n, new[] {5, 0}, new[] {0, 1}))
                .Should().Throw<IndexError>().WithMessage("tuple index out of range");
            new Action(() => np.tensordot(m, n, new[] {-3}, new[] {0}))
                .Should().Throw<IndexError>().WithMessage("tuple index out of range");

            // …but a mismatched extent found earlier in the list hides a later out-of-range axis.
            new Action(() => np.tensordot(m, n, new[] {0, 1}, new[] {0, 5}))
                .Should().Throw<ValueError>().WithMessage("shape-mismatch for sum");
        }

        #endregion
    }
}
