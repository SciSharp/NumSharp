using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    ///     <c>np.einsum</c>'s subscript language, against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The contraction now computes, so a VALID expression returns an array — and
    ///     <see cref="Contracts"/> asserts its SHAPE against NumPy's. Every shape below was produced
    ///     by running the same expression through NumPy 2.4.2; the value parity is gated separately by
    ///     <see cref="EinsumContractionTests"/>.
    ///     </para>
    ///     <para>
    ///     NumPy carries TWO einsum parsers — the C one behind the default <c>optimize=False</c> and
    ///     a Python one behind the <c>optimize</c> path — and they word the same rejection
    ///     differently. These are the C one's texts, since that is what a default call hits, and they
    ///     are unchanged by the contraction landing (the engine parses before it contracts).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class EinsumSubscriptParityTests
    {
        private static NDArray A23 => np.arange(6.0).reshape(2, 3);
        private static NDArray A32 => np.arange(6.0).reshape(3, 2);
        private static NDArray Sq3 => np.arange(9.0).reshape(3, 3);
        private static NDArray V3 => np.arange(3.0);

        /// <summary>
        ///     Asserts the expression parses and CONTRACTS to <paramref name="shape"/> — the shape
        ///     NumPy 2.4.2 actually returns for it.
        /// </summary>
        private static void Contracts(string shape, Func<NDArray> call)
        {
            var result = call();
            string.Join(",", result.shape).Should().Be(shape);
        }

        #region shapes the parser resolves

        [TestMethod]
        public void ExplicitOutput_ResolvesTheShapeNumPyProduces()
        {
            Contracts("2,2", () => np.einsum("ij,jk->ik", A23, A32));
            Contracts("3,2", () => np.einsum("ij->ji", A23));
            Contracts("", () => np.einsum("ij->", A23));
            Contracts("2", () => np.einsum("ij->i", A23));
            Contracts("3,3", () => np.einsum("i,j->ij", V3, V3));
            // (2,3)@(3,2)@(2,3) contracts to (2,3), not (2,2) — l comes from the third operand.
            Contracts("2,3", () => np.einsum("ij,jk,kl->il", A23, A32, A23));
        }

        [TestMethod]
        public void ImplicitOutput_IsEveryOnceUsedLabelInASCIIOrder()
        {
            // No "->": the labels that appear exactly once, ASCII-ordered — so an UPPER-case label
            // sorts before a lower-case one, which is why "zA" transposes and "ab" does not.
            Contracts("2,2", () => np.einsum("ij,jk", A23, A32));
            Contracts("3,2", () => np.einsum("zA", A23));
            Contracts("2,3", () => np.einsum("Az", A23));
            Contracts("3,3", () => np.einsum("i,j", V3, V3));
            Contracts("", () => np.einsum("ii", Sq3));
        }

        [TestMethod]
        public void RepeatedLabels_AreDiagonals()
        {
            Contracts("3", () => np.einsum("ii->i", Sq3));
            Contracts("3", () => np.einsum("iii->i", np.arange(27.0).reshape(3, 3, 3)));
            Contracts("3,3", () => np.einsum("iji->ij", np.arange(27.0).reshape(3, 3, 3)));
            Contracts("3", () => np.einsum("ii,ii->i", Sq3, Sq3));
        }

        [TestMethod]
        public void Ellipsis_BroadcastsAndAlignsRight()
        {
            Contracts("2", () => np.einsum("...i->...", A23));
            Contracts("2", () => np.einsum("...i,...i->...", A23, A23));
            Contracts("2,4", () => np.einsum("k...,jk", A32, np.arange(12.0).reshape(4, 3)));
            Contracts("2,4", () => np.einsum("ki,...k->i...", A32, np.arange(12.0).reshape(4, 3)));

            // An ellipsis matching zero dimensions simply vanishes.
            Contracts("3", () => np.einsum("...i->i", V3));
            Contracts("", () => np.einsum("...,...->...", NDArray.Scalar(2.0), NDArray.Scalar(3.0)));
        }

        [TestMethod]
        public void SizeOneOperandsBroadcastAgainstASharedLabel()
        {
            Contracts("2,3", () => np.einsum("ij,ij->ij", A23, np.ones(new Shape(1, 3))));
            Contracts("2,4", () => np.einsum("ij,jk->ik", A23, np.ones(new Shape(1, 4))));
        }

        [TestMethod]
        public void SpacesAreIgnored()
        {
            Contracts("3,2", () => np.einsum(" i j -> j i ", A23));
        }

        #endregion

        #region the sublist spelling

        [TestMethod]
        public void SublistAlphabetIsUPPERCaseFirst_WhichIsWhatKeepsIndexOrderAndASCIIOrderTheSame()
        {
            // NumPy 2.4.2's einsum_symbols is 'ABC…Zabc…z': index 0-25 are A-Z and 26-51 are a-z.
            // Guessing the intuitive order (a-z then A-Z) is SILENTLY wrong rather than an error —
            // it only shows when a single expression mixes the two halves, because the inferred
            // output is ASCII-sorted. This is that expression.
            Contracts("2,3", () => np.einsum(A23, new[] {0, 26}));
            Contracts("3,2", () => np.einsum(A23, new[] {26, 0}));

            Contracts("2,3", () => np.einsum(A23, new[] {0, 1}));
            Contracts("3,2", () => np.einsum(A23, new[] {1, 0}));
        }

        [TestMethod]
        public void SublistsAcceptOperandsAndAnOptionalOutputList()
        {
            Contracts("2,2", () => np.einsum(A23, new[] {0, 1}, A32, new[] {1, 2}));
            Contracts("2,2", () => np.einsum(A23, new[] {0, 1}, A32, new[] {1, 2}, new[] {0, 2}));
            Contracts("2", () => np.einsum(A23, new object[] {Slice.Ellipsis, 0}, new object[] {Slice.Ellipsis}));
        }

        [TestMethod]
        public void SublistsRejectNonIntegersAndOutOfRangeLabels()
        {
            new Action(() => np.einsum(A23, new object[] {0, "x"}))
                .Should().Throw<TypeError>().WithMessage("each subscript must be either an integer or an ellipsis");

            // A trailing operand with no subscript list is the same rejection.
            new Action(() => np.einsum(A23, new[] {0, 1}, A32))
                .Should().Throw<TypeError>().WithMessage("each subscript must be either an integer or an ellipsis");

            new Action(() => np.einsum(A23, new[] {0, 52}))
                .Should().Throw<ValueError>().WithMessage("subscript is not within the valid range [0, 52)");

            new Action(() => np.einsum(A23, new[] {0, -1}))
                .Should().Throw<ValueError>().WithMessage("subscript is not within the valid range [0, 52)");
        }

        #endregion

        #region grammar rejections, verbatim

        [TestMethod]
        public void InvalidCharactersAreRejectedWhereverTheyAppear()
        {
            new Action(() => np.einsum("i$,j->ij", V3, V3)).Should().Throw<ValueError>()
                .WithMessage("invalid subscript '$' in einstein sum subscripts string, subscripts must be letters");

            new Action(() => np.einsum("ij->i$", A23)).Should().Throw<ValueError>()
                .WithMessage("invalid subscript '$' in einstein sum subscripts string, subscripts must be letters");

            // '>' without its '-' never opens an arrow, so it is scanned as a label.
            new Action(() => np.einsum("i>j", V3)).Should().Throw<ValueError>()
                .WithMessage("invalid subscript '>' in einstein sum subscripts string, subscripts must be letters");
        }

        [TestMethod]
        public void AMalformedArrowIsReportedAsAMissingOne()
        {
            new Action(() => np.einsum("i-j", V3)).Should().Throw<ValueError>()
                .WithMessage("einstein sum subscript string does not contain proper '->' output specified");
        }

        [TestMethod]
        public void TheOperandCountMessagesReadBACKWARDS_AndAreReproducedAnyway()
        {
            // NumPy walks operand by operand and checks the delimiter that follows each one. The two
            // texts describe the OPERAND list where they mean the subscripts string, so both read
            // inverted: one operand against two terms says "more operands provided". Verbatim
            // regardless — a caller grepping NumPy's message must find it.
            new Action(() => np.einsum("ij,jk->ik", A23)).Should().Throw<ValueError>()
                .WithMessage("more operands provided to einstein sum function than specified in the subscripts string");

            new Action(() => np.einsum("ij->ij", A23, A32)).Should().Throw<ValueError>()
                .WithMessage("fewer operands provided to einstein sum function than specified in the subscripts string");
        }

        [TestMethod]
        public void OutputLabelsMustBeUniqueAndPresentInTheInput()
        {
            new Action(() => np.einsum("ij->ii", A23)).Should().Throw<ValueError>()
                .WithMessage("einstein sum subscripts string includes output subscript 'i' multiple times");

            new Action(() => np.einsum("i->ii", V3)).Should().Throw<ValueError>()
                .WithMessage("einstein sum subscripts string includes output subscript 'i' multiple times");

            new Action(() => np.einsum("ij->ik", A23)).Should().Throw<ValueError>()
                .WithMessage("einstein sum subscripts string included output subscript 'k' which never appeared in an input");
        }

        [TestMethod]
        public void EllipsisMustBeExactlyThreeDots_AndAtMostOnePerTerm()
        {
            foreach (string bad in new[] {"..i->", "....i->", "...i...->"})
            {
                new Action(() => np.einsum(bad, A23)).Should().Throw<ValueError>().WithMessage(
                    "einstein sum subscripts string contains a '.' that is not part of an ellipsis ('...') in operand 0");
            }

            new Action(() => np.einsum("...i->..i", A23)).Should().Throw<ValueError>().WithMessage(
                "einstein sum subscripts string contains a '.' that is not part of an ellipsis ('...') in the output");
        }

        [TestMethod]
        public void AnOutputMustCarryAnEllipsisWhenTheInputsBroadcast()
        {
            new Action(() => np.einsum("...i->i", A23)).Should().Throw<ValueError>().WithMessage(
                "output has more dimensions than subscripts given in einstein sum, but no '...' " +
                "ellipsis provided to broadcast the extra dimensions.");
        }

        [TestMethod]
        public void SubscriptCountMustMatchOperandRank()
        {
            // More labels than dimensions is caught as it happens...
            new Action(() => np.einsum("ijk->", A23)).Should().Throw<ValueError>()
                .WithMessage("einstein sum subscripts string contains too many subscripts for operand 0");

            // ...including against a 0-d operand, where the FIRST label is already one too many.
            new Action(() => np.einsum("i->", NDArray.Scalar(3.0))).Should().Throw<ValueError>()
                .WithMessage("einstein sum subscripts string contains too many subscripts for operand 0");

            // Fewer is only detectable at the end, and gets its own wording.
            new Action(() => np.einsum("i->", A23)).Should().Throw<ValueError>().WithMessage(
                "operand has more dimensions than subscripts given in einstein sum, but no '...' " +
                "ellipsis provided to broadcast the extra dimensions.");

            // An ellipsis that would have to match a negative number of dimensions.
            new Action(() => np.einsum("...ij->", V3)).Should().Throw<ValueError>()
                .WithMessage("einstein sum subscripts string contains too many subscripts for operand 0");
        }

        #endregion

        #region dimension rejections

        [TestMethod]
        public void AnImpossibleDiagonal_HasTWOMessagesAndTheConditionsPickBetweenThem()
        {
            // NumPy tries to answer a single-operand, out=-less, nothing-summed einsum with a VIEW,
            // and that attempt words the failure its own way. Any other shape takes the general
            // path, which names the operand index instead.
            new Action(() => np.einsum("ii->i", A23)).Should().Throw<ValueError>()
                .WithMessage("dimensions in single operand for collapsing index 'i' don't match (2 != 3)");

            // 'ii' sums the diagonal away, so the view attempt is abandoned before it starts.
            new Action(() => np.einsum("ii", A23)).Should().Throw<ValueError>()
                .WithMessage("dimensions in operand 0 for collapsing index 'i' don't match (2 != 3)");

            // An out= also rules the view out.
            new Action(() => np.einsum("ii->i", new[] {A23}, @out: np.zeros(new Shape(2))))
                .Should().Throw<ValueError>()
                .WithMessage("dimensions in operand 0 for collapsing index 'i' don't match (2 != 3)");

            // ...and so does a second operand.
            new Action(() => np.einsum("ii,jj->ij", A23, A32)).Should().Throw<ValueError>()
                .WithMessage("dimensions in operand 0 for collapsing index 'i' don't match (2 != 3)");
        }

        [TestMethod]
        public void EveryDiagonalIsCheckedBeforeAnyCrossOperandExtent()
        {
            // Operand 0 fixes j=3; operand 1's "jj" is an impossible diagonal on a (2,3) array. NumPy
            // collapses ALL operands' diagonals before the iterator compares extents, so this is a
            // diagonal error and not a size-of-label one. Doing it in one pass reverses the two.
            new Action(() => np.einsum("ij,jj->ij", np.ones(new Shape(2, 3)), A23))
                .Should().Throw<ValueError>()
                .WithMessage("dimensions in operand 1 for collapsing index 'j' don't match (2 != 3)");
        }

        [TestMethod]
        [Misaligned]
        public void ALabelWhoseExtentsDisagree_UsesNumPysOTHERWordingForTheSameError()
        {
            // NumPy's default path leaks its ITERATOR's text here — "operands could not be broadcast
            // together with remapped shapes [original->remapped]: (2,3)->(2,newaxis,3) (4,2)->(2,4)"
            // — which describes axis bookkeeping rather than the contraction. NumSharp raises the
            // wording NumPy's own einsumfunc.py parser uses for the identical condition. Note the
            // two sizes read swapped against the sentence: the first is the size already recorded.
            new Action(() => np.einsum("ij,jk->ik", A23, np.arange(8.0).reshape(4, 2)))
                .Should().Throw<ValueError>()
                .WithMessage("Size of label 'j' for operand 1 (3) does not match previous terms (4).");
        }

        #endregion

        #region keywords

        [TestMethod]
        public void OutIsValidatedForRank_BeforeAnythingIsComputed()
        {
            new Action(() => np.einsum("ij->ij", new[] {A23}, @out: np.zeros(new Shape(6))))
                .Should().Throw<ValueError>()
                .WithMessage("out parameter does not have the correct number of dimensions, has 1 but should have 2");

            new Action(() => np.einsum("ij->i", new[] {A23}, @out: np.zeros(new Shape(2, 1))))
                .Should().Throw<ValueError>()
                .WithMessage("out parameter does not have the correct number of dimensions, has 2 but should have 1");
        }

        [TestMethod]
        public void OrderAndCastingAreValidated()
        {
            new Action(() => np.einsum("ij->ji", new[] {A23}, order: 'Q'))
                .Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");

            new Action(() => np.einsum("ij->", new[] {A23}, casting: "bogus"))
                .Should().Throw<ValueError>()
                .WithMessage("casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got 'bogus')");
        }

        [TestMethod]
        public void NoOperandsIsRejectedBeforeTheSubscriptsAreEvenRead()
        {
            new Action(() => np.einsum("->", Array.Empty<NDArray>()))
                .Should().Throw<ValueError>()
                .WithMessage("must specify the einstein sum subscripts string and at least one operand");

            new Action(() => np.einsum(Array.Empty<object>()))
                .Should().Throw<ValueError>().WithMessage(
                    "must specify the einstein sum subscripts string and at least one operand, " +
                    "or at least one operand and its corresponding subscripts list");
        }

        [TestMethod]
        public void TooManyOperandsIsRejected()
        {
            var operands = new NDArray[70];
            for (int i = 0; i < operands.Length; i++)
                operands[i] = np.ones(new Shape(2));

            new Action(() => np.einsum(string.Join(",", new string('i', 70).ToCharArray()) + "->i", operands))
                .Should().Throw<ValueError>().WithMessage("too many operands");
        }

        #endregion

        [TestMethod]
        public void TheContractionComputes_ThroughTheMatrixProduct()
        {
            // The subscripts that used to resolve only a SHAPE now resolve a value. A plain matmul
            // reduces to np.matmul (hence OpenBLAS when a backend is referenced); the answer is
            // NumPy's. arange(6).reshape(2,3) @ arange(6).reshape(3,2) = [[10,13],[28,40]].
            np.einsum("ij,jk->ik", A23, A32).Should().BeOfValues(10, 13, 28, 40).And.BeShaped(2, 2);
        }
    }
}
