using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     Numeric NumPy-parity fixes the string validation found in <c>np.linalg.norm</c>, <c>np.linalg.matrix_power</c>,
    ///     <c>np.cross</c> and <c>np.linalg.tensorinv</c> (an empty text matrix converts to an empty float64 one, a text
    ///     operand meets the shape checks first). Every expected value was probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class LinAlgNumPyOrderTests
    {
        [TestMethod]
        public void Norm_OfAnEmptyOperand_MaxOrders_AreZero()
        {
            // NumPy 2.x reduces the inf / 1 orders with `.max(..., initial=0)`:
            // np.linalg.norm(np.zeros((0, 0)), np.inf) -> 0.0, ord=1 -> 0.0, np.zeros(0) inf -> 0.0, np.zeros((3, 0)) -> 0.0
            np.linalg.norm(np.zeros(new Shape(0L, 0L)), double.PositiveInfinity).GetDouble().Should().Be(0.0);
            np.linalg.norm(np.zeros(new Shape(0L, 0L)), 1).GetDouble().Should().Be(0.0);
            np.linalg.norm(np.zeros(new Shape(0L)), double.PositiveInfinity).GetDouble().Should().Be(0.0);
            np.linalg.norm(np.zeros(new Shape(3L, 0L)), double.PositiveInfinity).GetDouble().Should().Be(0.0);
            np.linalg.vector_norm(np.zeros(new Shape(0L)), ord: double.PositiveInfinity).GetDouble().Should().Be(0.0);
            var stacked = np.linalg.norm(np.zeros(new Shape(2L, 0L, 0L)), double.PositiveInfinity, new[] { 1, 2 });
            stacked.ToArray<double>().Should().Equal(0.0, 0.0);
        }

        [TestMethod]
        public void Norm_OfAnEmptyOperand_MinOrders_StillRaise()
        {
            // The min forms have no initial: np.linalg.norm(np.zeros((0, 0)), -1) -> ValueError.
            Action act = () => np.linalg.norm(np.zeros(new Shape(0L, 0L)), -1);
            act.Should().Throw<ArgumentException>().WithMessage("zero-size array to reduction operation minimum which has no identity");
        }

        [TestMethod]
        public void Norm_InitialZero_KeepsNonEmptyAnswers()
        {
            // initial=0 cannot change a non-empty answer: every |x| is >= 0, and a NaN still propagates.
            np.linalg.norm(np.array(new[,] { { 1.0, -2.0 }, { 3.0, 4.0 } }), double.PositiveInfinity).GetDouble().Should().Be(7.0);
            np.linalg.norm(np.array(new[,] { { 1.0, -2.0 }, { 3.0, 4.0 } }), 1).GetDouble().Should().Be(6.0);
            double.IsNaN(np.linalg.norm(np.array(new[] { 1.0, double.NaN }), double.PositiveInfinity).GetDouble()).Should().BeTrue();
            var f32 = np.linalg.norm(np.array(new[] { 1.5f, -2.5f }), double.PositiveInfinity);
            f32.dtype.Should().Be(np.float32);
            f32.GetSingle().Should().Be(2.5f);
        }

        [TestMethod]
        public void MatrixPower_One_ReturnsTheOperandItself()
        {
            // np.linalg.matrix_power(m, 1) is m -> True.
            var m = np.eye(2);
            ReferenceEquals(np.linalg.matrix_power(m, 1), m).Should().BeTrue();
        }

        [TestMethod]
        public void Cross_LeadingAxesBroadcast_BeforeAnythingElse()
        {
            // np.cross(np.ones((2, 3)), np.ones((3, 3))) -> np.broadcast's ValueError text.
            Action act = () => np.cross(np.ones(new Shape(2L, 3L)), np.ones(new Shape(3L, 3L)));
            act.Should().Throw<IncorrectShapeException>().WithMessage(
                "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (2,) and arg 1 with shape (3,).");
            // The broadcast wins over a bad axisc.
            Action both = () => np.cross(np.ones(new Shape(2L, 3L)), np.ones(new Shape(4L, 3L)), axisc: 5);
            both.Should().Throw<IncorrectShapeException>().WithMessage("*arg 1 with shape (4,).");
        }

        [TestMethod]
        public void Cross_Axisc_IsCheckedAgainstTheResultRank()
        {
            Action act = () => np.cross(np.ones(new Shape(2L, 3L)), np.ones(new Shape(2L, 3L)), axisc: 5);
            act.Should().Throw<AxisError>().WithMessage("axisc: axis 5 is out of bounds for array of dimension 2*");
            // Two 2-vectors give the z-components only, and NumPy never checks axisc there.
            np.cross(np.ones(new Shape(2L, 2L)), np.ones(new Shape(2L, 2L)), axisc: 5).ToArray<double>().Should().Equal(0.0, 0.0);
        }

        [TestMethod]
        public void Cross_TextOperand_MeetsTheShapeChecksFirst()
        {
            Action bcast = () => np.cross(np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }), np.ones(new Shape(3L, 3L)));
            bcast.Should().Throw<IncorrectShapeException>().WithMessage("shape mismatch: objects cannot be broadcast*");
            Action loop = () => np.cross(np.array(new[] { "a", "b", "c" }), np.ones(new Shape(3L)));
            loop.Should().Throw<TypeError>().WithMessage(
                "ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U32'), dtype('<U32')) -> None");
        }

        [TestMethod]
        public void TensorInv_IndPastTheRank_IsNumPysSquareCheck()
        {
            // np.linalg.tensorinv(np.eye(4).reshape(4, 2, 2), 3): prod(()) = 1 -> a (1, 16) matrix -> not square.
            Action act = () => np.linalg.tensorinv(np.eye(4).reshape(4, 2, 2), 3);
            act.Should().Throw<LinAlgError>().WithMessage("Last 2 dimensions of the array must be square");
        }
    }
}
