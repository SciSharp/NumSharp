using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     Tests for <see cref="np.broadcast_shapes"/>, verified against NumPy 2.4.2 output.
    ///     broadcast_shapes computes the joint broadcast shape of its arguments WITHOUT allocating,
    ///     accepting tuples / bare ints / int[] as shapes; the result is asserted through
    ///     <see cref="Shape.ToString"/> since the function returns a <see cref="Shape"/>.
    ///
    ///     NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.broadcast_shapes.html
    ///     NumPy source: numpy/lib/_stride_tricks_impl.py (broadcast_shapes / _broadcast_shape)
    /// </summary>
    [TestClass]
    public class np_broadcast_shapes_Test
    {
        // ================================================================
        //  Happy path — every argument form NumPy accepts
        // ================================================================

        [TestMethod]
        public void Empty_ReturnsScalarShape()
        {
            // np.broadcast_shapes() -> ()
            np.broadcast_shapes().ToString().Should().Be("()");
        }

        [TestMethod]
        public void BareInts_TreatedAsVectorShapes()
        {
            // np.broadcast_shapes(3, 1) -> (3,)  (each int is the shape (n,))
            np.broadcast_shapes(3, 1).ToString().Should().Be("(3)");
        }

        [TestMethod]
        public void MixedIntAndTuple()
        {
            // np.broadcast_shapes(5, (3,1)) -> (3, 5)
            np.broadcast_shapes(5, (3, 1)).ToString().Should().Be("(3, 5)");
        }

        [TestMethod]
        public void Basic_ThreeTuples()
        {
            // np.broadcast_shapes((1,2),(3,1),(3,2)) -> (3, 2)
            np.broadcast_shapes((1, 2), (3, 1), (3, 2)).ToString().Should().Be("(3, 2)");
        }

        [TestMethod]
        public void RankDifference_RightAligned()
        {
            // np.broadcast_shapes((6,7),(5,6,1),7,(5,1,7)) -> (5, 6, 7)
            np.broadcast_shapes((6, 7), (5, 6, 1), 7, (5, 1, 7)).ToString().Should().Be("(5, 6, 7)");
        }

        [TestMethod]
        public void ZeroDimension_WinsOverOne()
        {
            // A zero-length axis is NOT size-1: (0,1) broadcast (1,1) -> (0, 1)
            np.broadcast_shapes((0, 1), (1, 1)).ToString().Should().Be("(0, 1)");
            // (0,) with (0,) -> (0,)
            np.broadcast_shapes(0, 0).ToString().Should().Be("(0)");
        }

        [TestMethod]
        public void Single_ReturnsItself()
        {
            np.broadcast_shapes((2, 3)).ToString().Should().Be("(2, 3)");
        }

        [TestMethod]
        public void IntArrayShapes()
        {
            // int[] arguments (e.g. a.shape) convert to Shape and broadcast the same
            np.broadcast_shapes(new[] { 1, 4 }, new[] { 3, 1 }).ToString().Should().Be("(3, 4)");
        }

        // ================================================================
        //  Error parity — message is NumPy's verbatim (arg indices + shapes)
        // ================================================================

        [TestMethod]
        public void Incompatible_1D_ThrowsWithArgDetail()
        {
            // NumPy: ValueError; NumSharp: IncorrectShapeException (house convention for shape
            // broadcast errors), message byte-identical to NumPy including the two spaces and the
            // Python-repr shapes.
            ((Action)(() => np.broadcast_shapes(3, 4)))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  " +
                             "Mismatch is between arg 0 with shape (3,) and arg 1 with shape (4,).");
        }

        [TestMethod]
        public void Incompatible_NamesEstablishingArg_NotRunningShape()
        {
            // ((2,3),(3,),(4,)): arg 0 establishes the trailing extent 3, arg 1 (=3) matches it, and
            // arg 2 (=4) conflicts — so the report names the ESTABLISHING arg 0, not arg 1. (In C#
            // the 1-D shapes (3,) and (4,) are the bare ints 3 and 4 via int->Shape.)
            ((Action)(() => np.broadcast_shapes((2, 3), 3, 4)))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("*Mismatch is between arg 0 with shape (2, 3) and arg 2 with shape (4,).");
        }

        [TestMethod]
        public void Incompatible_LaterArgEstablishesExtent()
        {
            // (1,5,2): arg 0 is size-1 (never sets), arg 1 (=5) establishes, arg 2 (=2) conflicts.
            ((Action)(() => np.broadcast_shapes(1, 5, 2)))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("*Mismatch is between arg 1 with shape (5,) and arg 2 with shape (2,).");
        }

        [TestMethod]
        public void ZeroVsNonUnit_Incompatible()
        {
            // 0 and 2 do not broadcast (0 is not size-1).
            ((Action)(() => np.broadcast_shapes((0, 1), (2, 1))))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("*Mismatch is between arg 0 with shape (0, 1) and arg 1 with shape (2, 1).");
        }

        [TestMethod]
        public void NegativeDimension_ThrowsValueError()
        {
            // NumPy reaches this through np.empty(x); message verbatim.
            ((Action)(() => np.broadcast_shapes(-1)))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("negative dimensions are not allowed");
        }

        [TestMethod]
        public void NegativeDimension_ReportedBeforeBroadcastMismatch()
        {
            // A negative extent is rejected up front, ahead of an otherwise-incompatible pair.
            ((Action)(() => np.broadcast_shapes((3, -1), (4, 2))))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("negative dimensions are not allowed");
        }

        // ================================================================
        //  The pair a mismatch names: np.broadcast's multi-iterator scan
        // ================================================================

        /// <summary>
        ///     Two axes conflict: NumPy's PyArray_Broadcast scans axis by axis, so it names the pair of the FIRST axis (args 2
        ///     and 3 disagree on axis 0) — not the first operand that conflicts anywhere (arg 1 against arg 0 on axis 1), which
        ///     an operand-by-operand scan reported.
        /// </summary>
        [TestMethod]
        public void TwoConflictingAxes_NamesTheLeftmostAxisPair()
        {
            // np.broadcast_shapes((1,4), (1,5), (2,1), (3,1))
            ((Action)(() => np.broadcast_shapes((1, 4), (1, 5), (2, 1), (3, 1))))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  " +
                             "Mismatch is between arg 2 with shape (2, 1) and arg 3 with shape (3, 1).");
        }

        /// <summary>
        ///     Past 64 operands NumPy broadcasts in groups (np.broadcast takes 64): the 67th operand is argument 3 of the second
        ///     call, whose argument 0 is the shape the first 64 broadcast to.
        /// </summary>
        [TestMethod]
        public void PastSixtyFourOperands_NumbersTheOperandsWithinTheirGroup()
        {
            // shapes = [(1,)] * 70; shapes[3] = (4,); shapes[66] = (5,); np.broadcast_shapes(*shapes)
            var shapes = new Shape[70];
            for (int i = 0; i < shapes.Length; i++)
                shapes[i] = new Shape(1);
            shapes[3] = new Shape(4);
            shapes[66] = new Shape(5);
            ((Action)(() => np.broadcast_shapes(shapes)))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  " +
                             "Mismatch is between arg 0 with shape (4,) and arg 3 with shape (5,).");

            // Both operands in the second group: shapes[66] = (4,), shapes[68] = (5,) are its arguments 3 and 5.
            for (int i = 0; i < shapes.Length; i++)
                shapes[i] = new Shape(1);
            shapes[66] = new Shape(4);
            shapes[68] = new Shape(5);
            ((Action)(() => np.broadcast_shapes(shapes)))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  " +
                             "Mismatch is between arg 3 with shape (4,) and arg 5 with shape (5,).");

            // 130 operands are three groups (64, 63, 3): shapes[129] is argument 3 of the third call.
            shapes = new Shape[130];
            for (int i = 0; i < shapes.Length; i++)
                shapes[i] = new Shape(1);
            shapes[2] = new Shape(3, 1);
            shapes[129] = new Shape(2, 5);
            ((Action)(() => np.broadcast_shapes(shapes)))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  " +
                             "Mismatch is between arg 0 with shape (3, 1) and arg 3 with shape (2, 5).");
        }

        /// <summary>
        ///     A broadcast whose element count does not fit 64 bits is np.broadcast's ValueError, though each operand exists:
        ///     <c>np.broadcast_shapes((2**40,), (2**40, 1))</c>.
        /// </summary>
        [TestMethod]
        public void ElementCountOverflow_ThrowsValueError()
        {
            ((Action)(() => np.broadcast_shapes(new Shape(1L << 40), new Shape(1L << 40, 1))))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("broadcast dimensions too large.");
        }
    }
}
