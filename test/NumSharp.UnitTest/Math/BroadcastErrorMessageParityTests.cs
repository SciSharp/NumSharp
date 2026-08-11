using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.MathSuite
{
    /// <summary>
    ///     NumPy 2.4.2 parity for the ELEMENTWISE broadcast-mismatch error text.
    ///
    ///     <para>NumPy raises THREE different broadcast errors depending on the code path
    ///     (all probed against numpy 2.4.2):</para>
    ///     <list type="number">
    ///       <item><b>Elementwise ufunc</b> (<c>a + b</c>, <c>np.add</c>, comparisons, arctan2, shifts):
    ///         <c>operands could not be broadcast together with shapes (2,3) (2,) </c>
    ///         — note the trailing space and the Python-tuple shape repr.</item>
    ///       <item><b>broadcast helpers</b> (<c>broadcast_arrays</c>/<c>broadcast</c>/<c>broadcast_shapes</c>):
    ///         <c>shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 …</c></item>
    ///       <item><b>assignment</b> (<c>x[:] = y</c>):
    ///         <c>could not broadcast input array from shape (2,) into shape (2,3)</c></item>
    ///     </list>
    ///     Before this fix the elementwise path (case 1) leaked NumSharp's generic
    ///     <c>shape mismatch: objects cannot be broadcast to a single shape</c> instead of NumPy's
    ///     operand-listing text. These tests pin case 1 verbatim and guard that cases 2 &amp; 3 are
    ///     UNCHANGED (the fix is scoped to the <c>DefaultEngine.Broadcast(Shape,Shape)</c> wrapper,
    ///     which only the elementwise engine paths call).
    ///
    ///     <para>The exception TYPE stays <see cref="IncorrectShapeException"/> where NumPy raises
    ///     <c>ValueError</c> — the long-standing NumSharp house convention for shape errors (same as
    ///     the reshape family); only the MESSAGE is brought to parity.</para>
    /// </summary>
    [TestClass]
    public class BroadcastErrorMessageParityTests
    {
        private static Exception Capture(Action act)
        {
            try { act(); return null; }
            catch (Exception ex) { return ex; }
        }

        // ---------------------------------------------------------------------
        //  Case 1: elementwise ufuncs -> "operands could not be broadcast together with shapes X Y "
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Add_IncompatibleShapes_MatchesNumpyUfuncText()
        {
            var ex = Capture(() => { var _ = np.ones(new Shape(2, 3)) + np.ones(new Shape(2)); });
            ex.Should().BeOfType<IncorrectShapeException>();
            // Verbatim numpy 2.4.2, incl. the trailing space after the last shape.
            ex.Message.Should().Be("operands could not be broadcast together with shapes (2,3) (2,) ");
        }

        [TestMethod]
        public void Subtract_IncompatibleShapes_MatchesNumpyUfuncText()
        {
            var ex = Capture(() => { var _ = np.ones(new Shape(2, 3)) - np.ones(new Shape(2)); });
            ex.Message.Should().Be("operands could not be broadcast together with shapes (2,3) (2,) ");
        }

        [TestMethod]
        public void Multiply_IncompatibleShapes_MatchesNumpyUfuncText()
        {
            var ex = Capture(() => { var _ = np.ones(new Shape(2, 3)) * np.ones(new Shape(2)); });
            ex.Message.Should().Be("operands could not be broadcast together with shapes (2,3) (2,) ");
        }

        [TestMethod]
        public void Divide_IncompatibleShapes_MatchesNumpyUfuncText()
        {
            var ex = Capture(() => { var _ = np.ones(new Shape(2, 3)) / np.ones(new Shape(2)); });
            ex.Message.Should().Be("operands could not be broadcast together with shapes (2,3) (2,) ");
        }

        [TestMethod]
        public void ThreeDim_vs_TwoDim_ListsBothShapes()
        {
            var ex = Capture(() => { var _ = np.ones(new Shape(2, 3, 4)) + np.ones(new Shape(2, 4)); });
            ex.Message.Should().Be("operands could not be broadcast together with shapes (2,3,4) (2,4) ");
        }

        [TestMethod]
        public void OneDim_vs_OneDim_UsesTrailingCommaTuples()
        {
            // Pins the Python-tuple 1-D repr: "(3,)" and "(4,)", not "(3)"/"(4)".
            var ex = Capture(() => { var _ = np.ones(new Shape(3)) + np.ones(new Shape(4)); });
            ex.Message.Should().Be("operands could not be broadcast together with shapes (3,) (4,) ");
        }

        [TestMethod]
        public void Comparison_IncompatibleShapes_MatchesNumpyUfuncText()
        {
            var ex = Capture(() => { var _ = np.ones(new Shape(2, 3)) > np.ones(new Shape(2)); });
            ex.Should().BeOfType<IncorrectShapeException>();
            ex.Message.Should().Be("operands could not be broadcast together with shapes (2,3) (2,) ");
        }

        [TestMethod]
        public void Arctan2_IncompatibleShapes_MatchesNumpyUfuncText()
        {
            var ex = Capture(() => { var _ = np.arctan2(np.ones(new Shape(2, 3)), np.ones(new Shape(2))); });
            ex.Message.Should().Be("operands could not be broadcast together with shapes (2,3) (2,) ");
        }

        // ---------------------------------------------------------------------
        //  Regression guards: the fix must NOT leak into the other two paths
        // ---------------------------------------------------------------------

        [TestMethod]
        public void BroadcastArrays_KeepsItsOwnWording_NotTheUfuncText()
        {
            // broadcast_arrays calls Shape.Broadcast directly (bypassing the elementwise wrapper),
            // so it must keep NumPy's "shape mismatch: …" wording, NOT the ufunc operand-listing text.
            var ex = Capture(() => { var _ = np.broadcast_arrays(np.ones(new Shape(2, 3)), np.ones(new Shape(2))); });
            ex.Should().NotBeNull();
            ex.Message.Should().Contain("shape mismatch");
            ex.Message.Should().NotContain("operands could not be broadcast together");
        }

        [TestMethod]
        public void Assignment_KeepsInputIntoShapeText()
        {
            // x[:] = y with a non-broadcastable RHS -> NumPy's assignment-specific message (already correct).
            var ex = Capture(() => { var x = np.zeros(new Shape(2, 3)); x[":"] = np.ones(new Shape(2)); });
            ex.Should().NotBeNull();
            ex.GetType().Name.Should().Be("ValueError");
            ex.Message.Should().Be("could not broadcast input array from shape (2,) into shape (2,3)");
        }
    }
}
