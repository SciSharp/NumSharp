using System;

namespace NumSharp.Tests.Math
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

        /// <summary>
        ///     A view assignment into a STRIDED region (a column slice of a 2-D array, an empty one too) is NumPy's
        ///     <c>PyArray_AssignArray</c> as well: a ValueError with NumPy's text, raised before anything is written.
        ///     NumSharp's copy iterator raised <c>np.broadcast_to</c>'s IncorrectShapeException text there
        ///     ("operands could not be broadcast together with remapped shapes ...").
        /// </summary>
        [TestMethod]
        public void Assignment_IntoAStridedRegion_RaisesNumPysText()
        {
            // >>> x = np.zeros((3, 6)); x[:, 1:3] = np.ones(3)
            // ValueError: could not broadcast input array from shape (3,) into shape (3,2)
            var x = np.zeros(new Shape(3, 6));
            var ex = Capture(() => { x[":, 1:3"] = np.ones(new Shape(3)); });
            ex.Should().BeOfType<ValueError>();
            ex.Message.Should().Be("could not broadcast input array from shape (3,) into shape (3,2)");
            x.array_equal(np.zeros(new Shape(3, 6))).Should().BeTrue("nothing is written before the refusal");
            // >>> x = np.zeros((2, 4)); x[:, 1:2] = np.ones((3, 1))
            // ValueError: could not broadcast input array from shape (3,1) into shape (2,1)
            ex = Capture(() => { var y = np.zeros(new Shape(2, 4)); y[":, 1:2"] = np.ones(new Shape(3, 1)); });
            ex.Should().BeOfType<ValueError>();
            ex.Message.Should().Be("could not broadcast input array from shape (3,1) into shape (2,1)");
            // >>> x = np.zeros((3, 4)); x[:, 1:1] = np.ones(2)  — an empty region still requires the broadcast
            // ValueError: could not broadcast input array from shape (2,) into shape (3,0)
            ex = Capture(() => { var y = np.zeros(new Shape(3, 4)); y[":, 1:1"] = np.ones(new Shape(2)); });
            ex.Should().BeOfType<ValueError>();
            ex.Message.Should().Be("could not broadcast input array from shape (2,) into shape (3,0)");
            // >>> x[:, 1:1] = np.ones((1, 1, 1))  is a no-op (a one-element value fits any region)
            var z = np.zeros(new Shape(3, 4));
            z[":, 1:1"] = np.ones(new Shape(1, 1, 1));
            z.array_equal(np.zeros(new Shape(3, 4))).Should().BeTrue();
        }

        /// <summary>
        ///     NumPy strips a value's leading extent-1 axes beyond the region's rank before it broadcasts, and its error
        ///     prints the STRIPPED shape; into a strided region such a value is assigned. NumSharp printed the value's own
        ///     shape (<c>(1,2)</c> where NumPy prints <c>(2,)</c>) and refused the strided assignment.
        /// </summary>
        [TestMethod]
        public void Assignment_StripsLeadingUnitAxes_AsNumPysAssignArrayDoes()
        {
            // >>> x = np.zeros(14); x[1:2] = np.ones((1, 2))
            // ValueError: could not broadcast input array from shape (2,) into shape (1,)
            var ex = Capture(() => { var x = np.zeros(new Shape(14)); x["1:2"] = np.ones(new Shape(1, 2)); });
            ex.Should().BeOfType<ValueError>();
            ex.Message.Should().Be("could not broadcast input array from shape (2,) into shape (1,)");
            // >>> x = np.zeros((3, 3)); x[0] = np.ones((1, 2, 3))   (only the leading 1 is stripped)
            // ValueError: could not broadcast input array from shape (2,3) into shape (3,)
            ex = Capture(() => { var x = np.zeros(new Shape(3, 3)); x["0"] = np.ones(new Shape(1, 2, 3)); });
            ex.Should().BeOfType<ValueError>();
            ex.Message.Should().Be("could not broadcast input array from shape (2,3) into shape (3,)");
            // >>> x = np.zeros((2, 4)); x[:, 1:2] = np.ones((1, 1, 2))
            // ValueError: could not broadcast input array from shape (1,2) into shape (2,1)
            ex = Capture(() => { var x = np.zeros(new Shape(2, 4)); x[":, 1:2"] = np.ones(new Shape(1, 1, 2)); });
            ex.Should().BeOfType<ValueError>();
            ex.Message.Should().Be("could not broadcast input array from shape (1,2) into shape (2,1)");
            // >>> x = np.zeros((2, 6)); x[:, ::2] = np.arange(3.).reshape(1, 1, 3)
            // array([[0., 0., 1., 0., 2., 0.], [0., 0., 1., 0., 2., 0.]])
            var y = np.zeros(new Shape(2, 6));
            y[":, ::2"] = np.arange(3.0).reshape(1, 1, 3);
            y.array_equal(np.array(new double[,] { { 0, 0, 1, 0, 2, 0 }, { 0, 0, 1, 0, 2, 0 } })).Should().BeTrue();
        }
    }
}
