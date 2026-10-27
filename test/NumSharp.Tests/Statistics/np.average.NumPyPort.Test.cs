using System;

namespace NumSharp.Tests.Statistics
{
    /// <summary>
    ///     np.average's statements against NumPy 2.4.2 (<c>numpy/lib/_function_base_impl.py</c>, <c>average</c> and
    ///     <c>_weights_are_valid</c>), one test per D4 fix of the np-layer dtype plan's P2 family: NumPy's exception types,
    ///     NumPy's promotion of text weights and the refusals that follow it.
    /// </summary>
    [TestClass]
    public class np_average_NumPyPort_Tests
    {
        // ------------------------------------------------------------------------------- _weights_are_valid's types

        [TestMethod]
        public void WeightsOfAnotherShape_WithoutAnAxis_IsTypeError_OnEveryOperandKind()
        {
            // NumPy: `if axis is None: raise TypeError("Axis must be specified when shapes of a and weights differ.")`,
            // before the result dtype is computed - so numeric, U, S and StringDType operands all raise it.
            const string text = "Axis must be specified when shapes of a and weights differ.";
            var w = np.ones(new Shape(3));
            new Action(() => np.average(np.ones(new Shape(2, 3)), (int?)null, w)).Should().ThrowExactly<TypeError>().WithMessage(text);
            new Action(() => np.average(np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }), (int?)null, w))
                .Should().ThrowExactly<TypeError>().WithMessage(text);
            new Action(() => np.average(np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }).astype(np.dtype("S1")), (int?)null, w))
                .Should().ThrowExactly<TypeError>().WithMessage(text);
            new Action(() => np.average(np.array(2.5), (int?)null, np.ones(new Shape(1)))).Should().ThrowExactly<TypeError>().WithMessage(text);
        }

        [TestMethod]
        public void WeightsNotOfTheAxesShape_IsValueError_TheEmptyTupleIncluded()
        {
            // NumPy: `if wgt.shape != tuple(a.shape[ax] for ax in axis): raise ValueError(...)`. An empty tuple is an axis
            // given (not None): weights of another shape fail this check, also for a 0-d operand, whose only tuple is ().
            const string text = "Shape of weights must be consistent with shape of a along specified axis.";
            new Action(() => np.average(np.ones(new Shape(2, 3)), 0, np.ones(new Shape(3)))).Should().ThrowExactly<ValueError>().WithMessage(text);
            new Action(() => np.average(np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }), 0, np.ones(new Shape(3))))
                .Should().ThrowExactly<ValueError>().WithMessage(text);
            new Action(() => np.average(np.array(2.5), Array.Empty<int>(), np.ones(new Shape(1)))).Should().ThrowExactly<ValueError>().WithMessage(text);
            new Action(() => np.average(np.array(new[] { "a" }).astype(np.dtype("T")).reshape(Shape.Scalar), Array.Empty<int>(), np.ones(new Shape(1))))
                .Should().ThrowExactly<ValueError>().WithMessage(text);
        }
    }
}
