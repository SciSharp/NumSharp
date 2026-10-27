using System;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    ///     np.logical_and / or / xor / not against NumPy 2.4.2, one test per D4 fix of the np-layer dtype plan's P2 family
    ///     (the logical ufuncs, now <c>TensorEngine.LogicalAnd</c> ... <c>LogicalNot</c>): logical_not's answers for a text
    ///     operand under <c>dtype=</c>, <c>out=</c> and <c>where=</c>; logical_not's broadcast error over a numeric operand.
    /// </summary>
    [TestClass]
    public class np_logical_NumPyPort_Tests
    {
        // ------------------------------------------------------------------------------- logical_not over text

        [TestMethod]
        public void LogicalNot_TextOperand_AnswersAsNumPyUnderDtypeOutAndWhere()
        {
            // logical_not has no string loop and its loops are bool-output only, so with a text operand NumPy answers:
            // out= / where= parse errors first, then a dtype= other than bool finds no loop at all, while dtype=bool binds
            // the ?->? loop and the input cast from text fails; a StringDType operand's no-loop text names the requested
            // output class (all probed against 2.4.2).
            var u = np.array(new[] { "ab", "", "c" });
            var s = u.astype(np.dtype("S5"));
            var t = u.astype(np.dtype("T"));
            var readOnly = np.zeros(new Shape(3), np.@bool);
            readOnly.setflags(write: false);
            const string noLoop = "No loop matching the specified signature and casting was found for ufunc logical_not";

            new Action(() => np.logical_not(u, dtype: np.@bool)).Should().ThrowExactly<ArgumentException>()
                .WithMessage("Cannot cast ufunc 'logical_not' input from dtype('<U2') to dtype('bool') with casting rule 'same_kind'");
            new Action(() => np.logical_not(s, dtype: np.float64)).Should().ThrowExactly<IncorrectTypeException>().WithMessage(noLoop);
            new Action(() => np.logical_not(u, dtype: np.int8)).Should().ThrowExactly<IncorrectTypeException>().WithMessage(noLoop);
            new Action(() => np.logical_not(t, dtype: np.float64)).Should().ThrowExactly<TypeError>().WithMessage(
                "ufunc 'logical_not' did not contain a loop with signature matching types <class 'numpy.dtypes.StringDType'> -> <class 'numpy.dtypes.Float64DType'>");
            new Action(() => np.logical_not(t, dtype: np.@bool)).Should().ThrowExactly<TypeError>().WithMessage(
                "ufunc 'logical_not' did not contain a loop with signature matching types <class 'numpy.dtypes.StringDType'> -> <class 'numpy.dtypes.BoolDType'>");
            // The parse errors come first, ahead of the loop answer.
            new Action(() => np.logical_not(u, @out: readOnly, dtype: np.float64)).Should().ThrowExactly<ValueError>()
                .WithMessage("output array is read-only");
            new Action(() => np.logical_not(u, where: np.ones(new Shape(3), np.int64), dtype: np.float64)).Should()
                .Throw<ArgumentException>().WithMessage("Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
            // Without dtype= the answers were NumPy's already.
            new Action(() => np.logical_not(u)).Should().ThrowExactly<TypeError>().WithMessage(
                "ufunc 'logical_not' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''");
            new Action(() => np.logical_not(t)).Should().ThrowExactly<TypeError>().WithMessage(
                "ufunc 'logical_not' did not contain a loop with signature matching types <class 'numpy.dtypes.StringDType'> -> None");
        }

        // ------------------------------------------------------------------------------- logical_not's broadcast text

        [TestMethod]
        public void LogicalNot_NumericOperand_BroadcastErrorListsTheUnaryOperands()
        {
            // logical_not is unary: NumPy's iterator lists the operand, then out= and where= when given. NumSharp computes a
            // non-bool operand's logical NOT as the comparison x == 0, whose own error listed its hidden 0-d zero as an
            // operand ("(3,4) () (3,4,2)"). Texts probed against 2.4.2.
            var a = np.arange(12.0).reshape(3, 4);
            var wrong = np.zeros(new Shape(3, 4, 2), np.@bool);
            new Action(() => np.logical_not(a, @out: wrong)).Should().Throw<ArgumentException>()
                .WithMessage("operands could not be broadcast together with shapes (3,4) (3,4,2) ");
            new Action(() => np.logical_not(a, where: np.ones(new Shape(3, 4, 2), np.@bool))).Should().Throw<ArgumentException>()
                .WithMessage("operands could not be broadcast together with shapes (3,4) (3,4,2) ");
            new Action(() => np.logical_not(a, @out: np.zeros(new Shape(3, 4), np.@bool), where: np.ones(new Shape(3, 4, 2), np.@bool)))
                .Should().Throw<ArgumentException>().WithMessage("operands could not be broadcast together with shapes (3,4) (3,4) (3,4,2) ");
            new Action(() => np.logical_not(np.zeros(new Shape(0, 3), np.int8), @out: np.zeros(new Shape(0, 3, 2), np.@bool)))
                .Should().Throw<ArgumentException>().WithMessage("operands could not be broadcast together with shapes (0,3) (0,3,2) ");
            new Action(() => np.logical_not(a, @out: np.zeros(new Shape(4), np.@bool))).Should().Throw<ArgumentException>()
                .WithMessage("non-broadcastable output operand with shape (4,) doesn't match the broadcast shape (3,4)");
            // NumPy parses out= and where= before it broadcasts: a non-bool mask and a read-only out= are reported ahead of
            // an operand that does not fit.
            new Action(() => np.logical_not(a, @out: wrong, where: np.ones(new Shape(3, 4), np.int64))).Should()
                .Throw<ArgumentException>().WithMessage("Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
            var readOnly = np.zeros(new Shape(3, 4), np.@bool);
            readOnly.setflags(write: false);
            new Action(() => np.logical_not(a, @out: readOnly, where: np.ones(new Shape(3, 4, 2), np.@bool))).Should()
                .ThrowExactly<ValueError>().WithMessage("output array is read-only");
            // A bool operand takes the unary invert, whose text was NumPy's already.
            new Action(() => np.logical_not(a > 5, @out: wrong)).Should().Throw<ArgumentException>()
                .WithMessage("operands could not be broadcast together with shapes (3,4) (3,4,2) ");
        }
    }
}
