using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     A ufunc cast refusal is NumPy's <c>UFuncTypeError</c>, a <c>TypeError</c> subclass, and NumSharp raises
    ///     <see cref="TypeError"/> with NumPy's text for every one of them: an <c>out=</c> the loop cannot be cast into, a
    ///     <c>dtype=</c> loop an input cannot reach, np.around's float64 composition cast back into a bool or integer array,
    ///     whatever the route (parity QA-04). It is not an <see cref="ArgumentException"/>: code that caught the former house
    ///     type for these refusals must catch <see cref="TypeError"/>.
    /// </summary>
    /// <remarks>Every expected text is NumPy 2.4.2's for the same call (probed on win-amd64).</remarks>
    [TestClass]
    public class UfuncCastRefusalTests
    {
        /// <summary>Asserts the call raises exactly <see cref="TypeError"/> with NumPy's text.</summary>
        /// <param name="call">The refused call.</param>
        /// <param name="text">NumPy's <c>UFuncTypeError</c> text, verbatim.</param>
        private static void Refuses(Action call, string text) => call.Should().ThrowExactly<TypeError>().WithMessage(text);

        /// <summary>np.around of a bool array with decimals != 0 multiplies (or divides) in float64 and cannot cast back.</summary>
        [TestMethod]
        public void Around_BoolWithDecimals()
        {
            using var b = np.array(new[] { true, false });
            Refuses(() => np.around(b, 1),
                "Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('bool') with casting rule 'same_kind'");
            Refuses(() => np.around(b, -1),
                "Cannot cast ufunc 'divide' output from dtype('float64') to dtype('bool') with casting rule 'same_kind'");
        }

        /// <summary>np.evaluate's Round node refuses the same bool input at type inference, with the same text.</summary>
        [TestMethod]
        public void Evaluate_RoundOfBool()
        {
            using var b = np.array(new[] { true, false });
            Refuses(() => np.evaluate(NDExpr.Round(NDExpr.Arr(b), 1)),
                "Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('bool') with casting rule 'same_kind'");
            Refuses(() => np.evaluate(NDExpr.Round(NDExpr.Arr(b), -1)),
                "Cannot cast ufunc 'divide' output from dtype('float64') to dtype('bool') with casting rule 'same_kind'");
        }

        /// <summary>An out= the float64 loop cannot be cast into: a binary ufunc, a unary ufunc, and np.around's multiply.</summary>
        [TestMethod]
        public void OutCasts()
        {
            using var x = np.array(new[] { 1.5 });
            using var y = np.array(new[] { 1.0 });
            using var i4 = np.zeros(new Shape(1), np.int32);
            Refuses(() => np.add(x, y, i4),
                "Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
            Refuses(() => np.sqrt(x, i4),
                "Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
            Refuses(() => np.around(x, 1, i4),
                "Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
        }

        /// <summary>A dtype= loop an input cannot reach: binary refusals number the input, unary ones do not.</summary>
        [TestMethod]
        public void DtypeInputCasts()
        {
            using var x = np.array(new[] { 1.5 });
            using var y = np.array(new[] { 1.0 });
            using var z = np.array(new[] { new Complex(1, 1) });
            Refuses(() => np.add(x, y, dtype: NPTypeCode.Int32),
                "Cannot cast ufunc 'add' input 0 from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
            Refuses(() => np.fabs(z, dtype: NPTypeCode.Double),
                "Cannot cast ufunc 'fabs' input from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");
        }

        /// <summary>The refusal is not an <see cref="ArgumentException"/>: a handler for that type no longer sees it, one
        ///     for <see cref="TypeError"/> (NumPy's <c>except TypeError</c>) does.</summary>
        [TestMethod]
        public void IsNotAnArgumentException()
        {
            using var x = np.array(new[] { 1.5 });
            using var i4 = np.zeros(new Shape(1), np.int32);
            Exception caught = null;
            try
            {
                np.add(x, x, i4);
            }
            catch (Exception e)
            {
                caught = e;
            }

            caught.Should().BeOfType<TypeError>();
            caught.Should().NotBeAssignableTo<ArgumentException>();
        }
    }
}
