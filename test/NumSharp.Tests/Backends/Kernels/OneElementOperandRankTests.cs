using System;
using System.Collections.Generic;
using AwesomeAssertions;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     A binary ufunc or comparison whose one-element operand has MORE dimensions than the other operand returns the
    ///     broadcast shape, as in NumPy: <c>(7,) op (1,1)</c> is <c>(1,7)</c> and <c>(3,7) op (1,1,1)</c> is
    ///     <c>(1,3,7)</c> (plan <c>docs/plans/api-parameter-parity.md</c> X19). The scalar-broadcast shortcuts of the
    ///     binary, comparison and shift routes gave the result the array operand's shape, dropping the leading axes, and so
    ///     did np.power's old scalar-exponent fast path: 100 of the 165 calls of this file's first test were wrong before
    ///     the fix.
    /// </summary>
    /// <remarks>Shapes and values are NumPy 2.4.2's for the same calls (probed 2026-10-06).</remarks>
    [TestClass]
    public class OneElementOperandRankTests : TestClass
    {
        /// <summary>The binary calls the probe covered, each applied to (lhs, rhs).</summary>
        private static readonly (string Name, Func<NDArray, NDArray, NDArray> Call)[] FloatCalls =
        {
            ("add", (a, b) => np.add(a, b)), ("subtract", (a, b) => np.subtract(a, b)),
            ("multiply", (a, b) => np.multiply(a, b)), ("divide", (a, b) => np.divide(a, b)),
            ("power", (a, b) => np.power(a, b)), ("mod", (a, b) => np.mod(a, b)),
            ("floor_divide", (a, b) => np.floor_divide(a, b)), ("fmod", (a, b) => np.fmod(a, b)),
            ("maximum", (a, b) => np.maximum(a, b)), ("minimum", (a, b) => np.minimum(a, b)),
            ("fmax", (a, b) => np.fmax(a, b)), ("fmin", (a, b) => np.fmin(a, b)), ("arctan2", (a, b) => np.arctan2(a, b)),
            ("equal", (a, b) => np.equal(a, b)), ("less", (a, b) => np.less(a, b)),
            ("greater_equal", (a, b) => np.greater_equal(a, b)), ("logical_and", (a, b) => np.logical_and(a, b)),
            ("logical_or", (a, b) => np.logical_or(a, b)), ("logical_xor", (a, b) => np.logical_xor(a, b)),
            ("float_power", (a, b) => np.float_power(a, b)), ("operator +", (a, b) => a + b),
            ("operator *", (a, b) => a * b), ("operator <", (a, b) => a < b), ("operator ==", (a, b) => a == b),
        };

        /// <summary>The integer-only calls, applied to int32 operands.</summary>
        private static readonly (string Name, Func<NDArray, NDArray, NDArray> Call)[] IntCalls =
        {
            ("bitwise_and", (a, b) => np.bitwise_and(a, b)), ("bitwise_or", (a, b) => np.bitwise_or(a, b)),
            ("bitwise_xor", (a, b) => np.bitwise_xor(a, b)), ("left_shift", (a, b) => np.left_shift(a, b)),
            ("right_shift", (a, b) => np.right_shift(a, b)), ("gcd", (a, b) => np.gcd(a, b)), ("lcm", (a, b) => np.lcm(a, b)),
            ("add", (a, b) => np.add(a, b)), ("power", (a, b) => np.power(a, b)),
        };

        /// <summary>
        ///     Every binary ufunc, comparison, logical op, shift and operator returns NumPy's broadcast shape when a
        ///     one-element operand has more dimensions, whichever side it is on.
        /// </summary>
        [TestMethod]
        public void HigherRankOneElementOperand_KeepsItsLeadingAxes()
        {
            var cases = new (string Name, NDArray Lhs, NDArray Rhs, long[] Want)[]
            {
                ("(7,) op (1,1)", np.arange(1.0, 8.0), np.full(new Shape(1, 1), 2.0), new long[] { 1, 7 }),
                ("(1,1) op (7,)", np.full(new Shape(1, 1), 2.0), np.arange(1.0, 8.0), new long[] { 1, 7 }),
                ("(3,7) op (1,1,1)", np.arange(1.0, 22.0).reshape(3, 7), np.full(new Shape(1, 1, 1), 2.0), new long[] { 1, 3, 7 }),
                ("0-d op (1,)", NDArray.Scalar(3.0), np.full(new Shape(1), 2.0), new long[] { 1 }),
                ("(1,) op (1,1)", np.full(new Shape(1), 3.0), np.full(new Shape(1, 1), 2.0), new long[] { 1, 1 }),
            };
            var wrong = new List<string>();
            foreach (var (name, lhs, rhs, want) in cases)
            {
                foreach (var (fn, call) in FloatCalls)
                    Record(wrong, $"{fn} {name}", () => call(lhs, rhs), want);
                foreach (var (fn, call) in IntCalls)
                    Record(wrong, $"{fn} int32 {name}", () => call(lhs.astype(np.int32), rhs.astype(np.int32)), want);
            }
            string.Join(Environment.NewLine, wrong).Should().BeEmpty("a one-element operand of higher rank adds its leading axes");
        }

        /// <summary>The values along the kept axes are NumPy's too (the shortcut computed them right but shaped them wrong).</summary>
        [TestMethod]
        public void HigherRankOneElementOperand_ValuesAreNumPys()
        {
            var r = np.add(np.arange(1.0, 8.0), np.full(new Shape(1, 1), 2.0));
            r.shape.Should().Equal(1L, 7L);
            r.ToArray<double>().Should().Equal(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
            var c = np.less(np.full(new Shape(1, 1), 4.0), np.arange(1.0, 8.0));
            c.shape.Should().Equal(1L, 7L);
            c.ToArray<bool>().Should().Equal(false, false, false, false, true, true, true);
            var s = np.left_shift(np.arange(1, 4).astype(np.int32), np.full(new Shape(1, 1), 2).astype(np.int32));
            s.shape.Should().Equal(1L, 3L);
            s.ToArray<int>().Should().Equal(4, 8, 12);
        }

        /// <summary>Runs one call and records a line when its shape is not <paramref name="want"/> (or it throws).</summary>
        /// <param name="wrong">Collects one line per wrong call.</param>
        /// <param name="label">The call's label.</param>
        /// <param name="call">The call.</param>
        /// <param name="want">NumPy's result shape.</param>
        private static void Record(List<string> wrong, string label, Func<NDArray> call, long[] want)
        {
            try
            {
                var got = call().shape;
                if (!got.AsSpan().SequenceEqual(want))
                    wrong.Add($"{label}: ({string.Join(",", got)}) != NumPy ({string.Join(",", want)})");
            }
            catch (Exception e)
            {
                wrong.Add($"{label}: threw {e.GetType().Name}: {e.Message}");
            }
        }
    }
}
