using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.r_ / np.c_'s slice branch computes with the Python numbers its fields spell: digits are a Python int (exact at
    ///     any size), anything else a Python float. NumPy's <c>arange(start, stop, step)</c> then takes its int route when
    ///     every field is an int — Python's exact length, exact int64 values past 2**53 and Python's int division, whose
    ///     zero step is <c>division by zero</c> — and its float route otherwise (<c>float division by zero</c>). Every
    ///     expectation is NumPy 2.4.2's output (Windows and Linux agree).
    /// </summary>
    [TestClass]
    public class np_r_python_numbers_tests
    {
        /// <summary>
        ///     A zero step is Python's ZeroDivisionError from <c>(stop - start) / step</c>: int / int says
        ///     <c>division by zero</c>, a float on either side <c>float division by zero</c> (NumSharp: DivideByZeroException).
        ///     The int texts were <c>float division by zero</c>, the float route's.
        /// </summary>
        [TestMethod]
        public void ZeroStep_IsPythonsDivisionByZero_IntOrFloat()
        {
            // np.r_[0:5:0], np.r_[5::0], np.r_[::0], np.c_[0:5:0] -> ZeroDivisionError: division by zero
            new Action(() => { var _ = np.r_["0:5:0"]; }).Should().Throw<DivideByZeroException>().WithMessage("division by zero");
            new Action(() => { var _ = np.r_["5::0"]; }).Should().Throw<DivideByZeroException>().WithMessage("division by zero");
            new Action(() => { var _ = np.r_["::0"]; }).Should().Throw<DivideByZeroException>().WithMessage("division by zero");
            new Action(() => { var _ = np.c_["0:5:0"]; }).Should().Throw<DivideByZeroException>().WithMessage("division by zero");
            // np.r_[0.0:5:0], np.r_[0:5:0.0] -> ZeroDivisionError: float division by zero
            new Action(() => { var _ = np.r_["0.0:5:0"]; }).Should().Throw<DivideByZeroException>()
                .WithMessage("float division by zero");
            new Action(() => { var _ = np.r_["0:5:0.0"]; }).Should().Throw<DivideByZeroException>()
                .WithMessage("float division by zero");
        }

        /// <summary>
        ///     Python ints past 2**53 are exact: NumPy's length and values come from Python's int arithmetic. A double
        ///     reading of the bounds rounded 2**53 + 1 down and 2**53 + 3 up, which made four elements of two.
        /// </summary>
        [TestMethod]
        public void IntsPast2To53_AreExact()
        {
            // np.r_[9007199254740993:9007199254740995] -> int64 [9007199254740993, 9007199254740994]
            var a = np.r_["9007199254740993:9007199254740995"];
            a.Should().BeOfType(NPTypeCode.Int64).And.BeShaped(2);
            a.GetInt64(0).Should().Be(9007199254740993L);
            a.GetInt64(1).Should().Be(9007199254740994L);

            // np.r_[-2**63:-2**63+3] -> [-9223372036854775808, -9223372036854775807, -9223372036854775806]
            var b = np.r_["-9223372036854775808:-9223372036854775805"];
            b.Should().BeOfType(NPTypeCode.Int64).And.BeShaped(3);
            b.GetInt64(0).Should().Be(long.MinValue);
            b.GetInt64(2).Should().Be(long.MinValue + 2);

            // np.r_[2**63-3:2**63-1] -> [9223372036854775805, 9223372036854775806]
            var c = np.r_["9223372036854775805:9223372036854775807"];
            c.Should().BeShaped(2);
            c.GetInt64(0).Should().Be(long.MaxValue - 2);
            c.GetInt64(1).Should().Be(long.MaxValue - 1);

            // np.c_[9007199254740993:9007199254740995] -> (2, 1); np.r_[0:3, 9007199254740993:9007199254740994]
            var d = np.c_["9007199254740993:9007199254740995"];
            d.Should().BeOfType(NPTypeCode.Int64).And.BeShaped(2, 1);
            d.GetInt64(1, 0).Should().Be(9007199254740994L);
            var e = np.r_["0:3, 9007199254740993:9007199254740994"];
            e.Should().BeShaped(4);
            e.GetInt64(3).Should().Be(9007199254740993L);
        }

        /// <summary>
        ///     <c>-0</c> is the Python int 0, so where it meets a float field it converts to +0.0 (arange stores the int
        ///     start as 0.0); <c>-0.0</c> is a float and keeps its sign. The int field was read as the double -0.0.
        /// </summary>
        [TestMethod]
        public void NegativeZeroIntField_IsTheInt0()
        {
            // np.r_[-0:3.0] -> [0., 1., 2.] (signbit False); np.r_[-0.0:3.0] -> [-0., 1., 2.]; np.mgrid[-0:3.0] -> [0., 1., 2.]
            double first = np.r_["-0:3.0"].GetDouble(0);
            BitConverter.DoubleToInt64Bits(first).Should().Be(0L);
            BitConverter.DoubleToInt64Bits(np.r_["-0.0:3.0"].GetDouble(0)).Should().Be(BitConverter.DoubleToInt64Bits(-0.0));
            NDArray m = np.mgrid["-0:3.0"];
            BitConverter.DoubleToInt64Bits(m.GetDouble(0)).Should().Be(0L);
        }

        /// <summary>
        ///     <c>Infinity</c> and <c>NaN</c> are Python floats, not ints: np.r_[0:1:inf] is float64. They were read as
        ///     int literals (no point, no exponent), which made that slice int64.
        /// </summary>
        [TestMethod]
        public void InfinityAndNaNFields_AreFloats()
        {
            // np.r_[0:1:float('inf')] -> array([0.]) (float64: arange's quotient underflows to 0.0, one element)
            var a = np.r_["0:1:Infinity"];
            a.Should().BeOfType(NPTypeCode.Double).And.BeShaped(1).And.BeOfValues(0.0);
            // np.r_[0:float('inf')] -> ValueError: Maximum allowed size exceeded
            new Action(() => { var _ = np.r_["0:Infinity"]; }).Should().Throw<ValueError>()
                .WithMessage("Maximum allowed size exceeded");
            // np.r_[0:float('nan')] / np.r_[0:1:float('nan')] -> ValueError: arange: cannot compute length
            new Action(() => { var _ = np.r_["0:NaN"]; }).Should().Throw<ValueError>()
                .WithMessage("arange: cannot compute length");
            new Action(() => { var _ = np.r_["0:1:NaN"]; }).Should().Throw<ValueError>()
                .WithMessage("arange: cannot compute length");
        }
    }
}
