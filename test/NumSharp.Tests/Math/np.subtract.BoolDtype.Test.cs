using System;

namespace NumSharp.Tests.Math;

/// <summary>
///     <c>np.subtract</c> of two bool operands: NumPy's <c>PyUFunc_SubtractionTypeResolver</c> refuses the BOOL loop only,
///     after the loop is resolved, so a <c>dtype=</c> that names another loop subtracts the bools in it (np.linspace
///     subtracts bool ends with <c>dtype=float64</c>). Expectations are NumPy 2.4.2's output.
/// </summary>
[TestClass]
public class np_subtract_BoolDtype_Test
{
    /// <summary>The refusal NumPy raises for the bool loop.</summary>
    private const string BoolSubtract = "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, " +
                                        "the `^` operator, or the logical_xor function instead.";

    /// <summary>
    ///     <c>np.subtract([True, False, True], [False, True, True], dtype=float64)</c> is <c>[1., -1., 0.]</c> and with
    ///     <c>dtype=int8</c> <c>[1, -1, 0]</c>.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void BoolOperands_SubtractInTheRequestedLoop()
    {
        using var a = np.array(new[] { true, false, true });
        using var b = np.array(new[] { false, true, true });

        using (var asFloat = np.subtract(a, b, dtype: np.float64))
        {
            asFloat.typecode.Should().Be(NPTypeCode.Double);
            asFloat.ToArray<double>().Should().Equal(1.0, -1.0, 0.0);
        }
        using (var asInt8 = np.subtract(a, b, dtype: np.int8))
        {
            asInt8.typecode.Should().Be(NPTypeCode.SByte);
            asInt8.ToArray<sbyte>().Should().Equal((sbyte)1, (sbyte)-1, (sbyte)0);
        }
    }

    /// <summary>
    ///     The bool loop is still refused: no <c>dtype</c> (an <c>out=</c> float64 array does not choose the loop), or
    ///     <c>dtype=bool</c>.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void BoolLoop_IsRefused()
    {
        using var a = np.array(new[] { true, false, true });
        using var b = np.array(new[] { false, true, true });
        using var output = np.empty(new Shape(3), np.float64);

        Action plain = () => np.subtract(a, b);
        plain.Should().ThrowExactly<TypeError>().WithMessage(BoolSubtract);
        Action intoFloat = () => np.subtract(a, b, @out: output);
        intoFloat.Should().ThrowExactly<TypeError>().WithMessage(BoolSubtract);
        Action boolLoop = () => np.subtract(a, b, dtype: np.@bool);
        boolLoop.Should().ThrowExactly<TypeError>().WithMessage(BoolSubtract);
    }
}
