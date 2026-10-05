using System.Collections.Generic;
using static NumSharp.Tests.Utilities.RawBitCases;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs where a ufunc's <c>out=</c> route computes other bits than the plain call: NumPy runs the same inner
    ///     loop either way (the out array only replaces the allocation), so <c>f(x, out=o)</c> and <c>f(x)</c> always agree
    ///     there. Each test asserts NumPy 2.4.2's answer, dtype, shape and the raw bits of every element, so it FAILS while
    ///     the bug lives and passes once it is fixed. Remove <c>[OpenBugs]</c> from a test when its path is fixed and move
    ///     it next to the function's own tests.
    /// </summary>
    /// <remarks>
    ///     Every expected string is NumPy 2.4.2's output for the same call, probed on 2026-10-06, in
    ///     <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/>'s format: <c>dtype (shape) bits…</c>.
    /// </remarks>
    [TestClass]
    public class OpenBugsUfuncOutTests : TestClass
    {
        /// <summary>
        ///     <c>np.power(x, 2.0, out=o)</c> must be <c>x * x</c>, as the plain call is: NumPy's DOUBLE_power / FLOAT_power
        ///     take a fast path when the exponent's stride is 0 (-1 is <c>1/x</c>, 0 is 1, 0.5 is <c>sqrt</c>, 1 is <c>x</c>,
        ///     2 is <c>x*x</c>) on every route. NumSharp's plain call takes it too, its <c>out=</c> route calls the C
        ///     runtime's <c>pow</c>, which is one ULP off on about 0.06 % of float64 inputs (116 of 200,000 uniform values in
        ///     [-1000, 1000]) and 0.01 % of float32 ones: <c>757.7617475757002 ** 2</c> is <c>0x412185f5bb7003c1</c> in
        ///     NumPy, <c>...03c2</c> through NumSharp's out=.
        /// </summary>
        /// <remarks>
        ///     Root cause: the ufunc out= route (<c>DefaultEngine.UfuncOut.cs</c>) runs np.power's general inner loop, which
        ///     has no scalar-exponent branch; the fast paths live only on the allocating route. Plan
        ///     <c>docs/plans/api-parameter-parity.md</c> X13, found by the A6 work (2026-10-06): numpy.ma's
        ///     <c>abs(danom) ** 2</c> port first wrote power's result into its layout buffer with out= and drifted on 73
        ///     probe cases; it now calls np.power plainly and copies.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Power_Out_ScalarExponent_TakesNumPysFastPath()
        {
            var fails = new List<string>();
            var x = np.array(new[] { 757.7617475757002, -205.48057283278354 });
            var o = np.empty_like(x);
            Check(fails, "power f64 out= 2.0", () => { np.power(x, NDArray.Scalar(2.0), @out: o); return o; },
                  "float64 (2,) 412185f5bb7003c1 40e49dc8818783d1");
            var xf = np.array(new[] { -901.92664f });
            var of = np.empty_like(xf);
            Check(fails, "power f32 out= 2.0", () => { np.power(xf, NDArray.Scalar(2.0f), @out: of); return of; },
                  "float32 (1,) 494699fb");
            AssertNoMismatch(fails, "NumPy's power takes its scalar-exponent fast path on the out= route too");
        }
    }
}
