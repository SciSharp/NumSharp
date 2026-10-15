using System;
using System.Collections.Generic;
using System.Numerics;
using static NumSharp.Tests.Utilities.RawBitCases;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     <c>np.nanmean</c> as NumPy's <c>nanmean</c> (<c>numpy/lib/_nanfunctions_impl.py</c>), pinned to the bit: an input
    ///     that is not inexact IS <c>np.mean</c>; an inexact one gets the dtype check, the non-NaN count
    ///     (<c>keepdims=keepdims</c>), the sum of the NaN-zeroed K-order copy in its loop, and the in-place division (the
    ///     float64 loop for a real dtype, Smith's complex loop). Every expected string is NumPy 2.4.2's output for the same
    ///     call, in <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/>'s format (<c>dtype (shape) bits…</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Before the port (plan <c>docs/plans/api-parameter-parity.md</c> item A7, 2026-10-05) np.nanmean ran its own
    ///     struct kernels: an integer or bool input dropped <c>keepdims</c> for a flat reduction, float16 / float32 were
    ///     summed in double (a float16 overflow stayed finite), a complex sum was divided part by part (an infinite part
    ///     left the other finite), and there was no <c>dtype=</c>. An 11,415-case probe found 538 cases grossly off and
    ///     207 ULP off; the oracle tier <c>nanmean.jsonl</c> (zero excuses) covers the same ground.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_nanmean_numpy_port_tests
    {
        /// <summary>
        ///     An integer or bool input is np.mean with the same keywords, so a flat reduction keeps <c>keepdims</c>:
        ///     <c>(1, 1)</c>. The old route dropped it (0-d).
        /// </summary>
        [TestMethod]
        public void IntegerInput_IsNpMean_KeepdimsKept()
        {
            var fails = new List<string>();
            var i32 = np.arange(12).astype(np.int32).reshape(3, 4) * 7 - 20;
            Check(fails, "nanmean i32 (3,4) keepdims", () => np.nanmean(i32, keepdims: true), "float64 (1,1) 4032800000000000");
            Check(fails, "nanmean i32 (3,4) axis1 keepdims", () => np.nanmean(i32, 1, true),
                  "float64 (3,1) c023000000000000 4032800000000000 4047400000000000");
            var b = np.array(new bool[,] { { true, false, true }, { true, true, false } });
            Check(fails, "nanmean bool (2,3) keepdims", () => np.nanmean(b, keepdims: true), "float64 (1,1) 3fe5555555555555");
            AssertNoMismatch(fails, "NumPy's nanmean of an integer input is np.mean, keepdims included");
        }

        /// <summary>
        ///     A complex mean divides by the non-NaN count with NumPy's complex division, so an infinite part turns the other
        ///     part into NaN; an element with a NaN part is skipped.
        /// </summary>
        [TestMethod]
        public void Complex_DividesLikeNumPysComplexLoop()
        {
            var fails = new List<string>();
            var c = np.array(new Complex[,] { { new Complex(double.PositiveInfinity, 2), new Complex(double.NaN, 0) },
                                              { new Complex(1, 1), new Complex(3, -1) } });
            Check(fails, "nanmean c128 inf axis0", () => np.nanmean(c, 0),
                  "complex128 (2,) 7ff0000000000000 fff8000000000000 4008000000000000 bff0000000000000");
            AssertNoMismatch(fails, "NumPy's nanmean divides in its complex loop");
        }

        /// <summary>
        ///     float16 sums in float16 (it overflows to inf) and float32 in float32 (NumPy's pairwise order); <c>dtype=</c>
        ///     moves the sum into the requested loop.
        /// </summary>
        [TestMethod]
        public void FloatInputs_SumInTheirLoop()
        {
            var fails = new List<string>();
            var h = np.array(new[] { (Half)60000f, (Half)30000f, Half.NaN, (Half)20000f });
            Check(fails, "nanmean f16 overflow", () => np.nanmean(h), "float16 () 7c00");
            var f = (np.arange(1031) % 37 * 0.173 + 1000.0).astype(np.float32);
            f["::7"] = NDArray.Scalar(float.NaN);
            Check(fails, "nanmean f32 1031", () => np.nanmean(f), "float32 () 447ac64b");
            Check(fails, "nanmean f32 1031 dtype=f64", () => np.nanmean(f, dtype: np.float64), "float64 () 408f58c95f74d68b");
            AssertNoMismatch(fails, "NumPy's nanmean sums an inexact input in its own loop");
        }

        /// <summary>
        ///     A slice with no non-NaN element is <c>0 / 0</c>: NaN; an empty reduction too.
        /// </summary>
        [TestMethod]
        public void AllNaNOrEmptySlice_IsNaN()
        {
            var fails = new List<string>();
            var x = np.array(new double[,] { { 1.0, 2.0 }, { double.NaN, double.NaN } });
            Check(fails, "nanmean f64 all-NaN row axis1", () => np.nanmean(x, 1), "float64 (2,) 3ff8000000000000 nan");
            Check(fails, "nanmean f64 (0,3) axis0", () => np.nanmean(np.zeros(new Shape(0, 3)), 0), "float64 (3,) nan nan nan");
            AssertNoMismatch(fails, "NumPy's nanmean keeps 0/0 for a slice without a non-NaN element");
        }

        /// <summary>
        ///     The keyword and axis rules in NumPy's order: an integer <c>dtype=</c> for an inexact input is TypeError; a 0-d
        ///     inexact input reduces flat for axis 0 / -1 (nanmean reads the axis through np.sum) and refuses axis 1; a 0-d
        ///     integer input is np.mean and refuses axis 0.
        /// </summary>
        [TestMethod]
        public void Errors_AreNumPys()
        {
            var fails = new List<string>();
            CheckThrows<TypeError>(fails, "nanmean f64 dtype=i32", () => np.nanmean(np.array(new[] { 1.0, 2.0 }), dtype: np.int32),
                                   "If a is inexact, then dtype must be inexact");
            Check(fails, "nanmean f64 0-d axis0", () => np.nanmean(NDArray.Scalar(3.5), 0), "float64 () 400c000000000000");
            CheckThrows<AxisError>(fails, "nanmean f64 0-d axis1", () => np.nanmean(NDArray.Scalar(3.5), 1),
                                   "axis 1 is out of bounds for array of dimension 0");
            CheckThrows<AxisError>(fails, "nanmean i32 0-d axis0", () => np.nanmean(NDArray.Scalar(3), 0),
                                   "axis 0 is out of bounds for array of dimension 0");
            AssertNoMismatch(fails, "NumPy's nanmean checks the dtype, then reads the axis through np.sum (np.mean for integers)");
        }
    }
}
