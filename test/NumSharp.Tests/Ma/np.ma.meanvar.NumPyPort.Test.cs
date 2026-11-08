using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using static NumSharp.Tests.Utilities.RawBitCases;

namespace NumSharp.Tests.Ma
{
    /// <summary>
    ///     <c>np.ma.mean</c> / <c>np.ma.var</c> / <c>np.ma.std</c> / <c>np.ma.anom</c> as numpy.ma's own statements
    ///     (<c>numpy/ma/core.py</c>), pinned to the bit. Without a mask, ndarray's <c>_mean</c> / <c>_var</c> run on the masked
    ///     array, so their <c>true_divide(out=...)</c> steps meet <c>MaskedArray.__array_wrap__</c>'s divide domain (a quotient
    ///     with <c>|sum| * tiny &gt;= count</c> becomes 1.0, masked). With a mask, the mean is <c>sum * 1. / count</c>, the
    ///     variance squares <c>self - mean</c> in place (complex: <c>abs(danom) ** 2</c>), divides by <c>count - ddof</c> and
    ///     REPLACES the result's mask with <c>mask_or(mask.all(axis), cnt &lt;= 0)</c>. Every expected string is NumPy 2.4.2's
    ///     output for the same call in <see cref="MaDescribe"/>'s format: <c>masked</c> for the singleton, else the form
    ///     (<c>scalar</c> / <c>nomask</c> / <c>mask</c>), the data in <see cref="Describe"/>'s format at EVERY slot, and the
    ///     mask bits.
    /// </summary>
    /// <remarks>
    ///     Before the port (plan <c>docs/plans/api-parameter-parity.md</c> item A6, 2026-10-05) these four ran a masked
    ///     reduction of their own: the mean divided in its own order and never met the divide domain, a slice with an
    ///     unmasked inf kept inf where NumPy masks it into 0.0, the mask of the variance was the divide's, and anom
    ///     inherited all of it. A 1.9-million-call probe and the zero-excuse oracle tier <c>ma_meanvar.jsonl</c> (26,379
    ///     cases) cover the ground these tests pin by example.
    /// </remarks>
    [TestClass]
    public class np_ma_meanvar_numpy_port_tests
    {
        /// <summary>
        ///     Without a mask the mean is <c>_mean</c> on the masked array: its <c>true_divide(ret, count, out=ret)</c> goes
        ///     through <c>__array_wrap__</c>, which masks a quotient in the divide domain and writes 1.0 there; a 0-d result in
        ///     the domain is the <c>masked</c> singleton. A float32 sum that overflows is such a quotient.
        /// </summary>
        [TestMethod]
        public void Mean_WithoutMask_DividesThroughArrayWrapsDomain()
        {
            var fails = new List<string>();
            CheckMa(fails, "mean [inf,1]", () => np.ma.mean(Ma(new[] { double.PositiveInfinity, 1.0 })), "masked");
            CheckMa(fails, "mean [[1,2],[inf,3]] axis1",
                    () => np.ma.mean(Ma(new[,] { { 1.0, 2.0 }, { double.PositiveInfinity, 3.0 } }), 1),
                    "mask float64 (2,) 3ff8000000000000 3ff0000000000000 | 01");
            CheckMa(fails, "mean f32 overflow", () => np.ma.mean(Ma(new[] { 1e38f, 3e38f, 3e38f })), "masked");
            CheckMa(fails, "mean i32 (2,2) axis0", () => np.ma.mean(Ma(np.array(new[] { 1, 2, 3, 4 }).reshape(2, 2)), 0),
                    "nomask float64 (2,) 4000000000000000 4008000000000000");
            AssertNoMismatch(fails, "NumPy's nomask mean divides through MaskedArray.__array_wrap__'s domain");
        }

        /// <summary>
        ///     With a mask the mean is <c>dsum * 1. / cnt</c>: a float32 sum turns float64 (the strong <c>1.</c>), a float16
        ///     one sums in float32 and returns float16 (<c>np.float16(...)</c>), no unmasked element gives <c>masked</c>, and an
        ///     all-False mask stays a mask array (form <c>mask</c>, not nomask). The <c>* 1.</c> is a complex multiply by
        ///     <c>1+0j</c> for a complex sum, so an infinite imaginary part makes the real part NaN before the divide masks
        ///     the slot (the masked slot keeps NaN + NaNj, where a sum divided directly would keep NaN + inf j).
        /// </summary>
        [TestMethod]
        public void Mean_WithMask_IsSumTimesOneOverCount()
        {
            var fails = new List<string>();
            CheckMa(fails, "mean f32 masked", () => np.ma.mean(Ma(new[] { 1f, 2f, 3f, 4f }, new[] { false, false, true, false })),
                    "scalar float64 () 4002aaaaaaaaaaab");
            CheckMa(fails, "mean f16 masked", () => np.ma.mean(Ma(new[] { (Half)1, (Half)2, (Half)4 }, new[] { false, true, false })),
                    "scalar float16 () 4100");
            CheckMa(fails, "mean f16 (2,2) axis1",
                    () => np.ma.mean(Ma(np.array(new[] { (Half)1, (Half)2, (Half)3, (Half)4 }).reshape(2, 2),
                                        np.array(new[] { false, true, true, true }).reshape(2, 2)), 1),
                    "mask float16 (2,) 3c00 0000 | 01");
            CheckMa(fails, "mean all masked", () => np.ma.mean(Ma(new[] { 1.0, 2.0 }, new[] { true, true })), "masked");
            CheckMa(fails, "mean all-False axis0",
                    () => np.ma.mean(Ma(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }, new[,] { { false, false }, { false, false } }), 0),
                    "mask float64 (2,) 4000000000000000 4008000000000000 | 00");
            CheckMa(fails, "mean c (2,2) inf imag axis1",
                    () => np.ma.mean(Ma(np.array(new[] { new Complex(1, double.PositiveInfinity), new Complex(2, 0), new Complex(3, 0), new Complex(4, -1) }).reshape(2, 2),
                                        np.array(new[] { false, false, false, true }).reshape(2, 2)), 1),
                    "mask complex128 (2,) nan nan 4008000000000000 0000000000000000 | 10");
            AssertNoMismatch(fails, "NumPy's masked mean is sum * 1. / count");
        }

        /// <summary>
        ///     A masked variance REPLACES its mask with <c>mask_or(mask.all(axis), cnt &lt;= 0)</c>: a fully masked row is
        ///     masked, a row whose count does not exceed <c>ddof</c> is masked, and a row with an unmasked inf is 0.0, unmasked
        ///     (its mean is masked by ma's divide, so every anomaly of the row is masked and sums to 0).
        /// </summary>
        [TestMethod]
        public void Var_WithMask_ReplacesTheResultMask()
        {
            var fails = new List<string>();
            var x = Ma(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, double.PositiveInfinity, 9.0 } },
                       new[,] { { true, true, true }, { false, true, true }, { false, false, false } });
            CheckMa(fails, "var axis1", () => np.ma.var(x, 1),
                    "mask float64 (3,) 0000000000000000 0000000000000000 0000000000000000 | 100");
            CheckMa(fails, "var axis1 ddof1", () => np.ma.var(x, 1, ddof: 1),
                    "mask float64 (3,) 0000000000000000 0000000000000000 0000000000000000 | 110");
            CheckMa(fails, "std axis1 ddof1", () => np.ma.std(x, 1, ddof: 1),
                    "mask float64 (3,) 0000000000000000 0000000000000000 0000000000000000 | 110");
            AssertNoMismatch(fails, "NumPy's masked var replaces the divide's mask");
        }

        /// <summary>
        ///     A flat masked variance with <c>count - ddof &lt; 0</c> is a plain scalar, not <c>masked</c>: the divide by a
        ///     negative count is outside ma's divide domain, so <c>0 / -1</c> is <c>-0.0</c> and <c>0.5 / -1</c> is
        ///     <c>-0.5</c>. Its square root is in sqrt's domain only for the zero; std of the negative one is <c>masked</c>.
        /// </summary>
        [TestMethod]
        public void Var_WithMask_CountBelowDdof_IsAPlainScalar()
        {
            var fails = new List<string>();
            CheckMa(fails, "var ddof2", () => np.ma.var(Ma(new[] { 1.0, 2.0, 3.0 }, new[] { false, true, true }), ddof: 2),
                    "scalar float64 () 8000000000000000");
            CheckMa(fails, "var ddof3", () => np.ma.var(Ma(new[] { 1.0, 2.0, 4.0 }, new[] { false, false, true }), ddof: 3),
                    "scalar float64 () bfe0000000000000");
            CheckMa(fails, "std ddof3", () => np.ma.std(Ma(new[] { 1.0, 2.0, 4.0 }, new[] { false, false, true }), ddof: 3),
                    "masked");
            AssertNoMismatch(fails, "NumPy's masked var divides by a negative count outside the domain");
        }

        /// <summary>
        ///     A complex masked variance squares through <c>abs(danom) ** 2</c>: a slice with an infinite element has a masked
        ///     mean, so its variance is <c>masked</c>; a finite one rounds like NumPy's <c>hypot</c> squared. A square past the
        ///     float64 range is MASKED by ma.power (its <c>~isfinite</c> mask), so the slice's sum skips it: a column whose
        ///     squares all overflow sums to 0.0, and the replaced result mask (<c>mask_or</c>, all-False here) shrinks to
        ///     nomask.
        /// </summary>
        [TestMethod]
        public void Var_Complex_TakesAbsSquared()
        {
            var fails = new List<string>();
            CheckMa(fails, "var c inf",
                    () => np.ma.var(Ma(new[] { new Complex(1, 2), new Complex(3, -1), new Complex(double.PositiveInfinity, 0), new Complex(2, 2) },
                                       new[] { false, false, false, true })),
                    "masked");
            CheckMa(fails, "var c finite",
                    () => np.ma.var(Ma(new[] { new Complex(1, 2), new Complex(3, -1), new Complex(-1, 0.5) }, new[] { false, false, true })),
                    "scalar float64 () 400a000000000001");
            CheckMa(fails, "var c (2,2) axis0 huge",
                    () => np.ma.var(Ma(np.array(new[] { new Complex(1, 2), new Complex(3, -1), new Complex(2, -2), new Complex(1e200, 1e200) }).reshape(2, 2),
                                       np.array(new[] { false, false, false, false }).reshape(2, 2)), 0),
                    "nomask float64 (2,) 4011000000000000 0000000000000000");
            CheckMa(fails, "var c (3,2) axis0 overflowing squares",
                    () => np.ma.var(Ma(np.array(new[] { new Complex(1e160, 1e160), new Complex(1, 0), new Complex(-1e160, 0), new Complex(2, 0),
                                                        new Complex(3, 0), new Complex(5, 1) }).reshape(3, 2),
                                       np.array(new[] { false, false, false, false, true, false }).reshape(3, 2)), 0),
                    "nomask float64 (2,) 0000000000000000 4008e38e38e38e3b");
            AssertNoMismatch(fails, "NumPy's masked complex var is abs(danom) ** 2 through ma.power");
        }

        /// <summary>
        ///     <c>mean=</c> reaches <c>_var</c> as given without a mask, where a Python float is WEAK (float32 stays float32),
        ///     but goes through <c>ma.subtract</c> with a mask, whose <c>getdata</c> makes it a STRONG float64 array (the
        ///     variance turns float64). A NumPy scalar is strong either way.
        /// </summary>
        [TestMethod]
        public void Var_MeanParam_IsWeakWithoutAMask_StrongWithOne()
        {
            var fails = new List<string>();
            var f = new[] { 1f, 2f, 4f, 8f };
            CheckMa(fails, "var f32 nomask mean=1.5", () => np.ma.var(Ma(f), mean: 1.5), "scalar float32 () 41440000");
            CheckMa(fails, "var f32 masked mean=1.5", () => np.ma.var(Ma(f, new[] { false, false, false, true }), mean: 1.5),
                    "scalar float64 () 4002000000000000");
            CheckMa(fails, "var f32 nomask mean=np.float32", () => np.ma.var(Ma(f), mean: NDArray.Scalar(1.5f)),
                    "scalar float32 () 41440000");
            CheckMa(fails, "var f32 all-False mean=np.float64",
                    () => np.ma.var(Ma(f, new[] { false, false, false, false }), mean: NDArray.Scalar(1.5)),
                    "scalar float64 () 4028800000000000");
            AssertNoMismatch(fails, "NumPy's var takes mean= weak without a mask and strong through ma.subtract");
        }

        /// <summary>
        ///     Without a mask, <c>_var</c> squares the deviations IN PLACE (<c>multiply(x, x, out=x)</c>), and x is the
        ///     subtract's fresh result in NumPy's ufunc layout, so a permuted input's squares are summed in the input's
        ///     memory order. A C-ordered square buffer adds the same 24 values in another order: one ULP higher here
        ///     (<c>...1c71d</c>).
        /// </summary>
        [TestMethod]
        public void Var_WithoutMask_SquaresInPlace_InNumPysLayout()
        {
            var fails = new List<string>();
            var b = np.array(new[] { 21, -25, 49, -6, -3, 0, 8, 5, 0, 49, 30, 29, 20, 12, -16, 48, -4, -29, 34, -34, 35, 11, -39, -46 })
                      .reshape(2, 3, 4);
            var x = np.transpose(b, new[] { 2, 0, 1 });
            CheckMa(fails, "var i32 transposed 3-D", () => np.ma.var(Ma(x)), "scalar float64 () 408788a71c71c71c");
            AssertNoMismatch(fails, "NumPy sums the in-place squares in the subtract's memory order");
        }

        /// <summary>
        ///     <c>anom</c> is <c>self - mean</c> through ma.subtract: a float32 mean in the divide domain (an inf without a
        ///     mask) is <c>masked</c>, so every anomaly is masked and float64, with the minuend's data put back at the
        ///     masked slots; with an all-False mask the mean is a plain float64 inf (scalar arithmetic), so the anomalies are
        ///     <c>-inf</c> / NaN under the all-False mask; a fully masked slice's mean is masked along an axis too. The mean
        ///     is expanded only for a nonzero axis (<c>if not axis: return self - m</c>): a 0-d operand with a mask takes
        ///     axis 0 through ma's reduction (the result stays 0-d), and without a mask np.mean's axis check refuses it.
        /// </summary>
        [TestMethod]
        public void Anom_SubtractsTheMaskedMean()
        {
            var fails = new List<string>();
            var f = new[] { 1f, float.PositiveInfinity, 2f };
            CheckMa(fails, "anom f32 nomask inf", () => np.ma.anom(Ma(f)),
                    "mask float64 (3,) 3ff0000000000000 7ff0000000000000 4000000000000000 | 111");
            CheckMa(fails, "anom f32 all-False inf", () => np.ma.anom(Ma(f, new[] { false, false, false })),
                    "mask float64 (3,) fff0000000000000 nan fff0000000000000 | 000");
            CheckMa(fails, "anom all masked axis-1", () => np.ma.anom(Ma(new[] { 1.0, 2.0 }, new[] { true, true }), -1),
                    "mask float64 (2,) 3ff0000000000000 4000000000000000 | 11");
            CheckMa(fails, "anom (2,3) axis1",
                    () => np.ma.anom(Ma(new[,] { { 1.0, 2.0, 4.0 }, { 3.0, 5.0, 9.0 } }, new[,] { { false, true, false }, { false, false, false } }), 1),
                    "mask float64 (2,3) bff8000000000000 4000000000000000 3ff8000000000000 c005555555555556 bfe5555555555558 400aaaaaaaaaaaaa | 010000");
            CheckMa(fails, "anom 0-d mask False axis0", () => np.ma.anom(Ma(NDArray.Scalar(5.0), NDArray.Scalar(false)), 0),
                    "scalar float64 () 0000000000000000");
            CheckMa(fails, "anom 0-d mask True axis0", () => np.ma.anom(Ma(NDArray.Scalar(5.0), NDArray.Scalar(true)), 0), "masked");
            CheckMaThrows<AxisError>(fails, "anom 0-d nomask axis0", () => np.ma.anom(Ma(NDArray.Scalar(5.0)), 0),
                                     "axis 0 is out of bounds for array of dimension 0");
            AssertNoMismatch(fails, "NumPy's anom is self - mean through ma.subtract");
        }

        /// <summary>
        ///     NumPy's errors, verbatim: a 0-d operand whose mean is in the divide domain makes <c>_var</c> write into the
        ///     read-only <c>masked</c>; a complex deviation that carries a mask cannot take <c>x.view((f8, (2,)))</c>; a
        ///     <c>mean=</c> that does not broadcast fails the subtract with the ufunc's text; an axis past the rank is
        ///     AxisError. <c>std</c> ignores <c>mean=</c> (NumPy's <c>ma.std</c> forwards only <c>keepdims</c>).
        /// </summary>
        [TestMethod]
        public void Errors_AndStdIgnoringMean()
        {
            var fails = new List<string>();
            CheckMaThrows<ValueError>(fails, "var 0-d inf", () => np.ma.var(Ma(NDArray.Scalar(double.PositiveInfinity))),
                                      "output array is read-only");
            CheckMaThrows<ValueError>(fails, "var complex masked mean",
                                      () => np.ma.var(Ma(new[] { new Complex(1, 1), new Complex(2, 0) }),
                                                      mean: Ma(new[] { new Complex(1, 0) }, new[] { true })),
                                      "Changing the dtype to a subarray type is only supported if the total itemsize is unchanged");
            CheckMaThrows<IncorrectShapeException>(fails, "var mean (2,)",
                                                   () => np.ma.var(Ma(np.ones(new Shape(2, 3))), 1, mean: np.zeros(new Shape(2))),
                                                   "operands could not be broadcast together with shapes (2,3) (2,) ");
            CheckMaThrows<AxisError>(fails, "mean axis 2", () => np.ma.mean(Ma(np.ones(new Shape(2, 3))), 2),
                                     "axis 2 is out of bounds for array of dimension 2");
            CheckMa(fails, "std mean=10 ignored", () => np.ma.std(Ma(new[] { 1.0, 2.0, 4.0 }), mean: 10.0),
                    "scalar float64 () 3ff3f49c0b9ad4db");
            AssertNoMismatch(fails, "NumPy's ma reductions raise these errors and std ignores mean=");
        }

        // ---- helpers -------------------------------------------------------------------------------------------------

        /// <summary>A masked array over <paramref name="data"/> with no mask (nomask) or the given one (kept as an array
        /// even when all-False, as <c>np.ma.array(x, mask=...)</c> keeps it).</summary>
        /// <param name="data">The data: an NDArray, or a C# array (any rank) that <c>np.array</c> converts.</param>
        /// <param name="mask">The mask, or null for nomask.</param>
        /// <returns>The masked array.</returns>
        private static NDMaskedArray Ma(object data, object mask = null)
        {
            var d = data as NDArray ?? np.array((Array)data);
            var m = mask is null ? null : mask as NDArray ?? np.array((Array)mask);
            return np.ma.masked_array(d, m);
        }

        /// <summary>
        ///     Renders a masked result: <c>masked</c> for the singleton; else <c>scalar</c> (0-d without a mask),
        ///     <c>nomask</c> (an array without a mask) or <c>mask</c>, the data at every slot in <see cref="Describe"/>'s
        ///     format, and for a mask array <c> | </c> and one 0/1 digit per slot. NumPy's side prints the same format.
        /// </summary>
        /// <remarks>
        ///     A complex part that is NaN renders as <c>nan</c> (Describe prints complex parts raw): which NaN a complex
        ///     multiply or divide leaves is outside NumPy's contract (the oracle tokenizes it too), but a NaN where NumPy has a
        ///     number, or a number where it has NaN, still shows.
        /// </remarks>
        /// <param name="r">The result.</param>
        /// <returns>The rendering.</returns>
        private static string MaDescribe(NDMaskedArray r)
        {
            if (ReferenceEquals(r, np.ma.NDMasked))
                return "masked";
            var d = r.data;
            string data = d.typecode == NPTypeCode.Complex ? DescribeComplexNaNTokenized(d) : Describe(d);
            if (r._mask is null)
                return (d.ndim == 0 ? "scalar " : "nomask ") + data;
            var m = r._mask.ndim == 0 ? r._mask.reshape(1) : r._mask.flatten();
            var bits = new StringBuilder();
            for (long i = 0; i < m.size; i++)
                bits.Append(m.GetAtIndex<bool>(i) ? '1' : '0');
            return "mask " + data + " | " + bits;
        }

        /// <summary><see cref="Describe"/>'s rendering of a complex128 array with every NaN part as the token <c>nan</c>
        /// (the real then the imaginary word per element, as Describe prints them).</summary>
        /// <param name="d">A complex128 array.</param>
        /// <returns>The rendering.</returns>
        private static string DescribeComplexNaNTokenized(NDArray d)
        {
            string dims = string.Join(",", d.shape);
            string shape = d.ndim == 1 ? $"({dims},)" : $"({dims})";
            var flat = d.ndim == 0 ? d.reshape(1) : d.flatten();
            var parts = new List<string>();
            for (long i = 0; i < flat.size; i++)
            {
                var c = flat.GetAtIndex<Complex>(i);
                parts.Add(double.IsNaN(c.Real) ? "nan" : BitConverter.DoubleToUInt64Bits(c.Real).ToString("x16"));
                parts.Add(double.IsNaN(c.Imaginary) ? "nan" : BitConverter.DoubleToUInt64Bits(c.Imaginary).ToString("x16"));
            }
            return $"{d.dtype.name} {shape} {string.Join(" ", parts)}";
        }

        /// <summary>Records a line when a masked call's <see cref="MaDescribe"/> rendering differs from NumPy's (a throw
        /// is recorded too).</summary>
        /// <param name="fails">Collects the mismatches.</param>
        /// <param name="name">The case label.</param>
        /// <param name="call">The NumSharp call.</param>
        /// <param name="expected">NumPy 2.4.2's rendering.</param>
        private static void CheckMa(List<string> fails, string name, Func<NDMaskedArray> call, string expected)
        {
            string actual;
            try
            {
                actual = MaDescribe(call());
            }
            catch (Exception e)
            {
                actual = $"threw {e.GetType().Name}: {e.Message}";
            }
            if (actual != expected)
                fails.Add($"{name}: expected [{expected}] but NumSharp returned [{actual}]");
        }

        /// <summary>Records a line unless the masked call throws <typeparamref name="TException"/> whose message starts
        /// with NumPy's text (a prefix: .NET appends <c>(Parameter '...')</c> to an argument exception).</summary>
        /// <typeparam name="TException">The exception type NumPy's error maps to.</typeparam>
        /// <param name="fails">Collects the mismatches.</param>
        /// <param name="name">The case label.</param>
        /// <param name="call">The NumSharp call.</param>
        /// <param name="numpyMessage">NumPy's message, verbatim.</param>
        private static void CheckMaThrows<TException>(List<string> fails, string name, Func<NDMaskedArray> call, string numpyMessage)
            where TException : Exception
        {
            try
            {
                var r = call();
                fails.Add($"{name}: expected {typeof(TException).Name} [{numpyMessage}] but NumSharp returned [{MaDescribe(r)}]");
            }
            catch (TException e) when (e.Message.StartsWith(numpyMessage, StringComparison.Ordinal))
            {
                // NumPy's exception type with NumPy's text.
            }
            catch (Exception e)
            {
                fails.Add($"{name}: expected {typeof(TException).Name} [{numpyMessage}] but NumSharp threw {e.GetType().Name} [{e.Message}]");
            }
        }
    }
}
