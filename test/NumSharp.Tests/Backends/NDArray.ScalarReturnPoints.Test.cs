using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     NumPy's return points (step S2 of the scalar plan): which calls hand a 0-d result back as NumPy's SCALAR (read-only,
    ///     no explicit array) and which as a 0-d ndarray. Every expectation is NumPy 2.4.2's, probed live: each of the 98
    ///     ufuncs NumSharp has returns a scalar for 0-d operands (a tuple of scalars for np.modf / np.frexp / np.divmod), as
    ///     do clip, round, dot / vdot / inner, sinc, nan_to_num, interp and the reductions; np.where, np.copy, astype of an
    ///     array, the views (np.squeeze, .real, .T, view, reshape) and .imag of a real array return 0-d ndarrays.
    /// </summary>
    [TestClass]
    public class NDArrayScalarReturnPointsTests
    {
        /// <summary>The kind of one result, in NumPy's terms.</summary>
        /// <param name="r">The result.</param>
        /// <returns>"scalar", "ndarray0" (a 0-d ndarray) or "ndarray" (with dimensions).</returns>
        private static string Kind(object r) => r is NDArray nd
            ? nd.ndim != 0 ? "ndarray" : nd.IsNumPyScalar && !nd.flags.writeable ? "scalar" : "ndarray0"
            : "clr:" + r?.GetType().Name;

        /// <summary>
        ///     Every ufunc NumSharp exposes under NumPy's name, called through reflection with 0-d operands (integer ones for
        ///     the integer-only ufuncs), returns NumPy's scalar - the PyArray_Return at the ufunc boundary.
        /// </summary>
        [TestMethod]
        public void EveryUfunc_ZeroDimOperands_ReturnsTheScalar()
        {
            string[] specs =
            {
                "abs:1", "absolute:1", "add:2", "arccos:1", "arccosh:1", "arcsin:1", "arcsinh:1", "arctan:1", "arctan2:2",
                "arctanh:1", "asinh:1", "atanh:1", "bitwise_and:2", "bitwise_count:1", "bitwise_not:1", "bitwise_or:2",
                "bitwise_xor:2", "cbrt:1", "ceil:1", "conj:1", "conjugate:1", "copysign:2", "cos:1", "cosh:1", "deg2rad:1",
                "degrees:1", "divide:2", "divmod:2", "equal:2", "exp:1", "exp2:1", "expm1:1", "fabs:1", "float_power:2",
                "floor:1", "floor_divide:2", "fmax:2", "fmin:2", "fmod:2", "frexp:1", "gcd:2", "greater:2",
                "greater_equal:2", "heaviside:2", "hypot:2", "invert:1", "isfinite:1", "isinf:1", "isnan:1", "lcm:2",
                "ldexp:2", "left_shift:2", "less:2", "less_equal:2", "log:1", "log10:1", "log1p:1", "log2:1", "logaddexp:2",
                "logaddexp2:2", "logical_and:2", "logical_not:1", "logical_or:2", "logical_xor:2", "maximum:2", "minimum:2",
                "mod:2", "modf:1", "multiply:2", "negative:1", "nextafter:2", "not_equal:2", "positive:1", "power:2",
                "rad2deg:1", "radians:1", "reciprocal:1", "remainder:2", "right_shift:2", "rint:1", "sign:1", "signbit:1",
                "sin:1", "sinh:1", "spacing:1", "sqrt:1", "square:1", "subtract:2", "tan:1", "tanh:1", "true_divide:2",
                "trunc:1",
            };
            var integerOnly = new HashSet<string> { "bitwise_and", "bitwise_count", "bitwise_not", "bitwise_or", "bitwise_xor",
                "gcd", "lcm", "invert", "left_shift", "right_shift" };
            var failures = new List<string>();
            foreach (var spec in specs)
            {
                string name = spec.Split(':')[0];
                int nin = spec[^1] - '0';
                NDArray x = integerOnly.Contains(name) ? np.array(6L) : np.array(0.5);
                NDArray y = integerOnly.Contains(name) || name == "ldexp" ? np.array(3L) : np.array(0.25);
                var method = typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => m.Name == name && m.GetParameters().Length >= nin
                                && m.GetParameters().Take(nin).All(p => p.ParameterType == typeof(NDArray))
                                && m.GetParameters().Skip(nin).All(p => p.IsOptional))
                    .OrderBy(m => m.GetParameters().Length).FirstOrDefault();
                if (method is null)
                {
                    failures.Add($"{name}: no np.{name} overload over NDArray operands");
                    continue;
                }
                var argv = method.GetParameters().Select((p, i) => i == 0 ? x : i == 1 && nin == 2 ? y : p.DefaultValue).ToArray();
                object r = method.Invoke(null, argv);
                string kind = r is ITuple t ? string.Join(",", Enumerable.Range(0, t.Length).Select(i => Kind(t[i]))) : Kind(r);
                if (kind.Split(',').Any(k => k != "scalar"))
                    failures.Add($"np.{name}: {kind}");
            }
            failures.Should().BeEmpty("NumPy returns a scalar from every ufunc whose operands are all 0-d");
        }

        /// <summary>
        ///     The operators are the ufuncs: two 0-d operands (or a 0-d operand and a C# literal) give the scalar, and a
        ///     result written into <c>out=</c> comes back as that writeable array.
        /// </summary>
        [TestMethod]
        public void Operators_AndOut_FollowTheUfunc()
        {
            var a = np.array(2.5);
            var b = np.array(1.5);
            Kind(a + b).Should().Be("scalar");
            Kind(a * 2).Should().Be("scalar");
            Kind(-a).Should().Be("scalar");
            Kind(+a).Should().Be("scalar");
            Kind(a < b).Should().Be("scalar");
            Kind(np.float64(2.5) + np.int64(7)).Should().Be("scalar");

            var o = np.array(0.0);
            var r = np.add(a, b, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            r.flags.writeable.Should().BeTrue("NumPy returns out= itself, writeable");
            r.GetDouble().Should().Be(4.0);
        }

        /// <summary>
        ///     The functions NumPy ends in PyArray_Return give the scalar for 0-d operands; those that build an array (where,
        ///     copy, astype of an array, the views, .imag of a real array) give a 0-d ndarray.
        /// </summary>
        [TestMethod]
        public void Functions_ScalarOrZeroDimArray_AsNumPy()
        {
            var a = np.array(2.5);
            var b = np.array(1.5);
            var t = np.array(true);
            var cases = new (string Label, Func<object> Call, string Expected)[]
            {
                ("np.clip(a, 0, 1)", () => np.clip(a, 0, 1), "scalar"),
                ("np.round(a)", () => np.round_(a), "scalar"),
                ("np.around(a, 1)", () => np.around(a, 1), "scalar"),
                ("a.round(1)", () => a.round(1), "scalar"),
                ("np.isclose(a, b)", () => np.isclose(a, b), "scalar"),
                ("np.dot(a, b)", () => np.dot(a, b), "scalar"),
                ("a.dot(a)", () => a.dot(a), "scalar"),
                ("np.vdot(a, b)", () => np.vdot(a, b), "scalar"),
                ("np.inner(a, b)", () => np.inner(a, b), "scalar"),
                ("np.matmul(ones(2), ones(2))", () => np.matmul(np.ones(2), np.ones(2)), "scalar"),
                ("np.angle(a)", () => np.angle(a), "scalar"),
                ("np.sinc(a)", () => np.sinc(a), "scalar"),
                ("np.nan_to_num(a)", () => np.nan_to_num(a), "scalar"),
                ("np.interp(a, [0, 1], [0, 1])", () => np.interp(a, np.array(new[] { 0.0, 1.0 }), np.array(new[] { 0.0, 1.0 })), "scalar"),
                ("a.sum()", () => a.sum(), "scalar"),
                ("a.std()", () => a.std(), "scalar"),
                ("np.median(a)", () => np.median(a), "scalar"),
                ("np.where(t, a, b)", () => np.where(t, a, b), "ndarray0"),
                ("a.astype(np.float32)", () => a.astype(np.float32), "ndarray0"),
                ("a.copy()", () => a.copy(), "ndarray0"),
                ("np.squeeze(a)", () => np.squeeze(a), "ndarray0"),
                ("np.real(a)", () => np.real(a), "ndarray0"),
                ("np.imag(a)", () => np.imag(a), "ndarray0"),
                ("a.imag", () => a.imag, "ndarray0"),
                ("a.T", () => a.T, "ndarray0"),
                ("a.view()", () => a.view(), "ndarray0"),
                ("a.flatten()", () => a.flatten(), "ndarray"),
                ("np.atleast_1d(a)", () => np.atleast_1d(a), "ndarray"),
            };
            var failures = cases.Where(c => Kind(c.Call()) != c.Expected)
                .Select(c => $"{c.Label}: {Kind(c.Call())}, NumPy {c.Expected}").ToList();
            failures.Should().BeEmpty();
        }

        /// <summary>
        ///     NumPy's <c>out=...</c> (<see cref="NDArray.AsArrayResult"/>) keeps a fresh 0-d ufunc result an ARRAY - writeable
        ///     and explicit - so a port can write into it; np.var of a 0-d array squares its deviation in place that way.
        /// </summary>
        [TestMethod]
        public void EllipsisOut_KeepsTheArray()
        {
            var r = np.subtract(np.array(2.0), np.array(1.0)).AsArrayResult();
            r.flags.writeable.Should().BeTrue();
            np.isscalar(r).Should().BeFalse();
            np.var(np.array(3.0), mean: np.array(1.0)).GetDouble().Should().Be(4.0);
        }

        /// <summary>
        ///     A scalar's own copy and cast stay 0-d ARRAYS in NumSharp, where NumPy's scalar methods return scalars
        ///     (<c>np.float64(2.5).copy()</c> is <c>np.float64(2.5)</c>): library code copies a 0-d result to get a writeable
        ///     buffer, so <c>copy()</c> keeps returning one. The values are NumPy's.
        /// </summary>
        [TestMethod, Misaligned]
        public void ScalarCopyAndAstype_AreArrays()
        {
            var s = np.float64(2.5);
            Kind(s.copy()).Should().Be("ndarray0", "NumPy's np.float64(2.5).copy() is a scalar");
            Kind(s.astype(np.float32)).Should().Be("ndarray0", "NumPy's np.float64(2.5).astype(np.float32) is a scalar");
            s.copy().GetDouble().Should().Be(2.5);
        }
    }
}
