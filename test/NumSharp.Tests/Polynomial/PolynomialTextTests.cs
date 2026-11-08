using System;
using System.Collections;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     <c>numpy.polynomial</c> with text arguments — S, U and <c>StringDType</c> arrays, Python strs (C# strings) and
    ///     Python lists. NumPy has no text polynomial: its Python code meets the text with whatever its statements do, and
    ///     NumSharp follows them up to the first statement that would compute with the strings themselves. So every answer
    ///     a type, dtype or shape decides is NumPy's (pinned here, all probed against NumPy 2.4.2): as_series' common-type
    ///     refusal, np.array's stacking in <c>{p}line</c> (numbers turned into text), the string ufuncs' TypeErrors at
    ///     <c>c[:1]*0</c>, <c>c *= scl</c>, <c>x*0</c> / <c>2*x</c> / <c>1 - x</c>, <c>x + 0.0</c>, CPython's errors for strs
    ///     and lists; the recurrences that go on to repeat and concatenate the strings are the documented refusal
    ///     (<c>NotSupportedException</c>, the series algebra's object-arithmetic policy).
    /// </summary>
    [TestClass]
    public class PolynomialTextTests
    {
        /// <summary>The <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary><c>np.array('x')</c> (<c>&lt;U1</c>, 0-d).</summary>
        private static NDArray U0 => np.array("x");

        /// <summary><c>['ab', 'c', 'de']</c> (<c>&lt;U2</c>).</summary>
        private static NDArray U3 => np.array(new[] { "ab", "c", "de" });

        /// <summary><c>np.array(b'x')</c> (<c>|S1</c>, 0-d).</summary>
        private static NDArray S0 => np.array("x").astype(np.dtype("S1"));

        /// <summary><c>[b'ab', b'c', b'de']</c> (<c>|S2</c>).</summary>
        private static NDArray S3 => np.array(new[] { "ab", "c", "de" }).astype(np.dtype("S2"));

        /// <summary><c>np.array('x', StringDType())</c> (0-d).</summary>
        private static NDArray T0 => np.array(new[] { "x" }, T).reshape(Array.Empty<long>());

        /// <summary><c>['ab', 'c', 'de']</c> as <c>StringDType</c>.</summary>
        private static NDArray T3 => np.array(new[] { "ab", "c", "de" }, T);

        /// <summary><c>[['1', '2'], ['3', '4']]</c> (<c>&lt;U1</c>).</summary>
        private static NDArray U22 => np.array(new[,] { { "1", "2" }, { "3", "4" } });

        /// <summary>An empty <c>&lt;U1</c> array.</summary>
        private static NDArray Ue => np.array(Array.Empty<string>()).astype(np.dtype("U1"));

        /// <summary><c>[1.5, 2.0, 3.0]</c> (float64).</summary>
        private static NDArray F3 => np.array(new[] { 1.5, 2.0, 3.0 });

        /// <summary>The string multiply's TypeError without out= (NumPy's text).</summary>
        private const string OutKwarg =
            "The 'out' kwarg is necessary when using the string multiply ufunc directly. Use numpy.strings.multiply to multiply strings without specifying 'out'.";

        /// <summary>The dispatcher's refusal for operand classes (<c>StringDType</c>, <c>_PyLongDType</c>, ...).</summary>
        /// <param name="ufunc">The ufunc.</param>
        /// <param name="classes">The DType class names.</param>
        /// <returns>The text.</returns>
        private static string NoLoop(string ufunc, params string[] classes)
        {
            var reprs = classes.Select(c => $"<class 'numpy.dtypes.{c}'>").ToArray();
            string ins = reprs.Length == 1 ? reprs[0] : "(" + string.Join(", ", reprs) + ")";
            return $"ufunc '{ufunc}' did not contain a loop with signature matching types {ins} -> None";
        }

        /// <summary>The legacy resolver's refusal for operand descriptors (<c>dtype('&lt;U2')</c>, <c>dtype('float64')</c>).</summary>
        /// <param name="ufunc">The ufunc.</param>
        /// <param name="descrs">The descriptor reprs inside <c>dtype(...)</c>.</param>
        /// <returns>The text.</returns>
        private static string LegacyNoLoop(string ufunc, params string[] descrs)
            => $"ufunc '{ufunc}' did not contain a loop with signature matching types ({string.Join(", ", descrs.Select(d => $"dtype('{d}')"))}) -> None";

        /// <summary>The elements of a text array in C order.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The texts.</returns>
        private static string[] Texts(NDArray a)
        {
            object list = a.ndim == 0 ? a.tolist() : a.flatten().tolist();
            return list is IList items ? items.Cast<object>().Select(x => x is byte[] b ? System.Text.Encoding.Latin1.GetString(b) : (string)x).ToArray() : new[] { (string)list };
        }

        // ------------------------------------------------------------------------------------------- the checking routes

        [TestMethod]
        public void AsSeriesConsumers_RejectTextAsNoCommonType()
        {
            var pp = np.polynomial.polynomial;
            new Action(() => pp.polyadd(U3, F3)).Should().Throw<ValueError>().WithMessage("Coefficient arrays have no common type");
            new Action(() => np.polynomial.chebyshev.chebmul(F3, T3)).Should().Throw<ValueError>().WithMessage("Coefficient arrays have no common type");
            new Action(() => pp.polytrim(S3)).Should().Throw<ValueError>().WithMessage("Coefficient arrays have no common type");
            new Action(() => pp.polyadd("ab", F3)).Should().Throw<ValueError>().WithMessage("Coefficient arrays have no common type");
            pp.polyfromroots(Ue).ToArray<double>().Should().Equal(1.0);
            var t3 = T3;
            np.polynomial.polyutils.trimseq(t3).Should().BeSameAs(t3);
        }

        [TestMethod]
        public void VanderFitWeightsAndRoots_RaiseTheirFirstStatementsError()
        {
            var pp = np.polynomial.polynomial;
            new Action(() => pp.polyvander(U3, 2)).Should().Throw<TypeError>().WithMessage(LegacyNoLoop("add", "<U2", "float64"));
            new Action(() => pp.polyvander(S3, 2)).Should().Throw<TypeError>().WithMessage(LegacyNoLoop("add", "S2", "float64"));
            new Action(() => pp.polyvander(T3, 2)).Should().Throw<TypeError>().WithMessage(NoLoop("add", "StringDType", "_PyFloatDType"));
            new Action(() => pp.polyvander("ab", 2)).Should().Throw<TypeError>().WithMessage(LegacyNoLoop("add", "<U2", "float64"));
            // vander2d's stack: np.array's promotion of the points (a number turned into text of its width).
            new Action(() => pp.polyvander2d("ab", 1L, new[] { 1, 1 })).Should().Throw<TypeError>().WithMessage(LegacyNoLoop("add", "<U21", "float64"));
            new Action(() => pp.polyvander2d(U3, F3, new[] { 1, 1 })).Should().Throw<TypeError>().WithMessage(LegacyNoLoop("add", "<U32", "float64"));
            new Action(() => pp.polyvander2d(U3, T3, new[] { 1, 1 })).Should().Throw<TypeError>().WithMessage(NoLoop("add", "StringDType", "_PyFloatDType"));
            new Action(() => pp.polyvander2d("ab", new object[] { 1L, 2L }, new[] { 1, 1 })).Should().Throw<ValueError>().WithMessage(
                "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.");
            // fit: x = np.asarray(x) + 0.0 with a WEAK Python float.
            new Action(() => pp.polyfit(T3, F3, 1L)).Should().Throw<TypeError>().WithMessage(NoLoop("add", "StringDType", "_PyFloatDType"));
            new Action(() => pp.polyfit(F3, F3, 1L, w: T3)).Should().Throw<TypeError>().WithMessage(NoLoop("add", "StringDType", "_PyFloatDType"));
            // weights: NumPy's own ufunc statements.
            new Action(() => np.polynomial.chebyshev.chebweight(T3)).Should().Throw<TypeError>().WithMessage(NoLoop("add", "_PyFloatDType", "StringDType"));
            new Action(() => np.polynomial.legendre.legweight(U3)).Should().Throw<TypeError>().WithMessage(LegacyNoLoop("multiply", "<U2", "float64"));
            new Action(() => np.polynomial.laguerre.lagweight(T3)).Should().Throw<TypeError>().WithMessage(NoLoop("negative", "StringDType"));
            // polyvalfromroots: x - r with a weak Python int x.
            new Action(() => pp.polyvalfromroots(0L, T3)).Should().Throw<TypeError>().WithMessage(NoLoop("subtract", "_PyLongDType", "StringDType"));
            // trimcoef: tol < 0 is the less ufunc on text.
            new Action(() => np.polynomial.polyutils.trimcoef(F3, T0)).Should().Throw<TypeError>().WithMessage(NoLoop("less", "StringDType", "_PyLongDType"));
        }

        [TestMethod]
        public void MapparmsAndMapdomain_StrItemsAndPoints()
        {
            var pu = np.polynomial.polyutils;
            new Action(() => pu.mapparms(U3, new object[] { 0L, 1L })).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for -: 'numpy.str_' and 'numpy.str_'");
            new Action(() => pu.mapparms(new object[] { "ab", "c" }, new object[] { 0L, 1L })).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for -: 'str' and 'str'");
            new Action(() => pu.mapparms(new object[] { "a", 1L }, new object[] { 0L, 1L })).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for -: 'int' and 'str'");
            new Action(() => pu.mapparms(new object[] { 0L, 1L }, new object[] { 1L, "b" })).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for -: 'str' and 'int'");
            new Action(() => pu.mapparms(new object[] { 0L, 0L }, new object[] { "a", "b" })).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for -: 'str' and 'str'");
            new Action(() => pu.mapparms(new object[] { 0L, 1L }, new object[] { "a" })).Should().Throw<IndexError>()
                .WithMessage("list index out of range");
            new Action(() => pu.mapparms(("a", 1L), (0L, 1L))).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for -: 'int' and 'str'");
            new Action(() => pu.mapdomain("ab", new object[] { 0L, 1L }, new object[] { 1L, 3L })).Should().Throw<TypeError>()
                .WithMessage(LegacyNoLoop("multiply", "float64", "<U2"));
            new Action(() => pu.mapdomain(new object[] { "ab", "c" }, new object[] { 0L, 1L }, new object[] { 1L, 3L })).Should().Throw<TypeError>()
                .WithMessage(LegacyNoLoop("multiply", "float64", "<U2"));
        }

        // ------------------------------------------------------------------------------------------- {p}line

        [TestMethod]
        public void Line_StacksTextAsNpArrayDoes()
        {
            var pp = np.polynomial.polynomial;
            using (var r = pp.polyline(U0, 1L)) { r.dtype.str.Should().Be("<U21"); Texts(r).Should().Equal("x", "1"); }
            using (var r = pp.polyline(S0, 1L)) { r.dtype.str.Should().Be("|S21"); Texts(r).Should().Equal("x", "1"); }
            using (var r = pp.polyline(U0, 2.5)) { r.dtype.str.Should().Be("<U32"); Texts(r).Should().Equal("x", "2.5"); }
            using (var r = pp.polyline(U0, true)) { r.dtype.str.Should().Be("<U5"); Texts(r).Should().Equal("x", "True"); }
            using (var r = pp.polyline(U0, new Complex(0, 1))) { r.dtype.str.Should().Be("<U64"); Texts(r).Should().Equal("x", "1j"); }
            using (var r = pp.polyline(U0, 0L)) { r.dtype.str.Should().Be("<U1"); Texts(r).Should().Equal("x"); }
            using (var r = pp.polyline(U0, T0)) { r.dtype.Should().Be(T); Texts(r).Should().Equal("x", "x"); }
            using (var r = pp.polyline("x", 1L)) { r.dtype.str.Should().Be("<U21"); Texts(r).Should().Equal("x", "1"); }
            using (var r = pp.polyline(1L, "x")) { r.dtype.str.Should().Be("<U21"); Texts(r).Should().Equal("1", "x"); }
            using (var r = np.polynomial.hermite.hermline(U0, 3L)) { r.dtype.str.Should().Be("<U32"); Texts(r).Should().Equal("x", "1.5"); }
            new Action(() => pp.polyline(U3, 1L)).Should().Throw<ValueError>().WithMessage(
                "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.");
            // A text scale compares unequal to 0 everywhere: its truth value is decided by its size.
            new Action(() => pp.polyline(U0, U3)).Should().Throw<ValueError>()
                .WithMessage("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
            new Action(() => pp.polyline(U0, Ue)).Should().Throw<ValueError>()
                .WithMessage("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
            // A StringDType entry with a number: NumPy's stack is an object array (the documented refusal).
            new Action(() => pp.polyline(T0, 1L)).Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void Line_LaguerreAndHermiteArithmetic()
        {
            var lag = np.polynomial.laguerre;
            new Action(() => lag.lagline(U0, 1L)).Should().Throw<TypeError>().WithMessage(LegacyNoLoop("add", "<U1", "int64"));
            new Action(() => lag.lagline(1L, T0)).Should().Throw<TypeError>().WithMessage(NoLoop("add", "_PyLongDType", "StringDType"));
            new Action(() => lag.lagline(U0, U0)).Should().Throw<TypeError>().WithMessage(NoLoop("negative", "StrDType"));
            new Action(() => lag.lagline(U0, S0)).Should().Throw<TypeError>()
                .WithMessage("ufunc 'add' cannot use operands with types dtype('<U1') and dtype('S1')");
            new Action(() => lag.lagline("x", 1L)).Should().Throw<TypeError>().WithMessage("can only concatenate str (not \"int\") to str");
            new Action(() => lag.lagline(1L, "x")).Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for +: 'int' and 'str'");
            new Action(() => lag.lagline("x", "y")).Should().Throw<TypeError>().WithMessage("bad operand type for unary -: 'str'");
            new Action(() => np.polynomial.hermite.hermline(1L, "x")).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for /: 'str' and 'int'");
            new Action(() => np.polynomial.hermite.hermline(1L, T0)).Should().Throw<TypeError>()
                .WithMessage(NoLoop("divide", "StringDType", "_PyLongDType"));
        }

        [TestMethod]
        public void Line_PythonListsArePythonObjects()
        {
            // A list is never equal to 0 (Python's comparison), concatenates under +, and has no unary - or /.
            new Action(() => np.polynomial.polynomial.polyline(1L, new object[] { 0L })).Should().Throw<ValueError>().WithMessage(
                "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.");
            using (var r = np.polynomial.polynomial.polyline(new object[] { 1L }, new object[] { 2L }))
            {
                r.shape.Should().Equal(2L, 1L);
                r.ToArray<long>().Should().Equal(1L, 2L);
            }
            new Action(() => np.polynomial.laguerre.lagline(new object[] { 1L }, new object[] { 2L })).Should().Throw<TypeError>()
                .WithMessage("bad operand type for unary -: 'list'");
            new Action(() => np.polynomial.hermite.hermline(1L, new object[] { 2L })).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for /: 'list' and 'int'");
            // An ndarray scale compares elementwise: [0] != 0 is [False], so the one-coefficient branch.
            np.polynomial.polynomial.polyline(1L, np.array(new long[] { 0 })).ToArray<long>().Should().Equal(1L);
        }

        // ------------------------------------------------------------------------------------------- {p}der / {p}int

        [TestMethod]
        public void DerAndInt_CheckTheirArgumentsFirst_AndCopyForOrderZero()
        {
            var pp = np.polynomial.polynomial;
            var u3 = U3;
            using (var c = pp.polyder(u3, 0))
            {
                c.Should().NotBeSameAs(u3);
                Texts(c).Should().Equal("ab", "c", "de");
            }
            Texts(pp.polyint(T3, 0)).Should().Equal("ab", "c", "de");
            new Action(() => pp.polyder(U3, -1)).Should().Throw<ValueError>().WithMessage("The order of derivation must be non-negative");
            new Action(() => pp.polyder(U3, 1, null, 3)).Should().Throw<AxisError>().WithMessage("axis 3 is out of bounds for array of dimension 1*");
            new Action(() => pp.polyint(U3, -1)).Should().Throw<ValueError>().WithMessage("The order of integration must be non-negative");
            new Action(() => pp.polyint(U3, 1, new object[] { 1L, 2L })).Should().Throw<ValueError>().WithMessage("Too many integration constants");
            new Action(() => pp.polyint(U3, 1, null, new object[] { 0L, 1L })).Should().Throw<ValueError>().WithMessage("lbnd must be a scalar.");
            new Action(() => pp.polyint(U3, 1, null, null, new object[] { 1L, 2L })).Should().Throw<ValueError>().WithMessage("scl must be a scalar.");
        }

        [TestMethod]
        public void DerAndInt_StringUfuncStatements()
        {
            var pp = np.polynomial.polynomial;
            // m >= len(c): c[:1]*0, the string multiply without out= (S / U raise, StringDType repeats 0 times).
            new Action(() => pp.polyder(U0)).Should().Throw<TypeError>().WithMessage(OutKwarg);
            Texts(pp.polyder(T0)).Should().Equal("");
            Texts(np.polynomial.hermite_e.hermeder(T0)).Should().Equal("");
            // The first `c *= scl`: no loop for a float scale.
            new Action(() => pp.polyder(T3, 1, 0.5)).Should().Throw<TypeError>().WithMessage(NoLoop("multiply", "StringDType", "_PyFloatDType"));
            // An N-D series' first row statement: j*c[j] / c[0]*0 (S / U), Laguerre's negation (any text).
            new Action(() => pp.polyder(U22)).Should().Throw<TypeError>().WithMessage(OutKwarg);
            new Action(() => np.polynomial.laguerre.lagder(U22)).Should().Throw<TypeError>().WithMessage(NoLoop("negative", "StrDType"));
            new Action(() => pp.polyint(U22)).Should().Throw<TypeError>().WithMessage(OutKwarg);
            new Action(() => np.polynomial.laguerre.lagint(U22)).Should().Throw<TypeError>().WithMessage(NoLoop("negative", "StrDType"));
            new Action(() => pp.polyint(Ue)).Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 0 with size 0");
            // Past those statements NumPy repeats and concatenates the strings: the documented refusal.
            new Action(() => pp.polyder(U3)).Should().Throw<NotSupportedException>();
            new Action(() => pp.polyint(T3)).Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void Int_TextIntegrationConstants()
        {
            // len(k) first; then the order that uses k[i]: k[i] - {p}val(lbnd, tmp) has no subtract loop for text (a 1-D
            // array's elements convert as Python strs / bytes of their own length), c[0] += k[i] no add loop.
            var pp = np.polynomial.polynomial;
            new Action(() => pp.polyint(F3, 1, T3)).Should().Throw<ValueError>().WithMessage("Too many integration constants");
            new Action(() => pp.polyint(F3, 3, T3)).Should().Throw<TypeError>().WithMessage(LegacyNoLoop("subtract", "<U2", "float64"));
            new Action(() => pp.polyint(F3, 3, S3)).Should().Throw<TypeError>().WithMessage(LegacyNoLoop("subtract", "S2", "float64"));
            new Action(() => np.polynomial.laguerre.lagint(np.array(new[] { 0.0 }), 3, S3)).Should().Throw<TypeError>()
                .WithMessage(LegacyNoLoop("add", "float64", "S2"));
        }

        // ------------------------------------------------------------------------------------------- the {p}val family

        [TestMethod]
        public void Val_FirstStatementOnTextPoints_AndTheEmptySeries()
        {
            new Action(() => np.polynomial.polynomial.polyval(F3, Ue)).Should().Throw<IndexError>().WithMessage("index -1 is out of bounds for axis 0 with size 0");
            new Action(() => np.polynomial.chebyshev.chebval(F3, Ue)).Should().Throw<IndexError>().WithMessage("index -2 is out of bounds for axis 0 with size 0");
            new Action(() => np.polynomial.polynomial.polyval(U3, F3)).Should().Throw<TypeError>().WithMessage(OutKwarg);
            new Action(() => np.polynomial.hermite.hermval(U3, F3)).Should().Throw<TypeError>().WithMessage(OutKwarg);
            new Action(() => np.polynomial.laguerre.lagval(U3, F3)).Should().Throw<TypeError>().WithMessage(LegacyNoLoop("subtract", "int64", "<U2"));
            new Action(() => np.polynomial.chebyshev.chebval(U3, new object[] { 1L, 2L })).Should().Throw<TypeError>()
                .WithMessage(LegacyNoLoop("multiply", "float64", "<U2"));
            // A text series: the recurrence starts with the strings (the documented refusal).
            new Action(() => np.polynomial.polynomial.polyval(F3, U3)).Should().Throw<NotSupportedException>();
        }

        // ------------------------------------------------------------------------------------------- np.linalg.cond (found by the same sweep)

        [TestMethod]
        public void Cond_ArrayOrder_IsUnhashable()
        {
            // NumPy's `p in {2, -2}` hashes p: an ndarray (text or numeric) or a list cannot be hashed.
            var x = np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            new Action(() => np.linalg.cond(x, np.array(2L))).Should().Throw<TypeError>().WithMessage("unhashable type: 'numpy.ndarray'");
            new Action(() => np.linalg.cond(x, T3)).Should().Throw<TypeError>().WithMessage("unhashable type: 'numpy.ndarray'");
            new Action(() => np.linalg.cond(x, new object[] { 1L })).Should().Throw<TypeError>().WithMessage("unhashable type: 'list'");
            new Action(() => np.linalg.cond(np.zeros(new Shape(0, 0)), np.array(2L))).Should().Throw<LinAlgError>()
                .WithMessage("cond is not defined on empty arrays");
        }
    }
}
