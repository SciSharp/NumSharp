using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs in the KEYWORD surface of the max/min reduction family: <c>np.max</c>, <c>np.min</c>,
    ///     <c>np.amax</c>, <c>np.amin</c>, <c>np.nanmax</c>, <c>np.nanmin</c> and the methods <c>ndarray.max</c> /
    ///     <c>ndarray.min</c>. NumPy 2.4.2 gives every one of them <c>out=</c>, <c>initial=</c> and <c>where=</c>;
    ///     NumSharp has none of the three. A ported call that names one does not compile, and NumSharp cannot
    ///     write a max/min into a caller's buffer, start it from a value (which is also the only way NumPy reduces
    ///     an EMPTY slice without an error) or skip masked elements. Each test asserts NumPy's answer, so it FAILS
    ///     while the keywords are missing and passes once they behave like NumPy's.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why the calls go through reflection.</b> A direct <c>np.max(a, initial: 10.0)</c> does not compile
    ///     today, and a test file that does not compile cannot pin anything. <see cref="Np"/> and <see cref="Nd"/>
    ///     therefore bind the overload whose parameter names are NumPy's, in NumPy's order
    ///     (<c>a, axis, out, keepdims, initial, where</c>; the method form drops <c>a</c>). Today no such overload
    ///     exists, so every case records "no overload". Once the overload exists, the SAME cases check NumPy's
    ///     values with no edit here; remove <c>[OpenBugs]</c> then, and move the tests next to the functions' own.
    ///     </para>
    ///     <para>
    ///     <b>What NumPy does</b> (<c>numpy/_core/fromnumeric.py</c> and <c>_methods.py</c>): <c>np.max</c> /
    ///     <c>np.amax</c> / <c>ndarray.max</c> are <c>np.maximum.reduce</c>, <c>np.min</c> / <c>np.amin</c> /
    ///     <c>ndarray.min</c> are <c>np.minimum.reduce</c>, and <c>np.nanmax</c> / <c>np.nanmin</c> are
    ///     <c>np.fmax.reduce</c> / <c>np.fmin.reduce</c> for a float input. <c>initial</c> is folded in as one more
    ///     element of every output (cast to the reduction dtype like a Python scalar: <c>7.5</c> truncates into an
    ///     int64 reduction, <c>1000</c> overflows an int8 one); <c>where</c> broadcasts against the input and drops
    ///     the masked elements, and because max/min have no identity it REQUIRES <c>initial</c>;
    ///     <c>out</c> receives the result (cast from the reduction dtype, unsafe like every ufunc.reduce
    ///     <c>out</c>) and is returned itself.
    ///     </para>
    ///     <para>
    ///     <b>Error cases compare the MESSAGE only</b>; the label names NumPy's exception type. NumSharp's house
    ///     types match NumPy's in general, but a non-bool <c>where=</c> is a documented exception
    ///     (<c>ArgumentException</c> with NumPy's TypeError text, <c>DefaultEngine.UfuncOut.ValidateWhereMask</c>),
    ///     and this file pins the keywords, not that mapping.
    ///     </para>
    ///     <para>
    ///     <b>A sibling pin</b> (<see cref="AllAny_Where_NonBoolMask_RaisesNumPysCastError"/>): <c>np.all</c> /
    ///     <c>np.any</c> already HAVE <c>where=</c>, but accept a non-bool mask where NumPy raises.
    ///     </para>
    ///     <para>
    ///     <b>Also observed, not pinned:</b> NumSharp's static overloads are <c>(a, axis, keepdims, dtype)</c>. The
    ///     <c>dtype</c> parameter does not exist in NumPy (a ported call never passes it), but the THIRD positional
    ///     slot is <c>keepdims</c> where NumPy's is <c>out</c>, so a positional port such as <c>np.max(a, 1, o)</c>
    ///     does not compile either. NumPy's tuple <c>axis</c> is out of scope here: every NumSharp reduction is
    ///     single-axis, a library-wide gap.
    ///     </para>
    ///     <para>
    ///     Every expected string is NumPy 2.4.2's output for the same call, probed on 2026-10-04, in
    ///     <see cref="Describe"/>'s format: <c>dtype (shape) values…</c>, float bits as hex in logical C order.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OpenBugsReductionKeywordTests : TestClass
    {
        /// <summary>The <c>np.*</c> functions of the family; NumPy gives each the same signature.</summary>
        private static readonly string[] NpMaxMin = { "max", "min", "amax", "amin", "nanmax", "nanmin" };

        /// <summary>
        ///     NumPy 2.4.2's parameter names of every <c>np.*</c> function of the family, in NumPy's order
        ///     (<c>inspect.signature</c>: <c>(a, axis=None, out=None, keepdims=&lt;no value&gt;,
        ///     initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>).
        /// </summary>
        private static readonly string[] NpParameters = { "a", "axis", "out", "keepdims", "initial", "where" };

        /// <summary>The <c>ndarray</c> methods of the family (NumPy's ndarray has no <c>amax</c> / <c>amin</c> methods).</summary>
        private static readonly string[] NdArrayMaxMin = { "max", "min" };

        /// <summary>
        ///     NumPy 2.4.2's parameter names of <c>ndarray.max</c> / <c>ndarray.min</c>, in NumPy's order
        ///     (<c>a.max(axis=None, out=None, *, keepdims=&lt;no value&gt;, initial=&lt;no value&gt;,
        ///     where=&lt;no value&gt;)</c>; C# has no keyword-only marker, but the order is kept).
        /// </summary>
        private static readonly string[] NdArrayParameters = { "axis", "out", "keepdims", "initial", "where" };

        /// <summary>
        ///     Every function of the family must have an overload with NumPy's parameter NAMES in NumPy's ORDER, and
        ///     every parameter after the array must be optional, so a ported call (<c>np.max(a, axis: 1, out: o)</c>,
        ///     <c>a.min(where: m, initial: 9.0)</c>) compiles and binds. NumSharp's overloads are
        ///     <c>(a, axis, keepdims, dtype)</c> and <c>(axis, keepdims, dtype)</c> / <c>(dtype)</c>.
        /// </summary>
        /// <remarks>
        ///     The same check as <c>LinAlgSignatureParityTests</c>: a signature is API, and a missing or reordered
        ///     keyword passes every value test while it breaks every named-argument call ported from Python. Other
        ///     overloads may exist next to the NumPy form (NumSharp's <c>dtype</c> extension, the generic
        ///     <c>amax&lt;T&gt;</c>).
        /// </remarks>
        [TestMethod, OpenBugs]
        public void MaxMinFamily_TakesNumPysKeywords_InNumPysOrder()
        {
            var fails = new List<string>();
            foreach (var method in NpMaxMin)
                CheckSignature(fails, typeof(np), method, isStatic: true, NpParameters, requiredCount: 1);
            foreach (var method in NdArrayMaxMin)
                CheckSignature(fails, typeof(NDArray), method, isStatic: false, NdArrayParameters, requiredCount: 0);
            AssertNoMismatch(fails, "NumPy's max/min family takes (a, axis, out, keepdims, initial, where)");
        }

        /// <summary>
        ///     <c>out=</c> must receive the result and be returned ITSELF, for every function of the family, the
        ///     method forms and a 0-d out; the result is cast into an out of another dtype (int64, float32, bool), and
        ///     an out whose shape is not the result's raises NumPy's broadcast error. NumSharp has no <c>out</c>
        ///     parameter, so every reduction allocates and a caller's buffer is never written.
        /// </summary>
        /// <remarks>
        ///     NumPy routes <c>out</c> through <c>ufunc.reduce</c>: the reduction runs in the input's dtype and the
        ///     result is cast into <c>out</c> with the 'unsafe' rule, which is why a float64 max lands in a bool out
        ///     as <c>True</c>. A wrong-shape out is NpyIter's error, verbatim (trailing space included).
        /// </remarks>
        [TestMethod, OpenBugs]
        public void MaxMinFamily_Out_WritesIntoAndReturnsTheGivenArray()
        {
            var fails = new List<string>();
            CheckOut(fails, "np.max(a, axis=1, out=o)", o => Np("max", ("a", A()), ("axis", 1), ("out", o)), np.empty(new Shape(2), np.float64), "float64 (2,) 4014000000000000 4018000000000000");
            CheckOut(fails, "np.min(a, axis=0, out=o)", o => Np("min", ("a", A()), ("axis", 0), ("out", o)), np.empty(new Shape(3), np.float64), "float64 (3,) 3ff0000000000000 4000000000000000 4008000000000000");
            CheckOut(fails, "np.max(a, out=0-d o)", o => Np("max", ("a", A()), ("out", o)), np.empty(Shape.Scalar, np.float64), "float64 () 4018000000000000");
            CheckOut(fails, "np.amax(a, axis=1, out=o)", o => Np("amax", ("a", A()), ("axis", 1), ("out", o)), np.empty(new Shape(2), np.float64), "float64 (2,) 4014000000000000 4018000000000000");
            CheckOut(fails, "np.amin(a, axis=1, out=o)", o => Np("amin", ("a", A()), ("axis", 1), ("out", o)), np.empty(new Shape(2), np.float64), "float64 (2,) 3ff0000000000000 4000000000000000");
            CheckOut(fails, "np.nanmax([[1,nan],[nan,3]], axis=1, out=o)", o => Np("nanmax", ("a", Nans()), ("axis", 1), ("out", o)), np.empty(new Shape(2), np.float64), "float64 (2,) 3ff0000000000000 4008000000000000");
            CheckOut(fails, "np.nanmin([[1,nan],[nan,3]], axis=0, out=o)", o => Np("nanmin", ("a", Nans()), ("axis", 0), ("out", o)), np.empty(new Shape(2), np.float64), "float64 (2,) 3ff0000000000000 4008000000000000");
            CheckOut(fails, "a.max(axis=1, out=o)", o => Nd(A(), "max", ("axis", 1), ("out", o)), np.empty(new Shape(2), np.float64), "float64 (2,) 4014000000000000 4018000000000000");
            CheckOut(fails, "a.min(axis=1, out=o)", o => Nd(A(), "min", ("axis", 1), ("out", o)), np.empty(new Shape(2), np.float64), "float64 (2,) 3ff0000000000000 4000000000000000");
            CheckOut(fails, "np.max(a, axis=1, out=int64 o) casts", o => Np("max", ("a", A()), ("axis", 1), ("out", o)), np.empty(new Shape(2), np.int64), "int64 (2,) 5 6");
            CheckOut(fails, "np.max(a, axis=1, out=float32 o) casts", o => Np("max", ("a", A()), ("axis", 1), ("out", o)), np.empty(new Shape(2), np.float32), "float32 (2,) 40a00000 40c00000");
            CheckOut(fails, "np.max(a, axis=1, out=bool o) casts", o => Np("max", ("a", A()), ("axis", 1), ("out", o)), np.empty(new Shape(2), np.@bool), "bool (2,) True True");
            CheckRaises(fails, "np.max(a, axis=1, out=(3,)) (ValueError)", () => Np("max", ("a", A()), ("axis", 1), ("out", np.empty(new Shape(3), np.float64))), "operands could not be broadcast together with remapped shapes [original->remapped]: (3,)->(3,newaxis) (2,3) ");
            AssertNoMismatch(fails, "NumPy's max/min write the result into out= and return out itself");
        }

        /// <summary>
        ///     <c>initial=</c> must join every output as one more element: it can be the maximum (or minimum) itself,
        ///     it makes an EMPTY slice reducible (NumPy raises "zero-size array to reduction operation maximum which
        ///     has no identity" without it), a NaN on either side propagates through max/min while nanmax/nanmin skip
        ///     a NaN element (an all-NaN slice returns <c>initial</c>), it is cast to the reduction dtype (<c>7.5</c>
        ///     truncates into int64), it raises when it does not fit an integer dtype, and <c>initial=None</c> means
        ///     no initial. NumSharp has no <c>initial</c> parameter.
        /// </summary>
        /// <remarks>
        ///     The integer cases are NumPy's Python-int rule (NEP 50): a C# integer literal is a weak Python int, so
        ///     <c>initial: 1000</c> into an int8 reduction raises NumPy's OverflowError text rather than wrapping. An
        ///     implementation that types <c>initial</c> as <c>double?</c> cannot tell <c>7</c> from <c>7.0</c> and
        ///     loses exactness past 2^53 in an int64 reduction; <c>object</c> (or per-kind overloads) keeps it.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void MaxMinFamily_Initial_JoinsEveryOutputAsOneMoreElement()
        {
            var fails = new List<string>();
            Check(fails, "np.max(a, initial=10.0)", () => Np("max", ("a", A()), ("initial", 10.0)), "float64 () 4024000000000000");
            Check(fails, "np.max(a, axis=0, initial=4.5)", () => Np("max", ("a", A()), ("axis", 0), ("initial", 4.5)), "float64 (3,) 4012000000000000 4014000000000000 4018000000000000");
            Check(fails, "np.min(a, axis=1, initial=1.5)", () => Np("min", ("a", A()), ("axis", 1), ("initial", 1.5)), "float64 (2,) 3ff0000000000000 3ff8000000000000");
            Check(fails, "np.amin(a, initial=-2.0)", () => Np("amin", ("a", A()), ("initial", -2.0)), "float64 () c000000000000000");
            Check(fails, "np.amax(a, axis=1, keepdims=True, initial=4.5)", () => Np("amax", ("a", A()), ("axis", 1), ("keepdims", true), ("initial", 4.5)), "float64 (2,1) 4014000000000000 4018000000000000");
            Check(fails, "np.max(empty, initial=-1.0)", () => Np("max", ("a", np.zeros(new Shape(0), np.float64)), ("initial", -1.0)), "float64 () bff0000000000000");
            Check(fails, "np.max(zeros((0,3)), axis=0, initial=7.0)", () => Np("max", ("a", np.zeros(new Shape(0, 3), np.float64)), ("axis", 0), ("initial", 7.0)), "float64 (3,) 401c000000000000 401c000000000000 401c000000000000");
            Check(fails, "np.min(zeros((2,0)), axis=1, initial=inf)", () => Np("min", ("a", np.zeros(new Shape(2, 0), np.float64)), ("axis", 1), ("initial", double.PositiveInfinity)), "float64 (2,) 7ff0000000000000 7ff0000000000000");
            Check(fails, "np.max([nan, 1], initial=5.0)", () => Np("max", ("a", np.array(new[] { double.NaN, 1.0 })), ("initial", 5.0)), "float64 () nan");
            Check(fails, "np.max([1, 2], initial=nan)", () => Np("max", ("a", np.array(new[] { 1.0, 2.0 })), ("initial", double.NaN)), "float64 () nan");
            Check(fails, "np.max(int64 [1, 2], initial=7)", () => Np("max", ("a", np.array(new long[] { 1, 2 })), ("initial", 7)), "int64 () 7");
            Check(fails, "np.max(int64 [1, 2], initial=7.5) truncates", () => Np("max", ("a", np.array(new long[] { 1, 2 })), ("initial", 7.5)), "int64 () 7");
            Check(fails, "np.max(float32 [1, 2], initial=2.5)", () => Np("max", ("a", np.array(new[] { 1f, 2f })), ("initial", 2.5)), "float32 () 40200000");
            Check(fails, "np.max(bool [F, F], initial=True)", () => Np("max", ("a", np.array(new[] { false, false })), ("initial", true)), "bool () True");
            Check(fails, "np.max(a, initial=None) is no initial", () => Np("max", ("a", A()), ("initial", null)), "float64 () 4018000000000000");
            Check(fails, "np.nanmax([nan, 1], initial=5.0)", () => Np("nanmax", ("a", np.array(new[] { double.NaN, 1.0 })), ("initial", 5.0)), "float64 () 4014000000000000");
            Check(fails, "np.nanmin([[nan,nan],[2,nan]], axis=1, initial=0.5)", () => Np("nanmin", ("a", np.array(new[] { double.NaN, double.NaN, 2.0, double.NaN }).reshape(2, 2)), ("axis", 1), ("initial", 0.5)), "float64 (2,) 3fe0000000000000 3fe0000000000000");
            Check(fails, "np.nanmax([nan, nan], initial=3.0)", () => Np("nanmax", ("a", np.array(new[] { double.NaN, double.NaN })), ("initial", 3.0)), "float64 () 4008000000000000");
            Check(fails, "a.max(initial=10.0)", () => Nd(A(), "max", ("initial", 10.0)), "float64 () 4024000000000000");
            Check(fails, "a.min(axis=0, initial=0.5)", () => Nd(A(), "min", ("axis", 0), ("initial", 0.5)), "float64 (3,) 3fe0000000000000 3fe0000000000000 3fe0000000000000");
            CheckRaises(fails, "np.max(int8 [1, 2], initial=1000) (OverflowError)", () => Np("max", ("a", np.array(new sbyte[] { 1, 2 })), ("initial", 1000)), "Python integer 1000 out of bounds for int8");
            CheckRaises(fails, "np.min(uint8 [1, 2], initial=-1) (OverflowError)", () => Np("min", ("a", np.array(new byte[] { 1, 2 })), ("initial", -1)), "Python integer -1 out of bounds for uint8");
            AssertNoMismatch(fails, "NumPy folds initial= into every output of the reduction");
        }

        /// <summary>
        ///     <c>where=</c> must reduce only the selected elements: a bool mask broadcast against the input (a
        ///     <c>(2,1)</c> row mask, a <c>(3,)</c> column mask), with <c>keepdims</c>, for nanmax too, and an output
        ///     whose whole slice is masked off gets <c>initial</c>. Because max/min have no identity, NumPy REQUIRES
        ///     <c>initial</c> with any <c>where</c> (even an all-True mask), rejects a non-bool mask and a mask that
        ///     does not broadcast. NumSharp has no <c>where</c> parameter (<c>np.all</c> / <c>np.any</c> already
        ///     have one).
        /// </summary>
        /// <remarks>
        ///     The required-initial error names the ufunc the function reduces with: 'maximum' for np.max, 'minimum'
        ///     for np.min, 'fmax' for np.nanmax of a float array. The non-bool mask error is NumPy's TypeError text
        ///     (<c>_wheremask_converter</c> casts with the 'safe' rule).
        /// </remarks>
        [TestMethod, OpenBugs]
        public void MaxMinFamily_Where_ReducesOnlyTheSelectedElements()
        {
            var fails = new List<string>();
            Check(fails, "np.max(a, where=m, initial=-inf)", () => Np("max", ("a", A()), ("where", M()), ("initial", double.NegativeInfinity)), "float64 () 4018000000000000");
            Check(fails, "np.max(a, axis=1, where=m, initial=0.0)", () => Np("max", ("a", A()), ("axis", 1), ("where", M()), ("initial", 0.0)), "float64 (2,) 4008000000000000 4018000000000000");
            Check(fails, "np.min(a, axis=0, where=m, initial=100.0)", () => Np("min", ("a", A()), ("axis", 0), ("where", M()), ("initial", 100.0)), "float64 (3,) 3ff0000000000000 4000000000000000 4008000000000000");
            Check(fails, "np.amax(a, axis=0, where=[[T],[F]], initial=-1.0)", () => Np("amax", ("a", A()), ("axis", 0), ("where", np.array(new[] { true, false }).reshape(2, 1)), ("initial", -1.0)), "float64 (3,) 3ff0000000000000 4014000000000000 4008000000000000");
            Check(fails, "np.max(a, axis=1, where=[[F,F,F],[T,T,T]], initial=-7.0)", () => Np("max", ("a", A()), ("axis", 1), ("where", np.array(new[] { false, false, false, true, true, true }).reshape(2, 3)), ("initial", -7.0)), "float64 (2,) c01c000000000000 4018000000000000");
            Check(fails, "np.max(a, axis=1, keepdims=True, where=m, initial=0.0)", () => Np("max", ("a", A()), ("axis", 1), ("keepdims", true), ("where", M()), ("initial", 0.0)), "float64 (2,1) 4008000000000000 4018000000000000");
            Check(fails, "np.amin(a, where=[T,F,T], initial=9.0)", () => Np("amin", ("a", A()), ("where", np.array(new[] { true, false, true })), ("initial", 9.0)), "float64 () 3ff0000000000000");
            Check(fails, "np.nanmax([[nan,5],[4,nan]], axis=1, where=[[T,F],[T,T]], initial=-1.0)", () => Np("nanmax", ("a", np.array(new[] { double.NaN, 5.0, 4.0, double.NaN }).reshape(2, 2)), ("axis", 1), ("where", np.array(new[] { true, false, true, true }).reshape(2, 2)), ("initial", -1.0)), "float64 (2,) bff0000000000000 4010000000000000");
            Check(fails, "a.max(axis=1, where=m, initial=0.0)", () => Nd(A(), "max", ("axis", 1), ("where", M()), ("initial", 0.0)), "float64 (2,) 4008000000000000 4018000000000000");
            Check(fails, "a.min(where=m, initial=9.0)", () => Nd(A(), "min", ("where", M()), ("initial", 9.0)), "float64 () 3ff0000000000000");
            CheckRaises(fails, "np.max(a, where=m) without initial (ValueError)", () => Np("max", ("a", A()), ("where", M())), "reduction operation 'maximum' does not have an identity, so to use a where mask one has to specify 'initial'");
            CheckRaises(fails, "np.min(a, axis=1, where=m) without initial (ValueError)", () => Np("min", ("a", A()), ("axis", 1), ("where", M())), "reduction operation 'minimum' does not have an identity, so to use a where mask one has to specify 'initial'");
            CheckRaises(fails, "np.max(a, where=all True) without initial (ValueError)", () => Np("max", ("a", A()), ("where", np.ones(new Shape(2, 3), np.@bool))), "reduction operation 'maximum' does not have an identity, so to use a where mask one has to specify 'initial'");
            CheckRaises(fails, "np.nanmax(a, where=m) without initial (ValueError)", () => Np("nanmax", ("a", A()), ("where", M())), "reduction operation 'fmax' does not have an identity, so to use a where mask one has to specify 'initial'");
            CheckRaises(fails, "np.max(a, where=int64 mask, initial=0.0) (TypeError)", () => Np("max", ("a", A()), ("where", np.array(new long[] { 1, 0, 1, 0, 1, 1 }).reshape(2, 3)), ("initial", 0.0)), "Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
            CheckRaises(fails, "np.max(a, where=(2,) mask, initial=-1.0) (ValueError)", () => Np("max", ("a", A()), ("where", np.array(new[] { true, false })), ("initial", -1.0)), "operands could not be broadcast together with remapped shapes [original->remapped]: (2,3) (2,) ");
            AssertNoMismatch(fails, "NumPy's where= drops the masked elements and needs initial= for max/min");
        }

        /// <summary>
        ///     <c>np.all</c> / <c>np.any</c> / <c>ndarray.all</c> / <c>ndarray.any</c> already take <c>where=</c>, but a
        ///     NON-BOOL mask (int64, float64, uint8) must raise NumPy's cast error: NumPy converts the mask with the
        ///     'safe' rule, and only bool reaches bool. NumSharp reads the mask's truthiness and reduces, so a numeric
        ///     array passed by mistake (an index array, a weight vector) silently selects elements.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Logic/np.all.cs</c> <c>ApplyWhereForAll</c> / <c>ApplyWhereForAny</c> convert the mask
        ///     with <c>ToBool</c> (nonzero = True). The elementwise ufunc routes already reject a non-bool mask with
        ///     NumPy's text (<c>DefaultEngine.UfuncOut.ValidateWhereMask</c>), so the fix is that same check here.
        ///     Found on 2026-10-04 while validating this file's reflection plumbing against <c>np.all</c>, the one
        ///     reduction that already has NumPy's <c>where=</c>. These overloads exist, so the cases call them
        ///     directly.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void AllAny_Where_NonBoolMask_RaisesNumPysCastError()
        {
            var fails = new List<string>();
            var b = np.array(new[] { true, false, true, true, true, true }).reshape(2, 3);
            var int64Mask = np.array(new long[] { 1, 0, 1, 0, 1, 1 }).reshape(2, 3);
            var float64Mask = np.array(new[] { 1.0, 0.0, 1.0, 0.0, 1.0, 1.0 }).reshape(2, 3);
            var uint8Mask = np.array(new byte[] { 1, 0, 1, 0, 1, 1 }).reshape(2, 3);
            CheckRaises(fails, "np.all(b, where=int64 mask) (TypeError)", () => np.all(b, where: int64Mask),
                        "Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
            CheckRaises(fails, "np.any(b, where=int64 mask) (TypeError)", () => np.any(b, where: int64Mask),
                        "Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
            CheckRaises(fails, "np.all(b, where=float64 mask) (TypeError)", () => np.all(b, where: float64Mask),
                        "Cannot cast array data from dtype('float64') to dtype('bool') according to the rule 'safe'");
            CheckRaises(fails, "np.any(b, where=uint8 mask) (TypeError)", () => np.any(b, where: uint8Mask),
                        "Cannot cast array data from dtype('uint8') to dtype('bool') according to the rule 'safe'");
            CheckRaises(fails, "b.all(where=int64 mask) (TypeError)", () => b.all(where: int64Mask),
                        "Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
            CheckRaises(fails, "b.any(where=float64 mask) (TypeError)", () => b.any(where: float64Mask),
                        "Cannot cast array data from dtype('float64') to dtype('bool') according to the rule 'safe'");
            Check(fails, "np.all(b, where=bool mask) (already right)", () => np.all(b, where: M()), "bool () True");
            AssertNoMismatch(fails, "NumPy casts where= to bool with the 'safe' rule, which only bool passes");
        }

        // ======================================================================================================
        //  Helpers
        // ======================================================================================================

        /// <summary>The <c>(2, 3)</c> float64 array <c>[[1, 5, 3], [4, 2, 6]]</c> most cases reduce.</summary>
        /// <returns>A fresh C-contiguous array, so no case can see another case's writes.</returns>
        private static NDArray A() => np.array(new[] { 1.0, 5.0, 3.0, 4.0, 2.0, 6.0 }).reshape(2, 3);

        /// <summary>The <c>(2, 3)</c> bool mask <c>[[T, F, T], [F, T, T]]</c> the <c>where=</c> cases use.</summary>
        /// <returns>A fresh C-contiguous bool array.</returns>
        private static NDArray M() => np.array(new[] { true, false, true, false, true, true }).reshape(2, 3);

        /// <summary>The <c>(2, 2)</c> float64 array <c>[[1, nan], [nan, 3]]</c> the nanmax/nanmin <c>out=</c> cases use.</summary>
        /// <returns>A fresh C-contiguous array.</returns>
        private static NDArray Nans() => np.array(new[] { 1.0, double.NaN, double.NaN, 3.0 }).reshape(2, 2);

        /// <summary>
        ///     Calls the <c>np.*</c> function <paramref name="method"/> through its NumPy-shaped overload
        ///     (<see cref="NpParameters"/>), passing <paramref name="args"/> by name.
        /// </summary>
        /// <param name="method">The function name (<c>max</c>, <c>amin</c>, <c>nanmax</c>, …).</param>
        /// <param name="args">The arguments by NumPy name; every parameter not named takes its declared default.</param>
        /// <returns>What the function returns.</returns>
        /// <exception cref="MissingMethodException">No overload has NumPy's parameters — the state this file pins.</exception>
        /// <exception cref="ArgumentException">An argument name is not one of NumPy's parameters, or a value cannot be
        /// converted to its parameter's type (a bug in the test, not in NumSharp).</exception>
        private static NDArray Np(string method, params (string Name, object Value)[] args)
            => Call(typeof(np), null, method, NpParameters, args);

        /// <summary>
        ///     Calls the <see cref="NDArray"/> method <paramref name="method"/> on <paramref name="self"/> through its
        ///     NumPy-shaped overload (<see cref="NdArrayParameters"/>), passing <paramref name="args"/> by name.
        /// </summary>
        /// <param name="self">The array the method is called on.</param>
        /// <param name="method">The method name (<c>max</c> or <c>min</c>).</param>
        /// <param name="args">The arguments by NumPy name; every parameter not named takes its declared default.</param>
        /// <returns>What the method returns.</returns>
        /// <exception cref="MissingMethodException">No overload has NumPy's parameters — the state this file pins.</exception>
        /// <exception cref="ArgumentException">An argument name is not one of NumPy's parameters, or a value cannot be
        /// converted to its parameter's type.</exception>
        private static NDArray Nd(NDArray self, string method, params (string Name, object Value)[] args)
            => Call(typeof(NDArray), self, method, NdArrayParameters, args);

        /// <summary>
        ///     Finds the overload of <paramref name="owner"/>.<paramref name="method"/> whose parameter names are
        ///     exactly <paramref name="numpyParameters"/> and invokes it with <paramref name="args"/>. Binding by
        ///     NumPy's names (instead of by C# overload resolution) is what lets this file compile before the
        ///     overload exists and check it, unchanged, after.
        /// </summary>
        /// <param name="owner">The declaring type: <see cref="np"/> for the functions, <see cref="NDArray"/> for the methods.</param>
        /// <param name="target">The instance for a method, or <c>null</c> for a static function.</param>
        /// <param name="method">The member name.</param>
        /// <param name="numpyParameters">NumPy's parameter names in NumPy's order — the overload to bind.</param>
        /// <param name="args">The arguments by name.</param>
        /// <returns>The member's return value.</returns>
        /// <exception cref="MissingMethodException">No overload has exactly <paramref name="numpyParameters"/>.</exception>
        /// <exception cref="ArgumentException">An argument name is not in <paramref name="numpyParameters"/>, a
        /// parameter without a default was not given, or a value cannot be converted to its parameter's type.</exception>
        /// <remarks>
        ///     The member's own exception is rethrown with its original stack (not wrapped in
        ///     <see cref="TargetInvocationException"/>), so the error cases compare NumSharp's real message.
        /// </remarks>
        private static NDArray Call(Type owner, object target, string method, string[] numpyParameters, (string Name, object Value)[] args)
        {
            var flags = BindingFlags.Public | (target == null ? BindingFlags.Static : BindingFlags.Instance);
            var form = owner.GetMethods(flags).FirstOrDefault(m => m.Name == method
                && m.GetParameters().Select(p => p.Name).SequenceEqual(numpyParameters));
            if (form == null)
                throw new MissingMethodException(
                    $"{(target == null ? "np" : "ndarray")}.{method} has no overload ({string.Join(", ", numpyParameters)})");

            // A misspelled name would otherwise fall back to the parameter's default and pass for the wrong reason.
            foreach (var (name, _) in args)
                if (Array.IndexOf(numpyParameters, name) < 0)
                    throw new ArgumentException($"'{name}' is not one of NumPy's parameters of {method}");

            var parameters = form.GetParameters();
            var values = new object[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                int k = Array.FindIndex(args, x => x.Name == parameters[i].Name);
                if (k >= 0)
                    values[i] = ConvertArgument(args[k].Value, parameters[i].ParameterType);
                else if (parameters[i].HasDefaultValue)
                    values[i] = parameters[i].DefaultValue;
                else
                    throw new ArgumentException($"{method}: '{parameters[i].Name}' has no default and was not given");
            }

            try
            {
                return (NDArray)form.Invoke(target, values);
            }
            catch (TargetInvocationException e) when (e.InnerException != null)
            {
                // Unwrap so CheckRaises sees NumSharp's own exception and message.
                ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                throw;
            }
        }

        /// <summary>
        ///     Converts a test value to the type the NumPy-shaped overload declares for that parameter, so the cases do
        ///     not depend on how an implementation types <c>initial</c> (<c>object</c>, <c>double?</c>,
        ///     <c>NDArray</c>) or <c>where</c>.
        /// </summary>
        /// <param name="value">The value as the case wrote it; <c>null</c> is NumPy's <c>None</c>.</param>
        /// <param name="target">The parameter type.</param>
        /// <returns><paramref name="value"/> itself when it already fits (an <c>object</c> parameter receives the boxed
        /// C# literal and so keeps its int-vs-float kind), else the converted value.</returns>
        /// <exception cref="ArgumentException">The value cannot become a <paramref name="target"/>.</exception>
        private static object ConvertArgument(object value, Type target)
        {
            if (value is null || target.IsInstanceOfType(value))
                return value;
            var underlying = Nullable.GetUnderlyingType(target);
            if (underlying != null)
                return ConvertArgument(value, underlying);
            if (target == typeof(NDArray))
                return np.asanyarray(value);
            if (value is IConvertible && (target.IsPrimitive || target == typeof(decimal)))
                return Convert.ChangeType(value, target, CultureInfo.InvariantCulture);
            throw new ArgumentException($"cannot pass a {value.GetType().Name} as a {target.Name} parameter");
        }

        /// <summary>
        ///     Records a line when <paramref name="owner"/> has no overload of <paramref name="method"/> with exactly
        ///     <paramref name="expected"/> as its parameter names, or when that overload makes a NumPy keyword
        ///     mandatory.
        /// </summary>
        /// <param name="mismatches">Collects one line per failing function.</param>
        /// <param name="owner">The declaring type.</param>
        /// <param name="method">The member name.</param>
        /// <param name="isStatic">True for an <c>np.*</c> function, false for an <see cref="NDArray"/> method.</param>
        /// <param name="expected">NumPy's parameter names in NumPy's order.</param>
        /// <param name="requiredCount">How many leading parameters NumPy requires (<c>a</c> for a function, none for a
        /// method); every parameter after them must have a default.</param>
        private static void CheckSignature(List<string> mismatches, Type owner, string method, bool isStatic, string[] expected, int requiredCount)
        {
            var flags = BindingFlags.Public | (isStatic ? BindingFlags.Static : BindingFlags.Instance);
            var overloads = owner.GetMethods(flags).Where(m => m.Name == method).ToArray();
            string label = $"{(isStatic ? "np" : "ndarray")}.{method}";
            var form = overloads.FirstOrDefault(m => m.GetParameters().Select(p => p.Name).SequenceEqual(expected));
            if (form == null)
            {
                mismatches.Add($"{label}: no overload ({string.Join(", ", expected)}); NumSharp has "
                               + string.Join(" | ", overloads.Select(m => $"({string.Join(", ", m.GetParameters().Select(p => p.Name))})")));
                return;
            }

            var mandatory = form.GetParameters().Skip(requiredCount).Where(p => !p.HasDefaultValue).Select(p => p.Name).ToArray();
            if (mandatory.Length > 0)
                mismatches.Add($"{label}: NumPy defaults every keyword, but {string.Join(", ", mandatory)} has no default");
        }

        /// <summary>
        ///     Runs one value case and records a line when its <see cref="Describe"/> string differs from NumPy's. An
        ///     exception is recorded as a mismatch too, so one throwing case cannot hide the state of the others.
        /// </summary>
        /// <param name="mismatches">Collects one line per failing case; the test asserts it is empty.</param>
        /// <param name="name">The case label: the NumPy call it mirrors.</param>
        /// <param name="call">The NumSharp call under test.</param>
        /// <param name="expected">NumPy 2.4.2's result in <see cref="Describe"/>'s format.</param>
        private static void Check(List<string> mismatches, string name, Func<NDArray> call, string expected)
        {
            string actual;
            try
            {
                actual = Describe(call());
            }
            catch (Exception e)
            {
                actual = $"threw {e.GetType().Name}: {e.Message}";
            }

            if (actual != expected)
                mismatches.Add($"{name}: expected [{expected}] but NumSharp returned [{actual}]");
        }

        /// <summary>
        ///     Runs one <c>out=</c> case: the call must return <paramref name="out"/> ITSELF (NumPy's <c>r is out</c>)
        ///     and leave NumPy's result in it.
        /// </summary>
        /// <param name="mismatches">Collects one line per failing case.</param>
        /// <param name="name">The case label.</param>
        /// <param name="call">The call under test; it receives <paramref name="out"/> to pass as <c>out=</c>.</param>
        /// <param name="out">A fresh destination of the shape and dtype the case names.</param>
        /// <param name="expected">NumPy 2.4.2's content of the out array, in <see cref="Describe"/>'s format.</param>
        private static void CheckOut(List<string> mismatches, string name, Func<NDArray, NDArray> call, NDArray @out, string expected)
        {
            string actual;
            try
            {
                var result = call(@out);
                // A fresh array with the right values is not NumPy's contract: the caller holds `out`, not the return.
                actual = ReferenceEquals(result, @out) ? Describe(@out) : $"a new array ({Describe(result)}), not out";
            }
            catch (Exception e)
            {
                actual = $"threw {e.GetType().Name}: {e.Message}";
            }

            if (actual != expected)
                mismatches.Add($"{name}: expected out = [{expected}] but NumSharp returned [{actual}]");
        }

        /// <summary>
        ///     Runs one error case: the call must throw with NumPy's MESSAGE (the exception type is not compared; see
        ///     the class remarks).
        /// </summary>
        /// <param name="mismatches">Collects one line per failing case.</param>
        /// <param name="name">The case label; it names NumPy's exception type.</param>
        /// <param name="call">The call under test.</param>
        /// <param name="expectedMessage">NumPy 2.4.2's <c>str(e)</c>, verbatim.</param>
        private static void CheckRaises(List<string> mismatches, string name, Func<NDArray> call, string expectedMessage)
        {
            string actual;
            try
            {
                actual = $"returned [{Describe(call())}]";
            }
            catch (Exception e)
            {
                if (e.Message == expectedMessage)
                    return;
                actual = $"threw {e.GetType().Name}: {e.Message}";
            }

            mismatches.Add($"{name}: expected NumPy's error [{expectedMessage}] but NumSharp {actual}");
        }

        /// <summary>
        ///     Fails the test when any check recorded a mismatch, listing EVERY line, so a partial implementation shows
        ///     exactly which cases still differ.
        /// </summary>
        /// <param name="mismatches">The recorded lines; empty means every case matched NumPy.</param>
        /// <param name="because">The NumPy rule the cases check, shown in the failure message.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, when <paramref name="mismatches"/>
        /// is not empty.</exception>
        private static void AssertNoMismatch(List<string> mismatches, string because)
        {
            string.Join(Environment.NewLine, mismatches).Should().BeEmpty(because);
        }

        /// <summary>
        ///     Renders a result as <c>dtype (shape) values…</c>: the NumPy dtype name, the shape in tuple form
        ///     (<c>()</c>, <c>(3,)</c>, <c>(2,1)</c>), then every element in logical C order — float64 as 16 hex
        ///     digits, float32 as 8, any other dtype as its value (<c>True</c>/<c>False</c> for bool, the integer for
        ///     an integer dtype). Raw float bits make a <c>-0.0</c> or a 1-ULP difference visible.
        /// </summary>
        /// <param name="r">The result to render.</param>
        /// <returns>The rendering; a float NaN element is written as the token <c>nan</c>, because no case here is
        /// about NaN bits.</returns>
        private static string Describe(NDArray r)
        {
            string dims = string.Join(",", r.shape);
            string shape = r.ndim == 1 ? $"({dims},)" : $"({dims})";
            // flatten() copies in logical C order, the order NumPy's reshape(-1) printed the expected values in; a 0-d
            // result has one element, so a reshape view of it is enough.
            var flat = r.ndim == 0 ? r.reshape(1) : r.flatten();
            var parts = new List<string>();
            for (long i = 0; i < flat.size; i++)
            {
                switch (flat.typecode)
                {
                    case NPTypeCode.Double:
                    {
                        double v = flat.GetAtIndex<double>(i);
                        parts.Add(double.IsNaN(v) ? "nan" : BitConverter.DoubleToUInt64Bits(v).ToString("x16"));
                        break;
                    }
                    case NPTypeCode.Single:
                    {
                        float v = flat.GetAtIndex<float>(i);
                        parts.Add(float.IsNaN(v) ? "nan" : BitConverter.SingleToUInt32Bits(v).ToString("x8"));
                        break;
                    }
                    default:
                        parts.Add(Convert.ToString(flat.GetAtIndex(i), CultureInfo.InvariantCulture));
                        break;
                }
            }

            return $"{r.dtype.name} {shape} {string.Join(" ", parts)}";
        }
    }
}
