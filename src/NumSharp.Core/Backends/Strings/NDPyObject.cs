using System;
using System.Collections;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Printing;
using NumSharp.Utilities;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The few CPython object protocols NumPy's <c>StringDType</c> calls on an arbitrary Python object, applied to the C#
    ///     value that stands in for it: <c>str()</c>, <c>repr()</c>, <c>PyObject_RichCompareBool(a, b, Py_EQ)</c>, a hash
    ///     consistent with that equality, and the <c>na != na</c> truth test that marks a NaN-like missing-data object.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>The C# boundary map</b> (the house NEP 50 map, plus the string stores' rule for <see cref="char"/>):
    ///     <see langword="null"/> is <c>None</c>; <see cref="bool"/>, the integer primitives and <see cref="BigInteger"/> are
    ///     a Python <c>int</c> (<c>bool</c> prints <c>True</c> / <c>False</c>); <see cref="float"/> and <see cref="double"/>
    ///     are a Python <c>float</c>; <see cref="Complex"/> is a Python <c>complex</c>; <see cref="string"/> is a Python
    ///     <c>str</c> and so is a <see cref="char"/> (one UTF-16 unit — the rule every NumSharp string store follows);
    ///     <c>byte[]</c> is <c>bytes</c>; <see cref="Half"/> is the NumPy scalar <c>np.float16</c> (its repr is
    ///     <c>np.float16(1.5)</c>); <see cref="decimal"/> (no NumPy analog) prints like the <c>float64</c> it converts to;
    ///     an <see cref="NDArray"/> prints as <c>str(array)</c> / <c>repr(array)</c>; a value tuple is a Python
    ///     <c>tuple</c>, any other <see cref="IList"/> a <c>list</c>, an <see cref="IDictionary"/> a <c>dict</c>; anything
    ///     else is an opaque object whose <c>str()</c> and <c>repr()</c> are its <see cref="object.ToString"/>.
    ///     </para>
    ///     <para>
    ///     <b>Why it exists.</b> A <c>StringDType</c> descriptor keeps its <c>na_object</c> as a live Python object and runs
    ///     these protocols on it: <c>repr()</c> in the descriptor's repr (<c>StringDType(na_object='NA')</c>), <c>str()</c>
    ///     for the text an NA becomes in a cast to <c>U</c> / <c>S</c> (<c>'None'</c>, <c>'nan'</c>), equality for descriptor
    ///     equality and for finding a common instance, and the <c>!=</c> test for NaN-likeness. Keeping them in one place
    ///     makes every caller agree on what a C# value means to NumPy.
    ///     </para>
    /// </remarks>
    internal static class NDPyObject
    {
        /// <summary>
        ///     Python's <c>str(obj)</c> under the boundary map (see the class remarks): <c>'None'</c>, <c>'True'</c>,
        ///     <c>'5'</c>, <c>'1.5'</c>, <c>'nan'</c>, <c>'(1+2j)'</c>, the text itself, <c>"b'x'"</c> (Python's
        ///     <c>str(bytes)</c> is its repr), <c>'1.5'</c> for <c>np.float16(1.5)</c>.
        /// </summary>
        /// <param name="obj">The C# value standing in for a Python object.</param>
        /// <returns>The text <c>str()</c> returns.</returns>
        public static string Str(object obj)
        {
            switch (obj)
            {
                case null:
                    return "None";
                case string s:
                    return s;
                case char c:
                    return c.ToString();
                case byte[] b:
                    return PyRepr.Bytes(b); // bytes.__str__ is bytes.__repr__ (no -b warning in a default interpreter)
                case Half h:
                    return FormatFloat((double)h, FloatKind.Half); // np.float16's str: its own shortest digits
                case NDArray nd:
                    return nd.ToString(false);
                default:
                    if (TryFormatNumber(obj, out var text))
                        return text;
                    if (obj is IDictionary || obj is IList || obj is ITuple)
                        return Repr(obj); // a container's str() is its repr()
                    return obj.ToString();
            }
        }

        /// <summary>
        ///     Python's <c>repr(obj)</c> under the boundary map: <c>None</c>, <c>True</c>, <c>5</c>, <c>1.5</c>, <c>nan</c>,
        ///     <c>(1+2j)</c>, <c>'NA'</c> (CPython's quote choice and escapes), <c>b'\x80'</c>, <c>np.float16(1.5)</c>,
        ///     <c>(1,)</c>, <c>[]</c>, <c>{}</c>.
        /// </summary>
        /// <param name="obj">The C# value standing in for a Python object.</param>
        /// <returns>The text <c>repr()</c> returns.</returns>
        public static string Repr(object obj)
        {
            switch (obj)
            {
                case null:
                    return "None";
                case string s:
                    return PyRepr.Str(s);
                case char c:
                    return PyRepr.Str(c.ToString());
                case byte[] b:
                    return PyRepr.Bytes(b);
                case Half h:
                    // A NumPy scalar's repr names its type (NumPy 2 scalar reprs): np.float16(1.5), np.float16(nan).
                    return "np.float16(" + FormatFloat((double)h, FloatKind.Half) + ")";
                case NDArray nd:
                    return nd.ToString(true);
                case ITuple tuple:
                {
                    // Python's tuple repr: a one-element tuple keeps its trailing comma, the empty tuple is "()".
                    var sb = new StringBuilder("(");
                    for (int i = 0; i < tuple.Length; i++)
                    {
                        if (i > 0)
                            sb.Append(", ");
                        sb.Append(Repr(tuple[i]));
                    }
                    if (tuple.Length == 1)
                        sb.Append(',');
                    return sb.Append(')').ToString();
                }
                case IDictionary dict:
                {
                    var sb = new StringBuilder("{");
                    bool first = true;
                    foreach (DictionaryEntry e in dict)
                    {
                        if (!first)
                            sb.Append(", ");
                        first = false;
                        sb.Append(Repr(e.Key)).Append(": ").Append(Repr(e.Value));
                    }
                    return sb.Append('}').ToString();
                }
                case IList list:
                    return "[" + string.Join(", ", list.Cast<object>().Select(Repr)) + "]";
                default:
                    // A Python number's repr equals its str (int, float, complex, bool).
                    if (TryFormatNumber(obj, out var text))
                        return text;
                    return obj.ToString();
            }
        }

        /// <summary>
        ///     NumPy's <c>na_eq_cmp</c> — the equality a <c>StringDType</c> uses for two missing-data objects: the same object
        ///     (NumPy's identity shortcut, which is how <c>None</c> equals <c>None</c>); two Python floats that are both NaN
        ///     (<c>float('nan')</c> and <c>np.nan</c> are different objects but the same missing value); otherwise Python's
        ///     <c>==</c> with errors read as "not equal" (<c>PyObject_RichCompareBool</c>, a raised error cleared to 0).
        /// </summary>
        /// <param name="a">One missing-data object (null is <c>None</c>).</param>
        /// <param name="b">The other.</param>
        /// <returns>True when NumPy considers the two objects the same missing value.</returns>
        /// <remarks>
        ///     Python's <c>==</c> compares numbers by VALUE across types (<c>5 == 5.0</c>, <c>True == 1</c>,
        ///     <c>(1+0j) == 1</c>, <c>0.0 == -0.0</c>), so the numeric branch compares exactly, never through a C# boxed
        ///     <see cref="object.Equals(object)"/> (which is false for an <c>int</c> against a <c>double</c>). A NumPy
        ///     <c>np.float16</c> NaN is NOT a Python float, so two of them differ unless they are the same object; and a
        ///     comparison that involves a NumPy scalar follows NumPy's scalar rules (see <see cref="PyEq"/>).
        /// </remarks>
        public static bool Equal(object a, object b)
        {
            if (ReferenceEquals(a, b))
                return true; // catches None and other singletons like Pandas.NA
            if (a is null || b is null)
                return false;
            if (IsPyFloat(a) && IsPyFloat(b) && double.IsNaN(ToPyFloat(a)) && double.IsNaN(ToPyFloat(b)))
                return true; // nan check catches np.nan and float('nan')
            return PyEq(a, b);
        }

        /// <summary>
        ///     Python's <c>a == b</c> as <c>PyObject_RichCompareBool</c> answers it (the same object is equal to itself; an
        ///     error reads as "not equal"), under the boundary map.
        /// </summary>
        /// <param name="a">One value (null is <c>None</c>).</param>
        /// <param name="b">The other.</param>
        /// <returns>True when Python's <c>==</c> is truthy.</returns>
        /// <remarks>
        ///     <list type="bullet">
        ///     <item>Python numbers (<see cref="bool"/>, the integers, <see cref="BigInteger"/>, <see cref="float"/>,
        ///     <see cref="double"/>, <see cref="Complex"/>) compare EXACTLY, as CPython does: <c>2**53 + 1 != 2.0**53</c>,
        ///     a NaN equals nothing.</item>
        ///     <item>A comparison involving a NumPy scalar — <see cref="Half"/> (<c>np.float16</c>) or
        ///     <see cref="decimal"/> (NumSharp's own type, compared as the <c>np.float64</c> it converts to) — follows
        ///     NumPy's scalar rules: NEP 50 converts the Python number to the scalar's dtype first, so
        ///     <c>np.float16(2048) == 2049</c> and <c>np.float16(0.1) == 0.1</c> are True, <c>np.float16(inf) == 2**70</c>
        ///     is True, and a Python complex meets <c>np.float16</c> as complex64
        ///     (<c>np.float16(0.1) == 0.1+0j</c> is False).</item>
        ///     <item>Text compares ordinally (a <see cref="char"/> is a one-character <c>str</c>); <c>bytes</c> by content
        ///     and never equal to text; an <see cref="NDArray"/> only to itself (its <c>==</c> is an array whose truth
        ///     value raises past one element).</item>
        ///     <item>A tuple (<see cref="ITuple"/>) equals a tuple, a list (<see cref="IList"/>) a list, a dict
        ///     (<see cref="IDictionary"/>) a dict, item by item with Python's container rule: the same object, or
        ///     <c>==</c> (so a list holding one NaN equals itself only through the identity of the NaN object).</item>
        ///     </list>
        /// </remarks>
        internal static bool PyEq(object a, object b)
        {
            if (ReferenceEquals(a, b))
                return true;
            if (a is null || b is null)
                return false;
            if (IsNumber(a) && IsNumber(b))
                return NumbersEqual(a, b);
            // A NumPy scalar against a list / tuple: NumPy converts the sequence to an array and compares elementwise, so
            // the outcome is the truth value of that array — only a single element has one (more or none raises, read
            // as "not equal"). The comparison is the ARRAY's: strong promotion, not the weak scalar rule.
            if (IsNumPyScalar(a) && IsSequence(b))
                return ScalarEqualsSequence(a, b);
            if (IsNumPyScalar(b) && IsSequence(a))
                return ScalarEqualsSequence(b, a);
            switch (a)
            {
                case string sa:
                    return b is string sb ? string.Equals(sa, sb, StringComparison.Ordinal) : b is char cb && sa.Length == 1 && sa[0] == cb;
                case char ca:
                    return b is char cb2 ? ca == cb2 : b is string sb2 && sb2.Length == 1 && sb2[0] == ca;
                case byte[] ba:
                    return b is byte[] bb && ba.AsSpan().SequenceEqual(bb);
                case NDArray:
                    // ndarray == x is an array; its truth value raises for more than one element, which
                    // PyObject_RichCompareBool clears to "not equal". Only the identity above makes two arrays equal.
                    return false;
                case ITuple ta:
                {
                    if (b is not ITuple tb || ta.Length != tb.Length)
                        return false;
                    for (int i = 0; i < ta.Length; i++)
                        if (!PyEq(ta[i], tb[i]))
                            return false;
                    return true;
                }
                case IDictionary da:
                {
                    if (b is not IDictionary db || da.Count != db.Count)
                        return false;
                    // Python's dict ==: every key of one is a key of the other (by ==) and maps to an equal value. A
                    // linear key search per entry: sentinel dicts are tiny, and C#'s own key equality is not Python's.
                    foreach (DictionaryEntry ea in da)
                    {
                        bool found = false;
                        foreach (DictionaryEntry eb in db)
                        {
                            if (PyEq(ea.Key, eb.Key))
                            {
                                found = PyEq(ea.Value, eb.Value);
                                break;
                            }
                        }
                        if (!found)
                            return false;
                    }
                    return true;
                }
                case IList la:
                {
                    // A list (byte[] was taken above as bytes); never equal to a tuple, a string or bytes.
                    if (b is not IList lb || b is byte[] || la.Count != lb.Count)
                        return false;
                    for (int i = 0; i < la.Count; i++)
                        if (!PyEq(la[i], lb[i]))
                            return false;
                    return true;
                }
            }
            return a.Equals(b);
        }

        /// <summary>
        ///     A hash consistent with <see cref="Equal"/> and <see cref="PyEq"/>: equal objects hash alike. EVERY number
        ///     hashes to one value, because NumPy's scalar comparisons make numeric equality non-transitive
        ///     (<c>np.float16(2048) == 2048</c> and <c>== 2049</c>, while <c>2048 != 2049</c>), which no finer numeric hash
        ///     can follow; text hashes by its ordinal hash (a char as its one-character string), bytes by content, None to a
        ///     constant, tuples and lists by their items, dicts by their entries in any order.
        /// </summary>
        /// <param name="obj">A missing-data object (null is <c>None</c>).</param>
        /// <returns>The hash.</returns>
        /// <remarks>
        ///     NumPy hashes a <c>StringDType</c> as <c>hash((coerce, na_object))</c>, which RAISES for an unhashable sentinel
        ///     (a list, a dict) and gives <c>np.float16(2048)</c> and <c>2049</c> different hashes although the
        ///     descriptors compare equal. A C# hash must not throw and must agree with equality, so it is coarser instead:
        ///     descriptors with different numeric sentinels collide, which costs nothing for the few descriptors a program
        ///     holds.
        /// </remarks>
        public static int Hash(object obj)
        {
            switch (obj)
            {
                case null:
                    return 0x4E4F4E45; // "NONE"
                case string s:
                    return s.Length == 1 ? s[0].GetHashCode() : StringComparer.Ordinal.GetHashCode(s);
                case char c:
                    return c.GetHashCode(); // equal to the one-character string's hash above
                case byte[] b:
                {
                    var h = new HashCode();
                    h.AddBytes(b);
                    return h.ToHashCode();
                }
                case NDArray nd:
                    return System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(nd); // equal only to itself
                case var seq when IsSequence(seq) && HasSingleNumericLeaf(seq):
                    // A NumPy scalar equals a one-leaf numeric sequence (np.float16(1) == (1,)), so such a sequence must
                    // hash like the numbers.
                    return 0x4E554D21;
                case ITuple t:
                {
                    var h = new HashCode();
                    h.Add(0x54555031); // "TUP1": a tuple never equals a list holding the same items
                    for (int i = 0; i < t.Length; i++)
                        h.Add(Hash(t[i]));
                    return h.ToHashCode();
                }
                case IDictionary d:
                {
                    // Order-independent: two dicts with the same entries are equal whatever their enumeration order.
                    int acc = 0x44494354; // "DICT"
                    foreach (DictionaryEntry e in d)
                        acc ^= HashCode.Combine(Hash(e.Key), Hash(e.Value));
                    return acc;
                }
                case IList l:
                {
                    var h = new HashCode();
                    h.Add(0x4C495354); // "LIST"
                    foreach (var item in l)
                        h.Add(Hash(item));
                    return h.ToHashCode();
                }
            }
            if (IsNumber(obj))
                return 0x4E554D21; // "NUM!": see the summary — every number, NaN included
            return obj.GetHashCode();
        }

        /// <summary>
        ///     NumPy's NaN-likeness test of a missing-data object (<c>new_stringdtype_instance</c>): the object is NaN-like
        ///     when <c>na != na</c> is truthy or raises — a Python float NaN, an <c>np.float16</c> NaN, a complex with a NaN
        ///     part, an array whose <c>!=</c> has no single truth value.
        /// </summary>
        /// <param name="obj">The missing-data object (null is <c>None</c>, which is not NaN-like).</param>
        /// <returns>True for a NaN-like object.</returns>
        public static bool IsNaNLike(object obj)
        {
            switch (obj)
            {
                case null:
                    return false;
                case double d:
                    return double.IsNaN(d);
                case float f:
                    return float.IsNaN(f);
                case Half h:
                    return Half.IsNaN(h);
                case Complex z:
                    return double.IsNaN(z.Real) || double.IsNaN(z.Imaginary);
                case NDArray nd:
                    // `nd != nd` is an array: a single element has a truth value (NaN-like when that element is), more
                    // elements raise — and a raise counts as NaN-like (pd.NA's behavior).
                    return nd.size != 1 || IsNaNLike(nd.Storage.GetAtIndex(0));
                default:
                    return false;
            }
        }

        // ---------------------------------------------------------------------------------------------------------------
        // numbers
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>Whether a value is a Python <c>float</c> in the boundary map (C# <see cref="float"/> or <see cref="double"/>).</summary>
        /// <param name="obj">The value.</param>
        /// <returns>True for a Python float.</returns>
        private static bool IsPyFloat(object obj) => obj is double || obj is float;

        /// <summary>The value of a Python float (<see cref="IsPyFloat"/> must hold).</summary>
        /// <param name="obj">A boxed <see cref="float"/> or <see cref="double"/>.</param>
        /// <returns>The value as a double.</returns>
        private static double ToPyFloat(object obj) => obj is float f ? f : (double)obj;

        /// <summary>Whether a value is a number of the boundary map (any Python / NumPy numeric scalar).</summary>
        /// <param name="obj">The value.</param>
        /// <returns>True for bool, the integer primitives, BigInteger, float, double, Half, decimal and Complex.</returns>
        private static bool IsNumber(object obj)
            => obj is bool || obj is sbyte || obj is byte || obj is short || obj is ushort || obj is int || obj is uint
               || obj is long || obj is ulong || obj is BigInteger || obj is float || obj is double || obj is Half
               || obj is decimal || obj is Complex;

        /// <summary>One component of a number: an exact integer, or a float (finite or not).</summary>
        /// <param name="IsInteger">True when the component is an exact integral value.</param>
        /// <param name="Integer">The integral value (when <paramref name="IsInteger"/>).</param>
        /// <param name="Float">The value as a double (always set).</param>
        private readonly record struct Part(bool IsInteger, BigInteger? Integer, double Float);

        /// <summary>
        ///     Splits a number into its real and imaginary components, the real one exact for an integer and for a float
        ///     with an integral value (Python compares <c>2**53 + 1 == 2.0**53</c> as False, which a double comparison would
        ///     get wrong).
        /// </summary>
        /// <param name="obj">A number (<see cref="IsNumber"/> holds).</param>
        /// <returns>The real component and the imaginary value (0 for a real number).</returns>
        private static (Part re, double im) ToComplexParts(object obj)
        {
            switch (obj)
            {
                case bool v:
                    return (new Part(true, v ? BigInteger.One : BigInteger.Zero, v ? 1 : 0), 0);
                case BigInteger big:
                    return (new Part(true, big, (double)big), 0);
                case sbyte or short or int or long:
                {
                    long v = Convert.ToInt64(obj, CultureInfo.InvariantCulture);
                    return (new Part(true, v, v), 0);
                }
                case byte or ushort or uint or ulong:
                {
                    ulong v = Convert.ToUInt64(obj, CultureInfo.InvariantCulture);
                    return (new Part(true, v, v), 0);
                }
                case Complex z:
                    return (FloatPart(z.Real), z.Imaginary);
                case decimal m:
                    return (FloatPart(Converts.ToDouble(m)), 0);
                case Half h:
                    return (FloatPart((double)h), 0);
                case float f:
                    return (FloatPart(f), 0);
                default:
                    return (FloatPart((double)obj), 0);
            }
        }

        /// <summary>A float component, made exact when its value is a finite integer.</summary>
        /// <param name="d">The value.</param>
        /// <returns>The component.</returns>
        private static Part FloatPart(double d)
            => double.IsFinite(d) && Math.Floor(d) == d ? new Part(true, new BigInteger(d), d) : new Part(false, null, d);

        /// <summary>
        ///     Python's numeric <c>==</c>: equal real components (exactly, for integral values) and equal imaginary parts —
        ///     or NumPy's scalar comparison when a NumPy scalar takes part (<see cref="NumPyScalarEqual"/>).
        /// </summary>
        /// <param name="a">A number.</param>
        /// <param name="b">A number.</param>
        /// <returns>True when the two numbers are equal in Python.</returns>
        private static bool NumbersEqual(object a, object b)
        {
            if (a is Half || b is Half || a is decimal || b is decimal)
                return NumPyScalarEqual(a, b);
            var (ra, ia) = ToComplexParts(a);
            var (rb, ib) = ToComplexParts(b);
            // A NaN imaginary part is unequal to everything (IEEE), which `ia == ib` already gives.
            if (!(ia == ib))
                return false;
            if (ra.IsInteger && rb.IsInteger)
                return ra.Integer == rb.Integer;
            // An exact integer never equals a non-integral value — a fraction, an infinity, a NaN (CPython compares int
            // and float exactly, so 2**1100 != inf even though the int's double conversion overflows to inf).
            if (ra.IsInteger != rb.IsInteger)
                return false;
            // Both non-integral floats (or NaN / inf): IEEE equality decides (NaN != NaN, inf == inf).
            return ra.Float == rb.Float;
        }

        /// <summary>
        ///     NumPy's comparison of a NumPy scalar (<see cref="Half"/> = <c>np.float16</c>; <see cref="decimal"/>, compared
        ///     as the <c>np.float64</c> it converts to) with another number: both operands are brought to the comparison's
        ///     dtype — float64 when a decimal takes part (NumPy promotes float16 with float64 to float64), else float16 —
        ///     a Python number converting as a WEAK scalar (NEP 50: <c>int → float(int) → dtype</c>, <c>float → dtype</c>),
        ///     and compared there with IEEE equality.
        /// </summary>
        /// <param name="a">A number, at least one of <paramref name="a"/> / <paramref name="b"/> a NumPy scalar.</param>
        /// <param name="b">The other number.</param>
        /// <returns>True when NumPy's <c>==</c> is True.</returns>
        /// <remarks>
        ///     A Python complex meets <c>np.float16</c> as complex64 (there is no complex32) and <c>np.float64</c> as
        ///     complex128. A Python int past the double range makes NumPy raise <c>OverflowError: int too large to convert
        ///     to float</c>, which <c>PyObject_RichCompareBool</c>'s caller reads as "not equal".
        /// </remarks>
        private static bool NumPyScalarEqual(object a, object b)
        {
            bool float64 = a is decimal || b is decimal;
            if (a is Complex || b is Complex)
            {
                var z = a is Complex za ? za : (Complex)b;
                double v = ScalarToDouble(a is Complex ? b : a);
                if (float64)
                    return z.Imaginary == 0 && z.Real == v; // complex128
                // complex64: the float16 value widens exactly to float32, the Python complex rounds to float32 parts.
                return (float)z.Imaginary == 0f && (float)z.Real == (float)v;
            }
            if (!TryWeakToDouble(a, out double da) || !TryWeakToDouble(b, out double db))
                return false; // NumPy's OverflowError, read as "not equal"
            if (float64)
                return da == db;
            // float16: every operand rounds ONCE from its double value to float16 (NumPy's npy_double_to_half), then
            // compares; the halves widen to float exactly, which keeps IEEE semantics (NaN unequal, -0 == +0).
            return (float)BitConverter.UInt16BitsToHalf(DirectILKernelGenerator.DoubleToHalfBits(da))
                   == (float)BitConverter.UInt16BitsToHalf(DirectILKernelGenerator.DoubleToHalfBits(db));
        }

        /// <summary>Whether a value is a NumPy scalar of the boundary map (<see cref="Half"/>, or <see cref="decimal"/> as <c>np.float64</c>).</summary>
        /// <param name="obj">The value.</param>
        /// <returns>True for a NumPy scalar.</returns>
        private static bool IsNumPyScalar(object obj) => obj is Half || obj is decimal;

        /// <summary>Whether a value is a Python list or tuple of the boundary map (not <c>bytes</c>, not text).</summary>
        /// <param name="obj">The value.</param>
        /// <returns>True for an <see cref="ITuple"/> or an <see cref="IList"/> other than <c>byte[]</c>.</returns>
        private static bool IsSequence(object obj) => obj is ITuple || (obj is IList && obj is not byte[]);

        /// <summary>Whether a list / tuple is a nest of one-element sequences ending in a number (<c>[1]</c>, <c>((2.5,),)</c>).</summary>
        /// <param name="sequence">A list or tuple (<see cref="IsSequence"/>).</param>
        /// <returns>True when NumPy would compare it with a scalar as that single number.</returns>
        private static bool HasSingleNumericLeaf(object sequence)
        {
            object current = sequence;
            while (IsSequence(current))
            {
                int count = current is ITuple t ? t.Length : ((IList)current).Count;
                if (count != 1)
                    return false;
                current = current is ITuple t1 ? t1[0] : ((IList)current)[0];
            }
            return IsNumber(current);
        }

        /// <summary>
        ///     NumPy's <c>scalar == sequence</c>: <c>np.asarray(sequence)</c> compared elementwise, then the result's truth
        ///     value. True only when the (nested) sequence holds exactly ONE leaf and that leaf equals the scalar under the
        ///     array's STRONG promotion (<see cref="StrongScalarEqual"/>).
        /// </summary>
        /// <param name="scalar">A NumPy scalar (<see cref="IsNumPyScalar"/>).</param>
        /// <param name="sequence">A list or tuple (<see cref="IsSequence"/>).</param>
        /// <returns>True when NumPy's comparison is truthy; false when it is falsy or raises (several leaves, no leaf, a
        ///     ragged nest, a non-numeric leaf, which makes an object array).</returns>
        private static bool ScalarEqualsSequence(object scalar, object sequence)
        {
            // The single leaf of a nest of one-element sequences ([[x]], [(x,)]): any wider level means several leaves.
            object current = sequence;
            while (IsSequence(current))
            {
                int count = current is ITuple t ? t.Length : ((IList)current).Count;
                if (count != 1)
                    return false; // no leaf, or more than one: the array's truth value raises
                current = current is ITuple t1 ? t1[0] : ((IList)current)[0];
            }
            return IsNumber(current) && StrongScalarEqual(scalar, current);
        }

        /// <summary>
        ///     The elementwise <c>==</c> of a NumPy scalar and one element of a sequence NumPy turned into an array:
        ///     both meet in <c>np.result_type</c> of the scalar and the array's dtype (strong promotion: a Python int is
        ///     int64, a float float64, a complex complex128, a bool bool). <c>np.float16</c> meets int64 / float64 in
        ///     float64, complex in complex128, bool and float16 in float16; <c>np.float64</c> (a decimal) meets everything
        ///     real in float64 and complex in complex128.
        /// </summary>
        /// <param name="scalar">A NumPy scalar (<see cref="IsNumPyScalar"/>).</param>
        /// <param name="element">A number.</param>
        /// <returns>True when the elementwise comparison is True.</returns>
        /// <remarks>A Python int past uint64 makes an OBJECT array, whose elementwise <c>==</c> is the scalar's own weak
        ///     comparison (<see cref="NumPyScalarEqual"/>).</remarks>
        private static bool StrongScalarEqual(object scalar, object element)
        {
            if (element is BigInteger big && (big < long.MinValue || big > ulong.MaxValue))
                return NumPyScalarEqual(scalar, element);
            double s = ScalarToDouble(scalar);
            if (element is Complex z)
                return z.Imaginary == 0 && z.Real == s; // complex128: the scalar widens exactly
            if (scalar is Half && (element is Half || element is bool))
                return TryWeakToDouble(element, out double e16) && (float)s == (float)e16; // float16 (both exact there)
            // float64: an int64 / uint64 element rounds to the nearest double, a float is one already.
            return TryWeakToDouble(element, out double e) && s == e;
        }

        /// <summary>The value of a NumPy scalar of the boundary map as a double (exact for both).</summary>
        /// <param name="scalar">A <see cref="Half"/> or a <see cref="decimal"/>.</param>
        /// <returns>The double.</returns>
        private static double ScalarToDouble(object scalar) => scalar is Half h ? (double)h : Converts.ToDouble((decimal)scalar);

        /// <summary>
        ///     A number of the boundary map as the double it converts to on its way into a NumPy scalar comparison: a Python
        ///     int correctly rounded (CPython's <c>float(int)</c>), a bool as 0 / 1, a float widened, a NumPy scalar exactly.
        /// </summary>
        /// <param name="obj">A non-complex number.</param>
        /// <param name="value">The double.</param>
        /// <returns>False for a Python int past the double range (NumPy raises <c>OverflowError</c> there).</returns>
        private static bool TryWeakToDouble(object obj, out double value)
        {
            switch (obj)
            {
                case Half h:
                    value = (double)h;
                    return true;
                case decimal m:
                    value = Converts.ToDouble(m);
                    return true;
                case double d:
                    value = d;
                    return true;
                case float f:
                    value = f;
                    return true;
                case bool v:
                    value = v ? 1 : 0;
                    return true;
                case BigInteger big:
                    value = PythonInt.ToDouble(big);
                    return !double.IsInfinity(value);
                case byte or ushort or uint or ulong:
                    // .NET 8's (double)ulong double-rounds at and above 2**63: round through Python's float(int).
                    value = PythonInt.ToDouble(Convert.ToUInt64(obj, CultureInfo.InvariantCulture));
                    return true;
                default:
                    value = Convert.ToInt64(obj, CultureInfo.InvariantCulture); // sbyte..long: (double)long rounds correctly
                    return true;
            }
        }

        /// <summary>
        ///     The Python / NumPy <c>str()</c> of a number of the boundary map (the formatter the string stores use), or false
        ///     for a value that is not a number.
        /// </summary>
        /// <param name="obj">The value.</param>
        /// <param name="text">The text, when it is a number.</param>
        /// <returns>True when <paramref name="obj"/> is a number.</returns>
        private static bool TryFormatNumber(object obj, out string text)
        {
            switch (obj)
            {
                case bool v:
                    text = v ? "True" : "False";
                    return true;
                case BigInteger big:
                    text = big.ToString(CultureInfo.InvariantCulture);
                    return true;
                case sbyte or short or int or long:
                    text = Convert.ToInt64(obj, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
                    return true;
                case byte or ushort or uint or ulong:
                    text = Convert.ToUInt64(obj, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
                    return true;
                case float f:
                    // A Python float: the float32 value widened, printed by the double's repr.
                    text = FormatFloat(f, FloatKind.Double);
                    return true;
                case double d:
                    text = FormatFloat(d, FloatKind.Double);
                    return true;
                case decimal m:
                    // NumSharp's decimal prints as the float64 scalar it converts to (ArrayFormatter.ScalarStr's rule).
                    text = FormatFloat(Converts.ToDouble(m), FloatKind.Double);
                    return true;
                case Complex z:
                {
                    Span<char> buf = stackalloc char[NDTextFormat.MaxTextLength];
                    text = buf.Slice(0, NDTextFormat.FormatComplex(z, buf)).ToString();
                    return true;
                }
                default:
                    text = null;
                    return false;
            }
        }

        /// <summary>The NumPy scalar <c>str()</c> of a float of the given kind (shortest digits, NumPy's positional window).</summary>
        /// <param name="d">The value.</param>
        /// <param name="kind">Whose digits and window to use.</param>
        /// <returns>The text.</returns>
        private static string FormatFloat(double d, FloatKind kind)
        {
            Span<char> buf = stackalloc char[NDTextFormat.MaxTextLength];
            return buf.Slice(0, NDTextFormat.FormatFloat(d, kind, buf)).ToString();
        }
    }
}
