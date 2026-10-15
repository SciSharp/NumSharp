using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Printing;

// =============================================================================
// NDPolyIndexArgument.cs — polyutils._as_int: Python's operator.index for numpy.polynomial's integer arguments
// =============================================================================
//
// numpy.polynomial reads every integer argument that must be EXACTLY an integer (the Vandermonde degrees, and the
// degree counts of the fitting family) with one helper (NumPy 2.4.2 numpy/polynomial/polyutils.py):
//
//     def _as_int(x, desc):
//         try:
//             return operator.index(x)
//         except TypeError as e:
//             raise TypeError(f"{desc} must be an integer, received {x}") from e
//
// so WHICH values pass is operator.index — the __index__ protocol — and the error's text interpolates the value with an
// f-string, i.e. format(x, '') — NOT str(x). Both are ported here for every C# value the house boundary maps
// (NDPolyNumber's header), probed against NumPy 2.4.2:
//
// operator.index(x)
//   * a Python int / bool / BigInteger is itself (True is 1); a NumPy integer scalar and a 0-d INTEGER array are their
//     value — char (NumSharp's uint16-proxied integer dtype) included;
//   * everything else raises: a Python float / complex (even an integral one: 2.0), a NumPy float / complex / bool
//     scalar (np.True_ has no __index__), a 0-d float / complex / bool array, any array of one or more dims (even one
//     element), a str, None, a list or tuple.
// format(x, '')
//   * a Python scalar is its str: an int's digits, a float's shortest repr (1.0, 1e+20, -0.0, inf, nan), a complex's
//     repr ((2+0j), 1j), True / False, None, a str itself;
//   * a NumPy scalar and a 0-d array go through the scalar's __format__, which formats float(self) / int(self) /
//     complex(self) / bool(self) — so np.float16(1000) reads 1000.0 (its str would be 1e+03) and np.float32(0.1)
//     0.10000000149011612;
//   * an array of one or more dims is its str (array_str): [2], [1 2], [2.], [[2]];
//   * a list / tuple is its repr, whose items are repr'd: [1, 'a', None, 2.5], (2,), [np.float16(2.0)],
//     (array([1.5]),) — a NumPy scalar item reads np.<dtype>(<str>) (np.float16(1e+03), np.complex128(2+0j),
//     np.True_), an array item array_repr.
// NumSharp's decimal has no NumPy analog: the house prints it as a float64 (ArrayFormatter.ScalarStr), and so does this.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     <c>polyutils._as_int</c> — Python's <c>operator.index</c> — and the <c>format(x, '')</c> / <c>repr(x)</c> its error
    ///     text interpolates (see the file header).
    /// </summary>
    internal static class PolyIndexArgument
    {
        /// <summary>
        ///     <c>polyutils._as_int(x, desc)</c>: the integer <paramref name="x"/> stands for, under Python's <c>operator.index</c>.
        /// </summary>
        /// <param name="x">The argument: a C# integer / bool / <see cref="BigInteger"/> / char, or a 0-d integer
        ///     <see cref="NDArray"/>; anything else raises.</param>
        /// <param name="desc">The argument's name in the error text (NumPy's <c>desc</c>, e.g. <c>"deg"</c>).</param>
        /// <returns>The integer (any size — the caller decides what is too large).</returns>
        /// <exception cref="TypeError">NumPy's <c>{desc} must be an integer, received {x}</c>, with <c>x</c> formatted as
        ///     NumPy's f-string does (<see cref="Format"/>).</exception>
        public static BigInteger AsInt(object x, string desc)
        {
            if (TryIndex(x, out var v))
                return v;
            throw new TypeError($"{desc} must be an integer, received {Format(x)}");
        }

        /// <summary>
        ///     Python's <c>operator.index(x)</c>: whether <paramref name="x"/> implements <c>__index__</c> under the house mapping,
        ///     and its value.
        /// </summary>
        /// <param name="x">The value.</param>
        /// <param name="value">The integer when the result is true.</param>
        /// <returns>True for a Python int / bool, a NumPy integer scalar (char), or a 0-d integer array.</returns>
        private static bool TryIndex(object x, out BigInteger value)
        {
            switch (x)
            {
                case bool b: value = b ? BigInteger.One : BigInteger.Zero; return true;   // bool is an int subclass: True is 1
                case sbyte v: value = v; return true;
                case byte v: value = v; return true;
                case short v: value = v; return true;
                case ushort v: value = v; return true;
                case int v: value = v; return true;
                case uint v: value = v; return true;
                case long v: value = v; return true;
                case ulong v: value = v; return true;
                case BigInteger v: value = v; return true;
                case char v: value = v; return true;   // NumSharp's char dtype: a NumPy integer scalar (np.uint16's __index__)
                case NDArray nd when nd.ndim == 0 && nd.typecode != NPTypeCode.Boolean && PolyTyping.IsIntLike(nd.typecode):
                    // ndarray.__index__ accepts a 0-d INTEGER array only ("only integer scalar arrays can be converted to a
                    // scalar index" otherwise — a 0-d bool array included); its element is a boxed CLR integer (or char).
                    return TryIndex(nd.GetAtIndex(0), out value);
            }
            value = default;
            return false;
        }

        /// <summary>
        ///     Python's <c>format(x, '')</c> — what an f-string interpolates — under the house mapping (see the file header).
        /// </summary>
        /// <param name="x">The value (may be null).</param>
        /// <returns>The text NumPy's message would carry.</returns>
        public static string Format(object x)
        {
            switch (x)
            {
                case null: return "None";
                case string s: return s;
                case NDArray nd:
                    // A 0-d array formats its scalar (float(self) / int(self) / complex(self) / bool(self) — exactly the boxed
                    // CLR value's rules below); any other array is its str (array_str, the house's byte-exact port).
                    return nd.ndim == 0 ? Format(nd.GetAtIndex(0)) : nd.ToString();
            }
            if (TryFormatScalar(x, out var text))
                return text;
            // A typed C# array / Memory<T> of a dtype is an ndarray (the house mapping): its str.
            if (PolySequence.IsArrayLike(x))
            {
                using var a = np.asanyarray(x);
                return a.ToString();
            }
            // A list / tuple: its repr (items repr'd).
            if (PolySequence.IsSequence(x))
                return Repr(x);
            return x.ToString() ?? string.Empty;
        }

        /// <summary>
        ///     Python's <c>repr(x)</c> — how an item of a list or tuple reads inside the container's str — under the house mapping.
        /// </summary>
        /// <param name="x">The value (may be null).</param>
        /// <returns>The repr text.</returns>
        public static string Repr(object x)
        {
            switch (x)
            {
                case null: return "None";
                case string s: return PythonInt.Repr(s);
                case NDArray nd: return nd.ToString(true);   // array_repr: array([1.5]), array(2.)
                case Half h: return NumPyScalarRepr(h, NPTypeCode.Half);
                case char c: return NumPyScalarRepr(c, NPTypeCode.Char);
                case decimal m: return NumPyScalarRepr(m, NPTypeCode.Decimal);
            }
            // A Python scalar's repr is its str (int, float, complex, bool).
            if (TryFormatScalar(x, out var text))
                return text;
            if (PolySequence.IsArrayLike(x))
            {
                using var a = np.asanyarray(x);
                return a.ToString(true);
            }
            if (PolySequence.IsSequence(x))
            {
                var items = PolySequence.Items(x);
                bool tuple = x is ITuple;
                var sb = new StringBuilder();
                sb.Append(tuple ? '(' : '[');
                for (int i = 0; i < items.Length; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append(Repr(items[i]));
                }
                if (tuple && items.Length == 1) sb.Append(',');   // Python's 1-tuple: (2,)
                sb.Append(tuple ? ')' : ']');
                return sb.ToString();
            }
            return x.ToString() ?? string.Empty;
        }

        /// <summary>
        ///     <c>format(v, '')</c> of a Python scalar (C# bool / integer / BigInteger / float / double / Complex) or of a NumPy
        ///     scalar's value (Half, char, decimal — formatted as <c>float(self)</c> / <c>int(self)</c>, their <c>__format__</c>).
        /// </summary>
        /// <param name="v">The value.</param>
        /// <param name="text">The text when the result is true.</param>
        /// <returns>False for anything that is not a scalar.</returns>
        private static bool TryFormatScalar(object v, out string text)
        {
            switch (v)
            {
                case bool b: text = b ? "True" : "False"; return true;
                case sbyte or byte or short or ushort or int or uint or long or ulong:
                    text = Convert.ToString(v, CultureInfo.InvariantCulture); return true;
                case BigInteger bi: text = bi.ToString(CultureInfo.InvariantCulture); return true;
                // A C# float is a Python float of its double value; np.float16 / np.float32's __format__ is float(self).
                case double d: text = ArrayFormatter.PythonFloatRepr(d, FloatKind.Double); return true;
                case float f: text = ArrayFormatter.PythonFloatRepr((double)f, FloatKind.Double); return true;
                case Half h: text = ArrayFormatter.PythonFloatRepr((double)h, FloatKind.Double); return true;
                case Complex c: text = ArrayFormatter.PythonComplexRepr(c); return true;
                case char c: text = ((int)c).ToString(CultureInfo.InvariantCulture); return true;   // int(self)
                case decimal m: text = ArrayFormatter.ScalarStr(m, NPTypeCode.Decimal); return true;   // the house's float64 print
            }
            text = null;
            return false;
        }

        /// <summary>NumPy 2's repr of a NumPy scalar: <c>np.{dtype name}({str})</c> — <c>np.float16(2.0)</c>, <c>np.float16(1e+03)</c>.</summary>
        /// <param name="value">The boxed scalar.</param><param name="t">Its dtype.</param>
        /// <returns>The repr.</returns>
        private static string NumPyScalarRepr(object value, NPTypeCode t)
            => $"np.{t.AsNumpyDtypeName()}({ArrayFormatter.ScalarStr(value, t)})";
    }
}
