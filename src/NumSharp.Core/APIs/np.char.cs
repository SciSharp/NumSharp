using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>numpy.char</c> module (<see cref="CharModule"/>): <c>np.char.equal(a, b)</c> is spelled
        ///     <c>np.@char.equal(a, b)</c>, since <c>char</c> is a C# keyword. The object also converts implicitly to the
        ///     <c>Char</c> dtype descriptor this field held before the module existed, so every <c>DType</c> use of
        ///     <c>np.@char</c> — <c>astype(np.@char)</c>, <c>a.dtype == np.@char</c>, a <c>DType[]</c> initializer, a
        ///     <c>(DType, DType)</c> tuple key — keeps compiling and keeps its meaning.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     <b>What changed for a <c>DType</c> user.</b> The member's TYPE is now <see cref="CharModule"/>, so code that reads
        ///     a descriptor member straight off it (<c>np.@char.itemsize</c>) or compares references
        ///     (<c>ReferenceEquals(x, np.@char)</c>) must convert first: <c>((DType)np.@char).itemsize</c>. Boxing it to
        ///     <c>object</c> boxes the module, not the descriptor.
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/routines.char.html</para>
        /// </remarks>
        public static CharModule @char { get; } = new CharModule();
    }

    /// <summary>
    ///     The <c>numpy.char</c> module surface, reachable as <see cref="np.@char"/>: NumPy's legacy vectorized string
    ///     operations over fixed-width <c>S</c> / <c>U</c> arrays. This release covers the comparisons that strip trailing
    ///     whitespace (<see cref="equal"/>, <see cref="not_equal"/>, <see cref="less"/>, <see cref="less_equal"/>,
    ///     <see cref="greater"/>, <see cref="greater_equal"/>, <see cref="compare_chararrays(object, object, string, bool)"/>),
    ///     <see cref="add"/> and <see cref="multiply"/>; the rest of the module follows the plan in
    ///     <c>docs/plans/np-char.md</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Arguments are array_likes (<c>object</c>), read as Python objects.</b> An <see cref="NDArray"/> is used as it
    ///     is; a C# <see cref="string"/> is NumPy's <c>str</c> (a 0-d <c>U</c> array); a <c>byte[]</c> is NumPy's
    ///     <c>bytes</c> (a 0-d <c>S</c> array) and a <c>byte[][]</c> / any <c>IEnumerable&lt;byte[]&gt;</c> a list of
    ///     bytes (an <c>S</c> array); a <c>string[]</c> or an <c>object[]</c> of text is a list of str; a C# number or bool
    ///     is NumPy's Python scalar. Unlike <see cref="np.strings"/> (whose <c>byte[]</c> is a <c>uint8</c> array, the
    ///     general <see cref="NDArray"/> conversion), a <c>byte[]</c> here is text, because text is all these parameters
    ///     can mean.
    ///     </para>
    ///     <para>
    ///     <b>Python's <c>NotImplemented</c> is <c>null</c>.</b> NumPy's comparisons answer <c>NotImplemented</c> for an
    ///     <c>S</c> operand against a <c>U</c> one (the class check of <c>_umath_strings_richcompare</c>); the C# functions
    ///     return <c>null</c> there.
    ///     </para>
    ///     <para>
    ///     <b>Python's <c>None</c> has no array form.</b> NumPy reads <c>None</c> as an object array; NumSharp has no object
    ///     dtype, so a <c>null</c> array argument is an <see cref="ArgumentNullException"/> where NumPy raises its
    ///     no-string / no-loop error. The object path (<see cref="split"/>, <see cref="rsplit"/>, <see cref="splitlines"/>,
    ///     <see cref="join(object, object)"/>) is the exception: it models NumPy's object arrays, so there a <c>null</c>
    ///     argument IS Python's <c>None</c> — NumPy's <c>string operation on non-string array</c> as the text argument,
    ///     <c>can only join an iterable</c> as <c>join</c>'s sequence, whitespace as <c>split</c>'s separator.
    ///     </para>
    ///     <para>
    ///     <b>The <c>Char</c> dtype.</b> The module converts implicitly to the <c>Char</c> dtype descriptor
    ///     (<see cref="op_Implicit(CharModule)"/>), the meaning <c>np.@char</c> had before the module existed.
    ///     </para>
    /// </remarks>
    [ModuleName("np.char")]
    public sealed partial class CharModule
    {
        /// <summary>NumPy's message for a <c>cmp</c> that names no comparison (multiarraymodule.c).</summary>
        private const string CmpMessage = "comparison must be '==', '!=', '<', '>', '<=', '>='";

        /// <summary>
        ///     The unsized <c>S</c> request a list of bytes is built under: <see cref="NDStringArrays.FromManaged"/> then
        ///     discovers the width (the longest value, at least 1), as NumPy's <c>np.array([b'ab', b'c'])</c> does.
        /// </summary>
        private static readonly DType s_bytesRequest = np.dtype("S");

        /// <summary>The <c>Char</c> dtype descriptor <c>np.@char</c> converts to (the registry's singleton).</summary>
        private static readonly DType s_charDType = typeof(char);

        /// <summary>The single instance is <see cref="np.@char"/>; the module has no state.</summary>
        internal CharModule() { }

        /// <summary>
        ///     <c>np.@char</c> where a <see cref="DType"/> is expected: the <c>Char</c> dtype descriptor (NumSharp's UTF-16
        ///     code-unit dtype), the value the member held before it became the module.
        /// </summary>
        /// <param name="module">The module (<see cref="np.@char"/>).</param>
        /// <returns>The <c>Char</c> descriptor, or null for a null module.</returns>
        public static implicit operator DType(CharModule module) => module is null ? null : s_charDType;

        /// <summary>Python's <c>repr()</c> of the module object.</summary>
        /// <returns><c>&lt;module 'numpy.char'&gt;</c>.</returns>
        public override string ToString() => "<module 'numpy.char'>";

        // =============================================================================================================
        // The comparisons that strip trailing whitespace
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.char.equal(x1, x2)</c>: <c>x1 == x2</c> element-wise once each element has lost its trailing
        ///     NULs and ASCII whitespace — <c>compare_chararrays(x1, x2, '==', True)</c>. So <c>'a '</c> equals <c>'a'</c>,
        ///     where <c>np.equal</c> says they differ.
        /// </summary>
        /// <param name="x1">The first operand: an <c>S</c> / <c>U</c> array or a text array_like (see the class remarks).</param>
        /// <param name="x2">The second operand, of the same class.</param>
        /// <returns>The bool result of the broadcast shape (a 0-d array for two scalars), or null — NumPy's
        ///     <c>NotImplemented</c> — when one operand is <c>S</c> and the other <c>U</c>.</returns>
        /// <exception cref="ArgumentNullException">An operand is null (NumPy's <c>None</c>; see the class remarks).</exception>
        /// <exception cref="TypeError">An operand is not <c>S</c> / <c>U</c> text — a number, a bool, a NEP 55
        ///     <c>StringDType</c> array: <c>comparison of non-string arrays</c>.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's ValueError text).</exception>
        /// <remarks>
        ///     The stripped units are NUL, <c>' '</c>, <c>'\t'</c>, <c>'\n'</c>, <c>'\v'</c>, <c>'\f'</c>, <c>'\r'</c> for
        ///     both classes (NumPy's <c>NumPyOS_ascii_isspace</c>); other whitespace such as <c>'\xa0'</c> or U+3000 stays,
        ///     and a leading space stays. The result is laid out as NumPy's iterator allocates it (an F-ordered pair gives an
        ///     F result).
        /// </remarks>
        [NDScoped]
        public NDArray equal(object x1, object x2) => Compare("equal", x1, x2, nameof(x1), nameof(x2));

        /// <summary>
        ///     NumPy's <c>np.char.not_equal(x1, x2)</c>: <c>x1 != x2</c> element-wise after stripping trailing NULs and ASCII
        ///     whitespace — <c>compare_chararrays(x1, x2, '!=', True)</c>.
        /// </summary>
        /// <param name="x1">The first operand (see <see cref="equal"/>).</param>
        /// <param name="x2">The second operand.</param>
        /// <returns>The bool result, or null (NumPy's <c>NotImplemented</c>) for an <c>S</c> / <c>U</c> pair.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError">An operand is not <c>S</c> / <c>U</c> text.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        public NDArray not_equal(object x1, object x2) => Compare("not_equal", x1, x2, nameof(x1), nameof(x2));

        /// <summary>
        ///     NumPy's <c>np.char.greater_equal(x1, x2)</c>: <c>x1 &gt;= x2</c> element-wise after stripping trailing NULs
        ///     and ASCII whitespace — <c>compare_chararrays(x1, x2, '&gt;=', True)</c>. The stripped strings compare unit by
        ///     unit as unsigned values (bytes for <c>S</c>, UCS-4 units for <c>U</c>).
        /// </summary>
        /// <param name="x1">The first operand (see <see cref="equal"/>).</param>
        /// <param name="x2">The second operand.</param>
        /// <returns>The bool result, or null (NumPy's <c>NotImplemented</c>) for an <c>S</c> / <c>U</c> pair.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError">An operand is not <c>S</c> / <c>U</c> text.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        public NDArray greater_equal(object x1, object x2) => Compare("greater_equal", x1, x2, nameof(x1), nameof(x2));

        /// <summary>
        ///     NumPy's <c>np.char.less_equal(x1, x2)</c>: <c>x1 &lt;= x2</c> element-wise after stripping trailing NULs and
        ///     ASCII whitespace — <c>compare_chararrays(x1, x2, '&lt;=', True)</c>.
        /// </summary>
        /// <param name="x1">The first operand (see <see cref="equal"/>).</param>
        /// <param name="x2">The second operand.</param>
        /// <returns>The bool result, or null (NumPy's <c>NotImplemented</c>) for an <c>S</c> / <c>U</c> pair.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError">An operand is not <c>S</c> / <c>U</c> text.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        public NDArray less_equal(object x1, object x2) => Compare("less_equal", x1, x2, nameof(x1), nameof(x2));

        /// <summary>
        ///     NumPy's <c>np.char.greater(x1, x2)</c>: <c>x1 &gt; x2</c> element-wise after stripping trailing NULs and ASCII
        ///     whitespace — <c>compare_chararrays(x1, x2, '&gt;', True)</c>.
        /// </summary>
        /// <param name="x1">The first operand (see <see cref="equal"/>).</param>
        /// <param name="x2">The second operand.</param>
        /// <returns>The bool result, or null (NumPy's <c>NotImplemented</c>) for an <c>S</c> / <c>U</c> pair.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError">An operand is not <c>S</c> / <c>U</c> text.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        public NDArray greater(object x1, object x2) => Compare("greater", x1, x2, nameof(x1), nameof(x2));

        /// <summary>
        ///     NumPy's <c>np.char.less(x1, x2)</c>: <c>x1 &lt; x2</c> element-wise after stripping trailing NULs and ASCII
        ///     whitespace — <c>compare_chararrays(x1, x2, '&lt;', True)</c>.
        /// </summary>
        /// <param name="x1">The first operand (see <see cref="equal"/>).</param>
        /// <param name="x2">The second operand.</param>
        /// <returns>The bool result, or null (NumPy's <c>NotImplemented</c>) for an <c>S</c> / <c>U</c> pair.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError">An operand is not <c>S</c> / <c>U</c> text.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        public NDArray less(object x1, object x2) => Compare("less", x1, x2, nameof(x1), nameof(x2));

        /// <summary>
        ///     NumPy's <c>np.char.compare_chararrays(a1, a2, cmp, rstrip)</c>: the element-wise comparison <paramref name="cmp"/>
        ///     of two string arrays of one class, each element first stripped of its trailing NULs and ASCII whitespace when
        ///     <paramref name="rstrip"/> is true.
        /// </summary>
        /// <param name="a1">The first operand (an <c>S</c> / <c>U</c> array or a text array_like; see the class remarks).</param>
        /// <param name="a2">The second operand, of the same class.</param>
        /// <param name="cmp">The comparison as NumPy's Python <c>str</c>: <c>"=="</c>, <c>"!="</c>, <c>"&lt;"</c>,
        ///     <c>"&lt;="</c>, <c>"&gt;"</c> or <c>"&gt;="</c> — read, as NumPy's <c>s#</c> argument does, as its UTF-8
        ///     bytes.</param>
        /// <param name="rstrip">True to strip each element's trailing NULs and ASCII whitespace before comparing.</param>
        /// <returns>The bool result of the broadcast shape (a 0-d array for two scalars), or null — NumPy's
        ///     <c>NotImplemented</c> — for an <c>S</c> operand against a <c>U</c> one.</returns>
        /// <exception cref="TypeError"><paramref name="cmp"/> is null (<c>a bytes-like object is required, not
        ///     'NoneType'</c>); or an operand is not <c>S</c> / <c>U</c> text (<c>comparison of non-string arrays</c>).</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="cmp"/> holds a lone surrogate, which UTF-8 cannot encode
        ///     (CPython's <c>'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed</c>).</exception>
        /// <exception cref="ValueError"><paramref name="cmp"/> is not one of the six comparisons:
        ///     <c>comparison must be '==', '!=', '&lt;', '&gt;', '&lt;=', '&gt;='</c>.</exception>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <remarks>
        ///     NumPy's order, all probed: <paramref name="cmp"/> is checked first (so a bad <c>cmp</c> wins over operands that
        ///     are not text or do not broadcast), then both operands must be <c>S</c> / <c>U</c> (a NEP 55
        ///     <c>StringDType</c> array is not), then a class mix answers <c>NotImplemented</c> (before the broadcast), then
        ///     the broadcast.
        /// </remarks>
        [NDScoped]
        public NDArray compare_chararrays(object a1, object a2, string cmp, bool rstrip)
        {
            if (cmp is null)
                throw new TypeError("a bytes-like object is required, not 'NoneType'");
            // NumPy's "s#" reads a str as its UTF-8 bytes and checks their count: 'é' is two bytes and fails the first-char
            // test, a lone surrogate fails the encoding itself.
            int length = NDUtf8.GetByteCount(cmp);
            if (length is < 1 or > 2)
                throw new ValueError(CmpMessage);
            Span<byte> bytes = stackalloc byte[2];
            NDUtf8.Encode(cmp, bytes);
            return CompareParsed(a1, a2, ParseCmp(bytes.Slice(0, length)), rstrip);
        }

        /// <summary>
        ///     NumPy's <c>np.char.compare_chararrays(a1, a2, cmp, rstrip)</c> with <paramref name="cmp"/> given as NumPy's
        ///     Python <c>bytes</c> (<c>b'=='</c>), which its <c>s#</c> argument accepts too.
        /// </summary>
        /// <param name="a1">The first operand.</param>
        /// <param name="a2">The second operand.</param>
        /// <param name="cmp">The comparison's bytes: <c>"=="u8</c>, <c>"!="u8</c>, <c>"&lt;"u8</c>, <c>"&lt;="u8</c>,
        ///     <c>"&gt;"u8</c> or <c>"&gt;="u8</c>.</param>
        /// <param name="rstrip">True to strip each element's trailing NULs and ASCII whitespace before comparing.</param>
        /// <returns>The bool result, or null (NumPy's <c>NotImplemented</c>) for an <c>S</c> / <c>U</c> pair.</returns>
        /// <exception cref="TypeError"><paramref name="cmp"/> is null, or an operand is not <c>S</c> / <c>U</c> text.</exception>
        /// <exception cref="ValueError"><paramref name="cmp"/> is not one of the six comparisons.</exception>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        public NDArray compare_chararrays(object a1, object a2, byte[] cmp, bool rstrip)
        {
            if (cmp is null)
                throw new TypeError("a bytes-like object is required, not 'NoneType'");
            return CompareParsed(a1, a2, ParseCmp(cmp), rstrip);
        }

        // =============================================================================================================
        // add and multiply
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.char.add(x1, x2)</c>: element-wise string concatenation. In NumPy 2.4.2 it IS the <c>np.add</c>
        ///     ufunc (<c>np.char.add is np.strings.add is np.add</c>), so two strings of one class concatenate at width
        ///     <c>width(x1) + width(x2)</c> with every rule of <see cref="np.add(NDArray, NDArray, NDArray, NDArray, DType)"/>.
        /// </summary>
        /// <param name="x1">The first operand (an array or a text array_like; see the class remarks).</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">An output to write into and return, or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, or null.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c> (the string add promoter ignores it for two strings of one class).</param>
        /// <returns>The concatenation, or <paramref name="out"/>.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError">A bytes / str mix (<c>ufunc 'add' cannot use operands with types …</c>), a string and
        ///     a number (NumPy's no-loop text), or a result wider than 2^31 - 1 bytes.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive the result, or the operands do not
        ///     broadcast with <paramref name="out"/> / <paramref name="where"/>.</exception>
        /// <exception cref="IncorrectShapeException">The two operands do not broadcast.</exception>
        /// <remarks>A NEP 55 <c>StringDType</c> operand computes through <c>np.add</c>'s <c>StringDType</c> loop (with a
        ///     <c>StringDType</c> or <c>U</c> partner; NumPy's NA rules and its incompatible-NA TypeError apply).</remarks>
        [NDScoped]
        public NDArray add(object x1, object x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.add(TextOperand(x1, nameof(x1)), TextOperand(x2, nameof(x2)), @out, where, dtype);

        /// <summary>
        ///     NumPy's <c>np.char.multiply(a, i)</c>: every string repeated <c>i</c> times (a count below 1 gives an empty
        ///     string) — <c>np.strings.multiply(a, i)</c> with its <c>TypeError</c> re-raised as
        ///     <c>ValueError("Can only multiply by integers")</c>, the module's backward-compatible rule.
        /// </summary>
        /// <param name="a">The strings (an <c>S</c> / <c>U</c> array or a text array_like).</param>
        /// <param name="i">The repeat counts: an integer array_like, broadcast against <paramref name="a"/>.</param>
        /// <returns>A fresh text array of <paramref name="a"/>'s class, as wide as the longest repetition.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="ValueError">Every <c>TypeError</c> of <see cref="np.strings.multiply(NDArray, NDArray)"/>:
        ///     a count that is not an integer (a float, a bool, text), a non-text <paramref name="a"/>, an output width that
        ///     is no valid descriptor (a uint64 count, a repetition wider than any descriptor) — all
        ///     <c>Can only multiply by integers</c> (probed against NumPy 2.4.2). Also NumPy's own ValueErrors: an empty
        ///     broadcast result (<c>zero-size array to reduction operation maximum which has no identity</c>) and operands
        ///     that do not broadcast.</exception>
        /// <exception cref="OverflowException">A string's <c>len * i</c> exceeds <c>sys.maxsize</c> ("Overflow encountered
        ///     in string multiply"): an OverflowError, not a TypeError, so it passes through.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="a"/> and <paramref name="i"/> do not broadcast.</exception>
        /// <remarks>A NEP 55 <c>StringDType</c> <paramref name="a"/> with an integer count computes NumPy's <c>a * i</c>
        ///     (<see cref="np.strings.multiply(NDArray, NDArray)"/>'s <c>StringDType</c> branch); a TypeError there is re-raised
        ///     as the ValueError above, as for every class.</remarks>
        [NDScoped]
        public NDArray multiply(object a, object i)
        {
            var text = TextOperand(a, nameof(a));
            var counts = TextOperand(i, nameof(i));
            try
            {
                return np.strings.multiply(text, counts);
            }
            catch (TypeError e)
            {
                // defchararray.multiply: `except TypeError: raise ValueError("Can only multiply by integers")` (the TypeError
                // stays reachable as the inner exception, as Python chains it as the context).
                throw new ValueError("Can only multiply by integers", e);
            }
        }

        // =============================================================================================================
        // Shared pieces
        // =============================================================================================================

        /// <summary>
        ///     A <c>numpy.char</c> array argument as the array NumPy's <c>np.asanyarray</c> / <c>PyArray_FROM_O</c> makes of
        ///     the Python object it stands for (the class remarks): an <see cref="NDArray"/> itself, a <c>byte[]</c> a 0-d
        ///     <c>S</c> (Python <c>bytes</c>), a sequence of <c>byte[]</c> an <c>S</c> array, anything else by
        ///     <see cref="NDArray.FromPythonOperand"/> (a <see cref="string"/> is a 0-d <c>U</c>, a number a Python scalar).
        /// </summary>
        /// <param name="value">The argument.</param>
        /// <param name="name">The parameter name, for the null check.</param>
        /// <returns>The array (a new one, except an <see cref="NDArray"/> argument; a new one is reclaimed by the calling
        ///     member's scope).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <exception cref="NotSupportedException"><paramref name="value"/> has no NumSharp array form.</exception>
        internal static NDArray TextOperand(object value, string name)
        {
            switch (value)
            {
                case null:
                    throw new ArgumentNullException(name);
                case byte[] bytes:
                    // Python bytes: np.asanyarray(b'ab') is the 0-d |S2 (b'' is |S1).
                    return NDArray.Scalar(bytes);
                case IEnumerable<byte[]> values:
                    // A list of bytes: an S array as wide as its longest value.
                    return NDStringArrays.FromManaged(values as byte[][] ?? values.ToArray(), s_bytesRequest, 0, 'C');
                default:
                    return NDArray.FromPythonOperand(value);
            }
        }

        /// <summary>One of the six stripping comparisons: <c>compare_chararrays(x1, x2, op, True)</c>.</summary>
        /// <param name="ufunc">The comparison's ufunc name.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="name1">The first parameter's name (null check).</param>
        /// <param name="name2">The second parameter's name.</param>
        /// <returns>The bool result, or null for NumPy's <c>NotImplemented</c>.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError">An operand is not <c>S</c> / <c>U</c> text.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        private static NDArray Compare(string ufunc, object x1, object x2, string name1, string name2)
            => NDStringComparison.CompareChararrays(ufunc, TextOperand(x1, name1), TextOperand(x2, name2), rstrip: true);

        /// <summary><c>compare_chararrays</c> after its <c>cmp</c> parsing: the operands converted, then NumPy's checks and loop.</summary>
        /// <param name="a1">The first operand.</param>
        /// <param name="a2">The second operand.</param>
        /// <param name="ufunc">The parsed comparison's ufunc name.</param>
        /// <param name="rstrip">Whether to strip trailing NULs and ASCII whitespace first.</param>
        /// <returns>The bool result, or null for NumPy's <c>NotImplemented</c>.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError">An operand is not <c>S</c> / <c>U</c> text.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        private static NDArray CompareParsed(object a1, object a2, string ufunc, bool rstrip)
            => NDStringComparison.CompareChararrays(ufunc, TextOperand(a1, nameof(a1)), TextOperand(a2, nameof(a2)), rstrip);

        /// <summary>
        ///     NumPy's <c>cmp</c> parsing (multiarraymodule.c <c>compare_chararrays</c>): one or two bytes, the second (when
        ///     present) <c>'='</c> — <c>==</c> / <c>!=</c> / <c>&lt;=</c> / <c>&gt;=</c> — or one of <c>&lt;</c> /
        ///     <c>&gt;</c> alone.
        /// </summary>
        /// <param name="cmp">The comparison's bytes.</param>
        /// <returns>The comparison's ufunc name.</returns>
        /// <exception cref="ValueError">Any other byte string (<c>"="</c>, <c>"=&gt;"</c>, <c>"&lt;&gt;"</c>, <c>"==="</c>,
        ///     <c>""</c>): <c>comparison must be '==', '!=', '&lt;', '&gt;', '&lt;=', '&gt;='</c>.</exception>
        private static string ParseCmp(ReadOnlySpan<byte> cmp)
        {
            if (cmp.Length is < 1 or > 2)
                throw new ValueError(CmpMessage);
            if (cmp.Length == 2)
            {
                if (cmp[1] != (byte)'=')
                    throw new ValueError(CmpMessage);
                return cmp[0] switch
                {
                    (byte)'=' => "equal",
                    (byte)'!' => "not_equal",
                    (byte)'<' => "less_equal",
                    (byte)'>' => "greater_equal",
                    _ => throw new ValueError(CmpMessage),
                };
            }
            return cmp[0] switch
            {
                (byte)'<' => "less",
                (byte)'>' => "greater",
                _ => throw new ValueError(CmpMessage),
            };
        }
    }
}
