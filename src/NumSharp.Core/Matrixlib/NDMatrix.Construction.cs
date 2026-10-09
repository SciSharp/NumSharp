using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text;

// =============================================================================
// NDMatrix.Construction.cs — matrix.__new__(subtype, data, dtype=None, copy=True) and _convert_from_string
// =============================================================================
//
// NumPy's constructor has four branches, tested in this order (numpy/matrixlib/defmatrix.py):
//
//   1. data is a matrix      same dtype and copy=False returns data ITSELF; otherwise data.astype(dtype), a copy
//                            in K order (an F-ordered matrix stays F-ordered).
//   2. data is an ndarray    new = data.view(matrix) — the finalize rule, which is where "shape too large to be a
//                            matrix." comes from — then: another dtype → new.astype(dtype) (K-order copy, also
//                            with copy=False); copy → new.copy() (C order); else the view.
//   3. data is a str         _convert_from_string: a Python list of rows, each token an ast.literal_eval, then
//                            branch 4.
//   4. anything else         arr = np.array(data, dtype, copy); more than two axes is "matrix must be
//                            2-dimensional"; the matrix is a VIEW of arr (ndarray.__new__(buffer=arr)), so it
//                            never owns its data.
//
// The C# boundary follows the house mapping of NDPolySequence.cs: an NDArray is an ndarray (branch 1 or 2), a
// typed C# array of a NumSharp dtype is an ndarray built by np.array, a string is a str, and every other array,
// tuple or collection is a Python sequence coerced by NumPy's own discovery walk (PolySequence.ToArray), so a
// list of C# ints is int64 and a ragged list raises NumPy's inhomogeneous-shape error.
//
// =============================================================================

namespace NumSharp
{
    public partial class NDMatrix
    {
        /// <summary>
        ///     NumPy's <c>matrix(data, dtype, copy)</c> for an array or a matrix (branches 1 and 2 of the file
        ///     header). Use it through <see cref="np.matrix(NDArray, DType, bool)"/> and
        ///     <see cref="np.asmatrix(NDArray, DType)"/>.
        /// </summary>
        /// <param name="data">The array. A 0-d <see cref="NDMatrix"/> is the scalar stand-in and is treated as a
        ///     plain 0-d array.</param>
        /// <param name="dtype">The dtype of the result, or <c>null</c> to keep the dtype of <paramref name="data"/>.</param>
        /// <param name="copy"><c>false</c> returns a view (or the matrix itself) when no conversion is needed. A
        ///     dtype change copies whatever this flag says.</param>
        /// <returns>
        ///     <paramref name="data"/> itself for a matrix that needs no conversion and no copy; a view of
        ///     <paramref name="data"/> for an array in the same case; otherwise a new matrix that owns its data.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">More than two axes of <paramref name="data"/> are longer than 1
        ///     (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="IncorrectShapeException">More than two axes and one of extent 0.</exception>
        [NDScoped]
        internal static NDMatrix Create(NDArray data, DType dtype, bool copy)
        {
            if (data is null)
                throw new ArgumentNullException(nameof(data));

            if (data is NDMatrix matrix && matrix.ndim != 0)
            {
                // isinstance(data, matrix): the dtype decides alone; astype's default order is 'K'.
                if (!copy && (dtype is null || dtype == matrix.dtype))
                    return matrix;
                return From(((NDArray)matrix).astype(dtype ?? matrix.dtype, copy: true), matrix);
            }

            // new = data.view(subtype): the finalize rule runs here, before any conversion, so an array that
            // cannot be a matrix raises without a copy being made.
            NDMatrix view = ViewOf(data);
            if (dtype is not null && dtype != data.dtype)
                return From(((NDArray)view).astype(dtype, copy: true), view);
            if (copy)
                return From(((NDArray)view).copy(), view);
            return view;
        }

        /// <summary>
        ///     NumPy's <c>matrix(data, dtype, copy)</c> for a matrix literal such as <c>"1 2; 3 4"</c> (branch 3).
        /// </summary>
        /// <param name="data">The literal: rows separated by <c>;</c>, columns by commas and/or whitespace, square
        ///     brackets ignored. Every token is one Python literal (see <see cref="EvaluateLiteral"/>).</param>
        /// <param name="dtype">The dtype of the result, or <c>null</c> for NumPy's discovered dtype (all bool →
        ///     bool, integers → int64, a float anywhere → float64, a complex anywhere → complex128, no element at
        ///     all → float64).</param>
        /// <returns>A new matrix of shape <c>(rows, columns)</c>; <c>(1, 0)</c> for an empty literal.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">The rows have different lengths (<c>Rows not the same size.</c>), or a
        ///     token is Python syntax that is not a literal (a name, an arithmetic expression).</exception>
        /// <exception cref="SyntaxError">A token is not Python syntax.</exception>
        /// <exception cref="NotSupportedException">A token is a string, <c>None</c> or <c>...</c>, or an integer
        ///     that does not fit uint64: NumPy builds a str or object matrix, dtypes NumSharp does not have.</exception>
        /// <exception cref="TypeError">A complex token with a real <paramref name="dtype"/> (NumPy's
        ///     <c>float() argument must be a string or a real number, not 'complex'</c>).</exception>
        /// <exception cref="OverflowException">An integer token outside an integer <paramref name="dtype"/>'s
        ///     range (NumPy's <c>Python integer 300 out of bounds for int8</c>).</exception>
        [NDScoped]
        internal static NDMatrix Create(string data, DType dtype)
        {
            if (data is null)
                throw new ArgumentNullException(nameof(data));

            return FromArrayLike(ArrayFromSequence(ConvertFromString(data), dtype));
        }

        /// <summary>
        ///     NumPy's <c>np.array(sequence, dtype=dtype)</c> for a Python sequence: the shape is discovered from
        ///     the nesting, and with a <paramref name="dtype"/> every element is stored with that dtype's
        ///     <c>setitem</c> (<c>PyArray_Pack</c>), not cast from the discovered array.
        /// </summary>
        /// <param name="sequence">A value <see cref="PolySequence.IsSequence"/> accepts.</param>
        /// <param name="dtype">The requested dtype, or <c>null</c> for the discovered one.</param>
        /// <returns>A new array (reclaimed by the caller's scope when the caller wraps it in a view).</returns>
        /// <exception cref="ValueError">A ragged sequence (NumPy's inhomogeneous-shape text), or a float NaN for
        ///     an integer dtype.</exception>
        /// <exception cref="TypeError">A complex element for a real dtype.</exception>
        /// <exception cref="OverflowException">A Python int outside an integer dtype's range, or a float
        ///     infinity for an integer dtype.</exception>
        /// <exception cref="NotSupportedException">The sequence holds strings, <c>null</c> items or other
        ///     objects.</exception>
        /// <remarks>
        ///     The difference from a cast shows for Python scalars: <c>np.array([300], dtype=np.int8)</c> raises
        ///     where <c>np.array([300]).astype(np.int8)</c> wraps to 44, and a complex element is refused by a
        ///     real dtype instead of losing its imaginary part. A bool dtype takes each element's truth value. A
        ///     sequence that holds ARRAYS is cast, as NumPy casts array items.
        /// </remarks>
        [NDScopedCovered]
        private static unsafe NDArray ArrayFromSequence(object sequence, DType dtype)
        {
            // The walk that discovers the shape also raises NumPy's errors for a ragged or non-numeric sequence.
            NDArray discovered = PolySequence.ToArray(sequence);
            if (dtype is null || dtype == discovered.dtype)
                return discovered;

            var leaves = new List<object>(checked((int)discovered.size));
            if (!TryCollectScalarLeaves(sequence, leaves) || leaves.Count != discovered.size)
                return discovered.astype(dtype, copy: true);

            NPTypeCode code = dtype.GetTypeCode();
            int itemSize = code.SizeOf();
            var result = new NDArray(dtype, discovered.Shape.AsClean(), false);
            Span<byte> slot = stackalloc byte[Backends.ReductionInitial.SlotBytes];
            byte* destination = (byte*)result.Address;
            for (int i = 0; i < leaves.Count; i++)
            {
                // PyArray_Pack of one Python object: range-checked ints, truncated floats, truth values for bool.
                Backends.ReductionInitial.TryPack(leaves[i], code, slot);
                slot.Slice(0, itemSize).CopyTo(new Span<byte>(destination + (long)i * itemSize, itemSize));
            }

            return result;
        }

        /// <summary>
        ///     Collects the leaves of a nested Python sequence in C order when every leaf is a scalar.
        /// </summary>
        /// <param name="node">A sequence or a leaf.</param>
        /// <param name="leaves">Receives the scalar leaves.</param>
        /// <returns><c>false</c> when a leaf is an array or a string: those take NumPy's cast route.</returns>
        private static bool TryCollectScalarLeaves(object node, List<object> leaves)
        {
            if (node is null or NDArray or string)
                return false;
            if (!PolySequence.IsSequence(node))
            {
                // A typed C# array of a dtype is an ndarray leaf for the sequence coercion.
                if (node is Array)
                    return false;
                leaves.Add(node);
                return true;
            }

            foreach (object item in PolySequence.Items(node))
            {
                if (!TryCollectScalarLeaves(item, leaves))
                    return false;
            }

            return true;
        }

        /// <summary>
        ///     NumPy's <c>matrix(data, dtype, copy)</c> for any other value (branch 4, after dispatching the first
        ///     three): a typed C# array, a Python-like sequence, or a scalar.
        /// </summary>
        /// <param name="data">The value: an <see cref="NDArray"/>, a <see cref="string"/>, a C# array of a NumSharp
        ///     dtype (any rank up to 2), a tuple / <c>object[]</c> / jagged array / collection (a Python sequence,
        ///     nested to any depth up to 2 axes), or a C# number.</param>
        /// <param name="dtype">The dtype of the result, or <c>null</c> for the discovered one.</param>
        /// <param name="copy">Only read for an <see cref="NDArray"/>; every other value is converted into new
        ///     memory, as <c>np.array</c> does for a list.</param>
        /// <returns>The matrix. For branch 4 it is a view of the array <c>np.array</c> built, so
        ///     <c>flags.owndata</c> is false, as in NumPy.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c> (NumPy builds an object
        ///     matrix holding <c>None</c>).</exception>
        /// <exception cref="ValueError">The value has more than two axes (<c>matrix must be 2-dimensional</c>), a
        ///     sequence is ragged (NumPy's inhomogeneous-shape text), or — for an array — it cannot be a matrix.</exception>
        /// <exception cref="SyntaxError">A matrix literal holds a token that is not Python syntax.</exception>
        /// <exception cref="NotSupportedException">The value holds strings, <c>null</c> items or other objects:
        ///     NumPy's str / object dtypes.</exception>
        /// <exception cref="TypeError">A sequence holds a complex number and <paramref name="dtype"/> is real.</exception>
        /// <exception cref="OverflowException">A sequence holds a Python int outside an integer
        ///     <paramref name="dtype"/>'s range.</exception>
        /// <remarks>
        ///     With a <paramref name="dtype"/> a SEQUENCE is stored element by element with the dtype's own
        ///     conversion (see <see cref="ArrayFromSequence"/>); a typed C# array and a single number are cast
        ///     (they are an ndarray and a NumPy-style value for this method).
        /// </remarks>
        [NDScoped]
        internal static NDMatrix Create(object data, DType dtype, bool copy)
        {
            switch (data)
            {
                case null:
                    throw new ArgumentNullException(nameof(data), "np.matrix(None) is an object matrix in NumPy; NumSharp has no object dtype.");
                case NDArray array:
                    return Create(array, dtype, copy);
                case string literal:
                    return Create(literal, dtype);
            }

            // np.array(data, dtype=dtype, copy=copy): a sequence is always converted into new memory.
            if (PolySequence.IsSequence(data))
                return FromArrayLike(ArrayFromSequence(data, dtype));

            NDArray converted = data is Array typed ? np.array(typed) : np.asanyarray(data);
            if (dtype is not null && dtype != converted.dtype)
                converted = converted.astype(dtype, copy: true);
            return FromArrayLike(converted);
        }

        /// <summary>
        ///     The end of branch 4: rejects more than two axes and returns the matrix NumPy builds with
        ///     <c>ndarray.__new__(subtype, shape, dtype, buffer=arr, order=order)</c>, a view of the converted
        ///     array.
        /// </summary>
        /// <param name="array">The array <c>np.array</c> built (reclaimed by the caller's scope).</param>
        /// <returns>A matrix over the array's memory; <c>(1, 1)</c> for 0-d, <c>(1, N)</c> for 1-d. A 2-d
        ///     F-ordered array stays F-ordered (NumPy's <c>order = 'F'</c>).</returns>
        /// <exception cref="ValueError"><paramref name="array"/> has more than two axes.</exception>
        private static NDMatrix FromArrayLike(NDArray array)
        {
            // Unlike the ndarray branch, extent-1 axes are NOT squeezed here: np.matrix([[[1, 2]]]) raises.
            if (array.ndim > 2)
                throw new ValueError("matrix must be 2-dimensional");
            return ViewOf(array);
        }

        #region _convert_from_string

        /// <summary>
        ///     NumPy's <c>_convert_from_string</c>: the Python list of rows a matrix literal denotes.
        /// </summary>
        /// <param name="data">The literal.</param>
        /// <returns>One <c>object[]</c> per row (a Python list of lists); every item is a boxed
        ///     <see cref="bool"/>, <see cref="long"/>, <see cref="BigInteger"/>, <see cref="double"/> or
        ///     <see cref="Complex"/>, the kinds <see cref="PolySequence"/> coerces as Python scalars.</returns>
        /// <exception cref="ValueError">Rows of different lengths, or a token that is not a literal.</exception>
        /// <exception cref="SyntaxError">A token that is not Python syntax.</exception>
        /// <exception cref="NotSupportedException">A string, <c>None</c> or <c>...</c> token.</exception>
        /// <remarks>
        ///     NumPy's statements: remove every <c>[</c> and <c>]</c>; split on <c>;</c> into rows; split each row
        ///     on <c>,</c>; split each part on whitespace (<c>str.split()</c>); evaluate every token. The row
        ///     length check runs after each row, so a bad token in row 0 is reported before a short row 1.
        /// </remarks>
        private static object[] ConvertFromString(string data)
        {
            string stripped = data.Replace("[", string.Empty).Replace("]", string.Empty);
            string[] rows = stripped.Split(';');
            var result = new object[rows.Length];
            var items = new List<object>();
            int columns = 0;

            for (int r = 0; r < rows.Length; r++)
            {
                items.Clear();
                foreach (string part in rows[r].Split(','))
                {
                    // str.split(): runs of whitespace separate, leading and trailing whitespace is dropped.
                    int i = 0;
                    while (i < part.Length)
                    {
                        while (i < part.Length && IsPythonWhitespace(part[i]))
                            i++;
                        int start = i;
                        while (i < part.Length && !IsPythonWhitespace(part[i]))
                            i++;
                        if (i > start)
                            items.Add(EvaluateLiteral(part.Substring(start, i - start)));
                    }
                }

                if (r == 0)
                    columns = items.Count;
                else if (items.Count != columns)
                    throw new ValueError("Rows not the same size.");
                result[r] = items.ToArray();
            }

            return result;
        }

        /// <summary>Python's <c>str.isspace()</c> for one UTF-16 unit: .NET's white space plus the four C0
        ///     information separators U+001C..U+001F, which Python counts and .NET does not.</summary>
        /// <param name="c">The character.</param>
        /// <returns>Whether <c>str.split()</c> separates on it.</returns>
        private static bool IsPythonWhitespace(char c) => char.IsWhiteSpace(c) || (c >= '\u001c' && c <= '\u001f');

        #endregion

        #region ast.literal_eval of one token

        /// <summary>The node kinds of the small expression tree <see cref="LiteralParser"/> builds, named after
        ///     the <c>ast</c> classes whose names CPython prints in a "malformed node" error.</summary>
        private enum NodeKind
        {
            /// <summary>A literal value (<c>ast.Constant</c>).</summary>
            Constant,

            /// <summary>An identifier that is not <c>True</c>, <c>False</c> or <c>None</c> (<c>ast.Name</c>).</summary>
            Name,

            /// <summary>A prefix operator applied to an operand (<c>ast.UnaryOp</c>).</summary>
            UnaryOp,

            /// <summary>An infix arithmetic or bitwise operator (<c>ast.BinOp</c>).</summary>
            BinOp,

            /// <summary>A comparison (<c>ast.Compare</c>).</summary>
            Compare,
        }

        /// <summary>One node of a parsed token. A class, because the tree is tiny and built once per token.</summary>
        private sealed class Node
        {
            /// <summary>What the node is.</summary>
            public NodeKind Kind;

            /// <summary>The value of a <see cref="NodeKind.Constant"/>: a boxed <see cref="bool"/>,
            ///     <see cref="long"/>, <see cref="BigInteger"/>, <see cref="double"/> or <see cref="Complex"/>;
            ///     a <see cref="string"/> for a str literal; <see cref="NoneValue"/> or <see cref="EllipsisValue"/>.</summary>
            public object Value;

            /// <summary>The operator character of a unary or binary node (<c>+</c>, <c>-</c>, <c>~</c>, <c>*</c> ...).</summary>
            public char Operator;

            /// <summary>The operand of a unary node, the left operand of a binary one.</summary>
            public Node Left;

            /// <summary>The right operand of a binary node.</summary>
            public Node Right;
        }

        /// <summary>The value of Python's <c>None</c> literal (C# <c>null</c> would read as "no value").</summary>
        private static readonly object NoneValue = new object();

        /// <summary>The value of Python's <c>...</c> literal.</summary>
        private static readonly object EllipsisValue = new object();

        /// <summary>
        ///     <c>ast.literal_eval(token)</c> for one whitespace-free token of a matrix literal, restricted to the
        ///     values a numeric matrix can hold.
        /// </summary>
        /// <param name="token">The token, without white space.</param>
        /// <returns>A boxed <see cref="bool"/> (<c>True</c> / <c>False</c>), <see cref="long"/> or
        ///     <see cref="BigInteger"/> (a Python int), <see cref="double"/> (a Python float) or
        ///     <see cref="Complex"/> (a Python complex).</returns>
        /// <exception cref="SyntaxError">The token is not Python syntax (<c>1L</c>, <c>+</c>, <c>(1</c>).</exception>
        /// <exception cref="ValueError">The token is Python syntax but not a literal (a name such as <c>nan</c>,
        ///     <c>2*3</c>, <c>--1</c>): CPython's <c>malformed node or string</c>.</exception>
        /// <exception cref="NotSupportedException">The token is a str, <c>None</c> or <c>...</c> literal.</exception>
        /// <remarks>
        ///     Accepted, as by CPython 3.12: decimal, hexadecimal, octal and binary integers with <c>_</c>
        ///     separators; floats (<c>.5</c>, <c>5.</c>, <c>1e3</c>; <c>1e400</c> is inf); imaginary numbers
        ///     (<c>2j</c>); one leading sign; <c>real+imagj</c> / <c>real-imagj</c>; <c>True</c> / <c>False</c>; and
        ///     parentheses around any of them.
        /// </remarks>
        internal static object EvaluateLiteral(string token)
        {
            Node node = new LiteralParser(token).Parse();
            object value = ConvertNode(node);
            if (value is string || ReferenceEquals(value, NoneValue) || ReferenceEquals(value, EllipsisValue))
                throw new NotSupportedException(
                    $"The matrix literal token {token} is a str, None or Ellipsis: NumPy builds a str or object matrix, dtypes NumSharp does not have.");
            return value;
        }

        /// <summary>CPython's <c>literal_eval._convert</c> for the node kinds a single token can produce.</summary>
        /// <param name="node">The parsed token.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ValueError">The node is not a literal.</exception>
        private static object ConvertNode(Node node)
        {
            if (node.Kind == NodeKind.Constant)
                return node.Value;

            // BinOp with Add / Sub: a real left operand and an imaginary right operand make a complex literal.
            if (node.Kind == NodeKind.BinOp && (node.Operator == '+' || node.Operator == '-'))
            {
                object left = ConvertSignedNumber(node.Left);
                object right = ConvertNumber(node.Right);
                if (left is not Complex && right is Complex imaginary)
                {
                    // Python adds a real and a complex part by part: the real part is left + 0.0 (so -0.0 turns
                    // into +0.0) and the imaginary part is 0.0 ± imag.
                    double real = ToDouble(left) + imaginary.Real;
                    return node.Operator == '+'
                        ? new Complex(real, 0.0 + imaginary.Imaginary)
                        : new Complex(ToDouble(left) - imaginary.Real, 0.0 - imaginary.Imaginary);
                }
            }

            return ConvertSignedNumber(node);
        }

        /// <summary>CPython's <c>_convert_signed_num</c>: a number, optionally under ONE unary plus or minus.</summary>
        /// <param name="node">The node.</param>
        /// <returns>The signed number.</returns>
        /// <exception cref="ValueError">The node is not a (signed) number.</exception>
        private static object ConvertSignedNumber(Node node)
        {
            if (node.Kind == NodeKind.UnaryOp && (node.Operator == '+' || node.Operator == '-'))
            {
                object operand = ConvertNumber(node.Left);
                if (node.Operator == '+')
                    return operand;
                return operand switch
                {
                    long l => l == long.MinValue ? (object)(-(BigInteger)l) : -l,
                    BigInteger b => Narrow(-b),
                    double d => -d,
                    Complex c => -c, // negates both parts: -2j is (-0-2j) in Python too
                    _ => throw Malformed(node),
                };
            }

            return ConvertNumber(node);
        }

        /// <summary>CPython's <c>_convert_num</c>: an int, float or complex constant. A bool is NOT a number
        ///     there (<c>-True</c> is malformed).</summary>
        /// <param name="node">The node.</param>
        /// <returns>The number.</returns>
        /// <exception cref="ValueError">The node is anything else.</exception>
        private static object ConvertNumber(Node node)
        {
            if (node.Kind == NodeKind.Constant && node.Value is long or BigInteger or double or Complex)
                return node.Value;
            throw Malformed(node);
        }

        /// <summary>The real value of a Python int or float operand of a complex literal.</summary>
        /// <param name="number">A boxed <see cref="long"/>, <see cref="BigInteger"/> or <see cref="double"/>.</param>
        /// <returns>The value as a double (an int past the float range is infinity; CPython raises
        ///     <c>OverflowError</c> there, a literal nobody writes).</returns>
        private static double ToDouble(object number) => number switch
        {
            long l => l,
            BigInteger b => (double)b,
            _ => (double)number,
        };

        /// <summary>A Python int as the narrowest box: <see cref="long"/> when it fits.</summary>
        /// <param name="value">The integer.</param>
        /// <returns>A boxed <see cref="long"/> or <see cref="BigInteger"/>.</returns>
        private static object Narrow(BigInteger value)
            => value >= long.MinValue && value <= long.MaxValue ? (object)(long)value : value;

        /// <summary>CPython's <c>_raise_malformed_node</c>.</summary>
        /// <param name="node">The offending node.</param>
        /// <returns>The error to throw. CPython prints the node object's memory address, which has no
        ///     counterpart here; the address is printed as zeros so the rest of the text matches.</returns>
        private static ValueError Malformed(Node node)
            => new ValueError($"malformed node or string on line 1: <ast.{node.Kind} object at 0x0000000000000000>");

        /// <summary>
        ///     A recursive-descent parser for the Python expressions one matrix-literal token can be: numbers,
        ///     names, string literals, unary and binary operators, parentheses. It decides between "not Python"
        ///     (<see cref="SyntaxError"/>) and "Python, but perhaps not a literal" (a <see cref="Node"/> for
        ///     <see cref="ConvertNode"/> to judge).
        /// </summary>
        private ref struct LiteralParser
        {
            /// <summary>The token being parsed.</summary>
            private readonly ReadOnlySpan<char> _text;

            /// <summary>The position of the next unread character.</summary>
            private int _pos;

            /// <summary>Starts a parser over <paramref name="token"/>.</summary>
            /// <param name="token">The token (no white space).</param>
            public LiteralParser(string token)
            {
                _text = token;
                _pos = 0;
            }

            /// <summary>Parses the whole token as one expression.</summary>
            /// <returns>The expression tree.</returns>
            /// <exception cref="SyntaxError">The token is not one complete expression.</exception>
            public Node Parse()
            {
                Node node = ParseComparison();
                if (_pos < _text.Length)
                    throw Syntax(_text[_pos] == ')' ? "unmatched ')'" : "invalid syntax");
                return node;
            }

            /// <summary>Comparison operators (lowest precedence handled here): <c>&lt; &gt; == != &lt;= &gt;=</c>.</summary>
            /// <returns>The node.</returns>
            private Node ParseComparison()
            {
                Node left = ParseBinary(0);
                while (_pos < _text.Length)
                {
                    char c = _text[_pos];
                    bool two = _pos + 1 < _text.Length && _text[_pos + 1] == '=';
                    if ((c == '<' || c == '>') && !(_pos + 1 < _text.Length && _text[_pos + 1] == c))
                        _pos += two ? 2 : 1;
                    else if ((c == '=' || c == '!') && two)
                        _pos += 2;
                    else
                        break;
                    left = new Node { Kind = NodeKind.Compare, Operator = c, Left = left, Right = ParseBinary(0) };
                }

                return left;
            }

            /// <summary>
            ///     Binary operators by precedence climbing. Only the tree's SHAPE matters to the caller (a BinOp
            ///     with Add / Sub over a real and an imaginary constant is a complex literal; every other BinOp is
            ///     malformed), so the levels below addition share one precedence table.
            /// </summary>
            /// <param name="minimum">The lowest precedence this call may consume.</param>
            /// <returns>The node.</returns>
            private Node ParseBinary(int minimum)
            {
                Node left = ParseUnary();
                while (_pos < _text.Length)
                {
                    int width = OperatorAt(_pos, out int precedence, out bool rightAssociative);
                    if (width == 0 || precedence < minimum)
                        break;
                    char op = _text[_pos];
                    _pos += width;
                    Node right = ParseBinary(rightAssociative ? precedence : precedence + 1);
                    left = new Node { Kind = NodeKind.BinOp, Operator = width == 1 ? op : '\0', Left = left, Right = right };
                }

                return left;
            }

            /// <summary>The binary operator starting at <paramref name="at"/>, if any.</summary>
            /// <param name="at">The position.</param>
            /// <param name="precedence">Its precedence (higher binds tighter).</param>
            /// <param name="rightAssociative">Whether it is <c>**</c>.</param>
            /// <returns>The operator's length in characters, 0 when there is none.</returns>
            private int OperatorAt(int at, out int precedence, out bool rightAssociative)
            {
                rightAssociative = false;
                precedence = 0;
                char c = _text[at];
                char next = at + 1 < _text.Length ? _text[at + 1] : '\0';
                switch (c)
                {
                    case '|': precedence = 1; return 1;
                    case '^': precedence = 2; return 1;
                    case '&': precedence = 3; return 1;
                    case '<' when next == '<': precedence = 4; return 2;
                    case '>' when next == '>': precedence = 4; return 2;
                    case '+':
                    case '-': precedence = 5; return 1;
                    case '*' when next == '*': precedence = 7; rightAssociative = true; return 2;
                    case '*':
                    case '@':
                    case '%': precedence = 6; return 1;
                    case '/': precedence = 6; return next == '/' ? 2 : 1;
                    default: return 0;
                }
            }

            /// <summary>Prefix operators <c>+ - ~</c>, then an atom.</summary>
            /// <returns>The node.</returns>
            /// <exception cref="SyntaxError">The token ends where an operand must follow.</exception>
            private Node ParseUnary()
            {
                if (_pos >= _text.Length)
                    throw Syntax("invalid syntax");
                char c = _text[_pos];
                if (c == '+' || c == '-' || c == '~')
                {
                    _pos++;
                    return new Node { Kind = NodeKind.UnaryOp, Operator = c, Left = ParseUnary() };
                }

                return ParseAtom();
            }

            /// <summary>A number, a name, a string literal, <c>...</c> or a parenthesized expression.</summary>
            /// <returns>The node.</returns>
            /// <exception cref="SyntaxError">No atom starts here.</exception>
            private Node ParseAtom()
            {
                char c = _text[_pos];
                if (c == '(')
                {
                    _pos++;
                    if (_pos >= _text.Length)
                        throw Syntax("'(' was never closed");
                    Node inner = ParseComparison();
                    if (_pos >= _text.Length || _text[_pos] != ')')
                        throw Syntax(_pos >= _text.Length ? "'(' was never closed" : "invalid syntax");
                    _pos++;
                    return inner;
                }

                if (char.IsAsciiDigit(c) || (c == '.' && _pos + 1 < _text.Length && char.IsAsciiDigit(_text[_pos + 1])))
                    return ParseNumber();

                if (c == '.' && _text.Slice(_pos).StartsWith("..."))
                {
                    _pos += 3;
                    return new Node { Kind = NodeKind.Constant, Value = EllipsisValue };
                }

                if (c == '\'' || c == '"')
                    return ParseString(c);

                if (char.IsLetter(c) || c == '_')
                {
                    int start = _pos;
                    while (_pos < _text.Length && (char.IsLetterOrDigit(_text[_pos]) || _text[_pos] == '_'))
                        _pos++;
                    ReadOnlySpan<char> name = _text.Slice(start, _pos - start);
                    // A name directly followed by a quote is a string prefix (b'..', r"..") in Python.
                    if (_pos < _text.Length && (_text[_pos] == '\'' || _text[_pos] == '"'))
                        return ParseString(_text[_pos]);
                    if (name.SequenceEqual("True"))
                        return new Node { Kind = NodeKind.Constant, Value = true };
                    if (name.SequenceEqual("False"))
                        return new Node { Kind = NodeKind.Constant, Value = false };
                    if (name.SequenceEqual("None"))
                        return new Node { Kind = NodeKind.Constant, Value = NoneValue };
                    return new Node { Kind = NodeKind.Name };
                }

                throw Syntax("invalid syntax");
            }

            /// <summary>A quoted string literal up to its closing quote.</summary>
            /// <param name="quote">The quote character at the current position.</param>
            /// <returns>A constant node holding the text between the quotes (escapes are not decoded: the value is
            ///     refused by the caller whatever it is).</returns>
            /// <exception cref="SyntaxError">The closing quote is missing.</exception>
            private Node ParseString(char quote)
            {
                int start = ++_pos;
                while (_pos < _text.Length && _text[_pos] != quote)
                    _pos += _text[_pos] == '\\' ? 2 : 1;
                if (_pos >= _text.Length)
                    throw Syntax("unterminated string literal (detected at line 1)");
                string value = _text.Slice(start, _pos - start).ToString();
                _pos++;
                return new Node { Kind = NodeKind.Constant, Value = value };
            }

            /// <summary>
            ///     A Python numeric literal by CPython's tokenizer rules: integers in four bases with <c>_</c>
            ///     separators, floats, and the imaginary suffix.
            /// </summary>
            /// <returns>A constant node holding a boxed <see cref="long"/> / <see cref="BigInteger"/> (int),
            ///     <see cref="double"/> (float) or <see cref="Complex"/> (imaginary).</returns>
            /// <exception cref="SyntaxError">A malformed literal, with CPython's message for its kind.</exception>
            private Node ParseNumber()
            {
                int start = _pos;
                if (_text[_pos] == '0' && _pos + 1 < _text.Length)
                {
                    char prefix = char.ToLowerInvariant(_text[_pos + 1]);
                    if (prefix == 'x' || prefix == 'o' || prefix == 'b')
                        return ParsePrefixedInteger(prefix);
                }

                bool isFloat = false;
                ScanDigits("invalid decimal literal");
                if (_pos < _text.Length && _text[_pos] == '.')
                {
                    isFloat = true;
                    _pos++;
                    if (_pos < _text.Length && char.IsAsciiDigit(_text[_pos]))
                        ScanDigits("invalid decimal literal");
                }

                if (_pos < _text.Length && (_text[_pos] == 'e' || _text[_pos] == 'E'))
                {
                    int mark = _pos + 1;
                    if (mark < _text.Length && (_text[mark] == '+' || _text[mark] == '-'))
                        mark++;
                    // An 'e' that no digit follows is not an exponent; it then reads as a letter glued to the
                    // number, which is the "invalid decimal literal" below.
                    if (mark < _text.Length && char.IsAsciiDigit(_text[mark]))
                    {
                        isFloat = true;
                        _pos = mark;
                        ScanDigits("invalid decimal literal");
                    }
                }

                bool imaginary = _pos < _text.Length && (_text[_pos] == 'j' || _text[_pos] == 'J');
                ReadOnlySpan<char> digits = _text.Slice(start, _pos - start);
                if (imaginary)
                    _pos++;
                if (_pos < _text.Length && (char.IsLetterOrDigit(_text[_pos]) || _text[_pos] == '_'))
                    throw Syntax(imaginary ? "invalid imaginary literal" : "invalid decimal literal");

                string clean = RemoveUnderscores(digits);
                if (isFloat || imaginary)
                {
                    // An overflowing float literal is inf in Python, and double.Parse agrees.
                    double value = double.Parse(clean, NumberStyles.Float, CultureInfo.InvariantCulture);
                    return new Node { Kind = NodeKind.Constant, Value = imaginary ? new Complex(0.0, value) : (object)value };
                }

                // "0", "00", "0_0" are zero; any other integer with a leading zero is the Python 2 octal form.
                if (clean.Length > 1 && clean[0] == '0' && clean.AsSpan().TrimStart('0').Length > 0)
                    throw Syntax("leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers");
                return new Node { Kind = NodeKind.Constant, Value = Narrow(BigInteger.Parse(clean, NumberStyles.None, CultureInfo.InvariantCulture)) };
            }

            /// <summary>An integer with a <c>0x</c>, <c>0o</c> or <c>0b</c> prefix.</summary>
            /// <param name="prefix">The lower-case prefix letter.</param>
            /// <returns>A constant node holding the integer.</returns>
            /// <exception cref="SyntaxError">No digit follows the prefix, a digit is outside the base, or a
            ///     letter is glued to the literal.</exception>
            private Node ParsePrefixedInteger(char prefix)
            {
                int radix = prefix == 'x' ? 16 : prefix == 'o' ? 8 : 2;
                string kind = prefix == 'x' ? "hexadecimal" : prefix == 'o' ? "octal" : "binary";
                _pos += 2;
                BigInteger value = BigInteger.Zero;
                bool any = false, lastWasUnderscore = false;
                while (_pos < _text.Length)
                {
                    char c = _text[_pos];
                    if (c == '_')
                    {
                        // One underscore may sit between the prefix and a digit or between two digits.
                        if (lastWasUnderscore)
                            throw Syntax($"invalid {kind} literal");
                        lastWasUnderscore = true;
                        _pos++;
                        continue;
                    }

                    int digit = char.IsAsciiHexDigit(c) ? System.Convert.ToInt32(c.ToString(), 16) : -1;
                    if (digit < 0)
                        break;
                    if (digit >= radix)
                    {
                        // A decimal digit outside the base has its own message; a hex letter after an octal or
                        // binary literal is a glued letter.
                        if (char.IsAsciiDigit(c))
                            throw Syntax($"invalid digit '{c}' in {kind} literal");
                        break;
                    }

                    value = value * radix + digit;
                    any = true;
                    lastWasUnderscore = false;
                    _pos++;
                }

                if (!any || lastWasUnderscore || (_pos < _text.Length && (char.IsLetterOrDigit(_text[_pos]) || _text[_pos] == '_')))
                    throw Syntax($"invalid {kind} literal");
                return new Node { Kind = NodeKind.Constant, Value = Narrow(value) };
            }

            /// <summary>Consumes a run of decimal digits with single <c>_</c> separators between digits.</summary>
            /// <param name="message">CPython's message for a misplaced underscore.</param>
            /// <exception cref="SyntaxError">An underscore is not followed by a digit.</exception>
            private void ScanDigits(string message)
            {
                while (_pos < _text.Length)
                {
                    char c = _text[_pos];
                    if (char.IsAsciiDigit(c))
                    {
                        _pos++;
                    }
                    else if (c == '_')
                    {
                        if (_pos + 1 >= _text.Length || !char.IsAsciiDigit(_text[_pos + 1]))
                            throw Syntax(message);
                        _pos++;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            /// <summary>The digits of a literal without its <c>_</c> separators.</summary>
            /// <param name="literal">The literal text.</param>
            /// <returns>The text .NET's parsers accept.</returns>
            private static string RemoveUnderscores(ReadOnlySpan<char> literal)
            {
                if (literal.IndexOf('_') < 0)
                    return literal.ToString();
                var builder = new StringBuilder(literal.Length);
                foreach (char c in literal)
                    if (c != '_')
                        builder.Append(c);
                return builder.ToString();
            }

            /// <summary>A <see cref="SyntaxError"/> with the location suffix <c>ast.parse</c> appends for source
            ///     that has no file name.</summary>
            /// <param name="message">CPython's message.</param>
            /// <returns>The error to throw.</returns>
            private static SyntaxError Syntax(string message) => new SyntaxError(message + " (<unknown>, line 1)");
        }

        #endregion
    }
}
