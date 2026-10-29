using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

// =============================================================================
// NDCharObjectPath.cs — numpy.char's object path: np.char.split / rsplit / splitlines / join
// =============================================================================
//
// NUMPY'S STRUCTURE (numpy/_core/strings.py, multiarraymodule.c `_vec_string`, CPython 3.12's str / bytes methods)
//   split(a, sep, maxsplit)   = _vec_string(a, object_, 'split', [sep] + _clean_args(maxsplit))
//   rsplit(...)               = the same with 'rsplit'
//   splitlines(a, keepends)   = _vec_string(a, object_, 'splitlines', _clean_args(keepends))
//   join(sep, seq)            = _to_bytes_or_str_array(_vec_string(sep, object_, 'join', (seq,)), seq)
//   _vec_string converts its first argument (PyArray_Converter), picks the METHOD by its dtype — bytes.<name> for S,
//   str.<name> for U and StringDType, otherwise "string operation on non-string array" —, converts and broadcasts the
//   other arguments with a multi-iterator (PyArray_MultiIterFromObjects: np.asarray each, NumPy's "shape mismatch: objects
//   cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (2,) and arg 1 with shape (3,)." text),
//   allocates a C-ordered object result and walks the broadcast in C order: per element it makes the NumPy SCALAR of each
//   operand (PyArray_ToScalar) and calls the method with them. The first element (C order) that raises ends the call.
//
// WHAT AN ELEMENT IS TO THE METHOD (PyArray_ToScalar): an S element is np.bytes_ (trailing NULs dropped; the buffer
//   protocol gives its bytes), a U element np.str_ (trailing NULs dropped; a ONE-unit text above U+10FFFF is CPython's
//   SystemError "invalid maximum character passed to PyUnicode_New"; the buffer protocol gives its UCS-4 bytes), a
//   StringDType element a Python str or the descriptor's NA object, a numeric element its NumPy scalar (np.int64, np.bool,
//   np.float64: the buffer protocol gives its raw bytes), and an object-array element (a flat or rectangular C# list holding
//   null, NumPy's object array) the Python object itself. So bytes.split accepts ANY NumPy scalar as a separator (the bytes
//   of np.int64(5)), and str.split only a str or None ("must be str or None, not numpy.int64").
//
// CPYTHON'S ORDER PER ELEMENT (what NumSharp checks, in this order)
//   1. the scalars of the operands, in argument order (the U SystemError);
//   2. the method descriptor's own check: str.<name> applied to a non-str (a StringDType NA such as None) is
//      "descriptor 'split' for 'str' objects doesn't apply to a 'NoneType' object";
//   3. argument parsing: maxsplit through operator.index ("'numpy.float64' object cannot be interpreted as an integer",
//      "Python int too large to convert to C ssize_t"), keepends through PyObject_IsTrue;
//   4. the method body: the separator's type, "empty separator", then the pieces (a ONE-unit piece above U+10FFFF is the
//      SystemError again, as CPython creates each piece as a str).
//
// THE PORTS (CPython 3.12 Objects/stringlib/split.h, unicodeobject.c, bytes_methods.c, stringlib/join.h)
//   split_whitespace / split_char / split, rsplit_whitespace / rsplit_char / rsplit and splitlines, statement for statement,
//   over 1-byte units (bytes) or 4-byte code points (str). unicodeobject.c's split / rsplit return [self] when the
//   separator is longer than the string or of a wider kind (PyUnicode_KIND): for a U element that piece IS the np.str_
//   scalar, which NDListArray remembers (Python prints it as np.str_('...'), bytes.join reads it as a buffer).
//   Whitespace: Py_UNICODE_ISSPACE (29 code points) for str, Py_ISSPACE (6 bytes) for bytes; line boundaries: \n \v \f \r
//   \x1c \x1d \x1e \x85 \u2028 \u2029 for str, \n \r for bytes; \r\n is one boundary.
//
// RESULTS
//   split / rsplit / splitlines: an NDListArray (NumPy's object array of lists), C-ordered over the broadcast shape.
//   join: NumPy's _to_bytes_or_str_array — the joined values as np.asarray(result.tolist()) (as wide as the longest, at
//   least 1), then cast to the CLASS of seq (StrDType(n) / BytesDType(n) / StringDType() with default parameters); an empty
//   result is result.astype(seq.dtype) (any dtype, also numeric); an OBJECT seq (an NDListArray, an object list) makes
//   NumPy 2.4.2 call ObjectDType(n), which raises "currently only the no-argument instantiation is supported; use
//   `np.dtype` instead." after every element was joined — reproduced, as the rest of NumPy's errors.
//
// THE LOOP is NDIter (C order, EXTERNAL_LOOP over the broadcast operands, each with its own dtype: no casting, no buffer),
// the phase-2 plan's technique for the object-path family ("C# helpers per element"; NumPy spends 300-1,500 ns there).
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The object path of <c>numpy.char</c> — <c>np.char.split</c>, <c>rsplit</c>, <c>splitlines</c> and <c>join</c> —
    ///     and the <c>_vec_string</c> functions of <c>numpy.strings</c> (<c>translate</c>, NDCharObjectPath.Translate.cs):
    ///     NumPy's <c>_vec_string</c> over CPython's <c>str</c> / <c>bytes</c> methods, ported element for element (see the
    ///     file header for the structure, the per-element order and the result rules).
    /// </summary>
    internal static unsafe partial class NDCharObjectPath
    {
        /// <summary>NumPy's TypeError when the first <c>_vec_string</c> argument is not text (<c>_is_user_defined_string_array</c>).</summary>
        internal const string NonStringArrayText = "string operation on non-string array";

        /// <summary>CPython's SystemError for a one-character str above U+10FFFF (<c>PyUnicode_New</c>).</summary>
        private const string InvalidMaxCharText = "invalid maximum character passed to PyUnicode_New";

        /// <summary>CPython's ValueError for an empty separator (<c>stringlib split</c>).</summary>
        private const string EmptySeparatorText = "empty separator";

        /// <summary>CPython's TypeError of <c>PySequence_Fast</c> in <c>str.join</c> / <c>bytes.join</c>.</summary>
        private const string JoinIterableText = "can only join an iterable";

        /// <summary>
        ///     NumPy 2.4.2's TypeError when <c>_to_bytes_or_str_array</c> calls <c>ObjectDType(n)</c> for an object
        ///     <c>seq</c> (a dtype class cannot be instantiated with arguments).
        /// </summary>
        internal const string ObjectDTypeText = "currently only the no-argument instantiation is supported; use `np.dtype` instead.";

        /// <summary>CPython's OverflowError of <c>PyLong_AsSsize_t</c>.</summary>
        private const string SsizeOverflowText = "Python int too large to convert to C ssize_t";

        // =============================================================================================================
        // The arguments
        // =============================================================================================================

        /// <summary>
        ///     One argument of a <c>_vec_string</c> call as NumPy holds it after <c>np.asarray</c>: an array NDIter can walk.
        ///     A NumSharp array, or the array of a converted C# value, is that array; a C# list holding null (NumPy's object
        ///     array, e.g. <c>[' ', None]</c>) is an int64 array of handles into its elements, read back per element as the
        ///     Python objects they stand for; an <see cref="NDListArray"/> (<c>join</c>'s <c>seq</c>) walks its element
        ///     ranges.
        /// </summary>
        internal sealed class VecArg : IDisposable
        {
            /// <summary>The array NDIter walks (the argument, the object handles, or the list array's <c>Starts</c>).</summary>
            internal NDArray Array { get; }

            /// <summary>NumPy's object array elements (the C# objects), or null.</summary>
            internal object[] Objects { get; }

            /// <summary>An <see cref="NDListArray"/> argument, or null.</summary>
            [NDBorrowed] // the caller's list array (join's seq): the argument only walks it, its owner disposes it
            internal NDListArray Lists { get; }

            /// <summary>Whether <see cref="Array"/> was made here (released with the argument).</summary>
            private readonly bool _owned;

            /// <summary>Wraps a converted argument.</summary>
            /// <param name="array">The array NDIter walks.</param>
            /// <param name="owned">Whether the array was made by the conversion.</param>
            /// <param name="objects">The object elements, or null.</param>
            /// <param name="lists">The list array, or null.</param>
            private VecArg(NDArray array, bool owned, object[] objects = null, NDListArray lists = null)
            {
                Array = array;
                _owned = owned;
                Objects = objects;
                Lists = lists;
            }

            /// <summary>Whether the argument is not text: an object array, a list array, or a non-string dtype.</summary>
            internal bool IsText => Objects is null && Lists is null && (Array.Storage.IsFixedString || Array.Storage.IsVString);

            /// <summary>
            ///     One Python object as the single element of a 0-d object argument, NOT converted with
            ///     <c>np.asarray</c>: CPython's <c>str.__mod__</c> / <c>bytes.__mod__</c> receive the right operand of
            ///     <c>'%s' % x</c> as the object itself (a tuple gives its items one per spec, an ndarray, a list or a dict is a
            ///     mapping), where <see cref="Convert"/> would turn a tuple or an ndarray into an array to broadcast.
            /// </summary>
            /// <param name="value">The object (null is Python's <c>None</c>; the boundary map of the element readers applies:
            ///     a C# string is a str, a <c>byte[]</c> bytes, an element stand-in the str / bytes it stands for).</param>
            /// <returns>The argument: a 0-d int64 handle array over a one-object table. It owns the handle array, which
            ///     <see cref="Dispose"/> releases; the object itself stays the caller's.</returns>
            internal static VecArg OfObject(object value) => new VecArg(NDArray.Scalar(0L), true, objects: new[] { value });

            /// <summary>
            ///     NumPy's <c>np.asarray(value)</c> of one argument under <c>numpy.char</c>'s text rule
            ///     (<see cref="CharModule.TextOperand"/>), with a C# Python literal at NumPy's default dtype (a Python int is
            ///     int64, or uint64 past int64; a float float64; a complex complex128; an int past uint64 an object element)
            ///     and null as Python's <c>None</c> (<c>np.asarray(None)</c>: a 0-d object array holding None), an
            ///     <see cref="IDictionary"/> as a dict (the 0-d object array holding it), a C# list holding a dict as an
            ///     object array, and a tuple — or a generic list of Python numbers — as the sequence it is to NumPy
            ///     (<see cref="AsPythonSequence"/>: <c>np.asarray((1, 2))</c> is the int64 array <c>[1, 2]</c>).
            /// </summary>
            /// <param name="value">The argument (null is Python's <c>None</c>).</param>
            /// <param name="name">The parameter name (for the conversion's messages).</param>
            /// <returns>The converted argument.</returns>
            /// <exception cref="ValueError">A ragged nested list (NumPy's inhomogeneous-shape text).</exception>
            /// <exception cref="NotSupportedException"><paramref name="value"/> has no NumSharp array form.</exception>
            internal static VecArg Convert(object value, string name)
            {
                if (value is null)
                    return new VecArg(NDArray.Scalar(0L), true, objects: new object[] { null });
                // A dict (translate's table): np.asarray(dict) is the 0-d object array holding it, whose scalar is the dict.
                if (value is IDictionary)
                    return new VecArg(NDArray.Scalar(0L), true, objects: new[] { value });
                if (value is NDListArray lists)
                    return new VecArg(lists.Ids, false, lists: lists);
                value = AsPythonSequence(value);
                if (value is BigInteger big)
                {
                    // A Python int of any size (numpy.char's array conversion has no BigInteger form): np.asarray gives
                    // int64, uint64 past int64, or an object 0-d array holding the int past uint64.
                    var bigArray = NormalizeLiteral(big, out object bigObject);
                    return bigArray is null
                        ? new VecArg(NDArray.Scalar(0L), true, objects: new[] { bigObject })
                        : new VecArg(bigArray, true);
                }
                if (TryObjectList(value, out var dims, out var items))
                {
                    var handles = UnmanagedStorage.ArangeObj(0L, (long)items.Length, 1L, null);
                    var shaped = dims.Length == 1 ? handles : handles.reshape(Shape.Adopt(dims));
                    if (!ReferenceEquals(shaped, handles))
                        handles.Dispose();
                    return new VecArg(shaped, true, objects: items);
                }
                if (TryNumberList(value, out var numbers))
                    return new VecArg(numbers, true);
                var nd = CharModule.TextOperand(value, name);
                bool owned = !ReferenceEquals(nd, value);
                if (nd.WasPythonLiteral && nd.ndim == 0)
                {
                    // np.asarray(<Python scalar>): a Python int is int64 (uint64 past int64, an object past uint64), a float
                    // float64, a complex complex128 — the dtype whose name and bytes the methods then see.
                    var literal = nd.Nep50Operand;
                    var normalized = NormalizeLiteral(literal, out object pyObject);
                    if (owned)
                        nd.Dispose();
                    if (normalized is null)
                    {
                        // An int past uint64: NumPy's object 0-d array holding the Python int.
                        var handle = NDArray.Scalar(0L);
                        return new VecArg(handle, true, objects: new[] { pyObject });
                    }
                    return new VecArg(normalized, true);
                }
                return new VecArg(nd, owned);
            }

            /// <summary>
            ///     A tuple, an array holding tuples, or a generic list of Python numbers read as the Python sequence it stands
            ///     for: an <c>object[]</c> of its items (every tuple at any depth of a one-dimensional array as a nested
            ///     <c>object[]</c>), so the array rules of <see cref="Convert"/> see it as they see a C# array of the same
            ///     items — a list of ints is int64 (NumPy's <c>np.asarray([1, 2])</c>), not the C# element type's int32; a list
            ///     of pairs is two-dimensional; a tuple next to a None is NumPy's ragged-shape error. Any other value is
            ///     returned unchanged.
            /// </summary>
            /// <param name="value">The argument.</param>
            /// <returns>The <c>object[]</c> of a tuple, of an array holding tuples or of a generic list of numbers, else
            ///     <paramref name="value"/>.</returns>
            /// <remarks>
            ///     A tuple is always a sequence to NumPy's array coercion (<c>np.asarray([(1, 2), (3, 4)])</c> is a 2x2 int64
            ///     array; one tuple per element needs an object array, which no C# value spells), and NumSharp's array
            ///     conversion takes no value tuple, so this is the only reading a tuple gets. A generic list is converted only
            ///     when every item is a C# number or bool — the items whose dtype NumPy's discovery decides differently from
            ///     the C# element type; a list of strings, of bytes or of containers keeps the conversion it has always had.
            /// </remarks>
            private static object AsPythonSequence(object value)
            {
                switch (value)
                {
                    case ITuple tuple:
                    {
                        var items = new object[tuple.Length];
                        for (int i = 0; i < items.Length; i++)
                            items[i] = AsPythonSequence(tuple[i]);
                        return items;
                    }
                    case System.Array array when array.Rank == 1 && array is not byte[] && HoldsTuple(array):
                    {
                        var items = new object[array.Length];
                        int k = 0;
                        foreach (var item in array)
                            items[k++] = AsPythonSequence(item);
                        return items;
                    }
                    case IList list when value is not System.Array && list.Count > 0:
                    {
                        foreach (var item in list)
                        {
                            if (item is not (bool or sbyte or byte or short or ushort or int or uint or long or ulong
                                    or float or double or Complex or BigInteger))
                                return value;
                        }
                        var items = new object[list.Count];
                        list.CopyTo(items, 0);
                        return items;
                    }
                    default:
                        return value;
                }
            }

            /// <summary>Whether a one-dimensional array holds a tuple, directly or inside a nested one-dimensional array.</summary>
            /// <param name="array">The array.</param>
            /// <returns>True when a tuple is found.</returns>
            private static bool HoldsTuple(System.Array array)
            {
                foreach (var item in array)
                {
                    if (item is ITuple)
                        return true;
                    if (item is System.Array inner && inner.Rank == 1 && inner is not byte[] && HoldsTuple(inner))
                        return true;
                }
                return false;
            }

            /// <summary>
            ///     The NumPy array of a Python literal: int64 / uint64 / float64 / complex128, or null with the Python object
            ///     for an int no NumPy integer holds.
            /// </summary>
            /// <param name="literal">The boxed C# literal.</param>
            /// <param name="pyObject">The literal when no array holds it (else null).</param>
            /// <returns>The 0-d array, or null.</returns>
            private static NDArray NormalizeLiteral(object literal, out object pyObject)
            {
                pyObject = null;
                switch (literal)
                {
                    case float f:
                        return NDArray.Scalar((double)f);
                    case double d:
                        return NDArray.Scalar(d);
                    case Complex c:
                        return NDArray.Scalar(c);
                    case ulong u when u > long.MaxValue:
                        return NDArray.Scalar(u);
                    case BigInteger bi:
                        if (bi >= long.MinValue && bi <= long.MaxValue)
                            return NDArray.Scalar((long)bi);
                        if (bi > 0 && bi <= ulong.MaxValue)
                            return NDArray.Scalar((ulong)bi);
                        pyObject = bi;
                        return null;
                    default:
                        return NDArray.Scalar(System.Convert.ToInt64(literal));
                }
            }

            /// <summary>
            ///     Whether <paramref name="value"/> is a C# list NumPy would make an OBJECT array of: a C# array (not a
            ///     <c>byte[]</c>, which is one bytes value) or a non-generic list, rectangular, holding at least one null or a
            ///     Python int that neither int64 nor uint64 holds (<see cref="NeedsObjectArray"/>).
            /// </summary>
            /// <param name="value">The argument.</param>
            /// <param name="dims">The list's shape.</param>
            /// <param name="items">Its elements in C order.</param>
            /// <returns>True for an object list.</returns>
            /// <exception cref="ValueError">A ragged list holding null (NumPy's inhomogeneous-shape text).</exception>
            private static bool TryObjectList(object value, out long[] dims, out object[] items)
            {
                dims = null;
                items = null;
                Array array = value switch
                {
                    byte[] => null,
                    Array a => a,
                    IList list => ToObjectArray(list),
                    _ => null,
                };
                if (array is null || !NeedsObjectArray(array))
                    return false;
                var found = new List<long>();
                var values = new List<object>();
                Backends.UnmanagedStorage.CollectValues(array, found, values);
                dims = found.ToArray();
                items = values.ToArray();
                return true;
            }

            /// <summary>A non-generic list as an <c>object[]</c> (one level; nested arrays stay as they are).</summary>
            /// <param name="list">The list.</param>
            /// <returns>The array.</returns>
            private static object[] ToObjectArray(IList list)
            {
                var a = new object[list.Count];
                list.CopyTo(a, 0);
                return a;
            }

            /// <summary>
            ///     Whether a C# array holds, at any depth, an element NumPy's array coercion can only hold in an OBJECT array:
            ///     null (Python's None), a dict (an <see cref="IDictionary"/>: not a sequence to the coercion, so a leaf) or a
            ///     Python int outside [int64.min, uint64.max] (<c>np.asarray([1, 2**64])</c> is an object array).
            /// </summary>
            /// <param name="array">The array.</param>
            /// <returns>True when such an element exists.</returns>
            private static bool NeedsObjectArray(Array array)
            {
                foreach (var item in array)
                {
                    if (item is null || item is IDictionary)
                        return true;
                    if (item is BigInteger bi && (bi < long.MinValue || bi > ulong.MaxValue))
                        return true;
                    if (item is Array child && item is not byte[] && NeedsObjectArray(child))
                        return true;
                }
                return false;
            }

            /// <summary>
            ///     NumPy's <c>np.asarray</c> of a C# list of Python numbers (no text, no None): the array of the dtype NumPy's
            ///     array coercion discovers for its elements — bool for bools alone; int64 for Python ints (a bool among them
            ///     is an int); uint64 for ints past int64 alone; float64 once an int64 meets a uint64 int (their promotion:
            ///     <c>np.asarray([1, 2**63])</c> is float64) or a float appears; complex128 once a complex appears; float64 for
            ///     an empty list. A list holding anything else (text, a NumPy scalar such as <c>Half</c>) is not one: it goes
            ///     on to the text conversion.
            /// </summary>
            /// <param name="value">The argument.</param>
            /// <param name="array">Receives the array (null when the value is not such a list).</param>
            /// <returns>True for a list of Python numbers.</returns>
            /// <exception cref="ValueError">A ragged list (NumPy's inhomogeneous-shape text).</exception>
            private static bool TryNumberList(object value, out NDArray array)
            {
                array = null;
                if (value is not Array source || value is byte[])
                    return false;
                var dims = new List<long>();
                var leaves = new List<object>();
                Backends.UnmanagedStorage.CollectValues(source, dims, leaves);
                bool anyInt = false, anyUInt = false, anyFloat = false, anyComplex = false;
                foreach (var leaf in leaves)
                {
                    switch (leaf)
                    {
                        case bool:
                            break;   // alone they keep the bool dtype (the last branch); among numbers they promote
                        case sbyte or byte or short or ushort or int or uint or long:
                            anyInt = true;
                            break;
                        case ulong u:
                            // A Python int past int64 is discovered as uint64.
                            if (u > long.MaxValue)
                                anyUInt = true;
                            else
                                anyInt = true;
                            break;
                        case BigInteger bi:
                            if (bi >= long.MinValue && bi <= long.MaxValue)
                                anyInt = true;
                            else if (bi > 0 && bi <= ulong.MaxValue)
                                anyUInt = true;
                            else
                                return false;   // an object element: TryObjectList's case
                            break;
                        case float or double:
                            anyFloat = true;
                            break;
                        case Complex:
                            anyComplex = true;
                            break;
                        default:
                            return false;
                    }
                }
                Array typed;
                if (anyComplex)
                    typed = leaves.Select(PythonComplex).ToArray();
                else if (anyFloat || (anyInt && anyUInt) || leaves.Count == 0)
                    typed = leaves.Select(x => PythonComplex(x).Real).ToArray();
                else if (anyUInt)
                    typed = leaves.Select(x => x is bool b ? (b ? 1UL : 0UL) : x is BigInteger bi ? (ulong)bi : System.Convert.ToUInt64(x)).ToArray();
                else if (anyInt)
                    typed = leaves.Select(x => x is bool b ? (b ? 1L : 0L) : x is BigInteger bi ? (long)bi : System.Convert.ToInt64(x)).ToArray();
                else
                    typed = leaves.Select(x => (bool)x).ToArray();
                var flat = np.array(typed);
                if (dims.Count == 1)
                {
                    array = flat;
                    return true;
                }
                array = flat.reshape(Shape.Adopt(dims.ToArray()));
                if (!ReferenceEquals(array, flat))
                    flat.Dispose();
                return true;
            }

            /// <summary>
            ///     A Python number of a list as a complex128 value — an int correctly rounded to the nearest double (through
            ///     the 64-bit integer conversions: <c>(double)BigInteger</c> truncates), a bool as 0 / 1.
            /// </summary>
            /// <param name="x">The number (one of the kinds <see cref="TryNumberList"/> accepts).</param>
            /// <returns>The value.</returns>
            private static Complex PythonComplex(object x) => x switch
            {
                Complex c => c,
                bool b => b ? 1.0 : 0.0,
                float f => f,
                double d => d,
                ulong u => u,
                BigInteger bi => bi >= long.MinValue && bi <= long.MaxValue ? (double)(long)bi : (double)(ulong)bi,
                _ => (double)System.Convert.ToInt64(x),
            };

            /// <summary>The argument's shape for NumPy's broadcast (the outer shape of a list array).</summary>
            internal Shape BroadcastShape => Array.Shape;

            /// <summary>Releases what the conversion made.</summary>
            public void Dispose()
            {
                if (_owned)
                    Array.Dispose();
            }
        }

        // =============================================================================================================
        // The Python object of one element
        // =============================================================================================================

        /// <summary>The Python type of the object <c>PyArray_ToScalar</c> made of one element.</summary>
        private enum ElemKind : byte
        {
            /// <summary><c>np.str_</c> (a <c>U</c> element): a str with the buffer protocol (its UCS-4 code units).</summary>
            NpStr,

            /// <summary>A Python <c>str</c> (a <c>StringDType</c> element, a C# string of an object list): no buffer.</summary>
            Str,

            /// <summary><c>np.bytes_</c> (an <c>S</c> element): bytes with the buffer protocol.</summary>
            NpBytes,

            /// <summary>A Python <c>bytes</c> (a <c>byte[]</c> of an object list).</summary>
            Bytes,

            /// <summary>A NumPy numeric scalar (<c>np.int64</c>, <c>np.bool</c>, <c>np.float64</c>): its raw bytes as a buffer.</summary>
            Number,

            /// <summary>Any other Python object (None, a Python int / float / bool, a list, a <c>StringDType</c> NA object).</summary>
            Object,
        }

        /// <summary>The Python object of one element (a view into the operand's memory or its scratch).</summary>
        private struct Elem
        {
            /// <summary>The object's type.</summary>
            internal ElemKind Kind;

            /// <summary>A text kind's code units (4-byte code points for <c>str</c> kinds, bytes otherwise).</summary>
            internal byte* Units;

            /// <summary>Their number.</summary>
            internal long Length;

            /// <summary>A <see cref="ElemKind.Number"/>'s dtype.</summary>
            internal NPTypeCode Number;

            /// <summary>A <see cref="ElemKind.Number"/>'s raw element bytes.</summary>
            internal byte* Raw;

            /// <summary>An <see cref="ElemKind.Object"/>'s object (null is None).</summary>
            internal object Obj;
        }

        /// <summary>The kind of an operand, decided once per call.</summary>
        private enum OperandKind : byte
        {
            /// <summary>An <c>S</c> array.</summary>
            Bytes,

            /// <summary>A <c>U</c> array.</summary>
            Unicode,

            /// <summary>A <c>StringDType</c> array.</summary>
            VString,

            /// <summary>A numeric or bool array.</summary>
            Number,

            /// <summary>NumPy's object array (handles into C# objects).</summary>
            Objects,
        }

        /// <summary>
        ///     Reads the elements of one operand as Python objects: the operand's kind and widths, its allocator for
        ///     <c>StringDType</c>, and the native scratch a decoded text lives in until the next element.
        /// </summary>
        private sealed class Operand : IDisposable
        {
            /// <summary>The operand's kind.</summary>
            internal readonly OperandKind Kind;

            /// <summary>An <c>S</c> / <c>U</c> element width in code units.</summary>
            internal readonly int Chars;

            /// <summary>A numeric operand's dtype.</summary>
            internal readonly NPTypeCode Number;

            /// <summary>A <c>StringDType</c> operand's allocator (locked by the caller around the walk).</summary>
            internal readonly VStringAllocator Allocator;

            /// <summary>A <c>StringDType</c> operand's descriptor parameters (its NA object).</summary>
            internal readonly VStringParameters Params;

            /// <summary>An object operand's elements.</summary>
            internal readonly object[] Objects;

            /// <summary>
            ///     The pool buffer behind <see cref="_points"/> (a NumSharp array, so the scratch comes from and returns to the
            ///     buffer pool like every other buffer of the call), or null before the first decoded text.
            /// </summary>
            private NDArray _pointsArray;

            /// <summary>Decoded code points of the current element (a <c>StringDType</c> text, an object string).</summary>
            private uint* _points;

            /// <summary>The capacity of <see cref="_points"/>.</summary>
            private long _pointsCap;

            /// <summary>The pool buffer behind <see cref="_bytes"/>, or null before the first copied value.</summary>
            private NDArray _bytesArray;

            /// <summary>Copied bytes of the current element (an object <c>byte[]</c>, a converted number).</summary>
            private byte* _bytes;

            /// <summary>The capacity of <see cref="_bytes"/>.</summary>
            private long _bytesCap;

            /// <summary>Describes the operand of a converted argument.</summary>
            /// <param name="arg">The argument.</param>
            internal Operand(VecArg arg)
            {
                Objects = arg.Objects;
                var storage = arg.Array.Storage;
                if (Objects is not null)
                {
                    Kind = OperandKind.Objects;
                }
                else if (storage.IsFixedString)
                {
                    Kind = storage.StringSlice.IsUnicode ? OperandKind.Unicode : OperandKind.Bytes;
                    Chars = storage.StringSlice.Chars;
                }
                else if (storage.IsVString)
                {
                    Kind = OperandKind.VString;
                    Allocator = storage.VStringSlice.Allocator;
                    Params = storage.VStringSlice.Descr.VString;
                }
                else
                {
                    Kind = OperandKind.Number;
                    Number = storage.TypeCode;
                }
            }

            /// <summary>Whether the operand reads as Python <c>str</c> objects (a <c>U</c> or <c>StringDType</c> array).</summary>
            internal bool IsStrClass => Kind is OperandKind.Unicode or OperandKind.VString;

            /// <summary>
            ///     The Python object of the element at <paramref name="p"/> (<c>PyArray_ToScalar</c>): valid until the next
            ///     read of this operand.
            /// </summary>
            /// <param name="p">The element's address (an NDIter data pointer).</param>
            /// <param name="e">Receives the object.</param>
            /// <exception cref="SystemError">A <c>U</c> element whose text is one code unit above U+10FFFF.</exception>
            /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell.</exception>
            internal void Read(byte* p, out Elem e)
            {
                e = default;
                switch (Kind)
                {
                    case OperandKind.Bytes:
                        e.Kind = ElemKind.NpBytes;
                        e.Units = p;
                        e.Length = NDCharArray.TrimmedUnits(p, Chars, false);
                        return;
                    case OperandKind.Unicode:
                        e.Kind = ElemKind.NpStr;
                        e.Units = p;
                        e.Length = NDCharArray.TrimmedUnits(p, Chars, true);
                        // np.str_ of a one-unit text above U+10FFFF: PyUnicode_New refuses it (a longer text caps its max).
                        if (e.Length == 1 && *(uint*)p > 0x10FFFF)
                            throw new SystemError(InvalidMaxCharText);
                        return;
                    case OperandKind.VString:
                        ReadVString(p, ref e);
                        return;
                    case OperandKind.Number:
                        e.Kind = ElemKind.Number;
                        e.Number = Number;
                        e.Raw = p;
                        if (Number == NPTypeCode.Decimal)
                        {
                            // NumSharp's decimal is NumPy's float64 (its scalar, its name, its 8 bytes).
                            byte* slot = Bytes(8);
                            *(double*)slot = (double)*(decimal*)p;
                            e.Number = NPTypeCode.Double;
                            e.Raw = slot;
                        }
                        return;
                    default:
                        ReadObject(Objects[*(long*)p], ref e);
                        return;
                }
            }

            /// <summary>A <c>StringDType</c> element (<c>stringdtype_getitem</c>): a str, or the NA object.</summary>
            /// <param name="cell">The cell.</param>
            /// <param name="e">Receives the object.</param>
            /// <exception cref="OutOfMemoryException">A corrupt cell.</exception>
            private void ReadVString(byte* cell, ref Elem e)
            {
                int isNull = Allocator.load(cell, out byte* buf, out long size);
                if (isNull < 0)
                    throw new OutOfMemoryException("Failed to load string");
                if (isNull == 1)
                {
                    // An NA reads as the NA object itself (a string NA is that string); without an NA object as ''.
                    if (Params.HasNAObject && Params.na_object is not string)
                    {
                        e.Kind = ElemKind.Object;
                        e.Obj = Params.na_object;
                        return;
                    }
                    SetString(Params.HasNAObject ? (string)Params.na_object : string.Empty, ref e);
                    return;
                }
                e.Kind = ElemKind.Str;
                e.Units = (byte*)DecodeUtf8(buf, size, out long n);
                e.Length = n;
            }

            /// <summary>An object-list element as the Python object it stands for (the house boundary map): a C# string or
            ///     char a str, an element stand-in the str / bytes it stands for, a <c>byte[]</c> bytes, a <see cref="Half"/> an
            ///     np.float16, anything else the object itself.</summary>
            /// <param name="obj">The C# value (null is None).</param>
            /// <param name="e">Receives the object.</param>
            private void ReadObject(object obj, ref Elem e)
            {
                switch (obj)
                {
                    case string s:
                        SetString(s, ref e);
                        return;
                    case char c:
                        SetString(c.ToString(), ref e);
                        return;
                    case NDCharArray standIn when standIn.IsScalarStandIn:
                        // An element stand-in IS the Python str / bytes chararray element access returns (the boundary
                        // map), not the 0-d array that carries it: '%r' of it is "'ab'", never "array('ab', dtype='<U2')".
                        ReadObject(standIn.tolist(), ref e);
                        return;
                    case byte[] b:
                        e.Kind = ElemKind.Bytes;
                        e.Units = Bytes(b.Length);
                        b.AsSpan().CopyTo(new Span<byte>(e.Units, b.Length));
                        e.Length = b.Length;
                        return;
                    case Half h:
                        // np.float16 is a NumPy scalar (its buffer is its two bytes).
                        e.Kind = ElemKind.Number;
                        e.Number = NPTypeCode.Half;
                        e.Raw = Bytes(2);
                        *(Half*)e.Raw = h;
                        return;
                    default:
                        e.Kind = ElemKind.Object;
                        e.Obj = obj;
                        return;
                }
            }

            /// <summary>A Python str of a .NET string (its UTF-16 code units joined into code points, lone surrogates kept).</summary>
            /// <param name="s">The string.</param>
            /// <param name="e">Receives the object.</param>
            private void SetString(string s, ref Elem e)
            {
                e.Kind = ElemKind.Str;
                uint* dst = Points(s.Length);
                long n = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    char c = s[i];
                    if (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                    {
                        dst[n++] = (uint)char.ConvertToUtf32(c, s[i + 1]);
                        i++;
                    }
                    else
                    {
                        dst[n++] = c;
                    }
                }
                e.Units = (byte*)dst;
                e.Length = n;
            }

            /// <summary>Decodes UTF-8 text (valid: a <c>StringDType</c> stores only valid UTF-8) into code points.</summary>
            /// <param name="src">The text.</param>
            /// <param name="size">Its size in bytes.</param>
            /// <param name="count">Receives the number of code points.</param>
            /// <returns>The code points (the operand's scratch).</returns>
            private uint* DecodeUtf8(byte* src, long size, out long count)
            {
                uint* dst = Points(size);
                long n = 0;
                long i = 0;
                while (i < size)
                {
                    byte b = src[i];
                    if (b < 0x80)
                    {
                        dst[n++] = b;
                        i++;
                        continue;
                    }
                    var span = new ReadOnlySpan<byte>(src + i, (int)Math.Min(size - i, 4));
                    System.Text.Rune.DecodeFromUtf8(span, out var rune, out int consumed);
                    dst[n++] = (uint)rune.Value;
                    i += consumed;
                }
                count = n;
                return dst;
            }

            /// <summary>The code-point scratch with room for <paramref name="n"/> code points.</summary>
            /// <param name="n">The capacity needed.</param>
            /// <returns>The scratch.</returns>
            private uint* Points(long n)
            {
                if (n > _pointsCap)
                {
                    // Grown by doubling into a fresh pool buffer: the previous text is dead by now (a read replaces it), so
                    // nothing is copied over.
                    long cap = Math.Max(n, Math.Max(64, _pointsCap * 2));
                    var grown = new NDArray(NPTypeCode.UInt32, Shape.Vector(cap), false);
                    _pointsArray?.Dispose();
                    _pointsArray = grown;
                    _points = (uint*)grown.Storage.Address;
                    _pointsCap = cap;
                }
                return _points;
            }

            /// <summary>The byte scratch with room for <paramref name="n"/> bytes.</summary>
            /// <param name="n">The capacity needed.</param>
            /// <returns>The scratch.</returns>
            private byte* Bytes(long n)
            {
                if (n > _bytesCap)
                {
                    long cap = Math.Max(n, Math.Max(64, _bytesCap * 2));
                    var grown = new NDArray(NPTypeCode.Byte, Shape.Vector(cap), false);
                    _bytesArray?.Dispose();
                    _bytesArray = grown;
                    _bytes = (byte*)grown.Storage.Address;
                    _bytesCap = cap;
                }
                return _bytes;
            }

            /// <summary>Returns the scratch buffers to the pool.</summary>
            public void Dispose()
            {
                _pointsArray?.Dispose();
                _bytesArray?.Dispose();
                _pointsArray = _bytesArray = null;
                _points = null;
                _bytes = null;
                _pointsCap = _bytesCap = 0;
            }
        }

        // =============================================================================================================
        // The Python protocols an element meets
        // =============================================================================================================

        /// <summary>CPython's <c>Py_TYPE(obj)-&gt;tp_name</c> of an element, as the TypeErrors quote it.</summary>
        /// <param name="e">The element.</param>
        /// <returns><c>numpy.str_</c>, <c>str</c>, <c>numpy.bytes_</c>, <c>bytes</c>, <c>numpy.int64</c>, <c>NoneType</c> ...</returns>
        private static string TypeName(in Elem e) => e.Kind switch
        {
            ElemKind.NpStr => "numpy.str_",
            ElemKind.Str => "str",
            ElemKind.NpBytes => "numpy.bytes_",
            ElemKind.Bytes => "bytes",
            ElemKind.Number => "numpy." + e.Number.AsNumpyDtypeName(),
            _ => NDPyObject.TypeName(e.Obj),
        };

        /// <summary>Whether the element is a str instance (<c>PyUnicode_Check</c>).</summary>
        /// <param name="e">The element.</param>
        /// <returns>True for <c>np.str_</c> and Python <c>str</c>.</returns>
        private static bool IsStr(in Elem e) => e.Kind is ElemKind.NpStr or ElemKind.Str;

        /// <summary>Whether the element is None (an object-list null, a <c>StringDType</c> NA object None).</summary>
        /// <param name="e">The element.</param>
        /// <returns>True for None.</returns>
        private static bool IsNone(in Elem e) => e.Kind == ElemKind.Object && e.Obj is null;

        /// <summary>
        ///     The buffer protocol (<c>PyObject_GetBuffer</c>, <c>PyBUF_SIMPLE</c>): the bytes a buffer-capable object
        ///     exposes — <c>np.str_</c> its UCS-4 code units, <c>np.bytes_</c> / <c>bytes</c> their bytes, a NumPy number its
        ///     raw element bytes. A Python <c>str</c>, None and the other Python objects have none.
        /// </summary>
        /// <param name="e">The element.</param>
        /// <param name="data">Receives the bytes.</param>
        /// <param name="length">Receives their number.</param>
        /// <returns>True when the object has a buffer.</returns>
        private static bool TryBuffer(in Elem e, out byte* data, out long length)
        {
            switch (e.Kind)
            {
                case ElemKind.NpStr:
                    data = e.Units;
                    length = e.Length * 4;
                    return true;
                case ElemKind.NpBytes:
                case ElemKind.Bytes:
                    data = e.Units;
                    length = e.Length;
                    return true;
                case ElemKind.Number:
                    data = e.Raw;
                    length = e.Number.SizeOf();
                    return true;
                default:
                    data = null;
                    length = 0;
                    return false;
            }
        }

        /// <summary>
        ///     CPython's <c>_PyNumber_Index</c> + <c>PyLong_AsSsize_t</c> of an argument (a <c>maxsplit</c>): a NumPy integer
        ///     (NumSharp's <c>Char</c> is <c>uint16</c>) or a Python int / bool.
        /// </summary>
        /// <param name="e">The element.</param>
        /// <returns>The integer.</returns>
        /// <exception cref="TypeError">No <c>__index__</c>: <c>'numpy.float64' object cannot be interpreted as an integer</c>
        ///     (<c>np.bool</c> has none in NumPy 2).</exception>
        /// <exception cref="OverflowException">The integer exceeds <c>Py_ssize_t</c> (<c>Python int too large to convert to C
        ///     ssize_t</c>).</exception>
        private static long Index(in Elem e)
        {
            if (e.Kind == ElemKind.Number)
            {
                switch (e.Number)
                {
                    case NPTypeCode.SByte: return *(sbyte*)e.Raw;
                    case NPTypeCode.Byte: return *e.Raw;
                    case NPTypeCode.Int16: return *(short*)e.Raw;
                    case NPTypeCode.UInt16: return *(ushort*)e.Raw;
                    case NPTypeCode.Char: return *(ushort*)e.Raw;
                    case NPTypeCode.Int32: return *(int*)e.Raw;
                    case NPTypeCode.UInt32: return *(uint*)e.Raw;
                    case NPTypeCode.Int64: return *(long*)e.Raw;
                    case NPTypeCode.UInt64:
                        ulong u = *(ulong*)e.Raw;
                        if (u > long.MaxValue)
                            throw new OverflowException(SsizeOverflowText);
                        return (long)u;
                }
            }
            else if (e.Kind == ElemKind.Object)
            {
                switch (e.Obj)
                {
                    case bool b: return b ? 1 : 0;   // a Python bool is an int
                    case sbyte v: return v;
                    case byte v: return v;
                    case short v: return v;
                    case ushort v: return v;
                    case int v: return v;
                    case uint v: return v;
                    case long v: return v;
                    case ulong v:
                        if (v > long.MaxValue)
                            throw new OverflowException(SsizeOverflowText);
                        return (long)v;
                    case BigInteger v:
                        if (v < long.MinValue || v > long.MaxValue)
                            throw new OverflowException(SsizeOverflowText);
                        return (long)v;
                }
            }
            throw new TypeError($"'{TypeName(e)}' object cannot be interpreted as an integer");
        }

        /// <summary>CPython's <c>PyObject_IsTrue</c> of an argument (a <c>keepends</c>).</summary>
        /// <param name="e">The element.</param>
        /// <returns>The truth value (a NaN is true, an empty text false).</returns>
        private static bool Truth(in Elem e)
        {
            switch (e.Kind)
            {
                case ElemKind.NpStr:
                case ElemKind.Str:
                case ElemKind.NpBytes:
                case ElemKind.Bytes:
                    return e.Length > 0;
                case ElemKind.Number:
                    switch (e.Number)
                    {
                        case NPTypeCode.Boolean: return *e.Raw != 0;
                        case NPTypeCode.Half: return *(Half*)e.Raw != Half.Zero || Half.IsNaN(*(Half*)e.Raw);
                        case NPTypeCode.Single: return *(float*)e.Raw != 0f || float.IsNaN(*(float*)e.Raw);
                        case NPTypeCode.Double: return *(double*)e.Raw != 0d || double.IsNaN(*(double*)e.Raw);
                        case NPTypeCode.Complex:
                            var c = *(Complex*)e.Raw;
                            return c.Real != 0 || c.Imaginary != 0 || double.IsNaN(c.Real) || double.IsNaN(c.Imaginary);
                        default:
                            // Every integer width is nonzero when any of its bytes is.
                            int size = e.Number.SizeOf();
                            for (int i = 0; i < size; i++)
                                if (e.Raw[i] != 0)
                                    return true;
                            return false;
                    }
                default:
                    return e.Obj switch
                    {
                        null => false,
                        bool b => b,
                        double d => d != 0 || double.IsNaN(d),
                        float f => f != 0 || float.IsNaN(f),
                        decimal m => m != 0,
                        Complex c => c != Complex.Zero || double.IsNaN(c.Real) || double.IsNaN(c.Imaginary),
                        BigInteger bi => !bi.IsZero,
                        sbyte or byte or short or ushort or int or uint or long or ulong => System.Convert.ToDecimal(e.Obj) != 0,
                        ICollection col => col.Count > 0,
                        _ => true,
                    };
            }
        }

        /// <summary>CPython's <c>PyUnicode_KIND</c> of a str: 1 below U+0100, 2 below U+10000, 4 otherwise (1 when empty).</summary>
        /// <param name="units">The code points.</param>
        /// <param name="length">Their number.</param>
        /// <returns>1, 2 or 4.</returns>
        private static int StrKind(uint* units, long length)
        {
            uint max = 0;
            for (long i = 0; i < length; i++)
                if (units[i] > max)
                    max = units[i];
            return max < 0x100 ? 1 : max < 0x10000 ? 2 : 4;
        }

        /// <summary>
        ///     Code points as a .NET string (NumPy's <c>tolist()</c> of a str piece): above U+FFFF as a surrogate pair, a lone
        ///     surrogate unit kept.
        /// </summary>
        /// <param name="units">The code points.</param>
        /// <param name="count">Their number.</param>
        /// <returns>The string.</returns>
        /// <exception cref="SystemError">A unit above U+10FFFF (no .NET string holds it; NumSharp's U element access refuses
        ///     it the same way).</exception>
        internal static string Ucs4ToString(uint* units, int count)
        {
            var sb = new System.Text.StringBuilder(count);
            for (int i = 0; i < count; i++)
            {
                uint u = units[i];
                if (u > 0x10FFFF)
                    throw new SystemError(InvalidMaxCharText);
                if (u > 0xFFFF)
                    sb.Append(char.ConvertFromUtf32((int)u));
                else
                    sb.Append((char)u);
            }
            return sb.ToString();
        }

        // =============================================================================================================
        // CPython's split family, statement for statement (stringlib/split.h)
        // =============================================================================================================

        /// <summary>Whitespace of the unit's class: <c>Py_ISSPACE</c> for a byte, <c>Py_UNICODE_ISSPACE</c> for a code point.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="u">The unit.</param>
        /// <returns>True for whitespace.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsSpace<TUnit>(TUnit u) where TUnit : unmanaged
            => typeof(TUnit) == typeof(byte)
                ? NDCharArray.IsBytesWhitespace(Unsafe.As<TUnit, byte>(ref u))
                : NDCharArray.IsPyWhitespace(Unsafe.As<TUnit, uint>(ref u));

        /// <summary>A line boundary of the unit's class (<c>STRINGLIB_ISLINEBREAK</c>).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="u">The unit.</param>
        /// <returns>True for <c>\n \r</c> (bytes) or the eleven str boundaries.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsLineBreak<TUnit>(TUnit u) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(byte))
            {
                byte b = Unsafe.As<TUnit, byte>(ref u);
                return b == (byte)'\n' || b == (byte)'\r';
            }
            uint c = Unsafe.As<TUnit, uint>(ref u);
            return c <= 0x0D ? c >= 0x0A : c is 0x1C or 0x1D or 0x1E or 0x85 or 0x2028 or 0x2029;
        }

        /// <summary>The unit's value as a <see cref="uint"/> (for the <c>\r\n</c> test).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="u">The unit.</param>
        /// <returns>The value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Value<TUnit>(TUnit u) where TUnit : unmanaged
            => typeof(TUnit) == typeof(byte) ? Unsafe.As<TUnit, byte>(ref u) : Unsafe.As<TUnit, uint>(ref u);

        /// <summary>
        ///     <c>split_whitespace</c>: the runs of non-whitespace, at most <paramref name="maxcount"/> splits; the rest (its
        ///     leading whitespace skipped, its trailing kept) is the last piece.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="len">Its length.</param>
        /// <param name="maxcount">The split limit (<see cref="long.MaxValue"/> for none).</param>
        /// <param name="sink">Receives the pieces in order.</param>
        private static void SplitWhitespace<TUnit>(TUnit* s, long len, long maxcount, PieceSink sink) where TUnit : unmanaged
        {
            long i = 0, j;
            while (maxcount-- > 0)
            {
                while (i < len && IsSpace(s[i]))
                    i++;
                if (i == len)
                    break;
                j = i;
                i++;
                while (i < len && !IsSpace(s[i]))
                    i++;
                sink.Add((byte*)(s + j), i - j);
            }
            if (i < len)
            {
                // Only occurs when maxcount was reached: skip the whitespace, the rest is one piece.
                while (i < len && IsSpace(s[i]))
                    i++;
                if (i != len)
                    sink.Add((byte*)(s + i), len - i);
            }
        }

        /// <summary><c>split_char</c> / <c>split</c>: the pieces between the occurrences of a separator, at most
        ///     <paramref name="maxcount"/> splits.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="len">Its length.</param>
        /// <param name="sep">The separator.</param>
        /// <param name="sepLen">Its length (at least 1).</param>
        /// <param name="maxcount">The split limit.</param>
        /// <param name="sink">Receives the pieces in order.</param>
        private static void SplitSeparator<TUnit>(TUnit* s, long len, TUnit* sep, long sepLen, long maxcount, PieceSink sink)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            long i = 0;
            while (maxcount-- > 0)
            {
                // FASTSEARCH from i: the leftmost occurrence (split_char for a one-unit separator).
                long pos = sepLen == 1 ? FindUnit(s + i, len - i, *sep) : FindNeedle(s + i, len - i, sep, sepLen);
                if (pos < 0)
                    break;
                long j = i + pos;
                sink.Add((byte*)(s + i), j - i);
                i = j + sepLen;
            }
            sink.Add((byte*)(s + i), len - i);
        }

        /// <summary>
        ///     FASTSEARCH's general case: the leftmost occurrence of <paramref name="needle"/> (two units or more) in
        ///     <paramref name="s"/>[0 : <paramref name="n"/>), or -1. A short text runs a candidate scan (the needle's first
        ///     unit through <see cref="FindUnit{TUnit}"/>, then the rest compared); a long one CoreLib's vectorized search.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="n">Its length in units.</param>
        /// <param name="needle">The separator.</param>
        /// <param name="m">Its length (at least 2).</param>
        /// <returns>The offset of the occurrence, or -1.</returns>
        /// <remarks>
        ///     Any exact search returns the same leftmost occurrence, so the two routes agree by definition. The candidate scan
        ///     is quadratic on adversarial texts (a run of <c>a</c> searched for <c>aab</c>), so texts past 256 units take
        ///     CoreLib's search, whose first-and-last-unit filter keeps such texts fast.
        /// </remarks>
        private static long FindNeedle<TUnit>(TUnit* s, long n, TUnit* needle, long m) where TUnit : unmanaged, IEquatable<TUnit>
        {
            if (n > 256)
                return new ReadOnlySpan<TUnit>(s, checked((int)n)).IndexOf(new ReadOnlySpan<TUnit>(needle, checked((int)m)));
            long last = n - m;
            long i = 0;
            while (i <= last)
            {
                // A candidate start in [i, last]: the needle's first unit.
                long k = FindUnit(s + i, last - i + 1, *needle);
                if (k < 0)
                    return -1;
                i += k;
                if (UnitsEqual(s + i + 1, needle + 1, m - 1))
                    return i;
                i++;
            }
            return -1;
        }

        /// <summary>
        ///     FASTSEARCH's reverse general case: the rightmost occurrence of <paramref name="needle"/> (two units or more)
        ///     that ends at or before <paramref name="n"/> in <paramref name="s"/>, or -1 (the candidates scanned backward).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="n">The end of the searched range.</param>
        /// <param name="needle">The separator.</param>
        /// <param name="m">Its length (at least 2).</param>
        /// <returns>The offset of the occurrence, or -1.</returns>
        /// <remarks>See <see cref="FindNeedle{TUnit}"/> for the two routes.</remarks>
        private static long FindNeedleLast<TUnit>(TUnit* s, long n, TUnit* needle, long m) where TUnit : unmanaged, IEquatable<TUnit>
        {
            if (n > 256)
                return new ReadOnlySpan<TUnit>(s, checked((int)n)).LastIndexOf(new ReadOnlySpan<TUnit>(needle, checked((int)m)));
            // The candidate starts are [0, n - m]; scan them from the right.
            long end = n - m + 1;
            while (end > 0)
            {
                long k = FindUnitLast(s, end, *needle);
                if (k < 0)
                    return -1;
                if (UnitsEqual(s + k + 1, needle + 1, m - 1))
                    return k;
                end = k;
            }
            return -1;
        }

        /// <summary>Whether two unit runs are equal (a short compare loop; the runs are a separator's length).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="a">The first run.</param>
        /// <param name="b">The second run.</param>
        /// <param name="count">Their length in units.</param>
        /// <returns>True when every unit matches.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool UnitsEqual<TUnit>(TUnit* a, TUnit* b, long count) where TUnit : unmanaged, IEquatable<TUnit>
        {
            for (long k = 0; k < count; k++)
                if (!a[k].Equals(b[k]))
                    return false;
            return true;
        }

        /// <summary>
        ///     FASTSEARCH's one-unit case (<c>find_char</c>): the first occurrence of <paramref name="c"/> in
        ///     <paramref name="s"/>[0 : <paramref name="n"/>), or -1 — a vector compare per block, then the scalar tail.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="n">Its length in units.</param>
        /// <param name="c">The unit.</param>
        /// <returns>The offset, or -1.</returns>
        /// <remarks>
        ///     Not <c>ReadOnlySpan&lt;T&gt;.IndexOf</c>: under <c>TieredCompilation=0</c> the split loop calling CoreLib's
        ///     ReadyToRun search ran 3× slower per element (measured 2026-10-16: 203 against 64 µs per 1,000 elements, the
        ///     same code with <c>DOTNET_ReadyToRun=0</c>), and a short text pays the call's fixed cost anyway.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long FindUnit<TUnit>(TUnit* s, long n, TUnit c) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(byte))
            {
                byte* p = (byte*)s;
                byte v = Unsafe.As<TUnit, byte>(ref c);
                long i = 0;
                if (Vector256.IsHardwareAccelerated && n >= Vector256<byte>.Count)
                {
                    var target = Vector256.Create(v);
                    for (; i + Vector256<byte>.Count <= n; i += Vector256<byte>.Count)
                    {
                        uint mask = Vector256.Equals(Vector256.Load(p + i), target).ExtractMostSignificantBits();
                        if (mask != 0)
                            return i + BitOperations.TrailingZeroCount(mask);
                    }
                }
                else if (Vector128.IsHardwareAccelerated && n >= Vector128<byte>.Count)
                {
                    var target = Vector128.Create(v);
                    for (; i + Vector128<byte>.Count <= n; i += Vector128<byte>.Count)
                    {
                        uint mask = Vector128.Equals(Vector128.Load(p + i), target).ExtractMostSignificantBits();
                        if (mask != 0)
                            return i + BitOperations.TrailingZeroCount(mask);
                    }
                }
                for (; i < n; i++)
                    if (p[i] == v)
                        return i;
                return -1;
            }
            else
            {
                uint* p = (uint*)s;
                uint v = Unsafe.As<TUnit, uint>(ref c);
                long i = 0;
                if (Vector256.IsHardwareAccelerated && n >= Vector256<uint>.Count)
                {
                    var target = Vector256.Create(v);
                    for (; i + Vector256<uint>.Count <= n; i += Vector256<uint>.Count)
                    {
                        uint mask = Vector256.Equals(Vector256.Load(p + i), target).ExtractMostSignificantBits();
                        if (mask != 0)
                            return i + BitOperations.TrailingZeroCount(mask);
                    }
                }
                else if (Vector128.IsHardwareAccelerated && n >= Vector128<uint>.Count)
                {
                    var target = Vector128.Create(v);
                    for (; i + Vector128<uint>.Count <= n; i += Vector128<uint>.Count)
                    {
                        uint mask = Vector128.Equals(Vector128.Load(p + i), target).ExtractMostSignificantBits();
                        if (mask != 0)
                            return i + BitOperations.TrailingZeroCount(mask);
                    }
                }
                for (; i < n; i++)
                    if (p[i] == v)
                        return i;
                return -1;
            }
        }

        /// <summary>
        ///     FASTSEARCH's one-unit reverse case (<c>rfind_char</c>): the last occurrence of <paramref name="c"/> in
        ///     <paramref name="s"/>[0 : <paramref name="n"/>), or -1 — vector blocks from the end, then the scalar head.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="n">Its length in units.</param>
        /// <param name="c">The unit.</param>
        /// <returns>The offset, or -1.</returns>
        /// <remarks>See <see cref="FindUnit{TUnit}"/> for why CoreLib's <c>LastIndexOf</c> is not used.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long FindUnitLast<TUnit>(TUnit* s, long n, TUnit c) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(byte))
            {
                byte* p = (byte*)s;
                byte v = Unsafe.As<TUnit, byte>(ref c);
                long i = n;
                if (Vector256.IsHardwareAccelerated && n >= Vector256<byte>.Count)
                {
                    var target = Vector256.Create(v);
                    for (; i >= Vector256<byte>.Count; i -= Vector256<byte>.Count)
                    {
                        uint mask = Vector256.Equals(Vector256.Load(p + i - Vector256<byte>.Count), target).ExtractMostSignificantBits();
                        if (mask != 0)
                            return i - Vector256<byte>.Count + 31 - BitOperations.LeadingZeroCount(mask);
                    }
                }
                else if (Vector128.IsHardwareAccelerated && n >= Vector128<byte>.Count)
                {
                    var target = Vector128.Create(v);
                    for (; i >= Vector128<byte>.Count; i -= Vector128<byte>.Count)
                    {
                        uint mask = Vector128.Equals(Vector128.Load(p + i - Vector128<byte>.Count), target).ExtractMostSignificantBits();
                        if (mask != 0)
                            return i - Vector128<byte>.Count + 31 - BitOperations.LeadingZeroCount(mask);
                    }
                }
                while (--i >= 0)
                    if (p[i] == v)
                        return i;
                return -1;
            }
            else
            {
                uint* p = (uint*)s;
                uint v = Unsafe.As<TUnit, uint>(ref c);
                long i = n;
                if (Vector256.IsHardwareAccelerated && n >= Vector256<uint>.Count)
                {
                    var target = Vector256.Create(v);
                    for (; i >= Vector256<uint>.Count; i -= Vector256<uint>.Count)
                    {
                        uint mask = Vector256.Equals(Vector256.Load(p + i - Vector256<uint>.Count), target).ExtractMostSignificantBits();
                        if (mask != 0)
                            return i - Vector256<uint>.Count + 31 - BitOperations.LeadingZeroCount(mask);
                    }
                }
                else if (Vector128.IsHardwareAccelerated && n >= Vector128<uint>.Count)
                {
                    var target = Vector128.Create(v);
                    for (; i >= Vector128<uint>.Count; i -= Vector128<uint>.Count)
                    {
                        uint mask = Vector128.Equals(Vector128.Load(p + i - Vector128<uint>.Count), target).ExtractMostSignificantBits();
                        if (mask != 0)
                            return i - Vector128<uint>.Count + 31 - BitOperations.LeadingZeroCount(mask);
                    }
                }
                while (--i >= 0)
                    if (p[i] == v)
                        return i;
                return -1;
            }
        }

        /// <summary>
        ///     <c>rsplit_whitespace</c>: the runs of non-whitespace from the right, at most <paramref name="maxcount"/>
        ///     splits; the rest (its trailing whitespace skipped, its leading kept) is the first piece.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="len">Its length.</param>
        /// <param name="maxcount">The split limit.</param>
        /// <param name="sink">Receives the pieces in order.</param>
        /// <param name="pending">Scratch for the pieces found right to left.</param>
        private static void RSplitWhitespace<TUnit>(TUnit* s, long len, long maxcount, PieceSink sink, PendingPieces pending)
            where TUnit : unmanaged
        {
            pending.Clear();
            long i = len - 1, j;
            while (maxcount-- > 0)
            {
                while (i >= 0 && IsSpace(s[i]))
                    i--;
                if (i < 0)
                    break;
                j = i;
                i--;
                while (i >= 0 && !IsSpace(s[i]))
                    i--;
                pending.Push(i + 1, j + 1);
            }
            if (i >= 0)
            {
                // Only occurs when maxcount was reached: skip the whitespace, the beginning is one piece.
                while (i >= 0 && IsSpace(s[i]))
                    i--;
                if (i >= 0)
                    pending.Push(0, i + 1);
            }
            pending.FlushReversed((byte*)s, sizeof(TUnit), sink);
        }

        /// <summary><c>rsplit_char</c> / <c>rsplit</c>: the pieces between the occurrences of a separator from the right.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="len">Its length.</param>
        /// <param name="sep">The separator.</param>
        /// <param name="sepLen">Its length (at least 1).</param>
        /// <param name="maxcount">The split limit.</param>
        /// <param name="sink">Receives the pieces in order.</param>
        /// <param name="pending">Scratch for the pieces found right to left.</param>
        private static void RSplitSeparator<TUnit>(TUnit* s, long len, TUnit* sep, long sepLen, long maxcount, PieceSink sink,
                                                   PendingPieces pending)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            pending.Clear();
            long j = len;
            while (maxcount-- > 0)
            {
                // FAST_RSEARCH over str[0 : j]: the rightmost occurrence that ends at or before j (rsplit_char for one unit).
                long pos = sepLen == 1 ? FindUnitLast(s, j, *sep) : FindNeedleLast(s, j, sep, sepLen);
                if (pos < 0)
                    break;
                pending.Push(pos + sepLen, j);
                j = pos;
            }
            pending.Push(0, j);
            pending.FlushReversed((byte*)s, sizeof(TUnit), sink);
        }

        /// <summary>
        ///     <c>splitlines</c>: the lines between the boundaries (<c>\r\n</c> one boundary), each with its boundary when
        ///     <paramref name="keepends"/>; no piece after a trailing boundary, none for an empty text.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="len">Its length.</param>
        /// <param name="keepends">Keep the boundaries.</param>
        /// <param name="sink">Receives the lines in order.</param>
        private static void SplitLines<TUnit>(TUnit* s, long len, bool keepends, PieceSink sink) where TUnit : unmanaged
        {
            long i = 0, j = 0;
            while (i < len)
            {
                // Find a line and append it.
                while (i < len && !IsLineBreak(s[i]))
                    i++;
                // Skip the line break, reading CRLF as one line break.
                long eol = i;
                if (i < len)
                {
                    if (Value(s[i]) == '\r' && i + 1 < len && Value(s[i + 1]) == '\n')
                        i += 2;
                    else
                        i++;
                    if (keepends)
                        eol = i;
                }
                sink.Add((byte*)(s + j), eol - j);
                j = i;
            }
        }

        /// <summary>The pieces of an <c>rsplit</c>, found right to left and appended in text order (CPython's
        ///     <c>PyList_Reverse</c>).</summary>
        private sealed class PendingPieces
        {
            /// <summary>The (start, end) pairs as found.</summary>
            private long[] _pairs = new long[32];

            /// <summary>The number of pairs.</summary>
            private int _count;

            /// <summary>Forgets the pairs of the previous element.</summary>
            internal void Clear() => _count = 0;

            /// <summary>Records a piece.</summary>
            /// <param name="start">Its first unit.</param>
            /// <param name="end">One past its last unit.</param>
            internal void Push(long start, long end)
            {
                if (2 * _count + 2 > _pairs.Length)
                    Array.Resize(ref _pairs, _pairs.Length * 2);
                _pairs[2 * _count] = start;
                _pairs[2 * _count + 1] = end;
                _count++;
            }

            /// <summary>Appends the recorded pieces in text order (the reverse of their discovery).</summary>
            /// <param name="text">The text's first byte.</param>
            /// <param name="unit">The unit size in bytes.</param>
            /// <param name="sink">The sink.</param>
            internal void FlushReversed(byte* text, int unit, PieceSink sink)
            {
                for (int k = _count - 1; k >= 0; k--)
                {
                    long start = _pairs[2 * k], end = _pairs[2 * k + 1];
                    sink.Add(text + start * unit, end - start);
                }
            }
        }

        // =============================================================================================================
        // The pieces of a split
        // =============================================================================================================

        /// <summary>
        ///     The growing piece store of one call: every piece's units back to back and the offset after each piece — what
        ///     <see cref="NDListArray"/> keeps. NumSharp arrays, grown by doubling.
        /// </summary>
        private sealed class PieceSink : IDisposable
        {
            /// <summary>The unit size in bytes (1 for bytes pieces, 4 for str pieces).</summary>
            private readonly int _unit;

            /// <summary>The units' array.</summary>
            private NDArray _units;

            /// <summary>Its capacity in units.</summary>
            private long _unitCap;

            /// <summary>The units written.</summary>
            private long _unitLen;

            /// <summary>The offsets' array (capacity + 1 entries).</summary>
            private NDArray _offsets;

            /// <summary>Its capacity in pieces.</summary>
            private long _pieceCap;

            /// <summary>The pieces written.</summary>
            private long _pieceLen;

            /// <summary>The units' address.</summary>
            private byte* _u;

            /// <summary>The offsets' address.</summary>
            private long* _o;

            /// <summary>Creates an empty store.</summary>
            /// <param name="unit">The unit size in bytes.</param>
            /// <param name="unitCapacity">The initial unit capacity.</param>
            /// <param name="pieceCapacity">The initial piece capacity.</param>
            internal PieceSink(int unit, long unitCapacity, long pieceCapacity)
            {
                _unit = unit;
                _unitCap = Math.Max(16, unitCapacity);
                _pieceCap = Math.Max(16, pieceCapacity);
                _units = new NDArray(unit == 1 ? NPTypeCode.Byte : NPTypeCode.UInt32, Shape.Vector(_unitCap), false);
                _offsets = new NDArray(NPTypeCode.Int64, Shape.Vector(_pieceCap + 1), false);
                _u = (byte*)_units.Storage.Address;
                _o = (long*)_offsets.Storage.Address;
                _o[0] = 0;
            }

            /// <summary>The number of pieces written so far.</summary>
            internal long Count => _pieceLen;

            /// <summary>Appends one piece.</summary>
            /// <param name="src">Its first unit.</param>
            /// <param name="units">Its length in units.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal void Add(byte* src, long units)
            {
                if (_unitLen + units > _unitCap)
                    GrowUnits(_unitLen + units);
                if (_pieceLen + 1 > _pieceCap)
                    GrowPieces();
                CopyPiece(src, _u + _unitLen * _unit, units * _unit);
                _unitLen += units;
                _o[++_pieceLen] = _unitLen;
            }

            /// <summary>
            ///     Copies one piece: up to 32 bytes with two overlapping loads and stores of the widest fitting size (the
            ///     common piece, a word, is a few bytes), up to 512 bytes in 16-byte blocks, longer pieces through
            ///     <see cref="Buffer.MemoryCopy"/>. The source (the text) and the destination (the store) never overlap.
            /// </summary>
            /// <param name="src">The piece.</param>
            /// <param name="dst">Its slot in the store.</param>
            /// <param name="bytes">Its size in bytes.</param>
            /// <remarks>
            ///     A call per piece cost more than the split's scan of a short word (measured 2026-10-16: ~10 ns per piece
            ///     through CoreLib's memmove, and 3× that under <c>TieredCompilation=0</c>, whose precompiled memmove body is
            ///     never re-optimized).
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static void CopyPiece(byte* src, byte* dst, long bytes)
            {
                if (bytes <= 16)
                {
                    // Two overlapping moves of the widest size that fits cover any length in [size, 2 * size].
                    if (bytes >= 8)
                    {
                        Unsafe.WriteUnaligned(dst, Unsafe.ReadUnaligned<ulong>(src));
                        Unsafe.WriteUnaligned(dst + bytes - 8, Unsafe.ReadUnaligned<ulong>(src + bytes - 8));
                    }
                    else if (bytes >= 4)
                    {
                        Unsafe.WriteUnaligned(dst, Unsafe.ReadUnaligned<uint>(src));
                        Unsafe.WriteUnaligned(dst + bytes - 4, Unsafe.ReadUnaligned<uint>(src + bytes - 4));
                    }
                    else if (bytes > 0)
                    {
                        dst[0] = src[0];
                        dst[bytes - 1] = src[bytes - 1];
                        dst[bytes >> 1] = src[bytes >> 1];
                    }
                    return;
                }
                if (bytes <= 32)
                {
                    Unsafe.WriteUnaligned(dst, Unsafe.ReadUnaligned<Vector128<byte>>(src));
                    Unsafe.WriteUnaligned(dst + bytes - 16, Unsafe.ReadUnaligned<Vector128<byte>>(src + bytes - 16));
                    return;
                }
                if (bytes <= 512)
                {
                    // 16-byte blocks, then one block ending at the piece's end (it overlaps the last full block).
                    long k = 0;
                    for (; k + 16 <= bytes; k += 16)
                        Unsafe.WriteUnaligned(dst + k, Unsafe.ReadUnaligned<Vector128<byte>>(src + k));
                    if (k < bytes)
                        Unsafe.WriteUnaligned(dst + bytes - 16, Unsafe.ReadUnaligned<Vector128<byte>>(src + bytes - 16));
                    return;
                }
                Buffer.MemoryCopy(src, dst, bytes, bytes);
            }

            /// <summary>Grows the units to hold at least <paramref name="needed"/> units.</summary>
            /// <param name="needed">The units needed.</param>
            private void GrowUnits(long needed)
            {
                long cap = Math.Max(needed, _unitCap * 2);
                var grown = new NDArray(_units.typecode, Shape.Vector(cap), false);
                Buffer.MemoryCopy(_u, grown.Storage.Address, cap * _unit, _unitLen * _unit);
                _units.Dispose();
                _units = grown;
                _unitCap = cap;
                _u = (byte*)grown.Storage.Address;
            }

            /// <summary>Doubles the piece capacity.</summary>
            private void GrowPieces()
            {
                long cap = _pieceCap * 2;
                var grown = new NDArray(NPTypeCode.Int64, Shape.Vector(cap + 1), false);
                Buffer.MemoryCopy(_o, grown.Storage.Address, (cap + 1) * 8, (_pieceLen + 1) * 8);
                _offsets.Dispose();
                _offsets = grown;
                _pieceCap = cap;
                _o = (long*)grown.Storage.Address;
            }

            /// <summary>
            ///     CPython creates every str piece with <c>PyUnicode_New</c>: a ONE-unit piece above U+10FFFF is its SystemError
            ///     (a longer piece caps its maximum character). Checks the pieces from <paramref name="firstPiece"/> on (the
            ///     ones the current element appended; code-point pieces only).
            /// </summary>
            /// <param name="firstPiece">The current element's first piece.</param>
            /// <exception cref="SystemError">Such a piece.</exception>
            internal void ThrowIfLoneInvalid(long firstPiece)
            {
                uint* units = (uint*)_u;
                for (long k = firstPiece; k < _pieceLen; k++)
                    if (_o[k + 1] - _o[k] == 1 && units[_o[k]] > 0x10FFFF)
                        throw new SystemError(InvalidMaxCharText);
            }

            /// <summary>Hands the arrays over (the sink no longer owns them).</summary>
            /// <param name="units">The units.</param>
            /// <param name="offsets">The offsets.</param>
            internal void Release(out NDArray units, out NDArray offsets)
            {
                units = _units;
                offsets = _offsets;
                _units = null;
                _offsets = null;
            }

            /// <summary>Releases arrays not handed over (a call that raised).</summary>
            public void Dispose()
            {
                _units?.Dispose();
                _offsets?.Dispose();
                _units = null;
                _offsets = null;
            }
        }

        // =============================================================================================================
        // split / rsplit / splitlines
        // =============================================================================================================

        /// <summary>
        ///     <c>np.char.split</c> / <c>rsplit</c>: <c>_vec_string(a, object_, 'split' | 'rsplit', [sep] +
        ///     _clean_args(maxsplit))</c>.
        /// </summary>
        /// <param name="a">The text (converted).</param>
        /// <param name="sep">The separators (converted), or null for <c>sep=None</c>.</param>
        /// <param name="maxsplit">The split limits (converted), or null for <c>maxsplit=None</c>.</param>
        /// <param name="reverse">True for <c>rsplit</c>.</param>
        /// <returns>The lists.</returns>
        /// <exception cref="TypeError">See <see cref="SplitElement{TUnit}"/>; also a non-text <paramref name="a"/>
        ///     (<see cref="NonStringArrayText"/>, checked by the caller).</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast (NumPy's multi-iterator text).</exception>
        internal static NDListArray Split(VecArg a, VecArg sep, VecArg maxsplit, bool reverse)
        {
            // The multi-iterator's arguments: a, the separator (None when not given: a 0-d object), then maxsplit when given.
            var shapes = new List<Shape> { a.BroadcastShape, sep?.BroadcastShape ?? Shape.NewScalar() };
            if (maxsplit is not null)
                shapes.Add(maxsplit.BroadcastShape);
            var outer = np.broadcast_shapes(shapes.ToArray());

            var ops = new List<NDArray> { a.Array };
            if (sep is not null)
                ops.Add(sep.Array);
            if (maxsplit is not null)
                ops.Add(maxsplit.Array);
            int sepSlot = sep is null ? -1 : 1;
            int maxSlot = maxsplit is null ? -1 : sep is null ? 1 : 2;
            var readers = BuildReaders(a, sep, maxsplit, null);
            try
            {
                bool unicode = readers[0].Kind != OperandKind.Bytes;
                char kind = readers[0].Kind == OperandKind.Bytes ? 'S' : readers[0].Kind == OperandKind.Unicode ? 'U' : 'T';
                long width = readers[0].Kind == OperandKind.VString ? 8 : Math.Max(1, readers[0].Chars);
                long size = outer.size;
                var sink = new PieceSink(unicode ? 4 : 1, Math.Min(size * width, 1L << 24), Math.Min(size * 2, 1L << 22));
                var starts = new NDArray(NPTypeCode.Int64, outer, false);
                var ends = new NDArray(NPTypeCode.Int64, outer, false);
                NDArray self = null;
                try
                {
                    var state = new SplitState
                    {
                        Readers = readers, SepSlot = sepSlot, MaxSlot = maxSlot, Reverse = reverse, Sink = sink,
                        Starts = (long*)starts.Storage.Address, Ends = (long*)ends.Storage.Address, Pending = new PendingPieces(),
                        Kind = kind, Size = size,
                    };
                    if (size > 0)
                    {
                        Walk(ops, readers, (dataptrs, strides, n, aux) =>
                        {
                            if (unicode)
                                SplitRun<uint>(dataptrs, strides, n, state);
                            else
                                SplitRun<byte>(dataptrs, strides, n, state);
                        });
                    }
                    if (state.SelfFlags is not null)
                    {
                        // Only U elements can return their own np.str_ (unicodeobject.c's [self] shortcut).
                        self = new NDArray(NPTypeCode.Boolean, outer, true);
                        bool* flags = (bool*)self.Storage.Address;
                        foreach (long e in state.SelfFlags)
                            flags[e] = true;
                    }
                    sink.Release(out var units, out var offsets);
                    // The heap takes the arrays over (and reads and disposes the per-element self flags); on a failure the
                    // catch below disposes them again, which NDArray.Dispose ignores.
                    return NDListArray.FromElements(units, offsets, sink.Count, starts, ends, self, kind, outer);
                }
                catch
                {
                    starts.Dispose();
                    ends.Dispose();
                    self?.Dispose();
                    throw;
                }
                finally
                {
                    sink.Dispose();
                }
            }
            finally
            {
                foreach (var r in readers)
                    r?.Dispose();
            }
        }

        /// <summary>The state of one split call, shared by its NDIter chunks.</summary>
        [NDBorrowed] // the piece sink and the readers belong to Split / Splitlines, which release them in their finally
        private sealed class SplitState
        {
            /// <summary>The operand readers (a, then the separator and maxsplit when present).</summary>
            internal Operand[] Readers;

            /// <summary>The separator's operand slot, or -1.</summary>
            internal int SepSlot;

            /// <summary>Maxsplit's operand slot, or -1.</summary>
            internal int MaxSlot;

            /// <summary>True for <c>rsplit</c>.</summary>
            internal bool Reverse;

            /// <summary>The piece store.</summary>
            internal PieceSink Sink;

            /// <summary>The next element's first-piece slot (C order).</summary>
            internal long* Starts;

            /// <summary>The next element's end slot.</summary>
            internal long* Ends;

            /// <summary>The rsplit scratch.</summary>
            internal PendingPieces Pending;

            /// <summary>The class of the pieces.</summary>
            internal char Kind;

            /// <summary>The number of elements.</summary>
            internal long Size;

            /// <summary>The C-order position of the next element.</summary>
            internal long Position;

            /// <summary>The positions whose single piece is the element's own <c>np.str_</c> (null until one).</summary>
            internal List<long> SelfFlags;
        }

        /// <summary>One NDIter chunk of a split: every element of the chunk, in C order.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c> / <c>StringDType</c>.</typeparam>
        /// <param name="dataptrs">The chunk's operand pointers.</param>
        /// <param name="strides">Their byte strides.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="state">The call's state.</param>
        /// <exception cref="Exception">The first element's error (see <see cref="SplitElement{TUnit}"/>).</exception>
        private static void SplitRun<TUnit>(void** dataptrs, long* strides, long n, SplitState state)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            byte* pa = (byte*)dataptrs[0];
            long sa = strides[0];
            byte* ps = state.SepSlot < 0 ? null : (byte*)dataptrs[state.SepSlot];
            long ss = state.SepSlot < 0 ? 0 : strides[state.SepSlot];
            byte* pm = state.MaxSlot < 0 ? null : (byte*)dataptrs[state.MaxSlot];
            long sm = state.MaxSlot < 0 ? 0 : strides[state.MaxSlot];
            for (long i = 0; i < n; i++, pa += sa, ps += ss, pm += sm)
            {
                long first = state.Sink.Count;
                bool self = SplitElement<TUnit>(pa, ps, pm, state);
                if (self)
                    (state.SelfFlags ??= new List<long>()).Add(state.Position);
                state.Starts[state.Position] = first;
                state.Ends[state.Position] = state.Sink.Count;
                state.Position++;
            }
        }

        /// <summary>
        ///     One element of <c>str.split</c> / <c>bytes.split</c> (or the <c>rsplit</c> twins), in CPython's order: the
        ///     scalars, the descriptor check, <c>maxsplit</c>'s <c>__index__</c>, the separator's type, the empty separator,
        ///     the pieces.
        /// </summary>
        /// <typeparam name="TUnit">The text's unit type.</typeparam>
        /// <param name="pa">The text element.</param>
        /// <param name="ps">The separator element, or null.</param>
        /// <param name="pm">The maxsplit element, or null.</param>
        /// <param name="state">The call's state.</param>
        /// <returns>True when the element's single piece is its own <c>np.str_</c> (a <c>U</c> element's [self]).</returns>
        /// <exception cref="SystemError">A one-unit <c>U</c> text or separator above U+10FFFF, or such a piece.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> NA that is not a str (<c>descriptor 'split' for 'str' objects
        ///     doesn't apply to a 'NoneType' object</c>); a maxsplit without <c>__index__</c>; a str method's separator that
        ///     is not a str or None (<c>must be str or None, not numpy.int64</c>); a bytes method's separator without a
        ///     buffer (<c>a bytes-like object is required, not 'str'</c>).</exception>
        /// <exception cref="OverflowException">A maxsplit beyond <c>Py_ssize_t</c>.</exception>
        /// <exception cref="ValueError">An empty separator (<c>empty separator</c>).</exception>
        private static bool SplitElement<TUnit>(byte* pa, byte* ps, byte* pm, SplitState state)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            var readers = state.Readers;
            // 1. PyArray_ToScalar of each operand, in argument order.
            readers[0].Read(pa, out var text);
            Elem sep = default;
            bool hasSep = ps != null;
            if (hasSep)
                readers[state.SepSlot].Read(ps, out sep);
            Elem max = default;
            if (pm != null)
                readers[state.MaxSlot].Read(pm, out max);

            bool strMethod = typeof(TUnit) == typeof(uint);
            string name = state.Reverse ? "rsplit" : "split";
            // 2. The method descriptor: str.split applies only to a str (a StringDType NA object is not one).
            if (strMethod && !IsStr(text))
                throw new TypeError($"descriptor '{name}' for 'str' objects doesn't apply to a '{TypeName(text)}' object");
            // 3. Argument parsing: maxsplit through operator.index, a negative limit meaning none.
            long maxcount = pm != null ? Index(max) : -1;
            if (maxcount < 0)
                maxcount = long.MaxValue;

            TUnit* s = (TUnit*)text.Units;
            long len = text.Length;
            // 4. The method body.
            long firstPiece = state.Sink.Count;
            if (!hasSep || IsNone(sep))
            {
                if (state.Reverse)
                    RSplitWhitespace(s, len, maxcount, state.Sink, state.Pending);
                else
                    SplitWhitespace(s, len, maxcount, state.Sink);
                if (text.Kind == ElemKind.NpStr)
                    state.Sink.ThrowIfLoneInvalid(firstPiece);
                return false;
            }

            TUnit* needle;
            long needleLen;
            if (strMethod)
            {
                if (!IsStr(sep))
                    throw new TypeError($"must be str or None, not {TypeName(sep)}");
                needle = (TUnit*)sep.Units;
                needleLen = sep.Length;
                // unicodeobject.c: a separator longer than the string, or of a wider kind, cannot occur in it: [self]. The
                // text's kind matters only for a separator wider than kind 1 (no text is narrower than that).
                int needleKind = StrKind((uint*)needle, needleLen);
                if (len < needleLen || (needleKind > 1 && StrKind((uint*)s, len) < needleKind))
                {
                    state.Sink.Add((byte*)s, len);
                    return text.Kind == ElemKind.NpStr;
                }
            }
            else
            {
                if (!TryBuffer(sep, out byte* data, out long bytes))
                    throw new TypeError($"a bytes-like object is required, not '{TypeName(sep)}'");
                needle = (TUnit*)data;
                needleLen = bytes;
            }
            if (needleLen == 0)
                throw new ValueError(EmptySeparatorText);
            if (state.Reverse)
                RSplitSeparator(s, len, needle, needleLen, maxcount, state.Sink, state.Pending);
            else
                SplitSeparator(s, len, needle, needleLen, maxcount, state.Sink);
            if (text.Kind == ElemKind.NpStr)
                state.Sink.ThrowIfLoneInvalid(firstPiece);
            return false;
        }

        /// <summary>
        ///     <c>np.char.splitlines</c>: <c>_vec_string(a, object_, 'splitlines', _clean_args(keepends))</c> — no second
        ///     operand for <c>keepends=None</c> (<c>_vec_string_no_args</c>).
        /// </summary>
        /// <param name="a">The text (converted).</param>
        /// <param name="keepends">The keepends values (converted), or null for <c>keepends=None</c>.</param>
        /// <returns>The lists of lines.</returns>
        /// <exception cref="SystemError">A one-unit <c>U</c> text, a one-unit <c>U</c> keepends element or a one-unit line above
        ///     U+10FFFF.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> NA that is not a str (<c>descriptor 'splitlines' for 'str' objects
        ///     doesn't apply to a 'NoneType' object</c>).</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast (NumPy's multi-iterator text).</exception>
        internal static NDListArray Splitlines(VecArg a, VecArg keepends)
        {
            var outer = keepends is null ? a.BroadcastShape.Clean() : np.broadcast_shapes(a.BroadcastShape, keepends.BroadcastShape);
            var ops = new List<NDArray> { a.Array };
            if (keepends is not null)
                ops.Add(keepends.Array);
            var readers = BuildReaders(a, keepends);
            try
            {
                bool unicode = readers[0].Kind != OperandKind.Bytes;
                char kind = readers[0].Kind == OperandKind.Bytes ? 'S' : readers[0].Kind == OperandKind.Unicode ? 'U' : 'T';
                long width = readers[0].Kind == OperandKind.VString ? 8 : Math.Max(1, readers[0].Chars);
                long size = outer.size;
                var sink = new PieceSink(unicode ? 4 : 1, Math.Min(size * width, 1L << 24), Math.Min(size * 2, 1L << 22));
                var starts = new NDArray(NPTypeCode.Int64, outer, false);
                var ends = new NDArray(NPTypeCode.Int64, outer, false);
                try
                {
                    long* pStarts = (long*)starts.Storage.Address;
                    long* pEnds = (long*)ends.Storage.Address;
                    long position = 0;
                    if (size > 0)
                    {
                        Walk(ops, readers, (dataptrs, strides, n, aux) =>
                        {
                            byte* pa = (byte*)dataptrs[0];
                            long sa = strides[0];
                            byte* pk = keepends is null ? null : (byte*)dataptrs[1];
                            long sk = keepends is null ? 0 : strides[1];
                            for (long i = 0; i < n; i++, pa += sa, pk += sk)
                            {
                                long first = sink.Count;
                                if (unicode)
                                    SplitlinesElement<uint>(pa, pk, readers, sink);
                                else
                                    SplitlinesElement<byte>(pa, pk, readers, sink);
                                pStarts[position] = first;
                                pEnds[position] = sink.Count;
                                position++;
                            }
                        });
                    }
                    sink.Release(out var units, out var offsets);
                    return NDListArray.FromElements(units, offsets, sink.Count, starts, ends, null, kind, outer);
                }
                catch
                {
                    starts.Dispose();
                    ends.Dispose();
                    throw;
                }
                finally
                {
                    sink.Dispose();
                }
            }
            finally
            {
                foreach (var r in readers)
                    r?.Dispose();
            }
        }

        /// <summary>
        ///     One element of <c>str.splitlines</c> / <c>bytes.splitlines</c> in CPython's order: the scalars, the descriptor
        ///     check, <c>keepends</c>' truth (<c>PyObject_IsTrue</c>), the lines.
        /// </summary>
        /// <typeparam name="TUnit">The text's unit type.</typeparam>
        /// <param name="pa">The text element.</param>
        /// <param name="pk">The keepends element, or null.</param>
        /// <param name="readers">The operand readers.</param>
        /// <param name="sink">The piece store.</param>
        /// <exception cref="SystemError">A one-unit <c>U</c> scalar or line above U+10FFFF.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> NA that is not a str.</exception>
        private static void SplitlinesElement<TUnit>(byte* pa, byte* pk, Operand[] readers, PieceSink sink) where TUnit : unmanaged
        {
            readers[0].Read(pa, out var text);
            Elem keep = default;
            if (pk != null)
                readers[1].Read(pk, out keep);
            if (typeof(TUnit) == typeof(uint) && !IsStr(text))
                throw new TypeError($"descriptor 'splitlines' for 'str' objects doesn't apply to a '{TypeName(text)}' object");
            bool keepends = pk != null && Truth(keep);
            long first = sink.Count;
            SplitLines((TUnit*)text.Units, text.Length, keepends, sink);
            if (text.Kind == ElemKind.NpStr)
                sink.ThrowIfLoneInvalid(first);
        }

        // =============================================================================================================
        // join
        // =============================================================================================================

        /// <summary>
        ///     <c>np.char.join(sep, seq)</c>: <c>_to_bytes_or_str_array(_vec_string(sep, object_, 'join', (seq,)), seq)</c>.
        /// </summary>
        /// <param name="sep">The separators (converted, text: the caller checked NumPy's non-string error).</param>
        /// <param name="seq">The sequences (converted: text, numbers, an object list or an <see cref="NDListArray"/>).</param>
        /// <param name="emptyObject">Receives NumPy's empty OBJECT result (an empty <see cref="NDListArray"/> of the broadcast
        ///     shape) when <paramref name="seq"/> is an object array and the broadcast is empty; then the return is null.</param>
        /// <returns>The joined text: <c>U</c> / <c>S</c> as wide as the longest result (at least 1), <c>StringDType()</c>, or
        ///     for an empty broadcast an empty array of <paramref name="seq"/>'s dtype.</returns>
        /// <exception cref="SystemError">A one-unit <c>U</c> separator or sequence above U+10FFFF, or such a character inside a
        ///     <c>U</c> sequence (iterating it creates the character).</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> NA separator that is not a str (<c>descriptor 'join' for 'str'
        ///     objects doesn't apply to a 'NoneType' object</c>); a sequence that is not iterable (a number, None:
        ///     <c>can only join an iterable</c>); an item of the wrong type (<c>sequence item 0: expected str instance, int
        ///     found</c>, <c>sequence item 0: expected a bytes-like object, str found</c>); an object <paramref name="seq"/>
        ///     with a non-empty broadcast (<see cref="ObjectDTypeText"/>, after every element joined).</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast (NumPy's multi-iterator text).</exception>
        internal static NDArray Join(VecArg sep, VecArg seq, out NDListArray emptyObject)
        {
            emptyObject = null;
            var outer = np.broadcast_shapes(sep.BroadcastShape, seq.BroadcastShape);
            long size = outer.size;
            bool objectSeq = seq.Objects is not null || seq.Lists is not null;
            if (size == 0)
            {
                // _to_bytes_or_str_array: an empty result is result.astype(seq.dtype) — any dtype; an object seq's is an empty
                // object array.
                if (objectSeq)
                {
                    emptyObject = EmptyLists(outer, seq.Lists?.Kind ?? 'U');
                    return null;
                }
                return new NDArray(seq.Array.dtype, outer.Clean(), true);
            }

            var ops = new List<NDArray> { sep.Array };
            if (seq.Lists is not null)
            {
                // A list-array sequence is walked through its ids (NumPy's pointers); each element's list is read from
                // the shared heap.
                ops.Add(seq.Lists.Ids);
            }
            else
            {
                ops.Add(seq.Array);
            }
            var readers = seq.Lists is null ? BuildReaders(sep, seq) : BuildReaders(sep);
            try
            {
                bool strMethod = readers[0].IsStrClass;
                var join = new JoinState { Readers = readers, StrMethod = strMethod, Lists = seq.Lists };
                // Pass 1, in C order: every element's checks (NumPy raises the first error) and its result's length.
                Walk(ops, readers, (dataptrs, strides, n, aux) => JoinMeasureRun(dataptrs, strides, n, join));
                if (objectSeq)
                    throw new TypeError(ObjectDTypeText);

                // np.asarray(result.tolist()) then astype(type(seq.dtype)(n)): the class of seq, as wide as the longest result.
                var seqStorage = seq.Array.Storage;
                // A sequence that is not text joined nothing (pass 1 refused it: not iterable), so it never gets here.
                if (!seqStorage.IsVString && !seqStorage.IsFixedString)
                    throw new InvalidOperationException("a non-text sequence passed the join's checks");
                if (seqStorage.IsVString)
                    return JoinIntoVString(ops, readers, outer, join);
                bool unicodeOut = seqStorage.StringSlice.IsUnicode;
                var descr = unicodeOut ? DTypeRegistry.Str.Descr(Math.Max(1, join.MaxLength)) : DTypeRegistry.Bytes.Descr(Math.Max(1, join.MaxLength));
                var result = new NDArray(descr, outer.Clean(), true);
                // Only a str method over a str sequence has non-empty results: a bytes method, or a bytes sequence, joined
                // only empty elements (anything else raised in pass 1), whose results are the zero-filled elements.
                if (strMethod && unicodeOut && join.MaxLength > 0)
                {
                    join.Output = (byte*)result.Storage.Address;
                    join.OutputChars = descr.itemsize / 4;
                    join.Position = 0;
                    try
                    {
                        Walk(ops, readers, (dataptrs, strides, n, aux) => JoinWriteRun(dataptrs, strides, n, join));
                    }
                    catch
                    {
                        result.Dispose();
                        throw;
                    }
                }
                return result;
            }
            finally
            {
                foreach (var r in readers)
                    r?.Dispose();
            }
        }

        /// <summary>The state of one join call, shared by its NDIter chunks.</summary>
        [NDBorrowed] // the list-array seq is the caller's argument; the readers belong to Join, which releases them
        private sealed class JoinState
        {
            /// <summary>The operand readers (sep, then seq when it is not a list array).</summary>
            internal Operand[] Readers;

            /// <summary>True for <c>str.join</c> (a <c>U</c> / <c>StringDType</c> sep), false for <c>bytes.join</c>.</summary>
            internal bool StrMethod;

            /// <summary>A list-array seq, or null.</summary>
            internal NDListArray Lists;

            /// <summary>The longest result (code points for <c>str.join</c>, bytes for <c>bytes.join</c>).</summary>
            internal long MaxLength;

            /// <summary>The output's first element (pass 2).</summary>
            internal byte* Output;

            /// <summary>The output's width in code points (pass 2).</summary>
            internal int OutputChars;

            /// <summary>The next element's C-order position (pass 2).</summary>
            internal long Position;
        }

        /// <summary>Pass 1 over one NDIter chunk: each element's checks and result length.</summary>
        /// <param name="dataptrs">The operand pointers.</param>
        /// <param name="strides">Their byte strides.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="join">The call's state.</param>
        /// <exception cref="Exception">The first element's error (see <see cref="Join"/>).</exception>
        private static void JoinMeasureRun(void** dataptrs, long* strides, long n, JoinState join)
        {
            byte* ps = (byte*)dataptrs[0];
            long ss = strides[0];
            byte* pq = (byte*)dataptrs[1];
            long sq = strides[1];
            bool lists = join.Lists is not null;
            for (long i = 0; i < n; i++, ps += ss, pq += sq)
            {
                join.Readers[0].Read(ps, out var sep);
                long length;
                if (lists)
                {
                    // An object array of lists: no scalar to make, the element IS the list (its id in the shared heap).
                    CheckSeparator(sep, join.StrMethod);
                    length = MeasureListJoin(sep, join.Lists, *(long*)pq, join.StrMethod);
                }
                else
                {
                    join.Readers[1].Read(pq, out var seq);
                    CheckSeparator(sep, join.StrMethod);
                    length = MeasureJoin(sep, seq, join.StrMethod);
                }
                if (length > join.MaxLength)
                    join.MaxLength = length;
            }
        }

        /// <summary>The method descriptor's check: <c>str.join</c> applies only to a str (a <c>StringDType</c> NA such as
        ///     None is not one); a <c>bytes.join</c> sep is always <c>np.bytes_</c>.</summary>
        /// <param name="sep">The separator's object.</param>
        /// <param name="strMethod">True for <c>str.join</c>.</param>
        /// <exception cref="TypeError"><c>descriptor 'join' for 'str' objects doesn't apply to a 'NoneType' object</c>.</exception>
        private static void CheckSeparator(in Elem sep, bool strMethod)
        {
            if (strMethod && !IsStr(sep))
                throw new TypeError($"descriptor 'join' for 'str' objects doesn't apply to a '{TypeName(sep)}' object");
        }

        /// <summary>
        ///     <c>sep.join(seq)</c> of one text / number / object element, measured: <c>PySequence_Fast(seq)</c> (a str yields
        ///     its characters, bytes their int values, a Python list its items, anything else is not iterable), then the item
        ///     checks of <c>PyUnicode_Join</c> / <c>bytes_join</c>.
        /// </summary>
        /// <param name="sep">The separator.</param>
        /// <param name="seq">The sequence.</param>
        /// <param name="strMethod">True for <c>str.join</c>.</param>
        /// <returns>The result's length (code points or bytes).</returns>
        /// <exception cref="SystemError">A <c>U</c> sequence holding a unit above U+10FFFF (its character cannot be created).</exception>
        /// <exception cref="TypeError">Not iterable, or an item of the wrong type.</exception>
        private static long MeasureJoin(in Elem sep, in Elem seq, bool strMethod)
        {
            long sepLength = strMethod ? sep.Length : SeparatorBytes(sep);
            switch (seq.Kind)
            {
                case ElemKind.NpStr:
                case ElemKind.Str:
                    // Iterating the str creates each character: a unit above U+10FFFF is PyUnicode_New's SystemError.
                    if (seq.Kind == ElemKind.NpStr)
                        for (long k = 0; k < seq.Length; k++)
                            if (((uint*)seq.Units)[k] > 0x10FFFF)
                                throw new SystemError(InvalidMaxCharText);
                    if (seq.Length == 0)
                        return 0;
                    if (!strMethod)
                        throw new TypeError("sequence item 0: expected a bytes-like object, str found");
                    return seq.Length + (seq.Length - 1) * sepLength;
                case ElemKind.NpBytes:
                case ElemKind.Bytes:
                    if (seq.Length == 0)
                        return 0;
                    throw new TypeError(strMethod
                        ? "sequence item 0: expected str instance, int found"
                        : "sequence item 0: expected a bytes-like object, int found");
                case ElemKind.Object when seq.Obj is IList list && seq.Obj is not byte[]:
                    return MeasureItems(list, sepLength, strMethod);
                default:
                    throw new TypeError(JoinIterableText);
            }
        }

        /// <summary><c>sep.join(list)</c> of a Python list from an object array (its items as the boundary map reads them).</summary>
        /// <param name="list">The list.</param>
        /// <param name="sepLength">The separator's length in result units.</param>
        /// <param name="strMethod">True for <c>str.join</c>.</param>
        /// <returns>The result's length.</returns>
        /// <exception cref="TypeError">An item of the wrong type.</exception>
        private static long MeasureItems(IList list, long sepLength, bool strMethod)
        {
            long total = 0;
            for (int k = 0; k < list.Count; k++)
            {
                object item = list[k];
                if (strMethod)
                {
                    if (item is not (string or char))
                        throw new TypeError($"sequence item {k}: expected str instance, {NDPyObject.TypeName(item)} found");
                    total += item is string s ? NDStringCodec.CodePointCount(s) : 1;
                }
                else
                {
                    if (item is not byte[] b)
                        throw new TypeError($"sequence item {k}: expected a bytes-like object, {NDPyObject.TypeName(item)} found");
                    total += b.Length;
                }
            }
            return list.Count == 0 ? 0 : total + (list.Count - 1) * sepLength;
        }

        /// <summary>
        ///     <c>sep.join(pieces)</c> of one element of a list array, measured: <c>str.join</c> needs str pieces (a bytes
        ///     piece is <c>expected str instance, bytes found</c>), <c>bytes.join</c> buffers (a plain str piece is
        ///     <c>expected a bytes-like object, str found</c>, naming its position; an element's own <c>np.str_</c> has one).
        /// </summary>
        /// <param name="sep">The separator.</param>
        /// <param name="lists">The list array.</param>
        /// <param name="id">The element's list id.</param>
        /// <param name="strMethod">True for <c>str.join</c>.</param>
        /// <returns>The result's length.</returns>
        /// <exception cref="TypeError">An item of the wrong type.</exception>
        private static long MeasureListJoin(in Elem sep, NDListArray lists, long id, bool strMethod)
        {
            var heap = lists.Heap;
            var (start, end) = heap.Range(id);
            long count = end - start;
            if (count == 0)
                return 0;
            if (strMethod && lists.Kind == 'S')
                throw new TypeError("sequence item 0: expected str instance, bytes found");
            if (!strMethod && lists.Kind != 'S')
            {
                // bytes_join takes each item's buffer in order: the first plain str has none.
                for (long k = start; k < end; k++)
                    if (!heap.IsNpStr(k))
                        throw new TypeError($"sequence item {k - start}: expected a bytes-like object, str found");
            }
            long* offsets = heap.Offsets;
            long sepLength = strMethod ? sep.Length : SeparatorBytes(sep);
            long total = offsets[end] - offsets[start];
            // bytes.join reads an np.str_ piece through its buffer: four bytes a code point.
            if (!strMethod && lists.Kind != 'S')
                total *= 4;
            return total + (count - 1) * sepLength;
        }

        /// <summary>The bytes a <c>bytes.join</c> separator contributes (an <c>S</c> element: its bytes).</summary>
        /// <param name="sep">The separator.</param>
        /// <returns>Its length in bytes.</returns>
        private static long SeparatorBytes(in Elem sep) => TryBuffer(sep, out _, out long bytes) ? bytes : 0;

        /// <summary>Pass 2 over one NDIter chunk: each element's <c>str.join</c> of a <c>U</c> sequence, written.</summary>
        /// <param name="dataptrs">The operand pointers.</param>
        /// <param name="strides">Their byte strides.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="join">The call's state.</param>
        private static void JoinWriteRun(void** dataptrs, long* strides, long n, JoinState join)
        {
            byte* ps = (byte*)dataptrs[0];
            long ss = strides[0];
            byte* pq = (byte*)dataptrs[1];
            long sq = strides[1];
            long itemsize = join.OutputChars * 4L;
            for (long i = 0; i < n; i++, ps += ss, pq += sq)
            {
                join.Readers[0].Read(ps, out var sep);
                join.Readers[1].Read(pq, out var seq);
                uint* dst = (uint*)(join.Output + join.Position * itemsize);
                WriteJoin((uint*)sep.Units, sep.Length, (uint*)seq.Units, seq.Length, dst);
                join.Position++;
            }
        }

        /// <summary>The characters of a sequence with the separator between them (the result of <c>str.join</c>).</summary>
        /// <param name="sep">The separator's code points.</param>
        /// <param name="sepLength">Their number.</param>
        /// <param name="seq">The sequence's code points.</param>
        /// <param name="length">Their number.</param>
        /// <param name="dst">The output (room for the result; the rest of the element stays zero).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteJoin(uint* sep, long sepLength, uint* seq, long length, uint* dst)
        {
            for (long k = 0; k < length; k++)
            {
                if (k > 0)
                    for (long j = 0; j < sepLength; j++)
                        *dst++ = sep[j];
                *dst++ = seq[k];
            }
        }

        /// <summary>
        ///     A <c>StringDType</c> sequence's join: the results packed into a fresh <c>StringDType()</c> array (NumPy's
        ///     <c>ret.astype(StringDType)</c>: default parameters, so no NA object).
        /// </summary>
        /// <param name="ops">The NDIter operands.</param>
        /// <param name="readers">The operand readers.</param>
        /// <param name="outer">The broadcast shape.</param>
        /// <param name="join">The call's state (pass 1 done).</param>
        /// <returns>The result.</returns>
        /// <exception cref="OutOfMemoryException">The arena cannot grow.</exception>
        private static NDArray JoinIntoVString(List<NDArray> ops, Operand[] readers, Shape outer, JoinState join)
        {
            var result = new NDArray(np.dtypes.StringDType(), outer.Clean(), true);
            try
            {
                if (!join.StrMethod)
                    return result;   // a bytes method joined only empty elements: every result is '' (a zeroed cell)
                var allocator = result.Storage.VStringSlice.Allocator;
                byte* cells = (byte*)result.Storage.Address;
                long position = 0;
                byte[] utf8 = new byte[64];
                uint[] points = new uint[64];
                allocator.acquire_allocator();
                try
                {
                    Walk(ops, readers, (dataptrs, strides, n, aux) =>
                    {
                        byte* ps = (byte*)dataptrs[0];
                        long ss = strides[0];
                        byte* pq = (byte*)dataptrs[1];
                        long sq = strides[1];
                        for (long i = 0; i < n; i++, ps += ss, pq += sq)
                        {
                            readers[0].Read(ps, out var sep);
                            readers[1].Read(pq, out var seq);
                            long length = seq.Length == 0 ? 0 : seq.Length + (seq.Length - 1) * sep.Length;
                            if (length > 0)
                            {
                                if (points.Length < length)
                                    points = new uint[Math.Max(length, points.Length * 2)];
                                fixed (uint* p = points)
                                    WriteJoin((uint*)sep.Units, sep.Length, (uint*)seq.Units, seq.Length, p);
                                int bytes = EncodeUtf8(points, (int)length, ref utf8);
                                fixed (byte* u = utf8)
                                    if (allocator.newsize(u, bytes, cells + position * 16) < 0)
                                        throw new OutOfMemoryException("Failed to pack string in join");
                            }
                            position++;
                        }
                    });
                }
                finally
                {
                    allocator.release_allocator();
                }
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>Encodes code points (valid scalar values: a <c>StringDType</c> result) as UTF-8.</summary>
        /// <param name="points">The code points.</param>
        /// <param name="count">Their number.</param>
        /// <param name="utf8">The output buffer, grown when needed.</param>
        /// <returns>The number of bytes.</returns>
        private static int EncodeUtf8(uint[] points, int count, ref byte[] utf8)
        {
            if (utf8.Length < count * 4)
                utf8 = new byte[Math.Max(count * 4, utf8.Length * 2)];
            int n = 0;
            for (int k = 0; k < count; k++)
            {
                uint c = points[k];
                if (c < 0x80)
                {
                    utf8[n++] = (byte)c;
                }
                else if (c < 0x800)
                {
                    utf8[n++] = (byte)(0xC0 | (c >> 6));
                    utf8[n++] = (byte)(0x80 | (c & 0x3F));
                }
                else if (c < 0x10000)
                {
                    utf8[n++] = (byte)(0xE0 | (c >> 12));
                    utf8[n++] = (byte)(0x80 | ((c >> 6) & 0x3F));
                    utf8[n++] = (byte)(0x80 | (c & 0x3F));
                }
                else
                {
                    utf8[n++] = (byte)(0xF0 | (c >> 18));
                    utf8[n++] = (byte)(0x80 | ((c >> 12) & 0x3F));
                    utf8[n++] = (byte)(0x80 | ((c >> 6) & 0x3F));
                    utf8[n++] = (byte)(0x80 | (c & 0x3F));
                }
            }
            return n;
        }

        /// <summary>NumPy's empty object result of a join over an object sequence: an empty list array of the shape.</summary>
        /// <param name="outer">The broadcast shape (size 0).</param>
        /// <param name="kind">The pieces' class (irrelevant: there are no pieces).</param>
        /// <returns>The empty list array.</returns>
        private static NDListArray EmptyLists(Shape outer, char kind)
            => new NDListArray(new NDArray(NPTypeCode.Int64, outer.Clean(), true), NDListHeap.Empty(kind));

        // =============================================================================================================
        // The walk
        // =============================================================================================================

        /// <summary>The readers of the given arguments, in their slot order (null arguments skipped).</summary>
        /// <param name="args">The arguments.</param>
        /// <returns>The readers.</returns>
        private static Operand[] BuildReaders(params VecArg[] args)
        {
            var readers = new List<Operand>();
            foreach (var arg in args)
                if (arg is not null)
                    readers.Add(new Operand(arg));
            return readers.ToArray();
        }

        /// <summary>
        ///     The C-order walk of a <c>_vec_string</c> call: NDIter over the operands (broadcast, each with its own dtype: no
        ///     casting, no buffer; <c>EXTERNAL_LOOP</c> chunks), every <c>StringDType</c> operand's allocator held for the
        ///     walk.
        /// </summary>
        /// <param name="ops">The operands (the broadcast was checked with NumPy's text before).</param>
        /// <param name="readers">The readers (their allocators are locked).</param>
        /// <param name="chunk">Called per chunk, in C order.</param>
        /// <exception cref="Exception">Whatever <paramref name="chunk"/> throws; the allocators are released first.</exception>
        private static void Walk(List<NDArray> ops, Operand[] readers, NDInnerLoopFunc chunk)
        {
            var allocators = new List<VStringAllocator>();
            foreach (var r in readers)
                if (r.Allocator is not null)
                    allocators.Add(r.Allocator);
            var flags = new NDIterPerOpFlags[ops.Count];
            for (int k = 0; k < flags.Length; k++)
                flags[k] = NDIterPerOpFlags.READONLY;
            // The descriptor construction (null entries: each operand's own descriptor): a fixed-width text operand has no
            // type-code element size, its descriptor carries it.
            using var iter = NDIterRef.AdvancedNew(ops.Count, ops.ToArray(),
                NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.ZEROSIZE_OK | NDIterGlobalFlags.REFS_OK,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_NO_CASTING, flags, opDescrs: new DType[ops.Count]);
            if (allocators.Count > 0)
                VStringAllocator.acquire_allocators(allocators);
            try
            {
                iter.ForEach(chunk);
            }
            finally
            {
                if (allocators.Count > 0)
                    VStringAllocator.release_allocators(allocators);
            }
        }
    }
}
