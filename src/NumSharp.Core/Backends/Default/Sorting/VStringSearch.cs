using System;
using System.Numerics;
using NumSharp.Backends.Strings;

// =============================================================================
// VStringSearch.cs — np.searchsorted with a NEP 55 StringDType operand: NumPy 2.4.2's npy_binsearch / npy_argbinsearch
// answers over the StringDType compare, and over Python objects when the operands have no common StringDType
// =============================================================================
//
// WHAT NUMPY DOES (multiarray/item_selection.c PyArray_SearchSorted, npysort/binsearch.cpp; probed)
//   * The descriptor: `PyArray_DescrFromObject(keys, haystack.dtype)` — discovery promoted with the haystack's descriptor,
//     a failed promotion making it OBJECT. StringDType promotes with StringDType (the descriptor with the NA object, when
//     the two NA objects are compatible) and with U (the StringDType descriptor); with numbers, bools, bytes, or two
//     incompatible NA objects it fails: object.
//   * The keys, then the haystack (its rank checked first), are converted to that descriptor under the 'safe' rule: a U
//     ARRAY into StringDType is only 'same_kind' (TypeError "Cannot cast array data from dtype('<U1') to StringDType()
//     according to the rule 'safe'"); a Python str key is packed, not cast; everything converts to object.
//   * The generic binary search, for each key in C order:
//       if (cmp(last_key, key) < 0) max_idx = arr_len;                         // ('right': <= 0); the FIRST key compares
//       else { min_idx = 0; max_idx = max_idx < arr_len ? max_idx + 1 : arr_len; }   // with itself
//       last_key = key;
//       while (min_idx < max_idx) { mid = ...; if (cmp(arr[mid], key) < 0) min_idx = mid + 1; else max_idx = mid; }
//     With a sorter, arr[mid] is arr[sorter[mid]], and an index outside the haystack returns at once with "Sorter index
//     out of range." (that ValueError replaces an error already raised by a comparison).
//   * StringDType's compare (_compare): texts by NpyString_cmp; a NaN-like NA as the FIRST operand is 1 (even against
//     another NA: a NaN-like NA key never "equals" anything, so 'left' and 'right' both find the first NA), as the second
//     -1; a None-like NA raises ValueError("Cannot compare null that is not a nan-like value") at the first comparison
//     that touches it — the comparison then answers 0 and the search goes on, so the result is the error — even the
//     first key's comparison with itself; any other NA reads as the default string.
//   * OBJECT_compare: `a < b` (Python's rich comparison; a TypeError "'<' not supported between instances of 'str' and
//     'int'" is the call's error, every later comparison answering 0), else `a > b`, else 0. The elements are the Python
//     objects the arrays' getitem gives: a StringDType text's str, its NA object (or '' without one), numbers, bools,
//     bytes, strs.
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   NumPy's loop step for step (the carry and the self-comparison included), each comparison answered from the cells
//   (StringDType path) or from the elements read as Python objects (object path), with the pending-error semantics above.
// =============================================================================

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    ///     <c>np.searchsorted</c> with a <c>StringDType</c> operand: NumPy's generic binary search over the StringDType compare
    ///     or over Python objects (see the file header).
    /// </summary>
    internal static unsafe class VStringSearch
    {
        // ============================== the StringDType path ==============================

        /// <summary>
        ///     The search of <paramref name="keys"/> in <paramref name="haystack"/>, both <c>StringDType</c>, under the NA rule of
        ///     the common descriptor <paramref name="common"/> (NumPy converts both operands to it, which keeps every text and
        ///     every NA cell).
        /// </summary>
        /// <param name="haystack">The 1-D haystack (any stride; may be empty).</param>
        /// <param name="keys">The keys (any shape and layout; may be empty).</param>
        /// <param name="left">True for <c>side='left'</c>.</param>
        /// <param name="sorter">The validated sorter (C-contiguous int64 of the haystack's length), or null.</param>
        /// <param name="common">The common descriptor (its NA rule and default string decide the NA comparisons).</param>
        /// <returns>A fresh int64 array of the keys' shape (0-d for a 0-d key).</returns>
        /// <exception cref="ValueError">A sorter index outside the haystack that the search reaches (<c>Sorter index out of
        ///     range.</c>), else a comparison that touched a None-like NA (<c>Cannot compare null that is not a nan-like
        ///     value</c>).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt cell.</exception>
        internal static NDArray SearchSorted(NDArray haystack, NDArray keys, bool left, NDArray sorter, DType common)
        {
            var result = new NDArray(NPTypeCode.Int64,
                keys.ndim == 0 ? Shape.NewScalar() : Shape.Adopt((long[])keys.Shape.dimensions.Clone()), false);
            long m = keys.size;
            if (m == 0)
                return result;
            // NumPy's keys are a C-contiguous array: they are read in C order.
            using NDArray k = keys.Shape.IsContiguous ? null : NDArray.AsContiguousArray(keys, null);
            NDArray kc = k ?? keys;
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            var rule = NDVStringCells.RuleOf(common);
            byte[] defaultText = NDUtf8.Encode(common.VString.default_string);
            var hayAlloc = haystack.Storage.VStringSlice.Allocator;
            var keyAlloc = kc.Storage.VStringSlice.Allocator;
            // Both allocators locked (one lock when they are the same allocator): the arena bases stay valid.
            VStringAllocator.acquire_allocators(hayAlloc, keyAlloc);
            try
            {
                fixed (byte* d = defaultText)
                {
                    var ctx = new Ctx
                    {
                        HayCells = (byte*)haystack.Address + haystack.Shape.offset * (long)cell,
                        HayStride = haystack.Shape.strides[0] * (long)cell,
                        HayArena = hayAlloc.ArenaBase,
                        KeyCells = (byte*)kc.Address + kc.Shape.offset * (long)cell,
                        KeyArena = keyAlloc.ArenaBase,
                        Rule = rule,
                        DefaultText = defaultText.Length == 0 ? (byte*)1 : d,
                        DefaultSize = defaultText.Length,
                    };
                    long n = haystack.size;
                    long* sort = sorter is null ? null : (long*)sorter.Address + sorter.Shape.offset;
                    long* ret = (long*)result.Address;
                    bool sorterError = Search(ref ctx, n, m, left, sort, ret);
                    if (sorterError)
                        throw new ValueError("Sorter index out of range.");
                    if (ctx.NullError)
                        throw VStringSort.NullCompareError();
                }
            }
            finally
            {
                VStringAllocator.release_allocators(hayAlloc, keyAlloc);
            }
            return result;
        }

        /// <summary>The cells and the NA rule one StringDType search reads, and its pending error.</summary>
        private struct Ctx
        {
            /// <summary>The haystack's element 0.</summary>
            internal byte* HayCells;

            /// <summary>The byte distance between haystack cells.</summary>
            internal long HayStride;

            /// <summary>The haystack allocator's arena base.</summary>
            internal byte* HayArena;

            /// <summary>The keys' element 0 (C-contiguous cells).</summary>
            internal byte* KeyCells;

            /// <summary>The keys allocator's arena base.</summary>
            internal byte* KeyArena;

            /// <summary>The common descriptor's NA rule.</summary>
            internal VStringNARule Rule;

            /// <summary>The default string an NA reads as (None / String rules).</summary>
            internal byte* DefaultText;

            /// <summary>Its size.</summary>
            internal long DefaultSize;

            /// <summary>Set once a comparison touched a None-like NA (NumPy's raised-but-continuing error).</summary>
            internal bool NullError;
        }

        /// <summary>
        ///     NumPy's <c>npy_binsearch</c> / <c>npy_argbinsearch</c> loop over the StringDType compare.
        /// </summary>
        /// <param name="ctx">The cells and rule (its pending error is set here).</param>
        /// <param name="n">The haystack length.</param>
        /// <param name="m">The number of keys.</param>
        /// <param name="left">True for <c>side='left'</c> (less: <c>cmp &lt; 0</c>; right: <c>cmp &lt;= 0</c>).</param>
        /// <param name="sort">The sorter, or null.</param>
        /// <param name="ret">The output, one slot per key.</param>
        /// <returns>True when a sorter index outside the haystack was reached (the search stopped there).</returns>
        private static bool Search(ref Ctx ctx, long n, long m, bool left, long* sort, long* ret)
        {
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            long minIdx = 0, maxIdx = n;
            byte* lastKey = ctx.KeyCells;
            for (long k = 0; k < m; k++)
            {
                byte* key = ctx.KeyCells + k * cell;
                if (Less(ref ctx, lastKey, ctx.KeyArena, key, ctx.KeyArena, left))
                    maxIdx = n;
                else
                {
                    minIdx = 0;
                    maxIdx = maxIdx < n ? maxIdx + 1 : n;
                }
                lastKey = key;
                while (minIdx < maxIdx)
                {
                    long mid = minIdx + ((maxIdx - minIdx) >> 1);
                    long index = mid;
                    if (sort != null)
                    {
                        index = sort[mid];
                        if (index < 0 || index >= n)
                            return true;
                    }
                    if (Less(ref ctx, ctx.HayCells + index * ctx.HayStride, ctx.HayArena, key, ctx.KeyArena, left))
                        minIdx = mid + 1;
                    else
                        maxIdx = mid;
                }
                ret[k] = minIdx;
            }
            return false;
        }

        /// <summary>The search's ordering test: <c>cmp(a, b) &lt; 0</c> for 'left', <c>&lt;= 0</c> for 'right'.</summary>
        /// <param name="ctx">The rule and pending error.</param>
        /// <param name="a">The first cell.</param>
        /// <param name="arenaA">Its arena base.</param>
        /// <param name="b">The second cell.</param>
        /// <param name="arenaB">Its arena base.</param>
        /// <param name="left">The side.</param>
        /// <returns>The test's answer.</returns>
        private static bool Less(ref Ctx ctx, byte* a, byte* arenaA, byte* b, byte* arenaB, bool left)
        {
            int c = Compare(ref ctx, a, arenaA, b, arenaB);
            return left ? c < 0 : c <= 0;
        }

        /// <summary>
        ///     NumPy's StringDType <c>_compare</c> under the common rule: texts by <c>NpyString_cmp</c>; a NaN-like NA first
        ///     is 1, second -1; a None-like NA sets the pending error and answers 0; any other NA reads as the default string.
        /// </summary>
        /// <param name="ctx">The rule and pending error.</param>
        /// <param name="a">The first cell.</param>
        /// <param name="arenaA">Its arena base.</param>
        /// <param name="b">The second cell.</param>
        /// <param name="arenaB">Its arena base.</param>
        /// <returns>Negative, zero or positive.</returns>
        /// <exception cref="OutOfMemoryException">A corrupt cell.</exception>
        private static int Compare(ref Ctx ctx, byte* a, byte* arenaA, byte* b, byte* arenaB)
        {
            int ra = NDVStringCells.Load(a, arenaA, out byte* ba, out long sa);
            int rb = NDVStringCells.Load(b, arenaB, out byte* bb, out long sb);
            if (ra < 0 || rb < 0)
                throw VStringSort.CorruptCellError();
            if (ra == 1 || rb == 1)
            {
                switch (ctx.Rule)
                {
                    case VStringNARule.NaN:
                        return ra == 1 ? 1 : -1;
                    case VStringNARule.Null:
                        ctx.NullError = true;
                        return 0;
                    default:
                        if (ra == 1)
                        {
                            ba = ctx.DefaultText;
                            sa = ctx.DefaultSize;
                        }
                        if (rb == 1)
                        {
                            bb = ctx.DefaultText;
                            sb = ctx.DefaultSize;
                        }
                        break;
                }
            }
            return NDVStringCells.Order(ba, sa, bb, sb);
        }

        // ============================== the object path ==============================

        /// <summary>
        ///     The search when the operands have no common StringDType descriptor: every element read as the Python object
        ///     NumPy's conversion to object gives, compared with Python's <c>&lt;</c> / <c>&gt;</c> (OBJECT_compare).
        /// </summary>
        /// <param name="haystack">The 1-D haystack (any dtype; may be empty).</param>
        /// <param name="keys">The keys (any dtype, shape and layout; may be empty).</param>
        /// <param name="left">True for <c>side='left'</c>.</param>
        /// <param name="sorter">The validated sorter (C-contiguous int64), or null.</param>
        /// <returns>A fresh int64 array of the keys' shape (0-d for a 0-d key).</returns>
        /// <exception cref="ValueError">A sorter index outside the haystack that the search reaches.</exception>
        /// <exception cref="TypeError">The first comparison Python refuses (<c>'&lt;' not supported between instances of
        ///     'str' and 'int'</c>, the operands' types in the comparison's order).</exception>
        internal static NDArray SearchSortedObjects(NDArray haystack, NDArray keys, bool left, NDArray sorter)
        {
            var result = new NDArray(NPTypeCode.Int64,
                keys.ndim == 0 ? Shape.NewScalar() : Shape.Adopt((long[])keys.Shape.dimensions.Clone()), false);
            long m = keys.size;
            if (m == 0)
                return result;
            PyObj[] hay = ReadObjects(haystack);
            PyObj[] ks = ReadObjects(keys);
            long n = hay.Length;
            long* sort = sorter is null ? null : (long*)sorter.Address + sorter.Shape.offset;
            long* ret = (long*)result.Address;
            TypeError error = null;
            long minIdx = 0, maxIdx = n;
            int last = 0;
            for (long k = 0; k < m; k++)
            {
                if (LessObj(ks[last], ks[k], left, ref error))
                    maxIdx = n;
                else
                {
                    minIdx = 0;
                    maxIdx = maxIdx < n ? maxIdx + 1 : n;
                }
                last = (int)k;
                while (minIdx < maxIdx)
                {
                    long mid = minIdx + ((maxIdx - minIdx) >> 1);
                    long index = mid;
                    if (sort != null)
                    {
                        index = sort[mid];
                        if (index < 0 || index >= n)
                            throw new ValueError("Sorter index out of range.");
                    }
                    if (LessObj(hay[index], ks[k], left, ref error))
                        minIdx = mid + 1;
                    else
                        maxIdx = mid;
                }
                ret[k] = minIdx;
            }
            if (error != null)
                throw error;
            return result;
        }

        /// <summary>The object search's ordering test (<c>cmp &lt; 0</c> / <c>&lt;= 0</c>), OBJECT_compare's pending error.</summary>
        /// <param name="a">The first object.</param>
        /// <param name="b">The second object.</param>
        /// <param name="left">The side.</param>
        /// <param name="error">The pending error: once set, every comparison answers 0 (NumPy's <c>PyErr_Occurred</c> test).</param>
        /// <returns>The test's answer.</returns>
        private static bool LessObj(in PyObj a, in PyObj b, bool left, ref TypeError error)
        {
            int c;
            if (error != null)
                c = 0;
            else
            {
                c = PyObj.ObjectCompare(a, b, out error);
            }
            return left ? c < 0 : c <= 0;
        }

        /// <summary>
        ///     The Python objects NumPy's conversion to object gives for every element of <paramref name="a"/>, in C order: a
        ///     StringDType text's str, its NA object ('' without one), U strs, S bytes, Python numbers and bools; a char
        ///     element is the Python int of its uint16 (NumSharp's char is NumPy's uint16 wherever it meets NumPy).
        /// </summary>
        /// <param name="a">The array.</param>
        /// <returns>The objects.</returns>
        /// <remarks>The char rule is the array's, not a C# value's: the boundary map (<c>PyObj.From</c>) reads a C# char as a
        ///     one-character str, which is right for a value a caller writes but not for an element of a char array, whose
        ///     store, casts and text are uint16's (a char compared with StringDType text here is NumPy's int-and-str
        ///     TypeError, as the uint16 array is).</remarks>
        private static PyObj[] ReadObjects(NDArray a)
        {
            long n = a.size;
            var objs = new PyObj[n];
            if (n == 0)
                return objs;
            if (NDStringUfuncs.IsVString(a))
            {
                var p = a.dtype.VString;
                PyObj na = PyObj.From(p.HasNAObject ? p.na_object : string.Empty);
                using NDArray c = a.Shape.IsContiguous ? null : NDArray.AsContiguousArray(a, null);
                NDArray src = c ?? a;
                const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
                var allocator = src.Storage.VStringSlice.Allocator;
                allocator.acquire_allocator();
                try
                {
                    byte* cells = (byte*)src.Address + src.Shape.offset * (long)cell;
                    byte* arena = allocator.ArenaBase;
                    for (long i = 0; i < n; i++)
                    {
                        int r = NDVStringCells.Load(cells + i * cell, arena, out byte* buf, out long size);
                        if (r < 0)
                            throw VStringSort.CorruptCellError();
                        objs[i] = r == 1 ? na : PyObj.Text(new ReadOnlySpan<byte>(buf, checked((int)size)).ToArray());
                    }
                }
                finally
                {
                    allocator.release_allocator();
                }
                return objs;
            }
            // Fixed-width text and numbers: the element values in C order (GetAtIndex walks the logical order); a char
            // element as its uint16 number (see the remarks).
            using NDArray flat = a.Shape.IsContiguous && a.Shape.offset == 0
                ? null
                : NDArray.AsContiguousArray(a, null);
            NDArray s = flat ?? a;
            bool charElements = s.typecode == NPTypeCode.Char;
            for (long i = 0; i < n; i++)
            {
                object element = s.Storage.GetAtIndex(i);
                objs[i] = PyObj.From(charElements ? (ushort)(char)element : element);
            }
            return objs;
        }

        /// <summary>The kind of a <see cref="PyObj"/>.</summary>
        private enum PyKind : byte
        {
            /// <summary>Python <c>str</c> (UTF-8 bytes: their order is code-point order).</summary>
            Str,

            /// <summary>Python <c>bytes</c>.</summary>
            Bytes,

            /// <summary>Python <c>int</c>.</summary>
            Int,

            /// <summary>Python <c>bool</c> (an int for comparisons, named <c>bool</c> in errors).</summary>
            Bool,

            /// <summary>Python <c>float</c>.</summary>
            Float,

            /// <summary>Python <c>complex</c> (orders against nothing).</summary>
            Complex,

            /// <summary>Python <c>None</c> (orders against nothing).</summary>
            None,

            /// <summary>Any other object (orders against nothing).</summary>
            Other,
        }

        /// <summary>
        ///     A Python object as OBJECT_compare meets it: its kind, its comparable value, and its type name for errors.
        /// </summary>
        private readonly struct PyObj
        {
            /// <summary>The kind.</summary>
            private readonly PyKind _kind;

            /// <summary>A str's UTF-8 bytes or a bytes value.</summary>
            private readonly byte[] _bytes;

            /// <summary>An int's or bool's value.</summary>
            private readonly BigInteger _int;

            /// <summary>A float's value.</summary>
            private readonly double _float;

            /// <summary>The type name an error quotes (for <see cref="PyKind.Other"/>).</summary>
            private readonly string _otherName;

            /// <summary>Builds an object.</summary>
            /// <param name="kind">The kind.</param>
            /// <param name="bytes">The text or bytes.</param>
            /// <param name="i">The integer.</param>
            /// <param name="f">The float.</param>
            /// <param name="otherName">The type name of an opaque object.</param>
            private PyObj(PyKind kind, byte[] bytes, BigInteger i, double f, string otherName)
            {
                _kind = kind;
                _bytes = bytes;
                _int = i;
                _float = f;
                _otherName = otherName;
            }

            /// <summary>A Python str from UTF-8 bytes.</summary>
            /// <param name="utf8">The text.</param>
            /// <returns>The object.</returns>
            internal static PyObj Text(byte[] utf8) => new PyObj(PyKind.Str, utf8, default, 0, null);

            /// <summary>The Python object a C# value stands for (the boundary map of <see cref="NDPyObject"/>).</summary>
            /// <param name="value">The value (null is None).</param>
            /// <returns>The object.</returns>
            internal static PyObj From(object value) => value switch
            {
                null => new PyObj(PyKind.None, null, default, 0, null),
                string s => Text(NDUtf8.Encode(s)),
                char c => Text(NDUtf8.Encode(c.ToString())),
                byte[] b => new PyObj(PyKind.Bytes, b, default, 0, null),
                bool b => new PyObj(PyKind.Bool, null, b ? BigInteger.One : BigInteger.Zero, 0, null),
                sbyte v => new PyObj(PyKind.Int, null, v, 0, null),
                byte v => new PyObj(PyKind.Int, null, v, 0, null),
                short v => new PyObj(PyKind.Int, null, v, 0, null),
                ushort v => new PyObj(PyKind.Int, null, v, 0, null),
                int v => new PyObj(PyKind.Int, null, v, 0, null),
                uint v => new PyObj(PyKind.Int, null, v, 0, null),
                long v => new PyObj(PyKind.Int, null, v, 0, null),
                ulong v => new PyObj(PyKind.Int, null, v, 0, null),
                BigInteger v => new PyObj(PyKind.Int, null, v, 0, null),
                float v => new PyObj(PyKind.Float, null, default, v, null),
                double v => new PyObj(PyKind.Float, null, default, v, null),
                Half v => new PyObj(PyKind.Float, null, default, (double)v, null),
                decimal v => new PyObj(PyKind.Float, null, default, (double)v, null),
                Complex => new PyObj(PyKind.Complex, null, default, 0, null),
                _ => new PyObj(PyKind.Other, null, default, 0, NDPyObject.TypeName(value)),
            };

            /// <summary>The type name Python's comparison errors quote.</summary>
            private string TypeName => _kind switch
            {
                PyKind.Str => "str",
                PyKind.Bytes => "bytes",
                PyKind.Int => "int",
                PyKind.Bool => "bool",
                PyKind.Float => "float",
                PyKind.Complex => "complex",
                PyKind.None => "NoneType",
                _ => _otherName,
            };

            /// <summary>Whether the object is a real number (int, bool or float).</summary>
            private bool IsReal => _kind is PyKind.Int or PyKind.Bool or PyKind.Float;

            /// <summary>
            ///     NumPy's OBJECT_compare: <c>a &lt; b</c> → -1, else <c>a &gt; b</c> → 1, else 0; a pair Python cannot order
            ///     raises its TypeError (returned in <paramref name="error"/>, the answer 0).
            /// </summary>
            /// <param name="a">The first object.</param>
            /// <param name="b">The second object.</param>
            /// <param name="error">Receives the TypeError of an unorderable pair, or null.</param>
            /// <returns>-1, 0 or 1.</returns>
            internal static int ObjectCompare(in PyObj a, in PyObj b, out TypeError error)
            {
                error = null;
                if (a._kind == PyKind.Str && b._kind == PyKind.Str || a._kind == PyKind.Bytes && b._kind == PyKind.Bytes)
                {
                    int c = a._bytes.AsSpan().SequenceCompareTo(b._bytes);
                    return c < 0 ? -1 : c > 0 ? 1 : 0;
                }
                if (a.IsReal && b.IsReal)
                    return CompareReal(a, b);
                error = new TypeError($"'<' not supported between instances of '{a.TypeName}' and '{b.TypeName}'");
                return 0;
            }

            /// <summary>Python's exact comparison of two real numbers (a NaN orders against nothing: 0).</summary>
            /// <param name="a">The first number.</param>
            /// <param name="b">The second number.</param>
            /// <returns>-1, 0 or 1.</returns>
            private static int CompareReal(in PyObj a, in PyObj b)
            {
                bool af = a._kind == PyKind.Float, bf = b._kind == PyKind.Float;
                if (!af && !bf)
                    return a._int.CompareTo(b._int);
                if (af && bf)
                    return a._float < b._float ? -1 : a._float > b._float ? 1 : 0;
                return af ? -CompareIntFloat(b._int, a._float) : CompareIntFloat(a._int, b._float);
            }

            /// <summary>Python's exact int-against-float comparison (a NaN orders against nothing: 0).</summary>
            /// <param name="i">The integer.</param>
            /// <param name="d">The float.</param>
            /// <returns>-1, 0 or 1.</returns>
            private static int CompareIntFloat(BigInteger i, double d)
            {
                if (double.IsNaN(d))
                    return 0;
                if (double.IsPositiveInfinity(d))
                    return -1;
                if (double.IsNegativeInfinity(d))
                    return 1;
                double f = Math.Floor(d);
                var fi = new BigInteger(f);
                int c = i.CompareTo(fi);
                if (c != 0)
                    return c;
                // i == floor(d): equal when d is integral, below d otherwise.
                return d == f ? 0 : -1;
            }
        }
    }
}
