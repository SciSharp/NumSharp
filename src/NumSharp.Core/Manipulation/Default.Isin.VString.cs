using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Sorting;
using NumSharp.Backends.Strings;

// =============================================================================
// Default.Isin.VString.cs — np.isin when an operand is NEP 55 StringDType text (T), NumPy 2.4.2's loop path
// =============================================================================
//
// WHAT NUMPY DOES (numpy/lib/_arraysetops_impl.py, _isin; probed)
//   A StringDType descriptor has objects (dtype.hasobject is True), so `contains_object` sends every call to the LOOP path
//   whatever the sizes and whatever `kind` says ('table' is refused first: a StringDType array is not an integer one):
//       mask = zeros(len(ar1)); for a in ar2: mask |= (ar1 == a)          (invert: ones; mask &= (ar1 != a))
//   Iterating ar2 yields Python objects: a text cell's str (exactly, NULs included), an NA cell's NA object (None, a float
//   NaN, the sentinel str, or '' for a descriptor without one), a U element's str, an S element's bytes, a number. Then
//   `ar1 == a` converts `a` back into an array:
//     * a str becomes a U scalar — which drops its trailing NULs ('a\0' compares as 'a') — and ar1 compares as text: a
//       StringDType ar1 with ITS descriptor's NA rules (a NaN-like NA is False under == AND under !=; every other NA
//       compares as the descriptor's default string — '' for a None-like NA, whose == / != loop overwrites its null
//       answer with the compare of the loaded strings), a U ar1 as text;
//     * None (and an opaque object) becomes an object array, compared with Python's ==: only an NA cell whose NA object
//       equals it (None == None) is True;
//     * a number or bytes has no comparison loop with text: all False under ==, all True under !=. A numeric ar1 compares
//       numbers numerically and an S ar1 bytes by value.
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   The loop's OR / AND over ar2 is folded into set lookups: ar2's strings go into one set (NULs stripped), its None-like
//   or opaque NA object, its numbers and its bytes are kept apart, and each element of ar1 gets the loop's answer from its
//   kind: a text (or an NA read as a default string) is True exactly when the set holds it; a NaN-like NA is False, and
//   under invert True only when ar2 holds no string; a None-like NA is True when the set holds '' or one of ar2's object
//   values equals its NA object. A numeric ar1 runs NumPy's numeric comparison once per distinct number of ar2. One pass,
//   O(len(ar1) + len(ar2)), where NumPy compares every pair.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>_isin</c> when <paramref name="element"/> or <paramref name="test_elements"/> is a
        ///     <c>StringDType</c> array (the <c>kind</c> name already validated): the loop path, see the file header.
        /// </summary>
        /// <param name="element">The tested values (the result has their shape).</param>
        /// <param name="test_elements">The set (flattened).</param>
        /// <param name="invert">Whether the answer is <c>not in</c> (NumPy's AND of <c>!=</c>, which is NOT the negation of
        ///     the answer for a NaN-like NA).</param>
        /// <param name="kind">The validated <c>kind</c>: <c>"table"</c> is refused (a <c>StringDType</c> array is not an
        ///     integer array); <c>"sort"</c> and null both take the loop path.</param>
        /// <returns>A bool array of <paramref name="element"/>'s shape: a view of a fresh 1-D answer, as NumPy's reshape
        ///     returns it.</returns>
        /// <exception cref="ValueError"><c>kind='table'</c>.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> set whose NA object is a sequence or an array, held by
        ///     an NA cell: NumPy compares the element against that sequence ELEMENTWISE (a broadcast), which this does not
        ///     model.</exception>
        private static NDArray IsinVString(NDArray element, NDArray test_elements, bool invert, string kind)
        {
            if (kind == "table")
                throw new ValueError("The 'table' method is only supported for boolean or integer arrays. "
                    + "Please select 'sort' or None for kind.");
            var dims = (long[])element.Shape.dimensions.Clone();
            Shape shape = dims.Length == 0 ? Shape.NewScalar() : Shape.Adopt(dims);
            using NDArray flat = IsinVStringFlat(element, test_elements, invert);
            return flat.reshape(shape);
        }

        /// <summary>
        ///     The 1-D answer of <see cref="IsinVString"/>: one flag per element of <paramref name="element"/> in C order.
        /// </summary>
        /// <param name="element">The tested values.</param>
        /// <param name="test_elements">The set.</param>
        /// <param name="invert">Whether the answer is <c>not in</c>.</param>
        /// <returns>A fresh 1-D bool array.</returns>
        /// <exception cref="NotSupportedException">See <see cref="IsinVString"/>.</exception>
        /// <remarks>Scoped: the ravels and the per-number comparisons are reclaimed; only the answer survives.</remarks>
        [NDScoped]
        private static NDArray IsinVStringFlat(NDArray element, NDArray test_elements, bool invert)
        {
            NDArray ar1 = np.ravel(element);
            NDArray ar2 = np.ravel(test_elements);
            using var set = new IsinObjects();
            set.Collect(ar2);
            long n1 = ar1.size;
            var mask = new NDArray(NPTypeCode.Boolean, Shape.Vector(n1), false);
            if (n1 == 0)
                return mask;

            if (Backends.Strings.NDStringUfuncs.IsVString(ar1))
            {
                IsinVStringElements(ar1, set, invert, mask);
                return mask;
            }
            // Not a StringDType ar1: the loop's comparisons that can be True are text against text (a U ar1), bytes against
            // bytes (an S ar1) and numbers against numbers (a numeric ar1); every other pair has no loop (all False / True).
            unsafe
            {
                bool* m = (bool*)mask.Address;
                for (long i = 0; i < n1; i++)
                    m[i] = invert;
            }
            if (ar1.typecode == NPTypeCode.String)
            {
                IsinFixedWidthElements(ar1, set, invert, mask);
                return mask;
            }
            // A numeric ar1: NumPy's `ar1 == a` for each number of the set (a Python number is a weak scalar).
            foreach (object number in set.Numbers)
            {
                using NDArray scalar = NDArray.FromPythonOperand(number);
                using NDArray hit = invert ? np.not_equal(ar1, scalar) : np.equal(ar1, scalar);
                if (invert)
                    np.logical_and(mask, hit, @out: mask);
                else
                    np.logical_or(mask, hit, @out: mask);
            }
            return mask;
        }

        /// <summary>
        ///     The flags of a <c>StringDType</c> <paramref name="ar1"/>, cell by cell: a text (or an NA read as a default
        ///     string) by set membership, a NaN-like NA False (under invert: True only when the set holds no string), a
        ///     None-like NA as <c>''</c> by set membership or by its NA object against the set's object values.
        /// </summary>
        /// <param name="ar1">The flattened <c>StringDType</c> values (any stride).</param>
        /// <param name="set">The set's objects.</param>
        /// <param name="invert">Whether the answer is <c>not in</c>.</param>
        /// <param name="mask">The 1-D bool output, one flag per element.</param>
        private static unsafe void IsinVStringElements(NDArray ar1, IsinObjects set, bool invert, NDArray mask)
        {
            var rule = NDVStringCells.RuleOf(ar1.dtype);
            var p = ar1.dtype.VString;
            // Under == / != an NA compares as the descriptor's default string under every rule but the NaN-like one ('' for a
            // None-like NA: NumPy's loop overwrites the null answer with the compare of the loaded strings, an NA loading as
            // an empty one); a None-like NA also meets the set's object values through the object comparison.
            byte[] defaultText = NDUtf8.Encode(p.default_string);
            bool defaultHit;
            fixed (byte* d = defaultText)
                defaultHit = set.ContainsText(defaultText.Length == 0 ? (byte*)1 : d, defaultText.Length);
            bool nullHit = defaultHit || set.ObjectEqualsAny(p.na_object);
            bool* m = (bool*)mask.Address;
            long n = ar1.size;
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            var allocator = ar1.Storage.VStringSlice.Allocator;
            allocator.acquire_allocator();
            try
            {
                byte* cells = (byte*)ar1.Address + ar1.Shape.offset * (long)cell;
                long stride = ar1.Shape.strides[0] * (long)cell;
                byte* arena = allocator.ArenaBase;
                for (long i = 0; i < n; i++)
                {
                    int r = NDVStringCells.Load(cells + i * stride, arena, out byte* buf, out long size);
                    bool eq;
                    if (r == 0)
                        eq = set.ContainsText(buf, size);
                    else if (r < 0)
                        throw VStringSort.CorruptCellError();
                    else if (rule == VStringNARule.NaN)
                    {
                        // == is False against every value; != is False against a string and True otherwise.
                        m[i] = invert && !set.AnyString;
                        continue;
                    }
                    else if (rule == VStringNARule.Null)
                        eq = nullHit;
                    else
                        eq = defaultHit;
                    m[i] = invert ? !eq : eq;
                }
            }
            finally
            {
                allocator.release_allocator();
            }
        }

        /// <summary>
        ///     The flags of a fixed-width text <paramref name="ar1"/> against a set from a <c>StringDType</c> array: a
        ///     <c>U</c> element is True when the set holds its text; an <c>S</c> element when the set holds its bytes (only an
        ///     NA object can be bytes).
        /// </summary>
        /// <param name="ar1">The flattened <c>S</c> / <c>U</c> values.</param>
        /// <param name="set">The set's objects.</param>
        /// <param name="invert">Whether the answer is <c>not in</c>.</param>
        /// <param name="mask">The 1-D bool output (already holding <paramref name="invert"/>).</param>
        private static unsafe void IsinFixedWidthElements(NDArray ar1, IsinObjects set, bool invert, NDArray mask)
        {
            bool unicode = ar1.dtype.kind == 'U';
            if (unicode ? !set.AnyString : set.Bytes.Count == 0)
                return;
            bool* m = (bool*)mask.Address;
            long n = ar1.size;
            for (long i = 0; i < n; i++)
            {
                object value = ar1.Storage.GetAtIndex(i);
                bool eq;
                if (unicode)
                {
                    byte[] text = NDUtf8.Encode(((string)value).TrimEnd('\0'));
                    fixed (byte* t = text)
                        eq = set.ContainsText(text.Length == 0 ? (byte*)1 : t, text.Length);
                }
                else
                    eq = set.ContainsBytes((byte[])value);
                m[i] = invert ? !eq : eq;
            }
        }

        /// <summary>
        ///     The Python objects iterating a test set yields, sorted by how <c>ar1 == a</c> treats them: strings (NULs
        ///     stripped, as the U scalar a str becomes drops them) in a text set, None-like and opaque NA objects (object
        ///     comparison), numbers (numeric comparison) and bytes.
        /// </summary>
        private sealed class IsinObjects : IDisposable
        {
            /// <summary>The strings, as UTF-8 without trailing NULs.</summary>
            private readonly VStringDistinct.ByteTextSet _texts = new VStringDistinct.ByteTextSet();

            /// <summary>The objects NumPy compares with Python's <c>==</c> (None, opaque objects).</summary>
            private readonly List<object> _objects = new List<object>();

            /// <summary>Whether the set yielded a string at all (an empty string counts).</summary>
            internal bool AnyString { get; private set; }

            /// <summary>The numbers, one per distinct Python value.</summary>
            internal List<object> Numbers { get; } = new List<object>();

            /// <summary>The bytes values.</summary>
            internal List<byte[]> Bytes { get; } = new List<byte[]>();

            /// <summary>
            ///     Reads every element of the flattened set: a <c>StringDType</c> cell's text or NA object, a <c>U</c>
            ///     element's text, an <c>S</c> element's bytes, a number (a char element is NumPy's uint16 number).
            /// </summary>
            /// <param name="ar2">The flattened set.</param>
            /// <exception cref="NotSupportedException">An NA object that is a sequence or an array (see
            ///     <see cref="IsinVString"/>).</exception>
            internal unsafe void Collect(NDArray ar2)
            {
                long n = ar2.size;
                if (n == 0)
                    return;
                if (Backends.Strings.NDStringUfuncs.IsVString(ar2))
                {
                    var p = ar2.dtype.VString;
                    bool sawNA = false;
                    const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
                    var allocator = ar2.Storage.VStringSlice.Allocator;
                    allocator.acquire_allocator();
                    try
                    {
                        byte* cells = (byte*)ar2.Address + ar2.Shape.offset * (long)cell;
                        long stride = ar2.Shape.strides[0] * (long)cell;
                        byte* arena = allocator.ArenaBase;
                        for (long i = 0; i < n; i++)
                        {
                            int r = NDVStringCells.Load(cells + i * stride, arena, out byte* buf, out long size);
                            if (r < 0)
                                throw VStringSort.CorruptCellError();
                            if (r == 1)
                            {
                                sawNA = true;
                                continue;
                            }
                            AddText(buf, size);
                        }
                    }
                    finally
                    {
                        allocator.release_allocator();
                    }
                    // An NA cell yields the NA object (getitem), or '' for a descriptor without one.
                    if (sawNA)
                        AddObject(p.HasNAObject ? p.na_object : string.Empty);
                    return;
                }
                // Iterating a char set yields NumPy's uint16 numbers: a char ELEMENT is its uint16 proxy (as np.searchsorted's
                // VStringSearch.ReadObjects reads it), which no text equals. A C# char VALUE (an NA object, above) stays a str
                // in AddObject, the boundary map's Python str.
                bool chars = ar2.typecode == NPTypeCode.Char;
                for (long i = 0; i < n; i++)
                {
                    object value = ar2.Storage.GetAtIndex(i);
                    AddObject(chars ? (object)(ushort)(char)value : value);
                }
            }

            /// <summary>Adds one yielded object by kind.</summary>
            /// <param name="value">The object (null is None).</param>
            /// <exception cref="NotSupportedException">A sequence or an array.</exception>
            private unsafe void AddObject(object value)
            {
                switch (value)
                {
                    case string s:
                    {
                        byte[] text = NDUtf8.Encode(s);
                        fixed (byte* t = text)
                            AddText(text.Length == 0 ? (byte*)1 : t, text.Length);
                        return;
                    }
                    case char c:
                        AddObject(c.ToString());
                        return;
                    case byte[] bytes:
                        Bytes.Add(bytes);
                        return;
                    case bool or sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger
                        or float or double or decimal or Complex or Half:
                        foreach (object known in Numbers)
                            if (NDPyObject.Equal(known, value))
                                return;
                        Numbers.Add(value);
                        return;
                    case NDArray or ITuple or IList:
                        // np.asarray turns a sequence into an array and `ar1 == a` broadcasts against it elementwise.
                        throw new NotSupportedException(
                            $"np.isin against a StringDType test set whose NA object is a sequence ({NDPyObject.TypeName(value)}): " +
                            "NumPy compares each element with that sequence elementwise, which NumSharp does not model.");
                    default:
                        // None and opaque objects: an object array, Python's ==.
                        _objects.Add(value);
                        return;
                }
            }

            /// <summary>Adds a string (its trailing NULs stripped, as NumPy's U scalar drops them).</summary>
            /// <param name="buf">The UTF-8 text.</param>
            /// <param name="size">Its size.</param>
            private unsafe void AddText(byte* buf, long size)
            {
                while (size > 0 && buf[size - 1] == 0)
                    size--;
                _texts.Add(buf, size);
                AnyString = true;
            }

            /// <summary>Whether the set holds a string equal to the given UTF-8 text (compared exactly).</summary>
            /// <param name="buf">The text.</param>
            /// <param name="size">Its size.</param>
            /// <returns>True when present.</returns>
            internal unsafe bool ContainsText(byte* buf, long size) => _texts.Count > 0 && _texts.Contains(buf, size);

            /// <summary>Whether the set holds bytes equal to <paramref name="value"/>.</summary>
            /// <param name="value">The bytes.</param>
            /// <returns>True when present.</returns>
            internal bool ContainsBytes(byte[] value)
            {
                foreach (var b in Bytes)
                    if (b.AsSpan().SequenceEqual(value))
                        return true;
                return false;
            }

            /// <summary>Whether <paramref name="naObject"/> equals one of the set's object values under Python's
            ///     <c>==</c> (NumPy's object comparison of a None-like NA).</summary>
            /// <param name="naObject">The NA object of the tested array.</param>
            /// <returns>True when an equal object is present.</returns>
            internal bool ObjectEqualsAny(object naObject)
            {
                foreach (var o in _objects)
                    if (NDPyObject.PyEq(naObject, o))
                        return true;
                return false;
            }

            /// <summary>Returns the text set's bytes.</summary>
            public void Dispose() => _texts.Dispose();
        }
    }
}
