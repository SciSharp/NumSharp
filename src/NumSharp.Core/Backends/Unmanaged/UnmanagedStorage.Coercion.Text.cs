using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp.Backends
{
    // ---------------------------------------------------------------------------------------------------------------------
    // NumPy's array coercion of C# values that hold text: np.array(list_of_str) / np.array('abc') - the shape walk
    // (PyArray_DiscoverDTypeAndShape's update_shape rule over a rectangular or jagged C# array), the result descriptor's
    // discovery, and the store of every element into a fixed-width string, StringDType or numeric result - and the request
    // test np.asarray makes for a text dtype. The array object's own members (the np-layer dtype plan's D1), moved from
    // the text backend's NDStringArrays.
    //
    // Discovery. Without a dtype, an element set holding any string (or char) is
    // U, one of only byte[] is S; the width is the longest element (Python len(): code
    // points of a string, bytes of a byte[]), and a width of 0 (only empty values, or no values) becomes one
    // character — NumPy's array allocation rule. A string dtype may be sized (used as given: values truncate) or
    // unsized ("U", "S", np.str_: the width is discovered for that class, so
    // np.array(["ab"], dtype: "S") is S2 and a non-ASCII value raises NumPy's
    // UnicodeEncodeError).
    //
    // Values. null is Python's None: with a string dtype it stores "None" (NumPy's
    // str(None)); without one NumPy would build an object array, which NumSharp has not, so it is refused.
    // A number among text values stores as its str(). Without a dtype it promotes with the text by its own
    // descriptor (np.array(['x', 1]) is <U21: a Python int is int64, whose text width is 21; a Python
    // float 32, a complex 64, a bool 5); with an unsized string dtype it contributes the length of its str()
    // (np.array([1, 22], dtype='U') is <U2).
    //
    // A numeric dtype over text values (np.array(['12', '3'], dtype='i8')) packs every element with the
    // dtype's setitem (ReductionInitial.TryPack, NumPy's PyArray_Pack): a text is parsed by
    // CPython's int() / float() / complex() and its errors quote the Python object
    // (invalid literal for int() with base 10: '3.5'), a number is packed as itself (range-checked).
    //
    // The plan's phase P6 moved the ndmin padding it called (np.AtLeastNdView) to the array object
    // (NDArray.AtLeastNdView) and the file from beside its facades (Creation/) to Backends/Unmanaged/.
    // ---------------------------------------------------------------------------------------------------------------------
    public unsafe partial class UnmanagedStorage
    {
        /// <summary>
        ///     Whether <c>np.array</c> should take the string route for <paramref name="array"/>: its elements are text
        ///     (<see cref="string"/>), the requested dtype is a string dtype (over an object array that holds text or no
        ///     ndarray item: an object list of ndarrays is the coercion of a sequence's, which casts each array item), or an
        ///     <c>object</c> array holds text at any
        ///     nesting depth (a C# spelling of a Python list of lists: <c>new object[] { new[] { "a", "b" }, new[] { "c" } }</c>)
        ///     — or holds a <c>null</c> (Python's <c>None</c>) under a requested numeric dtype: that list too is built element
        ///     by element with the dtype's setitem (<see cref="Build"/>: <c>np.array([None, 1.0], dtype=float)</c> is
        ///     <c>[nan, 1.]</c>, an integer dtype raises <c>int(None)</c>'s TypeError at the first <c>None</c> in C order).
        /// </summary>
        /// <param name="array">The managed input (any rank, rectangular or jagged).</param>
        /// <param name="dtype">The requested dtype, or null.</param>
        /// <returns>True for the string route (the element-by-element builder).</returns>
        internal static bool IsTextInput(Array array, DType dtype)
        {
            var elementType = array.GetType().GetElementType();
            while (elementType is { IsArray: true } && elementType != typeof(byte[]))
                elementType = elementType.GetElementType();
            if (elementType == typeof(string))
                return true;
            // A string dtype (StringDType included) takes the Python objects of an object array (each stored by the dtype's
            // setitem) and the bytes values of a byte[][]; a typed numeric array is an ndarray, which np.array casts. So is
            // an object list of ndarrays without text: NumPy casts each array item into the class (the coercion of a
            // sequence does: [int32 rows] under 'U' is <U11), which the element builder cannot do.
            if (NDStringStorage.IsString(dtype) || dtype?.Meta is VStringDTypeMeta)
                return elementType == typeof(byte[])
                       || (elementType == typeof(object) && (ContainsText(array) || !ContainsArrayItem(array)));
            // An object list with None under a requested dtype: NumPy packs each element with the dtype's setitem, which the
            // typed conversion of the numeric route cannot do for a null (it threw ArgumentOutOfRangeException). Without a
            // dtype, None makes NumPy's object array, which the numeric route refuses as before.
            return elementType == typeof(object) && (ContainsText(array) || (dtype is not null && ContainsNull(array)));
        }

        /// <summary>
        ///     Whether an <c>object</c> array holds an ndarray item - an <see cref="NDArray"/>, a typed C# array of a NumSharp
        ///     dtype or a <c>Memory&lt;T&gt;</c> of one (<see cref="IsArrayLike"/>) - directly or inside a nested object array.
        ///     Stops at the first one.
        /// </summary>
        /// <param name="array">The array to search.</param>
        /// <returns>True when an ndarray item is found.</returns>
        private static bool ContainsArrayItem(Array array)
        {
            foreach (var item in array)
            {
                switch (item)
                {
                    case NDArray _:
                        return true;
                    case string _:
                    case byte[] _:
                        // A text or bytes value is one element, never an array item or a level to search.
                        break;
                    case Array child:
                        if (IsArrayLike(child) || ContainsArrayItem(child))
                            return true;
                        break;
                    default:
                        if (IsArrayLike(item))
                            return true;
                        break;
                }
            }
            return false;
        }

        /// <summary>
        ///     Whether an <c>object</c> array holds a <c>null</c> (Python's <c>None</c>) directly or inside a nested array, the
        ///     way <c>Collect</c> will walk it. Stops at the first one.
        /// </summary>
        /// <param name="array">The array to search.</param>
        /// <returns>True when a null element is found.</returns>
        private static bool ContainsNull(Array array)
        {
            foreach (var item in array)
            {
                switch (item)
                {
                    case null:
                        return true;
                    case string _:
                    case byte[] _:
                        // A text or bytes value is one element, never a level to search.
                        break;
                    case Array child:
                        if (ContainsNull(child))
                            return true;
                        break;
                }
            }
            return false;
        }

        /// <summary>
        ///     Whether an <c>object</c> array holds a text value — a <see cref="string"/> or a <c>byte[]</c> (one bytes
        ///     value) — directly or inside a nested array, the way <c>CollectValues</c> will walk it. Stops at the first one.
        /// </summary>
        /// <param name="array">The array to search.</param>
        /// <returns>True when a text value is found.</returns>
        private static bool ContainsText(Array array)
        {
            foreach (var item in array)
            {
                switch (item)
                {
                    case string _:
                    case byte[] _:
                        return true;
                    case Array child:
                        // A nested typed text array answers at once; a nested object array is searched in turn.
                        var childType = child.GetType().GetElementType();
                        while (childType is { IsArray: true } && childType != typeof(byte[]))
                            childType = childType.GetElementType();
                        if (childType == typeof(string) || ContainsText(child))
                            return true;
                        break;
                }
            }
            return false;
        }

        /// <summary>
        ///     <c>np.array(values, dtype)</c> for a managed array of text: the shape is the array's (a jagged array must be
        ///     rectangular), the dtype is discovered or validated, every element is encoded in C order.
        /// </summary>
        /// <param name="array">A rectangular (<c>string[,]</c>) or jagged (<c>string[][]</c>) array of text; with a string
        /// dtype also an <c>object</c> array or a jagged <c>byte[][]</c> (each innermost <c>byte[]</c> is one bytes value).</param>
        /// <param name="dtype">A string dtype (sized or not), or null to discover.</param>
        /// <param name="ndmin">Minimum number of dimensions: leading 1s are prepended as a VIEW of the built array (NumPy's result does not own its data).</param>
        /// <param name="order">The memory layout: <c>'C'</c>, <c>'F'</c>, or <c>'A'</c>/<c>'K'</c> (C for a fresh array).</param>
        /// <returns>The new array.</returns>
        /// <exception cref="ValueError">A ragged jagged array (NumPy's inhomogeneous-shape text).</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        /// <exception cref="NotSupportedException">A numeric dtype, a number among the values, or <c>None</c> without a dtype.</exception>
        internal static NDArray FromManagedText(Array array, DType dtype, int ndmin, char order)
        {
            if (array is null)
                throw new ArgumentNullException(nameof(array));

            long[] dims;
            IReadOnlyList<object> values;
            if (array is string[] strings)
            {
                // A flat string[] (the common input) is its own element list: no per-item walk through the non-generic
                // enumerator and no copy into a List — the array, read through its object view (array covariance; never
                // written).
                dims = new[] { strings.LongLength };
                values = strings;
            }
            else
            {
                var found = new List<long>();
                var collected = new List<object>();
                CollectValues(array, found, collected);
                dims = found.ToArray();
                values = collected;
            }
            var built = Build(dims, values, dtype, order);
            if (ndmin <= dims.Length)
                return built;
            // NumPy builds the array at its discovered rank and prepends the ones as a VIEW of it (OWNDATA False, the
            // prepended axes' strides C-like), so the result is a view here too; the built array's own reference goes.
            var view = NDArray.AtLeastNdView(built, ndmin);
            built.Dispose();
            return view;
        }

        /// <summary>
        ///     Whether an array already satisfies a dtype REQUEST (<c>np.asarray(a, dtype)</c> returns it uncopied): for the
        ///     numeric lanes the lane must match; for a string request the array must be a string array of that class whose
        ///     width is the requested one — an unsized request (<c>'U'</c>, <c>np.str_</c>) adapts to the array first, so a
        ///     <c>&lt;U4</c> array satisfies <c>'U'</c> but not <c>'U2'</c> or <c>'S'</c>.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="requested">The requested dtype (not null).</param>
        /// <returns>True when no conversion is needed.</returns>
        internal static bool SatisfiesRequest(NDArray a, DType requested)
        {
            if (a.typecode != NPTypeCode.String && !(requested.Meta is StringDTypeMeta))
                return a.typecode == requested.GetTypeCode();
            if (a.typecode != NPTypeCode.String || !(requested.Meta is StringDTypeMeta) || !requested.isnative)
                return false;
            var adapted = requested.itemsize == 0 ? DTypePromotion.CastDescrToDType(a.dtype, requested.Meta) : requested;
            return Strings.NDStringArrays.SameElement(a, adapted);
        }

        /// <summary>
        ///     <c>np.array(values, dtype)</c> for values already collected in C order with known dimensions — the store
        ///     <c>np.loadtxt</c> makes for a fixed-width string dtype after it read every field: a sized class keeps its width
        ///     (each value truncated to it), an unsized one takes the widest value's.
        /// </summary>
        /// <param name="dims">The result's dimensions (their product is <c>values.Count</c>).</param>
        /// <param name="values">The element values in C order: a <see cref="string"/>, a <c>byte[]</c> (one <c>S</c> value), a
        ///     number (stored as its text), or null (<c>None</c>).</param>
        /// <param name="dtype">The string dtype.</param>
        /// <returns>The new C-ordered array, which owns its data.</returns>
        /// <exception cref="TypeError">The discovered width exceeds the class's maximum.</exception>
        internal static NDArray FromValues(long[] dims, IReadOnlyList<object> values, DType dtype) => Build(dims, values, dtype, 'C');

        /// <summary>
        ///     <see cref="FromValues(long[], IReadOnlyList{object}, DType)"/> with NumPy's <c>order</c> for an array built from
        ///     Python objects (<c>np.array(list, dtype, order=order)</c>): the values are written in C order and an <c>'F'</c>
        ///     request lays a multi-dimensional result out in column-major order.
        /// </summary>
        /// <param name="dims">The result dimensions (the values fill them in C order).</param>
        /// <param name="values">The elements, one per position, as Python objects (text, bytes, numbers, None).</param>
        /// <param name="dtype">A string dtype (sized: values truncate; unsized: the width is discovered), or null.</param>
        /// <param name="order"><c>'F'</c> / <c>'f'</c> for a column-major result; any other (already validated) order is C.</param>
        /// <returns>The new array.</returns>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        /// <exception cref="NotSupportedException">A value that has no text form NumPy would give it (an object).</exception>
        internal static NDArray FromValues(long[] dims, IReadOnlyList<object> values, DType dtype, char order) => Build(dims, values, dtype, order);

        /// <summary>
        ///     NumPy's <c>PyArray_FromAny</c> over values already collected in C order (<c>PyArray_DiscoverDTypeAndShape</c>,
        ///     then <c>PyArray_AssignFromCache</c>): the descriptor is the requested one unless it is unsized
        ///     (<c>PyDataType_ISUNSIZED</c>: an unsized string class, or no request), when <see cref="Discover"/> finds the
        ///     string class and the widest value; the array is allocated with it, and every value is stored by the class's
        ///     <c>setitem</c> (<see cref="DTypeMeta.setitem"/>, NumPy's <c>PyArray_Pack</c>): a numeric class packs a Python
        ///     value with NumPy's checks (a str parsed by CPython's <c>int()</c> / <c>float()</c> / <c>complex()</c>, None as
        ///     NaN, <c>nan+nanj</c> or False), a fixed-width string class encodes its text (truncated, None as <c>'None'</c>,
        ///     a number as its <c>str()</c>), <c>StringDType</c> packs it through the buffer's allocator (its NA object as an
        ///     NA cell, a non-text value refused without coercion). The stores run as the class's
        ///     <see cref="DTypeMeta.SetFreshItems"/> over the new elements: <c>setitem</c> per value, <c>StringDType</c>'s
        ///     in one run that reserves its arena once (the same cells and arena bytes).
        /// </summary>
        /// <param name="dims">The result dimensions.</param>
        /// <param name="values">The elements, C order, <c>product(dims)</c> of them; null is Python's None.</param>
        /// <param name="dtype">The requested dtype or null.</param>
        /// <param name="order">The requested memory layout: <c>'F'</c> / <c>'f'</c> lays a result of two or more
        ///     dimensions out in column-major order (a copy of the C-ordered fill); any other (already validated) order is
        ///     C.</param>
        /// <returns>The new array, which owns its data.</returns>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>, or a text with a lone surrogate into a
        ///     <c>StringDType</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>, or bytes that are not UTF-8 into a
        ///     <c>StringDType</c>.</exception>
        /// <exception cref="ValueError">A text that is not a number of a numeric class's kind (quoting it), a NaN into an
        ///     integer class, or a non-text value into a <c>coerce=False</c> <c>StringDType</c>.</exception>
        /// <exception cref="OverflowException">A value that does not fit an integer class.</exception>
        /// <exception cref="TypeError">None into an integer or decimal class, a value NumPy cannot convert, or a discovered
        ///     width past the string class's maximum.</exception>
        /// <exception cref="NotSupportedException">None without a dtype (NumPy's object array), or a value with no text
        ///     form.</exception>
        private static NDArray Build(long[] dims, IReadOnlyList<object> values, DType dtype, char order)
        {
            // PyDataType_ISUNSIZED: no request, or an unsized string class, discovers the descriptor from the values; any
            // other request (a numeric class, a sized string class, StringDType) is the descriptor.
            var descr = dtype is null || dtype.itemsize == 0 ? Discover(values, dtype) : dtype;
            // NPY_NEEDS_INIT: a cell that holds a reference must be valid (empty) before setitem replaces it; every other
            // element's bytes are all written by its store, so the buffer is not zeroed first.
            var result = descr.Meta.HasReferences ? new NDArray(descr, new Shape(dims), true) : NDArray.ForOverwrite(descr, new Shape(dims));
            try
            {
                // The ALLOCATED descriptor stores (the allocation gives an unsized string class one character, and it is the
                // native form of a byte-swapped request), and the buffer goes along (StringDType's setitem finds the
                // allocator its cell belongs to there). PyArray_AssignFromCache's loop of PyArray_Pack is the class's run
                // over the new elements: setitem per value, StringDType's in one run that reserves its arena once.
                var stored = result.dtype;
                stored.Meta.SetFreshItems(stored, values, (byte*)result.Storage.Address, result.Storage.InternalArray);
                // The stores wrote through a raw pointer: the array must stay reachable until they are done.
                GC.KeepAlive(result);
                if (dims.Length > 1 && (order == 'F' || order == 'f'))
                {
                    var f = result.copy('F');
                    result.Dispose();
                    return f;
                }
                return result;
            }
            catch
            {
                // A refused value leaves a partly filled array: it is never handed out.
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     The dtype half of NumPy's <c>PyArray_DiscoverDTypeAndShape</c> over values already collected in C order: each
        ///     value's class (<see cref="DiscoverDTypeFromPyObject"/>, NumPy's <c>discover_dtype_from_pyobject</c>), its
        ///     descriptor (<see cref="FindScalarDescriptor"/>: the class's <c>discover_descr_from_pyobject</c>, cast to the
        ///     requested class), and the promotion of every value's descriptor in order (<c>handle_promotion</c>,
        ///     <see cref="DTypePromotion.PromoteTypes(NumSharp.DType, NumSharp.DType)"/>).
        /// </summary>
        /// <param name="values">The element values.</param>
        /// <param name="dtype">The requested dtype: an unsized one (an unsized string class) is discovered under, a sized
        ///     one is returned as it is; or null.</param>
        /// <returns>The descriptor to allocate: an unsized string class's singleton when every text is empty (the
        ///     allocation gives it one character), the requested class's singleton (or <c>U</c>'s) for no values.</returns>
        /// <exception cref="NotSupportedException">None without a dtype (NumPy's object array), a value of no NumPy type
        ///     without one, or a Python int past 64 bits without one.</exception>
        /// <exception cref="TypeError">A text too long for its string class (<c>string too large to store inside
        ///     array.</c>).</exception>
        /// <exception cref="ValueError">A Python int of more than 4300 decimal digits under a requested string class.</exception>
        /// <remarks>
        ///     <para>
        ///     What the slots give: a str holds its code points, a C# char one character, bytes their length; under a
        ///     requested string class a Python builtin (str, bytes, int, float, complex, bool: the string classes'
        ///     <c>is_known_scalar_type</c>) the length of its <c>str()</c>, and None <c>'None'</c>; any other value
        ///     its own descriptor, which the cast to the string class sizes to its <c>REQUIRED_STR_LEN</c> (a Python int is
        ///     int64, 21, or uint64, 20, by its value; a float or a NumPy float16 / decimal 32, a complex 64, a bool 5). The
        ///     promotion of a text with a number keeps the string class and the longer width; <c>U</c> absorbs <c>S</c>.
        ///     </para>
        ///     <para>
        ///     The callers pass values that hold text, bytes or a requested string class, so the result is a string
        ///     descriptor; values of numbers alone would be discovered as NumPy discovers them (<c>[1, 2]</c> is int64).
        ///     </para>
        /// </remarks>
        private static DType Discover(IReadOnlyList<object> values, DType dtype)
        {
            DTypeMeta fixedDType = null;
            if (dtype != null)
            {
                // A sized request is the descriptor; an unsized one names the class the values are discovered under.
                if (dtype.itemsize > 0)
                    return dtype;
                fixedDType = dtype.Meta;
            }
            else
            {
                // NumSharp's refusal, before any value is discovered: None without a request makes NumPy's array an object
                // one, and a string class request is the way to store it as text.
                foreach (var v in values)
                {
                    if (v is null)
                        throw new NotSupportedException(
                            "A null (None) among the values would make NumPy build an object array, which NumSharp does not have; " +
                            "pass a string dtype (np.str_ / \"U\") to store it as the text 'None', or replace it.");
                }
            }

            // handle_promotion: the first value's descriptor, then PyArray_PromoteTypes with each later one, in order.
            DType descr = null;
            foreach (var v in values)
            {
                var discovered = DiscoverDTypeFromPyObject(v, fixedDType);
                var d = FindScalarDescriptor(fixedDType, discovered, v);
                descr = descr is null ? d : DTypePromotion.PromoteTypes(descr, d);
            }

            // No values: the requested class's unsized singleton, or NumPy's default U.
            if (descr is null)
                return ((StringDTypeMeta)(fixedDType ?? DTypeRegistry.Str)).singleton;
            // Every text empty: the class singleton (an unsized descriptor, which the allocation gives one character).
            if (descr.Meta is StringDTypeMeta cls && descr.itemsize == 0)
                return cls.singleton;
            return descr;
        }

        /// <summary>
        ///     The descriptor an unsized string class takes for its widest element: <paramref name="cls"/>'s class of
        ///     <paramref name="width"/> characters, or the class singleton for a width of zero. It is <c>np.loadtxt</c>'s
        ///     staged unsized read, which measures the widths without the objects; the coercion of objects reaches the same
        ///     descriptor through the classes' discovery (<see cref="Discover"/>).
        /// </summary>
        /// <param name="cls">The string class (U or S).</param>
        /// <param name="width">The widest element's length: code points for U, bytes for S.</param>
        /// <returns>The descriptor.</returns>
        /// <exception cref="TypeError">A width past the class's maximum (<c>string too large to store inside
        ///     array.</c>).</exception>
        internal static DType UnsizedTextDescr(StringDTypeMeta cls, long width)
        {
            if (width > cls.MaxChars)
                throw new TypeError("string too large to store inside array.");
            return width == 0 ? cls.singleton : cls.Descr(width);
        }

        /// <summary>
        ///     Flattens a managed array of any rank into C-ordered element values and records its shape: a rectangular
        ///     multi-dimensional level contributes all its axes, a jagged level one axis per nesting, and the walk stops at an
        ///     element that is text — a <see cref="string"/>, or a <c>byte[]</c> (one bytes value).
        /// </summary>
        /// <param name="array">The managed array.</param>
        /// <param name="dims">Receives the dimensions.</param>
        /// <param name="values">Receives the elements in C order.</param>
        /// <exception cref="ValueError">A ragged jagged level (NumPy's inhomogeneous-shape text).</exception>
        internal static void CollectValues(Array array, List<long> dims, List<object> values)
        {
            int leafDepth = -1;
            Visit(array, 0, dims, values, ref leafDepth);
        }

        /// <summary>
        ///     One level of <see cref="CollectValues"/> (NumPy's <c>update_shape</c> rule): a level's axes are appended the first
        ///     time that depth is reached and must match on every later visit; every leaf value must sit at the same depth.
        /// </summary>
        /// <param name="array">The current level (rank r contributes r axes).</param>
        /// <param name="depth">The axis index this level starts at.</param>
        /// <param name="dims">The dimensions found so far.</param>
        /// <param name="values">The elements so far (C order).</param>
        /// <param name="leafDepth">The depth of the first leaf seen (-1 before any), which every later leaf must match.</param>
        /// <exception cref="ValueError">A level or a leaf disagrees with what an earlier sibling established.</exception>
        private static void Visit(Array array, int depth, List<long> dims, List<object> values, ref int leafDepth)
        {
            int rank = array.Rank;
            for (int r = 0; r < rank; r++)
            {
                long extent = array.GetLongLength(r);
                int axis = depth + r;
                if (axis == dims.Count)
                {
                    // A new axis — unless leaves were already found shallower than this: then this branch is deeper.
                    if (leafDepth >= 0 && axis >= leafDepth)
                        throw Inhomogeneous(dims, leafDepth);
                    dims.Add(extent);
                }
                else if (dims[axis] != extent)
                {
                    throw Inhomogeneous(dims, axis);
                }
            }

            int childDepth = depth + rank;
            foreach (var item in array)
            {
                // A nested array is one more level — except a byte[], which is ONE bytes value (Python's `bytes`).
                if (item is Array child && item is not byte[])
                {
                    Visit(child, childDepth, dims, values, ref leafDepth);
                    continue;
                }

                if (leafDepth < 0)
                {
                    // The first leaf: a sibling that already went deeper makes the input ragged.
                    if (childDepth != dims.Count)
                        throw Inhomogeneous(dims, childDepth);
                    leafDepth = childDepth;
                }
                else if (leafDepth != childDepth)
                {
                    throw Inhomogeneous(dims, Math.Min(leafDepth, childDepth));
                }
                values.Add(item);
            }
        }

        /// <summary>NumPy's ragged-sequence error: the shape agreed so far plus "inhomogeneous part".</summary>
        /// <param name="dims">The dimensions recorded.</param>
        /// <param name="agreedAxes">How many leading axes every element agreed on.</param>
        /// <returns>The exception (not thrown).</returns>
        private static ValueError Inhomogeneous(List<long> dims, int agreedAxes)
        {
            int n = Math.Min(agreedAxes, dims.Count);
            string shape = n == 1 ? $"({dims[0]},)" : "(" + string.Join(", ", dims.Take(n)) + ")";
            return new ValueError(
                $"setting an array element with a sequence. The requested array has an inhomogeneous shape after {n} dimensions. " +
                $"The detected shape was {shape} + inhomogeneous part.");
        }
    }
}
