using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The C# → string-array boundary: building <c>S</c>/<c>U</c> arrays from managed values — NumPy's
    ///     <c>np.array(list_of_str)</c> / <c>np.array('abc')</c> path (<c>PyArray_DiscoverDTypeAndShape</c> for the shape
    ///     and the length, then <c>setitem</c> per element), restricted to text values.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Discovery.</b> Without a dtype, an element set holding any <see cref="string"/> (or <see cref="char"/>) is
    ///     <c>U</c>, one of only <c>byte[]</c> is <c>S</c>; the width is the longest element (Python <c>len()</c>: code
    ///     points of a string, bytes of a <c>byte[]</c>), and a width of 0 (only empty values, or no values) becomes one
    ///     character — NumPy's array allocation rule. A string dtype may be sized (used as given: values truncate) or
    ///     unsized (<c>"U"</c>, <c>"S"</c>, <c>np.str_</c>: the width is discovered for that class, so
    ///     <c>np.array(["ab"], dtype: "S")</c> is <c>S2</c> and a non-ASCII value raises NumPy's
    ///     <c>UnicodeEncodeError</c>).
    ///     </para>
    ///     <para>
    ///     <b>Values.</b> <see langword="null"/> is Python's <c>None</c>: with a string dtype it stores <c>"None"</c> (NumPy's
    ///     <c>str(None)</c>); without one NumPy would build an <c>object</c> array, which NumSharp has not, so it is refused.
    ///     A number among text values stores as its <c>str()</c>. Without a dtype it promotes with the text by its own
    ///     descriptor (<c>np.array(['x', 1])</c> is <c>&lt;U21</c>: a Python int is int64, whose text width is 21; a Python
    ///     float 32, a complex 64, a bool 5); with an unsized string dtype it contributes the length of its <c>str()</c>
    ///     (<c>np.array([1, 22], dtype='U')</c> is <c>&lt;U2</c>).
    ///     </para>
    ///     <para>
    ///     <b>A numeric dtype</b> over text values (<c>np.array(['12', '3'], dtype='i8')</c>) packs every element with the
    ///     dtype's setitem (<see cref="ReductionInitial.TryPack"/>, NumPy's <c>PyArray_Pack</c>): a text is parsed by
    ///     CPython's <c>int()</c> / <c>float()</c> / <c>complex()</c> and its errors quote the Python object
    ///     (<c>invalid literal for int() with base 10: '3.5'</c>), a number is packed as itself (range-checked).
    ///     </para>
    /// </remarks>
    internal static unsafe class NDStringArrays
    {
        /// <summary>
        ///     Whether <c>np.array</c> should take the string route for <paramref name="array"/>: its elements are text
        ///     (<see cref="string"/>), the requested dtype is a string dtype, or an <c>object</c> array holds text at any
        ///     nesting depth (a C# spelling of a Python list of lists: <c>new object[] { new[] { "a", "b" }, new[] { "c" } }</c>).
        /// </summary>
        /// <param name="array">The managed input (any rank, rectangular or jagged).</param>
        /// <param name="dtype">The requested dtype, or null.</param>
        /// <returns>True for the string route.</returns>
        public static bool IsTextInput(Array array, DType dtype)
        {
            var elementType = array.GetType().GetElementType();
            while (elementType is { IsArray: true } && elementType != typeof(byte[]))
                elementType = elementType.GetElementType();
            if (elementType == typeof(string))
                return true;
            if (NDStringStorage.IsString(dtype))
                return elementType == typeof(object) || elementType == typeof(byte[]);
            return elementType == typeof(object) && ContainsText(array);
        }

        /// <summary>
        ///     Whether an <c>object</c> array holds a text value — a <see cref="string"/> or a <c>byte[]</c> (one bytes
        ///     value) — directly or inside a nested array, the way <c>Collect</c> will walk it. Stops at the first one.
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
        public static NDArray FromManaged(Array array, DType dtype, int ndmin, char order)
        {
            if (array is null)
                throw new ArgumentNullException(nameof(array));

            var dims = new List<long>();
            var values = new List<object>();
            Collect(array, dims, values);
            var built = Build(dims.ToArray(), values, dtype, order);
            if (ndmin <= dims.Count)
                return built;
            // NumPy builds the array at its discovered rank and prepends the ones as a VIEW of it (OWNDATA False, the
            // prepended axes' strides C-like), so the result is a view here too; the built array's own reference goes.
            var view = np.AtLeastNdView(built, ndmin);
            built.Dispose();
            return view;
        }

        /// <summary>
        ///     The explicit <c>(Array)</c> conversion of a fixed-width string array: a C# array of the array's shape whose
        ///     elements are the decoded values — <see cref="string"/> for <c>U</c>, <c>byte[]</c> for <c>S</c>, trailing NULs
        ///     stripped as NumPy's element access strips them (<c>arr.tolist()</c> with C# arrays instead of nested lists).
        ///     A 0-d array becomes a one-element vector, as <c>ToMuliDimArray</c> does for the numeric dtypes.
        /// </summary>
        /// <param name="a">A string array of any layout (read in logical C order).</param>
        /// <returns><c>string[]</c>, <c>string[,]</c>, ... or <c>byte[][]</c>, <c>byte[][,]</c>, ...</returns>
        /// <exception cref="SystemError">A <c>U</c> unit above <c>U+10FFFF</c> (the decoder's refusal, as NumPy's).</exception>
        /// <exception cref="InvalidOperationException">A dimension longer than a C# array can hold.</exception>
        public static Array ToManagedArray(NDArray a)
        {
            var elementType = a.Storage.StringSlice.IsUnicode ? typeof(string) : typeof(byte[]);
            var dims = a.Shape.dimensions;
            if (dims.Length == 0)
            {
                var one = Array.CreateInstance(elementType, 1);
                one.SetValue(a.Storage.GetAtIndex(0), 0);
                return one;
            }

            var lengths = new int[dims.Length];
            for (int d = 0; d < dims.Length; d++)
                lengths[d] = dims[d] <= int.MaxValue
                    ? (int)dims[d]
                    : throw new InvalidOperationException($"Dimension {d} has {dims[d]} elements, more than a C# array holds.");
            var result = Array.CreateInstance(elementType, lengths);
            long n = a.size;
            if (n == 0)
                return result;

            // A vector fills by flat index; higher ranks walk a C-order odometer in lockstep with the flat index, so each
            // element is decoded once and stored at its coordinates (GetAtIndex resolves strides, offset and broadcast).
            if (dims.Length == 1)
            {
                for (int i = 0; i < lengths[0]; i++)
                    result.SetValue(a.Storage.GetAtIndex(i), i);
                return result;
            }
            var coords = new int[dims.Length];
            for (long flat = 0; flat < n; flat++)
            {
                result.SetValue(a.Storage.GetAtIndex(flat), coords);
                for (int d = coords.Length - 1; d >= 0; d--)
                {
                    if (++coords[d] < lengths[d])
                        break;
                    coords[d] = 0;
                }
            }
            return result;
        }

        /// <summary>
        ///     The descriptor of the NumPy SCALAR an element read back is — what NumPy's iteration over a 1-D string array
        ///     yields (<c>for pv in p</c>), so the descriptor its errors name: a <c>str</c> is <c>&lt;U{len}</c> and a
        ///     <c>bytes</c> <c>S{len}</c>, at least one character (<c>np.str_('')</c> becomes <c>&lt;U1</c>, NumPy's allocation
        ///     rule), whatever width the array it came from has.
        /// </summary>
        /// <param name="value">An element value: a <see cref="string"/> (<c>U</c>) or a <c>byte[]</c> (<c>S</c>).</param>
        /// <returns>The scalar's descriptor.</returns>
        /// <exception cref="ArgumentException"><paramref name="value"/> is neither text nor bytes.</exception>
        public static DType ScalarDescr(object value) => value switch
        {
            string s => DTypeRegistry.Str.Descr(Math.Max(1, NDStringCodec.CodePointCount(s))),
            byte[] b => DTypeRegistry.Bytes.Descr(Math.Max(1, b.Length)),
            _ => throw new ArgumentException($"A string element is a string or a byte[], not {value?.GetType().Name ?? "null"}.", nameof(value)),
        };

        /// <summary>
        ///     <c>np.array(value, dtype)</c> for ONE text value — a 0-d array (<c>np.array('abc')</c> is <c>&lt;U3</c> with
        ///     shape <c>()</c>).
        /// </summary>
        /// <param name="value">A <see cref="string"/>, <c>byte[]</c>, <see cref="char"/> or null (None).</param>
        /// <param name="dtype">A string dtype, or null to discover.</param>
        /// <returns>The 0-d array.</returns>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        /// <exception cref="NotSupportedException">A numeric dtype or a non-text value.</exception>
        public static NDArray FromScalar(object value, DType dtype)
            => Build(Array.Empty<long>(), new List<object>(1) { value }, dtype, 'C');

        /// <summary>
        ///     <c>np.full(shape, fill_value, dtype)</c> / <c>np.full_like</c> for a string result — NumPy's
        ///     <c>a = empty(shape, dtype); copyto(a, fill_value, casting='unsafe')</c>: the result dtype is the requested one
        ///     (an unsized request is ONE character, NumPy's allocation rule) or the fill value's own; the value is encoded
        ///     once into a 0-d array and broadcast by the string copy route, which truncates (<c>'abc'</c> into <c>S1</c> is
        ///     <c>b'a'</c>) and converts <c>S ↔ U</c> with the ASCII checks.
        /// </summary>
        /// <param name="shape">The result dimensions and layout: a fresh dense layout (C, F or <c>np.full_like</c>'s KEEPORDER
        ///     permutation, <see cref="Shape.NewLikeLayout"/>) is kept, a view's layout is re-laid out in C order.</param>
        /// <param name="fillValue">A <see cref="string"/>, <c>byte[]</c>, <see cref="char"/>, null (None) or a 0-d string array.</param>
        /// <param name="dtype">The result dtype (a string dtype), or null to take the fill value's.</param>
        /// <returns>The filled array.</returns>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        /// <exception cref="StringComputationNotSupportedException">A numeric fill value (number to text is the computational
        ///     phase).</exception>
        public static NDArray Full(Shape shape, object fillValue, DType dtype)
        {
            using var value = fillValue is NDArray nd
                ? nd.copy()
                : FromScalar(fillValue, dtype?.Meta is StringDTypeMeta cls ? cls.Singleton : null);
            var descr = dtype ?? value.dtype;
            // The shape's own layout is kept when it describes a fresh buffer (np.full_like's KEEPORDER / F layouts); the
            // allocator re-lays a view's layout out in C order (Shape.ForFreshAllocation).
            var layout = shape.IsEmpty ? Shape.Adopt(Array.Empty<long>()) : shape;
            // The broadcast copy writes every element, so a U result skips the NEEDS_INIT zero fill; a failing conversion
            // disposes it.
            var result = NDArray.ForOverwrite(descr, layout);
            try
            {
                Iteration.NDIter.Copy(result.Storage, value.Storage);
            }
            catch
            {
                result.Dispose();
                throw;
            }
            return result;
        }

        /// <summary>
        ///     Turns a FRESH unit array — the result of a numeric operation run on unit views (<c>uint8</c> for <c>S</c>,
        ///     <c>uint32</c> for <c>U</c>, last axis = the characters of one element) — into the string array it holds,
        ///     without moving a byte when it is C-contiguous (<see cref="UnmanagedStorage.StringsOverUnits"/>); another layout is
        ///     first copied to C order.
        /// </summary>
        /// <param name="units">The unit result (this call takes ownership and disposes it).</param>
        /// <param name="descr">The sized, native string descriptor of the result.</param>
        /// <returns>The string array (owning its data).</returns>
        /// <exception cref="InvalidOperationException">The last axis is not one element of <paramref name="descr"/>.</exception>
        public static NDArray FromUnitArray(NDArray units, DType descr)
        {
            try
            {
                if (units.Shape.IsContiguous)
                    return new NDArray(UnmanagedStorage.StringsOverUnits(units.Storage, descr)) { TensorEngine = units.TensorEngine };
                using var dense = units.copy();
                return new NDArray(UnmanagedStorage.StringsOverUnits(dense.Storage, descr)) { TensorEngine = units.TensorEngine };
            }
            finally
            {
                units.Dispose(); // the string array holds its own counted reference on the buffer
            }
        }

        /// <summary>
        ///     The unit view of a string array as an <see cref="NDArray"/> the caller disposes (shares the memory).
        /// </summary>
        /// <param name="strings">A string array.</param>
        /// <returns>A <c>uint8</c>/<c>uint32</c> array of shape <c>shape + (chars,)</c>.</returns>
        public static NDArray UnitArray(NDArray strings)
            => new NDArray(strings.Storage.StringUnitView()) { TensorEngine = strings.TensorEngine };

        /// <summary>
        ///     <c>np.where(condition, x, y)</c> when <paramref name="x"/> or <paramref name="y"/> is a string array — NumPy's
        ///     <c>PyArray_Where</c>: the result dtype is <c>result_type(x, y)</c> over DESCRIPTORS (the wider width; <c>S</c> with
        ///     <c>U</c> gives <c>U</c>), the result is allocated by NumPy's iterator in KEEPORDER (the operands' memory order,
        ///     <see cref="Shape.NpyIterOutputShape"/>), and each element is the converted <paramref name="x"/> or
        ///     <paramref name="y"/> element. The selection runs on unit views — the condition gains a trailing axis of extent
        ///     1 so it broadcasts over each element's characters.
        /// </summary>
        /// <param name="condition">The condition (bool, numbers, or text — each read by its truth value; a non-empty string
        ///     is True).</param>
        /// <param name="x">Values where the condition holds.</param>
        /// <param name="y">Values elsewhere.</param>
        /// <param name="dims">The broadcast dimensions of the three (the caller validated the broadcast); adopted.</param>
        /// <returns>The new string array.</returns>
        /// <exception cref="SystemError">A <c>U</c> condition holds a unit above U+10FFFF (NumPy's bool cast of it).</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> value NumPy converts to the <c>U</c> result is not ASCII — the
        ///     element NumPy's loop converts first (see <see cref="ThrowIfWhereDecodeFails"/>).</exception>
        /// <remarks>
        ///     NumPy's iterator casts the condition to bool and the values to the result dtype buffer by buffer
        ///     (<c>NPY_ITER_ZEROSIZE_OK</c>), so an EMPTY result converts nothing: it is an empty array of the result dtype
        ///     whatever the condition and the values hold (<c>np.where(np.zeros(0, bool), np.zeros(0), np.zeros(0, 'U2'))</c> is
        ///     an empty <c>&lt;U32</c> array).
        /// </remarks>
        public static NDArray Where(NDArray condition, NDArray x, NDArray y, long[] dims)
        {
            var descr = DTypePromotion.ResultType(x, y);

            // NumPy's iterator allocates the result in the order of its operands (cond, x, y): stride-0 / extent-1 axes
            // abstain, C wins a tie — a transposed or F operand set gives a transposed or F result.
            var operandShapes = new[] { condition.Shape, x.Shape, y.Shape };
            var layout = Shape.NpyIterOutputShape((long[])dims.Clone(), operandShapes);
            if (layout.size == 0)
                return new NDArray(descr, layout, true);

            // A text condition is read by its truth values: NumPy's iterator casts it to bool (a non-empty string is True; a
            // U unit above U+10FFFF raises the cast's SystemError).
            using var cond = condition.typecode == NPTypeCode.Boolean ? condition.view()
                : condition.typecode == NPTypeCode.String ? NDStringTruth.Truth(condition, NDStringTruth.Flavor.Cast)
                : condition.astype(NPTypeCode.Boolean);
            // A numeric x / y is cast to the result's string dtype by the iterator's buffers: each number written as its
            // scalar's str(), truncated to the result width. Formatting cannot fail, so converting the operand whole first
            // gives the same result (and an S -> U decode below still names NumPy's element).
            NDArray xText = null, yText = null;
            try
            {
                if (x.typecode != NPTypeCode.String)
                    x = xText = x.astype(descr, copy: true, order: 'K', casting: "unsafe");
                if (y.typecode != NPTypeCode.String)
                    y = yText = y.astype(descr, copy: true, order: 'K', casting: "unsafe");
                return WhereText(cond, x, y, descr, dims, layout);
            }
            finally
            {
                xText?.Dispose();
                yText?.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="Where"/> once every value operand is text: NumPy's S → U decode check, then the selection over unit
        ///     views (or the contiguous fast path).
        /// </summary>
        /// <param name="cond">The bool condition (already the truth of a text condition).</param>
        /// <param name="x">Text values where the condition holds.</param>
        /// <param name="y">Text values elsewhere.</param>
        /// <param name="descr">The result descriptor.</param>
        /// <param name="dims">The broadcast dimensions.</param>
        /// <param name="layout">The result layout (NumPy's iterator output layout).</param>
        /// <returns>The new string array.</returns>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> value the result converts to <c>U</c> is not ASCII.</exception>
        private static NDArray WhereText(NDArray cond, NDArray x, NDArray y, DType descr, long[] dims, Shape layout)
        {
            // S -> U decoding is the only conversion of a string where that can fail; NumPy's loop decides which element
            // it reports, so that is found first, and the conversions below skip the ASCII check they no longer need.
            bool decodes = descr.Meta == DTypeRegistry.Str && (!x.Storage.StringSlice.IsUnicode || !y.Storage.StringSlice.IsUnicode);
            if (decodes)
                ThrowIfWhereDecodeFails(cond, x, y, descr, dims, layout);

            NDArray xs = SameElement(x, descr) ? x : ConvertUnchecked(x, descr);
            NDArray ys = SameElement(y, descr) ? y : ConvertUnchecked(y, descr);
            try
            {
                // The common case selects whole elements in one pass; any broadcast or strided operand runs on the unit views.
                if (TryWhereContiguous(cond, xs, ys, descr, dims, layout, out var selected))
                    return selected;
                using var condUnits = np.expand_dims(cond, -1);
                using var xu = UnitArray(xs);
                using var yu = UnitArray(ys);
                var result = FromUnitArray(np.where(condUnits, xu, yu), descr);
                if (result.Shape.IsContiguous && layout.IsContiguous)
                    return result;
                // NumPy's layout: the values move into an array laid out like the iterator's output.
                using (result)
                {
                    var laid = NDArray.ForOverwrite(descr, layout); // the copy writes every element
                    Iteration.NDIter.Copy(laid.Storage, result.Storage);
                    return laid;
                }
            }
            finally
            {
                if (!ReferenceEquals(xs, x))
                    xs.Dispose();
                if (!ReferenceEquals(ys, y))
                    ys.Dispose();
            }
        }

        /// <summary>
        ///     <c>np.where</c> over string values that need no broadcast and no layout work: the condition and both values
        ///     have the result's dimensions, are C-contiguous, and the values already hold the result descriptor. Then each
        ///     condition byte selects one whole element — <see cref="DirectILKernelGenerator.GetWhereBytesKernel"/>, one pass
        ///     over the result — where the unit route broadcasts the condition over each element's characters (2x slower
        ///     than NumPy at 1M <c>&lt;U8</c> elements).
        /// </summary>
        /// <param name="cond">The bool condition.</param>
        /// <param name="x">The values where the condition holds, in <paramref name="descr"/>.</param>
        /// <param name="y">The values elsewhere, in <paramref name="descr"/>.</param>
        /// <param name="descr">The result descriptor (its width is the kernel's element width).</param>
        /// <param name="dims">The broadcast dimensions.</param>
        /// <param name="layout">NumPy's result layout for the operands (C order when they are all C-contiguous).</param>
        /// <param name="result">The new array when the fast path applied, else null.</param>
        /// <returns>True when <paramref name="result"/> was produced; false for a shape or layout the kernel does not take
        ///     (nothing allocated).</returns>
        private static unsafe bool TryWhereContiguous(NDArray cond, NDArray x, NDArray y, DType descr, long[] dims, Shape layout,
                                                      out NDArray result)
        {
            result = null;
            long n = 1;
            foreach (var d in dims)
                n *= d;
            if (n == 0 || !layout.IsContiguous)
                return false;
            foreach (var operand in new[] { cond, x, y })
                if (!operand.Shape.IsContiguous || !SameDimensions(operand.Shape.dimensions, dims))
                    return false;
            var kernel = DirectILKernelGenerator.GetWhereBytesKernel(descr.itemsize);
            if (kernel is null)
                return false;

            long width = descr.itemsize;
            var r = NDArray.ForOverwrite(descr, layout); // the kernel writes every element
            kernel((bool*)(cond.Storage.Address + cond.Shape.offset),
                   x.Storage.Address + x.Shape.offset * width,
                   y.Storage.Address + y.Shape.offset * width,
                   r.Storage.Address + r.Shape.offset * width,
                   n);
            result = r;
            return true;
        }

        /// <summary>True when two dimension lists are equal (rank and every extent).</summary>
        /// <param name="a">One list.</param>
        /// <param name="b">The other.</param>
        /// <returns>True for equal ranks and extents.</returns>
        private static bool SameDimensions(long[] a, long[] b)
        {
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }

        /// <summary>
        ///     A copy of <paramref name="a"/> converted to <paramref name="descr"/> without the ASCII check, for a caller that
        ///     already raised NumPy's error for every element it will read (non-ASCII bytes of an element nobody reads carry
        ///     through as garbage units and are discarded).
        /// </summary>
        /// <param name="a">A string array.</param>
        /// <param name="descr">The target string descriptor.</param>
        /// <returns>A fresh C-ordered array of <paramref name="a"/>'s dimensions.</returns>
        private static NDArray ConvertUnchecked(NDArray a, DType descr)
        {
            var r = NDArray.ForOverwrite(descr, Shape.Adopt((long[])a.Shape.dimensions.Clone())); // fully written below
            NDStringCopy.CopyUnchecked(r.Storage, a.Storage);
            return r;
        }

        /// <summary>
        ///     NumPy's error for the <c>S → U</c> decode of a mixed-class <c>np.where</c> (<c>PyArray_Where</c>), raised before
        ///     anything is written, naming the element NumPy's loop meets first. The loop walks the iteration order — the
        ///     memory order of the result <paramref name="layout"/> — and converts in one of two ways: when the result's item
        ///     size is 1, 2, 4, 8 or 16 bytes (the "trivial copy loop") the iterator casts <paramref name="x"/> and
        ///     <paramref name="y"/> into its buffers <c>np.getbufsize()</c> elements at a time, x's chunk before y's, EVERY
        ///     element; otherwise only the element the condition takes is converted, element by element.
        /// </summary>
        /// <param name="cond">The bool condition.</param>
        /// <param name="x">The x operand.</param>
        /// <param name="y">The y operand.</param>
        /// <param name="descr">The <c>U</c> result descriptor.</param>
        /// <param name="dims">The broadcast dimensions.</param>
        /// <param name="layout">The result layout (its memory order is the iteration order).</param>
        /// <exception cref="UnicodeDecodeError">The first failing element, with the position of its first non-ASCII byte.</exception>
        private static void ThrowIfWhereDecodeFails(NDArray cond, NDArray x, NDArray y, DType descr, long[] dims, Shape layout)
        {
            foreach (var d in dims)
                if (d == 0)
                    return;
            bool xBytes = !x.Storage.StringSlice.IsUnicode, yBytes = !y.Storage.StringSlice.IsUnicode;
            // The common case: no byte at or above 0x80 in either bytes operand — nothing can fail (one max reduction).
            if ((!xBytes || AllAscii(x)) && (!yBytes || AllAscii(y)))
                return;

            using var xBad = xBytes ? NonAsciiInLayout(x, dims, layout) : null;
            using var yBad = yBytes ? NonAsciiInLayout(y, dims, layout) : null;
            int itemsize = descr.itemsize;
            if (itemsize == 1 || itemsize == 2 || itemsize == 4 || itemsize == 8 || itemsize == 16)
            {
                long ix = FirstTrue(xBad), iy = FirstTrue(yBad), buf = np.getbufsize();
                if (ix >= 0 && (iy < 0 || ix / buf <= iy / buf))
                    throw DecodeErrorAt(x, dims, layout, ix);
                if (iy >= 0)
                    throw DecodeErrorAt(y, dims, layout, iy);
                return;
            }

            // Element by element: the taken element of a bytes operand is the only one converted.
            using var condInLayout = new NDArray(NPTypeCode.Boolean, Shape.Adopt((long[])layout.dimensions.Clone(), (long[])layout.strides.Clone()), false);
            using (var cb = np.broadcast_to(cond, Shape.Adopt((long[])dims.Clone())))
                Iteration.NDIter.Copy(condInLayout.Storage, cb.Storage);
            long first = FirstTakenBad(condInLayout, xBad, yBad);
            if (first >= 0)
                throw DecodeErrorAt(ByteAt(condInLayout, first) != 0 ? x : y, dims, layout, first);
        }

        /// <summary>
        ///     Whether every unit of a bytes array is below 0x80 (one max reduction over its unit view): no element can fail
        ///     NumPy's ASCII decode into <c>U</c>. Used by <c>np.where</c>'s decode check and by the set routines, whose
        ///     answers over empty operands do not depend on the bytes unless the decode can fail.
        /// </summary>
        /// <param name="a">A non-empty <c>S</c> array.</param>
        /// <returns>True when no element can fail an ASCII decode.</returns>
        internal static bool AllAscii(NDArray a)
        {
            using var units = UnitArray(a);
            using var max = np.amax(units);
            return Convert.ToInt32(max.GetAtIndex(0)) < 0x80;
        }

        /// <summary>
        ///     A bool array laid out like the result (<paramref name="layout"/>): true where the broadcast element of
        ///     <paramref name="a"/> holds a byte at or above 0x80.
        /// </summary>
        /// <param name="a">A bytes operand.</param>
        /// <param name="dims">The broadcast dimensions.</param>
        /// <param name="layout">The result layout.</param>
        /// <returns>The verdicts, owned by the caller.</returns>
        private static NDArray NonAsciiInLayout(NDArray a, long[] dims, Shape layout)
        {
            using var units = UnitArray(a);
            using var max = np.amax(units, -1);
            using var bad = max >= (byte)0x80;
            using var spread = np.broadcast_to(bad, Shape.Adopt((long[])dims.Clone()));
            var laid = new NDArray(NPTypeCode.Boolean, Shape.Adopt((long[])layout.dimensions.Clone(), (long[])layout.strides.Clone()), false);
            Iteration.NDIter.Copy(laid.Storage, spread.Storage);
            return laid;
        }

        /// <summary>The first true byte of a dense bool array in MEMORY order, or -1 (null counts as none).</summary>
        /// <param name="mask">A dense, offset-0 bool array, or null.</param>
        /// <returns>The memory index.</returns>
        private static long FirstTrue(NDArray mask)
        {
            if (mask is null)
                return -1;
            var span = new ReadOnlySpan<byte>(mask.Storage.Address, checked((int)mask.size));
            return span.IndexOfAnyExcept((byte)0);
        }

        /// <summary>The byte at a memory index of a dense bool array.</summary>
        /// <param name="mask">A dense, offset-0 bool array.</param>
        /// <param name="memoryIndex">The memory index.</param>
        /// <returns>0 or 1.</returns>
        private static byte ByteAt(NDArray mask, long memoryIndex) => mask.Storage.Address[memoryIndex];

        /// <summary>
        ///     The first memory index where the TAKEN operand's element fails: the condition picks <paramref name="xBad"/>'s
        ///     verdict, else <paramref name="yBad"/>'s (a null verdict array is a unicode operand, which never fails).
        /// </summary>
        /// <param name="cond">The condition laid out like the result.</param>
        /// <param name="xBad">x's verdicts in the same layout, or null.</param>
        /// <param name="yBad">y's verdicts in the same layout, or null.</param>
        /// <returns>The memory index, or -1.</returns>
        private static long FirstTakenBad(NDArray cond, NDArray xBad, NDArray yBad)
        {
            long n = cond.size;
            byte* c = cond.Storage.Address;
            byte* xb = xBad is null ? null : xBad.Storage.Address;
            byte* yb = yBad is null ? null : yBad.Storage.Address;
            // A scalar walk on the (rare) error path only: it runs after a reduction proved some element is not ASCII.
            for (long i = 0; i < n; i++)
            {
                byte* verdicts = c[i] != 0 ? xb : yb;
                if (verdicts != null && verdicts[i] != 0)
                    return i;
            }
            return -1;
        }

        /// <summary>
        ///     NumPy's <c>UnicodeDecodeError</c> for the element at <paramref name="memoryIndex"/> of the result layout: the
        ///     index is turned into coordinates through the layout's strides (outermost first) and the broadcast element of
        ///     <paramref name="a"/> there is decoded until its first non-ASCII byte.
        /// </summary>
        /// <param name="a">The bytes operand whose element fails.</param>
        /// <param name="dims">The broadcast dimensions.</param>
        /// <param name="layout">The result layout.</param>
        /// <param name="memoryIndex">The failing element's memory index in the layout.</param>
        /// <returns>The exception (not thrown).</returns>
        private static Exception DecodeErrorAt(NDArray a, long[] dims, Shape layout, long memoryIndex)
        {
            var strides = layout.strides;
            var order = Enumerable.Range(0, dims.Length).OrderByDescending(d => strides[d]).ThenBy(d => d).ToArray();
            var coords = new long[dims.Length];
            long rest = memoryIndex;
            foreach (int d in order)
            {
                coords[d] = strides[d] == 0 ? 0 : rest / strides[d];
                rest -= coords[d] * strides[d];
            }
            long flat = 0;
            for (int d = 0; d < dims.Length; d++)
                flat = flat * dims[d] + coords[d];
            using var spread = np.broadcast_to(a, Shape.Adopt((long[])dims.Clone()));
            var value = (byte[])spread.Storage.GetAtIndex(flat);
            for (int i = 0; i < value.Length; i++)
                if (value[i] >= 0x80)
                    return NDStringCodec.DecodeError(value[i], i);
            return new InvalidOperationException("the element holds no non-ASCII byte");
        }

        /// <summary>Whether an array's elements are already <paramref name="descr"/>'s (class and width).</summary>
        /// <param name="a">The array.</param>
        /// <param name="descr">A string descriptor.</param>
        /// <returns>True when no conversion is needed.</returns>
        public static bool SameElement(NDArray a, DType descr)
            => a.typecode == NPTypeCode.String && a.dtype.Meta == descr.Meta && a.dtype.itemsize == descr.itemsize;

        /// <summary>
        ///     Whether an array already satisfies a dtype REQUEST (<c>np.asarray(a, dtype)</c> returns it uncopied): for the
        ///     numeric lanes the lane must match; for a string request the array must be a string array of that class whose
        ///     width is the requested one — an unsized request (<c>'U'</c>, <c>np.str_</c>) adapts to the array first, so a
        ///     <c>&lt;U4</c> array satisfies <c>'U'</c> but not <c>'U2'</c> or <c>'S'</c>.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="requested">The requested dtype (not null).</param>
        /// <returns>True when no conversion is needed.</returns>
        public static bool SatisfiesRequest(NDArray a, DType requested)
        {
            if (a.typecode != NPTypeCode.String && !(requested.Meta is StringDTypeMeta))
                return a.typecode == requested.GetTypeCode();
            if (a.typecode != NPTypeCode.String || !(requested.Meta is StringDTypeMeta) || !requested.isnative)
                return false;
            var adapted = requested.itemsize == 0 ? DTypePromotion.CastDescrToDType(a.dtype, requested.Meta) : requested;
            return SameElement(a, adapted);
        }

        /// <summary>
        ///     Discovers the descriptor, allocates, and encodes <paramref name="values"/> (already in C order).
        /// </summary>
        /// <param name="dims">The result dimensions.</param>
        /// <param name="values">The elements, C order, <c>product(dims)</c> of them.</param>
        /// <param name="dtype">The requested dtype or null.</param>
        /// <param name="order">The requested memory layout.</param>
        /// <returns>The new array.</returns>
        private static NDArray Build(long[] dims, List<object> values, DType dtype, char order)
        {
            // A numeric dtype: NumPy's discovery keeps the requested dtype and packs each element with its setitem.
            if (dtype != null && dtype.Meta is not StringDTypeMeta)
                return BuildNumeric(dims, values, dtype, order);

            var descr = Discover(values, dtype);
            var storage = UnmanagedStorage.AllocateString(new Shape(dims), descr, fillZeros: false);
            var slice = storage.StringSlice;
            byte* p = slice.Address;
            int itemsize = slice.ItemSize;
            for (int i = 0; i < values.Count; i++)
                NDStringStorage.WriteElement(p + (long)i * itemsize, slice.IsUnicode, slice.Chars, values[i]);

            var result = new NDArray(storage);
            if (dims.Length > 1 && (order == 'F' || order == 'f'))
            {
                var f = result.copy('F');
                result.Dispose();
                return f;
            }
            return result;
        }

        /// <summary>
        ///     NumPy's dtype discovery for text values: the class (requested, or <c>U</c> when any value is text, <c>S</c>
        ///     when the text is all bytes) and the width (requested, or the widest value).
        /// </summary>
        /// <param name="values">The element values.</param>
        /// <param name="dtype">The requested string dtype or null.</param>
        /// <returns>The descriptor to allocate (possibly unsized; the allocator grows 0 to one character).</returns>
        /// <exception cref="NotSupportedException">None without a dtype (NumPy's object array), or a value that is neither
        ///     text, bytes, a number nor None.</exception>
        /// <exception cref="TypeError">The width exceeds the class's maximum (<c>string too large to store inside
        ///     array.</c>).</exception>
        /// <remarks>
        ///     The width of a value: a string's code points, a byte[]'s length, 4 for <c>str(None)</c>; a number's is the
        ///     length of its <c>str()</c> under a requested unsized class (<c>PyArray_DTypeFromObjectStringDiscovery</c>), and
        ///     its descriptor's text width without one (NumPy discovers the number's own dtype, then
        ///     <c>PyArray_PromoteTypes</c> widens the string class to that dtype's <c>REQUIRED_STR_LEN</c>).
        /// </remarks>
        private static DType Discover(List<object> values, DType dtype)
        {
            StringDTypeMeta cls;
            if (dtype != null)
            {
                var requested = (StringDTypeMeta)dtype.Meta;
                if (dtype.itemsize > 0)
                    return dtype;
                cls = requested;
            }
            else
            {
                bool anyText = false, anyBytes = false;
                foreach (var v in values)
                {
                    switch (v)
                    {
                        case string _:
                        case char _:
                            anyText = true;
                            break;
                        case byte[] _:
                            anyBytes = true;
                            break;
                        case null:
                            throw new NotSupportedException(
                                "A null (None) among the values would make NumPy build an object array, which NumSharp does not have; " +
                                "pass a string dtype (np.str_ / \"U\") to store it as the text 'None', or replace it.");
                    }
                }
                // Unicode absorbs bytes (string_unicode_common_dtype); an empty value set is NumPy's default U. A number
                // never changes the class: the string class absorbs every numeric DType.
                cls = anyText || !anyBytes ? DTypeRegistry.Str : DTypeRegistry.Bytes;
            }

            long width = 0;
            foreach (var v in values)
            {
                long len = v switch
                {
                    string s => NDStringCodec.CodePointCount(s),
                    char _ => 1,
                    byte[] b => b.Length,
                    null => 4, // str(None) == 'None'
                    _ => dtype != null ? NDStringStorage.StrLength(v) : NumberTextWidth(v),
                };
                if (len > width)
                    width = len;
            }
            if (width > cls.MaxChars)
                throw new TypeError("string too large to store inside array.");
            return width == 0 ? cls.Singleton : cls.Descr(width);
        }

        /// <summary>
        ///     The text width a number contributes when NumPy promotes it with text (<c>np.array(['x', 1])</c>): its
        ///     discovered descriptor's <c>REQUIRED_STR_LEN</c> — a Python int is int64 (21; uint64, 20, when only that holds
        ///     it), a Python float / NumPy float16 / decimal 32, a complex 64, a bool 5.
        /// </summary>
        /// <param name="v">The number (boxed C# value).</param>
        /// <returns>The width in characters.</returns>
        /// <exception cref="NotSupportedException">A Python int past 64 bits (NumPy builds an object array), or a value that
        ///     is not a number.</exception>
        private static long NumberTextWidth(object v)
        {
            NPTypeCode discovered = v switch
            {
                bool => NPTypeCode.Boolean,
                sbyte or byte or short or ushort or int or uint or long => NPTypeCode.Int64,
                ulong u => u <= long.MaxValue ? NPTypeCode.Int64 : NPTypeCode.UInt64,
                System.Numerics.BigInteger bi when bi >= long.MinValue && bi <= long.MaxValue => NPTypeCode.Int64,
                System.Numerics.BigInteger bi when bi >= 0 && bi <= ulong.MaxValue => NPTypeCode.UInt64,
                System.Numerics.BigInteger => throw new NotSupportedException(
                    "A Python int past 64 bits among text values makes NumPy build an object array, which NumSharp does not have."),
                float or double => NPTypeCode.Double,
                Half => NPTypeCode.Half,
                decimal => NPTypeCode.Decimal,
                System.Numerics.Complex => NPTypeCode.Complex,
                _ => throw new NotSupportedException(
                    $"A {v.GetType().Name} value has no NumPy dtype: a text array takes text, bytes, numbers and None."),
            };
            return NDTextFormat.RequiredWidth(discovered);
        }

        /// <summary>
        ///     <c>np.array(values, dtype)</c> with a NUMERIC dtype over values that hold text: every element packed by the
        ///     dtype's setitem (<see cref="ReductionInitial.TryPack"/>), so a text is parsed by CPython's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c> (a bool dtype takes its truth value) and a number is range-checked as NumPy
        ///     packs a Python scalar.
        /// </summary>
        /// <param name="dims">The result dimensions.</param>
        /// <param name="values">The elements, C order.</param>
        /// <param name="dtype">The numeric dtype.</param>
        /// <param name="order">The requested memory layout.</param>
        /// <returns>The new array.</returns>
        /// <exception cref="ValueError">A text is not a number of the dtype's kind (quoting <c>'…'</c>).</exception>
        /// <exception cref="OverflowException">A value does not fit an integer dtype.</exception>
        /// <exception cref="TypeError">None into an integer dtype, or a value of a type NumPy cannot convert.</exception>
        private static NDArray BuildNumeric(long[] dims, List<object> values, DType dtype, char order)
        {
            NPTypeCode tc = dtype.typecode;
            var result = new NDArray(dtype, new Shape(dims), false);
            try
            {
                int size = tc.SizeOf();
                byte* p = (byte*)result.Address;
                Span<byte> slot = stackalloc byte[ReductionInitial.SlotBytes];
                for (int i = 0; i < values.Count; i++)
                {
                    PackNumeric(values[i], tc, slot);
                    slot.Slice(0, size).CopyTo(new Span<byte>(p + (long)i * size, size));
                }
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
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     NumPy's setitem of one creation value into a numeric dtype: <see cref="ReductionInitial.TryPack"/>, plus the
        ///     <c>None</c> rules it leaves to its caller — <c>MyPyFloat_AsDouble(None)</c> is NaN for a float dtype,
        ///     <c>CDOUBLE_setitem(None)</c> NaN+NaNj, <c>PyObject_IsTrue(None)</c> False, and <c>int(None)</c> a TypeError.
        /// </summary>
        /// <param name="value">The value (null is None).</param>
        /// <param name="tc">The numeric dtype.</param>
        /// <param name="slot">Receives the element.</param>
        /// <exception cref="TypeError">None into an integer dtype.</exception>
        private static void PackNumeric(object value, NPTypeCode tc, Span<byte> slot)
        {
            if (value is not null)
            {
                ReductionInitial.TryPack(value, tc, slot);
                return;
            }
            slot.Clear();
            switch (tc)
            {
                case NPTypeCode.Boolean:
                    return;
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal:
                case NPTypeCode.Complex:
                    // NPY_NAN is the POSITIVE quiet NaN (.NET's double.NaN has the sign bit set).
                    double nan = BitConverter.UInt64BitsToDouble(0x7FF8_0000_0000_0000UL);
                    if (tc == NPTypeCode.Complex)
                        System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref slot[0], new System.Numerics.Complex(nan, nan));
                    else
                        ReductionInitial.TryPack(nan, tc, slot);
                    return;
                default:
                    throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            }
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
        private static void Collect(Array array, List<long> dims, List<object> values)
        {
            int leafDepth = -1;
            Visit(array, 0, dims, values, ref leafDepth);
        }

        /// <summary>
        ///     One level of <see cref="Collect"/> (NumPy's <c>update_shape</c> rule): a level's axes are appended the first
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
