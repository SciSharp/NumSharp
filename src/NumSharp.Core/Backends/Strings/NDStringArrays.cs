using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     Helpers of the fixed-width string arrays that stay with the text backend: the descriptor of an element read back
    ///     (<see cref="ScalarDescr"/>), np.full's and np.where's text routes (<see cref="Full"/>, <see cref="Where"/>), the
    ///     unit-array views the string ufuncs compute on (<see cref="FromUnitArray"/>, <see cref="UnitArray"/>), and the
    ///     ASCII and element-equality tests (<see cref="AllAscii"/>, <see cref="SameElement"/>). The coercion of C# values
    ///     holding text into an array (NumPy's array coercion: the shape walk, the discovery, the stores) is the array
    ///     object's (<c>UnmanagedStorage.Coercion.Text.cs</c>).
    /// </summary>
    internal static unsafe class NDStringArrays
    {
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
        ///     <c>np.full(shape, fill_value, dtype)</c> / <c>np.full_like</c> for a string result — NumPy's
        ///     <c>a = empty(shape, dtype); copyto(a, fill_value, casting='unsafe')</c>: the result dtype is the requested one
        ///     (an unsized request is ONE character, NumPy's allocation rule) or the fill value's own; the value is encoded
        ///     once into a 0-d array and broadcast by the string copy route, which truncates (<c>'abc'</c> into <c>S1</c> is
        ///     <c>b'a'</c>) and converts <c>S ↔ U</c> with the ASCII checks.
        /// </summary>
        /// <param name="shape">The result dimensions and layout: a fresh dense layout (C, F or <c>np.full_like</c>'s KEEPORDER
        ///     permutation, <see cref="Shape.NewLikeLayout"/>) is kept, a view's layout is re-laid out in C order.</param>
        /// <param name="fillValue">A <see cref="string"/>, <c>byte[]</c>, <see cref="char"/>, null (None), a number (formatted
        ///     as its <c>str()</c>, NumPy's <c>copyto(casting='unsafe')</c>) or a 0-d array.</param>
        /// <param name="dtype">The result dtype (a string dtype), or null to take the fill value's.</param>
        /// <returns>The filled array.</returns>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        public static NDArray Full(Shape shape, object fillValue, DType dtype)
        {
            using var value = fillValue is NDArray nd
                ? nd.copy()
                : UnmanagedStorage.FromAnyScalar(fillValue, dtype?.Meta is StringDTypeMeta cls ? cls.singleton : null);
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
            // A StringDType value operand moves through the allocators (NDVStringMove), never the unit / byte routes.
            if (x.Storage.IsVString || y.Storage.IsVString)
                return NDVStringMove.Where(condition, x, y, dims);
            // NumPy's result_type(x, y) under NEP 50: a C# number literal is a WEAK Python scalar (Nep50Operand), which has
            // no common DType with text — np.where(c, u, 1) raises DTypePromotionError naming _PyLongDType, where a 0-d
            // number ARRAY (np.int64(1)) promotes to a string wide enough for its text (<U21).
            var descr = DTypePromotion.ResultType(x.Nep50Operand, y.Nep50Operand);

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
            // np.amax's whole call over the unit view is the engine's maximum.reduce (TensorEngine.ReduceAMax).
            using var max = units.TensorEngine.ReduceAMax(units, null);
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
            // Each element's largest unit: maximum.reduce over the unit axis (np.amax(units, -1)'s engine member).
            using var max = units.TensorEngine.ReduceAMax(units, -1);
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
    }
}
