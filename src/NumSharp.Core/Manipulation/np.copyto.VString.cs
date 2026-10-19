using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     <c>PyArray_AssignArray</c> / <c>PyArray_AssignRawScalar</c> when either operand is a NEP 55 <c>StringDType</c>
        ///     array: NumPy's checks in NumPy's order with descriptor-level cast rules, then a copy that moves every string
        ///     through the allocators (never the raw masked kernel, which would copy cells away from their allocator).
        /// </summary>
        /// <param name="dst">The destination.</param>
        /// <param name="src">The source (any rank).</param>
        /// <param name="where">The validated bool mask, or <c>null</c>.</param>
        /// <param name="casting">The casting level.</param>
        /// <exception cref="ValueError">A read-only destination, or a source / mask that does not broadcast.</exception>
        /// <exception cref="InvalidCastException">The cast is not allowed under <paramref name="casting"/> (NumPy's TypeError
        ///     text, "array data" or "scalar").</exception>
        /// <exception cref="System.Exception">The cast's own errors once it is allowed (a non-UTF-8 <c>S</c> source, a text that
        ///     does not parse into a number, an NA value into a non-nullable dtype).</exception>
        /// <remarks>
        ///     <para>
        ///     <b>Into <c>StringDType</c>.</b> A source of another descriptor is converted to the destination's first (the cast
        ///     route: <c>U</c> / <c>S</c> text, numbers as their <c>str()</c>, another <c>StringDType</c> with its NA rules), after
        ///     NumPy's broadcast checks. Without a mask the copy is the <c>StringDType</c> copy core
        ///     (<see cref="NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/>: broadcasting and NumPy's overlap rule). With a mask
        ///     the numeric <c>np.copyto</c> runs on MARKS (the source's element numbers into an <c>int64</c> array of the
        ///     destination's shape, under the mask) and the marked elements receive the strings (<see cref="NDVStringMove"/>).
        ///     </para>
        ///     <para>
        ///     <b>From <c>StringDType</c> into another dtype.</b> NumPy's broadcast and mask checks first, then only the elements
        ///     NumPy's transfer converts are converted to the destination's descriptor (the <c>StringDType</c> casts) and copied
        ///     by the destination dtype's own route (<see cref="CopytoFromVString"/>): a text that is not copied is never parsed.
        ///     </para>
        /// </remarks>
        private static unsafe void CopytoAssignVStrings(NDArray dst, NDArray src, NDArray @where, NPY_CASTING casting)
        {
            bool scalar = src.ndim == 0;
            DType from = src.dtype, to = dst.dtype;

            // AssignArray: a redundant self-copy (the same cells, an equal descriptor, the same layout) returns before every
            // check, the read-only one included.
            if (!scalar && src.Storage.IsVString && dst.Storage.IsVString && from.Equals(to)
                && FirstCell(src) == FirstCell(dst) && SameLayout(src.Shape, dst.Shape))
                return;

            // NumPy's text for a read-only destination (NumSharpException.ThrowIfNotWriteable's).
            if (!dst.Shape.IsWriteable)
                throw new ValueError("assignment destination is read-only");
            if (!DTypeCasting.CanCastTypeTo(from, to, casting))
                throw NDStringCasting.InvalidCast(from, to, casting, scalar);

            if (!dst.Storage.IsVString)
            {
                // A StringDType source into another dtype (the rule passed): NumPy's remaining checks, then the conversion of
                // exactly the elements its transfer converts.
                CopytoFromVString(dst, src, @where, to);
                return;
            }

            // NumPy's broadcast check of the source (leading extent-1 axes beyond dst's rank dropped first), then the 0-d
            // mask shortcut and the mask's broadcast check — all before anything is converted or written.
            if (!scalar)
                CopytoBroadcastStrides(dst.Shape.dimensions, src.Shape.dimensions, src.Shape.strides, "input array", stripLeadingUnitAxes: true);
            if (@where is not null && @where.ndim == 0)
            {
                if (!(bool)@where.GetAtIndex(0))
                    return;
                @where = null;
            }
            if (@where is not null)
                CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides, "where mask", stripLeadingUnitAxes: false);
            if (dst.size == 0)
                return;

            // The source in the destination's descriptor: a StringDType source is copied as it is (the copy core applies the
            // NA rules between two StringDType descriptors), any other one is converted.
            NDArray value = src.Storage.IsVString ? src : src.astype(to, true, 'K', "unsafe");
            try
            {
                if (@where is null)
                {
                    UnmanagedStorage source = value.ndim > dst.ndim ? StrippedSourceView(value, dst.ndim) : value.Storage;
                    NDIter.Copy(dst.Storage, source);
                    return;
                }
                using var marks = NDVStringMove.NewMarks(dst);
                using var index = NDVStringMove.ValueIndex(value);
                // The numeric masked copy broadcasts the element numbers (and the mask) exactly as the strings would be.
                copyto(marks, index, "unsafe", @where);
                NDVStringMove.ScatterMarks(dst, marks, value);
            }
            finally
            {
                if (!ReferenceEquals(value, src))
                    value.Dispose();
            }
        }

        /// <summary>
        ///     <c>PyArray_AssignArray</c> / <c>PyArray_AssignRawScalar</c> from a <c>StringDType</c> source into a destination of
        ///     another dtype, once the read-only check and the cast rule passed: NumPy's remaining checks in its order, and the
        ///     conversion of exactly the elements its transfer converts.
        /// </summary>
        /// <param name="dst">The writeable destination (not <c>StringDType</c>).</param>
        /// <param name="src">The <c>StringDType</c> source.</param>
        /// <param name="where">The validated bool mask, or <c>null</c>.</param>
        /// <param name="to">The destination's descriptor.</param>
        /// <exception cref="ValueError">The source or the mask does not broadcast to <paramref name="dst"/>
        ///     (<c>could not broadcast input array from shape (2,) into shape (3,)</c>, <c>… where mask …</c>), a converted text
        ///     is not a number of the destination's kind (<c>could not convert string to float: 'x'</c>), or a converted NA has
        ///     no value in the destination's dtype.</exception>
        /// <exception cref="OverflowException">A converted integer text does not fit the destination.</exception>
        /// <remarks>
        ///     <para><b>A 0-d source</b> (<c>PyArray_AssignRawScalar</c>): into a destination of more than one element the value
        ///     is converted ONCE, before the mask is read, so a bad text raises under any mask; into one element or none it is
        ///     converted inside the masked loop, only when that element is selected.</para>
        ///     <para><b>An array source</b> (<c>PyArray_AssignArray</c>): its broadcast check (leading extent-1 axes beyond the
        ///     destination's rank dropped), the 0-d mask shortcut, the mask's broadcast check; an empty destination converts
        ///     nothing. Without a mask every source element is converted (each reaches some destination element). With one,
        ///     only the selected elements are: the source's element numbers are copied under the mask into MARKS (the numeric
        ///     masked copy broadcasts them exactly as it would the strings), the marked texts are gathered in the destination's
        ///     C order and converted, and the values are written back under the same selection — a text that is not selected
        ///     is never parsed, as in NumPy's masked transfer. (NumPy visits a non-C destination in its memory order, so with
        ///     several bad selected texts the first one it reports can differ from the C-order first reported here.)</para>
        /// </remarks>
        private static void CopytoFromVString(NDArray dst, NDArray src, NDArray @where, DType to)
        {
            if (src.ndim == 0)
            {
                // A destination of more than one element takes the value converted once, up front: its parse error ignores
                // the mask, whose checks the numeric copy of the converted value then makes.
                if (dst.size > 1)
                {
                    using var once = src.astype(to, true, 'K', "unsafe");
                    CopytoAssignArray(dst, once, @where, NPY_CASTING.NPY_UNSAFE_CASTING);
                    return;
                }
                if (@where is not null)
                    CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides, "where mask", stripLeadingUnitAxes: false);
                // One element or none: the masked loop converts the value only for a selected element (a mask that
                // broadcasts into one element has one element).
                if (dst.size == 0 || (@where is not null && !(bool)@where.GetAtIndex(0)))
                    return;
                using var single = src.astype(to, true, 'K', "unsafe");
                CopytoAssignArray(dst, single, null, NPY_CASTING.NPY_UNSAFE_CASTING);
                return;
            }

            CopytoBroadcastStrides(dst.Shape.dimensions, src.Shape.dimensions, src.Shape.strides, "input array", stripLeadingUnitAxes: true);
            // "optimization: scalar boolean mask"
            if (@where is not null && @where.ndim == 0)
            {
                if (!(bool)@where.GetAtIndex(0))
                    return;
                @where = null;
            }
            if (@where is not null)
                CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides, "where mask", stripLeadingUnitAxes: false);
            if (dst.size == 0)
                return;

            if (@where is null)
            {
                using var converted = src.astype(to, true, 'K', "unsafe");
                CopytoAssignArray(dst, converted, null, NPY_CASTING.NPY_UNSAFE_CASTING);
                return;
            }

            // The selected elements' source numbers (-1 where nothing is copied), broadcast and masked by the numeric copy.
            using var marks = NDVStringMove.NewMarks(dst);
            using (var index = NDVStringMove.ValueIndex(src))
                copyto(marks, index, "unsafe", @where);
            using var zero = NDArray.Scalar(0L);
            using var selected = np.greater_equal(marks, zero);
            using var positions = np.flatnonzero(selected);
            if (positions.size == 0)
                return;
            // Gather the selected texts (destination C order) and convert only them.
            using var flatMarks = np.ravel(marks);
            using var sourceNumbers = np.take(flatMarks, positions);
            using var flatSource = np.ravel(src);
            using var picked = np.take(flatSource, sourceNumbers);
            using var values = picked.astype(to, true, 'K', "unsafe");
            // The values at their destination positions (a C-ordered staging array), written back under the same selection.
            using var staged = np.zeros(Shape.Adopt((long[])dst.Shape.dimensions.Clone()), to);
            np.put(staged, positions, values);
            copyto(dst, staged, "unsafe", selected);
        }

        /// <summary>The address of a <c>StringDType</c> array's first cell (<c>PyArray_DATA</c>; 16-byte cells).</summary>
        /// <param name="a">A <c>StringDType</c> array.</param>
        /// <returns><c>Storage.Address + offset * 16</c>.</returns>
        private static unsafe byte* FirstCell(NDArray a) => a.Storage.Address + a.Shape.offset * VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
    }
}
