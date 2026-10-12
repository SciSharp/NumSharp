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
        ///     <b>From <c>StringDType</c> into another dtype.</b> The source is converted whole to the destination's descriptor
        ///     (the <c>StringDType</c> casts), then copied by the destination dtype's own route.
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
                // A StringDType source into another dtype: converted whole, then that dtype's own copy (the rule passed).
                using var converted = src.astype(to, true, 'K', "unsafe");
                CopytoAssignArray(dst, converted, @where, NPY_CASTING.NPY_UNSAFE_CASTING);
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

        /// <summary>The address of a <c>StringDType</c> array's first cell (<c>PyArray_DATA</c>; 16-byte cells).</summary>
        /// <param name="a">A <c>StringDType</c> array.</param>
        /// <returns><c>Storage.Address + offset * 16</c>.</returns>
        private static unsafe byte* FirstCell(NDArray a) => a.Storage.Address + a.Shape.offset * VStringAllocator.CellSize;
    }
}
