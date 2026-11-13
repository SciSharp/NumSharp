using System;
using System.Globalization;

namespace NumSharp
{
    public partial class NDArray
    {
        // ---------------------------- np.lib.stride_tricks.as_strided ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.lib.stride_tricks.as_strided(x, shape, strides, subok, writeable)</c> after the facade's null
        ///     check: a view of <paramref name="x"/>'s memory with the given shape and BYTE strides, refused where it would
        ///     read outside <paramref name="x"/>'s buffer.
        /// </summary>
        /// <param name="x">The array whose memory is viewed (not null; the caller's own object, a subclass included).</param>
        /// <param name="shape">The view's shape; null keeps <paramref name="x"/>'s.</param>
        /// <param name="strides">The view's strides in bytes, one per dimension; null keeps <paramref name="x"/>'s array
        ///     interface strides (none for a C-contiguous <paramref name="x"/>: the C strides of the view's shape).</param>
        /// <param name="subok">Keep <paramref name="x"/>'s class (NumPy's <c>_maybe_view_as_subclass</c>).</param>
        /// <param name="writeable">False makes the view read-only; true leaves it as writeable as <paramref name="x"/>.</param>
        /// <returns>The view (OWNDATA false), of <paramref name="x"/>'s class with <paramref name="subok"/>, plain
        ///     otherwise.</returns>
        /// <exception cref="ValueError">Strides of another length than the shape (<c>mismatch in length of strides and
        ///     shape</c>), checked before a negative dimension (<c>negative dimensions are not allowed</c>); a stride that
        ///     is not a multiple of the item size, or a view reaching outside <paramref name="x"/>'s buffer (NumSharp's
        ///     refusals where NumPy reads arbitrary memory); a <c>numpy.matrix</c> view of more than two long axes
        ///     (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's statements: <c>x = np.array(x, copy=None, subok=subok)</c>, its <c>__array_interface__</c> with the
        ///     given shape / strides, <c>np.asarray(DummyArray(interface, base=x))</c> (strides of another length are
        ///     refused there, before the creation refuses a negative dimension), <c>_maybe_view_as_subclass</c>, then
        ///     <c>if array.flags.writeable and not writeable: array.flags.writeable = False</c>.
        ///     </para>
        ///     <para>
        ///     <see cref="SubclassAwareAttribute"/>: <paramref name="subok"/> reads the caller's class, so the public
        ///     function is not woven (a woven one would receive the plain view of a <c>numpy.matrix</c> and lose it). With
        ///     <paramref name="subok"/> false the view is plain whatever <paramref name="x"/> is, as NumPy's
        ///     <c>np.array(x, subok=False)</c> makes it.
        ///     </para>
        ///     <para>
        ///     Footgun, as in NumPy: the view is WRITEABLE even where several elements read one item (a zero or overlapping
        ///     stride), so one write changes all of them. NumSharp's <see cref="Shape"/> computes a zero-stride layout
        ///     read-only (its broadcast protection, which NumPy applies only in <c>broadcast_to</c> /
        ///     <c>broadcast_arrays</c>), so this view sets the bit back for a writeable <paramref name="x"/>.
        ///     </para>
        /// </remarks>
        [SubclassAware]
        internal static NDArray as_strided(NDArray x, Shape? shape, long[] strides, bool subok, bool writeable)
        {
            var src = x.Shape;
            long itemsize = x.dtypesize;

            // __array_interface__: x's shape, and x's strides unless x is C-contiguous (then None: the C strides of
            // whatever shape is used); a given shape / strides replaces its entry.
            long[] dims = shape.HasValue
                ? (long[])(shape.Value.dimensions ?? System.Array.Empty<long>()).Clone()
                : (long[])(src.dimensions ?? System.Array.Empty<long>()).Clone();
            long[] byteStrides = strides is not null ? (long[])strides.Clone()
                : src.IsContiguous ? null
                : x.strides;

            // PyArray_FromInterface: a strides tuple of another length is refused first, then the creation refuses a
            // negative dimension.
            if (byteStrides is not null && byteStrides.Length != dims.Length)
                throw new ValueError("mismatch in length of strides and shape");
            foreach (long d in dims)
                if (d < 0)
                    throw new ValueError("negative dimensions are not allowed");

            // NumSharp's strides count whole items: a byte stride between items cannot be represented (NumPy would read
            // a misaligned item).
            var elementStrides = new long[dims.Length];
            if (byteStrides is null)
            {
                long step = 1;
                for (int i = dims.Length - 1; i >= 0; i--)
                {
                    elementStrides[i] = step;
                    step *= Math.Max(dims[i], 1);
                }
            }
            else
            {
                for (int i = 0; i < dims.Length; i++)
                {
                    if (itemsize == 0 || byteStrides[i] % itemsize != 0)
                        throw new ValueError(string.Create(CultureInfo.InvariantCulture,
                            $"as_strided: a stride of {byteStrides[i]} bytes is not a multiple of the item size {itemsize} (NumSharp's strides count whole items)"));
                    elementStrides[i] = byteStrides[i] / itemsize;
                }
            }

            // The bounds NumPy does not check: every reachable item must lie inside x's buffer (an empty view reaches
            // none). The extremes are the first item plus each axis' most negative / most positive step.
            long bufferSize = src.bufferSize > 0 ? src.bufferSize : src.size;
            bool empty = false;
            long low = 0, high = 0;
            for (int i = 0; i < dims.Length; i++)
            {
                if (dims[i] == 0)
                {
                    empty = true;
                    break;
                }
                long reach = (dims[i] - 1) * elementStrides[i];
                if (reach < 0)
                    low += reach;
                else
                    high += reach;
            }
            if (!empty && (src.offset + low < 0 || src.offset + high >= bufferSize))
                throw new ValueError("as_strided: the view reaches outside the array's buffer (NumSharp refuses the memory NumPy would read past it)");

            // The view over x's buffer window: x's offset and buffer size, so it reads only memory x may read. NumPy's view
            // starts as writeable as x even with a zero stride, which NumSharp's Shape computes read-only: the bit is set
            // here, and Alias clears it again for a read-only x (a read-only array's view stays read-only).
            var viewShape = new Shape(dims, elementStrides, src.offset, bufferSize);
            if (!viewShape.IsWriteable && src.IsWriteable)
                viewShape = viewShape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE);
            // The alias holds its own counted reference to the buffer, so the view outlives a disposed x.
            var view = new NDArray(x.Storage.Alias(viewShape)) { TensorEngine = x.TensorEngine };

            // `if array.flags.writeable and not writeable: array.flags.writeable = False`.
            if (!writeable && view.Shape.IsWriteable)
                view.setflags(write: false);

            // _maybe_view_as_subclass: with subok a subclass operand's class (the matrix finalize rule included); the
            // plain view is released when the class made a new object over the same memory.
            if (!subok)
                return view;
            NDArray result;
            try
            {
                result = NDSubclass.NewSubtype(view, null, x, keepRank: false, NewSubtypeZeroDim.Array);
            }
            catch
            {
                // A matrix finalize refusal (more than two long axes): the plain view is this call's own, released now
                // rather than left to the finalizer.
                view.Dispose();
                throw;
            }
            if (!ReferenceEquals(result, view))
                view.Dispose();
            return result;
        }
    }
}
