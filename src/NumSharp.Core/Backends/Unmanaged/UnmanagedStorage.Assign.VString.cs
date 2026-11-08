using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    // ---------------------------------------------------------------------------------------------------------------------
    // PyArray_AssignArray / PyArray_AssignRawScalar for NEP 55 StringDType operands: every string moves through the
    // allocators (UnmanagedStorage.Assign.cs holds the numeric assignment).
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class UnmanagedStorage
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
        ///     <b>Into <c>StringDType</c>.</b> A source of another descriptor is converted to the destination's (the cast
        ///     route: <c>U</c> / <c>S</c> text, numbers as their <c>str()</c>, another <c>StringDType</c> with its NA rules), after
        ///     NumPy's broadcast checks. Without a mask the whole source is converted and the copy is the <c>StringDType</c> copy
        ///     core (<see cref="NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/>: broadcasting and NumPy's overlap rule). With a
        ///     mask the numeric <c>np.copyto</c> runs on MARKS (the source's element numbers into an <c>int64</c> array of the
        ///     destination's shape, under the mask) and the marked elements receive the strings (<see cref="NDVStringMove"/>);
        ///     a source of another descriptor then has only its selected elements converted
        ///     (<see cref="CopytoIntoVStringSelected"/>), as NumPy's masked transfer casts only the selected runs.
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
            if (!scalar && from.Meta.HasReferences && to.Meta.HasReferences && from.Equals(to)
                && FirstCell(src) == FirstCell(dst) && SameLayout(src.Shape, dst.Shape))
                return;

            // NumPy's text for a read-only destination (NumSharpException.ThrowIfNotWriteable's).
            if (!dst.Shape.IsWriteable)
                throw new ValueError("assignment destination is read-only");
            if (!DTypeCasting.CanCastTypeTo(from, to, casting))
                throw DTypeCasting.InvalidCastError(from, to, casting, scalar);

            if (!to.Meta.HasReferences)
            {
                // A StringDType source into another dtype (the rule passed): NumPy's remaining checks, then the conversion of
                // exactly the elements its transfer converts.
                CopytoFromVString(dst, src, @where, to);
                return;
            }

            // NumPy's broadcast check of the source (leading extent-1 axes beyond dst's rank dropped first), then the 0-d
            // mask shortcut and the mask's broadcast check — all before anything is converted or written.
            if (!scalar)
                Shape.AssignmentBroadcastStrides(dst.Shape.dimensions, src.Shape.dimensions, src.Shape.strides, "input array", stripLeadingUnitAxes: true);
            if (@where is not null && @where.ndim == 0)
            {
                if (!(bool)@where.GetAtIndex(0))
                    return;
                @where = null;
            }
            if (@where is not null)
                Shape.AssignmentBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides, "where mask", stripLeadingUnitAxes: false);
            if (dst.size == 0)
                return;

            // A source of another descriptor under a mask: only the elements the mask selects are converted, as NumPy's
            // masked transfer casts only the selected runs - a masked-off element is never read (it may not even be a value:
            // a ufunc's text-out temporary leaves its masked-off slots uninitialized).
            if (@where is not null && !from.Meta.HasReferences)
            {
                CopytoIntoVStringSelected(dst, src, @where, to);
                return;
            }

            // The source in the destination's descriptor: a StringDType source is copied as it is (the copy core applies the
            // NA rules between two StringDType descriptors), any other one is converted (without a mask every source element
            // reaches some destination element, so each is converted, as NumPy's transfer converts it).
            NDArray value = from.Meta.HasReferences ? src : src.astype(to, true, 'K', "unsafe");
            try
            {
                if (@where is null)
                {
                    UnmanagedStorage source = value.ndim > dst.ndim ? value.Storage.AliasWithoutLeadingAxes(dst.ndim) : value.Storage;
                    NDIter.Copy(dst.Storage, source);
                    return;
                }
                using var marks = NDVStringMove.NewMarks(dst);
                using var index = NDVStringMove.ValueIndex(value);
                // The numeric masked copy broadcasts the element numbers (and the mask) exactly as the strings would be.
                Copyto(marks, index, NPY_CASTING.NPY_UNSAFE_CASTING, @where);
                NDVStringMove.ScatterMarks(dst, marks, value);
            }
            finally
            {
                if (!ReferenceEquals(value, src))
                    value.Dispose();
            }
        }

        /// <summary>
        ///     The masked half of <see cref="CopytoAssignVStrings"/> for a source of another descriptor (a number, <c>U</c> or
        ///     <c>S</c> text, bool, ...) into a <c>StringDType</c> destination, once every check passed: the source elements a
        ///     selected position takes are converted to <paramref name="to"/> - each once - and moved in through the
        ///     allocators; no other source element is read or converted.
        /// </summary>
        /// <param name="dst">The writeable, non-empty <c>StringDType</c> destination.</param>
        /// <param name="src">The source (its broadcast to <paramref name="dst"/> checked).</param>
        /// <param name="where">The mask (not 0-d; its broadcast to <paramref name="dst"/> checked).</param>
        /// <param name="to">The destination's descriptor.</param>
        /// <exception cref="System.Exception">The cast's own errors for a SELECTED element (NumPy raises them too: an invalid
        ///     bool byte gives <c>invalid value encountered in bool to string cast</c>).</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts while the strings are allocated.</exception>
        /// <remarks>
        ///     <para>NumPy's masked transfer (<c>PyArray_GetMaskedDTypeTransferFunction</c>) casts each run the mask selects, so
        ///     a masked-off element is never cast. Converting the whole source first, as the unmasked copy does, would cast them
        ///     all: a masked-off bool byte that is no bool value (a ufunc's text-out temporary leaves its masked-off slots
        ///     uninitialized) raised <c>invalid value encountered in bool to string cast</c> where NumPy writes nothing (plan
        ///     np-layer NL1).</para>
        ///     <para>The selection is decided once on MARKS: the source's element numbers copied under the mask by the numeric
        ///     masked copy (which broadcasts them, and the mask, exactly as it would the values). Each named source element is
        ///     converted once, and the strings move through the allocators (<see cref="NDVStringMove.ScatterMarks"/>), each into
        ///     its destination once. Two routes, by whether a source element can feed several positions:</para>
        ///     <para><b>No broadcast</b> (the source has the destination's size): each source element feeds at most its own
        ///     position. The selected positions are found with the vectorized compare and nonzero kernels; all of them is the
        ///     unmasked copy's whole-source conversion; otherwise their source elements are gathered in the destination's C
        ///     order, converted, and scattered through a ramp of slots.</para>
        ///     <para><b>A broadcast source</b> (smaller than the destination): a source element is named when the mask
        ///     selects any position along the axes it is broadcast over, so the flags are the vectorized <c>any()</c> of the
        ///     mask over those axes (a scalar pass over every mark, branch-free, still measured 0.88x of the whole-source
        ///     copy for a row broadcast down 100 rows); all of them is the whole conversion again; otherwise the named ones
        ///     are converted and placed at their numbers in a value array of the source's length (its other cells never
        ///     read), which the original marks then scatter. NumPy casts a broadcast element once per selected position;
        ///     once per element gives the same strings and the same error (one per position measured 0.16-0.25x).</para>
        ///     <para>NumPy visits a non-C destination in its memory order, so with several bad selected elements the first one
        ///     it reports can differ from the one reported here, as in <see cref="CopytoFromVString"/>; the texts are equal.</para>
        /// </remarks>
        private static void CopytoIntoVStringSelected(NDArray dst, NDArray src, NDArray @where, DType to)
        {
            // Each destination element's source number (-1 where the mask leaves it alone).
            using var marks = NDVStringMove.NewMarks(dst);
            using (var index = NDVStringMove.ValueIndex(src))
                Copyto(marks, index, NPY_CASTING.NPY_UNSAFE_CASTING, @where);
            long sourceCount = src.size, positionCount = marks.size;

            if (sourceCount == positionCount)
            {
                // No broadcast: the selected positions name distinct source elements.
                using var zero = NDArray.Scalar(0L);
                using var selected = marks.TensorEngine.GreaterEqual(marks, zero, null, null, null);
                using var positions = selected.TensorEngine.FlatNonZero(selected);
                if (positions.size == 0)
                    return;
                if (positions.size == positionCount)
                {
                    // Everything selected: the whole source converted once, as the unmasked copy converts it.
                    using var all = src.astype(to, true, 'K', "unsafe");
                    NDVStringMove.ScatterMarks(dst, marks, all);
                    return;
                }
                using var flatMarks = NDArray.ravel(marks, 'C');
                using var sourceNumbers = TakeFrom(flatMarks, positions, null, null, ClipModes.ClipModeRaise);
                using var flatSource = NDArray.ravel(src, 'C');
                using var picked = TakeFrom(flatSource, sourceNumbers, null, null, ClipModes.ClipModeRaise);
                using var converted = picked.astype(to, true, 'K', "unsafe");
                // Destination position -> its converted value's number (0 .. k-1), then the strings moved in.
                using var slots = NDVStringMove.NewMarks(dst);
                using (var ramp = NDVStringMove.ValueIndex(converted))
                    PutTo(slots, positions, ramp, "raise");
                NDVStringMove.ScatterMarks(dst, slots, converted);
                return;
            }

            // A broadcast source: a source element is named when the mask selects any position along the axes it is broadcast
            // over. Those axes are dst's axes where the source (its trailing dimensions aligned to dst's, leading extra ones
            // all 1) has extent 1 while dst does not; any() over them with keepdims gives the flags in the source's own C
            // order (the order of its element numbers), with the vectorized reduction instead of a pass over every mark.
            int rank = dst.ndim;
            long[] dstDims = dst.Shape.dimensions;
            long[] srcDims = src.Shape.dimensions;
            int broadcastAxisCount = 0;
            var broadcastAxes = new int[rank];
            for (int axis = 0; axis < rank; axis++)
            {
                int srcAxis = axis + (srcDims.Length - rank);
                long srcExtent = srcAxis >= 0 ? srcDims[srcAxis] : 1;
                if (srcExtent == 1 && dstDims[axis] != 1)
                    broadcastAxes[broadcastAxisCount++] = axis;
            }
            Array.Resize(ref broadcastAxes, broadcastAxisCount);
            using var maskOverDst = NDArray.broadcast_to(@where, dst.Shape);
            using var named = maskOverDst.TensorEngine.Any(maskOverDst, broadcastAxes, null, keepdims: true);
            using var numbers = named.TensorEngine.FlatNonZero(named);   // ascending source numbers
            if (numbers.size == 0)
                return;
            if (numbers.size == sourceCount)
            {
                // Every source element reaches a selected position: one conversion of the whole source, scattered by the
                // marks (flat C-order source numbers: the converted array's own flat indices).
                using var all = src.astype(to, true, 'K', "unsafe");
                NDVStringMove.ScatterMarks(dst, marks, all);
                return;
            }

            // The named source elements, converted once each and placed at their source numbers in a value array of the
            // source's length (its other cells are never read), which the original marks then scatter.
            using var flatSrc = NDArray.ravel(src, 'C');
            using var namedValues = TakeFrom(flatSrc, numbers, null, null, ClipModes.ClipModeRaise);
            using var convertedNamed = namedValues.astype(to, true, 'K', "unsafe");
            using var value = new NDArray(to, Shape.Vector(sourceCount), true);
            using (var valueSlots = NDVStringMove.NewMarks(value))
            {
                using (var ramp = NDVStringMove.ValueIndex(convertedNamed))
                    PutTo(valueSlots, numbers, ramp, "raise");
                NDVStringMove.ScatterMarks(value, valueSlots, convertedNamed);
            }
            NDVStringMove.ScatterMarks(dst, marks, value);
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
                    Shape.AssignmentBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides, "where mask", stripLeadingUnitAxes: false);
                // One element or none: the masked loop converts the value only for a selected element (a mask that
                // broadcasts into one element has one element).
                if (dst.size == 0 || (@where is not null && !(bool)@where.GetAtIndex(0)))
                    return;
                using var single = src.astype(to, true, 'K', "unsafe");
                CopytoAssignArray(dst, single, null, NPY_CASTING.NPY_UNSAFE_CASTING);
                return;
            }

            Shape.AssignmentBroadcastStrides(dst.Shape.dimensions, src.Shape.dimensions, src.Shape.strides, "input array", stripLeadingUnitAxes: true);
            // "optimization: scalar boolean mask"
            if (@where is not null && @where.ndim == 0)
            {
                if (!(bool)@where.GetAtIndex(0))
                    return;
                @where = null;
            }
            if (@where is not null)
                Shape.AssignmentBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides, "where mask", stripLeadingUnitAxes: false);
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
                Copyto(marks, index, NPY_CASTING.NPY_UNSAFE_CASTING, @where);
            using var zero = NDArray.Scalar(0L);
            using var selected = marks.TensorEngine.GreaterEqual(marks, zero, null, null, null);
            using var positions = selected.TensorEngine.FlatNonZero(selected);
            if (positions.size == 0)
                return;
            // Gather the selected texts (destination C order) and convert only them.
            using var flatMarks = NDArray.ravel(marks, 'C');
            using var sourceNumbers = TakeFrom(flatMarks, positions, null, null, ClipModes.ClipModeRaise);
            using var flatSource = NDArray.ravel(src, 'C');
            using var picked = TakeFrom(flatSource, sourceNumbers, null, null, ClipModes.ClipModeRaise);
            using var values = picked.astype(to, true, 'K', "unsafe");
            // The values at their destination positions (a C-ordered staging array), written back under the same selection.
            using var staged = new NDArray(to, new Shape(dst.Shape.dimensions, copy: false), true);
            PutTo(staged, positions, values, "raise");
            Copyto(dst, staged, NPY_CASTING.NPY_UNSAFE_CASTING, selected);
        }

        /// <summary>The address of a <c>StringDType</c> array's first cell (<c>PyArray_DATA</c>; 16-byte cells).</summary>
        /// <param name="a">A <c>StringDType</c> array.</param>
        /// <returns><c>Storage.Address + offset * 16</c>.</returns>
        private static unsafe byte* FirstCell(NDArray a) => a.Storage.Address + a.Shape.offset * VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
    }
}
