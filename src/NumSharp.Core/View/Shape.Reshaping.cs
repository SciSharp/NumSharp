using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     Represents a shape of an N-D array.
    /// </summary>
    /// <remarks>Handles slicing, indexing based on coordinates or linear offset and broadcastted indexing.</remarks>
    public partial struct Shape
    {
        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        /// <exception cref="ArgumentException">If <paramref name="newShape"/>'s size == 0</exception>
        /// <param name="unsafe">When true, then guards are skipped.</param>
        [MethodImpl(OptimizeAndInline)]
        public readonly Shape Reshape(Shape newShape, bool @unsafe = true)
        {
            if (IsBroadcasted)
            {
                return _reshapeBroadcast(newShape, @unsafe);
            }

            // Handle -1 in reshape - returns new shape with inferred dimension
            newShape = _inferMissingDimension(newShape);

            if (!@unsafe)
            {
                // Check if this is a scalar reshape (ndim=0 shape from default constructor or empty dims)
                bool isScalarShape = (newShape.dimensions == null || newShape.dimensions.Length == 0);

                if (isScalarShape)
                {
                    // Scalar shapes are valid only when reshaping from size 1.
                    // NumPy renders the empty shape as "()" (probed: np.zeros(3).reshape(())).
                    if (size != 1)
                        throw ReshapeSizeMismatch(size, Array.Empty<long>());
                }
                else
                {
                    // A zero-size request against a non-empty array is just a size mismatch —
                    // NumPy reports it with the same text as any other (probed:
                    // np.zeros(6).reshape(0) -> "cannot reshape array of size 6 into shape (0,)").
                    if (size != newShape.size)
                        throw ReshapeSizeMismatch(size, newShape.dimensions);
                }
            }

            // NumPy-aligned: Create new shape with preserved offset and bufferSize
            long bufSize = bufferSize > 0 ? bufferSize : size;

            // Handle scalar shape (null/empty dimensions from default constructor)
            var newDims = newShape.dimensions ?? Array.Empty<long>();
            var newStrides = newShape.strides ?? Array.Empty<long>();

            var result = new Shape(
                newDims.Length > 0 ? (long[])newDims.Clone() : newDims,
                newStrides.Length > 0 ? (long[])newStrides.Clone() : newStrides,
                offset,
                bufSize
            );

            // A contiguous reshape is a VIEW over the same memory, so it inherits writeability:
            // reshaping a read-only array (a broadcast, or an 'r' memmap) must stay read-only. The
            // fresh Shape above defaults WRITEABLE back to true. (The copy-forcing non-contiguous
            // path clones to fresh owned memory via Shape.Clean(), which correctly resets it.)
            if (!IsWriteable)
                result = result.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            return result;
        }

        /// <summary>
        ///     Changes the shape representing this storage (broadcast version).
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        private readonly Shape _reshapeBroadcast(Shape newShape, bool @unsafe = true)
        {
            // Handle -1 in reshape
            newShape = _inferMissingDimension(newShape);

            if (!@unsafe)
            {
                // Check if this is a scalar reshape (ndim=0 shape from default constructor or empty dims)
                bool isScalarShape = (newShape.dimensions == null || newShape.dimensions.Length == 0);

                if (isScalarShape)
                {
                    // Scalar shapes are valid only when reshaping from size 1.
                    // NumPy renders the empty shape as "()" (probed: np.zeros(3).reshape(())).
                    if (size != 1)
                        throw ReshapeSizeMismatch(size, Array.Empty<long>());
                }
                else
                {
                    if (size != newShape.size)
                        throw ReshapeSizeMismatch(size, newShape.dimensions);
                }
            }

            // NumPy-aligned: preserve bufferSize from original shape for broadcast tracking
            long bufSize = bufferSize > 0 ? bufferSize : size;

            // Handle scalar shape (null/empty dimensions from default constructor)
            var newDims = newShape.dimensions ?? Array.Empty<long>();
            var newStrides = newShape.strides ?? Array.Empty<long>();

            return new Shape(
                newDims.Length > 0 ? (long[])newDims.Clone() : newDims,
                newStrides.Length > 0 ? (long[])newStrides.Clone() : newStrides,
                0,
                bufSize
            );
        }

        /// <summary>
        ///     Renders a requested shape the way NumPy renders it inside a reshape error message.
        /// </summary>
        /// <remarks>
        ///     Port of <c>convert_shape_to_string</c> (numpy/_core/src/multiarray/common.c). Three
        ///     quirks are load-bearing, and all three are observable in NumPy 2.4.2's own texts:
        ///     <list type="bullet">
        ///     <item>LEADING unknown (negative) dims are dropped entirely — <c>reshape(-1, 0)</c>
        ///     reports <c>(0)</c>, not <c>(newaxis,0)</c>.</item>
        ///     <item>An unknown dim anywhere after the first printed one reads <c>newaxis</c> —
        ///     <c>reshape(0, -1)</c> reports <c>(0,newaxis)</c>.</item>
        ///     <item>A one-element shape closes with <c>,)</c> so it reads as a Python 1-tuple —
        ///     <c>reshape(0)</c> reports <c>(0,)</c>. That is why <c>(0)</c> and <c>(0,)</c> both
        ///     appear in NumPy's messages and mean different things.</item>
        ///     </list>
        ///     There are no spaces after the commas; NumPy builds this by concatenation, not by
        ///     formatting a Python tuple.
        /// </remarks>
        internal static string ConvertShapeToString(long[] vals)
        {
            int n = vals?.Length ?? 0;

            // Skip the leading "newaxis" run; if that consumes everything, the shape prints as ().
            int i = 0;
            while (i < n && vals[i] < 0)
                i++;

            if (i == n)
                return "()";

            var sb = new StringBuilder();
            sb.Append('(').Append(vals[i++].ToString(CultureInfo.InvariantCulture));
            for (; i < n; i++)
            {
                if (vals[i] < 0)
                    sb.Append(",newaxis");
                else
                    sb.Append(',').Append(vals[i].ToString(CultureInfo.InvariantCulture));
            }

            return sb.Append(n == 1 ? ",)" : ")").ToString();
        }

        /// <summary>
        ///     NumPy's single reshape failure, verbatim (<c>raise_reshape_size_mismatch</c>).
        /// </summary>
        /// <remarks>
        ///     NumPy answers every reshape rejection except the multiple-unknown one with this ONE
        ///     text. NumSharp keeps its long-standing <see cref="IncorrectShapeException"/> type
        ///     (the house convention — NumPy's <c>ValueError</c> maps to it across the shape APIs)
        ///     but the message is NumPy's, character for character.
        /// </remarks>
        private static IncorrectShapeException ReshapeSizeMismatch(long size, long[] requested)
            => new IncorrectShapeException($"cannot reshape array of size {size} into shape {ConvertShapeToString(requested)}");

        /// <summary>
        ///     Resolves an unknown (negative) dimension and validates the requested shape's size.
        /// </summary>
        /// <remarks>
        ///     Port of NumPy's <c>_fix_unknown_dimension</c> (numpy/_core/src/multiarray/shape.c).
        ///     Two things it does that the previous hand-rolled version did not:
        ///     <list type="bullet">
        ///     <item><b>ANY negative dimension is the unknown one</b>, not just <c>-1</c> — probed
        ///     on 2.4.2, <c>np.zeros(6).reshape(-3)</c> is <c>(6,)</c> and <c>reshape(3,-5)</c> is
        ///     <c>(3,2)</c>. Matching only <c>-1</c> let a second negative through as a literal
        ///     dimension, so <c>reshape(-1,-2)</c> silently produced the shape <c>(-3,-2)</c> —
        ///     a negative-extent array, not an error.</item>
        ///     <item><b>The zero product is checked BEFORE the division</b>, which is the whole
        ///     reason <c>np.zeros((0,3)).reshape(-1,0)</c> used to raise
        ///     <see cref="DivideByZeroException"/>: a degenerate known dimension makes the divisor
        ///     0. NumPy folds it into the ordinary size-mismatch test
        ///     (<c>s_known == 0 || s_original % s_known != 0</c>) and reports
        ///     <c>cannot reshape array of size 0 into shape (0)</c>.</item>
        ///     </list>
        /// </remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        private readonly Shape _inferMissingDimension(Shape shape)
        {
            // Handle uninitialized shape (from default constructor) or scalar shapes
            if (shape.dimensions == null || shape.dimensions.Length == 0)
                return shape;

            var dims = shape.dimensions;
            int n = dims.Length;
            var indexOfUnknown = -1;
            long product = 1;

            for (int i = 0; i < n; i++)
            {
                if (dims[i] < 0)
                {
                    if (indexOfUnknown != -1)
                        throw new ValueError("can only specify one unknown dimension");
                    indexOfUnknown = i;
                }
                else if (product != 0 && dims[i] > long.MaxValue / product)
                {
                    // The known dims alone overflow the element counter. NumPy reports this as an
                    // ordinary size mismatch rather than letting the wrapped product through as a
                    // plausible-looking (and much smaller) size.
                    throw ReshapeSizeMismatch(size, dims);
                }
                else
                {
                    product *= dims[i];
                }
            }

            if (indexOfUnknown == -1)
                return shape; // Nothing to infer; the caller's size check handles the rest.

            if (this.IsBroadcasted)
            {
                throw new NotSupportedException("Reshaping a broadcasted array with a -1 (unknown) dimension is not supported.");
            }

            if (product == 0 || size % product != 0)
                throw ReshapeSizeMismatch(size, dims);

            // Create new dimensions array with inferred value
            var newDims = (long[])dims.Clone();
            newDims[indexOfUnknown] = size / product;

            // Compute new strides for the corrected dimensions
            var newStrides = ComputeContiguousStrides(newDims);

            return new Shape(newDims, newStrides, 0, 0);
        }

        /// <summary>
        ///     Expands one or more axes with size-1 dimensions, matching NumPy's
        ///     <c>np.expand_dims(a, axis)</c> tuple-axis semantics.
        /// </summary>
        /// <remarks>
        ///     Each axis is normalized against the FINAL output ndim
        ///     (<c>inputNdim + axes.Length</c>). Duplicate normalized positions
        ///     raise <see cref="ArgumentException"/> ("repeated axis"), matching
        ///     NumPy's <c>ValueError</c>. Out-of-range axes throw
        ///     <see cref="ArgumentException"/>.
        /// </remarks>
        /// <param name="axes">Positions in the expanded output where size-1 axes are placed.</param>
        /// <returns>A new <see cref="Shape"/> aliasing the same storage with size-1 dims inserted.</returns>
        public readonly Shape ExpandDimensions(int[] axes)
        {
            if (axes == null || axes.Length == 0)
                return this;

            int inputNdim = dimensions?.Length ?? 0;
            int outNdim = inputNdim + axes.Length;

            // Normalize each axis against the OUTPUT ndim, mirroring NumPy.
            var normalized = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                int ax = axes[i];
                int adjusted = ax >= 0 ? ax : outNdim + ax;
                if (adjusted < 0 || adjusted >= outNdim)
                    throw new ArgumentException($"axis {ax} is out of bounds for array of dimension {outNdim}");
                normalized[i] = adjusted;
            }

            // Detect duplicates against normalized positions (NumPy: ValueError "repeated axis").
            var seen = new HashSet<int>();
            for (int i = 0; i < normalized.Length; i++)
            {
                if (!seen.Add(normalized[i]))
                    throw new ArgumentException("repeated axis");
            }

            // Apply axes in ascending order so each ExpandDimension call sees a
            // stable "earlier dim has already been inserted" view.
            var sorted = (int[])normalized.Clone();
            Array.Sort(sorted);

            Shape result = this;
            for (int i = 0; i < sorted.Length; i++)
                result = result.ExpandDimension(sorted[i]);

            return result;
        }

        /// <summary>
        ///     Expands a specific <paramref name="axis"/> with 1 dimension.
        /// </summary>
        /// <param name="axis"></param>
        /// <returns></returns>
        [SuppressMessage("ReSharper", "LocalVariableHidesMember")]
        public readonly Shape ExpandDimension(int axis)
        {
            long[] newDims;
            long[] newStrides;

            if (IsScalar)
            {
                newDims = new long[] { 1 };
                newStrides = new long[] { 0 };
            }
            else
            {
                newDims = (long[])dimensions.Clone();
                newStrides = (long[])strides.Clone();

                // Allow negative axis specification
                if (axis < 0)
                {
                    axis = dimensions.Length + 1 + axis;
                    if (axis < 0)
                        throw new ArgumentException($"Effective axis {axis} is less than 0");
                }

                Arrays.Insert(ref newDims, axis, 1L);

                // Calculate proper stride for C-contiguous layout
                long newStride;
                if (axis >= dimensions.Length)
                {
                    // Appending at the end - use 1 (innermost stride)
                    newStride = 1;
                }
                else
                {
                    // Inserting before existing dimension
                    newStride = dimensions[axis] * strides[axis];
                }
                Arrays.Insert(ref newStrides, axis, newStride);
            }

            // Create new shape with preserved bufferSize
            long bufSize = bufferSize > 0 ? bufferSize : size;
            return new Shape(newDims, newStrides, offset, bufSize);
        }

        /// <summary>
        ///     Prepends <paramref name="count"/> leading length-1 axes in ONE allocation —
        ///     the closed form of <see cref="ExpandDimension"/><c>(0)</c> applied
        ///     <paramref name="count"/> times.
        /// </summary>
        /// <remarks>
        ///     Repeating <see cref="ExpandDimension"/> is quadratic: every step clones both
        ///     <c>dimensions</c> and <c>strides</c> and inserts into them, so padding to
        ///     ndim=100 000 one axis at a time copies ~10^10 longs (measured 27.6 s).
        ///     The result has a closed form: prepending never touches an existing entry, and
        ///     every prepended axis takes stride <c>dimensions[0] * strides[0]</c> — the first
        ///     insert computes it from the source, and each later one recomputes
        ///     <c>1 * that</c>. A 0-d source degenerates to stride 0 through
        ///     <see cref="ExpandDimension"/>'s scalar branch. So this is bit-identical to the
        ///     loop it replaces, cached flags included, at O(ndim).
        /// </remarks>
        internal readonly Shape PrependDimensions(int count)
        {
            if (count <= 0)
                return this;

            int inputNdim = dimensions?.Length ?? 0;
            var newDims = new long[inputNdim + count];
            var newStrides = new long[inputNdim + count];

            long padStride = inputNdim == 0 ? 0L : dimensions[0] * strides[0];
            for (int i = 0; i < count; i++)
            {
                newDims[i] = 1L;
                newStrides[i] = padStride;
            }

            for (int i = 0; i < inputNdim; i++)
            {
                newDims[count + i] = dimensions[i];
                newStrides[count + i] = strides[i];
            }

            // Create new shape with preserved bufferSize
            long bufSize = bufferSize > 0 ? bufferSize : size;
            return new Shape(newDims, newStrides, offset, bufSize);
        }
    }
}
