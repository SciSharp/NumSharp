using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Slicing

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UnmanagedStorage GetView(string slicing_notation) => GetView(Slice.ParseSlices(slicing_notation));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UnmanagedStorage GetView(params Slice[] slices)
        {
            if (slices == null)
                throw new ArgumentNullException(nameof(slices));
            // deal with ellipsis and newaxis if any before continuing into GetViewInternal
            int ellipsis_count = 0;
            int newaxis_count = 0;
            foreach (var slice in slices)
            {
                if (slice.IsEllipsis)
                    ellipsis_count++;
                if (slice.IsNewAxis)
                    newaxis_count++;
            }

            // deal with ellipsis
            if (ellipsis_count > 1)
                throw new ArgumentException("IndexError: an index can only have a single ellipsis ('...')");
            else if (ellipsis_count == 1)
                slices = ExpandEllipsis(slices).ToArray();

            // deal with newaxis
            if (newaxis_count > 0)
            {
                var view = this;
                for (var axis = 0; axis < slices.Length; axis++)
                {
                    var slice = slices[axis];
                    if (slice.IsNewAxis)
                    {
                        slices[axis] = Slice.All;
                        view = view.Alias(view.Shape.ExpandDimension(axis));
                    }
                }

                return view.GetViewInternal(slices);
            }

            // slicing without newaxis
            return GetViewInternal(slices);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [SuppressMessage("ReSharper", "PossibleInvalidOperationException")]
        private UnmanagedStorage GetViewInternal(params Slice[] slices)
        {
            // NOTE: GetViewInternal can not deal with Slice.Ellipsis or Slice.NewAxis! 
            //handle memory slice if possible
            if (!_shape.IsSliced)
            {
                var indices = new int[slices.Length];
                for (var i = 0; i < slices.Length; i++)
                {
                    var inputSlice = slices[i];
                    if (!inputSlice.IsIndex)
                    {
                        //incase it is a trailing :, e.g. [2,2, :] in a shape (3,3,5,5) -> (5,5)
                        if (i == slices.Length - 1 && inputSlice == Slice.All)
                        {
                            Array.Resize(ref indices, indices.Length - 1);
                            goto _getdata;
                        }

                        goto _perform_slice;
                    }

                    indices[i] = inputSlice.Start.Value;
                }

                _getdata:
                return GetData(indices);
            }

            //perform a regular slicing
            _perform_slice:

            // In case the slices selected are all ":"
            // ReSharper disable once ConvertIfStatementToReturnStatement
            if (!_shape.IsRecursive && slices.All(s => Equals(Slice.All, s)))
                return Alias();

            //handle broadcasted shape: materialize broadcast data into contiguous memory,
            //then slice the contiguous result. We must use a Clean() shape (not the broadcast
            //shape) so that strides match the contiguous data layout. Using _shape.Slice(slices)
            //would attach broadcast strides [1,0] to contiguous data [3,1], causing wrong offsets.
            if (_shape.IsBroadcasted)
            {
                var clonedData = CloneData();
                var cleanShape = _shape.Clean();
                return new UnmanagedStorage(clonedData, cleanShape).GetViewInternal(slices);
            }

            var slicedShape = _shape.Slice(slices);

            // Contiguous slice optimization:
            // If the slice describes a contiguous block of memory (step=1 for the
            // first partial dimension, all trailing dims fully taken), create an
            // offset InternalArray slice instead of a ViewInfo-based alias.
            // This makes Address point to the correct location, enabling:
            // - IsContiguous=true for the result
            // - Fast-path iteration in ToArray, ravel, copyto, etc.
            // - Proper view semantics (shares memory with original)
            //
            // This matches NumPy's architecture where slicing adjusts the data pointer.
            if (!slicedShape.IsRecursive && slicedShape.ViewInfo?.Slices != null)
            {
                var vi = slicedShape.ViewInfo;
                var origDims = vi.OriginalShape.dimensions;
                var origStrides = vi.OriginalShape.strides;
                var sdefs = vi.Slices;

                // Check contiguity: scan right-to-left.
                // Trailing dims must be fully taken (Start=0, Step=1, Count=origDim).
                // First partially-taken dim must have Step=1 (or Count<=1).
                // All dims left of that must have Count=1 (or be an index).
                bool contiguous = true;
                bool foundPartial = false;
                for (int i = sdefs.Length - 1; i >= 0; i--)
                {
                    var sd = sdefs[i];
                    int count = sd.IsIndex ? 1 : sd.Count;
                    bool isFull = !sd.IsIndex && sd.Start == 0 && sd.Step == 1 && sd.Count == origDims[i];

                    if (!foundPartial)
                    {
                        if (isFull) continue;
                        foundPartial = true;
                        if (sd.Step != 1 && count > 1) { contiguous = false; break; }
                    }
                    else
                    {
                        if (count != 1) { contiguous = false; break; }
                    }
                }

                if (contiguous && slicedShape.size > 0)
                {
                    // Compute linear start offset in the underlying InternalArray
                    int startOffset = 0;
                    for (int i = 0; i < sdefs.Length; i++)
                        startOffset += origStrides[i] * sdefs[i].Start;

                    // Create a clean shape (no ViewInfo) with the sliced dimensions
                    // This makes IsContiguous=true because there's no ViewInfo
                    var cleanShape = slicedShape.Clean();
                    return new UnmanagedStorage(InternalArray.Slice(startOffset, cleanShape.size), cleanShape);
                }
            }

            return Alias(slicedShape);
        }

        private IEnumerable<Slice> ExpandEllipsis(Slice[] slices)
        {
            // count dimensions without counting ellipsis or newaxis
            var count = 0;
            foreach (var slice in slices)
            {
                if (slice.IsNewAxis || slice.IsEllipsis)
                    continue;
                count++;
            }

            // expand 
            foreach (var slice in slices)
            {
                if (slice.IsEllipsis)
                {
                    for (int i = 0; i < Shape.NDim - count; i++)
                        yield return Slice.All;
                    continue;
                }

                yield return slice;
            }
        }

        #endregion
    }
}
