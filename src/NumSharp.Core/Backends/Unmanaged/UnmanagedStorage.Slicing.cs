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
            // For contiguous shapes, if all slices are indices, we can use GetData to get
            // a direct memory slice. For non-contiguous shapes (stepped, transposed), we
            // must go through the full slice path to create a proper view.
            if (_shape.IsContiguous)
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

            // In case the slices selected are all ":" - just return an alias
            if (slices.All(s => Equals(Slice.All, s)))
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

            // NumPy-aligned: All slices return views (aliases) that share memory with the original.
            // The slicedShape contains the correct offset and strides computed by Shape.Slice().
            // Views with non-zero offset or non-standard strides use coordinate-based access.
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
