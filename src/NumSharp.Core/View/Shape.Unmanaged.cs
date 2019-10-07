using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace NumSharp
{
    public partial struct Shape
    {
        /// <summary>
        ///     Get offset index out of coordinate indices.
        /// </summary>
        /// <param name="indices">A pointer to the coordinates to turn into linear offset</param>
        /// <param name="ndims">The number of dimensions</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        /// <remarks>Handles sliced indices and broadcasting</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public unsafe int GetOffset(int* indices, int ndims)
        {
            int offset;
            if (!IsSliced)
            {
                if (dimensions.Length == 0 && ndims == 1)
                    return indices[0];

                offset = 0;
                unchecked
                {
                    for (int i = 0; i < ndims; i++)
                        offset += strides[i] * indices[i];
                }

                if (IsBroadcasted)
                    return offset % BroadcastInfo.OriginalShape.size;

                return offset;
            }

            //if both sliced and broadcasted
            if (IsBroadcasted)
                return GetOffset_broadcasted(indices, ndims);

            // we are dealing with a slice

            var vi = ViewInfo;
            if (IsRecursive && vi.Slices == null)
            {
                // we are dealing with an unsliced recursively reshaped slice
                offset = GetOffset_IgnoreViewInfo(indices, ndims);
                var parent_coords = vi.ParentShape.GetCoordinates(offset, ignore_view_info: true);
                return vi.ParentShape.GetOffset(parent_coords);
            }

            var coords = new List<int>(ndims+10);
            for (int i = 0; i < ndims; i++)
                coords.Add( indices[i]);
            if (vi.UnreducedShape.IsScalar && ndims == 1 && indices[0] == 0 && !IsRecursive)
                return 0;
            if (ndims > vi.UnreducedShape.dimensions.Length)
                throw new ArgumentOutOfRangeException(nameof(indices), $"select has too many coordinates for this shape");
            var orig_ndim = vi.OriginalShape.NDim;
            if (orig_ndim > NDim && orig_ndim > ndims)
            {
                // fill in reduced dimensions in the provided coordinates 
                for (int i = 0; i < vi.OriginalShape.NDim; i++)
                {
                    var slice = ViewInfo.Slices[i];
                    if (slice.IsIndex)
                        coords.Insert(i, 0);
                    if (coords.Count == orig_ndim)
                        break;
                }
            }

            var orig_strides = vi.OriginalShape.strides;
            //var orig_dims = vi.OriginalShape.dimensions;
            offset = 0;
            unchecked
            {
                for (int i = 0; i < coords.Count; i++)
                {
                    // note: we can refrain from bounds checking here, because we should not allow negative indices at all, this should be checked higher up though.
                    //var coord = coords[i];
                    //var dim = orig_dims[i];
                    //if (coord < -dim || coord >= dim)
                    //    throw new ArgumentException($"index {coord} is out of bounds for axis {i} with a size of {dim}");
                    //if (coord < 0)
                    //    coord = dim + coord;
                    if (vi.Slices.Length <= i)
                    {
                        offset += orig_strides[i] * coords[i];
                        continue;
                    }

                    var slice = vi.Slices[i];
                    var start = slice.Start;
                    if (slice.IsIndex)
                        offset += orig_strides[i] * start; // the coord is irrelevant for index-slices (they are reduced dimensions)
                    else
                        offset += orig_strides[i] * (start + coords[i] * slice.Step);
                }
            }

            if (!IsRecursive)
                return offset;
            // we are dealing with a sliced recursively reshaped slice
            var parent_coords1 = vi.ParentShape.GetCoordinates(offset, ignore_view_info: true);
            return vi.ParentShape.GetOffset(parent_coords1);
        }

        /// <summary>
        ///     Get offset index out of coordinate indices.
        /// </summary>
        /// <param name="indices">The coordinates to turn into linear offset</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        /// <remarks>Handles sliced indices and broadcasting</remarks>
        [MethodImpl((MethodImplOptions)768)]
        private unsafe int GetOffset_broadcasted(int* indices, int ndims)
        {
            int offset;
            var vi = ViewInfo;
            var bi = BroadcastInfo;
            if (IsRecursive && vi.Slices == null)
            {
                // we are dealing with an unsliced recursively reshaped slice
                offset = GetOffset_IgnoreViewInfo(indices, ndims);
                var parent_coords = vi.ParentShape.GetCoordinates(offset, ignore_view_info: true);
                return vi.ParentShape.GetOffset(parent_coords);
            }

            var coords = new List<int>(ndims+10);
            for (int i = 0; i < ndims; i++)
                coords.Add(indices[i]);
            if (vi.UnreducedShape.IsScalar && ndims == 1 && indices[0] == 0 && !IsRecursive)
                return 0;
            if (ndims > vi.UnreducedShape.dimensions.Length)
                throw new ArgumentOutOfRangeException(nameof(indices), $"select has too many coordinates for this shape");
            var orig_ndim = vi.OriginalShape.NDim;
            if (orig_ndim > NDim && orig_ndim > ndims)
            {
                // fill in reduced dimensions in the provided coordinates 
                for (int i = 0; i < vi.OriginalShape.NDim; i++)
                {
                    var slice = ViewInfo.Slices[i];
                    if (slice.IsIndex)
                        coords.Insert(i, 0);
                    if (coords.Count == orig_ndim)
                        break;
                }
            }

            var orig_strides = vi.OriginalShape.strides;
            Shape unreducedBroadcasted;
            if (!bi.UnbroadcastShape.HasValue)
            {
                if (bi.OriginalShape.IsScalar)
                {
                    unreducedBroadcasted = vi.OriginalShape.Clone(true, false, false);
                    for (int i = 0; i < unreducedBroadcasted.NDim; i++)
                    {
                        unreducedBroadcasted.dimensions[i] = 1;
                        unreducedBroadcasted.strides[i] = 0;
                    }
                }
                else
                {
                    unreducedBroadcasted = vi.OriginalShape.Clone(true, false, false);
                    for (int i = Math.Abs(vi.OriginalShape.NDim - NDim), j = 0; i < unreducedBroadcasted.NDim; i++, j++)
                    {
                        if (strides[j] == 0)
                        {
                            unreducedBroadcasted.dimensions[i] = 1;
                            unreducedBroadcasted.strides[i] = 0;
                        }
                    }
                }

                bi.UnbroadcastShape = unreducedBroadcasted;
            }
            else
                unreducedBroadcasted = bi.UnbroadcastShape.Value;

            orig_strides = unreducedBroadcasted.strides;
            offset = 0;
            unchecked
            {
                for (int i = 0; i < coords.Count; i++)
                {
                    if (vi.Slices.Length <= i)
                    {
                        offset += orig_strides[i] * coords[i];
                        continue;
                    }

                    var slice = vi.Slices[i];
                    var start = slice.Start;
                    if (slice.IsIndex)
                        offset += orig_strides[i] * start; // the coord is irrelevant for index-slices (they are reduced dimensions)
                    else
                        offset += orig_strides[i] * (start + coords[i] * slice.Step);
                }
            }

            if (!IsRecursive)
                return offset;
            // we are dealing with a sliced recursively reshaped slice
            var parent_coords1 = vi.ParentShape.GetCoordinates(offset, ignore_view_info: true);
            return vi.ParentShape.GetOffset(parent_coords1);
        }

        /// <summary>
        /// Calculate the offset in an unsliced shape. If the shape is sliced, ignore the ViewInfo
        /// Note: to be used only inside of GetOffset()
        /// </summary>
        [MethodImpl((MethodImplOptions)768)]
        private unsafe int GetOffset_IgnoreViewInfo(int* indices, int ndims)
        {
            if (dimensions.Length == 0 && ndims == 1)
                return indices[0];

            int offset = 0;
            unchecked
            {
                for (int i = 0; i < ndims; i++)
                    offset += strides[i] * indices[i];
            }

            if (IsBroadcasted)
                return offset % BroadcastInfo.OriginalShape.size;

            return offset;
        }
    }
}
