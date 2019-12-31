using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
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
                var parent_coords = vi.ParentShape.GetCoordinates(offset);
                return vi.ParentShape.GetOffset(parent_coords);
            }

            var coords = new List<int>(ndims + 10);
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
            var parent_coords1 = vi.ParentShape.GetCoordinates(offset);
            return vi.ParentShape.GetOffset(parent_coords1);
        }

        /// <summary>
        ///     Gets the shape based on given <see cref="indicies"/> and the index offset (C-Contiguous) inside the current storage.
        /// </summary>
        /// <param name="indicies">The selection of indexes 0 based.</param>
        /// <returns></returns>
        /// <remarks>Used for slicing, returned shape is the new shape of the slice and offset is the offset from current address.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public unsafe (Shape Shape, int Offset) GetSubshape(int* dims, int ndims)
        {
            if (ndims == 0)
                return (this, 0);

            int offset;
            var dim = ndims;
            var newNDim = dimensions.Length - dim;
            if (IsBroadcasted)
            {
                var dimsClone = stackalloc int[ndims];
                for (int j = 0; j < ndims; j++)
                {
                    dimsClone[j] = dims[j];
                }

                Shape unreducedBroadcasted;
                if (!BroadcastInfo.UnreducedBroadcastedShape.HasValue)
                {
                    unreducedBroadcasted = this.Clone(true, false, false);
                    for (int i = 0; i < unreducedBroadcasted.NDim; i++)
                    {
                        if (unreducedBroadcasted.strides[i] == 0)
                            unreducedBroadcasted.dimensions[i] = 1;
                    }

                    BroadcastInfo.UnreducedBroadcastedShape = unreducedBroadcasted;
                }
                else
                    unreducedBroadcasted = BroadcastInfo.UnreducedBroadcastedShape.Value;

                //unbroadcast indices
                for (int i = 0; i < dim; i++)
                    dimsClone[i] = dimsClone[i] % unreducedBroadcasted[i];

                offset = unreducedBroadcasted.GetOffset(dimsClone, ndims);

                var retShape = new int[newNDim];
                var strides = new int[newNDim];
                var original = new int[newNDim];
                var original_strides = new int[newNDim];
                for (int i = 0; i < newNDim; i++)
                {
                    retShape[i] = this.dimensions[dim + i];
                    strides[i] = this.strides[dim + i];
                    original[i] = unreducedBroadcasted[dim + i];
                    original_strides[i] = unreducedBroadcasted.strides[dim + i];
                }

                return (new Shape(retShape, strides, new Shape(original, original_strides)), offset);
            }

            //compute offset
            offset = GetOffset(dims, ndims);

            var orig_shape = IsSliced ? ViewInfo.OriginalShape : this;
            if (offset >= orig_shape.Size)
                throw new IndexOutOfRangeException($"The offset {offset} is out of range in Shape {orig_shape.Size}");

            if (ndims == dimensions.Length)
                return (Scalar, offset);

            //compute subshape
            var innerShape = new int[newNDim];
            for (int i = 0; i < innerShape.Length; i++)
                innerShape[i] = this.dimensions[dim + i];

            //TODO! This is not full support of sliced,
            //TODO! when sliced it usually diverts from this function but it would be better if we add support for sliced arrays too.
            return (new Shape(innerShape), offset);
        }


        /// <summary>
        ///     Translates coordinates with negative indices, e.g:<br></br>
        ///     np.arange(9)[-1] == np.arange(9)[8]<br></br>
        ///     np.arange(9)[-2] == np.arange(9)[7]<br></br>
        /// </summary>
        /// <param name="dimensions">The dimensions these coordinates are targeting</param>
        /// <param name="coords">The coordinates.</param>
        /// <returns>Coordinates without negative indices.</returns>
        [SuppressMessage("ReSharper", "ParameterHidesMember"), MethodImpl((MethodImplOptions)512)]
        public static unsafe void InferNegativeCoordinates(int[] dimensions, int* coords, int coordsCount)
        {
            for (int i = 0; i < coordsCount; i++)
            {
                var curr = coords[i];
                if (curr < 0)
                    coords[i] = dimensions[i] + curr;
            }
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
                var parent_coords = vi.ParentShape.GetCoordinates(offset);
                return vi.ParentShape.GetOffset(parent_coords);
            }

            var coords = new List<int>(ndims + 10);
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
            if (!bi.UnreducedBroadcastedShape.HasValue)
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

                bi.UnreducedBroadcastedShape = unreducedBroadcasted;
            }
            else
                unreducedBroadcasted = bi.UnreducedBroadcastedShape.Value;

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
            var parent_coords1 = vi.ParentShape.GetCoordinates(offset);
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
