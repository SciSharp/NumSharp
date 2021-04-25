using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     Represents a shape of an N-D array.
    /// </summary>
    /// <remarks>Handles slicing, indexing based on coordinates or linear offset and broadcastted indexing.</remarks>
    public partial struct Shape : ICloneable, IEquatable<Shape>
    {
        internal ViewInfo ViewInfo;
        internal BroadcastInfo BroadcastInfo;

        /// <summary>
        ///     Does this Shape have modified strides, usually in scenarios like np.transpose.
        /// </summary>
        public bool ModifiedStrides;

        /// <summary>
        ///     True if the shape of this array was obtained by a slicing operation that caused the underlying data to be non-contiguous
        /// </summary>
        public readonly bool IsSliced
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ViewInfo != null;
        }

        /// <summary>
        ///     Does this Shape represents a non-sliced and non-broadcasted hence contagious unmanaged memory?
        /// </summary>
        public readonly bool IsContiguous
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => !IsSliced && !IsBroadcasted;
        }

        /// <summary>
        ///     Is this Shape a recusive view? (deeper than 1 view)
        /// </summary>
        public readonly bool IsRecursive
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ViewInfo != null && ViewInfo.ParentShape.IsEmpty == false;
        }

        /// <summary>
        ///     Dense data are stored contiguously in memory, addressed by a single index (the memory address). <br></br>
        ///     Array memory ordering schemes translate that single index into multiple indices corresponding to the array coordinates.<br></br>
        ///     0: Row major<br></br>
        ///     1: Column major
        /// </summary>
        internal const char layout = 'C';

        internal int _hashCode;
        internal int size;
        internal int[] dimensions;
        internal int[] strides;

        /// <summary>
        ///     Is this shape a broadcast and/or has modified strides?
        /// </summary>
        public readonly bool IsBroadcasted => BroadcastInfo != null;

        /// <summary>
        ///     Is this shape a scalar? (<see cref="NDim"/>==0 && <see cref="size"/> == 1)
        /// </summary>
        public bool IsScalar;

        /// <summary>
        /// True if the shape is not initialized.
        /// Note: A scalar shape is not empty.
        /// </summary>
        public readonly bool IsEmpty => _hashCode == 0;

        public readonly char Order => layout;

        /// <summary>
        ///     Singleton instance of a <see cref="Shape"/> that represents a scalar.
        /// </summary>
        public static readonly Shape Scalar = new Shape(new int[0]);

        /// <summary>
        ///     Create a new scalar shape
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Shape NewScalar() =>
            new Shape(new int[0]);

        /// <summary>
        ///     Create a new scalar shape
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Shape NewScalar(ViewInfo viewInfo) =>
            new Shape(new int[0]) {ViewInfo = viewInfo};

        /// <summary>
        ///     Create a new scalar shape
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Shape NewScalar(ViewInfo viewInfo, BroadcastInfo broadcastInfo) =>
            new Shape(new int[0]) {ViewInfo = viewInfo, BroadcastInfo = broadcastInfo};

        /// <summary>
        ///     Create a shape that represents a vector.
        /// </summary>
        /// <remarks>Faster than calling Shape's constructor</remarks>
        public static Shape Vector(int length)
        {
            var shape = new Shape {dimensions = new int[] {length}, strides = new int[] {1}, size = length};
            shape._hashCode = ( /*shape.layout*/ layout * 397) ^ (length * 397) * (length * 397);
            return shape;
        }

        /// <summary>
        ///     Create a shape that represents a vector.
        /// </summary>
        /// <remarks>Faster than calling Shape's constructor</remarks>
        public static Shape Vector(int length, ViewInfo viewInfo)
        {
            var shape = new Shape
            {
                dimensions = new[] {length},
                strides = new int[] {1},
                //layout = 'C',
                size = length,
                ViewInfo = viewInfo
            };

            shape._hashCode = ( /*shape.layout*/ layout * 397) ^ (length * 397) * (length * 397);
            return shape;
        }

        /// <summary>
        ///     Create a shape that represents a matrix.
        /// </summary>
        /// <remarks>Faster than calling Shape's constructor</remarks>
        public static Shape Matrix(int rows, int cols)
        {
            var shape = new Shape {dimensions = new[] {rows, cols}, strides = new int[] {cols, 1}, size = rows * cols};

            unchecked
            {
                int hash = ( /*shape.layout*/ layout * 397);
                int size = 1;
                foreach (var v in shape.dimensions)
                {
                    size *= v;
                    hash ^= (size * 397) * (v * 397);
                }

                shape._hashCode = hash;
            }

            shape.IsScalar = false;
            return shape;
        }

        public readonly int NDim
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => dimensions.Length;
        }

        public readonly int[] Dimensions
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => dimensions;
        }

        public readonly int[] Strides
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => strides;
        }

        /// <summary>
        ///     The linear size of this shape.
        /// </summary>
        public readonly int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => size;
        }

        public Shape(Shape other)
        {
            if (other.IsEmpty)
            {
                this = default;
                return;
            }

            //this.layout = other.layout;
            this._hashCode = other._hashCode;
            this.size = other.size;
            this.dimensions = (int[])other.dimensions.Clone();
            this.strides = (int[])other.strides.Clone();
            this.IsScalar = other.IsScalar;
            this.ViewInfo = other.ViewInfo?.Clone();
            this.BroadcastInfo = other.BroadcastInfo?.Clone();
            this.ModifiedStrides = other.ModifiedStrides;
        }

        public Shape(int[] dims, int[] strides)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            if (strides == null)
                throw new ArgumentNullException(nameof(strides));

            if (dims.Length != strides.Length)
                throw new ArgumentException($"While trying to construct a shape, given dimensions and strides does not match size ({dims.Length} != {strides.Length})");

            //layout = 'C';
            size = 1;
            unchecked
            {
                //calculate hash and size
                if (dims.Length > 0)
                {
                    int hash = (layout * 397);
                    foreach (var v in dims)
                    {
                        size *= v;
                        hash ^= (size * 397) * (v * 397);
                    }

                    _hashCode = hash;
                }
                else
                    _hashCode = 0;
            }

            this.strides = strides;
            this.dimensions = dims;
            IsScalar = size == 1 && dims.Length == 0;
            ViewInfo = null;
            BroadcastInfo = null;
            ModifiedStrides = false;
        }

        public Shape(int[] dims, int[] strides, Shape originalShape)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            if (strides == null)
                throw new ArgumentNullException(nameof(strides));

            if (dims.Length != strides.Length)
                throw new ArgumentException($"While trying to construct a shape, given dimensions and strides does not match size ({dims.Length} != {strides.Length})");

            //layout = 'C';
            size = 1;
            unchecked
            {
                //calculate hash and size
                if (dims.Length > 0)
                {
                    int hash = (layout * 397);
                    foreach (var v in dims)
                    {
                        size *= v;
                        hash ^= (size * 397) * (v * 397);
                    }

                    _hashCode = hash;
                }
                else
                    _hashCode = 0;
            }

            this.strides = strides;
            this.dimensions = dims;
            IsScalar = size == 1 && dims.Length == 0;
            ViewInfo = null;
            BroadcastInfo = new BroadcastInfo() {OriginalShape = originalShape};
            ModifiedStrides = false;
        }

        [MethodImpl((MethodImplOptions)512)]
        public Shape(params int[] dims)
        {
            if (dims == null)
            {
                strides = dims = dimensions = new int[0];
            }
            else
            {
                dimensions = dims;
                strides = new int[dims.Length];
            }

            unchecked
            {
                size = 1;
                //layout = 'C';
                if (dims.Length > 0)
                {
                    int hash = (layout * 397);
                    foreach (var v in dims)
                    {
                        size *= v;
                        hash ^= (size * 397) * (v * 397);
                    }

                    _hashCode = hash;
                }
                else
                    _hashCode = int.MinValue; //scalar's hashcode is int.minvalue

                if (dims.Length != 0)
                {
                    strides[strides.Length - 1] = 1;
                    for (int i = strides.Length - 1; i >= 1; i--)
                        strides[i - 1] = strides[i] * dims[i];
                }
            }

            IsScalar = _hashCode == int.MinValue;
            ViewInfo = null;
            BroadcastInfo = null;
            ModifiedStrides = false;
        }

        /// <summary>
        ///     An empty shape without any fields set except all are default.
        /// </summary>
        /// <remarks>Used internally.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static Shape Empty(int ndim)
        {
            return new Shape {dimensions = new int[ndim], strides = new int[ndim]};
            //default vals already sets: ret.layout = 0;
            //default vals already sets: ret.size = 0;
            //default vals already sets: ret._hashCode = 0;
            //default vals already sets: ret.IsScalar = false;
            //default vals already sets: ret.ViewInfo = null;
        }


        [MethodImpl((MethodImplOptions)768)]
        private readonly void _computeStrides()
        {
            if (dimensions.Length == 0)
                return;

            unchecked
            {
                strides[strides.Length - 1] = 1;
                for (int idx = strides.Length - 1; idx >= 1; idx--)
                    strides[idx - 1] = strides[idx] * dimensions[idx];
            }
        }


        [MethodImpl((MethodImplOptions)768)]
        private readonly void _computeStrides(int axis)
        {
            if (dimensions.Length == 0)
                return;
            
            if (axis == 0)
                strides[0] = strides[1] * dimensions[1];
            else
                unchecked
                {
                    if (axis == strides.Length - 1)
                        strides[strides.Length - 1] = 1;
                    else
                        strides[axis - 1] = strides[axis] * dimensions[axis];
                }
        }

        public readonly int this[int dim]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => dimensions[dim < 0 ? dimensions.Length + dim : dim];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => dimensions[dim < 0 ? dimensions.Length + dim : dim] = value;
        }

        /// <summary>
        ///     Retrieve the transformed offset if <see cref="IsSliced"/> is true, otherwise returns <paramref name="offset"/>.
        /// </summary>
        /// <param name="offset">The offset within the bounds of <see cref="size"/>.</param>
        /// <returns>The transformed offset.</returns>
        /// <remarks>Avoid using unless it is unclear if shape is sliced or not.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly int TransformOffset(int offset)
        {
            // ReSharper disable once ConvertIfStatementToReturnStatement
            if (ViewInfo == null && BroadcastInfo == null)
                return offset;

            return GetOffset(GetCoordinates(offset));
        }

        /// <summary>
        ///     Get offset index out of coordinate indices.
        ///
        ///     The offset is the absolute offset in memory for the given coordinates.
        ///     Even for shapes that were sliced and reshaped after slicing and sliced again (and so forth)
        ///     this returns the absolute memory offset.
        ///
        ///     Note: the inverse operation to this is GetCoordinatesFromAbsoluteIndex
        /// </summary>
        /// <param name="indices">The coordinates to turn into linear offset</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        /// <remarks>Handles sliced indices and broadcasting</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public readonly int GetOffset(params int[] indices)
        {
            int offset;
            if (!IsSliced)
            {
                if (dimensions.Length == 0 && indices.Length == 1)
                    return indices[0];

                offset = 0;
                unchecked
                {
                    for (int i = 0; i < indices.Length; i++)
                        offset += strides[i] * indices[i];
                }

                if (IsBroadcasted)
                    return offset % BroadcastInfo.OriginalShape.size;

                return offset;
            }

            //if both sliced and broadcasted
            if (IsBroadcasted)
                return GetOffset_broadcasted(indices);

            // we are dealing with a slice

            var vi = ViewInfo;
            if (IsRecursive && vi.Slices == null)
            {
                // we are dealing with an unsliced recursively reshaped slice
                offset = GetOffset_IgnoreViewInfo(indices);
                var parent_coords = vi.ParentShape.GetCoordinates(offset);
                return vi.ParentShape.GetOffset(parent_coords);
            }

            var coords = new List<int>(indices);
            if (vi.UnreducedShape.IsScalar && indices.Length == 1 && indices[0] == 0 && !IsRecursive)
                return 0;
            if (indices.Length > vi.UnreducedShape.dimensions.Length)
                throw new ArgumentOutOfRangeException(nameof(indices), $"select has too many coordinates for this shape");
            var orig_ndim = vi.OriginalShape.NDim;
            if (orig_ndim > NDim && orig_ndim > indices.Length)
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
        ///     Get offset index out of coordinate indices.
        /// </summary>
        /// <param name="index">The coordinates to turn into linear offset</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        /// <remarks>Handles sliced indices and broadcasting</remarks>
        [MethodImpl((MethodImplOptions)768)]
        internal readonly int GetOffset_1D(int index)
        {
            int offset;
            if (!IsSliced)
            {
                if (dimensions.Length == 0)
                    return index;

                offset = 0;
                unchecked
                {
                    offset += strides[0] * index;
                }

                if (IsBroadcasted)
                    return offset % BroadcastInfo.OriginalShape.size;

                return offset;
            }

            //if both sliced and broadcasted
            if (IsBroadcasted)
                return GetOffset_broadcasted_1D(index);

            // we are dealing with a slice

            var vi = ViewInfo;
            if (IsRecursive && vi.Slices == null)
            {
                // we are dealing with an unsliced recursively reshaped slice
                offset = GetOffset_IgnoreViewInfo(index);
                var parent_coords = vi.ParentShape.GetCoordinates(offset);
                return vi.ParentShape.GetOffset(parent_coords);
            }

            var coords = new List<int>(1) {index};
            if (vi.UnreducedShape.IsScalar && index == 0 && !IsRecursive)
                return 0;
            if (1 > vi.UnreducedShape.dimensions.Length)
                throw new ArgumentOutOfRangeException(nameof(index), $"select has too many coordinates for this shape");
            var orig_ndim = vi.OriginalShape.NDim;
            if (orig_ndim > NDim && orig_ndim > 1)
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
        /// Calculate the offset in an unsliced shape. If the shape is sliced, ignore the ViewInfo
        /// Note: to be used only inside of GetOffset()
        /// </summary>
        [MethodImpl((MethodImplOptions)768)]
        private readonly int GetOffset_IgnoreViewInfo(params int[] indices)
        {
            if (dimensions.Length == 0 && indices.Length == 1)
                return indices[0];

            int offset = 0;
            unchecked
            {
                for (int i = 0; i < indices.Length; i++)
                    offset += strides[i] * indices[i];
            }

            if (IsBroadcasted)
                return offset % BroadcastInfo.OriginalShape.size;

            return offset;
        }

        /// <summary>
        ///     Get offset index out of coordinate indices.
        /// </summary>
        /// <param name="indices">The coordinates to turn into linear offset</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        /// <remarks>Handles sliced indices and broadcasting</remarks>
        [MethodImpl((MethodImplOptions)768)]
        private readonly int GetOffset_broadcasted(params int[] indices)
        {
            int offset;
            var vi = ViewInfo;
            var bi = BroadcastInfo;
            if (IsRecursive && vi.Slices == null)
            {
                // we are dealing with an unsliced recursively reshaped slice
                offset = GetOffset_IgnoreViewInfo(indices);
                var parent_coords = vi.ParentShape.GetCoordinates(offset);
                return vi.ParentShape.GetOffset(parent_coords);
            }

            var coords = new List<int>(indices);
            if (vi.UnreducedShape.IsScalar && indices.Length == 1 && indices[0] == 0 && !IsRecursive)
                return 0;
            if (indices.Length > vi.UnreducedShape.dimensions.Length)
                throw new ArgumentOutOfRangeException(nameof(indices), $"select has too many coordinates for this shape");
            var orig_ndim = vi.OriginalShape.NDim;
            if (orig_ndim > NDim && orig_ndim > indices.Length)
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
            Shape unreducedBroadcasted = resolveUnreducedBroadcastedShape();

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
        ///     Get offset index out of coordinate indices.
        /// </summary>
        /// <param name="index">The coordinates to turn into linear offset</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        /// <remarks>Handles sliced indices and broadcasting</remarks>
        [MethodImpl((MethodImplOptions)768)]
        private readonly int GetOffset_broadcasted_1D(int index)
        {
            int offset;
            var vi = ViewInfo;
            var bi = BroadcastInfo;
            if (IsRecursive && vi.Slices == null)
            {
                // we are dealing with an unsliced recursively reshaped slice
                offset = GetOffset_IgnoreViewInfo(index);
                var parent_coords = vi.ParentShape.GetCoordinates(offset);
                return vi.ParentShape.GetOffset(parent_coords);
            }

            var coords = new List<int>(1) {index};
            if (vi.UnreducedShape.IsScalar && index == 0 && !IsRecursive)
                return 0;
            if (1 > vi.UnreducedShape.dimensions.Length)
                throw new ArgumentOutOfRangeException(nameof(index), $"select has too many coordinates for this shape");
            var orig_ndim = vi.OriginalShape.NDim;
            if (orig_ndim > NDim && orig_ndim > 1)
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
            Shape unreducedBroadcasted = resolveUnreducedBroadcastedShape();

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
        ///     Gets the shape based on given <see cref="indicies"/> and the index offset (C-Contiguous) inside the current storage.
        /// </summary>
        /// <param name="indicies">The selection of indexes 0 based.</param>
        /// <returns></returns>
        /// <remarks>Used for slicing, returned shape is the new shape of the slice and offset is the offset from current address.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public readonly (Shape Shape, int Offset) GetSubshape(params int[] indicies)
        {
            if (indicies.Length == 0)
                return (this, 0);

            int offset;
            var dim = indicies.Length;
            var newNDim = dimensions.Length - dim;
            if (IsBroadcasted)
            {
                indicies = (int[])indicies.Clone(); //we must copy because we make changes to it.
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
                    indicies[i] = indicies[i] % unreducedBroadcasted[i];

                offset = unreducedBroadcasted.GetOffset(indicies);

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
            offset = GetOffset(indicies);

            var orig_shape = IsSliced ? ViewInfo.OriginalShape : this;
            if (offset >= orig_shape.Size)
                throw new IndexOutOfRangeException($"The offset {offset} is out of range in Shape {orig_shape.Size}");

            if (indicies.Length == dimensions.Length)
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
        ///  Gets coordinates in this shape from index in this shape (slicing is ignored).
        ///  Example: Shape (2,3)
        /// 0 => [0, 0]
        /// 1 => [0, 1]
        /// ...
        /// 6 => [1, 2]
        /// </summary>
        /// <param name="offset">the index if you would iterate from 0 to shape.size in row major order</param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public readonly int[] GetCoordinates(int offset)
        {
            int[] coords = null;

            if (strides.Length == 1)
                coords = new int[] {offset};

            int counter = offset;
            coords = new int[strides.Length];
            int stride;
            for (int i = 0; i < strides.Length; i++)
            {
                unchecked
                {
                    stride = strides[i];
                    if (stride == 0)
                    {
                        coords[i] = 0;
                    }
                    else
                    {
                        coords[i] = counter / stride;
                        counter -= coords[i] * stride;
                    }
                }
            }

            return coords;
        }

        /// <summary>
        ///     Retrievs the coordinates in current shape (potentially sliced and reshaped) from index in original array.<br></br>
        ///     Note: this is the inverse operation of GetOffset<br></br>
        ///     Example: Shape a (2,3) => sliced to b (2,2) by a[:, 1:]<br></br>
        ///     The absolute indices in a are:<br></br>
        ///     [0, 1, 2,<br></br>
        ///      3, 4, 5]<br></br>
        ///     The absolute indices in b are:<br></br>
        ///     [1, 2,<br></br>
        ///      4, 5]<br></br>
        ///     <br></br>
        ///     <br></br>
        ///     Examples:<br></br>
        ///     a.GetCoordinatesFromAbsoluteIndex(1) returns [0, 1]<br></br>
        ///     b.GetCoordinatesFromAbsoluteIndex(0) returns [0, 0]<br></br>
        ///     b.GetCoordinatesFromAbsoluteIndex(0) returns [] because it is out of shape<br></br>
        /// </summary>
        /// <param name="offset">Is the index in the original array before it was sliced and/or reshaped</param>
        /// <remarks>Note: due to slicing the absolute indices (offset in memory) are different from what GetCoordinates would return, which are relative indices in the shape.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public readonly int[] GetCoordinatesFromAbsoluteIndex(int offset)
        {
            if (!IsSliced)
                return GetCoordinates(offset);

            // handle sliced shape
            int[] parent_coords = null;
            if (IsRecursive)
            {
                var parent = ViewInfo.ParentShape;
                var unreshaped_parent_coords = parent.GetCoordinatesFromAbsoluteIndex(offset);
                var parent_shape_offset = parent.GetOffset_IgnoreViewInfo(unreshaped_parent_coords);
                var orig_shape = ViewInfo.OriginalShape.IsEmpty ? this : ViewInfo.OriginalShape;
                parent_coords = orig_shape.GetCoordinates(parent_shape_offset);
            }
            else
                parent_coords = ViewInfo.OriginalShape.GetCoordinates(offset);

            if (ViewInfo.Slices == null)
                return parent_coords;
            return ReplaySlicingOnCoords(parent_coords, ViewInfo.Slices);
        }

        [MethodImpl((MethodImplOptions)768)]
        private int[] ReplaySlicingOnCoords(int[] parentCoords, SliceDef[] slices)
        {
            var coords = new List<int>();
            for (int i = 0; i < parentCoords.Length; i++)
            {
                var slice = slices[i];
                var coord = parentCoords[i];
                if (slice.Count == -1) // this is a Slice.Index so we remove this dim from coords
                    continue;
                if (slice.Count == 0) // this is a Slice.None which means there is no set of coordinates that can index anything in this shape
                    return new int[0];
                if (slice.Start > coord && slice.Step > 0 || slice.Start < coord && slice.Step < 0) // outside of the slice, return empty coords
                    return new int[0];
                if (coord % Math.Abs(slice.Step) != 0) // coord is between the steps, so we are "outside" of this shape, return empty coords
                    return new int[0];
                coords.Add((coord - slice.Start) / slice.Step);
            }

            return coords.ToArray();
        }

        [MethodImpl((MethodImplOptions)768)]
        public void ChangeTensorLayout(char order = 'C')
        {
            return; //currently this does nothing.
            //layout = order;
            _computeStrides();
            ComputeHashcode();
        }

        [MethodImpl((MethodImplOptions)768)]
        public static int GetSize(int[] dims)
        {
            int size = 1;
            unchecked
            {
                for (int i = 0; i < dims.Length; i++)
                    size *= dims[i];
            }

            return size;
        }

        public static int[] GetAxis(ref Shape shape, int axis)
        {
            return GetAxis(shape.dimensions, axis);
        }

        public static int[] GetAxis(Shape shape, int axis)
        {
            return GetAxis(shape.dimensions, axis);
        }

        public static int[] GetAxis(int[] dims, int axis)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            if (dims.Length == 0)
                return new int[0];

            if (axis <= -1) axis = dims.Length - 1;
            if (axis >= dims.Length)
                throw new AxisOutOfRangeException(dims.Length, axis);

            return dims.RemoveAt(axis);
        }

        /// <summary>
        ///     Extracts the shape of given <paramref name="array"/>.
        /// </summary>
        /// <remarks>Supports both jagged and multi-dim.</remarks>
        [MethodImpl((MethodImplOptions)512)]
        public static int[] ExtractShape(Array array)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            bool isJagged = false;

            {
                var type = array.GetType();
                isJagged = array.Rank == 1 && type.IsArray && type.GetElementType().IsArray;
            }

            var l = new List<int>(16);
            if (isJagged)
            {
                // ReSharper disable once PossibleNullReferenceException
                Array arr = array;
                do
                {
                    l.Add(arr.Length);
                    arr = arr.GetValue(0) as Array;
                } while (arr != null && arr.GetType().IsArray);
            }
            else
            {
                //jagged or regular
                for (int dim = 0; dim < array.Rank; dim++)
                {
                    l.Add(array.GetLength(dim));
                }
            }

            return l.ToArray();
        }

        private Shape resolveUnreducedBroadcastedShape()
        {
            var bi = BroadcastInfo;
            if (bi.UnreducedBroadcastedShape.HasValue)
                return bi.UnreducedBroadcastedShape.Value;

            Shape unreducedBroadcasted;
            var vi = ViewInfo;
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
            return unreducedBroadcasted;
        }

        /// <summary>
        ///     Recalculate hashcode from current dimension and layout.
        /// </summary>
        [MethodImpl((MethodImplOptions)768)]
        internal void ComputeHashcode()
        {
            if (dimensions.Length > 0)
            {
                unchecked
                {
                    size = 1;
                    int hash = (layout * 397);
                    foreach (var v in dimensions)
                    {
                        size *= v;
                        hash ^= (size * 397) * (v * 397);
                    }

                    _hashCode = hash;
                }
            }
        }

        #region Slicing support

        [MethodImpl((MethodImplOptions)768)]
        public readonly Shape Slice(string slicing_notation) =>
            this.Slice(NumSharp.Slice.ParseSlices(slicing_notation));

        [MethodImpl((MethodImplOptions)768)]
        public readonly Shape Slice(params Slice[] input_slices)
        {
            if (IsEmpty)
                throw new InvalidOperationException("Unable to slice an empty shape.");

            //if (IsBroadcasted)
            //    throw new NotSupportedException("Unable to slice a shape that is broadcasted.");

            var slices = new List<SliceDef>(16);
            var sliced_axes_unreduced = new List<int>();
            for (int i = 0; i < NDim; i++)
            {
                var dim = Dimensions[i];
                var slice = input_slices.Length > i ? input_slices[i] : NumSharp.Slice.All; //fill missing selectors
                var slice_def = slice.ToSliceDef(dim);
                slices.Add(slice_def);
                var count = Math.Abs(slices[i].Count); // for index-slices count would be -1 but we need 1.
                sliced_axes_unreduced.Add(count);
            }

            if (IsSliced && ViewInfo.Slices != null)
            {
                // merge new slices with existing ones and insert the indices of the parent shape that were previously reduced
                for (int i = 0; i < ViewInfo.OriginalShape.NDim; i++)
                {
                    var orig_slice = ViewInfo.Slices[i];
                    if (orig_slice.IsIndex)
                    {
                        slices.Insert(i, orig_slice);
                        sliced_axes_unreduced.Insert(i, 1);
                        continue;
                    }

                    slices[i] = ViewInfo.Slices[i].Merge(slices[i]);
                    sliced_axes_unreduced[i] = Math.Abs(slices[i].Count);
                }
            }

            var sliced_axes = sliced_axes_unreduced.Where((dim, i) => !slices[i].IsIndex).ToArray();
            var origin = (this.IsSliced && ViewInfo.Slices != null) ? this.ViewInfo.OriginalShape : this;
            var viewInfo = new ViewInfo() {OriginalShape = origin, Slices = slices.ToArray(), UnreducedShape = new Shape(sliced_axes_unreduced.ToArray()),};

            if (IsRecursive)
                viewInfo.ParentShape = ViewInfo.ParentShape;

            if (sliced_axes.Length == 0) //is it a scalar
                return NewScalar(viewInfo);

            return new Shape(sliced_axes) {ViewInfo = viewInfo};
        }

        #endregion

        #region Implicit Operators

        public static explicit operator int[](Shape shape) =>
            (int[])shape.dimensions.Clone(); //we clone to avoid any changes

        public static implicit operator Shape(int[] dims) =>
            new Shape(dims);

        public static explicit operator int(Shape shape) =>
            shape.Size;

        public static explicit operator Shape(int dim) =>
            Shape.Vector(dim);

        public static explicit operator (int, int)(Shape shape) =>
            shape.dimensions.Length == 2 ? (shape.dimensions[0], shape.dimensions[1]) : (0, 0);

        public static implicit operator Shape((int, int) dims) =>
            Shape.Matrix(dims.Item1, dims.Item2);

        public static explicit operator (int, int, int)(Shape shape) =>
            shape.dimensions.Length == 3 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2]) : (0, 0, 0);

        public static implicit operator Shape((int, int, int) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3);

        public static explicit operator (int, int, int, int)(Shape shape) =>
            shape.dimensions.Length == 4 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3]) : (0, 0, 0, 0);

        public static implicit operator Shape((int, int, int, int) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);

        public static explicit operator (int, int, int, int, int)(Shape shape) =>
            shape.dimensions.Length == 5 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3], shape.dimensions[4]) : (0, 0, 0, 0, 0);

        public static implicit operator Shape((int, int, int, int, int) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5);

        public static explicit operator (int, int, int, int, int, int)(Shape shape) =>
            shape.dimensions.Length == 6 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3], shape.dimensions[4], shape.dimensions[5]) : (0, 0, 0, 0, 0, 0);

        public static implicit operator Shape((int, int, int, int, int, int) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5, dims.Item6);

        #endregion

        #region Deconstructor

        public readonly void Deconstruct(out int dim1, out int dim2)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
        }

        public readonly void Deconstruct(out int dim1, out int dim2, out int dim3)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
        }

        public readonly void Deconstruct(out int dim1, out int dim2, out int dim3, out int dim4)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
            dim4 = dims[3];
        }

        public readonly void Deconstruct(out int dim1, out int dim2, out int dim3, out int dim4, out int dim5)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
            dim4 = dims[3];
            dim5 = dims[4];
        }

        public readonly void Deconstruct(out int dim1, out int dim2, out int dim3, out int dim4, out int dim5, out int dim6)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
            dim4 = dims[3];
            dim5 = dims[4];
            dim6 = dims[5];
        }

        #endregion

        #region Equality

        public static bool operator ==(Shape a, Shape b)
        {
            if (a.IsEmpty && b.IsEmpty)
                return true;

            if (a.IsEmpty || b.IsEmpty)
                return false;

            if (a.size != b.size || a.NDim != b.NDim)
                return false;

            var dim = a.NDim;
            for (int i = 0; i < dim; i++)
            {
                if (a[i] != b[i])
                    return false;
            }

            return true;
        }

        public static bool operator !=(Shape a, Shape b)
        {
            return !(a == b);
        }

        public override readonly bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj))
            {
                return false;
            }

            if (obj.GetType() != this.GetType())
            {
                return false;
            }

            return Equals((Shape)obj);
        }

        /// <summary>Indicates whether the current object is equal to another object of the same type.</summary>
        /// <param name="other">An object to compare with this object.</param>
        /// <returns>true if the current object is equal to the <paramref name="other">other</paramref> parameter; otherwise, false.</returns>
        public readonly bool Equals(Shape other)
        {
            if ((_hashCode == 0 && _hashCode == other._hashCode) || dimensions == null && other.dimensions == null) //they are empty.
                return true;

            if ((dimensions == null && other.dimensions != null) || (dimensions != null && other.dimensions == null)) //they are empty.
                return false;

            if (size != other.size /*|| layout != other.layout*/ || dimensions.Length != other.dimensions.Length)
                return false;

            // ReSharper disable once LoopCanBeConvertedToQuery
            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] != other.dimensions[i])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>Serves as the default hash function.</summary>
        /// <returns>A hash code for the current object.</returns>
        public override int GetHashCode()
        {
            // ReSharper disable once NonReadonlyMemberInGetHashCode
            return _hashCode;
        }

        #endregion

        /// <summary>
        ///     Translates coordinates with negative indices, e.g:<br></br>
        ///     np.arange(9)[-1] == np.arange(9)[8]<br></br>
        ///     np.arange(9)[-2] == np.arange(9)[7]<br></br>
        /// </summary>
        /// <param name="dimensions">The dimensions these coordinates are targeting</param>
        /// <param name="coords">The coordinates.</param>
        /// <returns>Coordinates without negative indices.</returns>
        [SuppressMessage("ReSharper", "ParameterHidesMember"), MethodImpl((MethodImplOptions)512)]
        public static int[] InferNegativeCoordinates(int[] dimensions, int[] coords)
        {
            for (int i = 0; i < coords.Length; i++)
            {
                var curr = coords[i];
                if (curr < 0)
                    coords[i] = dimensions[i] + curr;
            }

            return coords;
        }

        /// <summary>
        ///     Flag this shape as stride-modified.
        /// </summary>
        /// <param name="value"></param>
        public void SetStridesModified(bool value)
        {
            ModifiedStrides = value;
        }

        public override string ToString() =>
            "(" + string.Join(", ", dimensions) + ")";

        /// <summary>Creates a new object that is a copy of the current instance.</summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        readonly object ICloneable.Clone() =>
            Clone(true, false, false);

        /// <summary>
        ///     Creates a complete copy of this Shape.
        /// </summary>
        /// <param name="deep">Should make a complete deep clone or a shallow if false.</param>
        public readonly Shape Clone(bool deep = true, bool unview = false, bool unbroadcast = false)
        {
            if (IsEmpty)
                return default;

            if (IsScalar)
            {
                if ((unview || ViewInfo == null) && (unbroadcast || BroadcastInfo == null))
                    return Scalar;

                return NewScalar(unview ? null : ViewInfo?.Clone(), unbroadcast ? null : BroadcastInfo?.Clone());
            }

            if (deep && unview && unbroadcast)
                return new Shape((int[])this.dimensions.Clone());

            if (!deep && !unview && !unbroadcast)
                return this; //basic struct reassign

            var ret = deep ? new Shape(this) : (Shape)MemberwiseClone();

            if (unview)
                ret.ViewInfo = null;

            if (unbroadcast)
            {
                ret.BroadcastInfo = null;
                ret._computeStrides();
            }

            return ret;
        }

        /// <summary>
        ///     Returns a clean shape based on this.
        ///     Cleans ViewInfo and returns a newly constructed.
        /// </summary>
        /// <returns></returns>
        public readonly Shape Clean()
        {
            if (IsScalar)
                return NewScalar();

            return new Shape((int[])this.dimensions.Clone());
        }
    }
}
