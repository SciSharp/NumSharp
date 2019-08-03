using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Text;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     Represents a shape of an N-D array.
    /// </summary>
    /// <remarks>Handles slicing, indexing based on coordinates or linear offset and broadcastted indexing.</remarks>
    public struct Shape : ICloneable, IEquatable<Shape>
    {
        internal ViewInfo ViewInfo;

        /// <summary>
        /// True if the shape of this array was obtained by a slicing operation that caused the underlying data to be non-contiguous
        /// </summary>
        public bool IsSliced
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ViewInfo != null;
        }

        public bool IsRecursive
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
        internal char layout;

        internal int _hashCode;
        internal int size;
        internal int[] dimensions;
        internal int[] strides;

        public bool IsScalar;

        /// <summary>
        /// True if the shape is not initialized.
        /// Note: A scalar shape is not empty.
        /// </summary>
        public bool IsEmpty => _hashCode == 0;

        public char Order => layout;

        /// <summary>
        ///     Singleton instance of a <see cref="Shape"/> that represents a scalar.
        /// </summary>
        public static readonly Shape Scalar = new Shape(Array.Empty<int>()) {size = 1, _hashCode = int.MinValue};

        /// <summary>
        ///     Create a new scalar shape
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Shape NewScalar() => new Shape(Array.Empty<int>()) {size = 1, _hashCode = int.MinValue};

        /// <summary>
        ///     Create a new scalar shape
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Shape NewScalar(ViewInfo viewInfo) => new Shape(Array.Empty<int>()) {size = 1, _hashCode = int.MinValue, ViewInfo = viewInfo};

        /// <summary>
        ///     Create a shape that represents a vector.
        /// </summary>
        /// <remarks>Faster than calling Shape's constructor</remarks>
        public static Shape Vector(int length)
        {
            var shape = new Shape {dimensions = new int[] {length}, strides = new int[] {1}, layout = 'C', size = length};

            unchecked
            {
                shape._hashCode = 26599 ^ length * 397;
            }

            shape.IsScalar = false;
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
                layout = 'C',
                size = length,
                ViewInfo = viewInfo
            };

            unchecked
            {
                shape._hashCode = 26599 ^ length * 397;
            }

            shape.IsScalar = false;
            return shape;
        }

        /// <summary>
        ///     Create a shape that represents a matrix.
        /// </summary>
        /// <remarks>Faster than calling Shape's constructor</remarks>
        public static Shape Matrix(int rows, int cols)
        {
            var shape = new Shape {dimensions = new[] {rows, cols}, strides = new int[] {cols, 1}, layout = 'C', size = rows * cols};

            unchecked
            {
                shape._hashCode = (26599 ^ rows * 397) ^ cols * 397; //('C' * 397)
            }

            shape.IsScalar = false;
            return shape;
        }

        public int NDim
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => dimensions.Length;
        }

        public int[] Dimensions
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => dimensions;
        }

        public int[] Strides
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => strides;
        }

        /// <summary>
        ///     The linear size of this shape.
        /// </summary>
        public int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => size;
        }

        public Shape(Shape other)
        {
            this.layout = other.layout;
            this._hashCode = other._hashCode;
            this.size = other.size;
            this.dimensions = (int[])other.dimensions.Clone();
            this.strides = (int[])other.strides.Clone();
            this.IsScalar = other.IsScalar;
            this.ViewInfo = other.ViewInfo?.Clone();
        }

        [MethodImpl((MethodImplOptions)512)]
        public Shape(params int[] dims)
        {
            if (dims == null)
            {
                strides = dims = dimensions = Array.Empty<int>();
            }
            else
            {
                dimensions = dims;
                strides = new int[dims.Length];
            }

            layout = 'C';
            size = 1;
            unchecked
            {
                if (dims.Length > 0)
                {
                    int hash = (layout * 397);
                    foreach (var v in dims)
                        hash ^= (size *= v) * 397;
                    _hashCode = hash;
                }
                else
                    _hashCode = 0;

                if (dims.Length != 0)
                    if (layout == 'C')
                    {
                        strides[strides.Length - 1] = 1;
                        for (int idx = strides.Length - 1; idx >= 1; idx--)
                            strides[idx - 1] = strides[idx] * dims[idx];
                    }
                    else
                    {
                        strides[0] = 1;
                        for (int idx = 1; idx < strides.Length; idx++)
                            strides[idx] = strides[idx - 1] * dims[idx - 1];
                    }
            }

            IsScalar = size == 1 && dims.Length == 0;
            ViewInfo = null;
        }

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
        private void _SetDimOffset()
        {
            if (dimensions.Length == 0)
                return;

            unchecked
            {
                if (layout == 'C')
                {
                    strides[strides.Length - 1] = 1;
                    for (int idx = strides.Length - 1; idx >= 1; idx--)
                        strides[idx - 1] = strides[idx] * dimensions[idx];
                }
                else
                {
                    strides[0] = 1;
                    for (int idx = 1; idx < strides.Length; idx++)
                        strides[idx] = strides[idx - 1] * dimensions[idx - 1];
                }
            }
        }

        public int this[int dim]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Dimensions[dim];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Dimensions[dim] = value;
        }

        /// <summary>
        ///     Retrieve the transformed offset if <see cref="IsSliced"/> is true, otherwise returns <paramref name="offset"/>.
        /// </summary>
        /// <param name="offset">The offset within the bounds of <see cref="size"/>.</param>
        /// <returns>The transformed offset.</returns>
        /// <remarks>Avoid using unless it is unclear if shape is sliced or not.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int TransformOffset(int offset)
        {
            // ReSharper disable once ConvertIfStatementToReturnStatement
            if (ViewInfo == null)
                return offset;

            return GetOffset(GetCoordinates(offset));
        }

        /// <summary>
        ///     Get offset index out of coordinate indices.
        /// </summary>
        /// <param name="indices">The coordinates to turn into linear offset</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        /// <remarks>Handles sliced indices and broadcasting</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public int GetOffset(params int[] indices)
        {
            if (!IsSliced)
                return GetOffset_IgnoreViewInfo(indices);

            // we are dealing with a slice
            int offset;
            var vi = ViewInfo;
            if (IsRecursive && vi.Slices == null)
            {
                // we are dealing with an unsliced recursively reshaped slice
                offset = GetOffset_IgnoreViewInfo(indices);
                var parent_coords = vi.ParentShape.GetCoordinates(offset, ignore_view_info: true);
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
        private int GetOffset_IgnoreViewInfo(params int[] indices)
        {
            if (dimensions.Length == 0 && indices.Length == 1)
                return indices[0];

            int offset = 0;
            unchecked
            {
                for (int i = 0; i < indices.Length; i++)
                    offset += strides[i] * indices[i];
            }

            return offset;
        }


        /// <summary>
        ///     Gets the shape based on given <see cref="indicies"/> and the index offset (C-Contiguous) inside the current storage.
        /// </summary>
        /// <param name="indicies">The selection of indexes 0 based.</param>
        /// <returns></returns>
        /// <remarks>Used for slicing, returned shape is the new shape of the slice and offset is the offset from current address.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public (Shape Shape, int Offset) GetSubshape(params int[] indicies)
        {
            if (indicies.Length == 0)
                return (this, 0);

            //compute offset
            int offset = GetOffset(indicies);

            var orig_shape = IsSliced ? ViewInfo.OriginalShape : this;
            if (offset >= orig_shape.Size)
                throw new IndexOutOfRangeException($"The offset {offset} is out of range in Shape {orig_shape.Size}");

            if (indicies.Length == dimensions.Length)
                return (Scalar, offset);

            //compute subshape
            var dim = indicies.Length;
            var innerShape = new int[dimensions.Length - dim];
            for (int i = 0; i < innerShape.Length; i++)
            {
                innerShape[i] = dimensions[dim + i];
            }

            return (innerShape, offset);
        }

        /// <summary>
        ///     Transforms offset index into coordinates that matches this shape.
        /// </summary>
        /// <param name="offset"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public int[] GetCoordinates(int offset, bool ignore_view_info = false)
        {
            int[] coords = null;
            if (strides.Length == 1)
                coords = new int[] {offset};
            else if (layout == 'C')
            {
                int counter = offset;
                coords = new int[strides.Length];
                for (int i = 0; i < strides.Length; i++)
                {
                    coords[i] = counter / strides[i];
                    counter -= coords[i] * strides[i];
                }
            }
            else
            {
                int counter = offset;
                coords = new int[strides.Length];
                for (int i = strides.Length - 1; i >= 0; i--)
                {
                    coords[i] = counter / strides[i];
                    counter -= coords[i] * strides[i];
                }
            }

            if (IsSliced && !ignore_view_info)
            {
                // TODO! undo dimensionality reduction
                for (int i = 0; i < coords.Length; i++)
                {
                    var slice = ViewInfo.Slices[i];
                    coords[i] = (coords[i] / slice.Step) - slice.Start;
                }
            }

            return coords;
        }

        [MethodImpl((MethodImplOptions)768)]
        public void ChangeTensorLayout(char order = 'C')
        {
            layout = order;
            _SetDimOffset();
            ComputeHashcode();
        }

        [MethodImpl((MethodImplOptions)768)]
        public static int GetSize(int[] dims)
        {
            int size = 1;
            unchecked
            {
                for (int idx = 0; idx < dims.Length; idx++)
                    size *= dims[idx];
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
                return Array.Empty<int>();

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
                        hash ^= (size *= v) * 397;
                    _hashCode = hash;
                }
            }
        }

        #region Slicing support

        [MethodImpl((MethodImplOptions)768)]
        public Shape Slice(string slicing_notation) => this.Slice(NumSharp.Slice.ParseSlices(slicing_notation));

        [MethodImpl((MethodImplOptions)768)]
        public Shape Slice(params Slice[] input_slices)
        {
            if (IsEmpty)
                throw new InvalidOperationException("Unable to slice an empty shape.");

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

        public static explicit operator int[](Shape shape) => (int[])shape.dimensions.Clone(); //we clone to avoid any changes
        public static implicit operator Shape(int[] dims) => new Shape(dims);

        public static explicit operator int(Shape shape) => shape.Size;
        public static explicit operator Shape(int dim) => Shape.Vector(dim);

        public static explicit operator (int, int)(Shape shape) => shape.dimensions.Length == 2 ? (shape.dimensions[0], shape.dimensions[1]) : (0, 0); //TODO! this should return (0,0) but rather (dim[0], dim[1]) regardless of size.
        public static implicit operator Shape((int, int) dims) => Shape.Matrix(dims.Item1, dims.Item2);

        public static explicit operator (int, int, int)(Shape shape) => shape.dimensions.Length == 3 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2]) : (0, 0, 0);
        public static implicit operator Shape((int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3);

        public static explicit operator (int, int, int, int)(Shape shape) => shape.dimensions.Length == 4 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3]) : (0, 0, 0, 0);
        public static implicit operator Shape((int, int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);

        public static explicit operator (int, int, int, int, int)(Shape shape) => shape.dimensions.Length == 5 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3], shape.dimensions[4]) : (0, 0, 0, 0, 0);
        public static implicit operator Shape((int, int, int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5);

        public static explicit operator (int, int, int, int, int, int)(Shape shape) => shape.dimensions.Length == 6 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3], shape.dimensions[4], shape.dimensions[5]) : (0, 0, 0, 0, 0, 0);
        public static implicit operator Shape((int, int, int, int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5, dims.Item6);

        #endregion

        #region Deconstructor

        public void Deconstruct(out int dim1, out int dim2)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
        }

        public void Deconstruct(out int dim1, out int dim2, out int dim3)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
        }

        public void Deconstruct(out int dim1, out int dim2, out int dim3, out int dim4)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
            dim4 = dims[3];
        }

        public void Deconstruct(out int dim1, out int dim2, out int dim3, out int dim4, out int dim5)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
            dim4 = dims[3];
            dim5 = dims[4];
        }

        public void Deconstruct(out int dim1, out int dim2, out int dim3, out int dim4, out int dim5, out int dim6)
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

        public override bool Equals(object obj)
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
        public bool Equals(Shape other)
        {
            if ((_hashCode == 0 && _hashCode == other._hashCode) || dimensions == null && other.dimensions == null) //they are empty.
                return true;

            if ((dimensions == null && other.dimensions != null) || (dimensions != null && other.dimensions == null)) //they are empty.
                return false;

            if (size != other.size || layout != other.layout || dimensions.Length != other.dimensions.Length)
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
        ///     Expands a specific <paramref name="axis"/> with 1 dimension.
        /// </summary>
        /// <param name="axis"></param>
        /// <returns></returns>
        [SuppressMessage("ReSharper", "LocalVariableHidesMember")]
        internal Shape ExpandDimension(int axis)
        {
            Shape ret;
            if (IsScalar)
            {
                ret = Vector(1);
                ret.strides[0] = 0;
            }
            else
            {
                ret = Clone(true, true);
            }

            var dimensions = ret.dimensions;
            var strides = ret.strides;
            // Allow negative axis specification
            if (axis < 0)
            {
                axis = dimensions.Length + 1 + axis;
                if (axis < 0)
                {
                    throw new ArgumentException($"Effective axis {axis} is less than 0");
                }
            }

            Arrays.Insert(ref dimensions, axis, 1);
            Arrays.Insert(ref strides, axis, 0);
            ret.dimensions = dimensions;
            ret.strides = strides;
            if (IsSliced)
            {
                ret.ViewInfo = new ViewInfo() { ParentShape = this, Slices = null };
            }

            ret.ComputeHashcode();
            return ret;
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


        public override string ToString() => "(" + string.Join(", ", dimensions) + ")";

        /// <summary>Creates a new object that is a copy of the current instance.</summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        object ICloneable.Clone() => Clone(true, false);

        /// <summary>
        ///     Creates a complete copy of this Shape.
        /// </summary>
        /// <param name="deep">Should make a complete deep clone or a shallow if false.</param>
        public Shape Clone(bool deep = true, bool unview = false)
        {
            if (IsScalar)
            {
                if (unview || ViewInfo == null)
                    return Scalar;

                return NewScalar(ViewInfo.Clone());
            }

            if (!deep && !unview)
                return this; //basic struct reassign

            var ret = deep ? new Shape(this) : (Shape)MemberwiseClone();
            if (unview)
                ret.ViewInfo = null;

            return ret;
        }

        /// <summary>
        ///     Returns a clean shape based on this.
        ///     Cleans ViewInfo and returns a newly constructed.
        /// </summary>
        /// <returns></returns>
        public Shape Clean()
        {
            if (IsScalar)
                return NewScalar();

            return new Shape((int[])this.dimensions.Clone());
        }
    }
}
