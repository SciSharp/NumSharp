using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Text;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp
{
    public struct Shape : ICloneable, IEquatable<Shape>, IComparable<Shape>, IComparable
    {
        private static readonly int[] _vectorStrides = { 1 };

        internal ViewInfo ViewInfo;

        public bool IsSliced
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ViewInfo != null;
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
        public bool IsEmpty => _hashCode == 0;
        public char Order => layout;

        /// <summary>
        ///     Singleton instance of a <see cref="Shape"/> that represents a scalar.
        /// </summary>
        public static readonly Shape Scalar = new Shape(Array.Empty<int>()) { size = 1, _hashCode = int.MinValue };

        /// <summary>
        ///     Create a shape that represents a vector.
        /// </summary>
        /// <remarks>Faster than calling Shape's constructor</remarks>
        public static Shape Vector(int length)
        {
            var shape = new Shape { dimensions = new[] { length }, strides = _vectorStrides, layout = 'C', size = length };

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
            var shape = new Shape { dimensions = new[] { length }, strides = _vectorStrides, layout = 'C', size = length, ViewInfo = viewInfo };

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
            var shape = new Shape { dimensions = new[] { rows, cols }, strides = new int[] { cols, 1 }, layout = 'C', size = rows * cols };

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
            this.ViewInfo = other.ViewInfo;
        }

        [MethodImpl((MethodImplOptions)512)]
        public Shape(params int[] dims)
        {
            this.ViewInfo = null;
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
        }

        [MethodImpl((MethodImplOptions)768)]
        public static Shape Empty(int ndim)
        {
            return new Shape { dimensions = new int[ndim], strides = new int[ndim] };
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
        ///     get store position by shape<br></br>
        ///     for example: 2 x 2 row major<br></br>
        ///     [[1, 2, 3], [4, 5, 6]]<br></br>
        ///     GetIndexInShape(0, 1) = 1<br></br>
        ///     GetIndexInShape(1, 1) = 5
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public int GetIndexInShape(params int[] select)
        {
            int offset;
            if (IsSliced)
            {
                if (select.Length > dimensions.Length)
                    throw new InvalidEnumArgumentException($"select has too many coordinates for this shape");
                // TODO: perf opt
                var vi = ViewInfo;
                if (dimensions.Length == 0 && select.Length == 1)
                {
                    var slice = vi.Slices[0];
                    var start = slice.Step < 0 ? slice.Stop.Value - 1 : slice.Start.Value;
                    return start + select[0] * slice.Step;
                }

                offset = 0;
                unchecked
                {
                    for (int i = 0; i < @select.Length; i++)
                    {
                        if (vi.Slices.Length <= i)
                        {
                            offset += strides[i] * @select[i];
                            continue;
                        }
                        var slice = vi.Slices[i];
                        var start = slice.Step < 0 ? slice.Stop.Value - 1 : slice.Start.Value;

                        offset += strides[i] * (start + @select[i] * slice.Step);
                    }
                }
                return offset;
            }

            // no slicing
            if (dimensions.Length == 0 && select.Length == 1)
                return select[0];

            offset = 0;
            unchecked
            {
                for (int i = 0; i < @select.Length; i++)
                    offset += strides[i] * @select[i];
            }
            return offset;
        }

        /// <summary>
        ///     Gets the shape based on given <see cref="indicies"/> and the index offset (C-Contegious) inside the current storage.
        /// </summary>
        /// <param name="indicies">The selection of indexes 0 based.</param>
        /// <returns></returns>
        /// <remarks>Used for slicing, return's shape is the new shape of the slice and offset is the offset from current address.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public (Shape Shape, int Offset) GetSubshape(params int[] indicies)
        {
            if (indicies.Length == 0)
                return (this, 0);

            //compute offset
            int offset = 0;
            if (dimensions.Length == 0 && indicies.Length == 1)
                offset = indicies[0];
            else
            {
                unchecked
                {
                    for (int i = 0; i < indicies.Length; i++)
                        offset += strides[i] * indicies[i];
                }
            }

            if (offset >= size)
                throw new IndexOutOfRangeException($"Shape({string.Join(", ", indicies)})");

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
        ///     get position in shape by store position<br></br>
        ///     [[1, 2, 3], [4, 5, 6]]<br></br>
        ///     GetDimIndexOutShape(1) = (0, 1)<br></br>
        ///     GetDimIndexOutShape(4) = (1, 1)
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public int[] GetDimIndexOutShape(int select)
        {
            int[] dimIndexes = null;
            if (strides.Length == 1)
                dimIndexes = new int[] { select };

            else if (layout == 'C')
            {
                int counter = select;
                dimIndexes = new int[strides.Length];

                for (int idx = 0; idx < strides.Length; idx++)
                {
                    dimIndexes[idx] = counter / strides[idx];
                    counter -= dimIndexes[idx] * strides[idx];
                }
            }
            else
            {
                int counter = select;
                dimIndexes = new int[strides.Length];

                for (int idx = strides.Length - 1; idx >= 0; idx--)
                {
                    dimIndexes[idx] = counter / strides[idx];
                    counter -= dimIndexes[idx] * strides[idx];
                }
            }

            return dimIndexes;
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

        public static int[] GetAxis(in int[] dims, int axis)
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
        internal void ComputeHashcode()
        {
            if (dimensions.Length > 0)
            {
                unchecked
                {
                    int hash = (layout * 397);
                    foreach (var v in dimensions)
                        hash ^= (size *= v) * 397;
                    _hashCode = hash;
                }
            }
        }

        #region Slicing support

        public Shape Slice(string slicing_notation)
            => this.Slice(NumSharp.Slice.ParseSlices(slicing_notation));

        public Shape Slice(params Slice[] input_slices)
        {
            var slices = input_slices.Length != Dimensions.Length ? input_slices : new Slice[Dimensions.Length];
            var sliced_axes_unreduced = new int[Dimensions.Length];
            for (int i = 0; i < Dimensions.Length; i++)
            {
                var dim = Dimensions[i];
                var slice = input_slices.Length > i ? input_slices[i] : NumSharp.Slice.All();
                slices[i] = this.IsSliced ? ViewInfo.Slices[i].Merge(slice.Sanitize(dim)) : slice.Sanitize(dim);
                sliced_axes_unreduced[i] = slices[i].GetSize();
            };
            var sliced_axes = sliced_axes_unreduced.Where((dim, i) => !slices[i].IsIndex).ToArray();
            var viewInfo = new ViewInfo() { OriginalShape = this, Slices = slices, UnreducedShape = new Shape(sliced_axes_unreduced) };
            return new Shape(sliced_axes) { ViewInfo = viewInfo };
        }

        #endregion

        #region Implicit Operators

        public static explicit operator int[] (Shape shape) => (int[])shape.dimensions.Clone(); //we clone to avoid any changes
        public static implicit operator Shape(int[] dims) => new Shape(dims);

        public static explicit operator int(Shape shape) => shape.Size;
        public static implicit operator Shape(int dim) => new Shape(dim);

        public static explicit operator (int, int) (Shape shape) => shape.dimensions.Length == 2 ? (shape.dimensions[0], shape.dimensions[1]) : (0, 0);
        public static implicit operator Shape((int, int) dims) => new Shape(dims.Item1, dims.Item2);

        public static explicit operator (int, int, int) (Shape shape) => shape.dimensions.Length == 3 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2]) : (0, 0, 0);
        public static implicit operator Shape((int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3);

        public static explicit operator (int, int, int, int) (Shape shape) => shape.dimensions.Length == 4 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3]) : (0, 0, 0, 0);
        public static implicit operator Shape((int, int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);

        public static explicit operator (int, int, int, int, int) (Shape shape) => shape.dimensions.Length == 5 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3], shape.dimensions[4]) : (0, 0, 0, 0, 0);
        public static implicit operator Shape((int, int, int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5);

        public static explicit operator (int, int, int, int, int, int) (Shape shape) => shape.dimensions.Length == 6 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3], shape.dimensions[4], shape.dimensions[5]) : (0, 0, 0, 0, 0, 0);
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

        /// <summary>Compares the current instance with another object of the same type and returns an integer that indicates whether the current instance precedes, follows, or occurs in the same position in the sort order as the other object. </summary>
        /// <param name="other">An object to compare with this instance. </param>
        /// <returns>A value that indicates the relative order of the objects being compared. The return value has these meanings: Value Meaning Less than zero This instance precedes <paramref name="other" /> in the sort order.  Zero This instance occurs in the same position in the sort order as <paramref name="other" />. Greater than zero This instance follows <paramref name="other" /> in the sort order. </returns>
        public int CompareTo(Shape other)
        {
            return size.CompareTo(other.size);
        }

        /// <summary>Compares the current instance with another object of the same type and returns an integer that indicates whether the current instance precedes, follows, or occurs in the same position in the sort order as the other object.</summary>
        /// <param name="obj">An object to compare with this instance. </param>
        /// <returns>A value that indicates the relative order of the objects being compared. The return value has these meanings: Value Meaning Less than zero This instance precedes <paramref name="obj" /> in the sort order. Zero This instance occurs in the same position in the sort order as <paramref name="obj" />. Greater than zero This instance follows <paramref name="obj" /> in the sort order. </returns>
        /// <exception cref="T:System.ArgumentException">
        /// <paramref name="obj" /> is not the same type as this instance. </exception>
        public int CompareTo(object obj)
        {
            if (ReferenceEquals(null, obj)) return 1;
            return obj is Shape other ? CompareTo(other) : throw new ArgumentException($"Object must be of type {nameof(Shape)}");
        }

        /// <summary>Returns a value that indicates whether a <see cref="T:NumSharp.NewStuff.Shape" /> value is less than another <see cref="T:NumSharp.NewStuff.Shape" /> value.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if <paramref name="left" /> is less than <paramref name="right" />; otherwise, false.</returns>
        public static bool operator <(Shape left, Shape right)
        {
            return left.CompareTo(right) < 0;
        }

        /// <summary>Returns a value that indicates whether a <see cref="T:NumSharp.NewStuff.Shape" /> value is greater than another <see cref="T:NumSharp.NewStuff.Shape" /> value.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if <paramref name="left" /> is greater than <paramref name="right" />; otherwise, false.</returns>
        public static bool operator >(Shape left, Shape right)
        {
            return left.CompareTo(right) > 0;
        }

        /// <summary>Returns a value that indicates whether a <see cref="T:NumSharp.NewStuff.Shape" /> value is less than or equal to another <see cref="T:NumSharp.NewStuff.Shape" /> value.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if <paramref name="left" /> is less than or equal to <paramref name="right" />; otherwise, false.</returns>
        public static bool operator <=(Shape left, Shape right)
        {
            return left.CompareTo(right) <= 0;
        }

        /// <summary>Returns a value that indicates whether a <see cref="T:NumSharp.NewStuff.Shape" /> value is greater than or equal to another <see cref="T:NumSharp.NewStuff.Shape" /> value.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if <paramref name="left" /> is greater than <paramref name="right" />; otherwise, false.</returns>
        public static bool operator >=(Shape left, Shape right)
        {
            return left.CompareTo(right) >= 0;
        }

        #endregion

        public override string ToString()
        {
            return "(" + string.Join(", ", dimensions) + ")";
        }

        /// <summary>Creates a new object that is a copy of the current instance.</summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        object ICloneable.Clone()
        {
            return Clone(true);
        }

        /// <summary>
        ///     Creates a complete copy of this Shape.
        /// </summary>
        /// <param name="deep">Should make a complete deep clone or a shallow if false.</param>
        public Shape Clone(bool deep = true)
        {
            return deep ? new Shape(this) : this;
        }
    }
}
