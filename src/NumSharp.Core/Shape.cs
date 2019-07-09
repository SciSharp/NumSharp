using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    public struct Shape : ICloneable, IEquatable<Shape>, IComparable<Shape>, IComparable
    {
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
        public static Shape Scalar { get; } = new Shape(Array.Empty<int>()) {size = 1};

        [MethodImpl((MethodImplOptions)768)]
        public static Shape Empty(int ndim)
        {
            return new Shape {dimensions = new int[ndim], strides = new int[ndim]};
            //default vals already sets: ret.layout = 0;
            //default vals already sets: ret.size = 0;
            //default vals already sets: ret._hashCode = 0;
            //default vals already sets: ret.IsScalar = false;
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
            this.dimensions = other.dimensions;
            this.strides = other.strides;
            this.IsScalar = other.IsScalar;
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
                    if (layout == 0)
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
        private void _SetDimOffset()
        {
            if (dimensions.Length == 0)
                return;

            unchecked
            {
                if (layout == 0)
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
            if (dimensions.Length == 0 && select.Length == 1)
                return select[0];

            int idx = 0;
            unchecked
            {
                for (int i = 0; i < select.Length; i++)
                    idx += strides[i] * select[i];
            }

            return idx;
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
        public int GetIndexInShape(Slice slice, params int[] select) => GetIndexInShape(@select);

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
                throw new ArgumentException("Selection indexes cannot be an empty collection.", nameof(indicies));

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
                dimIndexes = new int[] {select};

            else if (layout == 0)
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

                for (int idx = strides.Length - 1; idx > -1; idx--)
                {
                    dimIndexes[idx] = counter / strides[idx];
                    counter -= dimIndexes[idx] * strides[idx];
                }
            }

            return dimIndexes;
        }

        public void ChangeTensorLayout(char order = 'C')
        {
            layout = order;
            strides = new int[dimensions.Length];
            _SetDimOffset();
        }

        [MethodImpl((MethodImplOptions)768)]
        public void Reshape(params int[] dims)
        {
            this = new Shape(dims);
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

        //TODO! figure what to name this.
        public static int[] GetShape(int[] dims, int axis = -1)
        {
            switch (axis)
            {
                case -1:
                    return dims;
                case 0:
                    return dims.Skip(1).Take(dims.Length - 1).ToArray();
                case 1:
                    return new int[] {dims[0]}.Concat(dims.Skip(2).Take(dims.Length - 2)).ToArray();
                case 2:
                    return dims.Take(2).ToArray();
                default:
                    throw new NotImplementedException($"GetCoordinates shape: {string.Join(", ", dims)} axis: {axis}");
            }
        }

        #region Slicing support

        public Shape Slice(Slice[] slices, bool reduce = false)
        {
            var sliced_axes = Dimensions.Select((dim, i) => slices[i].GetSize(dim));
            if (reduce)
                sliced_axes = sliced_axes.Where((dim, i) => !slices[i].IsIndex);
            return new Shape(sliced_axes.ToArray());
        }

        #endregion

        #region Implicit Operators

        public static implicit operator int[](Shape shape) => shape.dimensions;
        public static implicit operator Shape(int[] dims) => new Shape(dims);

        public static implicit operator int(Shape shape) => shape.Size;
        public static implicit operator Shape(int dim) => new Shape(dim);

        public static implicit operator (int, int)(Shape shape) => shape.dimensions.Length == 2 ? (shape.dimensions[0], shape.dimensions[1]) : (0, 0);
        public static implicit operator Shape((int, int) dims) => new Shape(dims.Item1, dims.Item2);

        public static implicit operator (int, int, int)(Shape shape) => shape.dimensions.Length == 3 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2]) : (0, 0, 0);
        public static implicit operator Shape((int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3);

        public static implicit operator (int, int, int, int)(Shape shape) => shape.dimensions.Length == 4 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3]) : (0, 0, 0, 0);
        public static implicit operator Shape((int, int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);

        #endregion

        #region Equality

        public static bool operator ==(Shape a, Shape b)
        {
            return Equals(a, b);
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
            return Clone();
        }

        /// <summary>
        ///     Creates a complete copy of this Shape.
        /// </summary>
        public Shape Clone()
        {
            return new Shape(this);
        }
    }
}
