using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    public partial class Shape : ICloneable, IEquatable<Shape>
    {
        /// <summary>
        ///     Dense data are stored contiguously in memory, addressed by a single index (the memory address). 
        ///     Array memory ordering schemes translate that single index into multiple indices corresponding to the array coordinates.
        ///     0: Row major
        ///     1: Column major
        /// </summary>
        private int layout;

        private int size;
        private int[] dimensions;
        private int[] strides;

        public string Order => layout == 1 ? "F" : "C";

        public int NDim => dimensions.Length;

        public int[] Dimensions => dimensions;

        public int[] Strides => strides;

        /// <summary>
        ///     The linear size of this shape.
        /// </summary>
        public int Size => size;

        public Shape(params int[] dims)
        {
            Reshape(dims);
        }

        public int this[int dim]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Dimensions[dim];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Dimensions[dim] = value;
        }

        protected void _SetDimOffset()
        {
            if (dimensions.Length == 0)
                return;

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

        /// <summary>
        ///     get store position by shape<br></br>
        ///     for example: 2 x 2 row major<br></br>
        ///     [[1, 2, 3], [4, 5, 6]]<br></br>
        ///     GetIndexInShape(0, 1) = 1<br></br>
        ///     GetIndexInShape(1, 1) = 5
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public int GetIndexInShape(params int[] select)
        {
            if (NDim == 0 && select.Length == 1)
                return select[0];

            int idx = 0;

            for (int i = 0; i < select.Length; i++)
                idx += strides[i] * select[i];

            return idx;
        }

        public int GetIndexInShape(Slice slice, params int[] select)
        {
            if (NDim == 0 && select.Length == 1)
                return select[0];

            int idx = 0; //todo! IL optimization?
            for (int i = 0; i < select.Length; i++)
                idx += strides[i] * select[i];

            return idx;
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

        public void ChangeTensorLayout(string order = "C")
        {
            layout = order == "C" ? 0 : 1;
            strides = new int[dimensions.Length];
            _SetDimOffset();
        }

        public void Reshape(params int[] dims)
        {
            dimensions = dims;
            strides = new int[dimensions.Length];

            size = 1;

            for (int idx = 0; idx < dims.Length; idx++)
                size *= dims[idx];
            _SetDimOffset();
        }

        public static int GetSize(int[] dims)
        {
            int size = 1;

            for (int idx = 0; idx < dims.Length; idx++)
                size *= dims[idx];

            return size;
        }

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
            if (b is null) return false;
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

            if (ReferenceEquals(this, obj))
            {
                return true;
            }

            if (obj.GetType() != this.GetType())
            {
                return false;
            }

            return Equals((Shape) obj);
        }

        /// <summary>Indicates whether the current object is equal to another object of the same type.</summary>
        /// <param name="other">An object to compare with this object.</param>
        /// <returns>true if the current object is equal to the <paramref name="other">other</paramref> parameter; otherwise, false.</returns>
        public bool Equals(Shape other) {
            if (ReferenceEquals(null, other))
            {
                return false;
            }

            if (ReferenceEquals(this, other))
            {
                return true;
            }

            return layout == other.layout && Enumerable.SequenceEqual(dimensions, other.dimensions);
        }

        /// <summary>Serves as the default hash function.</summary>
        /// <returns>A hash code for the current object.</returns>
        public override int GetHashCode() {
            unchecked
            {
                int ret = (layout * 397);
                foreach (var d in dimensions) {
                    ret ^= d;
                }

                return ret;
            }
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
            return new Shape(this.dimensions) {layout = this.layout};
        }
    }
}
