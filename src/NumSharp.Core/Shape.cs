using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp
{
    public partial class Shape
    {
        /// <summary>
        /// Dense data are stored contiguously in memory, addressed by a single index (the memory address). 
        /// Array memory ordering schemes translate that single index into multiple indices corresponding to the array coordinates.
        /// 0: Row major
        /// 1: Column major
        /// </summary>
        private int layout;
        public string Order => layout == 1 ? "F" : "C";
        
        public int NDim => dimensions.Length;

        private int[] dimensions;
        public int[] Dimensions => dimensions;

        private int[] dimOffset;
        public int[] DimOffset => dimOffset;

        private int size;
        public int Size => size;

        public Shape(params int[] dims)
        {
            ReShape(dims);
        }

        public int this[int dim]
        {
            get
            {
                return Dimensions[dim];
            }

            set
            {
                Dimensions[dim] = value;
            }
        }

        protected void _SetDimOffset()
        {
            if (dimensions.Length == 0)
                return;

            if (layout == 0)
            {
                dimOffset[dimOffset.Length - 1] = 1;
                for (int idx = dimOffset.Length - 1; idx >= 1; idx--)
                    dimOffset[idx - 1] = dimOffset[idx] * dimensions[idx];
            }
            else
            {
                dimOffset[0] = 1;
                for (int idx = 1; idx < dimOffset.Length; idx++)
                    dimOffset[idx] = dimOffset[idx - 1] * dimensions[idx - 1];
            }
        }

        /// <summary>
        /// get store position by shape
        /// for example: 2 x 2 row major
        /// [[1, 2, 3], [4, 5, 6]]
        /// GetIndexInShape(0, 1) = 1
        /// GetIndexInShape(1, 1) = 5
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public int GetIndexInShape(params int[] select)
        {
            int idx = 0;

            for (int i = 0; i < select.Length; i++)
                idx += dimOffset[i] * select[i];

            return idx;
        }

        /// <summary>
        /// get position in shape by store position
        /// [[1, 2, 3], [4, 5, 6]]
        /// GetDimIndexOutShape(1) = (0, 1)
        /// GetDimIndexOutShape(4) = (1, 1)
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public int[] GetDimIndexOutShape(int select)
        {
            int[] dimIndexes = null;
            if (dimOffset.Length == 1)
                dimIndexes = new int[] {select};

            else if(layout == 0)
            {
                int counter = select;
                dimIndexes = new int[dimOffset.Length];

                for (int idx = 0; idx < dimOffset.Length; idx++)
                {
                    dimIndexes[idx] = counter / dimOffset[idx];
                    counter -= dimIndexes[idx] * dimOffset[idx];
                }
            }
            else
            {
                int counter = select;
                dimIndexes = new int[dimOffset.Length];

                for (int idx = dimOffset.Length - 1; idx > -1; idx--)
                {
                    dimIndexes[idx] = counter / dimOffset[idx];
                    counter -= dimIndexes[idx] * dimOffset[idx];
                }
            }

            return dimIndexes;
        }

        public void  ChangeTensorLayout(string order = "C")
        {
            layout = order == "C" ? 0 : 1;
            dimOffset = new int[dimensions.Length];
            _SetDimOffset();
        }

        public void ReShape(params int[] dims)
        {
            dimensions = dims;
            dimOffset = new int[dimensions.Length];

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
                    return new int[] { dims[0] }.Concat(dims.Skip(2).Take(dims.Length - 2)).ToArray();
                case 2:
                    return dims.Take(2).ToArray();
                default:
                    throw new NotImplementedException($"GetCoordinates shape: {string.Join(", ", dims)} axis: {axis}");
            }
        }

        public static implicit operator int[] (Shape shape) => shape.dimensions;
        public static implicit operator Shape(int[] dims) => new Shape(dims);

        public static implicit operator int(Shape shape) => shape.Size;
        public static implicit operator Shape(int dim) => new Shape(dim);

        public static implicit operator (int, int)(Shape shape) => shape.dimensions.Length == 2 ? (shape.dimensions[0], shape.dimensions[1]) : (0, 0);
        public static implicit operator Shape((int, int) dims) => new Shape(dims.Item1, dims.Item2);

        public static implicit operator (int, int, int) (Shape shape) => shape.dimensions.Length == 3 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2]) : (0, 0, 0);
        public static implicit operator Shape((int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3);

        public static implicit operator (int, int, int, int) (Shape shape) => shape.dimensions.Length == 4 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3]) : (0, 0, 0, 0);
        public static implicit operator Shape((int, int, int, int) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);
        #region Equality

        public static bool operator ==(Shape a, Shape b)
        {
            if (b is null) return false;
            return Enumerable.SequenceEqual(a.Dimensions, b?.Dimensions);
        }

        public static bool operator !=(Shape a, Shape b)
        {
            return !(a == b);
        }

        public override bool Equals(object obj)
        {
            if (obj.GetType() != typeof(Shape))
                return false;
            return Enumerable.SequenceEqual(Dimensions, ((Shape)obj).Dimensions);
        }

        public override int GetHashCode()
        {
            // TODO: this hashcode function is actually not very useful
            return base.GetHashCode();
        }

        #endregion

        public override string ToString()
        {
            return "(" + string.Join(", ", dimensions) + ")";
        }

    }
}
