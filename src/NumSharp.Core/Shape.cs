using System;
using NumSharp.Interfaces;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp
{
    public partial class Shape : IShape
    {
        protected int _TensorLayout;
        public int TensorLayout {get {return _TensorLayout;}}
        protected int[] _Dimensions;
        protected int[] _DimOffset;
        protected int _size;
        public int NDim => _Dimensions.Length;
        public int[] Dimensions {get{return _Dimensions;}}
        public int[] DimOffset {get{return _DimOffset;}}
        public int Size {get{return _size;}}
        protected void _SetDimOffset()
        {
            if (this._Dimensions.Length == 0)
            {

            }
            else
            {
                _DimOffset[_DimOffset.Length - 1] = 1;
                for (int idx = _DimOffset.Length - 1; idx >= 1; idx--)
                    _DimOffset[idx - 1] = _DimOffset[idx] * this._Dimensions[idx];
            }
        }

        public Shape(params int[] dims)
        {
            ReShape(dims);
        }

        public Shape(IEnumerable<int> shape) : this(shape.ToArray())
        {
            
        }

        /// <summary>
        /// get store position by shape
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
                idx += _DimOffset[i] * select[i];
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
            if (this._DimOffset.Length == 1)
                dimIndexes = new int[] {select};
            /*else if (this._TensorLayout == 1)
            {
                int counter = select;
                dimIndexes = new int[_DimOffset.Length];

                for (int idx = _DimOffset.Length-1; idx > -1;idx--)
                {
                    dimIndexes[idx] = counter / _DimOffset[idx];
                    counter -= dimIndexes[idx] * _DimOffset[idx];
                }
            }
            else*/
            {
                int counter = select;
                dimIndexes = new int[_DimOffset.Length];

                for (int idx = 0; idx < _DimOffset.Length;idx++)
                {
                    dimIndexes[idx] = counter / _DimOffset[idx];
                    counter -= dimIndexes[idx] * _DimOffset[idx];
                }    
            }

            return dimIndexes;
        }

        public void  ChangeTensorLayout(int layout)
        {
            _DimOffset = new int[this._Dimensions.Length];

            layout = (layout == 0) ? 1 : layout;
            
            _TensorLayout = layout;
            _SetDimOffset();
        }

        public void ReShape(params int[] dims)
        {
            this._Dimensions = dims;
            this._DimOffset = new int[this._Dimensions.Length];
            this._TensorLayout = 1;

            this._size = 1;

            for (int idx = 0; idx < dims.Length; idx++)
                _size *= dims[idx];
            this._SetDimOffset();
        }

        public static implicit operator int(Shape shape) => shape.Size;
        public static implicit operator (int, int)(Shape shape) => shape._Dimensions.Length == 2 ? (shape._Dimensions[0], shape._Dimensions[1]) : (0, 0);
        public static implicit operator (int, int, int) (Shape shape) => shape._Dimensions.Length == 3 ? (shape._Dimensions[0], shape._Dimensions[1], shape._Dimensions[2]) : (0, 0, 0);
        public static implicit operator int[](Shape shape) => shape._Dimensions;
        public (int, int) BiShape => _Dimensions.Length == 2 ? (_Dimensions[0], _Dimensions[1]) : (0, 0);
        public (int, int, int) TriShape => _Dimensions.Length == 3 ? (_Dimensions[0], _Dimensions[1], _Dimensions[2]) : (0, 0, 0);
        public static implicit operator Shape(int[] dims) => new Shape(dims);
        public static implicit operator Shape(int dim) => new Shape(dim);
    }
}
