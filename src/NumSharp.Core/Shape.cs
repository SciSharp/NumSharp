using System;
using NumSharp.Core.Interfaces;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.Core
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
            if (this._TensorLayout == 1)
            {
                _DimOffset[0] = 1;

                for(int idx = 1;idx < _DimOffset.Length;idx++)
                    _DimOffset[idx] = _DimOffset[idx-1] * this._Dimensions[idx-1];
            }
            else if ( _TensorLayout == 2)
            {
                _DimOffset[_DimOffset.Length-1] = 1;
                for(int idx = _DimOffset.Length-1;idx >= 1;idx--)
                    _DimOffset[idx-1] = _DimOffset[idx] * this._Dimensions[idx];
            }
        }
        public Shape(params int[] shape)
        {
            if (shape.Length == 0)
                throw new Exception("Shape cannot be empty.");
            
            this._Dimensions = shape;
            this._DimOffset = new int[this._Dimensions.Length] ;
            this._TensorLayout = 1;

            this._size = 1;

            for (int idx =0; idx < shape.Length;idx++)
                _size *= shape[idx];
            this._SetDimOffset();
        }
        public Shape(IEnumerable<int> shape) : this(shape.ToArray())
        {
            
        }
        public int GetIndexInShape(params int[] select)
        {
            int idx = 0;
            for (int i = 0; i < select.Length; i++)
            {
                idx += _DimOffset[i] * select[i];
            }

            return idx;
        }
        public int[] GetDimIndexOutShape(int select)
        {
            int[] dimIndexes = null;
            if (this._DimOffset.Length == 1)
                dimIndexes = new int[] {select};
            else if (this._TensorLayout == 1)
            {
                int counter = select;
                dimIndexes = new int[_DimOffset.Length];

                for (int idx = _DimOffset.Length-1; idx > -1;idx--)
                {
                    dimIndexes[idx] = counter / _DimOffset[idx];
                    counter -= dimIndexes[idx] * _DimOffset[idx];
                }
            }
            else
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
        public int UniShape => _Dimensions[0];

        public (int, int) BiShape => _Dimensions.Length == 2 ? (_Dimensions[0], _Dimensions[1]) : (0, 0);

        public (int, int, int) TriShape => _Dimensions.Length == 3 ? (_Dimensions[0], _Dimensions[1], _Dimensions[2]) : (0, 0, 0);
    }
}
