using System;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray roll(int shift, int axis )
        {
            if (axis > this.ndim)
                throw new IncorrectShapeException();
            
            Array data = this.Storage.GetData();
            Array result = this.Storage.CloneData();

            shift = (axis == 0) ? (-1) * shift : shift;

            shift = ((shift % this.shape[axis]) < 0) ? shift+this.shape[axis] : shift; 

            for ( int idx = 0; idx < this.size;idx++)
            {
                int[] indexes = this.Storage.Shape.GetDimIndexOutShape(idx);
                indexes[axis] = ( indexes[axis] + shift) % this.shape[axis];

                result.SetValue(data.GetValue(idx),this.Storage.Shape.GetIndexInShape(indexes));
            }
            
            NDArray resultNDArray = new NDArray(this.dtype,this.shape);

            resultNDArray.Storage.SetData(result);

            return resultNDArray;
        }
        public NDArray roll(int shift)
        {
            shift = (-1) * shift;

            int tensorLayout = this.Storage.TensorLayout;

            this.Storage.ChangeTensorLayout(2);

            Array cpy = Array.CreateInstance(this.dtype, this.size);

            shift = ((shift % this.size) < 0) ? shift+this.size : shift; 

            //shift = shift % this.size;

            var strg = this.Storage.GetData();

            Array.Copy(strg,shift,cpy,0,this.size-shift);

            for (int idx = 0; idx < shift;idx++)
                cpy.SetValue( strg.GetValue(idx) ,this.size-shift+idx );

            var returnValue = new NDArray(this.dtype);

            returnValue.Storage.Allocate(dtype, new Shape(this.shape), 1);

            returnValue.Storage.SetData(cpy);

            this.Storage.ChangeTensorLayout(tensorLayout);

            return returnValue;
        }
    }
}