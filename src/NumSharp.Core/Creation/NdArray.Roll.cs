using System;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray roll(int shift, int axis )
        {
            if (axis > this.ndim)
                throw new IncorrectShapeException();
            
            shift = (axis == 0) ? (-1) * shift : shift;

            shift = ((shift % this.shape[axis]) < 0) ? shift+this.shape[axis] : shift;

            switch (dtype.Name)
            {
                case "Int32":
                    {
                        var data = this.Data<int>();
                        var newData = new int[this.size];
                        for (int idx = 0; idx < this.size; idx++)
                        {
                            int[] indexes = this.Storage.Shape.GetDimIndexOutShape(idx);
                            indexes[axis] = (indexes[axis] + shift) % this.shape[axis];
                            newData[this.Storage.Shape.GetIndexInShape(indexes)] = data[idx];
                        }
                        return new NDArray(newData, this.shape);
                    }

                case "Single":
                    {
                        var data = this.Data<float>();
                        var newData = new float[this.size];
                        for (int idx = 0; idx < this.size; idx++)
                        {
                            int[] indexes = this.Storage.Shape.GetDimIndexOutShape(idx);
                            indexes[axis] = (indexes[axis] + shift) % this.shape[axis];
                            newData[this.Storage.Shape.GetIndexInShape(indexes)] = data[idx];
                        }
                        return new NDArray(newData, this.shape);
                    }

                case "Double":
                    {
                        var data = this.Data<double>();
                        var newData = new double[this.size];
                        for (int idx = 0; idx < this.size; idx++)
                        {
                            int[] indexes = this.Storage.Shape.GetDimIndexOutShape(idx);
                            indexes[axis] = (indexes[axis] + shift) % this.shape[axis];
                            newData[this.Storage.Shape.GetIndexInShape(indexes)] = data[idx];
                        }
                        return new NDArray(newData, this.shape);
                    }

                default:
                    throw new NotImplementedException($"NDArray.roll {dtype.Name}");
            }
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