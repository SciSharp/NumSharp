using NumSharp.Backends;
using System;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray ones(Type dtype = null, params int[] shapes)
        {
            dtype = (dtype == null ) ? typeof(double) : dtype;

            int dataLength = 1;
            for (int idx = 0; idx < shapes.Length;idx++)
                dataLength *= shapes[idx];

            Array dataArray = Array.CreateInstance(dtype,dataLength);

            for (int idx = 0; idx < dataLength;idx++)
                dataArray.SetValue(1,idx);
            
            this.Storage = new ArrayStorage(dtype);
            this.Storage.Allocate(new Shape(shapes));
            
            this.Storage.SetData(dataArray);

            return this;
        }
        public NDArray ones(params int[] shapes)
        {
            return this.ones(typeof(double),shapes);
        }
    }
}
