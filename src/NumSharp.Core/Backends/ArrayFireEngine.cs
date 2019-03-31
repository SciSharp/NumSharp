using NumSharp.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Backends
{
    public class ArrayFireEngine : ITensorEngine
    {
        public Type DType => throw new NotImplementedException();

        public int DTypeSize => throw new NotImplementedException();

        public Shape Shape => throw new NotImplementedException();

        public void Allocate(Type dtype, Shape shape)
        {
            throw new NotImplementedException();
        }

        public void Allocate(Array values)
        {
            throw new NotImplementedException();
        }

        public void ChangeDataType(Type dtype)
        {
            throw new NotImplementedException();
        }

        public object Clone()
        {
            throw new NotImplementedException();
        }

        public Array CloneData()
        {
            throw new NotImplementedException();
        }

        public Array CloneData(Type dtype)
        {
            throw new NotImplementedException();
        }

        public T[] CloneData<T>()
        {
            throw new NotImplementedException();
        }

        public NDArray Dot(NDArray x, NDArray y)
        {
            var dtype = x.dtype;

            switch (dtype.Name)
            {
                case "Int32":
                    break;
            }

            throw new NotImplementedException("SimdEngine.dot");
        }

        public Array GetData()
        {
            throw new NotImplementedException();
        }

        public Array GetData(Type dtype)
        {
            throw new NotImplementedException();
        }

        public T[] GetData<T>()
        {
            throw new NotImplementedException();
        }

        public object GetData(params int[] indexes)
        {
            throw new NotImplementedException();
        }

        public T GetData<T>(params int[] indexes)
        {
            throw new NotImplementedException();
        }

        public void Reshape(params int[] dimensions)
        {
            throw new NotImplementedException();
        }

        public void SetData(Array values)
        {
            throw new NotImplementedException();
        }

        public void SetData<T>(Array values)
        {
            throw new NotImplementedException();
        }

        public void SetData(object value, params int[] indexes)
        {
            throw new NotImplementedException();
        }

        public void SetData(Array values, Type dtype)
        {
            throw new NotImplementedException();
        }
    }
}
