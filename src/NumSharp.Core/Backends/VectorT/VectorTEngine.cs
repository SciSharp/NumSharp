using NumSharp.Interfaces;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace NumSharp.Backends.VectorT
{
    /// <summary>
    /// 
    /// </summary>
    public class VectorTEngine<Td> : IStorage where Td : struct
    {
        private Vector<Td> vectors;

        public Type DType { get; set; }

        public Shape Shape { get; set; }

        public int DTypeSize => throw new NotImplementedException();

        public void Allocate(Type dtype, Shape shape)
        {
            DType = dtype;
            Shape = shape;
            vectors = Vector<Td>.Zero;
        }

        public void Allocate(Array values)
        {
            throw new NotImplementedException();
        }

        public void ChangeDataType(Type dtype)
        {
            throw new NotImplementedException();
        }

        public void ChangeTensorLayout(int order)
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

        public void SetData(object value, params int[] indexes)
        {
            throw new NotImplementedException();
        }

        public void SetData(Array values, Type dtype)
        {
            throw new NotImplementedException();
        }

        public void SetData<T>(Array values)
        {
            
        }

        public void SetData<T>(T value, int offset)
        {
            throw new NotImplementedException();
        }
    }
}
