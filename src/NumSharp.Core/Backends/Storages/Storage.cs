using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public abstract class Storage : IStorage
    {
        public NPTypeCode DType { get; set; }
        public Shape Shape { get; set; }
        protected unsafe void* address;
        public unsafe void* Address => address;

        public virtual void Allocate(Shape shape)
            => throw new NotImplementedException("");
        public NPTypeCode TypeCode => DType;
        public virtual TensorEngine Engine => new DefaultEngine();

        Type IStorage.DType => throw new NotImplementedException();

        public int DTypeSize => throw new NotImplementedException();

        public IArraySlice InternalArray => throw new NotImplementedException();

        public int Count => throw new NotImplementedException();

        

        public static Storage Allocate<T>(T x) where T : unmanaged
        {
            if (x is int int_x)
            {
                return new StorageOfInt32(int_x);
            }
            return null;
        }

        public static Storage Allocate<T>(T[] x, Shape? shape = null) where T : unmanaged
        {
            if (x is int[] int_x)
            {
                return new StorageOfInt32(int_x, shape);
            }
            return null;
        }

        public unsafe ReadOnlySpan<T> Read<T>()
        {
            return new ReadOnlySpan<T>(Address, Shape.Size);
        }

        public void Reshape(Shape shape)
        {
            throw new NotImplementedException();
        }

        public IStorage Alias(Shape shape)
        {
            throw new NotImplementedException();
        }

        public IStorage Alias()
        {
            throw new NotImplementedException();
        }

        public void Allocate(IArraySlice values, Shape shape)
        {
            throw new NotImplementedException();
        }

        public IStorage Cast(Type type)
        {
            throw new NotImplementedException();
        }

        public ArraySlice<T> CloneData<T>() where T : unmanaged
        {
            throw new NotImplementedException();
        }

        public bool GetBoolean(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public byte GetByte(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public int GetInt32(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public int GetInt64(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public float GetSingle(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public double GetDouble(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public ValueType GetAtIndex(int index)
        {
            throw new NotImplementedException();
        }

        public T GetAtIndex<T>(int index)
        {
            throw new NotImplementedException();
        }

        public ValueType GetValue<T>(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public ArraySlice<T> GetData<T>() where T : unmanaged
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(Array values)
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(Array values, Type dtype)
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(Array values, NPTypeCode typeCode)
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(IArraySlice values)
        {
            throw new NotImplementedException();
        }

        public void SetAtIndex(object value, int index)
        {
            throw new NotImplementedException();
        }

        public void SetBoolean(bool value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetByte(byte value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetInt32(int value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetInt64(long value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetSingle(float value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetDouble(double value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetData(IArraySlice value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetData(NDArray value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetValue(ValueType value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetValue(object value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public IStorage Cast(NPTypeCode typeCode)
        {
            throw new NotImplementedException();
        }

        public IStorage CastIfNecessary(NPTypeCode typeCode)
        {
            throw new NotImplementedException();
        }

        public IArraySlice CloneData()
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(IArraySlice values, Type dtype)
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(NDArray nd)
        {
            throw new NotImplementedException();
        }

        public void SetData(object value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetValue<T>(T value, params int[] indices) where T : unmanaged
        {
            throw new NotImplementedException();
        }

        T IStorage.GetValue<T>(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public ValueType GetValue(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public IArraySlice GetData()
        {
            throw new NotImplementedException();
        }

        public NDArray GetData(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void Reshape(Shape shape, bool copy)
        {
            throw new NotImplementedException();
        }

        public void CopyTo<T>(T[] array) where T : unmanaged
        {
            throw new NotImplementedException();
        }

        public void CopyTo<T>(IMemoryBlock slice)
        {
            throw new NotImplementedException();
        }

        public unsafe void CopyTo<T>(void* address)
        {
            throw new NotImplementedException();
        }

        public void ExpandDimension(int axis)
        {
            throw new NotImplementedException();
        }

        public IStorage GetView(params Slice[] slices)
        {
            throw new NotImplementedException();
        }

        public IStorage GetView(string slicing_notation)
        {
            throw new NotImplementedException();
        }

        public unsafe T[] ToArray<T>() where T : unmanaged
        {
            throw new NotImplementedException();
        }

        public IStorage Clone()
        {
            throw new NotImplementedException();
        }

        public unsafe IStorage GetData(int* dims, int ndims)
        {
            throw new NotImplementedException();
        }

        IStorage IStorage.GetData(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetShapeUnsafe(Shape shape)
        {
            throw new NotImplementedException();
        }
    }
}
