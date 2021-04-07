using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public interface IStorage
    {
        void Allocate(Shape shape);
        unsafe void* Address { get; }
        unsafe ReadOnlySpan<T> Read<T>();

        Type DType { get; }
        Shape Shape { get; }
        int Count { get; }
        int DTypeSize { get; }
        void Reshape(Shape shape);
        void Reshape(Shape shape, bool copy);
        TensorEngine Engine { get; }
        NPTypeCode TypeCode { get; }
        IStorage Alias(Shape shape);
        IStorage Alias();
        void Allocate(IArraySlice values, Shape shape);
        IStorage Cast(Type type);
        IStorage Cast(NPTypeCode typeCode);
        IStorage CastIfNecessary(NPTypeCode typeCode);

        void CopyTo<T>(T[] array) where T : unmanaged;
        void CopyTo<T>(IMemoryBlock slice);
        unsafe void CopyTo<T>(void* address);

        IStorage Clone();

        void ExpandDimension(int axis);

        IArraySlice InternalArray { get; }
        ArraySlice<T> CloneData<T>() where T : unmanaged;
        IArraySlice CloneData();
        bool GetBoolean(params int[] indices);
        byte GetByte(params int[] indices);
        int GetInt32(params int[] indices);
        int GetInt64(params int[] indices);
        float GetSingle(params int[] indices);
        double GetDouble(params int[] indices);
        ValueType GetAtIndex(int index);
        T GetAtIndex<T>(int index);
        T GetValue<T>(params int[] indices) where T : unmanaged;
        ValueType GetValue(params int[] indices);
        ArraySlice<T> GetData<T>() where T : unmanaged;
        IArraySlice GetData();
        IStorage GetData(params int[] indices);
        unsafe IStorage GetData(int* dims, int ndims);

        IStorage GetView(params Slice[] slices);
        IStorage GetView(string slicing_notation);

        void ReplaceData(Array values);
        void ReplaceData(Array values, Type dtype);
        void ReplaceData(Array values, NPTypeCode typeCode);
        void ReplaceData(IArraySlice values);
        void ReplaceData(IArraySlice values, Type dtype);
        void ReplaceData(NDArray nd);
        void SetData(object value, params int[] indices);
        unsafe void SetAtIndex(object value, int index);
        void SetBoolean(bool value, params int[] indices);
        void SetByte(byte value, params int[] indices);
        void SetInt32(int value, params int[] indices);
        void SetInt64(long value, params int[] indices);
        void SetSingle(float value, params int[] indices);
        void SetDouble(double value, params int[] indices);
        void SetData(IArraySlice value, params int[] indices);
        void SetData(NDArray value, params int[] indices);
        void SetValue(ValueType value, params int[] indices);
        void SetValue(object value, params int[] indices);
        void SetValue<T>(T value, params int[] indices) where T : unmanaged;
        void SetShapeUnsafe(Shape shape);
        unsafe T[] ToArray<T>() where T : unmanaged;
    }
}
