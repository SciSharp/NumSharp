using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public interface IStorage
    {
        void Allocate(Shape shape, Type dtype = null);
        void Allocate(IArraySlice values, Shape shape, bool copy = false);

        unsafe void* Address { get; set; }
        unsafe ReadOnlySpan<T> Read<T>();

        Type DType { get; }
        Shape Shape { get; set; }
        ref Shape ShapeReference { get; }
        int Count { get; set; }
        int DTypeSize { get; }
        void Reshape(Shape shape, bool copy = false);
        void Reshape(ref Shape shape, bool copy = false);
        TensorEngine Engine { get; }
        NPTypeCode TypeCode { get; }
        IStorage Alias(Shape shape);
        IStorage Alias();
        
        IStorage Cast(Type type);
        IStorage Cast(NPTypeCode typeCode);
        IStorage CastIfNecessary(NPTypeCode typeCode);

        bool CopyTo(IntPtr ptr);
        bool CopyTo(IMemoryBlock block);
        unsafe bool CopyTo<T>(IMemoryBlock<T> block) where T : unmanaged;
        unsafe bool CopyTo<T>(T* address) where T : unmanaged;
        bool CopyTo<T>(T[] array) where T : unmanaged;
        
        IStorage Clone();

        void ExpandDimension(int axis);

        IArraySlice InternalArray { get; set; }
        void SetInternalArray(IArraySlice array);

        ArraySlice<T> CloneData<T>() where T : unmanaged;
        IArraySlice CloneData();
        bool GetBoolean(params int[] indices);
        byte GetByte(params int[] indices);
        int GetInt32(params int[] indices);
        long GetInt64(params int[] indices);
        float GetSingle(params int[] indices);
        double GetDouble(params int[] indices);
        ValueType GetAtIndex(int index);
        unsafe T GetAtIndex<T>(int index) where T : unmanaged;
        T GetValue<T>(params int[] indices) where T : unmanaged;
        ValueType GetValue(params int[] indices);
        ArraySlice<T> GetData<T>() where T : unmanaged;
        IArraySlice GetData();
        IStorage GetData(params int[] indices);
        unsafe IStorage GetData(int* dims, int ndims);

        IStorage GetView(params Slice[] slices);
        IStorage GetView(string slicing_notation);
        IStorage GetViewInternal(params Slice[] slices);

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
