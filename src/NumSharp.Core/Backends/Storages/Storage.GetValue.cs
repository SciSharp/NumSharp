using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Backends
{
    /*
     * Performance Rank
     * 1. Super fast 10x than 2  
     *  var span = nd.Read<int>();
        for (int j = 0; j < size; j++)
            span[j]

     * 2. Faster
     *  for (int j = 0; j < size; j++)
     *      GetAtIndex<T>(j)
     */
    public abstract partial class Storage
    {
        public bool GetBoolean(params int[] indices)
            => (bool)_internalArray[_shape.GetOffset(indices)];

        public byte GetByte(params int[] indices)
            => (byte)_internalArray[_shape.GetOffset(indices)];

        public virtual int GetInt32(params int[] indices)
            => (int)_internalArray[_shape.GetOffset(indices)];

        public long GetInt64(params int[] indices)
            => (long)_internalArray[_shape.GetOffset(indices)];

        public float GetSingle(params int[] indices)
            => (float)_internalArray[_shape.GetOffset(indices)];

        public double GetDouble(params int[] indices)
            => (double)_internalArray[_shape.GetOffset(indices)];

        public unsafe ValueType GetAtIndex(int index)
            => _typecode switch
            {
                NPTypeCode.Boolean => *((bool*)Address + _shape.TransformOffset(index)),
                NPTypeCode.Byte => *((byte*)Address + _shape.TransformOffset(index)),
                NPTypeCode.Char => *((char*)Address + _shape.TransformOffset(index)),
                NPTypeCode.Int32 => *((int*)Address + _shape.TransformOffset(index)),
                NPTypeCode.Int64 => *((long*)Address + _shape.TransformOffset(index)),
                NPTypeCode.Single => *((float*)Address + _shape.TransformOffset(index)),
                NPTypeCode.Double => *((double*)Address + _shape.TransformOffset(index)),
                _ => throw new NotSupportedException()
            };

        public unsafe T GetAtIndex<T>(int index) where T : unmanaged
            => *((T*)Address + _shape.TransformOffset(index));

        public T GetValue<T>(params int[] indices) where T : unmanaged
            => GetAtIndex<T>(_shape.GetOffset(indices));

        public ValueType GetValue(params int[] indices)
            => GetAtIndex(_shape.GetOffset(indices));
    }
}
