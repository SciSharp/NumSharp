using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public abstract partial class Storage
    {
        public bool GetBoolean(params int[] indices)
            => (bool)_internalArray[_shape.GetOffset(indices)];

        public byte GetByte(params int[] indices)
            => (byte)_internalArray[_shape.GetOffset(indices)];

        public int GetInt32(params int[] indices)
            => (int)_internalArray[_shape.GetOffset(indices)];

        public long GetInt64(params int[] indices)
            => (long)_internalArray[_shape.GetOffset(indices)];

        public float GetSingle(params int[] indices)
            => (float)_internalArray[_shape.GetOffset(indices)];

        public double GetDouble(params int[] indices)
            => (double)_internalArray[_shape.GetOffset(indices)];

        public ValueType GetAtIndex(int index)
            => (ValueType)_internalArray[index];

        public T GetAtIndex<T>(int index)
            => (T)_internalArray[index];

        public T GetValue<T>(params int[] indices) where T : unmanaged
            => GetAtIndex<T>(_shape.GetOffset(indices));

        public ValueType GetValue(params int[] indices)
            => GetAtIndex(_shape.GetOffset(indices));
    }
}
