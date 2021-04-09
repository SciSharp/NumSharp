using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public abstract partial class Storage
    {
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

        public virtual ValueType GetAtIndex(int index)
        {
            throw new NotImplementedException();
        }

        public T GetAtIndex<T>(int index)
            => Read<T>()[index];

        public ValueType GetValue<T>(params int[] indices)
        {
            throw new NotImplementedException();
        }

        T IStorage.GetValue<T>(params int[] indices)
        {
            var offset = _shape.GetOffset(indices);
            return GetAtIndex<T>(offset);
        }

        public ValueType GetValue(params int[] indices)
        {
            throw new NotImplementedException();
        }
    }
}
