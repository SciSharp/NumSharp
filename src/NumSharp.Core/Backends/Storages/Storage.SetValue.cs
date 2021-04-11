using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public abstract partial class Storage
    {
        public void SetValue(ValueType value, params int[] indices)
        {
            var offset = _shape.GetOffset(indices);
            SetAtIndex(value, offset);
        }

        public void SetValue(object value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public unsafe void SetValue<T>(T value, params int[] indices) where T : unmanaged
            => *((T*) _address + _shape.GetOffset(indices)) = value;

        public void SetAtIndex(object value, int index)
            => _internalArray.SetIndex(index, value);

        public void SetAtIndex(ValueType value, int index)
            => _internalArray.SetIndex(index, value);

        public unsafe void SetBoolean(bool value, params int[] indices)
        {
            *((bool*)Address + _shape.GetOffset(indices)) = value;
        }

        public unsafe void SetByte(byte value, params int[] indices)
        {
            *((byte*)Address + _shape.GetOffset(indices)) = value;
        }

        public unsafe void SetInt32(int value, params int[] indices)
        {
            *((int*)Address + _shape.GetOffset(indices)) = value;
        }

        public unsafe void SetInt64(long value, params int[] indices)
        {
            *((long*)Address + _shape.GetOffset(indices)) = value;
        }

        public unsafe void SetSingle(float value, params int[] indices)
        {
            *((float*)Address + _shape.GetOffset(indices)) = value;
        }

        public unsafe void SetDouble(double value, params int[] indices)
        {
            *((double*)Address + _shape.GetOffset(indices)) = value;
        }
    }
}
