using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfByte : Storage
    {
        byte[] data;

        public override unsafe void* Address
        {
            get
            {
                if (_address != null)
                    return _address;

                fixed (byte* ptr = &data[0])
                    return ptr;
            }
            set => base.Address = value;
        }

        public StorageOfByte()
        {
            _typecode = NPTypeCode.Byte;
        }

        public StorageOfByte(byte x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfByte(byte[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape)
            => Init(new byte[shape.Size], shape);

        unsafe void Init(byte[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Byte;
            Shape = shape ?? new Shape(x.Length);
            data = x;
            _internalArray = ArraySlice.FromArray(data);
            _address = _internalArray.Address;
        }

        public override ValueType GetAtIndex(int index)
            => data[index];
    }
}
