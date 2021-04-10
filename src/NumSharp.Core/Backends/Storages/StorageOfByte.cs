using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfByte : Storage
    {
        public StorageOfByte()
        {
            _typecode = NPTypeCode.Byte;
        }

        public StorageOfByte(byte x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfByte(byte[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape, Type dtype = null)
            => Init(new byte[shape.Size], shape);

        unsafe void Init(byte[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Byte;
            _shape = shape ?? new Shape(x.Length);
            _internalArray = ArraySlice.FromArray(x);
            _address = _internalArray.Address;
        }
    }
}
