using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfInt64 : Storage
    {
        public StorageOfInt64()
        {
            _typecode = NPTypeCode.Int64;
        }

        public StorageOfInt64(long x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfInt64(long[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape, Type dtype = null)
            => Init(new long[shape.Size], shape);

        unsafe void Init(long[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Int64;
            _shape = shape ?? new Shape(x.Length);
            _internalArray = ArraySlice.FromArray(x);
            _address = _internalArray.Address;
        }
    }
}
