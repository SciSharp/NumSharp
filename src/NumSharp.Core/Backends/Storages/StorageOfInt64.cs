using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfInt64 : Storage
    {
        long[] data;

        public override unsafe void* Address
        {
            get
            {
                if (_address != null)
                    return _address;

                fixed (long* ptr = &data[0])
                    return ptr;
            }
            set => base.Address = value;
        }

        public StorageOfInt64()
        {
            _typecode = NPTypeCode.Int64;
        }

        public StorageOfInt64(long x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfInt64(long[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape)
            => Init(new long[shape.Size], shape);

        unsafe void Init(long[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Int64;
            Shape = shape ?? new Shape(x.Length);
            data = x;
            _internalArray = ArraySlice.FromArray(data);
            _address = _internalArray.Address;
        }

        public override ValueType GetAtIndex(int index)
            => data[index];
    }
}
