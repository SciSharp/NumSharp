using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfInt32 : Storage
    {
        public StorageOfInt32()
        {
            _typecode = NPTypeCode.Int32;
        }

        public StorageOfInt32(int x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfInt32(int[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape, Type dtype = null)
            => Init(new int[shape.Size], shape);

        unsafe void Init(int[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Int32;
            _shape = shape ?? new Shape(x.Length);
            _internalArray = ArraySlice.FromArray(x);
            _address = _internalArray.Address;
        }
    }
}
