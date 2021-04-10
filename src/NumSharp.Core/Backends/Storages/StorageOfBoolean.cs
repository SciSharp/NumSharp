using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfBoolean : Storage
    {
        public StorageOfBoolean()
        {
            _typecode = NPTypeCode.Boolean;
        }

        public StorageOfBoolean(bool x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfBoolean(bool[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape, Type dtype = null)
            => Init(new bool[shape.Size], shape);

        unsafe void Init(bool[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Boolean;
            _shape = shape ?? new Shape(x.Length);
            _internalArray = ArraySlice.FromArray(x);
            _address = _internalArray.Address;
        }
    }
}
