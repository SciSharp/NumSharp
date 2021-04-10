using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfSingle : Storage
    {
        public StorageOfSingle()
        {
            _typecode = NPTypeCode.Single;
        }

        public StorageOfSingle(float x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfSingle(float[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape, Type dtype = null)
            => Init(new float[shape.Size], shape);

        unsafe void Init(float[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Single;
            _shape = shape ?? new Shape(x.Length);
            _internalArray = ArraySlice.FromArray(x);
            _address = _internalArray.Address;
        }
    }
}
