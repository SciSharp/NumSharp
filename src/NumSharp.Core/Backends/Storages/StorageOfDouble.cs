using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfDouble : Storage
    {
        public StorageOfDouble()
        {
            _typecode = NPTypeCode.Double;
        }

        public StorageOfDouble(double x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfDouble(double[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape)
            => Init(new double[shape.Size], shape);

        unsafe void Init(double[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Double;
            _shape = shape ?? new Shape(x.Length);
            _internalArray = ArraySlice.FromArray(x);
            _address = _internalArray.Address;
        }
    }
}
