using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfDouble : Storage
    {
        double[] data;

        public override unsafe void* Address
        {
            get
            {
                if (_address != null)
                    return _address;

                fixed (double* ptr = &data[0])
                    return ptr;
            }
            set => base.Address = value;
        }

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
            Shape = shape ?? new Shape(x.Length);
            data = x;
            _internalArray = ArraySlice.FromArray(data);
            _address = _internalArray.Address;
        }

        public override ValueType GetAtIndex(int index)
            => data[index];

        public override void SetAtIndex(ValueType value, int index)
            => data[index] = (double)value;
    }
}
