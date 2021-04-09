using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfSingle : Storage
    {
        float[] data;

        public override unsafe void* Address
        {
            get
            {
                if (_address != null)
                    return _address;

                fixed (float* ptr = &data[0])
                    return ptr;
            }
            set => base.Address = value;
        }

        public StorageOfSingle()
        {
            _typecode = NPTypeCode.Single;
        }

        public StorageOfSingle(float x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfSingle(float[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape)
            => Init(new float[shape.Size], shape);

        unsafe void Init(float[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Single;
            _shape = shape ?? new Shape(x.Length);
            data = x;
            _internalArray = ArraySlice.FromArray(data);
            _address = _internalArray.Address;
        }

        public override ValueType GetAtIndex(int index)
            => data[index];
    }
}
