using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfBoolean : Storage
    {
        bool[] data;

        public override unsafe void* Address
        {
            get
            {
                if (_address != null)
                    return _address;

                fixed (bool* ptr = &data[0])
                    return ptr;
            }
            set => base.Address = value;
        }

        public StorageOfBoolean()
        {
            _typecode = NPTypeCode.Boolean;
        }

        public StorageOfBoolean(bool x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfBoolean(bool[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape)
            => Init(new bool[shape.Size], shape);

        unsafe void Init(bool[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Boolean;
            Shape = shape ?? new Shape(x.Length);
            data = x;
            _internalArray = ArraySlice.FromArray(data);
            _address = _internalArray.Address;
        }

        public override ValueType GetAtIndex(int index)
            => data[index];
    }
}
