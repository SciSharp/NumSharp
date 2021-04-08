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

        public StorageOfSingle(IArraySlice x, Shape shape)
            => _Allocate(shape, x);

        public override void Allocate(Shape shape)
            => Init(new float[shape.Size], shape);

        protected unsafe override void _Allocate(Shape shape, IArraySlice array)
        {
            _typecode = NPTypeCode.Single;
            _internalArray = (ArraySlice<float>)array;
            _shape = shape;
            _address = _internalArray.Address;
        }

        unsafe void Init(float[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Single;
            _shape = shape ?? new Shape(x.Length);
            data = x;
        }

        public unsafe override IStorage Alias()
        {
            var r = new StorageOfDouble();
            r.Shape = Shape;
            r.Address = _address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }

        public unsafe override IStorage Alias(Shape shape)
        {
            var r = new StorageOfDouble();
            r.Shape = shape;
            r.Address = _address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }

        public override ValueType GetAtIndex(int index)
            => data[index];
    }
}
