using System;

namespace NumSharp.Backends
{
    public class StorageOfBoolean : Storage
    {
        bool[] data;

        public override unsafe void* Address
        {
            get
            {
                fixed (bool* ptr = &data[0])
                    return ptr;
            }
            set => base.Address = value;
        }

        public StorageOfBoolean()
        {
            _typecode = NPTypeCode.Int32;
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
        }

        public unsafe override IStorage Alias()
        {
            var r = new StorageOfBoolean();
            r.Shape = Shape;
            r.Address = _address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }

        public unsafe override IStorage Alias(Shape shape)
        {
            var r = new StorageOfBoolean();
            r.Shape = shape;
            r.Address = _address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }
    }
}
