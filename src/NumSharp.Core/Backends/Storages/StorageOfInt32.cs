using System;

namespace NumSharp.Backends
{
    public class StorageOfInt32 : Storage
    {
        int[] data;

        public override unsafe void* Address
        {
            get
            {
                fixed (int* ptr = &data[0])
                    return ptr;
            }
            set => base.Address = value;
        }

        public StorageOfInt32()
        {
            _typecode = NPTypeCode.Int32;
        }

        public StorageOfInt32(int x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfInt32(int[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape)
            => Init(new int[shape.Size], shape);

        unsafe void Init(int[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Int32;
            Shape = shape ?? new Shape(x.Length);
            data = x;
        }

        public unsafe override IStorage Alias()
        {
            var r = new StorageOfInt32();
            r.Shape = Shape;
            r.Address = _address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }

        public unsafe override IStorage Alias(Shape shape)
        {
            var r = new StorageOfInt32();
            r.Shape = shape;
            r.Address = _address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }

        public override IStorage Clone()
        {
            var x = new int[data.Length];
            Buffer.BlockCopy(data, 0, x, 0, data.Length * sizeof(int));
            var r = new StorageOfInt32(x, Shape);
            return r;
        }
    }
}
