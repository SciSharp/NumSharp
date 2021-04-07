using System;
using System.Runtime.InteropServices;
using NumSharp;

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
            DType = NPTypeCode.Int32;
        }

        public StorageOfBoolean(bool x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfBoolean(bool[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape)
            => Init(new bool[shape.Size], shape);

        unsafe void Init(bool[] x, Shape? shape = null)
        {
            DType = NPTypeCode.Boolean;
            Shape = shape ?? new Shape(x.Length);
            data = x;
            // var handle = GCHandle.Alloc(x, GCHandleType.Pinned);
            // address = handle.AddrOfPinnedObject().ToPointer();
            // fixed (int* ptr = &x[0])
            // address = ptr;
            // handle.Free();
        }

        public override TensorEngine Engine => BackendFactory.GetEngine(DType);

        public unsafe override IStorage Alias()
        {
            var r = new StorageOfBoolean();
            r.Shape = Shape;
            r.Address = address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }

        public unsafe override IStorage Alias(Shape shape)
        {
            var r = new StorageOfBoolean();
            r.Shape = shape;
            r.Address = address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }
    }
}
