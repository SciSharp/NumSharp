using System;
using System.Runtime.InteropServices;
using NumSharp;

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
            DType = NPTypeCode.Int32;
        }

        public StorageOfInt32(int x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfInt32(int[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape)
            => Init(new int[shape.Size], shape);

        unsafe void Init(int[] x, Shape? shape = null)
        {
            DType = NPTypeCode.Int32;
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
            var r = new StorageOfInt32();
            r.Shape = Shape;
            r.Address = address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }

        public unsafe override IStorage Alias(Shape shape)
        {
            var r = new StorageOfInt32();
            r.Shape = shape;
            r.Address = address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }
    }
}
