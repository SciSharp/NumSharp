using System;

namespace NumSharp.Backends
{
    public class StorageOfDouble : Storage
    {
        double[] data;

        public override unsafe void* Address
        {
            get
            {
                fixed (double* ptr = &data[0])
                    return ptr;
            }
            set => base.Address = value;
        }

        public StorageOfDouble()
        {
            DType = NPTypeCode.Double;
        }

        public StorageOfDouble(double x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfDouble(double[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape)
            => Init(new double[shape.Size], shape);

        unsafe void Init(double[] x, Shape? shape = null)
        {
            DType = NPTypeCode.Double;
            Shape = shape ?? new Shape(x.Length);
            data = x;
        }

        public override TensorEngine Engine => BackendFactory.GetEngine(DType);

        public unsafe override IStorage Alias()
        {
            var r = new StorageOfDouble();
            r.Shape = Shape;
            r.Address = address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }

        public unsafe override IStorage Alias(Shape shape)
        {
            var r = new StorageOfDouble();
            r.Shape = shape;
            r.Address = address;
            r.Count = Shape.size; //incase shape is sliced
            return r;
        }

        public override ValueType GetAtIndex(int index)
            => data[index];
    }
}
