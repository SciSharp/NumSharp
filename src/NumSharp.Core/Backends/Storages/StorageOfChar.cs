using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public class StorageOfChar : Storage
    {
        public StorageOfChar()
        {
            _typecode = NPTypeCode.Char;
        }

        public StorageOfChar(char x)
            => Init(new[] { x }, NumSharp.Shape.Scalar);

        public StorageOfChar(char[] x, Shape? shape = null)
            => Init(x, shape);

        public override void Allocate(Shape shape, Type dtype = null)
            => Init(new char[shape.Size], shape);

        unsafe void Init(char[] x, Shape? shape = null)
        {
            _typecode = NPTypeCode.Char;
            _shape = shape ?? new Shape(x.Length);
            _internalArray = ArraySlice.FromArray(x);
            _address = _internalArray.Address;
        }
    }
}
