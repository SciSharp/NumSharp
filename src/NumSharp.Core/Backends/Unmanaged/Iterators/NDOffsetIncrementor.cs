using System.Runtime.CompilerServices;

namespace NumSharp.Backends.Unmanaged
{
    public class NDOffsetIncrementor
    {
        private readonly NDIndexArrayIncrementor incr;
        private readonly int[] strides;
        private readonly int shapeSize;
        public int Offset;

        public NDOffsetIncrementor(ref Shape shape) : this(shape.dimensions, shape.strides, shape.size)
        { }

        public NDOffsetIncrementor(Shape shape) : this(shape.dimensions, shape.strides, shape.size)
        { }

        public NDOffsetIncrementor(int[] dims, int[] strides, int shapeSize)
        {
            this.strides = strides;
            this.shapeSize = shapeSize;
            incr = new NDIndexArrayIncrementor(dims);
        }

        public bool HasNext => Offset < shapeSize;

        public void Reset()
        {
            incr.Reset();
            Offset = 0;
        }

        [MethodImpl((MethodImplOptions)512)]
        public int Next()
        {
            var index = incr.Next();
            if (index == null)
                return -1;

            int offset = 0;
            unchecked
            {
                for (int i = 0; i < index.Length; i++)
                    offset += strides[i] * index[i];
            }

            Offset = offset;
            return offset;
        }
    }
}
