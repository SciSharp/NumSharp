using System.Runtime.CompilerServices;

namespace NumSharp.Backends.Unmanaged
{
    public class NDOffsetIncrementor
    {
        private readonly NDIndexArrayIncrementor incr;
        private readonly int[] strides;
        private readonly int[] index;
        private bool hasNext;

        public NDOffsetIncrementor(ref Shape shape) : this(shape.dimensions, shape.strides)
        { }

        public NDOffsetIncrementor(Shape shape) : this(shape.dimensions, shape.strides)
        { }

        public NDOffsetIncrementor(int[] dims, int[] strides)
        {
            this.strides = strides;
            incr = new NDIndexArrayIncrementor(dims);
            index = incr.Index;
            hasNext = true;
        }

        public bool HasNext => hasNext;

        public void Reset()
        {
            incr.Reset();
            hasNext = true;
        }

        [MethodImpl((MethodImplOptions)512)]
        public int Next()
        {
            if (!hasNext)
                return -1;

            int offset = 0;
            unchecked
            {
                for (int i = 0; i < index.Length; i++)
                    offset += strides[i] * index[i];
            }

            if (incr.Next() == null)
                hasNext = false;

            //TODO! we need to support slice here!

            return offset;
        }
    }
}
