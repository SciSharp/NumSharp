using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    public class NDOffsetIncrementor
    {
        private readonly NDCoordinatesIncrementor incr;
        private readonly int[] index;
        private bool hasNext;

        public NDOffsetIncrementor(Shape shape)
        {
            incr = new NDCoordinatesIncrementor(shape.dimensions);
            index = incr.Index;
            hasNext = true;
        }

        public NDOffsetIncrementor(int[] dims) : this(new Shape(dims)) {}

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
                    ;//offset += strides[i] * index[i];
            }

            if (incr.Next() == null)
                hasNext = false;

            //TODO! we need to support slice here!

            return offset;
        }
    }

    public class NDOffsetIncrementorAutoresetting
    {
        private readonly NDCoordinatesIncrementor incr;
        private readonly int[] index;

        public NDOffsetIncrementorAutoresetting(Shape shape)
        {
            incr = new NDCoordinatesIncrementor(shape.dimensions, incrementor => incrementor.Reset());
            index = incr.Index;
        }

        public NDOffsetIncrementorAutoresetting(int[] dims) : this(new Shape(dims)) { }


        public bool HasNext => true;

        public void Reset()
        {
            incr.Reset();
        }

        [MethodImpl((MethodImplOptions)512)]
        public int Next()
        {
            int offset = 0;
            unchecked
            {
                for (int i = 0; i < index.Length; i++)
                    ;//offset += strides[i] * index[i];
            }

            incr.Next();

            //TODO! we need to support slice here!

            return offset;
        }
    }
}
