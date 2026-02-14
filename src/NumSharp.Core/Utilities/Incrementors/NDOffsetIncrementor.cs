using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    public class NDOffsetIncrementor
    {
        private readonly ValueCoordinatesIncrementor incr;
        private readonly int[] index;
        private bool hasNext;
        private readonly Shape shape;

        public NDOffsetIncrementor(Shape shape)
        {
            this.shape = shape;
            incr = new ValueCoordinatesIncrementor(shape.dimensions);
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

        [MethodImpl(Optimize)]
        public int Next()
        {
            if (!hasNext)
                return -1;

            int offset = 0;
            if (shape.IsSliced)
            {
                offset = shape.GetOffset(index);
            }
            else
            {
                unchecked
                {
                    for (int i = 0; i < index.Length; i++)
                        offset += shape.strides[i] * index[i];
                }
            }

            if (incr.Next() == null)
                hasNext = false;

            return offset;
        }
    }

    public class NDOffsetIncrementorAutoresetting
    {
        private readonly ValueCoordinatesIncrementor incr;
        private readonly int[] index;
        private readonly Shape shape;

        public NDOffsetIncrementorAutoresetting(Shape shape)
        {
            this.shape = shape;
            incr = new ValueCoordinatesIncrementor(shape.dimensions, (ref ValueCoordinatesIncrementor incrementor) => incrementor.Reset());
            index = incr.Index;
        }

        public NDOffsetIncrementorAutoresetting(int[] dims) : this(new Shape(dims)) { }


        public bool HasNext => true;

        public void Reset()
        {
            incr.Reset();
        }

        [MethodImpl(Optimize)]
        public int Next()
        {
            int offset = 0;
            if (shape.IsSliced)
            {
                offset = shape.GetOffset(index);
            }
            else
            {
                unchecked
                {
                    for (int i = 0; i < index.Length; i++)
                        offset += shape.strides[i] * index[i];
                }
            }

            incr.Next();

            return offset;
        }
    }
}
