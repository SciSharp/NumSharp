using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends.Unmanaged
{
    public class NDOffsetIncrementor
    {
        private readonly NDIndexArrayIncrementor incr;
        private readonly int[] strides;
        private readonly int shapeSize;
        public long Offset;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public NDOffsetIncrementor(ref Shape shape) : this(shape.dimensions, shape.strides, shape.size)
        { }

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
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
        public long Next()
        {
            var index = incr.Next();
            if (index == null)
                return -1;

            long offset = 0;
            unchecked
            {
                for (long i = 0; i < index.Length; i++)
                    offset += strides[i] * index[i];
            }

            Offset = offset;
            return offset;
        }
    }

    public class NDIndexArrayIncrementor
    {
        private readonly int[] dimensions;
        private readonly int resetto;
        public long[] Index;
        private int subcursor;
        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public NDIndexArrayIncrementor(ref Shape shape)
        {
            dimensions = shape.dimensions;
            Index = new long[dimensions.Length];
            resetto = subcursor = dimensions.Length - 1;
        }

        public NDIndexArrayIncrementor(int[] dims)
        {
            dimensions = dims;
            Index = new long[dims.Length];
            resetto = subcursor = dimensions.Length - 1;
        }

        public void Reset()
        {
            Index = new long[dimensions.Length];
            subcursor = resetto;
        }

        [MethodImpl((MethodImplOptions)512)]
        public long[] Next()
        {
            if (++Index[subcursor] >= dimensions[subcursor])
            {
                _repeat:
                Index[subcursor] = 0;

                do
                {
                    if (--subcursor <= -1) //TODO somehow can we skip all ones?
                        return null; //finished
                } while (dimensions[subcursor] <= 1);

                ++Index[subcursor];
                if (Index[subcursor] >= dimensions[subcursor])
                    goto _repeat;

                subcursor = resetto;
            }

            //Console.Write("[");
            //for (int i = 0; i < dimensions.Length; i++)
            //{
            //    Console.Write($"{Index[i]}, ");
            //}
            //
            //Console.WriteLine("]");
            return Index;
        }
    }
}
