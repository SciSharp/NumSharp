using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends.Unmanaged
{
    public class NDIndexArrayIncrementor
    {
        private readonly Action<NDIndexArrayIncrementor> endCallback;
        private readonly int[] dimensions;
        private readonly int resetto;
        public readonly int[] Index;
        private int subcursor;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public NDIndexArrayIncrementor(ref Shape shape)
        {
            dimensions = shape.dimensions;
            Index = new int[dimensions.Length];
            resetto = subcursor = dimensions.Length - 1;
        }

        public NDIndexArrayIncrementor(ref Shape shape, Action<NDIndexArrayIncrementor> endCallback) : this(ref shape)
        {
            this.endCallback = endCallback;
        }

        public NDIndexArrayIncrementor(int[] dims)
        {
            dimensions = dims;
            Index = new int[dims.Length];
            resetto = subcursor = dimensions.Length - 1;
        }

        public NDIndexArrayIncrementor(int[] dims, Action<NDIndexArrayIncrementor> endCallback) : this(dims) {
            this.endCallback = endCallback;
        }

        public void Reset()
        {
            Array.Clear(Index, 0, Index.Length);
            subcursor = resetto;
        }

        [MethodImpl((MethodImplOptions)512)]
        public int[] Next()
        {
            if (++Index[subcursor] >= dimensions[subcursor])
            {
                _repeat:
                Index[subcursor] = 0;

                do
                {
                    if (--subcursor <= -1)
                    {
                        //TODO somehow can we skip all ones?
                        endCallback?.Invoke(this);
                        return null;
                    }
                } while (dimensions[subcursor] <= 1);

                ++Index[subcursor];
                if (Index[subcursor] >= dimensions[subcursor])
                    goto _repeat;

                subcursor = resetto;
            }

            return Index;
        }
    }

    public class NDIndexArrayIncrementorAutoResetting
    {
        private readonly int[] dimensions;
        private readonly int resetto;
        public readonly int[] Index;
        private int subcursor;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public NDIndexArrayIncrementorAutoResetting(ref Shape shape)
        {
            dimensions = shape.dimensions;
            Index = new int[dimensions.Length];
            resetto = subcursor = dimensions.Length - 1;
        }

        public NDIndexArrayIncrementorAutoResetting(int[] dims)
        {
            dimensions = dims;
            Index = new int[dims.Length];
            resetto = subcursor = dimensions.Length - 1;
        }

        public void Reset()
        {
            Array.Clear(Index, 0, Index.Length);
            subcursor = resetto;
        }

        [MethodImpl((MethodImplOptions)512)]
        public int[] Next()
        {
            if (++Index[subcursor] >= dimensions[subcursor])
            {
                _repeat:
                Index[subcursor] = 0;

                do
                {
                    if (--subcursor <= -1)
                    {
                        //TODO somehow can we skip all ones?
                        Reset();
                        return Next(); //finished
                    }
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
