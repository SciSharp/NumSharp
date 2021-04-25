using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    public struct ValueNDCoordinatesIncrementor
    {
        public delegate void EndCallbackHandler(ref ValueNDCoordinatesIncrementor incr);
        private readonly EndCallbackHandler endCallback;
        private readonly int[] dimensions;
        private readonly int resetto;
        public readonly int[] Index;
        private int subcursor;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public ValueNDCoordinatesIncrementor(ref Shape shape)
        {
            if (shape.IsEmpty || shape.size == 0)
                throw new InvalidOperationException("Can't construct NDCoordinatesIncrementor with an empty shape.");

            dimensions = shape.IsScalar ? new[] {1} : shape.dimensions;
            Index = new int[dimensions.Length];
            resetto = subcursor = dimensions.Length - 1;
            endCallback = null;
        }

        public ValueNDCoordinatesIncrementor(ref Shape shape, EndCallbackHandler endCallback) : this(ref shape)
        {
            this.endCallback = endCallback;
        }

        public ValueNDCoordinatesIncrementor(int[] dims)
        {
            if (dims == null)
                throw new InvalidOperationException("Can't construct NDCoordinatesIncrementor with an empty shape.");

            if (dims.Length == 0)
                dims = new int[] {1};

            dimensions = dims;
            Index = new int[dims.Length];
            resetto = subcursor = dimensions.Length - 1;
            endCallback = null;
        }

        public ValueNDCoordinatesIncrementor(int[] dims, EndCallbackHandler endCallback) : this(dims)
        {
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
            if (subcursor <= -1)
                return null;

            if (++Index[subcursor] >= dimensions[subcursor])
            {
                _repeat:
                Index[subcursor] = 0;

                do
                {
                    if (--subcursor <= -1)
                    {
                        //TODO somehow can we skip all ones?
                        endCallback?.Invoke(ref this);
                        if (subcursor >= 0) //if callback has resetted it
                            return Index;
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

    public struct ValueNDCoordinatesIncrementorAutoResetting
    {
        private readonly int[] dimensions;
        private readonly int resetto;
        public readonly int[] Index;
        private int subcursor;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public ValueNDCoordinatesIncrementorAutoResetting(ref Shape shape)
        {
            if (shape.IsEmpty || shape.size == 0)
                throw new InvalidOperationException("Can't construct NDCoordinatesIncrementorAutoResetting with an empty shape.");

            dimensions = shape.dimensions;
            Index = new int[dimensions.Length];
            resetto = subcursor = dimensions.Length - 1;
        }

        public ValueNDCoordinatesIncrementorAutoResetting(int[] dims)
        {
            if (dims == null)
                throw new InvalidOperationException("Can't construct NDCoordinatesIncrementorAutoResetting with an empty shape.");

            if (dims.Length == 0)
                dims = new int[] {1};

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
                        return Index; //finished
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
}
