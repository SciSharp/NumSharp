using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    public class NDExtendedCoordinatesIncrementor
    {
        private readonly Action<NDExtendedCoordinatesIncrementor> endCallback;
        private readonly int _extendBy;
        private readonly int nonExtendedLength;
        private readonly int[] dimensions;
        private readonly int resetto;
        public int[] Index;
        private int subcursor;
        public bool ResetEntireArray { get; set; }

        /// <param name="extendBy">By how many items should <see cref="Index"/> be extended</param>
        public NDExtendedCoordinatesIncrementor(Shape shape, int extendBy, Action<NDExtendedCoordinatesIncrementor> endCallback = null)
        {
            if (shape.IsEmpty || shape.size == 0)
                throw new InvalidOperationException("Can't construct NDExtendedCoordinatesIncrementor with an empty shape.");

            _extendBy = extendBy;
            dimensions = shape.IsScalar ? new int[] {1} : shape.dimensions;
            nonExtendedLength = dimensions.Length;
            Index = new int[nonExtendedLength + extendBy];
            resetto = subcursor = dimensions.Length - 1;
            this.endCallback = endCallback;
        }

        /// <param name="dims">The dims has to be not extended, use <see cref="Array.Resize{T}"/> if it already extended</param>
        /// <param name="extendBy">By how many items should <see cref="Index"/> be extended</param>
        public NDExtendedCoordinatesIncrementor(int[] dims, int extendBy, Action<NDExtendedCoordinatesIncrementor> endCallback = null) : this(new Shape(dims), extendBy, endCallback) { }

        public void Reset()
        {
            if (ResetEntireArray)
                Array.Clear(Index, 0, Index.Length);
            else
                Array.Clear(Index, 0, dimensions.Length);

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
                        endCallback?.Invoke(this);
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

        [MethodImpl((MethodImplOptions)512)]
        public int[] Next(params int[] extendedIndices)
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
                        endCallback?.Invoke(this);
                        if (subcursor >= 0)
                        {
                            //if callback has resetted it
                            for (int i = 0; i < extendedIndices.Length; i++) 
                                Index[nonExtendedLength + i] = extendedIndices[i];
                            return Index;
                        }

                        return null;
                    }
                } while (dimensions[subcursor] <= 1);

                ++Index[subcursor];
                if (Index[subcursor] >= dimensions[subcursor])
                    goto _repeat;

                subcursor = resetto;
            }

            for (int i = 0; i < extendedIndices.Length; i++)
                Index[nonExtendedLength + i] = extendedIndices[i];
            return Index;
        }
    }
}
