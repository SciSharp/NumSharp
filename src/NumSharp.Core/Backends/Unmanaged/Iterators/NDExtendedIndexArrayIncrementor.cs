using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends.Unmanaged {
    public class NDExtendedIndexArrayIncrementor
    {
        private readonly int _extendBy;
        private readonly int nonExtendedLength;
        private readonly int[] dimensions;
        private readonly int resetto;
        public int[] Index;
        private int subcursor;

        /// <param name="extendBy">By how many items should <see cref="Index"/> be extended</param>
        public NDExtendedIndexArrayIncrementor(ref Shape shape, int extendBy)
        {
            _extendBy = extendBy;
            dimensions = shape.dimensions;
            nonExtendedLength = dimensions.Length;
            Index = new int[nonExtendedLength + extendBy];
            resetto = subcursor = dimensions.Length - 1;
        }

        /// <param name="dims">The dims has to be not extended, use <see cref="Array.Resize{T}"/> if it already extended</param>
        /// <param name="extendBy">By how many items should <see cref="Index"/> be extended</param>
        public NDExtendedIndexArrayIncrementor(int[] dims, int extendBy)
        {
            dimensions = dims;
            _extendBy = extendBy;
            nonExtendedLength = dimensions.Length;
            Index = new int[nonExtendedLength + extendBy];
            resetto = subcursor = dimensions.Length - 1;
        }

        public void Reset()
        {
            Index = new int[nonExtendedLength + _extendBy];
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
                        //TODO somehow can we skip all ones entierly?
                        Reset();
                        return Next();
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