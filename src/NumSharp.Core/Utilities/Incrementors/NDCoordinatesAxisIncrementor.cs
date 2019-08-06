using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    public class NDCoordinatesAxisIncrementor
    {
        public int Axis;
        private readonly Action<NDCoordinatesAxisIncrementor> endCallback;
        private readonly int[] dimensions;
        private readonly int resetto;
        public readonly Slice[] Slices;
        public readonly int[] Index;
        private int subcursor;

        public NDCoordinatesAxisIncrementor(ref Shape shape, int axis)
        {
            if (shape.IsEmpty || shape.size == 0)
                throw new InvalidOperationException("Can't construct NDCoordinatesAxisIncrementor with an empty shape.");

            if (shape.NDim == 1)
                throw new InvalidOperationException("Can't construct NDCoordinatesAxisIncrementor with a vector shape.");

            if (shape.IsScalar)
                throw new InvalidOperationException("Can't construct NDCoordinatesAxisIncrementor with a scalar shape.");

            if (axis < 0 || axis >= shape.dimensions.Length)
                throw new ArgumentOutOfRangeException(nameof(axis));

            Axis = axis;

            dimensions = shape.dimensions;
            Index = new int[dimensions.Length];
            if (axis == dimensions.Length - 1)
                resetto = subcursor = dimensions.Length - 2;
            else
                resetto = subcursor = dimensions.Length - 1;

            Slices = new Slice[dimensions.Length];
            for (int i = 0; i <= resetto; i++)
                Slices[i] = Slice.Index(0); //it has to be new instances because we increment them individually.
            Slices[Axis] = Slice.All;
        }

        public NDCoordinatesAxisIncrementor(ref Shape shape, int axis, Action<NDCoordinatesAxisIncrementor> endCallback) : this(ref shape, axis)
        {
            this.endCallback = endCallback;
        }

        public NDCoordinatesAxisIncrementor(int[] dims, int axis)
        {
            if (dims == null)
                throw new InvalidOperationException("Can't construct NDCoordinatesAxisIncrementor with an empty shape.");

            if (dims.Length == 1)
                throw new InvalidOperationException("Can't construct NDCoordinatesAxisIncrementor with a vector shape.");

            if (axis < 0 || axis >= dims.Length)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (dims.Length == 0)
                dims = new int[] {1};

            dimensions = dims;
            Axis = axis;
            Index = new int[dims.Length];
            if (axis == dimensions.Length - 1)
                resetto = subcursor = dimensions.Length - 2;
            else
                resetto = subcursor = dimensions.Length - 1;

            Slices = new Slice[dims.Length];
            for (int i = 0; i <= resetto; i++)
                Slices[i] = Slice.Index(0); //it has to be new instances because we increment them individually.
            Slices[Axis] = Slice.All;
        }

        public NDCoordinatesAxisIncrementor(int[] dims, int axis, Action<NDCoordinatesAxisIncrementor> endCallback) : this(dims, axis)
        {
            this.endCallback = endCallback;
        }

        public void Reset()
        {
            Array.Clear(Index, 0, Index.Length);
            for (int i = 0; i <= resetto; i++)
                Slices[i] = Slice.Index(0); //it has to be new instances because we increment them individually.

            Slices[Axis] = Slice.All;
            subcursor = resetto;
        }

        [MethodImpl((MethodImplOptions)512)]
        public Slice[] Next()
        {
            if (subcursor <= -1)
                return null;

            if (++Index[subcursor] >= dimensions[subcursor])
            {
                _repeat:
                Index[subcursor] = 0;
                Slices[subcursor] = Slice.Index(0);
                do
                {
                    if (--subcursor <= -1)
                    {
                        //TODO somehow can we skip all ones?
                        endCallback?.Invoke(this);
                        if (subcursor >= 0) //if callback has resetted it
                            return Slices;
                        return null;
                    }
                } while (dimensions[subcursor] <= 1 || Axis == subcursor);

                ++Slices[subcursor];
                ++Index[subcursor];
                if (Index[subcursor] >= dimensions[subcursor])
                    goto _repeat;

                subcursor = resetto;
            }
            else
                ++Slices[subcursor];

            return Slices;
        }
    }
}
