using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Produce an object that mimics broadcasting.
        /// </summary>
        /// <returns>Broadcast the input parameters against one another, and return an object that encapsulates the result. Amongst others, it has shape and nd properties, and may be used as an iterator.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast.html</remarks>
        public static Broadcast broadcast(NDArray nd1, NDArray nd2)
        {
            return new Broadcast { shape = DefaultEngine.ResolveReturnShape(nd1.Shape, nd2.Shape), iters = new[] { nd1.AsIterator(), nd2.AsIterator() } };
        }

        public class Broadcast
        {
            public Shape shape { get; internal set; }

            //It shouldn't be used unless it is a very advanced code...
            public int index => throw new NotSupportedException("NumSharp does not implement iterators exactly like numpy does.");
            public NDIterator[] iters { get; internal set; }

            public int nd
            {
                get => ndim;
            }

            public int ndim => shape.NDim;
            public int size => shape.size;

            void reset()
            {
                if (iters == null)
                    return;
                for (int i = 0; i < ndim; i++)
                    iters[i].Reset();
            }
        }
    }
}
