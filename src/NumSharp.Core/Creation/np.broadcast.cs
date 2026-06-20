using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Produce an object that mimics broadcasting.
        /// </summary>
        /// <returns>Broadcast the input parameters against one another, and return an object that encapsulates the result. Amongst others, it has shape and nd properties, and may be used as an iterator.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html</remarks>
        public static Broadcast broadcast(NDArray nd1, NDArray nd2)
        {
            return new Broadcast(nd1, nd2);
        }

        public class Broadcast
        {
            private readonly NDArray _op1;
            private readonly NDArray _op2;
            private NDIterator[] _iters;

            /// <summary>
            ///     Parameterless constructor retained for source/binary compatibility
            ///     (the public surface previously exposed an implicit one). An instance
            ///     built this way has no operands, so <see cref="iters"/> stays null
            ///     unless explicitly assigned.
            /// </summary>
            public Broadcast() { }

            internal Broadcast(NDArray op1, NDArray op2)
            {
                _op1 = op1;
                _op2 = op2;
                shape = Shape.ResolveReturnShape(op1.Shape, op2.Shape);
            }

            public Shape shape { get; internal set; }

            //It shouldn't be used unless it is a very advanced code...
            public int index => throw new NotSupportedException("NumSharp does not implement iterators exactly like numpy does.");

            /// <summary>
            ///     Per-operand iterators (NumPy's <c>broadcast.iters</c>). Built lazily
            ///     on first access: <see cref="broadcast"/> only resolves the broadcast
            ///     <see cref="shape"/> eagerly, so the common shape/size/ndim usage
            ///     allocates no iterators at all. Each entry is an <see cref="NDIterator"/>
            ///     (itself NpyIter-backed).
            /// </summary>
            public NDIterator[] iters
            {
                get => _iters ??= (_op1 is not null && _op2 is not null)
                    ? new[] { _op1.AsIterator(), _op2.AsIterator() }
                    : null;
                internal set => _iters = value;
            }

            public int nd
            {
                get => ndim;
            }

            public int ndim => shape.NDim;
            public long size => shape.size;

            void reset()
            {
                if (_iters == null)
                    return;
                for (int i = 0; i < ndim; i++)
                    _iters[i].Reset();
            }
        }
    }
}
