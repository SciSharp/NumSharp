using System;
using NumSharp.Backends.Iteration;

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
            private NpyFlatIterator[] _iters;

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
            ///     Per-operand flat iterators (NumPy's <c>broadcast.iters</c>): one entry per input,
            ///     each iterating that operand broadcast to the result <see cref="shape"/> in C-order
            ///     (e.g. np.broadcast([1,2,3], [[10],[20]]).iters[0] yields 1,2,3,1,2,3). Built lazily
            ///     on first access — np.broadcast() only resolves the shape eagerly — and backed by
            ///     <see cref="np.broadcast_to(NDArray, Shape)"/> + <see cref="NpyFlatIterator"/>, the
            ///     NpyIter-aligned replacement for the removed NDIterator.
            /// </summary>
            public NpyFlatIterator[] iters
            {
                get => _iters ??= (_op1 is not null && _op2 is not null)
                    ? new[]
                    {
                        new NpyFlatIterator(broadcast_to(_op1, shape)),
                        new NpyFlatIterator(broadcast_to(_op2, shape)),
                    }
                    : null;
                internal set => _iters = value;
            }

            public int nd
            {
                get => ndim;
            }

            public int ndim => shape.NDim;
            public long size => shape.size;

            /// <summary>
            ///     Number of operands being broadcast together (NumPy's <c>broadcast.numiter</c>,
            ///     equal to <c>len(iters)</c>). NumSharp's <see cref="np.broadcast(NDArray, NDArray)"/>
            ///     always combines exactly two operands, so this is 2 (NumPy itself accepts 1..32).
            /// </summary>
            public int numiter => 2;
        }
    }
}
