using System;
using System.Linq;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Produce an object that mimics broadcasting.
        /// </summary>
        /// <param name="arrays">
        ///     The arrays to broadcast against one another. NumPy accepts 0..64 operands; the same
        ///     range is honored here (65+ raises <see cref="ValueError"/>).
        /// </param>
        /// <returns>Broadcast the input parameters against one another, and return an object that encapsulates the result. Amongst others, it has shape and nd properties, and may be used as an iterator.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html</remarks>
        public static Broadcast broadcast(params NDArray[] arrays)
        {
            return new Broadcast(arrays);
        }

        /// <summary>
        ///     Two-operand overload — the most common case. Retained as an explicit overload so
        ///     <c>np.broadcast(a, b)</c> stays binary-compatible and skips the params-array allocation.
        /// </summary>
        public static Broadcast broadcast(NDArray nd1, NDArray nd2)
        {
            return new Broadcast(nd1, nd2);
        }

        public class Broadcast
        {
            private readonly NDArray[] _ops;
            private NpyFlatIterator[] _iters;

            /// <summary>
            ///     Parameterless constructor retained for source/binary compatibility
            ///     (the public surface previously exposed an implicit one). An instance
            ///     built this way has no operands, so <see cref="iters"/> stays null
            ///     unless explicitly assigned.
            /// </summary>
            public Broadcast() { }

            internal Broadcast(NDArray op1, NDArray op2) : this(new[] { op1, op2 }) { }

            /// <summary>
            ///     Broadcast N operands together (NumPy's <c>np.broadcast(*args)</c>). The result
            ///     <see cref="shape"/> is the broadcast of every operand's shape (resolved eagerly;
            ///     iterators are built lazily on first <see cref="iters"/> access). Incompatible
            ///     shapes raise <see cref="IncorrectShapeException"/>; more than 64 operands raise
            ///     <see cref="ValueError"/> — both matching NumPy 2.x.
            /// </summary>
            internal Broadcast(params NDArray[] ops)
            {
                _ops = ops ?? Array.Empty<NDArray>();

                // NumPy 2.x caps the multi-iterator at 64 operands ("Need at least 0 and at most
                // 64 array objects."). 0 operands is legal and yields a 0-d (scalar) broadcast.
                if (_ops.Length > 64)
                    throw new ValueError("Need at least 0 and at most 64 array objects.");

                shape = _ops.Length == 0
                    ? Shape.Scalar
                    : Shape.ResolveReturnShape(_ops.Select(o => o.Shape).ToArray());
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
            ///     NpyIter-aligned replacement for the removed NDIterator. There are <see cref="numiter"/>
            ///     entries (0 when constructed without operands).
            /// </summary>
            public NpyFlatIterator[] iters
            {
                get => _iters ??= _ops is null
                    ? null
                    : _ops.Select(o => new NpyFlatIterator(broadcast_to(o, shape))).ToArray();
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
            ///     equal to <c>len(iters)</c>). Equals the operand count passed to
            ///     <see cref="np.broadcast(NDArray[])"/> (0 for the parameterless constructor).
            /// </summary>
            public int numiter => _ops?.Length ?? 0;
        }
    }
}
