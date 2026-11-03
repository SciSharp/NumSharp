using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Broadcast any number of arrays against each other (NumPy 2.4.2's <c>numpy.broadcast_arrays</c>): an array
        ///     already of the common shape comes back as itself, every other one as a view, writeable only when its array is.
        /// </summary>
        /// <param name="ndArrays">The arrays to broadcast.</param>
        /// <returns>One array per input: the input itself when its shape is the broadcast shape, else a view on it. Views are
        ///     typically not contiguous, and more than one element of a broadcasted array may refer to a single memory
        ///     location: copy first to write to them independently. A view of a read-only array is read-only.</returns>
        /// <exception cref="IncorrectShapeException">The arrays do not broadcast together.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html. See
        ///     <see cref="NDArray.broadcast_arrays(NDArray[])"/> for the writeable rule (parity QA-34). A result that IS an
        ///     input must not be disposed as if it were a new view.</remarks>
        public static NDArray[] broadcast_arrays(params NDArray[] ndArrays)
        {
            using var plain = NDMatrixBoundary.PlainFor(ndArrays);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // The statement is the array object's (NDArray.broadcast_arrays, Backends/NDArray.StrideTricks.cs).
            return NDArray.broadcast_arrays(ndArrays);
        }

        /// <summary>
        ///     Broadcast two arrays against each other (NumPy 2.4.2's <c>numpy.broadcast_arrays(lhs, rhs)</c>): an array
        ///     already of the common shape comes back as itself, the other as a view, writeable only when its array is.
        /// </summary>
        /// <param name="lhs">An array to broadcast.</param>
        /// <param name="rhs">An array to broadcast.</param>
        /// <returns>The two results: each the input itself when its shape is the broadcast shape, else a view on it. Views
        ///     are typically not contiguous, and more than one element of a broadcasted array may refer to a single memory
        ///     location: copy first to write to them independently. A view of a read-only array is read-only.</returns>
        /// <exception cref="IncorrectShapeException">The arrays do not broadcast together.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html; the same statement as the
        ///     params overload (<see cref="NDArray.BroadcastArraysMember"/> per operand).</remarks>
        public static (NDArray Lhs, NDArray Rhs) broadcast_arrays(NDArray lhs, NDArray rhs)
        {
            using var plain = NDMatrixBoundary.PlainFor(lhs, rhs);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            var (leftShape, rightShape) = Shape.Broadcast(lhs.Shape, rhs.Shape);
            // NumPy 2.4.2's rule per operand: a matching operand as itself, else a view writeable iff the operand is.
            return (NDArray.BroadcastArraysMember(lhs, leftShape), NDArray.BroadcastArraysMember(rhs, rightShape));
        }
    }
}
