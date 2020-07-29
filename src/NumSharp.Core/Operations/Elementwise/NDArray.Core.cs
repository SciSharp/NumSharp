using System;
using System.Diagnostics;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities.Maths;

namespace NumSharp
{
    public partial class NDArray
    {
        protected static BinaryOperationIndex OpBinary = new BinaryOperationIndex(typeof(NDArray), nameof(BinaryOperation));
        protected static BinaryOperationIndex OpBinaryLeft = new BinaryOperationIndex(typeof(NDArray), nameof(BinaryOperationLeft));
        protected static BinaryOperationIndex OpBinaryRight = new BinaryOperationIndex(typeof(NDArray), nameof(BinaryOperationRight));

        /// <summary>
        /// Aplies the (x, y) => z projection delegate to each element of <see cref="NDArray"/> and converts it to <typeparamref name="TResult"/>.
        /// </summary>
        /// <typeparam name="TLElement">The type of a left <see cref="NDArray"/> element.</typeparam>
        /// <typeparam name="TRElement">The type of a right scalar element.</typeparam>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to the projector.</param>
        /// <param name="rhs">The right <typeparamref name="TRElement"/> scalar value to pass to the projector.</param>
        /// <param name="projector">The projection delegate that handles the array elements.</param>
        /// <returns>The resulting <see cref="NDArray{TDType}"/> that contains the projection result.</returns>
        private static unsafe NDArray BinaryOperationLeft<TLElement, TRElement, TResult>(NDArray lhs, TRElement rhs, BinaryOperator<TLElement, TRElement, TResult> @operator)
            where TLElement : unmanaged
            where TRElement : unmanaged
            where TResult : unmanaged
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));
            if (@operator is null)
                throw new ArgumentNullException(nameof(@operator));

            if (typeof(TLElement).GetTypeCode() != lhs.GetTypeCode)
                throw new ArgumentException($"The left argument array type {lhs.GetTypeCode} does not match the expected type {typeof(TLElement).GetTypeCode()}.", nameof(lhs));

            var result = new NDArray(typeof(TResult).GetTypeCode(), lhs.Shape.Clean());

            var lhsAddress = (TLElement*)lhs.Address;
            var resultAddress = (TResult*)result.Address;

            @operator.ParallelFor(0, result.size, resultAddress, lhsAddress, rhs);

            return result;
        }

        /// <summary>
        /// Aplies the (x, y) => z projection delegate to each element of <see cref="NDArray"/> and converts it to <typeparamref name="TResult"/>.
        /// </summary>
        /// <typeparam name="TLElement">The type of a left scalar element.</typeparam>
        /// <typeparam name="TRElement">The type of a right <see cref="NDArray"/> element.</typeparam>
        /// <param name="lhs">The right <typeparamref name="TRElement"/> scalar value to pass to the projector.</param>
        /// <param name="rhs">The left <see cref="NDArray"/> to pass to the projector.</param>
        /// <param name="projector">The projection delegate that handles the array elements.</param>
        /// <returns>The resulting <see cref="NDArray{TDType}"/> that contains the projection result.</returns>
        private static unsafe NDArray BinaryOperationRight<TLElement, TRElement, TResult>(TLElement lhs, NDArray rhs, BinaryOperator<TLElement, TRElement, TResult> @operator)
            where TLElement : unmanaged
            where TRElement : unmanaged
            where TResult : unmanaged
        {
            if (rhs is null)
                throw new ArgumentNullException(nameof(rhs));
            if (@operator is null)
                throw new ArgumentNullException(nameof(@operator));

            if (typeof(TLElement).GetTypeCode() != rhs.GetTypeCode)
                throw new ArgumentException($"The right argument array type {rhs.GetTypeCode} does not match the expected type {typeof(TRElement).GetTypeCode()}.", nameof(lhs));

            var result = new NDArray(typeof(TResult).GetTypeCode(), rhs.Shape.Clean());

            var rhsAddress = (TRElement*)rhs.Address;
            var resultAddress = (TResult*)result.Address;

            @operator.ParallelFor(0, result.size, resultAddress, lhs, rhsAddress);

            return result;
        }

        /// <summary>
        /// Aplies the (x, y) => z projection delegate to each element of left and right <see cref="NDArray"/> arguments.
        /// </summary>
        /// <typeparam name="TLElement">The type of a left <see cref="NDArray"/> element.</typeparam>
        /// <typeparam name="TRElement">The type of a right <see cref="NDArray"/> element.</typeparam>
        /// <typeparam name="TResult">The type of a resultint <see cref="NDArray"/> element.</typeparam>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to the projector.</param>
        /// <param name="rhs">The right <see cref="NDArray"/> to pass to the projector.</param>
        /// <param name="operator">The projection delegate that handles the array elements.</param>
        /// <returns>The resulting <see cref="NDArray"/> that contains the projection result.</returns>
        public static unsafe NDArray BinaryOperation<TLElement, TRElement, TResult>(NDArray lhs, NDArray rhs, BinaryOperator<TLElement, TRElement, TResult> @operator)
            where TLElement : unmanaged
            where TRElement : unmanaged
            where TResult : unmanaged
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs is null)
                throw new ArgumentNullException(nameof(rhs));
            if (@operator is null)
                throw new ArgumentNullException(nameof(@operator));

            if (typeof(TLElement).GetTypeCode() != lhs.GetTypeCode)
                throw new ArgumentException($"The left argument array type {lhs.GetTypeCode} does not match the expected type {typeof(TLElement).GetTypeCode()}.", nameof(lhs));
            if (typeof(TRElement).GetTypeCode() != rhs.GetTypeCode)
                throw new ArgumentException($"The right argument array type {rhs.GetTypeCode} does not match the expected type {typeof(TLElement).GetTypeCode()}.", nameof(rhs));

            var (ls, rs) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
            var resultShape = ls.Clean();
            var result = new NDArray(typeof(TResult).GetTypeCode(), resultShape);

            var lhsAddress = (TLElement*)lhs.Address;
            var rhsAddress = (TRElement*)rhs.Address;
            var resultAddress = (TResult*)result.Address;

            switch ((l: ls.IsLinear, r: rs.IsLinear))
            {
                // ls.IsLinear && rs.IsLinear
                case var isLinear when isLinear.l && isLinear.r:
                    Debug.Assert(ls.size == result.size && rs.size == result.size);
                    switch (result.size)
                    {
                        case var length when rs.IsBroadcasted && rs.BroadcastInfo.OriginalShape.IsScalar:
                            @operator.ParallelFor(0, length, resultAddress, lhsAddress, *rhsAddress);
                            break;
                        case var length when ls.IsBroadcasted && ls.BroadcastInfo.OriginalShape.IsScalar:
                            @operator.ParallelFor(0, length, resultAddress, *lhsAddress, rhsAddress);
                            break;
                        case var length:
                            @operator.ParallelFor(0, length, resultAddress, lhsAddress, rhsAddress);
                            break;
                    }
                    break;

                // ls.IsLinear && !rs.IsLinear
                case var isLinear when isLinear.l:
                    switch (result.size)
                    {
                        case var length when rs.IsBroadcasted && rs.BroadcastInfo.OriginalShape.IsScalar:
                            @operator.ParallelFor(0, length, resultAddress, lhsAddress, *rhsAddress);
                            break;
                        default:
                            @operator.IncrementFor(resultAddress, lhsAddress, rhsAddress, ref rs);
                            break;
                    }
                    break;

                // !ls.IsLinear && rs.IsLinear
                case var isLinear when isLinear.r:
                    switch (result.size)
                    {
                        case var length when ls.IsBroadcasted && ls.BroadcastInfo.OriginalShape.IsScalar && *lhsAddress is var lval:
                            @operator.ParallelFor(0, length, resultAddress, *lhsAddress, rhsAddress);
                            break;

                        default:
                            @operator.IncrementFor(resultAddress, lhsAddress, ref ls, rhsAddress);
                            break;
                    }
                    break;

                // !ls.IsLinear && !rs.IsLinear
                default:
                    @operator.IncrementFor(resultAddress, lhsAddress, ref ls, rhsAddress, ref rs);
                    break;
            }
            return result;
        }
    }
}
