using System;

namespace NumSharp.Utilities.Maths
{
    /// <summary>
    /// Encapsulates the <see cref="BinaryLOperation{TLElement, TRElement, TResult}"/> operation data necessary to call the operation based on argument type codes.
    /// </summary>
    /// <typeparam name="TLElement">The type of a left operation operand.</typeparam>
    /// <typeparam name="TRElement">The type of a right operation operand.</typeparam>
    /// <typeparam name="TResult">THe type of an operation result.</typeparam>
    public class BinaryLOperation<TLElement, TRElement, TResult> : BinaryOperation
        where TLElement : unmanaged
        where TRElement : unmanaged
        where TResult : unmanaged
    {
        private readonly Func<NDArray, TRElement, BinaryOperator<TLElement, TRElement, TResult>, NDArray> _operation;

        /// <summary>
        /// Creates the new instance of an  <see cref="BinaryOperation{TLElement, TRElement, TResult}"/> operation.
        /// </summary>
        /// <param name="operation">The operation delegate to invoke when operation is executed.</param>
        public BinaryLOperation(Func<NDArray, TRElement, BinaryOperator<TLElement, TRElement, TResult>, NDArray> operation)
        {
            _operation = operation ?? throw new ArgumentNullException(nameof(operation));
        }

        /// <summary>
        /// Gets the <see cref="BinaryOperation"/> operation type signature key.
        /// </summary>
        public override (NPTypeCode, NPTypeCode) GetTypeKey => (typeof(TLElement).GetTypeCode(), typeof(TRElement).GetTypeCode());

        /// <summary>
        /// Invokes the operator on <see cref="NDArray"/> by matching an <see cref="BinaryOperator"/> with the requested operator signature.
        /// </summary>
        /// <param name="lhs">The left scalar operand to call the operator for.</param>
        /// <param name="rhs">The right <see cref="NDArray"/> operand to call the operator for.</param>
        /// <param name="operator">The <see cref="BinaryOperator"/> operator that maps the operator signature.</param>
        /// <returns>The resulting <see cref="NDArray"/> returned.</returns>
        public override NDArray Invoke(ValueType lhs, NDArray rhs, BinaryOperator @operator)
        {
            throw new NotSupportedException($"The binary operation on a scalar and an array is not supported.");
        }

        /// <summary>
        /// Invokes the operator on <see cref="NDArray"/> by matching an <see cref="BinaryOperator"/> with the requested operator signature.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> operand to call the operator for.</param>
        /// <param name="rhs">The right scalar operand to call the operator for.</param>
        /// <param name="operator">The <see cref="BinaryOperator"/> operator that maps the operator signature.</param>
        /// <returns>The resulting <see cref="NDArray"/> returned.</returns>
        public override NDArray Invoke(NDArray lhs, ValueType rhs, BinaryOperator @operator)
        {
            if (@operator is null)
                throw new ArgumentNullException(nameof(@operator));

            if (@operator is BinaryOperator<TLElement, TRElement, TResult> generic)
                return _operation(lhs, (TRElement)rhs, generic);
            else
                throw new ArgumentException($"The operator signature is not compatible with operation.");
        }

        /// <summary>
        /// Invokes the operator on <see cref="NDArray"/> by matching an <see cref="BinaryOperator"/> with the requested operator signature.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> operand to call the operator for.</param>
        /// <param name="rhs">The right <see cref="NDArray"/> operand to call the operator for.</param>
        /// <param name="operator">The <see cref="BinaryOperator"/> operator that maps the operator signature.</param>
        /// <returns>The resulting <see cref="NDArray"/> returned.</returns>
        public override NDArray Invoke(NDArray lhs, NDArray rhs, BinaryOperator @operator)
        {
            throw new NotSupportedException($"The binary operation on two arrays is not supported.");
        }

        /// <summary>
        /// Invokes the operator on <see cref="NDArray"/> by matching an <see cref="BinaryOperator{TLElement, TRElement, TResult}"/> with the requested operator signature.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> operand to call the operator for.</param>
        /// <param name="rhs">The right <see cref="NDArray"/> operand to call the operator for.</param>
        /// <param name="operator">The <see cref="BinaryOperator{TLElement, TRElement, TResult}"/> operator that maps the operator signature.</param>
        /// <returns>The resulting <see cref="NDArray"/> returned.</returns>
        public NDArray Invoke(NDArray lhs, TRElement rhs, BinaryOperator<TLElement, TRElement, TResult> @operator)
        {
            if (@operator is null)
                throw new ArgumentNullException(nameof(@operator));

            return _operation(lhs, rhs, @operator);
        }
    }
}
