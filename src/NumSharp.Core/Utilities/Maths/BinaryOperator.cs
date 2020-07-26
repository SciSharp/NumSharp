using System;
using System.Threading.Tasks;

namespace NumSharp.Utilities.Maths
{
    /// <summary>
    /// Encapsulates the <see cref="BinaryOperator"/> operator caller data necessary to call the operator based on argument type codes.
    /// </summary>
    public abstract class BinaryOperator
    {
        /// <summary>
        /// Gets the <see cref="BinaryOperator"/> operator caller type code signature key.
        /// </summary>
        public abstract (NPTypeCode, NPTypeCode) Key { get; }

        /// <summary>
        /// Gets the <see cref="BinaryOperator"/> operator caller return type code.
        /// </summary>
        public abstract NPTypeCode ReturnCode { get; }
    }

    /// <summary>
    /// Encapsulates the <see cref="BinaryOperator{TLElement, TRElement, TResult}"/> operator caller data necessary to call the operator based on argument type codes.
    /// </summary>
    /// <typeparam name="TLElement">The type of a left operator operand.</typeparam>
    /// <typeparam name="TRElement">The type of a right operator operand.</typeparam>
    /// <typeparam name="TResult">THe type of an operator result.</typeparam>
    public class BinaryOperator<TLElement, TRElement, TResult> : BinaryOperator
        where TLElement : unmanaged
        where TRElement : unmanaged
        where TResult : unmanaged
    {
        private readonly Func<TLElement, TRElement, TResult> _operator;

        /// <summary>
        /// Creates the new instance of an  <see cref="BinaryOperator{TLElement, TRElement, TResult}"/> operator caller.
        /// </summary>
        /// <param name="operator">The operator delegate to invoke when operator is executed.</param>
        public BinaryOperator(Func<TLElement, TRElement, TResult> @operator)
        {
            _operator = @operator ?? throw new ArgumentNullException(nameof(@operator));
        }

        /// <summary>
        /// Gets the <see cref="BinaryOperator"/> operator caller type code signature key.
        /// </summary>
        public override (NPTypeCode, NPTypeCode) Key => (typeof(TLElement).GetTypeCode(), typeof(TRElement).GetTypeCode());

        /// <summary>
        /// Gets the <see cref="BinaryOperator"/> operator caller return type code.
        /// </summary>
        public override NPTypeCode ReturnCode => typeof(TResult).GetTypeCode();

        /// <summary>
        /// Runs the <see cref="Parallel.For(int, int, Action{int})"/> operation for every array element and a scalar element.
        /// </summary>
        /// <param name="index">The starting index for an array operation.</param>
        /// <param name="length">The sequence length for an array operation.</param>
        /// <param name="result">The operator result pointer to place the value to.</param>
        /// <param name="lhs">The left operator argument pointer to get the value from.</param>
        /// <param name="rhs">The right operator argument to get the value from.</param>
        public unsafe void ParallelFor(int index, int length, TResult* result, TLElement* lhs, TRElement rhs)
        {
            Parallel.For(index, length, i => *(result + i) = _operator(*(lhs + i), rhs));
        }

        /// <summary>
        /// Runs the <see cref="Parallel.For(int, int, Action{int})"/> operation for a scalar element and every array element.
        /// </summary>
        /// <param name="index">The starting index for an array operation.</param>
        /// <param name="length">The sequence length for an array operation.</param>
        /// <param name="result">The operator result pointer to place the value to.</param>
        /// <param name="lhs">The left operator argument to get the value from.</param>
        /// <param name="rhs">The right operator argument pointer to get the value from.</param>
        public unsafe void ParallelFor(int index, int length, TResult* result, TLElement lhs, TRElement* rhs)
        {
            Parallel.For(index, length, i => *(result + i) = _operator(lhs, *(rhs + i)));
        }

        /// <summary>
        /// Runs the <see cref="Parallel.For(int, int, Action{int})"/> operation for every array element in both sides.
        /// </summary>
        /// <param name="index">The starting index for an array operation.</param>
        /// <param name="length">The sequence length for an array operation.</param>
        /// <param name="result">The operator result pointer to place the value to.</param>
        /// <param name="lhs">The left operator argument pointer to get the value from.</param>
        /// <param name="rhs">The right operator argument pointer to get the value from.</param>
        public unsafe void ParallelFor(int index, int length, TResult* result, TLElement* lhs, TRElement* rhs)
        {
            Parallel.For(index, length, i => *(result + i) = _operator(*(lhs + i), *(rhs + i)));
        }

        /// <summary>
        /// Runs the coordinate incrementor in linear and non-linear arrays.
        /// </summary>
        /// <param name="result">The operator result pointer to place the value to.</param>
        /// <param name="lhs">The left operator argument pointer to get the value from.</param>
        /// <param name="rhs">The right operator argument pointer to get the value from.</param>
        /// <param name="rs">The shape of a right array argument.</param>
        public unsafe void IncrementFor(TResult* result, TLElement* lhs, TRElement* rhs, ref Shape rs)
        {
            var incrementor = new NDCoordinatesIncrementor(ref rs);
            var current = incrementor.Index;
            Func<int[], int> rsOffset = rs.GetOffset;
            do
                *result++ = _operator(*lhs++, *(rhs + rsOffset(current)));
            while (incrementor.Next() != null);
        }

        /// <summary>
        /// Runs the coordinate incrementor in non-linear and linear arrays.
        /// </summary>
        /// <param name="result">The operator result pointer to place the value to.</param>
        /// <param name="lhs">The left operator argument pointer to get the value from.</param>
        /// <param name="ls">The shape of a left array argument.</param>
        /// <param name="rhs">The right operator argument pointer to get the value from.</param>
        public unsafe void IncrementFor(TResult* result, TLElement* lhs, ref Shape ls, TRElement* rhs)
        {
            var incrementor = new NDCoordinatesIncrementor(ref ls);
            var current = incrementor.Index;
            Func<int[], int> lsOffset = ls.GetOffset;
            do
                *result++ = _operator(*(lhs + lsOffset(current)), *rhs++);
            while (incrementor.Next() != null);
        }

        /// <summary>
        /// Runs the coordinate incrementor in non-linear and non-linear arrays.
        /// </summary>
        /// <param name="result">The operator result pointer to place the value to.</param>
        /// <param name="lhs">The left operator argument pointer to get the value from.</param>
        /// <param name="ls">The shape of a left array argument.</param>
        /// <param name="rhs">The right operator argument pointer to get the value from.</param>
        /// <param name="rs">The shape of a right array argument.</param>
        public unsafe void IncrementFor(TResult* result, TLElement* lhs, ref Shape ls, TRElement* rhs, ref Shape rs)
        {
            var incrementor = new NDCoordinatesIncrementor(ref ls);
            var current = incrementor.Index;
            Func<int[], int> rsOffset = rs.GetOffset;
            Func<int[], int> lsOffset = ls.GetOffset;
            do
                *result++ = _operator(*(lhs + lsOffset(current)), *(rhs + rsOffset(current)));
            while (incrementor.Next() != null);
        }
    }
}
