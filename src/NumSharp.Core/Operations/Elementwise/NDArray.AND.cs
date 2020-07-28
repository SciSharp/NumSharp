using System;
using NumSharp.Generic;
using NumSharp.Utilities.Maths;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Performs bitwise and operation on the elements of two <see cref="NDArray"/> values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray"/> that contains the the operation result.</returns>
        public static NDArray operator &(NDArray lhs, NDArray rhs)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs is null)
                throw new ArgumentNullException(nameof(rhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetTypeCode;
            var @operator = Operator.OpBitwiseAnd.Get(ltc, rtc);
            var operation = OpBinary.Get(ltc, rtc, @operator.ReturnCode);
            return operation.Invoke(lhs, rhs, @operator);
        }

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, char rhs)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetType().GetTypeCode();
            var @operator = Operator.OpBitwiseAnd.Get(ltc, rtc);
            var operation = OpBinaryLeft.Get(ltc, rtc, @operator.ReturnCode);
            return operation.Invoke(lhs, (ValueType)rhs, @operator);
        }

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, byte rhs)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetType().GetTypeCode();
            var @operator = Operator.OpBitwiseAnd.Get(ltc, rtc);
            var operation = OpBinaryLeft.Get(ltc, rtc, @operator.ReturnCode);
            return operation.Invoke(lhs, (ValueType)rhs, @operator);
        }

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, short rhs)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetType().GetTypeCode();
            var @operator = Operator.OpBitwiseAnd.Get(ltc, rtc);
            var operation = OpBinaryLeft.Get(ltc, rtc, @operator.ReturnCode);
            return operation.Invoke(lhs, (ValueType)rhs, @operator);
        }

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, ushort rhs)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetType().GetTypeCode();
            var @operator = Operator.OpBitwiseAnd.Get(ltc, rtc);
            var operation = OpBinaryLeft.Get(ltc, rtc, @operator.ReturnCode);
            return operation.Invoke(lhs, (ValueType)rhs, @operator);
        }

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, int rhs)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetType().GetTypeCode();
            var @operator = Operator.OpBitwiseAnd.Get(ltc, rtc);
            var operation = OpBinaryLeft.Get(ltc, rtc, @operator.ReturnCode);
            return operation.Invoke(lhs, (ValueType)rhs, @operator);
        }

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, uint rhs)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetType().GetTypeCode();
            var @operator = Operator.OpBitwiseAnd.Get(ltc, rtc);
            var operation = OpBinaryLeft.Get(ltc, rtc, @operator.ReturnCode);
            return operation.Invoke(lhs, (ValueType)rhs, @operator);
        }

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, long rhs)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetType().GetTypeCode();
            var @operator = Operator.OpBitwiseAnd.Get(ltc, rtc);
            var operation = OpBinaryLeft.Get(ltc, rtc, @operator.ReturnCode);
            return operation.Invoke(lhs, (ValueType)rhs, @operator);
        }

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, ulong rhs)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetType().GetTypeCode();
            var @operator = Operator.OpBitwiseAnd.Get(ltc, rtc);
            var operation = OpBinaryLeft.Get(ltc, rtc, @operator.ReturnCode);
            return operation.Invoke(lhs, (ValueType)rhs, @operator);
        }
    }
}
