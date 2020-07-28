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
        public static NDArray operator &(NDArray lhs, NDArray rhs) => OpBinary.Invoke(lhs, rhs, Operator.OpBitwiseAnd);

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, char rhs) => OpBinaryLeft.Invoke(lhs, (ValueType)rhs, Operator.OpBitwiseAnd);

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, byte rhs) => OpBinaryLeft.Invoke(lhs, (ValueType)rhs, Operator.OpBitwiseAnd);

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, short rhs) => OpBinaryLeft.Invoke(lhs, (ValueType)rhs, Operator.OpBitwiseAnd);

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, ushort rhs) => OpBinaryLeft.Invoke(lhs, (ValueType)rhs, Operator.OpBitwiseAnd);

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, int rhs) => OpBinaryLeft.Invoke(lhs, (ValueType)rhs, Operator.OpBitwiseAnd);

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, uint rhs) => OpBinaryLeft.Invoke(lhs, (ValueType)rhs, Operator.OpBitwiseAnd);

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, long rhs) => OpBinaryLeft.Invoke(lhs, (ValueType)rhs, Operator.OpBitwiseAnd);

        /// <summary>
        /// Performs bitwise and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray operator &(NDArray lhs, ulong rhs) => OpBinaryLeft.Invoke(lhs, (ValueType)rhs, Operator.OpBitwiseAnd);
    }
}
