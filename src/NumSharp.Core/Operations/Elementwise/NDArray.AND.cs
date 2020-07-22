using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Aplies the (x, y) => z projection delegate to each element of <see cref="NDArray"/> and converts it to <typeparamref name="TResult"/>.
        /// </summary>
        /// <typeparam name="TElement">The type of <see cref="NDArray"/> element.</typeparam>
        /// <typeparam name="TResult">The type of resulting element.</typeparam>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to projector.</param>
        /// <param name="rhs">The right <see cref="NDArray"/> to pass to projector.</param>
        /// <param name="projector">The projection delegate that handles the array elements.</param>
        /// <returns>The resulting <see cref="NDArray{TDType}"/> that contains the projection result.</returns>
        private static NDArray<TResult> Apply<TElement, TResult>(NDArray lhs, TElement rhs, Func<TElement, TElement, TResult> projector)
            where TElement : unmanaged
            where TResult : unmanaged
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));
            if (projector is null)
                throw new ArgumentNullException(nameof(projector));

            var result = new NDArray(typeof(TResult), lhs.shape);
            var storage = result.Storage.GetData<TResult>();

            var lha = lhs.Storage.GetData<TElement>();

            for (var i = 0; i < storage.Count; i++)
                storage[i] = projector(lha[i], rhs);

            return result.AsGeneric<TResult>();
        }

        /// <summary>
        /// Aplies the (x, y) => z projection delegate to each element of <see cref="NDArray"/> and a scalar and converts it to <typeparamref name="TResult"/>.
        /// </summary>
        /// <typeparam name="TElement">The type of <see cref="NDArray"/> element.</typeparam>
        /// <typeparam name="TResult">The type of resulting element.</typeparam>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to projector.</param>
        /// <param name="rhs">The right <typeparamref name="TElement"/> scalar value to pass to projector.</param>
        /// <param name="projector">The projection delegate that handles the array elements.</param>
        /// <returns>The resulting <see cref="NDArray{TDType}"/> that contains the projection result.</returns>
        private static NDArray<TResult> Apply<TElement, TResult>(NDArray lhs, NDArray rhs, Func<TElement, TElement, TResult> projector)
            where TElement : unmanaged
            where TResult : unmanaged
        {
            if (rhs is null)
                throw new ArgumentNullException(nameof(rhs));
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));
            if (projector is null)
                throw new ArgumentNullException(nameof(projector));

            var result = new NDArray(typeof(TResult), lhs.shape);
            var storage = result.Storage.GetData<TResult>();

            var lha = lhs.Storage.GetData<TElement>();
            var rha = rhs.Storage.GetData<TElement>();

            for (var i = 0; i < storage.Count; i++)
                storage[i] = projector(lha[i], rha[i]);

            return result.AsGeneric<TResult>();
        }

        /// <summary>
        /// Performs bitwize and operation on the elements of two <see cref="NDArray"/> values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the the operation result.</returns>
        public static NDArray<bool> operator &(NDArray lhs, NDArray rhs)
        {
            return Apply<bool, bool>(lhs, rhs, (r, l) => r & l);
        }

        /// <summary>
        /// Performs bitwize and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray<byte> operator &(NDArray lhs, byte rhs)
        {
            return Apply<byte, byte>(lhs, rhs, (r, l) => (byte)(r & l));
        }

        /// <summary>
        /// Performs bitwize and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray<ushort> operator &(NDArray lhs, ushort rhs)
        {
            return Apply<ushort, ushort>(lhs, rhs, (r, l) => (ushort)(r & l));
        }

        /// <summary>
        /// Performs bitwize and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray<uint> operator &(NDArray lhs, uint rhs)
        {
            return Apply<uint, uint>(lhs, rhs, (r, l) => r & l);
        }

        /// <summary>
        /// Performs bitwize and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray<ulong> operator &(NDArray lhs, ulong rhs)
        {
            return Apply<ulong, ulong>(lhs, rhs, (r, l) => r & l);
        }

        /// <summary>
        /// Performs bitwize and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray<char> operator &(NDArray lhs, char rhs)
        {
            return Apply<char, char>(lhs, rhs, (r, l) => (char)(r & l));
        }

        /// <summary>
        /// Performs bitwize and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray<short> operator &(NDArray lhs, short rhs)
        {
            return Apply<short, short>(lhs, rhs, (r, l) => (short)(r & l));
        }

        /// <summary>
        /// Performs bitwize and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray<int> operator &(NDArray lhs, int rhs)
        {
            return Apply<int, int>(lhs, rhs, (r, l) => r & l);
        }

        /// <summary>
        /// Performs bitwize and operation on the elements of <see cref="NDArray"/> and the scalar values.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> to pass to perform the operation on.</param>
        /// <param name="rhs">The right scalar value to pass to perform the operation on.</param>
        /// <returns>The resulting <see cref="NDArray{bool}"/> that contains the operation result.</returns>
        public static NDArray<long> operator &(NDArray lhs, long rhs)
        {
            return Apply<long, long>(lhs, rhs, (r, l) => r & l);
        }
    }
}
