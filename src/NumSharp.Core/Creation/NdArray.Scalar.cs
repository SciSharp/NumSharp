using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Creates a scalar <see cref="NDArray"/> of <see cref="value"/> and <see cref="dtype"/>.
        /// </summary>
        /// <param name="value">The value of the scalar</param>
        /// <param name="dtype">The type of the scalar.</param>
        /// <returns></returns>
        /// <remarks>In case when <see cref="value"/> is not <see cref="dtype"/>, <see cref="Convert.ChangeType(object,System.Type)"/> will be called.</remarks>
        public static NDArray Scalar(object value, Type dtype)
        {
            return new NDArray(UnmanagedStorage.Scalar(Convert.ChangeType(value, dtype)));
        }

        /// <summary>
        ///     Creates a scalar <see cref="NDArray"/> of <see cref="value"/> and <see cref="dtype"/>.
        /// </summary>
        /// <param name="value">The value of the scalar</param>
        /// <returns></returns>
        /// <remarks>In case when <see cref="value"/> is not <see cref="dtype"/>, <see cref="Convert.ChangeType(object,System.Type)"/> will be called.</remarks>
        public static NDArray Scalar(object value)
        {
            return new NDArray(UnmanagedStorage.Scalar(value));
        }

        /// <summary>
        ///     Creates a scalar <see cref="NDArray"/> of <see cref="value"/> and <see cref="dtype"/>.
        /// </summary>
        /// <param name="value">The value of the scalar</param>
        /// <returns></returns>
        /// <remarks>In case when <see cref="value"/> is not <see cref="dtype"/>, <see cref="Convert.ChangeType(object,System.Type)"/> will be called.</remarks>
        public static NDArray Scalar(ValueType value)
        {
            return new NDArray(UnmanagedStorage.Scalar(value));
        }

        /// <summary>
        ///     Creates a scalar <see cref="NDArray"/> of <see cref="value"/> and <see cref="dtype"/>.
        /// </summary>
        /// <param name="value">The value of the scalar</param>
        /// <returns></returns>
        /// <remarks>In case when <see cref="value"/> is not <see cref="dtype"/>, <see cref="Convert.ChangeType(object,System.Type)"/> will be called.</remarks>
        public static NDArray Scalar<T>(T value) where T : unmanaged
        {
            return new NDArray(UnmanagedStorage.Scalar(value));
        }

        /// <summary>
        ///     Creates a scalar <see cref="NDArray"/> of <see cref="value"/> and <see cref="dtype"/>.
        /// </summary>
        /// <param name="value">The value of the scalar, attempt to convert will be performed</param>
        /// <returns></returns>
        /// <remarks>In case when <see cref="value"/> is not <see cref="dtype"/>, <see cref="Convert.ChangeType(object,System.Type)"/> will be called.</remarks>
        public static NDArray Scalar<T>(object value) where T : unmanaged
        {
            return new NDArray(UnmanagedStorage.Scalar(value as T? ?? Converts.ChangeType<T>(value, InfoOf<T>.NPTypeCode)));
        }

        /// <summary>
        ///     Creates a scalar <see cref="NDArray"/> of <see cref="value"/> and <see cref="dtype"/>.
        /// </summary>
        /// <param name="value">The value of the scalar</param>
        /// <param name="typeCode">The type code of the scalar.</param>
        /// <returns></returns>
        /// <remarks>In case when <see cref="value"/> is not <see cref="dtype"/>, <see cref="Convert.ChangeType(object,System.Type)"/> will be called.</remarks>
        public static NDArray Scalar(object value, NPTypeCode typeCode)
        {
            return new NDArray(UnmanagedStorage.Scalar(value, typeCode));
        }
    }
}
