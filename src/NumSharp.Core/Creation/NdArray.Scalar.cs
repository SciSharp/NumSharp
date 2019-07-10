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
            var ndArray = new NDArray(dtype, new int[0]);
            ndArray.Storage.ReplaceData(Arrays.Wrap(dtype.GetTypeCode(), Convert.ChangeType(value, dtype)));

            return ndArray;
        }

        /// <summary>
        ///     Creates a scalar <see cref="NDArray"/> of <see cref="value"/> and <see cref="dtype"/>.
        /// </summary>
        /// <param name="value">The value of the scalar</param>
        /// <returns></returns>
        /// <remarks>In case when <see cref="value"/> is not <see cref="dtype"/>, <see cref="Convert.ChangeType(object,System.Type)"/> will be called.</remarks>
        public static NDArray Scalar(object value)
        {
            var dtype = value.GetType();
            var ndArray = new NDArray(dtype, new int[0]);
            ndArray.Storage.ReplaceData(Arrays.Wrap(dtype.GetTypeCode(), value));

            return ndArray;
        }

        /// <summary>
        ///     Creates a scalar <see cref="NDArray"/> of <see cref="value"/> and <see cref="dtype"/>.
        /// </summary>
        /// <param name="value">The value of the scalar</param>
        /// <returns></returns>
        /// <remarks>In case when <see cref="value"/> is not <see cref="dtype"/>, <see cref="Convert.ChangeType(object,System.Type)"/> will be called.</remarks>
        public static NDArray Scalar<T>(T value) where T : unmanaged
        {
            var nd = new NDArray(typeof(T));
            nd.Storage = new UnmanagedStorage(value);
            var dtype = value.GetType();
            var ndArray = new NDArray(dtype, new int[0]);
            ndArray.Storage.ReplaceData(Arrays.Wrap(dtype.GetTypeCode(), value));

            return ndArray;
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
            var type = typeCode.AsType();
            var ndArray = new NDArray(type, new int[0]);
            ndArray.Storage.ReplaceData(Arrays.Wrap(typeCode, Convert.ChangeType(value, type))); //todo! create a NPConvert to support NPTypeCode

            return ndArray;
        }
    }
}
