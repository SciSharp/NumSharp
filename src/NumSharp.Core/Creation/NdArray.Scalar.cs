using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

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
        /// <exception cref="ArgumentNullException">when dtype is null</exception>
        /// <exception cref="InvalidCastException">This conversion is not supported. -or- value is null and conversionType is a value type. -or- value does not implement the IConvertible interface. </exception>
        public static NDArray Scalar(object value, Type dtype)
        {
            if (dtype == null)
            {
                throw new ArgumentNullException(nameof(dtype));
            }

            var ndArray = new NDArray(dtype, new int[0]);
            var arr = Array.CreateInstance(dtype, 1);
            if (dtype != value.GetType())
                value = Convert.ChangeType(value, dtype);

            arr.SetValue(value, 0);
            ndArray.Storage.SetData(arr);

            return ndArray;
        }

        /// <summary>
        ///     Creates a scalar <see cref="NDArray"/> of <see cref="value"/> from a known type <typeparamref name="T"/>.
        /// </summary>
        /// <param name="value">The value of the scalar</param>
        /// <returns></returns>
        /// <remarks>In case when <see cref="value"/> is not <see cref="dtype"/>, <see cref="Convert.ChangeType(object,System.Type)"/> will be called.</remarks>
        public static NDArray Scalar<T>(T value)
        {
            var ndArray = new NDArray(typeof(T), new int[0]);
            ndArray.Storage.SetData(new T[] {value});

            return ndArray;
        }
    }
}
