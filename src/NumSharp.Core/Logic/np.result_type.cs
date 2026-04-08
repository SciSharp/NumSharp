using System;
using System.Linq;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the arguments.
        /// </summary>
        /// <param name="arrays_and_dtypes">
        /// Arrays and/or dtype arguments. Can be any mix of NDArray, NPTypeCode, or Type.
        /// </param>
        /// <returns>The result type from combining the inputs.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.result_type.html
        ///
        /// Type promotion is the operation of determining the result type of an operation
        /// involving operands of different types.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.result_type(NPTypeCode.Int32, NPTypeCode.Int64)    // Int64
        /// np.result_type(NPTypeCode.Int32, NPTypeCode.Single)   // Double
        /// np.result_type(a, b)  // where a, b are NDArrays
        /// </code>
        /// </example>
        public static NPTypeCode result_type(params object[] arrays_and_dtypes)
        {
            if (arrays_and_dtypes == null || arrays_and_dtypes.Length == 0)
                throw new ArgumentException("At least one array or dtype must be provided", nameof(arrays_and_dtypes));

            var types = arrays_and_dtypes.Select(GetTypeCode).Where(t => t != NPTypeCode.Empty).ToArray();

            if (types.Length == 0)
                throw new ArgumentException("No valid types found in arguments", nameof(arrays_and_dtypes));

            if (types.Length == 1)
                return types[0];

            // Use existing find_common_type infrastructure
            return _FindCommonType_Array(types);
        }

        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the arguments.
        /// </summary>
        /// <param name="types">One or more NPTypeCode values.</param>
        /// <returns>The result type from combining the inputs.</returns>
        public static NPTypeCode result_type(params NPTypeCode[] types)
        {
            if (types == null || types.Length == 0)
                throw new ArgumentException("At least one type must be provided", nameof(types));

            if (types.Length == 1)
                return types[0];

            return _FindCommonType_Array(types);
        }

        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the arguments.
        /// </summary>
        /// <param name="arrays">One or more NDArray objects.</param>
        /// <returns>The result type from combining the array dtypes.</returns>
        public static NPTypeCode result_type(params NDArray[] arrays)
        {
            if (arrays == null || arrays.Length == 0)
                throw new ArgumentException("At least one array must be provided", nameof(arrays));

            if (arrays.Length == 1)
                return arrays[0].GetTypeCode;

            return _FindCommonType(arrays);
        }

        /// <summary>
        /// Helper to extract NPTypeCode from various input types.
        /// </summary>
        private static NPTypeCode GetTypeCode(object input)
        {
            return input switch
            {
                NPTypeCode tc => tc,
                Type t => t.GetTypeCode(),
                NDArray arr => arr.GetTypeCode,
                string s => dtype(s).typecode,
                _ => NPTypeCode.Empty
            };
        }
    }
}
