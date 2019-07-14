using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Extensions;
using NumSharp.Utilities.Linq;

namespace NumSharp
{
    public static partial class np
    {
        #region Privates

        const int __len_test_types = 19;
        const string __test_types = "?bBhHiIlLqQefdgFDGO";

        /// <summary>
        ///  b -> boolean<br></br>
        ///  u -> unsigned integer<br></br>
        ///  i -> signed integer<br></br>
        ///  f -> floating point<br></br>
        ///  c -> complex<br></br>
        ///  M -> datetime<br></br>
        ///  m -> timedelta<br></br>
        ///  S -> string<br></br>
        ///  U -> Unicode string<br></br>
        ///  V -> record<br></br>
        ///  O -> Python object
        /// </summary>
        private static readonly char[] _kind_list = {'b', 'u', 'i', 'f', 'c', 'S', 'U', 'V', 'O', 'M', 'm'};

#if __REGEN
	        %foreach supported_dtypes%
                {NPTypeCode.#1, 10000 },
	        %
#else
#endif

        #endregion

        // @formatter:off — disable formatter after this line
        private static NPTypeCode[] powerOrder =
        {
            NPTypeCode.Boolean, 
            NPTypeCode.Byte, //Int8
            NPTypeCode.Byte, //unit8
            NPTypeCode.Int16, 
            NPTypeCode.UInt16, 
            NPTypeCode.Int32, 
            NPTypeCode.UInt32, 
            NPTypeCode.Int32, 
            NPTypeCode.UInt32, 
            NPTypeCode.Int64, 
            NPTypeCode.UInt64, 
            //NPTypeCode.Single, //Float16
            NPTypeCode.Single, //Float32
            NPTypeCode.Double, //Float64
            NPTypeCode.Double,
            NPTypeCode.Decimal,
            NPTypeCode.Decimal,
            NPTypeCode.Complex, //Complex64
            NPTypeCode.Complex, //Complex128
            //NPTypeCode.Complex, //Complex128
            NPTypeCode.NDArray, 
            NPTypeCode.Single,
        };
        // @formatter:off — disable formatter after this line
        private static (NPTypeCode Type, int Priority)[] powerPriorities =
        {
            (NPTypeCode.Boolean, NPTypeCode.Boolean.GetPriority()),
            (NPTypeCode.Byte, NPTypeCode.Byte.GetPriority()), //Int8
            (NPTypeCode.Byte, NPTypeCode.Byte.GetPriority()), //unit8
            (NPTypeCode.Int16, NPTypeCode.Int16.GetPriority()),
            (NPTypeCode.UInt16, NPTypeCode.UInt16.GetPriority()),
            (NPTypeCode.Int32, NPTypeCode.Int32.GetPriority()),
            (NPTypeCode.UInt32, NPTypeCode.UInt32.GetPriority()),
            (NPTypeCode.Int32, NPTypeCode.Int32.GetPriority()),
            (NPTypeCode.UInt32, NPTypeCode.UInt32.GetPriority()),
            (NPTypeCode.Int64, NPTypeCode.Int64.GetPriority()),
            (NPTypeCode.UInt64, NPTypeCode.UInt64.GetPriority()),
            //NPTypeCode.Single, NPTypeCode.Single.GetPriority()), //Float16
            (NPTypeCode.Single, NPTypeCode.Single.GetPriority()), //Float32
            (NPTypeCode.Double, NPTypeCode.Double.GetPriority()), //Float64
            (NPTypeCode.Double, NPTypeCode.Double.GetPriority()),
            (NPTypeCode.Decimal, NPTypeCode.Decimal.GetPriority()),
            (NPTypeCode.Decimal, NPTypeCode.Decimal.GetPriority()),
            (NPTypeCode.Complex, NPTypeCode.Complex.GetPriority()), //Complex64
            (NPTypeCode.Complex, NPTypeCode.Complex.GetPriority()), //Complex128
            //NPTypeCode.Complex, //Complex128
            (NPTypeCode.NDArray, NPTypeCode.NDArray.GetPriority()),
            (NPTypeCode.Single, NPTypeCode.Single.GetPriority()),
        };
        // @formatter:on — enable formatter after this line

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(NPTypeCode[] array_types, NPTypeCode[] scalar_types)
        {
            return _FindCommonType(array_types ?? Array.Empty<NPTypeCode>(), scalar_types ?? Array.Empty<NPTypeCode>());
        }

        #region Overloads

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(Type[] array_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), Array.Empty<NPTypeCode>());
        }

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(Type[] array_types, Type[] scalar_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), scalar_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>());
        }

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(string[] array_types, string[] scalar_types)
        {
            return _FindCommonType(array_types?.Select(v => np.dtype(v).typecode).ToArray() ?? Array.Empty<NPTypeCode>(), scalar_types?.Select(v => np.dtype(v).typecode).ToArray() ?? Array.Empty<NPTypeCode>());
        }

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(Type[] array_types, NPTypeCode[] scalar_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), scalar_types ?? Array.Empty<NPTypeCode>());
        }

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(NPTypeCode[] array_types, Type[] scalar_types)
        {
            return _FindCommonType(array_types ?? Array.Empty<NPTypeCode>(), scalar_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>());
        }

        #endregion

        internal static NPTypeCode ResolveCommonArrayType(Type dtype_left, Type dtype_right)
        {
            return _FindCommonType(new NPTypeCode[] {dtype_left.GetTypeCode(), dtype_right.GetTypeCode()}, Array.Empty<NPTypeCode>());
        }

        internal static NPTypeCode ResolveCommonScalarType(Type dtype_left, Type dtype_right)
        {
            return _FindCommonType(Array.Empty<NPTypeCode>(), new NPTypeCode[] {dtype_left.GetTypeCode(), dtype_right.GetTypeCode()});
        }

        internal static NPTypeCode ResolveCommonArrayType(NPTypeCode dtype_left, NPTypeCode dtype_right)
        {
            return _FindCommonType(new NPTypeCode[] {dtype_left, dtype_right}, Array.Empty<NPTypeCode>());
        }

        internal static NPTypeCode ResolveCommonScalarType(NPTypeCode dtype_left, NPTypeCode dtype_right)
        {
            return _FindCommonType(Array.Empty<NPTypeCode>(), new NPTypeCode[] {dtype_left, dtype_right});
        }

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl((MethodImplOptions)512)]
        internal static NPTypeCode _FindCommonType(NPTypeCode[] array_types, NPTypeCode[] scalar_types)
        {
            NPTypeCode maxa = _can_coerce_all(array_types);
            NPTypeCode maxsc = _can_coerce_all(scalar_types);

            if (maxa == NPTypeCode.Empty)
                return maxsc;

            if (maxsc == NPTypeCode.Empty)
                return maxa;

            int index_a;
            int index_sc;
            try
            {
                index_a = maxa != NPTypeCode.Empty ? Array.IndexOf(_kind_list, DType._kind_list_map[maxa]) : -1;
                index_sc = maxsc != NPTypeCode.Empty ? Array.IndexOf(_kind_list, DType._kind_list_map[maxsc]) : -1;
            }
            catch (Exception)
            {
                return NPTypeCode.Empty;
            }

            if (index_sc > index_a)
                return _find_common_coerce(maxsc, maxa);
            else
                return maxa;
        }

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl((MethodImplOptions)512)]
        internal static NPTypeCode _FindCommonType_Scalar(params NPTypeCode[] scalar_types)
        {
            return _can_coerce_all(scalar_types);
        }


        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl((MethodImplOptions)512)]
        internal static NPTypeCode _FindCommonType_Array(params NPTypeCode[] array_types)
        {
            return _can_coerce_all(array_types);
        }

        #region Private of find_common_type

        private static NPTypeCode _can_coerce_all(NPTypeCode[] dtypelist, int start = 0)
        {
            int N = dtypelist.Length;
            if (N == 0)
                return NPTypeCode.Empty;
            if (N == 1)
                return dtypelist[0];

            //incase they are all equal
            for (int i = 0; i < N; i++)
            {
                for (int k = 0; k < N; k++)
                {
                    if (i != k && dtypelist[i] != dtypelist[k])
                        goto _false;
                }
            }

            return dtypelist[0];

            _false:
            int[] sizes = dtypelist.Select(d => d.SizeOf()).ToArray();
            int first_size = sizes[0];

            //incase we have same sizes, find out by priority
            if (sizes.All(s => s == first_size))
            {
                //find get largest priority out of dtypelist
                NPTypeCode maxType = NPTypeCode.Empty;
                int maxpriority = 0;
                for (int i = 0; i < N; i++)
                {
                    int priority = dtypelist[i].GetPriority();
                    if (priority > maxpriority)
                    {
                        maxpriority = priority;
                        maxType = dtypelist[i];
                    }
                }

                //then we gotta get the next size of that group.
                for (int i = 0; i < powerPriorities.Length; i++)
                {
                    var curr = powerPriorities[i];
                    if (curr.Priority > maxpriority)
                    {
                        return curr.Type;
                    }
                }

                //there aren't bigger, return largest
                return maxType;
            }

            //incase any of dtypelist is the largest group type
            foreach (NPTypeCode curr in dtypelist)
            {
                int currgrp = curr.GetGroup();
                if (dtypelist.Except(curr.Yield()).All(c => currgrp > c.GetGroup()))
                    return curr;
            }

            //go vanilla (taken from numpy's source)
            int n = start;
            while (n < powerOrder.Length)
            {
                NPTypeCode newdtype = powerOrder[n];
                int numcoerce = dtypelist.Count(x => NPTypeCodeExtensions.CompareTo(newdtype, x) > 0);
                if (numcoerce == N)
                    return newdtype;
                n += 1;
            }

            return NPTypeCode.Empty;
        }

        // Keep incrementing until a common type both can be coerced to
        //  is found.  Otherwise, return None
        private static NPTypeCode _find_common_coerce(NPTypeCode a, NPTypeCode b)
        {
            if (a > b)
                return a;
            if (a == NPTypeCode.Empty)
                return b;
            if (b == NPTypeCode.Empty)
                return a;

            int thisind;
            try
            {
                thisind = __test_types.IndexOf((char)a.ToTYPECHAR());
            }
            catch
            {
                return NPTypeCode.Empty;
            }

            return _can_coerce_all(new NPTypeCode[] {a, b}, thisind);
        }

        #endregion
    }
}
