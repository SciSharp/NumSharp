using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp
{
    [SuppressMessage("ReSharper", "StaticMemberInitializerReferesToMemberBelow")]
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
	        %foreach all_dtypes%
                {NPTypeCode.#1, 10000 },
	        %
#else
#endif

        #endregion

        internal static readonly Dictionary<(Type, Type), Type> _typemap_arr_arr;
        internal static readonly Dictionary<(NPTypeCode, NPTypeCode), NPTypeCode> _nptypemap_arr_arr;
        internal static readonly Dictionary<(Type, Type), Type> _typemap_arr_scalar;
        internal static readonly Dictionary<(NPTypeCode, NPTypeCode), NPTypeCode> _nptypemap_arr_scalar;

        [SuppressMessage("ReSharper", "UseObjectOrCollectionInitializer")]
        static np()
        {
            #region arr_arr

            _typemap_arr_arr = new Dictionary<(Type, Type), Type>(180);
            _typemap_arr_arr.Add((np.@bool, np.@bool), np.@bool);
            _typemap_arr_arr.Add((np.@bool, np.uint8), np.uint8);
            _typemap_arr_arr.Add((np.@bool, np.int16), np.int16);
            _typemap_arr_arr.Add((np.@bool, np.uint16), np.uint16);
            _typemap_arr_arr.Add((np.@bool, np.int32), np.int32);
            _typemap_arr_arr.Add((np.@bool, np.uint32), np.uint32);
            _typemap_arr_arr.Add((np.@bool, np.int64), np.int64);
            _typemap_arr_arr.Add((np.@bool, np.uint64), np.uint64);
            _typemap_arr_arr.Add((np.@bool, np.float32), np.float32);
            _typemap_arr_arr.Add((np.@bool, np.float64), np.float64);
            _typemap_arr_arr.Add((np.@bool, np.complex64), np.complex64);
            _typemap_arr_arr.Add((np.@bool, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.@bool, np.@char), np.@char);

            _typemap_arr_arr.Add((np.uint8, np.@bool), np.uint8);
            _typemap_arr_arr.Add((np.uint8, np.uint8), np.uint8);
            _typemap_arr_arr.Add((np.uint8, np.int16), np.int16);
            _typemap_arr_arr.Add((np.uint8, np.uint16), np.uint16);
            _typemap_arr_arr.Add((np.uint8, np.int32), np.int32);
            _typemap_arr_arr.Add((np.uint8, np.uint32), np.uint32);
            _typemap_arr_arr.Add((np.uint8, np.int64), np.int64);
            _typemap_arr_arr.Add((np.uint8, np.uint64), np.uint64);
            _typemap_arr_arr.Add((np.uint8, np.float32), np.float32);
            _typemap_arr_arr.Add((np.uint8, np.float64), np.float64);
            _typemap_arr_arr.Add((np.uint8, np.complex64), np.complex64);
            _typemap_arr_arr.Add((np.uint8, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.uint8, np.@char), np.uint8);

            _typemap_arr_arr.Add((np.@char, np.@char), np.@char);
            _typemap_arr_arr.Add((np.@char, np.@bool), np.@char);
            _typemap_arr_arr.Add((np.@char, np.uint8), np.uint8);
            _typemap_arr_arr.Add((np.@char, np.int16), np.int16);
            _typemap_arr_arr.Add((np.@char, np.uint16), np.uint16);
            _typemap_arr_arr.Add((np.@char, np.int32), np.int32);
            _typemap_arr_arr.Add((np.@char, np.uint32), np.uint32);
            _typemap_arr_arr.Add((np.@char, np.int64), np.int64);
            _typemap_arr_arr.Add((np.@char, np.uint64), np.uint64);
            _typemap_arr_arr.Add((np.@char, np.float32), np.float32);
            _typemap_arr_arr.Add((np.@char, np.float64), np.float64);
            _typemap_arr_arr.Add((np.@char, np.complex64), np.complex64);
            _typemap_arr_arr.Add((np.@char, np.@decimal), np.@decimal);

            _typemap_arr_arr.Add((np.int16, np.@bool), np.int16);
            _typemap_arr_arr.Add((np.int16, np.uint8), np.int16);
            _typemap_arr_arr.Add((np.int16, np.int16), np.int16);
            _typemap_arr_arr.Add((np.int16, np.uint16), np.int32);
            _typemap_arr_arr.Add((np.int16, np.int32), np.int32);
            _typemap_arr_arr.Add((np.int16, np.uint32), np.int64);
            _typemap_arr_arr.Add((np.int16, np.int64), np.int64);
            _typemap_arr_arr.Add((np.int16, np.uint64), np.float64);
            _typemap_arr_arr.Add((np.int16, np.float32), np.float32);
            _typemap_arr_arr.Add((np.int16, np.float64), np.float64);
            _typemap_arr_arr.Add((np.int16, np.complex64), np.complex64);
            _typemap_arr_arr.Add((np.int16, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.int16, np.@char), np.int16);

            _typemap_arr_arr.Add((np.uint16, np.@bool), np.uint16);
            _typemap_arr_arr.Add((np.uint16, np.uint8), np.uint16);
            _typemap_arr_arr.Add((np.uint16, np.int16), np.int32);
            _typemap_arr_arr.Add((np.uint16, np.uint16), np.uint16);
            _typemap_arr_arr.Add((np.uint16, np.int32), np.int32);
            _typemap_arr_arr.Add((np.uint16, np.uint32), np.uint32);
            _typemap_arr_arr.Add((np.uint16, np.int64), np.int64);
            _typemap_arr_arr.Add((np.uint16, np.uint64), np.uint64);
            _typemap_arr_arr.Add((np.uint16, np.float32), np.float32);
            _typemap_arr_arr.Add((np.uint16, np.float64), np.float64);
            _typemap_arr_arr.Add((np.uint16, np.complex64), np.complex64);
            _typemap_arr_arr.Add((np.uint16, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.uint16, np.@char), np.uint16);

            _typemap_arr_arr.Add((np.int32, np.@bool), np.int32);
            _typemap_arr_arr.Add((np.int32, np.uint8), np.int32);
            _typemap_arr_arr.Add((np.int32, np.int16), np.int32);
            _typemap_arr_arr.Add((np.int32, np.uint16), np.int32);
            _typemap_arr_arr.Add((np.int32, np.int32), np.int32);
            _typemap_arr_arr.Add((np.int32, np.uint32), np.int64);
            _typemap_arr_arr.Add((np.int32, np.int64), np.int64);
            _typemap_arr_arr.Add((np.int32, np.uint64), np.float64);
            _typemap_arr_arr.Add((np.int32, np.float32), np.float64);
            _typemap_arr_arr.Add((np.int32, np.float64), np.float64);
            _typemap_arr_arr.Add((np.int32, np.complex64), np.complex128);
            _typemap_arr_arr.Add((np.int32, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.int32, np.@char), np.int32);

            _typemap_arr_arr.Add((np.uint32, np.@bool), np.uint32);
            _typemap_arr_arr.Add((np.uint32, np.uint8), np.uint32);
            _typemap_arr_arr.Add((np.uint32, np.int16), np.int64);
            _typemap_arr_arr.Add((np.uint32, np.uint16), np.uint32);
            _typemap_arr_arr.Add((np.uint32, np.int32), np.int64);
            _typemap_arr_arr.Add((np.uint32, np.uint32), np.uint32);
            _typemap_arr_arr.Add((np.uint32, np.int64), np.int64);
            _typemap_arr_arr.Add((np.uint32, np.uint64), np.uint64);
            _typemap_arr_arr.Add((np.uint32, np.float32), np.float64);
            _typemap_arr_arr.Add((np.uint32, np.float64), np.float64);
            _typemap_arr_arr.Add((np.uint32, np.complex64), np.complex128);
            _typemap_arr_arr.Add((np.uint32, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.uint32, np.@char), np.uint32);

            _typemap_arr_arr.Add((np.int64, np.@bool), np.int64);
            _typemap_arr_arr.Add((np.int64, np.uint8), np.int64);
            _typemap_arr_arr.Add((np.int64, np.int16), np.int64);
            _typemap_arr_arr.Add((np.int64, np.uint16), np.int64);
            _typemap_arr_arr.Add((np.int64, np.int32), np.int64);
            _typemap_arr_arr.Add((np.int64, np.uint32), np.int64);
            _typemap_arr_arr.Add((np.int64, np.int64), np.int64);
            _typemap_arr_arr.Add((np.int64, np.uint64), np.float64);
            _typemap_arr_arr.Add((np.int64, np.float32), np.float64);
            _typemap_arr_arr.Add((np.int64, np.float64), np.float64);
            _typemap_arr_arr.Add((np.int64, np.complex64), np.complex128);
            _typemap_arr_arr.Add((np.int64, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.int64, np.@char), np.int64);

            _typemap_arr_arr.Add((np.uint64, np.@bool), np.uint64);
            _typemap_arr_arr.Add((np.uint64, np.uint8), np.uint64);
            _typemap_arr_arr.Add((np.uint64, np.int16), np.float64);
            _typemap_arr_arr.Add((np.uint64, np.uint16), np.uint64);
            _typemap_arr_arr.Add((np.uint64, np.int32), np.float64);
            _typemap_arr_arr.Add((np.uint64, np.uint32), np.uint64);
            _typemap_arr_arr.Add((np.uint64, np.int64), np.float64);
            _typemap_arr_arr.Add((np.uint64, np.uint64), np.uint64);
            _typemap_arr_arr.Add((np.uint64, np.float32), np.float64);
            _typemap_arr_arr.Add((np.uint64, np.float64), np.float64);
            _typemap_arr_arr.Add((np.uint64, np.complex64), np.complex128);
            _typemap_arr_arr.Add((np.uint64, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.uint64, np.@char), np.uint64);

            _typemap_arr_arr.Add((np.float32, np.@bool), np.float32);
            _typemap_arr_arr.Add((np.float32, np.uint8), np.float32);
            _typemap_arr_arr.Add((np.float32, np.int16), np.float32);
            _typemap_arr_arr.Add((np.float32, np.uint16), np.float32);
            _typemap_arr_arr.Add((np.float32, np.int32), np.float64);
            _typemap_arr_arr.Add((np.float32, np.uint32), np.float64);
            _typemap_arr_arr.Add((np.float32, np.int64), np.float64);
            _typemap_arr_arr.Add((np.float32, np.uint64), np.float64);
            _typemap_arr_arr.Add((np.float32, np.float32), np.float32);
            _typemap_arr_arr.Add((np.float32, np.float64), np.float64);
            _typemap_arr_arr.Add((np.float32, np.complex64), np.complex64);
            _typemap_arr_arr.Add((np.float32, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.float32, np.@char), np.float32);

            _typemap_arr_arr.Add((np.float64, np.@bool), np.float64);
            _typemap_arr_arr.Add((np.float64, np.uint8), np.float64);
            _typemap_arr_arr.Add((np.float64, np.int16), np.float64);
            _typemap_arr_arr.Add((np.float64, np.uint16), np.float64);
            _typemap_arr_arr.Add((np.float64, np.int32), np.float64);
            _typemap_arr_arr.Add((np.float64, np.uint32), np.float64);
            _typemap_arr_arr.Add((np.float64, np.int64), np.float64);
            _typemap_arr_arr.Add((np.float64, np.uint64), np.float64);
            _typemap_arr_arr.Add((np.float64, np.float32), np.float64);
            _typemap_arr_arr.Add((np.float64, np.float64), np.float64);
            _typemap_arr_arr.Add((np.float64, np.complex64), np.complex128);
            _typemap_arr_arr.Add((np.float64, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.float64, np.@char), np.float64);

            _typemap_arr_arr.Add((np.complex64, np.@bool), np.complex64);
            _typemap_arr_arr.Add((np.complex64, np.uint8), np.complex64);
            _typemap_arr_arr.Add((np.complex64, np.int16), np.complex64);
            _typemap_arr_arr.Add((np.complex64, np.uint16), np.complex64);
            _typemap_arr_arr.Add((np.complex64, np.int32), np.complex128);
            _typemap_arr_arr.Add((np.complex64, np.uint32), np.complex128);
            _typemap_arr_arr.Add((np.complex64, np.int64), np.complex128);
            _typemap_arr_arr.Add((np.complex64, np.uint64), np.complex128);
            _typemap_arr_arr.Add((np.complex64, np.float32), np.complex64);
            _typemap_arr_arr.Add((np.complex64, np.float64), np.complex128);
            _typemap_arr_arr.Add((np.complex64, np.complex64), np.complex64);
            _typemap_arr_arr.Add((np.complex64, np.@decimal), np.complex64);
            _typemap_arr_arr.Add((np.complex64, np.@char), np.complex64);

            _typemap_arr_arr.Add((np.@decimal, np.@bool), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.uint8), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.int16), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.uint16), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.int32), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.uint32), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.int64), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.uint64), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.float32), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.float64), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.complex64), np.complex128);
            _typemap_arr_arr.Add((np.@decimal, np.@decimal), np.@decimal);
            _typemap_arr_arr.Add((np.@decimal, np.@char), np.@decimal);

            _nptypemap_arr_arr = new Dictionary<(NPTypeCode, NPTypeCode), NPTypeCode>(_typemap_arr_arr.Count);
            foreach (var tc in _typemap_arr_arr) _nptypemap_arr_arr[(tc.Key.Item1.GetTypeCode(), tc.Key.Item2.GetTypeCode())] = tc.Value.GetTypeCode();

            #endregion

            #region arr_scalar

            _typemap_arr_scalar = new Dictionary<(Type, Type), Type>();
            _typemap_arr_scalar.Add((np.@bool, np.@bool), np.@bool);
            _typemap_arr_scalar.Add((np.@bool, np.uint8), np.uint8);
            _typemap_arr_scalar.Add((np.@bool, np.int16), np.int16);
            _typemap_arr_scalar.Add((np.@bool, np.uint16), np.uint16);
            _typemap_arr_scalar.Add((np.@bool, np.int32), np.int32);
            _typemap_arr_scalar.Add((np.@bool, np.uint32), np.uint32);
            _typemap_arr_scalar.Add((np.@bool, np.int64), np.int64);
            _typemap_arr_scalar.Add((np.@bool, np.uint64), np.uint64);
            _typemap_arr_scalar.Add((np.@bool, np.float32), np.float32);
            _typemap_arr_scalar.Add((np.@bool, np.float64), np.float64);
            _typemap_arr_scalar.Add((np.@bool, np.complex64), np.complex64);

            _typemap_arr_scalar.Add((np.uint8, np.@bool), np.uint8);
            _typemap_arr_scalar.Add((np.uint8, np.uint8), np.uint8);
            _typemap_arr_scalar.Add((np.uint8, np.@char), np.uint8);
            _typemap_arr_scalar.Add((np.uint8, np.int16), np.int16);
            _typemap_arr_scalar.Add((np.uint8, np.uint16), np.uint8);
            _typemap_arr_scalar.Add((np.uint8, np.int32), np.int32);
            _typemap_arr_scalar.Add((np.uint8, np.uint32), np.uint8);
            _typemap_arr_scalar.Add((np.uint8, np.int64), np.int64);
            _typemap_arr_scalar.Add((np.uint8, np.uint64), np.uint8);
            _typemap_arr_scalar.Add((np.uint8, np.float32), np.float32);
            _typemap_arr_scalar.Add((np.uint8, np.float64), np.float64);
            _typemap_arr_scalar.Add((np.uint8, np.complex64), np.complex64);

            _typemap_arr_scalar.Add((np.@char, np.@char), np.@char);
            _typemap_arr_scalar.Add((np.@char, np.@bool), np.@char);
            _typemap_arr_scalar.Add((np.@char, np.uint8), np.@char);
            _typemap_arr_scalar.Add((np.@char, np.int16), np.int16);
            _typemap_arr_scalar.Add((np.@char, np.uint16), np.uint16);
            _typemap_arr_scalar.Add((np.@char, np.int32), np.int32);
            _typemap_arr_scalar.Add((np.@char, np.uint32), np.uint32);
            _typemap_arr_scalar.Add((np.@char, np.int64), np.int64);
            _typemap_arr_scalar.Add((np.@char, np.uint64), np.uint64);
            _typemap_arr_scalar.Add((np.@char, np.float32), np.float32);
            _typemap_arr_scalar.Add((np.@char, np.float64), np.float64);
            _typemap_arr_scalar.Add((np.@char, np.complex64), np.complex64);

            _typemap_arr_scalar.Add((np.int16, np.@bool), np.int16);
            _typemap_arr_scalar.Add((np.int16, np.uint8), np.int16);
            _typemap_arr_scalar.Add((np.int16, np.@char), np.int16);
            _typemap_arr_scalar.Add((np.int16, np.int16), np.int16);
            _typemap_arr_scalar.Add((np.int16, np.uint16), np.int16);
            _typemap_arr_scalar.Add((np.int16, np.int32), np.int16);
            _typemap_arr_scalar.Add((np.int16, np.uint32), np.int16);
            _typemap_arr_scalar.Add((np.int16, np.int64), np.int16);
            _typemap_arr_scalar.Add((np.int16, np.uint64), np.int16);
            _typemap_arr_scalar.Add((np.int16, np.float32), np.float32);
            _typemap_arr_scalar.Add((np.int16, np.float64), np.float64);
            _typemap_arr_scalar.Add((np.int16, np.complex64), np.complex64);

            _typemap_arr_scalar.Add((np.uint16, np.@bool), np.uint16);
            _typemap_arr_scalar.Add((np.uint16, np.uint8), np.uint16);
            _typemap_arr_scalar.Add((np.uint16, np.@char), np.uint16);
            _typemap_arr_scalar.Add((np.uint16, np.int16), np.int32);
            _typemap_arr_scalar.Add((np.uint16, np.uint16), np.uint16);
            _typemap_arr_scalar.Add((np.uint16, np.int32), np.int32);
            _typemap_arr_scalar.Add((np.uint16, np.uint32), np.uint16);
            _typemap_arr_scalar.Add((np.uint16, np.int64), np.int64);
            _typemap_arr_scalar.Add((np.uint16, np.uint64), np.uint16);
            _typemap_arr_scalar.Add((np.uint16, np.float32), np.float32);
            _typemap_arr_scalar.Add((np.uint16, np.float64), np.float64);
            _typemap_arr_scalar.Add((np.uint16, np.complex64), np.complex64);

            _typemap_arr_scalar.Add((np.int32, np.@bool), np.int32);
            _typemap_arr_scalar.Add((np.int32, np.uint8), np.int32);
            _typemap_arr_scalar.Add((np.int32, np.@char), np.int32);
            _typemap_arr_scalar.Add((np.int32, np.int16), np.int32);
            _typemap_arr_scalar.Add((np.int32, np.uint16), np.int32);
            _typemap_arr_scalar.Add((np.int32, np.int32), np.int32);
            _typemap_arr_scalar.Add((np.int32, np.uint32), np.int32);
            _typemap_arr_scalar.Add((np.int32, np.int64), np.int32);
            _typemap_arr_scalar.Add((np.int32, np.uint64), np.int32);
            _typemap_arr_scalar.Add((np.int32, np.float32), np.float64);
            _typemap_arr_scalar.Add((np.int32, np.float64), np.float64);
            _typemap_arr_scalar.Add((np.int32, np.complex64), np.complex128);

            _typemap_arr_scalar.Add((np.uint32, np.@bool), np.uint32);
            _typemap_arr_scalar.Add((np.uint32, np.uint8), np.uint32);
            _typemap_arr_scalar.Add((np.uint32, np.@char), np.uint32);
            _typemap_arr_scalar.Add((np.uint32, np.int16), np.int64);
            _typemap_arr_scalar.Add((np.uint32, np.uint16), np.uint32);
            _typemap_arr_scalar.Add((np.uint32, np.int32), np.int64);
            _typemap_arr_scalar.Add((np.uint32, np.uint32), np.uint32);
            _typemap_arr_scalar.Add((np.uint32, np.int64), np.int64);
            _typemap_arr_scalar.Add((np.uint32, np.uint64), np.uint32);
            _typemap_arr_scalar.Add((np.uint32, np.float32), np.float64);
            _typemap_arr_scalar.Add((np.uint32, np.float64), np.float64);
            _typemap_arr_scalar.Add((np.uint32, np.complex64), np.complex128);

            _typemap_arr_scalar.Add((np.int64, np.@bool), np.int64);
            _typemap_arr_scalar.Add((np.int64, np.uint8), np.int64);
            _typemap_arr_scalar.Add((np.int64, np.@char), np.int64);
            _typemap_arr_scalar.Add((np.int64, np.int16), np.int64);
            _typemap_arr_scalar.Add((np.int64, np.uint16), np.int64);
            _typemap_arr_scalar.Add((np.int64, np.int32), np.int64);
            _typemap_arr_scalar.Add((np.int64, np.uint32), np.int64);
            _typemap_arr_scalar.Add((np.int64, np.int64), np.int64);
            _typemap_arr_scalar.Add((np.int64, np.uint64), np.int64);
            _typemap_arr_scalar.Add((np.int64, np.float32), np.float64);
            _typemap_arr_scalar.Add((np.int64, np.float64), np.float64);
            _typemap_arr_scalar.Add((np.int64, np.complex64), np.complex128);

            _typemap_arr_scalar.Add((np.uint64, np.@bool), np.uint64);
            _typemap_arr_scalar.Add((np.uint64, np.uint8), np.uint64);
            _typemap_arr_scalar.Add((np.uint64, np.@char), np.uint64);
            _typemap_arr_scalar.Add((np.uint64, np.int16), np.float64);
            _typemap_arr_scalar.Add((np.uint64, np.uint16), np.uint64);
            _typemap_arr_scalar.Add((np.uint64, np.int32), np.float64);
            _typemap_arr_scalar.Add((np.uint64, np.uint32), np.uint64);
            _typemap_arr_scalar.Add((np.uint64, np.int64), np.float64);
            _typemap_arr_scalar.Add((np.uint64, np.uint64), np.uint64);
            _typemap_arr_scalar.Add((np.uint64, np.float32), np.float64);
            _typemap_arr_scalar.Add((np.uint64, np.float64), np.float64);
            _typemap_arr_scalar.Add((np.uint64, np.complex64), np.complex128);

            _typemap_arr_scalar.Add((np.float32, np.@bool), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.uint8), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.@char), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.int16), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.uint16), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.int32), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.uint32), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.int64), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.uint64), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.float32), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.float64), np.float32);
            _typemap_arr_scalar.Add((np.float32, np.complex64), np.complex64);

            _typemap_arr_scalar.Add((np.float64, np.@bool), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.uint8), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.@char), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.int16), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.uint16), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.int32), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.uint32), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.int64), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.uint64), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.float32), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.float64), np.float64);
            _typemap_arr_scalar.Add((np.float64, np.complex64), np.complex128);

            _typemap_arr_scalar.Add((np.complex64, np.@bool), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.uint8), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.@char), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.int16), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.uint16), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.int32), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.uint32), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.int64), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.uint64), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.float32), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.float64), np.complex64);
            _typemap_arr_scalar.Add((np.complex64, np.complex64), np.complex64);

            _typemap_arr_scalar.Add((np.@decimal, np.@bool), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.uint8), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.@char), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.int16), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.uint16), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.int32), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.uint32), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.int64), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.uint64), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.float32), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.float64), np.@decimal);
            _typemap_arr_scalar.Add((np.@decimal, np.complex64), np.complex128);
            _typemap_arr_scalar.Add((np.@decimal, np.@decimal), np.@decimal);
            _typemap_arr_scalar.Add((np.@bool, np.@decimal), np.@bool);
            _typemap_arr_scalar.Add((np.uint8, np.@decimal), np.uint8);
            _typemap_arr_scalar.Add((np.@char, np.@decimal), np.@char);
            _typemap_arr_scalar.Add((np.int16, np.@decimal), np.int16);
            _typemap_arr_scalar.Add((np.uint16, np.@decimal), np.uint16);
            _typemap_arr_scalar.Add((np.int32, np.@decimal), np.int32);
            _typemap_arr_scalar.Add((np.uint32, np.@decimal), np.uint32);
            _typemap_arr_scalar.Add((np.int64, np.@decimal), np.int64);
            _typemap_arr_scalar.Add((np.uint64, np.@decimal), np.uint64);
            _typemap_arr_scalar.Add((np.float32, np.@decimal), np.float32);
            _typemap_arr_scalar.Add((np.float64, np.@decimal), np.float64);
            _typemap_arr_scalar.Add((np.complex64, np.@decimal), np.complex128);

            _nptypemap_arr_scalar = new Dictionary<(NPTypeCode, NPTypeCode), NPTypeCode>(_typemap_arr_scalar.Count);
            foreach (var tc in _typemap_arr_scalar) _nptypemap_arr_scalar[(tc.Key.Item1.GetTypeCode(), tc.Key.Item2.GetTypeCode())] = tc.Value.GetTypeCode();

            #endregion
        }

        // @formatter:off — disable formatter after this line
        internal static NPTypeCode[] powerOrder =
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
            NPTypeCode.Double,
            NPTypeCode.Decimal,
            NPTypeCode.Decimal,
            NPTypeCode.Complex, //Complex64
            NPTypeCode.Complex, //Complex128
            //NPTypeCode.Complex, //Complex128
            NPTypeCode.Single,
        };
        // @formatter:off — disable formatter after this line

        private static readonly (NPTypeCode Type, int Priority)[] powerPriorities =
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
            (NPTypeCode.Double, NPTypeCode.Double.GetPriority() + 1),
            (NPTypeCode.Decimal, NPTypeCode.Decimal.GetPriority()),
            (NPTypeCode.Decimal, NPTypeCode.Decimal.GetPriority()),
            (NPTypeCode.Complex, NPTypeCode.Complex.GetPriority()), //Complex64
            (NPTypeCode.Complex, NPTypeCode.Complex.GetPriority()), //Complex128
            //NPTypeCode.Complex, //Complex128
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

        internal static NPTypeCode _FindCommonArrayType(Type dtype_left, Type dtype_right)
        {
            return _nptypemap_arr_arr[(dtype_left.GetTypeCode(), dtype_right.GetTypeCode())];
        }

        internal static NPTypeCode _FindCommonArrayType(NPTypeCode dtype_left, NPTypeCode dtype_right)
        {
            return _nptypemap_arr_arr[(dtype_left, dtype_right)];
        }

        internal static NPTypeCode _FindCommonScalarType(Type dtype_left, Type dtype_right)
        {
            return _FindCommonType(Array.Empty<NPTypeCode>(), new NPTypeCode[] {dtype_left.GetTypeCode(), dtype_right.GetTypeCode()});
        }

        internal static NPTypeCode _FindCommonScalarType(NPTypeCode dtype_left, NPTypeCode dtype_right)
        {
            return _FindCommonType(Array.Empty<NPTypeCode>(), new NPTypeCode[] {dtype_left, dtype_right});
        }

        internal static NPTypeCode _FindCommonArrayScalarType(NPTypeCode dtypeArray, NPTypeCode dtypeScalar)
        {
            return _nptypemap_arr_scalar[(dtypeArray, dtypeScalar)];
        }

        internal static NPTypeCode _FindCommonArrayScalarType(Type dtypeArray, Type dtypeScalar)
        {
            return _nptypemap_arr_scalar[(dtypeArray.GetTypeCode(), dtypeScalar.GetTypeCode())];
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
                return _nptypemap_arr_scalar[(maxa, maxsc)];
            else
                return maxa;
        }

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl((MethodImplOptions)512)]
        internal static NPTypeCode _FindCommonType(List<NPTypeCode> array_types, List<NPTypeCode> scalar_types)
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

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl((MethodImplOptions)512)]
        internal static NPTypeCode _FindCommonType(params NDArray[] involvedArrays)
        {
            List<NPTypeCode> scalar = new List<NPTypeCode>(involvedArrays.Length);
            List<NPTypeCode> list = new List<NPTypeCode>(involvedArrays.Length);
            foreach (var nd in involvedArrays)
            {
                if (nd.Shape.IsScalar)
                    scalar.Add(nd.GetTypeCode);
                else
                    list.Add(nd.GetTypeCode);
            }

            return _FindCommonType(list, scalar);
        }

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        [MethodImpl((MethodImplOptions)512)]
        public static NPTypeCode find_common_type(params string[] involvedTypes)
        {
            return _can_coerce_all(involvedTypes.Select(s => dtype(s).typecode).ToArray());
        }

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl((MethodImplOptions)512)]
        internal static NPTypeCode _FindCommonType(NDArray firstNDArray, NDArray secondNDArray)
        {
            var lscalar = firstNDArray.Shape.IsScalar;
            var rscalar = secondNDArray.Shape.IsScalar;
            if (!lscalar && !rscalar)
                return _FindCommonArrayType(firstNDArray.GetTypeCode, secondNDArray.GetTypeCode);

            if (lscalar && rscalar)
                return _FindCommonScalarType(firstNDArray.GetTypeCode, secondNDArray.GetTypeCode);

            if (lscalar)
                return _FindCommonArrayScalarType(secondNDArray.GetTypeCode, firstNDArray.GetTypeCode);

            //rscalar is true
            return _FindCommonArrayScalarType(firstNDArray.GetTypeCode, secondNDArray.GetTypeCode);
        }

        #region Private of find_common_type

        [MethodImpl((MethodImplOptions)512)]
        private static NPTypeCode _can_coerce_all(NPTypeCode[] dtypelist)
        {
            int N = dtypelist.Length;

            if (N == 0)
                return NPTypeCode.Empty;
            if (N == 1)
                return dtypelist[0];
            NPTypeCode ret = default;
            while (N >= 2)
            {
                ret = _nptypemap_arr_arr[(dtypelist[N - 1], dtypelist[N - 2])];
                N -= 2;
            }

            if (N == 1)
            {
                ret = _nptypemap_arr_arr[(ret, dtypelist[0])];
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)512)]
        private static NPTypeCode _can_coerce_all(List<NPTypeCode> dtypelist)
        {
            int N = dtypelist.Count;

            if (N == 0)
                return NPTypeCode.Empty;
            if (N == 1)
                return dtypelist[0];
            NPTypeCode ret = default;

            while (N >= 2)
            {
                ret = _nptypemap_arr_arr[(dtypelist[N - 1], dtypelist[N - 2])];
                N -= 2;
            }

            if (N == 1)
            {
                ret = _nptypemap_arr_arr[(ret, dtypelist[0])];
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)512)]
        private static NPTypeCode _can_coerce_all(NPTypeCode[] dtypelist, int start)
        {
            int N = dtypelist.Length;
            if (start > 0)
            {
                var len = N - start;
                var sub = new NPTypeCode[len];
                Array.Copy(dtypelist, start, sub, len, len);
                dtypelist = sub;
                N = sub.Length;
            }

            if (N == 0)
                return NPTypeCode.Empty;
            if (N == 1)
                return dtypelist[0];
            NPTypeCode ret = default;
            while (N >= 2)
            {
                ret = _nptypemap_arr_arr[(dtypelist[N - 1], dtypelist[N - 2])];
                N -= 2;
            }

            if (N == 1)
            {
                ret = _nptypemap_arr_arr[(ret, dtypelist[0])];
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)512)]
        private static NPTypeCode _can_coerce_all(List<NPTypeCode> dtypelist, int start)
        {
            int N = dtypelist.Count;
            if (start > 0)
            {
                var len = N - start;
                var sub = new List<NPTypeCode>(len);
                for (int i = start; i < N; i++)
                    sub[i - start] = dtypelist[i];
                dtypelist = sub;
                N = sub.Count;
            }

            if (N == 0)
                return NPTypeCode.Empty;
            if (N == 1)
                return dtypelist[0];
            NPTypeCode ret = default;

            while (N >= 2)
            {
                ret = _nptypemap_arr_arr[(dtypelist[N - 1], dtypelist[N - 2])];
                N -= 2;
            }

            if (N == 1)
            {
                ret = _nptypemap_arr_arr[(ret, dtypelist[0])];
            }

            return ret;
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
