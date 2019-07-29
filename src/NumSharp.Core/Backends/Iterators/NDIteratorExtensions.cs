using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public static class NDIteratorExtensions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDIterator<T> AsIterator<T>(this NDArray nd, bool autoreset = false) where T : unmanaged
        {
            return new NDIterator<T>(nd, autoreset);
        }

        public static NDIterator AsIterator(this NDArray nd, bool autoreset = false)
        {
#if _REGEN
            #region Compute
		    switch (nd.GetTypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return new NDIterator<#2>(nd, autoreset);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (nd.GetTypeCode)
            {
                case NPTypeCode.Boolean: return new NDIterator<bool>(nd, autoreset);
                case NPTypeCode.Byte: return new NDIterator<byte>(nd, autoreset);
                case NPTypeCode.Int16: return new NDIterator<short>(nd, autoreset);
                case NPTypeCode.UInt16: return new NDIterator<ushort>(nd, autoreset);
                case NPTypeCode.Int32: return new NDIterator<int>(nd, autoreset);
                case NPTypeCode.UInt32: return new NDIterator<uint>(nd, autoreset);
                case NPTypeCode.Int64: return new NDIterator<long>(nd, autoreset);
                case NPTypeCode.UInt64: return new NDIterator<ulong>(nd, autoreset);
                case NPTypeCode.Char: return new NDIterator<char>(nd, autoreset);
                case NPTypeCode.Double: return new NDIterator<double>(nd, autoreset);
                case NPTypeCode.Single: return new NDIterator<float>(nd, autoreset);
                case NPTypeCode.Decimal: return new NDIterator<decimal>(nd, autoreset);
                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
        }

        public static NDIterator AsIterator(this UnmanagedStorage us, bool autoreset = false)
        {
#if _REGEN
            #region Compute
		    switch (us.TypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return new NDIterator<#2>(us, autoreset);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (us.TypeCode)
            {
                case NPTypeCode.Boolean: return new NDIterator<bool>(us, autoreset);
                case NPTypeCode.Byte: return new NDIterator<byte>(us, autoreset);
                case NPTypeCode.Int16: return new NDIterator<short>(us, autoreset);
                case NPTypeCode.UInt16: return new NDIterator<ushort>(us, autoreset);
                case NPTypeCode.Int32: return new NDIterator<int>(us, autoreset);
                case NPTypeCode.UInt32: return new NDIterator<uint>(us, autoreset);
                case NPTypeCode.Int64: return new NDIterator<long>(us, autoreset);
                case NPTypeCode.UInt64: return new NDIterator<ulong>(us, autoreset);
                case NPTypeCode.Char: return new NDIterator<char>(us, autoreset);
                case NPTypeCode.Double: return new NDIterator<double>(us, autoreset);
                case NPTypeCode.Single: return new NDIterator<float>(us, autoreset);
                case NPTypeCode.Decimal: return new NDIterator<decimal>(us, autoreset);
                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
        }

        public static NDIterator AsIterator(this IArraySlice arr, Shape shape)
        {
#if _REGEN
            #region Compute
		    switch (arr.TypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return new NDIterator<#2>(arr, shape, null);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (arr.TypeCode)
            {
                case NPTypeCode.Boolean: return new NDIterator<bool>(arr, shape, null);
                case NPTypeCode.Byte: return new NDIterator<byte>(arr, shape, null);
                case NPTypeCode.Int16: return new NDIterator<short>(arr, shape, null);
                case NPTypeCode.UInt16: return new NDIterator<ushort>(arr, shape, null);
                case NPTypeCode.Int32: return new NDIterator<int>(arr, shape, null);
                case NPTypeCode.UInt32: return new NDIterator<uint>(arr, shape, null);
                case NPTypeCode.Int64: return new NDIterator<long>(arr, shape, null);
                case NPTypeCode.UInt64: return new NDIterator<ulong>(arr, shape, null);
                case NPTypeCode.Char: return new NDIterator<char>(arr, shape, null);
                case NPTypeCode.Double: return new NDIterator<double>(arr, shape, null);
                case NPTypeCode.Single: return new NDIterator<float>(arr, shape, null);
                case NPTypeCode.Decimal: return new NDIterator<decimal>(arr, shape, null);
                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
        }

        public static NDIterator AsIterator(this IArraySlice arr, Shape shape, bool autoreset)
        {
#if _REGEN
            #region Compute
		    switch (arr.TypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return new NDIterator<#2>(arr, shape, null, autoreset);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (arr.TypeCode)
            {
                case NPTypeCode.Boolean: return new NDIterator<bool>(arr, shape, null, autoreset);
                case NPTypeCode.Byte: return new NDIterator<byte>(arr, shape, null, autoreset);
                case NPTypeCode.Int16: return new NDIterator<short>(arr, shape, null, autoreset);
                case NPTypeCode.UInt16: return new NDIterator<ushort>(arr, shape, null, autoreset);
                case NPTypeCode.Int32: return new NDIterator<int>(arr, shape, null, autoreset);
                case NPTypeCode.UInt32: return new NDIterator<uint>(arr, shape, null, autoreset);
                case NPTypeCode.Int64: return new NDIterator<long>(arr, shape, null, autoreset);
                case NPTypeCode.UInt64: return new NDIterator<ulong>(arr, shape, null, autoreset);
                case NPTypeCode.Char: return new NDIterator<char>(arr, shape, null, autoreset);
                case NPTypeCode.Double: return new NDIterator<double>(arr, shape, null, autoreset);
                case NPTypeCode.Single: return new NDIterator<float>(arr, shape, null, autoreset);
                case NPTypeCode.Decimal: return new NDIterator<decimal>(arr, shape, null, autoreset);
                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
        }

        public static NDIterator AsIterator(this IArraySlice arr, Shape shape, Shape broadcastShape, bool autoReset)
        {
#if _REGEN
            #region Compute
		    switch (arr.TypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return new NDIterator<#2>(arr, shape, broadcastShape, autoReset);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (arr.TypeCode)
            {
                case NPTypeCode.Boolean: return new NDIterator<bool>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.Byte: return new NDIterator<byte>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.Int16: return new NDIterator<short>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.UInt16: return new NDIterator<ushort>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.Int32: return new NDIterator<int>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.UInt32: return new NDIterator<uint>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.Int64: return new NDIterator<long>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.UInt64: return new NDIterator<ulong>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.Char: return new NDIterator<char>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.Double: return new NDIterator<double>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.Single: return new NDIterator<float>(arr, shape, broadcastShape, autoReset);
                case NPTypeCode.Decimal: return new NDIterator<decimal>(arr, shape, broadcastShape, autoReset);
                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
        }
    }
}
