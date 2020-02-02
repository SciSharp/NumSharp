using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static class NDIteratorExtensions
    {

        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="nd"/>.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="nd">The ndarray to iterate.</param>
        /// <param name="autoreset">Should this iterator loop forever?</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDIterator<T> AsIterator<T>(this NDArray nd, bool autoreset = false) where T : unmanaged
        {
            return new NDIterator<T>(nd, autoreset);
        }

        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="nd"/>.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="nd">The ndarray to iterate.</param>
        /// <param name="autoreset">Should this iterator loop forever?</param>
        public static NDIterator AsIterator(this NDArray nd, bool autoreset = false)
        {
#if _REGEN1
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
			    case NPTypeCode.Int32: return new NDIterator<int>(nd, autoreset);
			    case NPTypeCode.Int64: return new NDIterator<long>(nd, autoreset);
			    case NPTypeCode.Single: return new NDIterator<float>(nd, autoreset);
			    case NPTypeCode.Double: return new NDIterator<double>(nd, autoreset);
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }

        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="nd"/>.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="us">The ndarray to iterate.</param>
        /// <param name="autoreset">Should this iterator loop forever?</param>
        public static NDIterator AsIterator(this UnmanagedStorage us, bool autoreset = false)
        {
#if _REGEN1
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
			    case NPTypeCode.Int32: return new NDIterator<int>(us, autoreset);
			    case NPTypeCode.Int64: return new NDIterator<long>(us, autoreset);
			    case NPTypeCode.Single: return new NDIterator<float>(us, autoreset);
			    case NPTypeCode.Double: return new NDIterator<double>(us, autoreset);
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }

        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="arr"/> as if it were shaped like <paramref name="shape"/>.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arr">The IArraySlice to iterate.</param>
        /// <param name="autoreset">Should this iterator loop forever?</param>
        public static NDIterator AsIterator(this IArraySlice arr, Shape shape)
        {
#if _REGEN1
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
			    case NPTypeCode.Int32: return new NDIterator<int>(arr, shape, null);
			    case NPTypeCode.Int64: return new NDIterator<long>(arr, shape, null);
			    case NPTypeCode.Single: return new NDIterator<float>(arr, shape, null);
			    case NPTypeCode.Double: return new NDIterator<double>(arr, shape, null);
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }

        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="arr"/> as if it were shaped like <paramref name="shape"/>.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arr">The IArraySlice to iterate.</param>
        /// <param name="autoreset">Should this iterator loop forever?</param>
        /// <param name="shape">The original shape, non-broadcasted, to represent this iterator.</param>
        public static NDIterator AsIterator(this IArraySlice arr, Shape shape, bool autoreset)
        {
#if _REGEN1
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
			    case NPTypeCode.Int32: return new NDIterator<int>(arr, shape, null, autoreset);
			    case NPTypeCode.Int64: return new NDIterator<long>(arr, shape, null, autoreset);
			    case NPTypeCode.Single: return new NDIterator<float>(arr, shape, null, autoreset);
			    case NPTypeCode.Double: return new NDIterator<double>(arr, shape, null, autoreset);
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }
        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="arr"/> as if it were shaped like <paramref name="shape"/>.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arr">The IArraySlice to iterate.</param>
        /// <param name="autoreset">Should this iterator loop forever?</param>
        /// <param name="shape">The original shape, non-broadcasted.</param>
        /// <param name="broadcastShape">The broadcasted shape of <paramref name="shape"/></param>
        public static NDIterator AsIterator(this IArraySlice arr, Shape shape, Shape broadcastShape, bool autoReset)
        {
#if _REGEN1
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
			    case NPTypeCode.Int32: return new NDIterator<int>(arr, shape, broadcastShape, autoReset);
			    case NPTypeCode.Int64: return new NDIterator<long>(arr, shape, broadcastShape, autoReset);
			    case NPTypeCode.Single: return new NDIterator<float>(arr, shape, broadcastShape, autoReset);
			    case NPTypeCode.Double: return new NDIterator<double>(arr, shape, broadcastShape, autoReset);
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }
    }
}
