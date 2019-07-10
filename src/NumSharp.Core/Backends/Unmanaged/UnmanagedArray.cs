using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends.Unmanaged
{
    public static class UnmanagedArray
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <remarks>Returns a copy.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<TOut> Cast<TIn, TOut>(this UnmanagedArray<TIn> source) where TIn : unmanaged where TOut : unmanaged
        {
            unsafe
            {
                var ret = new UnmanagedArray<TOut>(source.Count);
                var src = source.Address;
                var dst = ret.Address;
                var len = source.Count;
                var tc = Type.GetTypeCode(typeof(TOut));
                for (int i = 0; i < len; i++)
                {
                    *(dst + i) = (TOut)Convert.ChangeType((object)*(src + i), tc);
                } //TODO! seperate class for NP!

                return ret;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <remarks>Returns a copy.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<TOut> Cast<TIn, TOut>(this IUnmanagedArray source) where TIn : unmanaged where TOut : unmanaged
        {
            unsafe
            {
                var ret = new UnmanagedArray<TOut>(source.Count);
                var src = (TIn*)source.Address;
                var dst = ret.Address;
                var len = source.Count;
                var tc = Type.GetTypeCode(typeof(TOut));
                for (int i = 0; i < len; i++)
                {
                    *(dst + i) = (TOut)Convert.ChangeType((object)*(src + i), tc);
                } //TODO! seperate class for NP!

                return ret;
            }
        }
    }
}
