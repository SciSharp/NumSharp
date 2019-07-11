using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Backends.Unmanaged
{
    public static class UnmanagedMemoryBlock
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <remarks>Returns a copy.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<TOut> Cast<TIn, TOut>(this UnmanagedMemoryBlock<TIn> source) where TIn : unmanaged where TOut : unmanaged
        {
            unsafe
            {
                var ret = new UnmanagedMemoryBlock<TOut>(source.Count);
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
        public static UnmanagedMemoryBlock<TOut> Cast<TIn, TOut>(this IUnmanagedArray source) where TIn : unmanaged where TOut : unmanaged
        {
            unsafe
            {
                var len = ((IMemoryBlock)source).Count;
                var ret = new UnmanagedMemoryBlock<TOut>(len);
                var src = (TIn*)source.Address;
                var dst = ret.Address;
                var tc = Type.GetTypeCode(typeof(TOut));
                for (int i = 0; i < len; i++)
                {
                    *(dst + i) = (TOut)Convert.ChangeType(*(src + i), tc);
                } //TODO! seperate class for NP!

                return ret;
            }
        }
    }
}
