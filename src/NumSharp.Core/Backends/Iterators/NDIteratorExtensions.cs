using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

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
        [MethodImpl(Inline)]
        public static NDIterator<T> AsIterator<T>(this NDArray nd, bool autoreset = false) where T : unmanaged
        {
            return new NDIterator<T>(nd, autoreset);
        }

        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="nd"/>.
        /// </summary>
        /// <param name="nd">The ndarray to iterate.</param>
        /// <param name="autoreset">Should this iterator loop forever?</param>
        public static NDIterator AsIterator(this NDArray nd, bool autoreset = false)
        {
            return NpFunc.Invoke(nd.GetTypeCode, CreateFromNDArray<int>, nd, autoreset);
        }

        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="us"/>.
        /// </summary>
        /// <param name="us">The storage to iterate.</param>
        /// <param name="autoreset">Should this iterator loop forever?</param>
        public static NDIterator AsIterator(this UnmanagedStorage us, bool autoreset = false)
        {
            return NpFunc.Invoke(us.TypeCode, CreateFromStorage<int>, us, autoreset);
        }

        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="arr"/> as if it were shaped like <paramref name="shape"/>.
        /// </summary>
        /// <param name="arr">The IArraySlice to iterate.</param>
        /// <param name="shape">The shape to iterate with.</param>
        public static NDIterator AsIterator(this IArraySlice arr, Shape shape)
        {
            return NpFunc.Invoke(arr.TypeCode, CreateFromSlice<int>, arr, shape);
        }

        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="arr"/> as if it were shaped like <paramref name="shape"/>.
        /// </summary>
        /// <param name="arr">The IArraySlice to iterate.</param>
        /// <param name="shape">The original shape, non-broadcasted, to represent this iterator.</param>
        /// <param name="autoreset">Should this iterator loop forever?</param>
        public static NDIterator AsIterator(this IArraySlice arr, Shape shape, bool autoreset)
        {
            return NpFunc.Invoke(arr.TypeCode, CreateFromSliceAuto<int>, arr, shape, autoreset);
        }

        /// <summary>
        ///     Creates a new iterator to iterate given <paramref name="arr"/> as if it were shaped like <paramref name="shape"/>.
        /// </summary>
        /// <param name="arr">The IArraySlice to iterate.</param>
        /// <param name="shape">The original shape, non-broadcasted.</param>
        /// <param name="broadcastShape">The broadcasted shape of <paramref name="shape"/></param>
        /// <param name="autoReset">Should this iterator loop forever?</param>
        public static NDIterator AsIterator(this IArraySlice arr, Shape shape, Shape broadcastShape, bool autoReset)
        {
            return NpFunc.Invoke(arr.TypeCode, CreateFromSliceBroadcast<int>, arr, shape, broadcastShape, autoReset);
        }

        private static NDIterator CreateFromNDArray<T>(NDArray nd, bool autoreset) where T : unmanaged
            => new NDIterator<T>(nd, autoreset);

        private static NDIterator CreateFromStorage<T>(UnmanagedStorage us, bool autoreset) where T : unmanaged
            => new NDIterator<T>(us, autoreset);

        private static NDIterator CreateFromSlice<T>(IArraySlice arr, Shape shape) where T : unmanaged
            => new NDIterator<T>(arr, shape, null);

        private static NDIterator CreateFromSliceAuto<T>(IArraySlice arr, Shape shape, bool autoreset) where T : unmanaged
            => new NDIterator<T>(arr, shape, null, autoreset);

        private static NDIterator CreateFromSliceBroadcast<T>(IArraySlice arr, Shape shape, Shape broadcastShape, bool autoReset) where T : unmanaged
            => new NDIterator<T>(arr, shape, broadcastShape, autoReset);
    }
}
