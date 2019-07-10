using System.Runtime.InteropServices;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     Provides a cache for size of <typeparamref name="T"/>.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class SizeOf<T>
    {
        public static readonly int Size = Marshal.SizeOf<T>();
    }
}
