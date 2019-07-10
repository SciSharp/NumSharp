using System.Runtime.InteropServices;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     Provides a cache for properties of <typeparamref name="T"/> that requires computation.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class InfoOf<T>
    {
        public static readonly int Size = Marshal.SizeOf<T>();
        public static readonly NPTypeCode NPTypeCode;
        public static readonly T Zero = default;
        public static readonly T MaxValue;
        public static readonly T MinValue;

        static InfoOf()
        {
            NPTypeCode = typeof(T).GetTypeCode();
            MaxValue = (T)NPTypeCode.MaxValue();
            MinValue = (T)NPTypeCode.MinValue();
        }
    }
}
