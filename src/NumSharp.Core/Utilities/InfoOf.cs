using System;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.InteropServices;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     Provides a cache for properties of <typeparamref name="T"/> that requires computation.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [SuppressMessage("ReSharper", "StaticMemberInGenericType")]
    internal class InfoOf<T>
    {
        public static readonly int Size;
        public static readonly NPTypeCode NPTypeCode;
        public static readonly T Zero;
        public static readonly T MaxValue;
        public static readonly T MinValue;

        static InfoOf()
        {
            NPTypeCode = typeof(T).GetTypeCode();
            Size = NPTypeCode.SizeOf();
            Zero = default;
            try
            {
                MaxValue = (T)NPTypeCode.MaxValue();
                MinValue = (T)NPTypeCode.MinValue();
            }
            catch (ArgumentOutOfRangeException) { }
        }
    }
}
