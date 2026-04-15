using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Static utility methods for type information.
    /// </summary>
    public static class InfoOf
    {
        /// <summary>
        /// Get the size in bytes of the given NPTypeCode.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int GetSize(NPTypeCode typeCode) => typeCode.SizeOf();
    }

    /// <summary>
    ///     Provides a cache for properties of <typeparamref name="T"/> that requires computation.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [SuppressMessage("ReSharper", "StaticMemberInGenericType")]
    public class InfoOf<T>
    {
        public static readonly int Size;
        public static readonly NPTypeCode NPTypeCode;
        public static readonly T Zero;
        public static readonly T MaxValue;
        public static readonly T MinValue;

        static InfoOf()
        {
            NPTypeCode = typeof(T).GetTypeCode();
            Zero = default;
            try
            {
                MaxValue = (T)NPTypeCode.MaxValue();
                MinValue = (T)NPTypeCode.MinValue();
            }
            catch (ArgumentOutOfRangeException) { }

            switch (NPTypeCode)
            {
                case NPTypeCode.Boolean:
                    Size = 1;
                    break;
                case NPTypeCode.Char:
                    Size = 2;
                    break;
                case NPTypeCode.SByte:
                    Size = 1;
                    break;
                case NPTypeCode.Byte:
                    Size = 1;
                    break;
                case NPTypeCode.Int16:
                    Size = 2;
                    break;
                case NPTypeCode.UInt16:
                    Size = 2;
                    break;
                case NPTypeCode.Int32:
                    Size = 4;
                    break;
                case NPTypeCode.UInt32:
                    Size = 4;
                    break;
                case NPTypeCode.Int64:
                    Size = 8;
                    break;
                case NPTypeCode.UInt64:
                    Size = 8;
                    break;
                case NPTypeCode.Half:
                    Size = 2;
                    break;
                case NPTypeCode.Single:
                    Size = 4;
                    break;
                case NPTypeCode.Double:
                    Size = 8;
                    break;
                case NPTypeCode.Decimal:
                    Size = 16;
                    break;
                case NPTypeCode.String:
                    break;
                case NPTypeCode.Complex:
                    Size = Marshal.SizeOf<T>();
                    break;
                default:
                    // NPTypeCode.Empty covers non-NumPy types (DateTime, DateTimeOffset,
                    // TimeSpan, DateTime64, user structs). Marshal.SizeOf requires
                    // unmanaged structs and throws for DateTime; Unsafe.SizeOf works
                    // for any struct layout.
                    Size = Unsafe.SizeOf<T>();
                    break;
            }
        }
    }
}
