using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Create a vector ndarray of type <see cref="string"/>.
        /// Encode string array.
        /// format: [numOfRow lenOfRow1 lenOfRow2 contents]
        /// sample: [2 2 4 aacccc]
        /// </summary>
        /// <param name="strArray"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray array(string[] strArray)
        {
            if (strArray == null)
                throw new ArgumentNullException(nameof(strArray));
            if (strArray.Length == 0)
                return new NDArray(NPTypeCode.String, 0);

            // convert to bytes
            string meta = $"{strArray.Length}";
            foreach (var str in strArray)
                meta += $" {str.Length}";
            meta += $":{string.Join("", strArray)}";
            return new NDArray(meta.ToCharArray());
        }

        /// <summary>
        ///     Create a vector ndarray of type <see cref="char"/>.
        /// </summary>
        /// <param name="str">The string to turn into NDArray(NPTypeCode.Char)</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray array(string str)
        {
            if (str == null)
                throw new ArgumentNullException(nameof(str));

            if (str.Length == 0)
                return new NDArray(NPTypeCode.Char, 0);

            unsafe
            {
                var ret = new ArraySlice<char>(new UnmanagedMemoryBlock<char>(str.Length));
                fixed (char* strChars = str)
                {
                    var src = strChars;
                    var dst = ret.Address;
                    var len = sizeof(char) * str.Length;
                    Buffer.MemoryCopy(src, dst, len, len);
                }

                return new NDArray(ret);
            }
        }
    }
}
