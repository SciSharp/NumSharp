using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Converts a string to a vector ndarray of bytes.
        /// </summary>
        /// <param name="str"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray FromString(string str) => np.array(str);

        /// <summary>
        ///     Converts the entire <see cref="NDArray"/> to a string.
        /// </summary>
        /// <remarks>Performs a copy due to String .net-framework limitations.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static string AsString(NDArray arr)
        {
            unsafe
            {
                Debug.Assert(arr.typecode == NPTypeCode.Char);
                return new string((char*)arr.Address, 0, arr.size);
            }
        }

        /// <summary>
        /// Convert to String[] from NDArray
        /// </summary>
        /// <param name="arr"></param>
        /// <returns></returns>
        public static string[] AsStringArray(NDArray arr)
        {
            var chars = arr.ToString();
            int pos = chars.IndexOf(' ');
            int numOfRows = int.Parse(chars.Substring(0, pos));
            int posOfContent = chars.IndexOf(':');
            string contents = chars.Substring(posOfContent + 1);
            int[] strLengths = chars.Substring(pos + 1, posOfContent - pos - 1)
                .Split(' ')
                .Select(x => int.Parse(x))
                .ToArray();
            string[] strArray = new string[numOfRows];
            int currentPos = 0;
            for (int i = 0; i < numOfRows; i++)
            {
                strArray[i] = contents.Substring(currentPos, strLengths[i]);
                currentPos += strLengths[i];
            }

            return strArray;
        }

        /// <summary>
        ///     Get a string out of a vector of chars.
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        /// <remarks>Performs a copy due to String .net-framework limitations.</remarks>
        public string GetString(params int[] indices)
        {
            unsafe
            {
                if (Shape.dimensions.Length - 1 != indices.Length)
                    throw new ArgumentOutOfRangeException(nameof(indices), "GetString(int[]) can only accept coordinates that point to a vector of chars.");

                Debug.Assert(typecode == NPTypeCode.Char);

                UnmanagedStorage src = Storage.GetData(indices);
                Debug.Assert(src.Shape.NDim == 1);

                if (!Shape.IsContiguous || Shape.ModifiedStrides)
                {
                    //this works faster than cloning.
                    var ret = new string('\0', src.Count);
                    fixed (char* retChars = ret)
                    {
                        var dst = new UnmanagedStorage(new ArraySlice<char>(new UnmanagedMemoryBlock<char>(retChars, ret.Length)), src.Shape.Clean());
                        MultiIterator.Assign(dst, src);
                    }

                    return ret;
                }

                //new string always performs a copy, there is no need to keep reference to arr's unmanaged storage.
                return new string((char*)src.Address, 0, src.Count);
            }
        }

        public void SetString(string value, params int[] indices)
        {
            Debug.Assert(typecode == NPTypeCode.Char);

            // ReSharper disable once ReplaceWithStringIsNullOrEmpty
            if (value == null || value.Length == 0)
                throw new ArgumentException("Value cannot be null or empty.", nameof(value));

            if (Shape.dimensions.Length - 1 != indices.Length)
                throw new ArgumentOutOfRangeException(nameof(indices), "SetString(string, int[] indices) can only accept coordinates that point to a vector of chars.");

            unsafe
            {
                if (Shape.IsContiguous)
                {
                    var dst = (char*)Address + Shape.GetOffset(indices);
                    fixed (char* src = value)
                    {
                        var len = sizeof(char) * value.Length;
                        Buffer.MemoryCopy(src, dst, len, len);
                    }
                }
                else
                {
                    fixed (char* strChars = value)
                    {
                        SetData(new ArraySlice<char>(new UnmanagedMemoryBlock<char>(strChars, value.Length)), indices);
                    }
                }
            }
        }

        /// <summary>
        ///     Get a string out of a vector of chars.
        /// </summary>
        /// <remarks>Performs a copy due to String .net-framework limitations.</remarks>
        public string GetStringAt(int offset)
        {
            Debug.Assert(typecode == NPTypeCode.Char);
            Debug.Assert(offset < Array.Count);

            return GetString(Shape.GetCoordinates(offset));
        }

        public void SetStringAt(string value, int offset)
        {
            Debug.Assert(typecode == NPTypeCode.Char);
            Debug.Assert(offset < Array.Count);

            SetString(value, Shape.GetCoordinates(offset));
        }
    }
}
