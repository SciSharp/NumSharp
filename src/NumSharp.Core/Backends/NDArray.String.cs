using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Converts a string to a 1-D <see cref="char"/> array of its UTF-16 code units — NumSharp's character-vector
        ///     view of text (the <see cref="NPTypeCode.Char"/> dtype), distinct from NumPy's fixed-width string dtypes:
        ///     <c>np.array("abc")</c> is the 0-d <c>&lt;U3</c> array, <c>NDArray.FromString("abc")</c> the char vector
        ///     <c>['a', 'b', 'c']</c>.
        /// </summary>
        /// <param name="str">The text.</param>
        /// <returns>A new char vector (an empty string gives an empty vector).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="str"/> is null.</exception>
        public static NDArray FromString(string str)
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
                    var len = sizeof(char) * str.Length;
                    Buffer.MemoryCopy(strChars, ret.Address, len, len);
                }

                return new NDArray(ret);
            }
        }

        /// <summary>
        ///     Whether <paramref name="other"/> holds elements of exactly this array's element type — the same storage lane
        ///     and, for a fixed-width string, the same class and width — so its bytes can be copied element for element
        ///     without a conversion. The lane alone is not enough for strings: <c>&lt;U1</c> and <c>&lt;U5</c> (and <c>S5</c>)
        ///     all live in <see cref="NPTypeCode.String"/>, and copying one as the other reads or writes past its elements.
        /// </summary>
        /// <param name="other">The other array.</param>
        /// <returns>True when no element conversion is needed. Two NEP 55 <c>StringDType</c> arrays need EQUAL descriptors (the
        ///     same <c>na_object</c> and <c>coerce</c>): a copy between other descriptors converts NA values, and their elements
        ///     still move through the allocators, never as raw bytes.</returns>
        internal bool HasSameElementType(NDArray other)
        {
            if (typecode != other.typecode)
                return false;
            if (typecode != NPTypeCode.String)
                return true;
            var mine = dtype;
            var theirs = other.dtype;
            if (mine.Meta is VStringDTypeMeta || theirs.Meta is VStringDTypeMeta)
                return mine.Equals(theirs);
            return mine.Meta == theirs.Meta && mine.itemsize == theirs.itemsize;
        }

        /// <summary>
        ///     Converts the entire <see cref="NDArray"/> to a string.
        /// </summary>
        /// <remarks>Performs a copy due to String .net-framework limitations.</remarks>
        [MethodImpl(Inline)]
        public static string AsString(NDArray arr)
        {
            unsafe
            {
                Debug.Assert(arr.typecode == NPTypeCode.Char);
                if (arr.size > int.MaxValue)
                    throw new InvalidOperationException("String size exceeds int.MaxValue.");
                return new string((char*)arr.Address, 0, (int)arr.size);
            }
        }

        /// <summary>
        ///     The elements of a fixed-width Unicode (<c>U</c>) array as C# strings, in C order — NumPy's
        ///     <c>arr.ravel().tolist()</c>. Each value has its trailing NULs removed (NumPy's <c>getitem</c>); any layout
        ///     (strided, transposed, broadcast) is read in logical order.
        /// </summary>
        /// <param name="arr">A <c>U</c> array of any shape.</param>
        /// <returns>The values.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="InvalidCastException"><paramref name="arr"/> is not a <c>U</c> array (an <c>S</c> array's
        /// elements are bytes: read them with <c>ToArray&lt;byte[]&gt;</c>-style element access).</exception>
        /// <exception cref="SystemError">A unit above <c>U+10FFFF</c>.</exception>
        /// <remarks>
        ///     <b>Breaking:</b> this used to parse the <c>"N l1 l2:contents"</c> char-vector layout the old
        ///     <c>np.array(string[])</c> produced; that layout no longer exists.
        ///     <para>A NEP 55 <c>StringDType</c> array is accepted too: each element is its text, an NA element the descriptor's
        ///     NA object when that is a string or <c>None</c> (null); an NA element of any other NA object (a float NaN) cannot
        ///     be a C# string and is refused — read such arrays with <see cref="tolist"/>.</para>
        /// </remarks>
        public static string[] AsStringArray(NDArray arr)
        {
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));
            if (arr.Storage.IsVString)
                return Backends.Strings.NDVStringMove.ToStrings(arr);
            if (!(arr.Storage.StringSlice is { IsUnicode: true }))
                throw new InvalidCastException($"AsStringArray reads a fixed-width Unicode (U) array, not {arr.dtype.ToString(true)}.");
            var result = new string[arr.size];
            if (arr.Shape.IsContiguous)
            {
                // A C-contiguous array is one run of elements: decode each straight from its units (the per-element
                // GetAtIndex path re-derives the element's offset and dispatches on the dtype every time).
                unsafe
                {
                    var slice = arr.Storage.StringSlice;
                    int chars = slice.Chars;
                    uint* units = (uint*)(arr.Storage.Address + arr.Shape.offset * (long)slice.ItemSize);
                    for (long i = 0; i < result.LongLength; i++)
                        result[i] = Backends.Strings.NDStringCodec.ReadUcs4(units + i * chars, chars);
                }
                return result;
            }
            for (long i = 0; i < result.LongLength; i++)
                result[i] = (string)arr.Storage.GetAtIndex(i);
            return result;
        }

        /// <summary>
        ///     Reads a vector of chars (the <see cref="NPTypeCode.Char"/> dtype) as one C# string: the innermost axis at the
        ///     given outer coordinates. This is NumSharp's char-vector API, not NumPy element access: a fixed-width string
        ///     array (<c>U</c> / <c>S</c>) stores one string per ELEMENT — read it with <see cref="item(long)"/> or
        ///     <see cref="AsStringArray"/>.
        /// </summary>
        /// <param name="indices">The coordinates of the vector: one fewer than the array's rank.</param>
        /// <returns>A new string holding a copy of the vector's chars.</returns>
        /// <exception cref="InvalidOperationException">The array is not a <see cref="NPTypeCode.Char"/> array (reading a
        ///     <c>U</c> array's UCS-4 bytes as UTF-16 chars would return garbage), or the vector is longer than
        ///     <see cref="int.MaxValue"/> chars.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="indices"/> does not address a vector.</exception>
        /// <remarks>Performs a copy due to String .net-framework limitations.</remarks>
        public string GetString(params long[] indices)
        {
            ThrowIfNotCharVector(nameof(GetString));
            unsafe
            {
                if (Shape.dimensions.Length - 1 != indices.Length)
                    throw new ArgumentOutOfRangeException(nameof(indices), "GetString(long[]) can only accept coordinates that point to a vector of chars.");

                Debug.Assert(typecode == NPTypeCode.Char);

                UnmanagedStorage src = Storage.GetData(indices);
                Debug.Assert(src.Shape.NDim == 1);

                if (!Shape.IsContiguous)
                {
                    //this works faster than cloning.
                    if (src.Count > int.MaxValue)
                        throw new InvalidOperationException("String size exceeds int.MaxValue.");
                    var ret = new string('\0', (int)src.Count);
                    fixed (char* retChars = ret)
                    {
                        var dst = new UnmanagedStorage(new ArraySlice<char>(new UnmanagedMemoryBlock<char>(retChars, ret.Length)), src.Shape.Clean());
                        NDIter.Copy(dst, src);
                    }

                    return ret;
                }

                //new string always performs a copy, there is no need to keep reference to arr's unmanaged storage.
                if (src.Count > int.MaxValue)
                    throw new InvalidOperationException("String size exceeds int.MaxValue.");
                return new string((char*)src.Address, 0, (int)src.Count);
            }
        }

        /// <summary>
        ///     Writes a C# string's UTF-16 chars into a vector of chars (the <see cref="NPTypeCode.Char"/> dtype) at the given
        ///     outer coordinates — NumSharp's char-vector API. A fixed-width string array (<c>U</c> / <c>S</c>) takes a
        ///     string per ELEMENT through its indexer (<c>a[i] = "text"</c>, truncated to the element width like NumPy).
        /// </summary>
        /// <param name="value">The text (its chars are copied from the vector's start; a longer text writes past the
        ///     vector's end only within the buffer the coordinates address, so keep it within the innermost length).</param>
        /// <param name="indices">The coordinates of the vector: one fewer than the array's rank.</param>
        /// <exception cref="InvalidOperationException">The array is not a <see cref="NPTypeCode.Char"/> array.</exception>
        /// <exception cref="ArgumentException"><paramref name="value"/> is null or empty.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="indices"/> does not address a vector.</exception>
        /// <exception cref="ValueError">The array is read-only.</exception>
        public void SetString(string value, params long[] indices)
        {
            ThrowIfNotCharVector(nameof(SetString));
            NumSharpException.ThrowIfNotWriteable(Shape);

            // ReSharper disable once ReplaceWithStringIsNullOrEmpty
            if (value == null || value.Length == 0)
                throw new ArgumentException("Value cannot be null or empty.", nameof(value));

            if (Shape.dimensions.Length - 1 != indices.Length)
                throw new ArgumentOutOfRangeException(nameof(indices), "SetString(string, long[] indices) can only accept coordinates that point to a vector of chars.");

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
        ///     <see cref="GetString"/> addressed by a flat offset of a char vector's element (its coordinates minus the
        ///     innermost one select the vector).
        /// </summary>
        /// <param name="offset">A flat element offset inside the vector to read.</param>
        /// <returns>A new string holding a copy of the vector's chars.</returns>
        /// <exception cref="InvalidOperationException">The array is not a <see cref="NPTypeCode.Char"/> array.</exception>
        /// <remarks>Performs a copy due to String .net-framework limitations.</remarks>
        public string GetStringAt(long offset)
        {
            ThrowIfNotCharVector(nameof(GetStringAt));
            Debug.Assert(offset < Array.Count);

            return GetString(Shape.GetCoordinates(offset));
        }

        /// <summary>
        ///     <see cref="SetString"/> addressed by a flat offset of a char vector's element.
        /// </summary>
        /// <param name="value">The text.</param>
        /// <param name="offset">A flat element offset inside the vector to write.</param>
        /// <exception cref="InvalidOperationException">The array is not a <see cref="NPTypeCode.Char"/> array.</exception>
        /// <exception cref="ArgumentException"><paramref name="value"/> is null or empty.</exception>
        /// <exception cref="ValueError">The array is read-only.</exception>
        public void SetStringAt(string value, long offset)
        {
            ThrowIfNotCharVector(nameof(SetStringAt));
            Debug.Assert(offset < Array.Count);

            SetString(value, Shape.GetCoordinates(offset));
        }

        /// <summary>
        ///     The guard of the char-vector API: these members reinterpret the buffer as UTF-16 chars, which is only right
        ///     for the <see cref="NPTypeCode.Char"/> dtype. A <c>Debug.Assert</c> used to stand here, so a Release build read
        ///     a <c>U</c> array's UCS-4 bytes as chars and returned garbage without an error.
        /// </summary>
        /// <param name="member">The calling member, named in the message.</param>
        /// <exception cref="InvalidOperationException">The array is not a <see cref="NPTypeCode.Char"/> array.</exception>
        private void ThrowIfNotCharVector(string member)
        {
            if (typecode == NPTypeCode.Char)
                return;
            throw new InvalidOperationException(
                $"{member} works on a vector of chars (the Char dtype), not on {dtype.ToString(true)}. A fixed-width string " +
                "array stores one string per element: read one with item(...), all of them with NDArray.AsStringArray, and " +
                "write one through the indexer (a[i] = \"text\").");
        }
    }
}
