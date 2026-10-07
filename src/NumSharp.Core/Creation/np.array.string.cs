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
        ///     Creates a 1-D fixed-width Unicode array from <paramref name="strArray"/> — NumPy's
        ///     <c>np.array(['a', 'bb'])</c>, <c>array(['a', 'bb'], dtype='&lt;U2')</c>: the width is the longest value in code
        ///     points (at least 1), each element NUL-padded.
        /// </summary>
        /// <param name="strArray">The values. A null entry is Python's <c>None</c>, which NumPy would only store in an
        /// <c>object</c> array, so it is refused here; pass a string dtype through <see cref="array(Array, DType, int, bool, char)"/>
        /// to store it as the text <c>"None"</c>.</param>
        /// <returns>The <c>U</c> array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="strArray"/> is null.</exception>
        /// <exception cref="NotSupportedException">An entry is null.</exception>
        /// <remarks>
        ///     <b>Breaking (NumPy parity):</b> this overload used to encode the strings into a <see cref="char"/> vector
        ///     (<c>"N l1 l2:contents"</c>), read back by the <c>NDArray.AsStringArray</c> layout parser.
        /// </remarks>
        [MethodImpl(Inline)]
        public static NDArray array(string[] strArray)
        {
            if (strArray == null)
                throw new ArgumentNullException(nameof(strArray));
            return Backends.Strings.NDStringArrays.FromManaged(strArray, null, 0, 'C');
        }
    }
}
