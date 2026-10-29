using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     <c>np.copyto(dst, src)</c> with a Python <c>str</c> source: NumPy's <c>PyArray_FromAny(src)</c> makes it a 0-d
        ///     Unicode array of its own length (<c>'abcdef'</c> is <c>&lt;U6</c>), which is then assigned like any 0-d source —
        ///     truncated into a shorter <c>U</c> destination, ASCII-encoded into an <c>S</c> one (under a casting rule that
        ///     allows <c>U → S</c>: <c>'unsafe'</c>).
        /// </summary>
        /// <param name="dst">The array into which the text is copied.</param>
        /// <param name="src">The text (not null).</param>
        /// <param name="casting">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</param>
        /// <param name="where">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="dst"/> or <paramref name="src"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidCastException">The cast from <c>&lt;U{len}</c> to the destination's dtype is not allowed:
        ///     <c>Cannot cast scalar from dtype('&lt;U6') to dtype('S3') according to the rule 'same_kind'</c>.</exception>
        /// <exception cref="UnicodeEncodeError">The text is not ASCII and the destination is <c>S</c>.</exception>
        /// <exception cref="OverflowException">The destination is an integer dtype and the text is an integer outside it
        ///     (NumPy's LP64 OverflowError texts).</exception>
        /// <exception cref="SystemError">The text is one unit above U+10FFFF (only reachable through a raw view).</exception>
        /// <exception cref="ValueError">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>; also a numeric destination
        ///     the text does not parse into (<c>invalid literal for int() with base 10: np.str_('x')</c> — NumPy parses the 0-d
        ///     array the text became, so it quotes the NumPy scalar).</exception>
        /// <exception cref="TypeError">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <exception cref="ArgumentException">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <remarks>
        ///     A C# <see cref="string"/> would otherwise reach <see cref="copyto(NDArray, NDArray, string, NDArray)"/> through
        ///     NumSharp's implicit <c>string → NDArray</c> conversion, which still reads <c>"[1, 2]"</c>-shaped text as a numeric
        ///     matrix (a legacy NumSharp shorthand); this overload keeps every text a text, as NumPy does.
        ///     <para><see cref="NDAnchoredAttribute"/>: the destination and the mask stay reachable while the copy runs, as on
        ///     <see cref="copyto(NDArray, NDArray, string, NDArray)"/> (the text's own array is the <c>using</c> local).</para>
        /// </remarks>
        [NDAnchored]
        public static void copyto(NDArray dst, string src, string casting = "same_kind", NDArray @where = null)
        {
            NPY_CASTING rule = CopytoCastingConverter(casting);
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            using var value = NDStringArrays.FromScalar(src, null);
            DefaultEngine.ValidateWhereMask(@where);
            UnmanagedStorage.CopytoAssignArray(dst, value, @where, rule);
        }
    }
}
