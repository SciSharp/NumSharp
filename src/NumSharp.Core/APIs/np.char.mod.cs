using System;
using NumSharp.Backends.Strings;

// =============================================================================
// np.char.mod.cs — numpy.char's mod, the % formatting it re-exports from numpy.strings
// =============================================================================
//
// numpy/_core/defchararray.py (NumPy 2.4.2) imports mod from numpy.strings, so np.char.mod IS np.strings.mod (the same
// function object):
//
//     def mod(a, values):
//         return _to_bytes_or_str_array(_vec_string(a, np.object_, '__mod__', (values,)), a)
//
// CPython's % per element (bytes.__mod__ for S, str.__mod__ for U and StringDType) against the broadcast element of
// np.asarray(values), the results a U / S array as wide as the longest one. The re-export converts `a` with the module's
// text rule (CharModule.TextOperand: a byte[] is Python bytes) and calls np.strings.mod, so every answer — a PLAIN result
// for a chararray operand (np.asarray drops the class), the layout, every error and its order — is np.strings'. The
// chararray's % operator (NDCharArray.Operators.cs) is NumPy's asarray(mod(self, values)) over this function.
//
// =============================================================================

namespace NumSharp
{
    public sealed partial class CharModule
    {
        /// <summary>
        ///     NumPy's <c>np.char.mod(a, values)</c>: <c>a % values</c> element-wise — CPython's <c>%</c> formatting of each
        ///     element of <paramref name="a"/> with the broadcast element of <paramref name="values"/>. It IS
        ///     <see cref="np.strings.mod(NDArray, object)"/> (NumPy re-exports the same function object).
        /// </summary>
        /// <param name="a">The format strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a C# string is a str, a <c>byte[]</c> Python bytes).</param>
        /// <param name="values">The values, converted as NumPy's <c>np.asarray</c> converts them and broadcast against
        ///     <paramref name="a"/>, ONE value per element (see <see cref="np.strings.mod(NDArray, object)"/>: a C# number is a
        ///     Python number, a tuple or a list a sequence, a dict a mapping for <c>%(key)s</c> specs, null Python's
        ///     <c>None</c>).</param>
        /// <returns>A fresh C-ordered PLAIN array over the broadcast shape — <c>U</c> / <c>S</c> (by <paramref name="a"/>'s
        ///     class) as wide as the longest result, at least 1; a default <c>StringDType</c> array for a <c>StringDType</c>
        ///     operand; an empty array of <paramref name="a"/>'s dtype when the broadcast is empty. A chararray operand gives a
        ///     plain array too (<c>np.asarray</c> drops the class); <c>chararray % values</c> is the chararray form.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>); the
        ///     value count does not fit the format (<c>not enough arguments for format string</c>, <c>not all arguments
        ///     converted during string formatting</c>); a conversion the value's type refuses (<c>%d format: a real number is
        ///     required, not numpy.str_</c>); <c>%(key)s</c> without a mapping; a <c>StringDType</c> result holding a surrogate.
        ///     </exception>
        /// <exception cref="ValueError">A malformed format (<c>incomplete format</c>, <c>unsupported format character 'y'
        ///     (0x79) at index 1</c>); a NaN under an integer conversion; a ragged <paramref name="values"/> list.</exception>
        /// <exception cref="OverflowException">An infinity under an integer conversion; a <c>%c</c> argument out of range;
        ///     an int too large for a float.</exception>
        /// <exception cref="IndexError">A <c>%(key)s</c> lookup on a NumPy scalar.</exception>
        /// <exception cref="KeyError">A <c>%(key)s</c> missing from a dict.</exception>
        /// <exception cref="SystemError">A <c>U</c> element of one unit past U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="NotSupportedException">A value with no NumSharp form (an object array of lists).</exception>
        /// <remarks>
        ///     <c>np.@char.mod(new[] { "%d", "%x" }, 255)</c> is <c>["255", "ff"]</c> (<c>U3</c>); an error ends the call at
        ///     the first element (C order) that raises, as in NumPy. Over <c>S</c>, <c>%s</c> writes a NumPy scalar's raw bytes
        ///     (the buffer protocol): <c>np.@char.mod(new[] { new byte[] { (byte)'%', (byte)'s' } }, "X")</c> holds the
        ///     <c>np.str_</c>'s UCS-4 bytes <c>b"X\x00\x00\x00"</c>, read back as <c>b"X"</c> in an <c>S4</c> result.
        /// </remarks>
        [NDScoped]
        public NDArray mod(object a, object values) => np.strings.mod(TextOperand(a, nameof(a)), values);
    }
}
