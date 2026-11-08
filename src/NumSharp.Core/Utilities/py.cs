using System;
using System.Numerics;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Implements Python utility functions that are often used in connection with numpy
    /// </summary>
    public static class py
    {
        public static int[] range(int n)
        {
            var a = new int[n];
            for (int i = 0; i < n; i++)
                a[i] = i;
            return a;
        }
        /// <summary>
        ///     CPython 3.12's <c>complex(str)</c>: parses one Python complex literal into a <see cref="Complex"/>, with the
        ///     grammar NumPy uses for a <c>str → complex128</c> conversion (<c>np.array(['1+2j']).astype(complex)</c>).
        /// </summary>
        /// <param name="input">The text: <c>'10+5j'</c>, <c>'3-2j'</c>, <c>'4j'</c>, <c>'j'</c>, <c>'5'</c>, <c>'1e3'</c>,
        ///     <c>'inf'</c> / <c>'nan'</c> in any case, <c>'(1+2j)'</c>, either <c>j</c> or <c>J</c>, surrounding whitespace,
        ///     underscores between digits, non-ASCII decimal digits.</param>
        /// <returns>The complex value.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="input"/> is null.</exception>
        /// <exception cref="ValueError">The text is not a complex literal (CPython's text: <c>complex() arg is a malformed
        ///     string</c>) — the empty string and whitespace included.</exception>
        /// <remarks>
        ///     This replaced a culture-dependent regular expression (<c>double.Parse</c> in the CURRENT culture, so under
        ///     de-DE <c>'1.5'</c> read 15) that refused <c>'1e3'</c>, <c>'inf'</c>, <c>'(1+2j)'</c>, whitespace and
        ///     underscores, accepted <c>'1+'</c>, and threw <see cref="FormatException"/>. The parser is the one every
        ///     NumSharp text → complex route runs (<see cref="Converts.ToComplex(string)"/>, <c>astype</c>, setitem), so the
        ///     answers agree.
        /// </remarks>
        public static Complex complex(string input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            return Converts.ParseNumPyText<Complex>(input, NPTypeCode.Complex);
        }
    }
}
