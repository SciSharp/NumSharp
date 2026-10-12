using System;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>numpy.strings</c> module: the vectorized string operations over fixed-width <c>S</c> / <c>U</c> arrays.
        ///     Strings phase 2 fills it subphase by subphase (P2.3: <see cref="add"/> and <see cref="multiply"/>; the length,
        ///     search, strip, pad and case families follow in P2.5 / P2.6).
        /// </summary>
        /// <remarks>
        ///     A C# <see cref="string"/> argument of a <c>numpy.strings</c> function is NumPy's Python <c>str</c> — a 0-d <c>U</c>
        ///     scalar of its length (<see cref="NDArray.Scalar(object)"/>): text is the only thing these parameters can mean, so
        ///     NumSharp's legacy <c>string → NDArray</c> conversion (a char array) does not apply here. A <c>byte[]</c> stays a
        ///     numeric <c>uint8</c> array; build an <c>S</c> scalar with <see cref="NDArray.Scalar(object)"/> for NumPy's
        ///     <c>bytes</c>.
        ///     <para>
        ///     <c>[ModuleName("np.strings")]</c> files its members under NumPy's <c>numpy.strings</c> in the API inventory
        ///     (<c>coverage/NumSharp.Tools.ApiInventory</c>), whose guard refuses a public nested static class of <c>np</c>
        ///     that names no module.
        ///     </para>
        /// </remarks>
        [ModuleName("np.strings")]
        public static partial class strings
        {
            /// <summary>
            ///     Element-wise string concatenation: NumPy's <c>np.strings.add</c>, which IS the <c>np.add</c> ufunc (probed:
            ///     <c>np.strings.add is np.add</c>). Two strings of one class concatenate at width <c>width(x1) + width(x2)</c>.
            /// </summary>
            /// <param name="x1">The first operand.</param>
            /// <param name="x2">The second operand.</param>
            /// <param name="out">An output to write into and return (a narrower or wider text array receives the result
            ///     truncated / zero-padded, NumPy's output cast), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c> (the string add promoter ignores it for two strings of one class).</param>
            /// <returns>The concatenation, or <paramref name="out"/>.</returns>
            /// <exception cref="TypeError">A bytes / str mix ("ufunc 'add' cannot use operands with types …"), a string and a
            ///     number (NumPy's no-loop text), or a result wider than 2^31 - 1 bytes ("addition result string of length …
            ///     is too large to store inside array.").</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive the result (the same_kind output cast,
            ///     e.g. a <c>U</c> result into an <c>S</c> out), or the operands do not broadcast with
            ///     <paramref name="out"/> / <paramref name="where"/>.</exception>
            /// <exception cref="IncorrectShapeException">The two operands do not broadcast.</exception>
            /// <exception cref="UnicodeDecodeError">An <c>S</c> result cast into a <c>U</c> <paramref name="out"/> meets a byte
            ///     ≥ 0x80.</exception>
            public static NDArray add(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
                => np.add(x1, x2, @out, where, dtype);

            /// <summary>Concatenation of an array and a C# string (NumPy's Python <c>str</c>, a 0-d <c>U</c> scalar).</summary>
            /// <param name="x1">The first operand.</param>
            /// <param name="x2">The string (not null).</param>
            /// <param name="out">An output to write into and return, or null.</param>
            /// <param name="where">A bool mask, or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>.</param>
            /// <returns>The concatenation, or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x2"/> is null.</exception>
            /// <exception cref="TypeError">An <c>S</c> or numeric <paramref name="x1"/> (see
            ///     <see cref="add(NDArray, NDArray, NDArray, NDArray, DType)"/>).</exception>
            public static NDArray add(NDArray x1, string x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                using var text = TextScalar(x2, nameof(x2));
                return np.add(x1, text, @out, where, dtype);
            }

            /// <summary>Concatenation of a C# string (NumPy's Python <c>str</c>) and an array, the string first.</summary>
            /// <param name="x1">The string (not null).</param>
            /// <param name="x2">The second operand.</param>
            /// <param name="out">An output to write into and return, or null.</param>
            /// <param name="where">A bool mask, or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>.</param>
            /// <returns>The concatenation, or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            /// <exception cref="TypeError">An <c>S</c> or numeric <paramref name="x2"/>.</exception>
            public static NDArray add(string x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                using var text = TextScalar(x1, nameof(x1));
                return np.add(text, x2, @out, where, dtype);
            }

            /// <summary>Concatenation of two C# strings: a 0-d <c>U</c> result (NumPy: <c>np.strings.add('a', 'b')</c> is <c>np.str_('ab')</c>).</summary>
            /// <param name="x1">The first string (not null).</param>
            /// <param name="x2">The second string (not null).</param>
            /// <returns>The 0-d concatenation.</returns>
            /// <exception cref="ArgumentNullException">A string is null.</exception>
            public static NDArray add(string x1, string x2)
            {
                using var a = TextScalar(x1, nameof(x1));
                using var b = TextScalar(x2, nameof(x2));
                return np.add(a, b);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.multiply(a, i)</c>: every string repeated <c>i</c> times (a count below 1 gives an empty
            ///     string), the result sized to the longest repetition — NumPy's two passes, statement for statement: the
            ///     length of every string, an overflow check, the output width <c>max(len(a) * max(i, 0))</c>, then the
            ///     <c>multiply</c> loop into an array laid out like <c>a</c> (<c>np.empty_like(a, shape=…)</c>).
            /// </summary>
            /// <param name="a">The strings (<c>S</c> or <c>U</c>).</param>
            /// <param name="i">The repeat counts: an integer array (any width, not bool), broadcast against <paramref name="a"/>.</param>
            /// <returns>A fresh text array of <paramref name="a"/>'s class (0-d for a 0-d operand pair).</returns>
            /// <exception cref="ArgumentNullException">An operand is null.</exception>
            /// <exception cref="TypeError"><paramref name="i"/> is not an integer array ("unsupported type float64 for operand
            ///     'i'", checked first); <paramref name="a"/> is not text (<c>str_len</c>'s no-loop text); or the output width
            ///     is not a valid descriptor (NumPy builds it from a string: a uint64 count makes the sizes float64, so
            ///     <c>'U6.0'</c> is "data type 'U6.0' not understood", as in NumPy 2.4.2).</exception>
            /// <exception cref="OverflowException">A string's <c>len * i</c> exceeds <c>sys.maxsize</c> ("Overflow encountered in
            ///     string multiply").</exception>
            /// <exception cref="ValueError">The broadcast result is empty ("zero-size array to reduction operation maximum
            ///     which has no identity": NumPy takes the max of the sizes).</exception>
            /// <exception cref="IncorrectShapeException"><paramref name="a"/> and <paramref name="i"/> do not broadcast.</exception>
            public static NDArray multiply(NDArray a, NDArray i)
            {
                if (a is null)
                    throw new ArgumentNullException(nameof(a));
                if (i is null)
                    throw new ArgumentNullException(nameof(i));
                // if not np.issubdtype(i.dtype, np.integer): raise TypeError(...) — bool is not a NumPy integer.
                if (!i.typecode.IsInteger())
                    throw new TypeError($"unsupported type {i.dtype} for operand 'i'");
                // i = np.maximum(i, 0)   (the C# literal is NumPy's weak Python int: i keeps its dtype)
                using var reps = np.maximum(i, 0);
                // a_len = str_len(a)
                using var aLen = StrLen(a);
                // if np.any(a_len > sys.maxsize / np.maximum(i, 1)): raise OverflowError(...)
                // sys.maxsize / x is the double loop: the Python int becomes 9.223372036854775807e18 and x a double.
                using (var floor = np.maximum(reps, 1))
                using (var divisor = floor.astype(np.float64))
                using (var maxsize = NDArray.Scalar(9.223372036854775807e18))
                using (var limit = np.true_divide(maxsize, divisor))
                using (var over = np.greater(aLen, limit))
                {
                    if (np.any(over))
                        throw new OverflowException("Overflow encountered in string multiply");
                }
                // buffersizes = a_len * i   (int64 times a uint64 count is float64, NumPy's promotion)
                using var buffersizes = np.multiply(aLen, reps);
                // out_dtype = f"{a.dtype.char}{buffersizes.max()}"   (raises for an empty result, as NumPy's max does)
                using var largest = np.amax(buffersizes);
                var outDtype = OutputDescriptor(a.dtype, largest);
                // out = np.empty_like(a, shape=buffersizes.shape, dtype=out_dtype)
                var @out = np.empty_like(a, outDtype, buffersizes.Shape);
                try
                {
                    // return _multiply_ufunc(a, i, out=out)
                    return np.multiply(a, reps, @out);
                }
                catch
                {
                    @out.Dispose();
                    throw;
                }
            }

            /// <summary>NumPy's <c>np.strings.multiply</c> of a C# string (NumPy's Python <c>str</c>): a 0-d text operand.</summary>
            /// <param name="a">The string (not null).</param>
            /// <param name="i">The repeat counts (an integer array).</param>
            /// <returns>The repetitions (0-d for a scalar count).</returns>
            /// <exception cref="ArgumentNullException">An operand is null.</exception>
            /// <exception cref="TypeError"><paramref name="i"/> is not an integer array.</exception>
            /// <exception cref="OverflowException">A repetition exceeds <c>sys.maxsize</c> characters.</exception>
            public static NDArray multiply(string a, NDArray i)
            {
                using var text = TextScalar(a, nameof(a));
                return multiply(text, i);
            }

            /// <summary>
            ///     NumPy's <c>str_len</c> for the internal size pass of <see cref="multiply(NDArray, NDArray)"/>: the length of
            ///     every string (trailing NULs never count, embedded ones do), int64.
            /// </summary>
            /// <param name="x">The operand.</param>
            /// <returns>A fresh int64 array of <paramref name="x"/>'s dimensions.</returns>
            /// <exception cref="TypeError"><paramref name="x"/> is not text: NumPy's <c>ufunc 'str_len' did not contain a loop
            ///     with signature matching types &lt;class 'numpy.dtypes.Int64DType'&gt; -&gt; None</c>.</exception>
            internal static NDArray StrLen(NDArray x)
            {
                if (!NDStringUfuncs.IsString(x))
                    throw new TypeError($"ufunc 'str_len' did not contain a loop with signature matching types {x.dtype.Meta} -> None");
                return NDStringArithmetic.Length(x);
            }

            /// <summary>
            ///     NumPy's <c>f"{a.dtype.char}{buffersizes.max()}"</c> descriptor: the class letter followed by the largest
            ///     size printed as NumPy prints that scalar (<c>6</c> for an int64 size, <c>6.0</c> for a float64 one), parsed
            ///     by <see cref="np.dtype(string)"/>. A size of 0 is NumPy's unsized <c>'U0'</c>, which an allocation turns
            ///     into width 1.
            /// </summary>
            /// <param name="text">The text operand's descriptor (its class letter).</param>
            /// <param name="largest">The 0-d largest size (int64, or float64 after a uint64 count).</param>
            /// <returns>The output descriptor.</returns>
            /// <exception cref="TypeError">The string is not a descriptor NumPy understands (a float size, a width past the
            ///     class limit): "data type 'U6.0' not understood".</exception>
            private static DType OutputDescriptor(DType text, NDArray largest)
            {
                string spec = $"{text.@char}{largest.ToString(false)}";
                DType parsed;
                try
                {
                    parsed = np.dtype(spec);
                }
                catch (NotSupportedException)
                {
                    // NumPy raises TypeError for a descriptor string it cannot parse.
                    throw new TypeError($"data type '{spec}' not understood");
                }
                // An unsized flexible descriptor ('U0' / 'S0') is width 1 once allocated (NumPy's empty_like).
                return parsed.itemsize == 0 ? ((StringDTypeMeta)parsed.Meta).Descr(1) : parsed;
            }

            /// <summary>A C# string as NumPy's Python <c>str</c>: a fresh 0-d <c>U</c> scalar of its length.</summary>
            /// <param name="text">The string.</param>
            /// <param name="name">The parameter name, for the null check.</param>
            /// <returns>The 0-d array (owned by the caller).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="text"/> is null.</exception>
            private static NDArray TextScalar(string text, string name)
                => NDArray.Scalar(text ?? throw new ArgumentNullException(name));
        }
    }
}
