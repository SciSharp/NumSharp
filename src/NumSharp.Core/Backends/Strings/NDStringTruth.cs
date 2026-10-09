using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The truth value of fixed-width strings: an element is true when any of its code units is nonzero — a non-empty
    ///     string once NumPy strips the trailing NULs (<c>'\0' → False</c>, <c>' ' → True</c>, <c>'False' → True</c>,
    ///     <c>'\0a' → True</c>; probed against 2.4.2). Every NumPy function that reads a string's truth reaches it through
    ///     one of TWO routes, which differ in one case only:
    ///     <list type="bullet">
    ///       <item><description><b>The bool cast</b> (<see cref="Flavor.Cast"/>): <c>astype(bool)</c>,
    ///       <c>asarray(dtype=bool)</c>, <c>any</c> / <c>all</c> (<c>logical_or</c> / <c>logical_and.reduce</c>), the logical
    ///       ufuncs, <c>where(c, x, y)</c>, <c>place</c>, <c>putmask</c>, <c>copyto</c> into bool. NumPy's legacy
    ///       string-to-bool cast builds a Python <c>str</c> per element from its units minus the trailing NULs, and CPython
    ///       refuses only a ONE-unit string above U+10FFFF (<c>PyUnicode_New(1, ch)</c> raises <c>SystemError: invalid
    ///       maximum character passed to PyUnicode_New</c>); a longer string's maximum character is capped at U+10FFFF, so
    ///       <c>[0x110000, 'A']</c> casts to True while <c>[0x110000, '\0']</c> raises (probed 2.4.2).</description></item>
    ///       <item><description><b>The nonzero function</b> (<see cref="Flavor.Nonzero"/>): <c>nonzero</c>,
    ///       <c>count_nonzero</c>, <c>argwhere</c>, <c>flatnonzero</c>, <c>where(c)</c>, <c>compress</c>, <c>extract</c>,
    ///       <c>trim_zeros</c>, <c>bool(0-d)</c>. <c>STRING_nonzero</c> / <c>UNICODE_nonzero</c> test the units directly and
    ///       never raise.</description></item>
    ///     </list>
    ///     Both are computed by one IL kernel (<see cref="ILKernelGenerator.GetTextTruthKernel"/>); the cast flavor also
    ///     tests for that lone unit. The result is an ordinary bool array, so every caller then runs its existing
    ///     bool machinery (SIMD nonzero, the any / all reductions with axis / keepdims / out / where, np.where, ...), which
    ///     is NumPy's own structure: a cast, then the bool loop.
    /// </summary>
    internal static class NDStringTruth
    {
        /// <summary>NumPy's text for a one-unit <c>U</c> element above U+10FFFF met by the string-to-bool cast (a SystemError).</summary>
        internal const string InvalidCodePointMessage = "invalid maximum character passed to PyUnicode_New";

        /// <summary>The route a caller's NumPy function reads the truth through (see the type summary).</summary>
        internal enum Flavor
        {
            /// <summary>The string-to-bool cast: raises SystemError for a <c>U</c> element that is one unit above U+10FFFF.</summary>
            Cast,

            /// <summary>The dtype's nonzero function: never raises.</summary>
            Nonzero,
        }

        /// <summary>
        ///     The truth array of a string array: a fresh bool array of its dimensions, laid out as NumPy's
        ///     <c>astype(bool, order=order)</c> lays out its result (<c>'K'</c>: the input's memory order).
        /// </summary>
        /// <param name="text">The string array (<c>S</c> or <c>U</c>, any layout, 0-d included).</param>
        /// <param name="flavor">The route (decides whether an invalid code point raises).</param>
        /// <param name="order">The result layout: <c>'K'</c> (default), <c>'A'</c>, <c>'C'</c> or <c>'F'</c>.</param>
        /// <returns>The bool array (owned by the caller).</returns>
        /// <exception cref="ArgumentException"><paramref name="text"/> is not a string array.</exception>
        /// <exception cref="SystemError"><see cref="Flavor.Cast"/> over a <c>U</c> array with an element that is one unit
        ///     above U+10FFFF (its first unit above it, every later unit zero).</exception>
        internal static NDArray Truth(NDArray text, Flavor flavor, char order = 'K')
        {
            if (!NDStringUfuncs.IsString(text))
                throw new ArgumentException($"A truth array is computed from a string array, not {text?.dtype}.", nameof(text));
            var result = new NDArray(NPTypeCode.Boolean, ResultShape(text.Shape, order), false);
            try
            {
                Fill(text, result, flavor);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Writes the truth of every element of <paramref name="text"/> into <paramref name="result"/> (same dimensions,
        ///     any layouts): one kernel call when both walk one linear order, else an NDIter pass.
        /// </summary>
        /// <param name="text">The string array.</param>
        /// <param name="result">A bool array of the same dimensions, not overlapping <paramref name="text"/>.</param>
        /// <param name="flavor">The route.</param>
        /// <exception cref="SystemError">The cast flavor met a <c>U</c> element that is one unit above U+10FFFF.</exception>
        private static unsafe void Fill(NDArray text, NDArray result, Flavor flavor)
        {
            if (result.size == 0)
                return;
            int unit = text.dtype.kind == 'U' ? 4 : 1;
            bool check = flavor == Flavor.Cast && unit == 4;
            var kernel = GetKernel(unit, text.dtype.itemsize, check);
            int invalid = 0;

            var ts = text.Shape;
            var rs = result.Shape;
            // Linear walk: both C-contiguous or both F-contiguous (a 0-d / one-element pair is both).
            if (ts.IsContiguous && rs.IsContiguous || ts.IsFContiguous && rs.IsFContiguous)
            {
                void** ptrs = stackalloc void*[2];
                long* strides = stackalloc long[2];
                ptrs[0] = (byte*)text.Address + ts.offset * (long)text.dtype.itemsize;
                ptrs[1] = (byte*)result.Address + rs.offset;
                strides[0] = text.dtype.itemsize;
                strides[1] = 1;
                kernel(ptrs, strides, result.size, &invalid);
                GC.KeepAlive(text);
                GC.KeepAlive(result);
            }
            else
            {
                using var iter = NDIterRef.AdvancedNew(2, new[] { text, result },
                    NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, opDescrs: new DType[2]);
                iter.ForEach(kernel, &invalid);
            }

            if (invalid != 0)
                throw new SystemError(InvalidCodePointMessage);
        }

        /// <summary>
        ///     The layout of a truth array: NumPy's <c>PyArray_NewLikeArray</c> for the requested order — <c>'K'</c> keeps a
        ///     strided input's sorted stride order (and C / F for a contiguous one), <c>'A'</c> / <c>'C'</c> / <c>'F'</c>
        ///     resolve as for any copy (the rule <see cref="NDIter.CopyAs"/> applies to a string copy).
        /// </summary>
        /// <param name="source">The string array's shape.</param>
        /// <param name="order">The order character.</param>
        /// <returns>A dense, offset-0 shape of the source's dimensions.</returns>
        private static Shape ResultShape(Shape source, char order)
        {
            if (source.NDim == 0)
                return Shape.NewScalar();
            if ((order == 'K' || order == 'k') && !source.IsContiguous && !source.IsFContiguous)
                return source.KeepOrder();
            return Shape.Adopt((long[])source.dimensions.Clone(), OrderResolver.Resolve(order, source));
        }

        /// <summary>The truth kernel: the IL kernel, or the managed element loop where dynamic code is unavailable.</summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="checkCodePoints">The cast flavor over <c>U</c>: also flag an element that is one unit above
        ///     U+10FFFF.</param>
        /// <returns>The kernel (never null).</returns>
        internal static NDInnerLoopFunc GetKernel(int unit, int width, bool checkCodePoints)
            => ILKernelGenerator.GetTextTruthKernel(unit, width, checkCodePoints)
               ?? ManagedKernel(unit, width, checkCodePoints);

        /// <summary>
        ///     The truth as a managed <see cref="NDInnerLoopFunc"/> (hosts without dynamic code; the IL kernel is tested
        ///     against it): <c>out = any byte nonzero</c>, and for the cast flavor <c>*(int*)auxdata = 1</c> when a U element
        ///     is one unit above U+10FFFF — its first unit above it and every later byte zero (the one-unit <c>str</c>
        ///     CPython's <c>PyUnicode_New</c> refuses; a longer string builds with its maximum character capped).
        /// </summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="checkCodePoints">The cast flavor over <c>U</c>.</param>
        /// <returns>The kernel.</returns>
        internal static unsafe NDInnerLoopFunc ManagedKernel(int unit, int width, bool checkCodePoints)
            => (dataptrs, strides, count, auxdata) =>
            {
                byte* p = (byte*)dataptrs[0], po = (byte*)dataptrs[1];
                long sp = strides[0], so = strides[1];
                bool invalid = false;
                for (long i = 0; i < count; i++, p += sp, po += so)
                {
                    bool nonzero = false;
                    for (int o = 0; o < width; o++)
                        nonzero |= p[o] != 0;
                    if (checkCodePoints && unit == 4 && width >= 4 && Unsafe.ReadUnaligned<uint>(p) > 0x10FFFF)
                    {
                        // A first unit above U+10FFFF fails only when the string is that one unit: every later byte zero.
                        bool restNonzero = false;
                        for (int o = 4; o < width; o++)
                            restNonzero |= p[o] != 0;
                        invalid |= !restNonzero;
                    }
                    *po = nonzero ? (byte)1 : (byte)0;
                }
                if (invalid)
                    *(int*)auxdata = 1;
            };
    }
}
