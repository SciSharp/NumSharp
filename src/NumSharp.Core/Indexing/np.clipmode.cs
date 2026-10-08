
namespace NumSharp
{
    public static partial class np
    {
        /// <summary>The <c>raise</c> clip mode (NumPy's <c>NPY_RAISE</c>) in the numbering the take / put / choose /
        ///     ravel_multi_index kernels share: an out-of-range index is an error.</summary>
        internal const int ClipModeRaise = 0;

        /// <summary>The <c>wrap</c> clip mode (NumPy's <c>NPY_WRAP</c>): an index is reduced modulo the axis length.</summary>
        internal const int ClipModeWrap = 1;

        /// <summary>The <c>clip</c> clip mode (NumPy's <c>NPY_CLIP</c>): an index is clamped to the axis, so a NEGATIVE
        ///     index becomes 0 rather than counting from the end.</summary>
        internal const int ClipModeClip = 2;

        /// <summary>
        ///     NumPy's <c>PyArray_ClipmodeConverter</c> for a <c>mode</c> string (<c>conversion_utils.c</c>: its
        ///     <c>string_converter_helper</c> around <c>clipmode_parser</c>), the one parser behind the <c>mode</c> of
        ///     <c>np.take</c>, <c>np.put</c>, <c>np.choose</c> and <c>np.ravel_multi_index</c>.
        /// </summary>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c> or <c>"clip"</c>, spelled exactly; <c>null</c> is NumPy's
        ///     <c>None</c>, which means <c>raise</c>.</param>
        /// <returns><see cref="ClipModeRaise"/>, <see cref="ClipModeWrap"/> or <see cref="ClipModeClip"/>.</returns>
        /// <exception cref="ValueError">A string whose first character selects a mode but which is not that mode's exact
        ///     lowercase spelling (<c>"Raise"</c>, <c>"clipped"</c>): <c>Use one of 'clip', 'raise', or 'wrap' for clip
        ///     mode</c>. Any other string, the empty one included: <c>clipmode must be one of 'clip', 'raise', or 'wrap'
        ///     (got 'bogus')</c>, the argument in Python's <c>repr</c>.</exception>
        /// <remarks>
        ///     The callers run it where NumPy's argument parsing runs it, so a bad mode is reported before anything else is
        ///     checked: before a read-only <c>np.put</c> target, an out-of-range <c>np.take</c> axis or an empty source.
        ///     NumPy's integer modes (<c>np.RAISE</c> and its siblings) have no C# spelling here; the C# signatures take a
        ///     string. An empty string used to mean <c>raise</c> in NumSharp's own parser; NumPy refuses it.
        /// </remarks>
        internal static int ClipModeConverter(string mode)
        {
            if (mode is null)
                return ClipModeRaise;

            // clipmode_parser selects the mode by the FIRST character alone (either case), then demands the exact
            // lowercase spelling; a selected-but-misspelled mode raises its own text, which string_converter_helper
            // propagates instead of formatting the generic one.
            if (mode.Length >= 1)
            {
                switch (mode[0])
                {
                    case 'C':
                    case 'c':
                        return mode == "clip" ? ClipModeClip : throw new ValueError(ClipModeMisspelled);
                    case 'W':
                    case 'w':
                        return mode == "wrap" ? ClipModeWrap : throw new ValueError(ClipModeMisspelled);
                    case 'R':
                    case 'r':
                        return mode == "raise" ? ClipModeRaise : throw new ValueError(ClipModeMisspelled);
                }
            }

            // The parser returned -1 without an error (empty string, or an unknown first character): the helper's
            // "%s %s (got %R)" with the ORIGINAL argument's repr. (Fully qualified: inside np, PyRepr names np's own
            // einsum_path helper method.)
            string repr = global::NumSharp.Backends.Printing.PyRepr.Str(mode);
            throw new ValueError($"clipmode must be one of 'clip', 'raise', or 'wrap' (got {repr})");
        }

        /// <summary>
        ///     NumPy's <c>PyArray_ConvertClipmodeSequence</c>: one clip mode per dimension, from a sequence of mode strings
        ///     (<c>np.ravel_multi_index(..., mode=('raise', 'wrap'))</c>).
        /// </summary>
        /// <param name="modes">One mode per dimension, each parsed by <see cref="ClipModeConverter"/>; <c>null</c> is
        ///     NumPy's <c>None</c> (every dimension raises).</param>
        /// <param name="n">The number of dimensions the modes must cover.</param>
        /// <returns>The <paramref name="n"/> parsed modes.</returns>
        /// <exception cref="ValueError">A sequence of another length (<c>list of clipmodes has wrong length (1 instead of
        ///     2)</c>), or a mode <see cref="ClipModeConverter"/> refuses (the first bad one, in order).</exception>
        internal static int[] ConvertClipmodeSequence(string[] modes, int n)
        {
            var result = new int[n];
            if (modes is null)
                return result;   // ClipModeRaise is 0: every dimension raises.

            if (modes.Length != n)
                throw new ValueError($"list of clipmodes has wrong length ({modes.Length} instead of {n})");
            for (int i = 0; i < n; i++)
                result[i] = ClipModeConverter(modes[i]);
            return result;
        }

        /// <summary><c>clipmode_parser</c>'s text for a mode whose first character selects a mode it does not spell.</summary>
        private const string ClipModeMisspelled = "Use one of 'clip', 'raise', or 'wrap' for clip mode";
    }
}
