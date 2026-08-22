using System;
using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Build a 2-D array from a nested sequence of blocks: <c>[[A, B], [C, D]]</c> assembles one
        ///     matrix by joining the blocks in each inner list left-to-right and stacking the resulting
        ///     rows top-to-bottom.
        /// </summary>
        /// <param name="obj">
        ///     Rows of blocks. Each inner array holds the blocks of one row; they are concatenated along
        ///     the last axis (horizontally). The rows are then concatenated along axis 0 (vertically).
        /// </param>
        /// <returns>The assembled 2-D array.</returns>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.bmat</c> ≙ <c>matrix(concatenate([concatenate(row, axis=-1) for
        ///     row in obj], axis=0))</c>. It is <b>pure block assembly (concatenation) — it performs NO
        ///     matrix multiplication</b>, so no BLAS/OpenBLAS path is involved; the numerics come entirely
        ///     from <see cref="concatenate(NDArray[], int?, NDArray, NPTypeCode?, string)"/> and the 2-D
        ///     coercion from <see cref="asmatrix(NDArray, Type)"/>. NumSharp has no dedicated <c>matrix</c>
        ///     subclass (NumPy's is pending-deprecated), so the result is a plain 2-D <see cref="NDArray"/>
        ///     — the special matrix operators (<c>*</c> as matmul, <c>**</c> as matrix power, <c>.H</c>,
        ///     <c>.I</c>) are NOT provided. The result dtype follows NumPy's two-stage
        ///     <see cref="result_type(NDArray[])"/> promotion (per row, then across rows). Block shapes are
        ///     validated by <c>concatenate</c>, so a ragged width or a per-row height mismatch raises the
        ///     usual concatenation shape error. See <see cref="block(object)"/> for the N-D generalization.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.bmat.html
        /// </remarks>
        public static NDArray bmat(NDArray[][] obj)
        {
            if (obj is null)
                throw new ArgumentNullException(nameof(obj));

            // Mirror NumPy exactly: each row is concatenated along the last axis, then the rows are
            // concatenated along axis 0. Doing it in two stages (rather than one flat concatenate)
            // reproduces NumPy's per-row-then-across-rows promotion and shape checks bit-for-bit.
            var rows = new NDArray[obj.Length];
            for (int i = 0; i < obj.Length; i++)
            {
                if (obj[i] is null)
                    throw new ArgumentNullException($"{nameof(obj)}[{i}]");
                rows[i] = concatenate(obj[i], -1);
            }

            return asmatrix(concatenate(rows, 0));
        }

        /// <summary>
        ///     Build a 2-D array from a flat sequence of blocks placed side by side: <c>[A, B]</c> joins the
        ///     blocks along the last axis, then coerces the result to 2-D.
        /// </summary>
        /// <param name="obj">The blocks, concatenated left-to-right (along the last axis).</param>
        /// <returns>The assembled 2-D array.</returns>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.bmat([A, B, …])</c> ≙ <c>matrix(concatenate(obj, axis=-1))</c> —
        ///     the flat-list branch, equivalent to a single-row <c>[[A, B, …]]</c>. A flat list of 1-D
        ///     blocks concatenates to 1-D and is then coerced to a <c>(1, N)</c> row (NumPy's
        ///     <c>matrix()</c> finalize). No matrix products; see <see cref="bmat(NDArray[][])"/>.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.bmat.html
        /// </remarks>
        public static NDArray bmat(NDArray[] obj)
        {
            if (obj is null)
                throw new ArgumentNullException(nameof(obj));

            return asmatrix(concatenate(obj, -1));
        }

        /// <summary>
        ///     Interpret a single array as a matrix — a 2-D copy. A 0-D input becomes <c>(1, 1)</c>, a 1-D
        ///     input of length N becomes <c>(1, N)</c>, and a 2-D input is copied unchanged.
        /// </summary>
        /// <param name="obj">Input array.</param>
        /// <returns>A fresh 2-D array holding <paramref name="obj"/>'s values.</returns>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.bmat(ndarray)</c> ≙ <c>matrix(obj)</c>, which <b>copies</b>
        ///     (<c>matrix</c>'s default <c>copy=True</c>) — unlike <see cref="asmatrix(NDArray, Type)"/>,
        ///     which returns a view. The copy is taken first, then coerced to 2-D, so the result never
        ///     aliases <paramref name="obj"/>. https://numpy.org/doc/stable/reference/generated/numpy.bmat.html
        /// </remarks>
        public static NDArray bmat(NDArray obj)
        {
            if (obj is null)
                throw new ArgumentNullException(nameof(obj));

            // NumPy's bmat(ndarray) is matrix(obj) with copy=True, so the result must not alias obj.
            return asmatrix(obj.copy());
        }

        /// <summary>
        ///     Build a 2-D array from a matrix string whose tokens name the blocks: rows are separated by
        ///     <c>';'</c> and the blocks within a row by commas and/or whitespace, e.g.
        ///     <c>"A, B; C, D"</c>. Each token is resolved to an <see cref="NDArray"/> through the supplied
        ///     dictionaries (trying <paramref name="ldict"/> first, then <paramref name="gdict"/>); the
        ///     blocks of each row are joined along the last axis and the rows are stacked along axis 0.
        /// </summary>
        /// <param name="obj">Matrix string such as <c>"A, B; C, D"</c>.</param>
        /// <param name="ldict">
        ///     Local name → array map, consulted first (NumPy's <c>ldict</c>).
        /// </param>
        /// <param name="gdict">
        ///     Global name → array map, consulted when a name is absent from <paramref name="ldict"/>
        ///     (NumPy's <c>gdict</c>). Optional.
        /// </param>
        /// <returns>The assembled 2-D array.</returns>
        /// <exception cref="NameError">A token is not present in either dictionary.</exception>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.bmat(str, ldict, gdict)</c> → <c>matrix(_from_string(...))</c>.
        ///     Still pure concatenation (no matrix products).
        ///     <para>
        ///     <b>Divergence — the name dictionary is required.</b> NumPy resolves a bare token against the
        ///     CALLER'S Python frame (<c>sys._getframe().f_back</c>) when <c>gdict</c> is <c>None</c>; C#
        ///     has no equivalent, so at least one of <paramref name="ldict"/> / <paramref name="gdict"/>
        ///     must be supplied (this is the same reason <see cref="r_"/>/<see cref="c_"/> omit the
        ///     <c>bmat</c> branch). Resolution tries <paramref name="ldict"/> then
        ///     <paramref name="gdict"/>; a token absent from both — including a numeric literal such as
        ///     <c>"1"</c>, which NumPy also treats as a name — raises <see cref="NameError"/>
        ///     ("name '{token}' is not defined"), matching NumPy verbatim. Unlike
        ///     <see cref="asmatrix(string, Type)"/>'s literal parser, brackets are NOT stripped (they
        ///     become part of a token and fail to resolve), exactly as NumPy's <c>_from_string</c> leaves
        ///     them. https://numpy.org/doc/stable/reference/generated/numpy.bmat.html
        ///     </para>
        /// </remarks>
        public static NDArray bmat(string obj, IDictionary<string, NDArray> ldict,
            IDictionary<string, NDArray> gdict = null)
        {
            if (obj is null)
                throw new ArgumentNullException(nameof(obj));
            if (ldict is null && gdict is null)
                throw new ArgumentException(
                    "bmat's string form needs a name→array dictionary in C#: NumPy resolves the tokens " +
                    "against the caller's Python frame, which has no C# equivalent. Pass ldict (and/or gdict).",
                    nameof(ldict));

            return asmatrix(BmatFromString(obj, ldict, gdict));
        }

        /// <summary>
        ///     Port of NumPy's <c>defmatrix._from_string</c>: split on <c>';'</c> for rows, split each row
        ///     on commas and whitespace for the block names, resolve each name, then
        ///     <c>concatenate(row_blocks, axis=-1)</c> per row and <c>concatenate(rows, axis=0)</c>.
        /// </summary>
        private static NDArray BmatFromString(string data, IDictionary<string, NDArray> ldict,
            IDictionary<string, NDArray> gdict)
        {
            // NumPy: rows = str.split(';'); within a row, split on ',' then on whitespace — i.e. tokens
            // are the comma/whitespace-separated words. RemoveEmptyEntries reproduces the two-stage split
            // (an empty row yields zero tokens, so its concatenate raises, as in NumPy).
            var separators = new[] { ',', ' ', '\t', '\n', '\r' };
            string[] rowStrings = data.Split(';');

            var rowArrays = new NDArray[rowStrings.Length];
            for (int r = 0; r < rowStrings.Length; r++)
            {
                string[] tokens = rowStrings[r].Split(separators, StringSplitOptions.RemoveEmptyEntries);
                var cols = new NDArray[tokens.Length];
                for (int c = 0; c < tokens.Length; c++)
                    cols[c] = BmatResolveName(tokens[c], ldict, gdict);

                rowArrays[r] = concatenate(cols, -1);
            }

            return concatenate(rowArrays, 0);
        }

        /// <summary>
        ///     Resolves one block name, trying <paramref name="ldict"/> first then <paramref name="gdict"/>
        ///     (NumPy's order), raising <see cref="NameError"/> when neither defines it.
        /// </summary>
        private static NDArray BmatResolveName(string name, IDictionary<string, NDArray> ldict,
            IDictionary<string, NDArray> gdict)
        {
            if (ldict is not null && ldict.TryGetValue(name, out NDArray v))
                return v;
            if (gdict is not null && gdict.TryGetValue(name, out v))
                return v;

            throw new NameError($"name '{name}' is not defined");
        }
    }
}
