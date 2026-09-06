using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

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
        ///     <see cref="result_type(NDArray[])"/> promotion (per row, then across rows). Errors reproduce
        ///     NumPy's <see cref="ValueError"/> verbatim (see <see cref="bmat(ITuple)"/> remarks). See
        ///     <see cref="block(object)"/> for the N-D generalization.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.bmat.html
        /// </remarks>
        // Scope: bmat builds intermediate row concatenations / a parsed-block copy and then wraps the
        // final concatenation in an asmatrix VIEW — so every intermediate (and, for bmat(NDArray),
        // the obj.copy() the view aliases) is orphaned once the view owns the buffer (measured: one
        // bucketed buffer escaped per call). [NDScoped] reclaims the intermediates while yielding the
        // asmatrix view; ARC keeps the view's shared buffer alive through it (rule R1 / invariant I3).
        [NDScoped]
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
                rows[i] = BmatConcat(obj[i], -1);
            }

            return asmatrix(BmatConcat(rows, 0));
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
        [NDScoped]
        public static NDArray bmat(NDArray[] obj)
        {
            if (obj is null)
                throw new ArgumentNullException(nameof(obj));

            return asmatrix(BmatConcat(obj, -1));
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
        [NDScoped]
        public static NDArray bmat(NDArray obj)
        {
            if (obj is null)
                throw new ArgumentNullException(nameof(obj));

            // NumPy's bmat(ndarray) is matrix(obj) with copy=True, so the result must not alias obj.
            return asmatrix(obj.copy());
        }

        /// <summary>
        ///     Build a 2-D array from C# tuples of blocks — <c>(A, B)</c> is a single side-by-side row and
        ///     <c>((A, B), (C, D))</c> is a nested block matrix — the tuple spelling of NumPy's
        ///     <c>np.bmat((A, B))</c> / <c>np.bmat(((A, B), (C, D)))</c>.
        /// </summary>
        /// <param name="obj">
        ///     A <see cref="ITuple"/> (any <c>ValueTuple</c>/<c>Tuple</c>): if its first entry is a bare
        ///     block (<see cref="NDArray"/>) the whole tuple is one row joined along the last axis;
        ///     otherwise each entry is a row (itself a tuple / <c>NDArray[]</c> / sequence of blocks),
        ///     concatenated per row and stacked along axis 0.
        /// </param>
        /// <returns>The assembled 2-D array.</returns>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.bmat</c>'s <c>isinstance(obj, (tuple, list))</c> branch. Mixed
        ///     forms work too — <c>([A, B], [C, D])</c> (a tuple of lists) — since a row may be a tuple, an
        ///     <c>NDArray[]</c>, or any <see cref="IEnumerable"/> of blocks; a non-<see cref="NDArray"/>
        ///     entry is passed through <see cref="asanyarray(in object, Type)"/> exactly as NumPy runs
        ///     <c>asarray</c> over the entries. Still pure concatenation (no matrix products).
        ///     <para>
        ///     <b>Error parity.</b> The errors reproduce <c>numpy.concatenate</c>'s contract as
        ///     <see cref="ValueError"/> with NumPy's verbatim text (see <see cref="BmatConcat"/>): an empty
        ///     tuple / row → "need at least one array to concatenate"; a leading 0-D block (a scalar, or a
        ///     <c>null</c> — NumPy's <c>None</c> becomes a 0-D array) → "zero-dimensional arrays cannot be
        ///     concatenated"; a later block whose rank differs from the first → "all the input arrays must
        ///     have same number of dimensions, but the array at index 0 has {n} dimension(s) and the array
        ///     at index {k} has {m} dimension(s)". A width/height mismatch between well-ranked blocks keeps
        ///     <c>concatenate</c>'s <see cref="IncorrectShapeException"/> — same verbatim NumPy text, the
        ///     library-wide house exception type for shape/alignment errors.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.bmat.html
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray bmat(ITuple obj)
        {
            if (obj is null)
                throw new ArgumentNullException(nameof(obj));

            // Port of numpy.bmat's tuple/list loop (defmatrix.py):
            //   for row in obj:
            //       if isinstance(row, ndarray): return matrix(concatenate(obj, axis=-1))   # flat row
            //       else: arr_rows.append(concatenate(row, axis=-1))
            //   return matrix(concatenate(arr_rows, axis=0))
            var arrRows = new List<NDArray>(obj.Length);
            for (int i = 0; i < obj.Length; i++)
            {
                object row = obj[i];
                if (row is NDArray)
                    return asmatrix(BmatConcat(BmatTupleToBlocks(obj), -1));
                arrRows.Add(BmatConcat(BmatRowToBlocks(row), -1));
            }

            return asmatrix(BmatConcat(arrRows.ToArray(), 0));
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
        [NDScoped]
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
        ///     Concatenates one row's (or the stacked rows') blocks, reproducing <c>numpy.concatenate</c>'s
        ///     error contract as <see cref="ValueError"/> with NumPy's verbatim message text so bmat's
        ///     errors match NumPy exactly. NumPy takes the rank from <c>arrays[0]</c>, so a leading 0-D
        ///     block reports "zero-dimensional…" while a *later* rank mismatch reports "same number of
        ///     dimensions…"; a <c>null</c> block is treated as a 0-D array (NumPy's <c>asarray(None)</c>).
        ///     Well-ranked blocks with mismatched extents fall through to
        ///     <see cref="concatenate(NDArray[], int?, NDArray, NPTypeCode?, string)"/>, whose
        ///     <see cref="IncorrectShapeException"/> carries the same verbatim NumPy text.
        /// </summary>
        private static NDArray BmatConcat(NDArray[] blocks, int axis)
        {
            if (blocks.Length == 0)
                throw new ValueError("need at least one array to concatenate");

            // NumPy: ndim = PyArray_NDIM(arrays[0]); if (ndim == 0) -> "zero-dimensional…".
            // A null block == a 0-D array (NumPy's asarray(None)).
            int ndim0 = blocks[0]?.ndim ?? 0;
            if (ndim0 == 0)
                throw new ValueError("zero-dimensional arrays cannot be concatenated");

            for (int k = 1; k < blocks.Length; k++)
            {
                int ndimk = blocks[k]?.ndim ?? 0;
                if (ndimk != ndim0)
                    throw new ValueError(
                        "all the input arrays must have same number of dimensions, but the array at " +
                        $"index 0 has {ndim0} dimension(s) and the array at index {k} has {ndimk} dimension(s)");
            }

            // All blocks share ndim ≥ 1 (so none is null here); concatenate does the extent check + copy.
            return concatenate(blocks, axis);
        }

        /// <summary>Materialises every entry of a flat tuple <c>(A, B, …)</c> to a block.</summary>
        private static NDArray[] BmatTupleToBlocks(ITuple t)
        {
            var blocks = new NDArray[t.Length];
            for (int i = 0; i < t.Length; i++)
                blocks[i] = BmatElementToBlock(t[i]);
            return blocks;
        }

        /// <summary>
        ///     Materialises a row (a tuple / <c>NDArray[]</c> / sequence of blocks) into its block array.
        /// </summary>
        private static NDArray[] BmatRowToBlocks(object row)
        {
            switch (row)
            {
                case NDArray[] a:
                    return a;
                case ITuple t:
                    return BmatTupleToBlocks(t);
                case string:
                    throw new ArgumentException("bmat: a row must be a sequence of blocks, not a string.");
                case IEnumerable ie:
                {
                    var list = new List<NDArray>();
                    foreach (var e in ie)
                        list.Add(BmatElementToBlock(e));
                    return list.ToArray();
                }
                default:
                    throw new ArgumentException(
                        $"bmat: each row must be a sequence of blocks, got {row?.GetType().Name ?? "null"}.");
            }
        }

        /// <summary>
        ///     Converts a tuple entry to a block: an <see cref="NDArray"/> passes through, <c>null</c> stays
        ///     <c>null</c> (a 0-D array to <see cref="BmatConcat"/>, matching NumPy's <c>None</c>), and any
        ///     other value goes through <see cref="asanyarray(in object, Type)"/> — as NumPy runs
        ///     <c>asarray</c> over the entries (a scalar so produced is 0-D and rejected identically).
        /// </summary>
        private static NDArray BmatElementToBlock(object e)
            => e switch
            {
                null => null,
                NDArray nd => nd,
                _ => asanyarray(e)
            };

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

                rowArrays[r] = BmatConcat(cols, -1);
            }

            return BmatConcat(rowArrays, 0);
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
