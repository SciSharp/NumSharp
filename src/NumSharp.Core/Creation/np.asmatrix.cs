using System;
using System.Collections.Generic;
using System.Globalization;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Interpret the input as a matrix — a 2-D array. Unlike a copy, <c>asmatrix</c> does not copy
        ///     if the input is already an ndarray; it returns a 2-D view that shares memory.
        /// </summary>
        /// <param name="data">Input array.</param>
        /// <param name="dtype">Data-type of the output. <c>null</c> keeps the input dtype.</param>
        /// <returns>
        ///     <paramref name="data"/> interpreted as a 2-D array. A 0-D input becomes shape <c>(1, 1)</c>,
        ///     a 1-D input of length N becomes <c>(1, N)</c>, and a 2-D input is returned unchanged.
        /// </returns>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.asmatrix</c> ≙ <c>matrix(data, copy=False)</c>. NumSharp has no
        ///     dedicated <c>matrix</c> subclass (NumPy's is pending-deprecated), so the result is a plain
        ///     2-D <see cref="NDArray"/> — the special matrix operators (<c>*</c> as matmul, <c>**</c> as
        ///     matrix power, <c>.H</c>, <c>.I</c>) are NOT provided. The dimensional coercion matches NumPy's
        ///     <c>matrix.__array_finalize__</c> exactly, including the &gt;2-D behaviour: axes of length 1 are
        ///     dropped and the result must then be 2-D, otherwise a <see cref="ValueError"/>
        ///     ("shape too large to be a matrix.") is raised. The view is preserved for strided, transposed
        ///     and reversed inputs (no copy). A dtype change casts (which copies), then coerces the copy.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html
        /// </remarks>
        public static NDArray asmatrix(NDArray data, Type dtype = null)
        {
            if (data is null)
                throw new ArgumentNullException(nameof(data));

            var a = data;
            // matrix(data, copy=False): an ndarray of a different dtype is cast (a copy), matching NumPy's
            // `data.view(subtype).astype(intype)` before the 2-D finalize runs on the result.
            if (dtype != null && !Equals(a.dtype, dtype))
                a = a.astype(dtype, copy: true);

            return CoerceToMatrix(a);
        }

        /// <summary>Convenience overload taking <see cref="NPTypeCode"/>.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html</remarks>
        public static NDArray asmatrix(NDArray data, NPTypeCode dtype)
            => asmatrix(data, dtype == NPTypeCode.Empty ? null : dtype.AsType());

        /// <summary>Convenience overload taking a NumPy-style dtype string (e.g. <c>"float32"</c>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html</remarks>
        public static NDArray asmatrix(NDArray data, string dtype)
            => asmatrix(data, dtype == null ? null : np.dtype(dtype).type);

        /// <summary>
        ///     Interpret a matrix string as a 2-D array. Rows are separated by <c>';'</c> and columns by
        ///     commas and/or whitespace, e.g. <c>"1 2; 3 4"</c>. Surrounding brackets are ignored. The dtype
        ///     is inferred (integer when every element is an integer, otherwise double) unless
        ///     <paramref name="dtype"/> is given.
        /// </summary>
        /// <param name="data">Matrix string such as <c>"1 2; 3 4"</c>.</param>
        /// <param name="dtype">Data-type of the output. <c>null</c> infers it from the values.</param>
        /// <exception cref="ValueError">If the rows are not all the same length.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html</remarks>
        public static NDArray asmatrix(string data, Type dtype = null)
        {
            if (data is null)
                throw new ArgumentNullException(nameof(data));

            var mat = ParseMatrixString(data);
            return dtype == null ? mat : mat.astype(dtype, copy: false);
        }

        /// <summary>Convenience overload taking a matrix string and <see cref="NPTypeCode"/>.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html</remarks>
        public static NDArray asmatrix(string data, NPTypeCode dtype)
            => asmatrix(data, dtype == NPTypeCode.Empty ? null : dtype.AsType());

        /// <summary>
        ///     Coerces <paramref name="a"/> to a 2-D view following NumPy's <c>matrix.__array_finalize__</c>:
        ///     2-D is unchanged; &gt;2-D drops length-1 axes and must land on exactly 2-D; a squeezed/native
        ///     1-D of length N becomes (1, N); 0-D becomes (1, 1). All reshapes are stride-preserving views.
        /// </summary>
        private static NDArray CoerceToMatrix(NDArray a)
        {
            var s = a.Shape;
            int nd = s.NDim;

            if (nd == 2)
                return a; // already a matrix — shared-memory view returned as-is

            long[] dims = s.dimensions;
            long[] strides = s.strides;

            if (nd > 2)
            {
                // Keep only axes with length > 1 (NumPy: tuple(x for x in shape if x > 1)).
                int keep = 0;
                for (int i = 0; i < nd; i++)
                    if (dims[i] > 1) keep++;

                var kd = new long[keep];
                var ks = new long[keep];
                for (int i = 0, j = 0; i < nd; i++)
                    if (dims[i] > 1) { kd[j] = dims[i]; ks[j] = strides[i]; j++; }

                if (keep > 2)
                    throw new ValueError("shape too large to be a matrix.");

                // Dropping a length-0 axis would change the element count — NumPy surfaces this as a
                // reshape failure. Guard it so the degenerate empty >2-D case raises rather than aliasing
                // past the buffer.
                long keptSize = 1;
                for (int i = 0; i < keep; i++) keptSize *= kd[i];
                if (keptSize != s.size)
                    throw new ValueError($"cannot reshape array of size {s.size} into matrix shape");

                if (keep == 2)
                    return Alias(a, new Shape(kd, ks, s.offset, s.bufferSize));

                // keep is 0 or 1 — fall through to the low-rank rules using the squeezed view.
                dims = kd;
                strides = ks;
                nd = keep;
            }

            if (nd == 1)
            {
                // (N,) -> (1, N): prepend a length-1 axis, preserving the element stride (a view).
                var oneD = new Shape((long[])dims.Clone(), (long[])strides.Clone(), s.offset, s.bufferSize);
                return Alias(a, oneD.ExpandDimension(0));
            }

            // nd == 0 (scalar) -> (1, 1).
            var scalarShape = new Shape(Array.Empty<long>(), Array.Empty<long>(), s.offset, s.bufferSize);
            return Alias(a, scalarShape.ExpandDimension(0).ExpandDimension(0));
        }

        private static NDArray Alias(NDArray a, Shape shape)
            => new NDArray(a.Storage.Alias(shape)) { TensorEngine = a.TensorEngine };

        /// <summary>
        ///     Parses a NumPy matrix string ("1 2; 3 4") into a 2-D NDArray. Mirrors NumPy's
        ///     <c>_convert_from_string</c>: strip brackets, split rows on ';', split each row on commas and
        ///     whitespace, then integer-typed unless any element needs a floating-point representation.
        /// </summary>
        private static NDArray ParseMatrixString(string data)
        {
            string cleaned = data.Replace("[", "").Replace("]", "");
            string[] rows = cleaned.Split(';');
            var separators = new[] { ',', ' ', '\t', '\n', '\r' };

            var tokenRows = new List<string[]>(rows.Length);
            int ncols = -1;
            bool anyFloat = false;

            foreach (var row in rows)
            {
                var toks = row.Split(separators, StringSplitOptions.RemoveEmptyEntries);
                if (ncols == -1)
                    ncols = toks.Length;
                else if (toks.Length != ncols)
                    throw new ValueError("Rows not the same size.");

                foreach (var t in toks)
                    if (!IsIntegerToken(t)) { anyFloat = true; }

                tokenRows.Add(toks);
            }

            int nrows = tokenRows.Count;

            // An element-free matrix (e.g. "" or "   ") has no tokens to infer an integer dtype from;
            // NumPy's np.array([[]]) defaults such an empty array to float64, so match that.
            if (anyFloat || (long)nrows * ncols == 0)
            {
                var buf = new double[nrows, ncols];
                for (int r = 0; r < nrows; r++)
                    for (int c = 0; c < ncols; c++)
                        buf[r, c] = double.Parse(tokenRows[r][c], CultureInfo.InvariantCulture);
                return np.array(buf);
            }
            else
            {
                var buf = new long[nrows, ncols];
                for (int r = 0; r < nrows; r++)
                    for (int c = 0; c < ncols; c++)
                        buf[r, c] = long.Parse(tokenRows[r][c], CultureInfo.InvariantCulture);
                return np.array(buf);
            }
        }

        /// <summary>True when <paramref name="token"/> parses as an integer literal (no decimal point / exponent).</summary>
        private static bool IsIntegerToken(string token)
            => long.TryParse(token, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out _);
    }
}
