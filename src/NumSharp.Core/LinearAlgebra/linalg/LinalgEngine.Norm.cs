using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.norm(x, ord, int[] axis, keepdims)</c> past the class boundary: NumPy's
        ///     <c>x.astype(float)</c> of a text operand, the float conversion of an integer / bool one, the fast path of
        ///     the default order over the whole array, then the one-axis (vector) or two-axis (matrix) branch.
        /// </summary>
        /// <param name="x">The operand (plain: the facade applied the class boundary).</param>
        /// <param name="ord">The order: null, a number, an infinity, or <c>"fro"</c> / <c>"f"</c> / <c>"nuc"</c>.</param>
        /// <param name="axis">The reduced axes, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <returns>The norm: a 0-d array for a whole-array norm, else the reduced array.</returns>
        /// <exception cref="ValueError">An order that is invalid for the number of reduced axes, duplicate axes, more
        ///     than two reduced axes (<c>Improper number of dimensions to norm.</c>), a min order of an empty operand, or a
        ///     text element that does not parse as a float.</exception>
        /// <exception cref="AxisError">An axis out of range.</exception>
        /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
        /// <exception cref="OpenBlasMissingBackendException">A matrix order 2, -2 or <c>"nuc"</c> over a non-empty
        ///     matrix with no matrix backend installed.</exception>
        [NDScoped]
        internal static NDArray Norm(NDArray x, object ord, int[] axis, bool keepdims)
        {
            // NumPy's norm converts a non-inexact input with x.astype(float): a text array is parsed (each element by
            // float(), the first failing one raising NumPy's text), and the norm runs on the float64 numbers.
            using var parsed = Backends.Strings.NDStringUfuncs.IsText(x) ? x.astype(np.float64, true, 'K', "unsafe") : null;
            if (parsed is not null)
                x = parsed;
            var a = ToFloating(x);
            int nd = a.ndim;

            // NumPy's fast path: with no axis and the default order, the whole array is one
            // vector — and it reaches it through `x.dot(x)`, so this mirrors that call rather
            // than composing a reduction with a different summation order.
            if (axis is null &&
                (ord is null
                 || (IsText(ord, out var word) && (word == "f" || word == "fro") && nd == 2)
                 || (IsNumber(ord, out double two) && two == 2d && nd == 1)))
            {
                var flat = np.ravel(a);
                var squared = a.typecode == NPTypeCode.Complex
                    ? np.sum(AbsSquared(flat))
                    : np.dot(flat, flat);
                var whole = np.sqrt(squared);
                return keepdims ? np.reshape(whole, Ones(nd)) : whole;
            }

            int[] axes = axis ?? AllAxes(nd);

            // NumPy's one-axis branch hands the axis to its reductions as it is (they check it), so a string order is
            // refused before any axis check: VectorNorm normalizes where those reductions would.
            if (axes.Length == 1)
                return VectorNorm(a, ord, axes[0], keepdims);
            if (axes.Length == 2)
                return MatrixNorm(a, ord, axes, nd, keepdims);

            throw new ValueError("Improper number of dimensions to norm.");
        }

        /// <summary>
        ///     The body of <c>np.linalg.vector_norm(x, int[] axis, keepdims, ord)</c> past the class boundary: NumPy's
        ///     tuple branch (the reduced axes moved to the front and merged into one, then norm's one-axis branch over
        ///     axis 0, through the public <c>np.linalg.norm</c>) or, without axes, the norm of the flattened operand.
        /// </summary>
        /// <param name="x">The operand (plain: the facade applied the class boundary).</param>
        /// <param name="axis">The reduced axes, any number of them, or null for the flattened array.</param>
        /// <param name="keepdims">Leave the reduced axes in the result with length 1.</param>
        /// <param name="ord">The vector order; null means NumPy's default of 2.</param>
        /// <returns>The vector norm over the axes.</returns>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ValueError">An axis appears twice ("repeated axis"), a text order, or -inf of an empty
        ///     operand.</exception>
        /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
        [NDScoped]
        internal static NDArray VectorNormOverAxes(NDArray x, int[] axis, bool keepdims, object ord)
        {
            object order = ord ?? 2;
            if (axis is not null)
            {
                int nd = x.ndim;
                var normalized = Backends.DefaultEngine.normalize_axis_tuple(axis, nd);

                // rest = the axes not reduced, in order; newshape = axis + rest (the raw axes: transpose accepts them).
                var isReduced = new bool[nd];
                foreach (int ax in normalized)
                    isReduced[ax] = true;
                var permutation = new int[nd];
                var mergedShape = new long[1 + nd - normalized.Length];
                long mergedLength = 1;
                for (int i = 0; i < axis.Length; i++)
                {
                    permutation[i] = axis[i];
                    mergedLength *= x.Shape.dimensions[normalized[i]];
                }
                mergedShape[0] = mergedLength;
                for (int d = 0, k = axis.Length, r = 1; d < nd; d++)
                {
                    if (isReduced[d])
                        continue;
                    permutation[k++] = d;
                    mergedShape[r++] = x.Shape.dimensions[d];
                }

                var merged = np.transpose(x, permutation).reshape(Shape.Adopt(mergedShape));
                var res = np.linalg.norm(merged, order, new[] {0}, false);
                if (!keepdims)
                    return res;

                // keepdims: the input's shape with every reduced axis set to 1 (NumPy reshapes: after the merge it
                // cannot reuse norm's own keepdims).
                var kept = (long[])x.Shape.dimensions.Clone();
                foreach (int ax in normalized)
                    kept[ax] = 1;
                return np.reshape(res, Shape.Adopt(kept));
            }

            // Flatten first, then reduce — and restore the rank afterwards when keepdims asks,
            // because the ravel has already erased it.
            var flattened = np.linalg.norm(np.ravel(ToFloating(x)), order, new[] {0}, false);
            return keepdims ? np.reshape(flattened, Ones(x.ndim)) : flattened;
        }

        /// <summary>
        ///     NumPy's one-axis branch of <c>norm</c>: the vector order <paramref name="ord"/> over <paramref name="axis"/>.
        ///     The inf order reduces with <c>initial=0</c>, so an empty vector's inf-norm is 0.0 where -inf still raises.
        /// </summary>
        /// <param name="a">The floating operand.</param>
        /// <param name="ord">The order: null / 2, a number, or +-inf (text orders are refused).</param>
        /// <param name="axis">The axis as given (negative values count from the end): NumPy checks it in the reduction a
        ///     numeric order runs, so a text order is refused whatever the axis.</param>
        /// <param name="keepdims">Keep the axis as length 1.</param>
        /// <returns>The reduced norm.</returns>
        /// <exception cref="ValueError">A text order (<c>Invalid norm order '...' for vectors</c>), or -inf of an empty
        ///     axis (NumPy's empty-min error).</exception>
        /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
        /// <exception cref="AxisError">A numeric order with an axis out of range (NumPy's reductions' error).</exception>
        [NDScopedCovered] // always dispatched from the [NDScoped] Norm body; rides its scope
        private static NDArray VectorNorm(NDArray a, object ord, int axis, bool keepdims)
        {
            if (ord is null || (IsNumber(ord, out double n) && n == 2d))
            {
                // NumPy's add.reduce checks the axis here.
                axis = NormalizeAxis(axis, a.ndim);
                // NumPy's special case: sqrt(add.reduce((x.conj() * x).real, axis)). For a real float32 / float64
                // operand the product is x * x, and np.evaluate streams Sum(x * x, axis) in add.reduce's own schedule:
                // the same bits as the eager product + np.sum (each product rounded, then summed in NumPy's order),
                // with no N-element temporary (a (10000, 10) slab: 45 us against 152 us for the eager pair). x * x
                // also keeps a negative NaN's sign as NumPy's x.conj() * x does, where |x| * |x| cleared it. float16
                // and complex keep the eager form (np.evaluate's float16 sum is not NumPy's HALF schedule, and a
                // complex |x|^2 is the real sum of squares), as does a host without dynamic code (no fused kernels).
                if (Backends.Kernels.DirectILKernelGenerator.Enabled && a.typecode is NPTypeCode.Double or NPTypeCode.Single)
                    return np.sqrt(np.evaluate(NDExpr.Sum((NDExpr)a * a, axis, keepdims)));
                return np.sqrt(np.sum(AbsSquared(a), axis, keepdims));
            }

            if (IsText(ord, out var word))
                throw new ValueError($"Invalid norm order '{word}' for vectors");

            double order = AsNumber(ord);
            // Every remaining branch is a reduction over the axis, which checks it: after the order's refusal above.
            axis = NormalizeAxis(axis, a.ndim);

            // NumPy 2.x reduces with initial=0 (`abs(x).max(axis=axis, keepdims=keepdims, initial=0)`): an EMPTY
            // vector's inf-norm is 0.0, not the empty-max ValueError. Every |x| is >= 0, so a non-empty answer is
            // unchanged (a NaN still propagates).
            if (double.IsPositiveInfinity(order))
                return np.amax(np.abs(a), axis, keepdims: keepdims, initial: 0);
            if (double.IsNegativeInfinity(order))
                return np.amin(np.abs(a), axis, keepdims: keepdims);
            if (order == 0d)
            {
                // NumPy counts non-zeros in `x.real.dtype`, NOT `x.dtype` — so a COMPLEX operand's
                // zero-"norm" is float64, not complex. For every real dtype the real component's
                // dtype is the dtype itself, so only Complex needs the substitution.
                var countType = a.typecode == NPTypeCode.Complex ? NPTypeCode.Double : a.typecode;
                return np.sum(np.not_equal(a, NDArray.Scalar(0)).astype(countType), axis, keepdims);
            }
            if (order == 1d)
                return np.sum(np.abs(a), axis, keepdims);

            var raised = np.power(np.abs(a), NDArray.Scalar(order));
            var reduced = np.sum(raised, axis, keepdims);
            return np.power(reduced, NDArray.Scalar(1d / order));
        }

        /// <summary>
        ///     NumPy's two-axis branch of <c>norm</c>: the matrix order <paramref name="ord"/> over the axis pair. The 1
        ///     and inf orders reduce with <c>initial=0</c> (an empty matrix gives 0.0); -1 and -inf keep the empty-min
        ///     error; 2, -2 and "nuc" go through the singular values.
        /// </summary>
        /// <param name="a">The floating operand.</param>
        /// <param name="ord">The order: null / "fro" / "f", "nuc", or a number among 1, -1, 2, -2, +-inf.</param>
        /// <param name="axes">The (row, column) axes as given.</param>
        /// <param name="nd">The operand's rank.</param>
        /// <param name="keepdims">Put both reduced axes back as length 1.</param>
        /// <returns>The reduced norm.</returns>
        /// <exception cref="ValueError">Duplicate axes, an invalid matrix order, or a min order of an empty
        ///     matrix.</exception>
        /// <exception cref="AxisError">An axis out of range.</exception>
        /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
        /// <exception cref="OpenBlasMissingBackendException">An order 2, -2 or "nuc" over a non-empty matrix with no matrix backend.</exception>
        [NDScopedCovered] // always dispatched from the [NDScoped] Norm body; rides its scope
        private static NDArray MatrixNorm(NDArray a, object ord, int[] axes, int nd, bool keepdims)
        {
            int rowAxis = NormalizeAxis(axes[0], nd);
            int colAxis = NormalizeAxis(axes[1], nd);
            if (rowAxis == colAxis)
                throw new ValueError("Duplicate axes given.");

            NDArray result;

            if (IsText(ord, out var word))
            {
                switch (word)
                {
                    case "f":
                    case "fro":
                        result = FrobeniusOverBoth(a, rowAxis, colAxis);
                        break;
                    case "nuc":
                        result = SingularValueNorm(a, rowAxis, colAxis, "nuc");
                        break;
                    default:
                        throw new ValueError("Invalid norm order for matrices.");
                }
            }
            else if (ord is null)
            {
                result = FrobeniusOverBoth(a, rowAxis, colAxis);
            }
            else
            {
                double order = AsNumber(ord);

                // NumPy reduces one axis FIRST and then shifts the surviving axis index down if
                // the reduced one sat before it — so these pairs are not interchangeable. The two
                // max forms reduce with initial=0 like NumPy 2.x (`.max(axis=..., initial=0)`): an
                // EMPTY matrix's 1-norm and inf-norm are 0.0; the min forms keep NumPy's empty-min
                // error.
                if (order == 2d || order == -2d)
                    result = SingularValueNorm(a, rowAxis, colAxis, order == 2d ? "2" : "-2");
                else if (order == 1d)
                    result = np.amax(np.sum(np.abs(a), rowAxis), Shift(colAxis, rowAxis), initial: 0);
                else if (order == -1d)
                    result = np.amin(np.sum(np.abs(a), rowAxis), Shift(colAxis, rowAxis));
                else if (double.IsPositiveInfinity(order))
                    result = np.amax(np.sum(np.abs(a), colAxis), Shift(rowAxis, colAxis), initial: 0);
                else if (double.IsNegativeInfinity(order))
                    result = np.amin(np.sum(np.abs(a), colAxis), Shift(rowAxis, colAxis));
                else
                    throw new ValueError("Invalid norm order for matrices.");
            }

            if (!keepdims)
                return result;

            // Put the two reduced axes back as length-1, lowest index first so the second
            // insertion is not displaced by the first.
            int first = rowAxis < colAxis ? rowAxis : colAxis;
            int second = rowAxis < colAxis ? colAxis : rowAxis;
            return np.expand_dims(np.expand_dims(result, first), second);
        }

        /// <summary>The Frobenius norm over an axis pair: the square root of the summed <c>|x|²</c>.</summary>
        /// <param name="a">The floating operand.</param>
        /// <param name="rowAxis">The normalized row axis.</param>
        /// <param name="colAxis">The normalized column axis.</param>
        /// <returns>The norm, both axes reduced.</returns>
        [NDScopedCovered] // only reached via MatrixNorm, itself under the [NDScoped] Norm scope
        private static NDArray FrobeniusOverBoth(NDArray a, int rowAxis, int colAxis)
        {
            var squared = AbsSquared(a);
            var once = np.sum(squared, rowAxis);
            return np.sqrt(np.sum(once, Shift(colAxis, rowAxis)));
        }

        /// <summary>An axis index after a LOWER-numbered axis has been reduced away.</summary>
        /// <param name="axis">The surviving axis, as numbered before the reduction.</param>
        /// <param name="removed">The reduced axis.</param>
        /// <returns><paramref name="axis"/> one lower when <paramref name="removed"/> sat before it, else unchanged.</returns>
        private static int Shift(int axis, int removed) => axis > removed ? axis - 1 : axis;

        /// <summary>
        ///     NumPy's <c>_multi_svd_norm</c>: the singular values over the axis pair, reduced by the largest (2, with
        ///     <c>initial=0</c>), the smallest (-2) or their sum ("nuc").
        /// </summary>
        /// <param name="a">The floating operand.</param>
        /// <param name="rowAxis">The normalized row axis.</param>
        /// <param name="colAxis">The normalized column axis.</param>
        /// <param name="which">"2", "-2" or "nuc".</param>
        /// <returns>The reduced norm.</returns>
        /// <exception cref="OpenBlasMissingBackendException">No matrix backend computes the singular values of a non-empty operand.</exception>
        /// <exception cref="ValueError">"-2" of an empty matrix (the empty-min error).</exception>
        [NDScopedCovered] // only reached via MatrixNorm, itself under the [NDScoped] Norm scope
        private static NDArray SingularValueNorm(NDArray a, int rowAxis, int colAxis, string which)
        {
            // NumPy's _multi_svd_norm: move the two matrix axes to the end, take the singular values
            // over them, then reduce — 2 → largest, -2 → smallest, 'nuc' → sum of the singulars.
            // NumPy calls `_multi_svd_norm(x, row, col, amax, 0)`: the largest singular value of an
            // EMPTY matrix is 0.0 (singular values are >= 0, so a non-empty answer is unchanged).
            // 'nuc' passes initial=0 to the sum, which the add identity already is.
            var y = np.moveaxis(a, new[] {rowAxis, colAxis}, new[] {-2, -1});
            var s = np.linalg.svdvals(y);
            return which switch
            {
                "2" => np.amax(s, -1, initial: 0),
                "-2" => np.amin(s, -1),
                _ => np.sum(s, -1), // "nuc"
            };
        }

        /// <summary>
        ///     NumPy computes norms in floating point — a bool or integer operand is cast up
        ///     before anything else happens, which is why <c>norm(arange(3))</c> is float64.
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <returns><paramref name="x"/> itself for a float16 / float32 / float64 / decimal / complex operand, else its
        ///     float64 cast.</returns>
        private static NDArray ToFloating(NDArray x)
            => x.typecode switch
            {
                NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double
                    or NPTypeCode.Decimal or NPTypeCode.Complex => x,
                _ => x.TensorEngine.Cast(x, typeof(double), copy: false)
            };

        /// <summary>Every axis of a rank, in order.</summary>
        /// <param name="nd">The rank.</param>
        /// <returns><c>[0, 1, ..., nd - 1]</c>.</returns>
        private static int[] AllAxes(int nd)
        {
            var all = new int[nd];
            for (int i = 0; i < nd; i++)
                all[i] = i;
            return all;
        }

        /// <summary>A shape of length-1 axes: keepdims after a whole-array reduction.</summary>
        /// <param name="nd">The rank.</param>
        /// <returns><paramref name="nd"/> ones.</returns>
        private static int[] Ones(int nd)
        {
            var ones = new int[nd];
            for (int i = 0; i < nd; i++)
                ones[i] = 1;
            return ones;
        }

        /// <summary>
        ///     NumPy's <c>normalize_axis_index</c> for the norm family (norm's matrix branch, and the reduce its vector
        ///     branch runs through, which takes the axis as a TUPLE - so a 0-d input has no axis 0 here).
        /// </summary>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <param name="nd">The operand's rank.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        private static int NormalizeAxis(int axis, int nd) => Backends.DefaultEngine.check_and_adjust_axis(nd, axis);

        /// <summary>
        ///     Python's <c>x.shape[axis]</c>: the tuple index np.linalg.cross reads BEFORE it normalizes anything, so an
        ///     axis out of range is the IndexError Python raises there, not an AxisError.
        /// </summary>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <param name="nd">The operand's rank.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range").</exception>
        private static int ShapeIndex(int axis, int nd)
        {
            if (axis < -nd || axis >= nd)
                throw new IndexError("tuple index out of range");
            return axis < 0 ? axis + nd : axis;
        }

        /// <summary>Whether an order is a name (a C# string: NumPy's str order).</summary>
        /// <param name="ord">The order argument.</param>
        /// <param name="word">The name, or null.</param>
        /// <returns>True when <paramref name="ord"/> is a string.</returns>
        private static bool IsText(object ord, out string word)
        {
            word = ord as string;
            return word is not null;
        }

        /// <summary>Whether an order is a number, and its value.</summary>
        /// <param name="ord">The order argument.</param>
        /// <param name="value">The order as a double, or 0.</param>
        /// <returns>False for null and for a string; true otherwise (with the converted value).</returns>
        /// <exception cref="TypeError">An order that does not convert to a double.</exception>
        private static bool IsNumber(object ord, out double value)
        {
            value = 0;
            if (ord is null || ord is string)
                return false;
            value = AsNumber(ord);
            return true;
        }

        /// <summary>An order's value as a double.</summary>
        /// <param name="ord">The order argument (not a string).</param>
        /// <returns>The converted value.</returns>
        /// <exception cref="TypeError">An order that does not convert: <c>'ord' must be None, a number or one of 'fro',
        ///     'f', 'nuc'</c>.</exception>
        private static double AsNumber(object ord)
        {
            try
            {
                return Convert.ToDouble(ord, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception e) when (e is FormatException or InvalidCastException or OverflowException)
            {
                throw new TypeError("'ord' must be None, a number or one of 'fro', 'f', 'nuc'", e);
            }
        }
    }
}
