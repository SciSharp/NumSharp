using System;
using System.Collections.Generic;
using NumSharp.Backends;

// =============================================================================
// QuantileText.cs — np.quantile / np.percentile over fixed-width text (S / U): NumPy 2.4.2's take path
// =============================================================================
//
// WHAT NUMPY DOES (numpy/lib/_function_base_impl.py: _ureduce, _quantile_ureduce_func, _quantile)
//   * _ureduce normalizes the axes, indexes `out` with `(Ellipsis,) + index_out` when keepdims is set, merges several
//     reduced axes into one last axis (np.moveaxis of the kept axes to the front, then reshape), calls the reduction, and
//     either returns `out` itself or, with keepdims, `r[(Ellipsis,) + index_r]` — which fails for a 0-d result, because
//     ndarray.take returns a NumPy SCALAR then and a np.str_ / np.bytes_ cannot be indexed with a tuple.
//   * _quantile_ureduce_func flattens the data (axis=None) or copies it, and _quantile moves the axis first and computes
//     the virtual indexes. A method whose gamma needs no fix (lower, higher, nearest, inverted_cdf, closest_observation),
//     or 'linear' with INTEGER virtual indexes (an integer or bool q), takes the "supports_integers" path:
//     `arr.partition(virtual_indexes.ravel(), axis=0)` and `take(arr, virtual_indexes, axis=0, out=out)`. Every other
//     method interpolates (_lerp subtracts text: an error, raised before this file is reached).
//   * Text has no selection loop, so the partition is NumPy's generic quicksort of every line; the k-th string of a sorted
//     line is the same whatever order equal strings took. PyArray_TakeFrom then checks, in order: out's shape ("output
//     array does not match result of ndarray.take"), the safe cast from out's dtype to the data's, an empty axis with a
//     non-empty result ("cannot do a non-empty take from an empty axes."), and every index ("index -1 is out of bounds for
//     axis 0 with size 0", the original value); the values reach out through an unsafe cast (U2 into U1 truncates).
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   The statements above in NumPy's order, with the line sort of TextSort (np.sort along the moved axis) for the partition
//   and NumSharp's take for the gather.
// =============================================================================

namespace NumSharp.Statistics
{
    /// <summary>
    ///     <c>np.quantile</c> / <c>np.percentile</c> over fixed-width text for the methods NumPy computes on text (the
    ///     selection methods, and <c>linear</c> with an integer q): see the file header.
    /// </summary>
    internal static class QuantileText
    {
        /// <summary>
        ///     NumPy's take path of <c>_quantile</c> over text, wrapped in <c>_ureduce</c>'s keepdims and out handling, with the
        ///     checks between them in NumPy's order: the out index (keepdims), q's rank (<c>_quantile_ureduce_func</c>), the
        ///     method and the interpolating methods' refusal (<c>_quantile</c>), then the take.
        /// </summary>
        /// <param name="op">The NumPy function name (<c>quantile</c> or <c>percentile</c>), for the refusals' texts.</param>
        /// <param name="a">The text data (only read).</param>
        /// <param name="q">The quantiles in [0, 1] (a percentile already divided by 100), flattened in C order.</param>
        /// <param name="qShape">q's shape: empty for a scalar q.</param>
        /// <param name="qIsInteger">True when q is an integer (or bool) array: NumPy's virtual index <c>(n - 1) * q</c> is then
        ///     an integer, and 'linear' takes the take path.</param>
        /// <param name="axes">The reduced axes (already validated by the caller, as NumPy's <c>_ureduce</c> first does), or
        ///     null for every axis.</param>
        /// <param name="out">An optional output array, or null.</param>
        /// <param name="method">The method name.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The selected strings, of shape <c>q.shape + kept dims</c> (0-d for a scalar q over every axis, NumPy's
        ///     scalar); <paramref name="out"/> itself when given.</returns>
        /// <exception cref="IndexError">keepdims with an <paramref name="out"/> of too few dimensions (NumPy's
        ///     <c>out[(Ellipsis,) + index_out]</c>); an interpolating method over an empty reduced axis (<c>index 0 is out of
        ///     bounds for axis 0 with size 0</c>); the take over an empty axis with a non-empty result (<c>cannot do a
        ///     non-empty take from an empty axes.</c>), or whose indexes are still checked (<c>index -1 is out of bounds for
        ///     axis 0 with size 0</c>).</exception>
        /// <exception cref="ValueError">A q of rank above 2 (<c>q must be a scalar or 1d</c>), or <paramref name="out"/> of
        ///     another shape than the result (<c>output array does not match result of ndarray.take</c>).</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0: <c>_ureduce</c>'s
        ///     merge <c>reshape(kept + (-1,))</c> cannot resolve its -1 (<c>cannot reshape array of size 0 into shape
        ///     (0,newaxis)</c>, NumPy's ValueError text under the house reshape exception).</exception>
        /// <exception cref="ValueError">An unknown method (NumPy's <c>'x' is not a valid method</c> text; a ValueError is an
        ///     <see cref="ArgumentException"/>).</exception>
        /// <exception cref="TypeError">An interpolating method (NumPy's <c>_lerp</c> subtracts the text: see
        ///     <see cref="Backends.Strings.NDStringUfuncs.QuantileError"/>); <paramref name="out"/>'s dtype does not cast
        ///     safely to the data's (<c>Cannot cast array data from dtype('&lt;U5') to dtype('&lt;U2') according to the rule
        ///     'safe'</c>; <c>Cannot cast scalar from ...</c> for a 0-d out); or keepdims without <paramref name="out"/> over
        ///     a 0-d result (<c>string indices must be integers, not 'tuple'</c> for <c>U</c>, <c>byte indices must be
        ///     integers or slices, not tuple</c> for <c>S</c>: NumPy indexes its scalar).</exception>
        /// <remarks>Scoped: the flattened or merged data, the sorted lines and the indexes are reclaimed; the result (or
        ///     the caller's <paramref name="out"/>) survives.</remarks>
        [NDScoped]
        internal static NDArray Compute(string op, NDArray a, double[] q, long[] qShape, bool qIsInteger, int[] axes,
            NDArray @out, string method, bool keepdims)
        {
            int nd = a.ndim;
            int[] reduced = axes is null ? null : QuantileEngine.NormalizeAxes(axes, nd);

            // _ureduce: with keepdims, the reduction writes into out[(Ellipsis,) + index_out], a view with the reduced
            // axes taken at 0 (every axis for axis=None) — indexed before the reduction function runs.
            NDArray target = @out;
            if (@out is not null && keepdims)
            {
                var index = new Slice[1 + nd];
                index[0] = Slice.Ellipsis;
                for (int d = 0; d < nd; d++)
                    index[1 + d] = reduced is null || Array.IndexOf(reduced, d) >= 0 ? Slice.Index(0) : Slice.All;
                target = @out[index];
            }

            // The data along one axis: _quantile_ureduce_func's flatten (axis=None), the one reduced axis, or _ureduce's
            // merge of several reduced axes into one last axis behind the kept ones — `a.reshape(a.shape[:nkeep] + (-1,))`,
            // whose -1 NumPy refuses to resolve when a kept axis is 0 long ("cannot reshape array of size 0 into shape
            // (0,newaxis)"), before q's rank is looked at.
            NDArray work;
            int axis;
            if (reduced is null)
            {
                work = a.flatten();
                axis = 0;
            }
            else if (reduced.Length == 1)
            {
                work = a;
                axis = reduced[0];
            }
            else
            {
                var keep = new List<int>();
                for (int d = 0; d < nd; d++)
                    if (Array.IndexOf(reduced, d) < 0)
                        keep.Add(d);
                var destinations = new int[keep.Count];
                for (int i = 0; i < destinations.Length; i++)
                    destinations[i] = i;
                NDArray movedKeep = np.moveaxis(a, keep.ToArray(), destinations);
                var merged = new long[keep.Count + 1];
                for (int d = 0; d < keep.Count; d++)
                    merged[d] = movedKeep.Shape.dimensions[d];
                merged[keep.Count] = -1;
                work = movedKeep.reshape(Shape.Adopt(merged));
                axis = keep.Count;
            }

            // _quantile_ureduce_func: `if q.ndim > 2: raise ValueError("q must be a scalar or 1d")`.
            if (qShape.Length > 2)
                throw new ValueError("q must be a scalar or 1d");

            // _quantile: the method (unknown names refused), then the interpolating methods' refusal (an empty reduced axis
            // first: their arr[previous_indexes] indexes it). A selection method — or 'linear' with an integer q — goes on.
            var refusal = Backends.Strings.NDStringUfuncs.QuantileAnswer(op, a, method, axes, qShape.Length == 0, qIsInteger, q);
            if (refusal is not null)
                throw refusal;
            QuantileMethod parsed = QuantileEngine.ParseMethod(method);

            // _quantile: the axis moved first; values_count = arr.shape[0]. 'linear' with an integer q indexes
            // (n - 1) * q itself (an integer, so its floor); every selection method keeps its own index rule, which for an
            // EMPTY line differs from linear's (inverted_cdf clips n * q - 1 = -1 to 0 where linear's (0 - 1) * 1 is -1).
            NDArray arr = np.moveaxis(work, axis, 0);
            long n = arr.Shape.dimensions[0];
            long[] indexes = VirtualIndexes(n, q, qIsInteger && parsed == QuantileMethod.Linear ? QuantileMethod.Lower : parsed);

            // take(arr, virtual_indexes, axis=0, out=out): its shape is q.shape + arr.shape[1:].
            var resultShape = new long[qShape.Length + arr.ndim - 1];
            long resultSize = 1;
            for (int d = 0; d < qShape.Length; d++)
            {
                resultShape[d] = qShape[d];
                resultSize *= qShape[d];
            }
            for (int d = 1; d < arr.ndim; d++)
            {
                resultShape[qShape.Length + d - 1] = arr.Shape.dimensions[d];
                resultSize *= arr.Shape.dimensions[d];
            }

            if (target is not null)
            {
                // PyArray_TakeFrom: out's shape first, then the safe cast from out's dtype to the data's.
                if (!SameDims(target.Shape.dimensions, resultShape))
                    throw new ValueError("output array does not match result of ndarray.take");
                // PyArray_FromArray(out, data dtype): npy_set_invalid_cast_error names a 0-d out a "scalar".
                if (!np.can_cast(target.dtype, arr.dtype, "safe"))
                    throw new TypeError($"Cannot cast {(target.ndim == 0 ? "scalar" : "array data")} from {target.dtype.ToString(true)} " +
                                        $"to {arr.dtype.ToString(true)} according to the rule 'safe'");
            }
            if (n == 0 && resultSize != 0)
                throw new IndexError("cannot do a non-empty take from an empty axes.");
            // npy_fasttake checks every index (the outer product before the axis is one line), in C order of q.
            foreach (long index in indexes)
                if (index < -n || index >= n)
                    throw new IndexError($"index {index} is out of bounds for axis 0 with size {n}");

            // The partition: text has no selection loop, NumPy's generic quicksort sorts each line; the strings at the
            // virtual indexes of a sorted line do not depend on the order of equal strings.
            NDArray sorted = n > 1 ? np.sort(arr, 0) : arr;
            NDArray picks = qShape.Length == 0 ? NDArray.Scalar(indexes[0]) : np.array(indexes).reshape(Shape.Adopt((long[])qShape.Clone()));
            NDArray result = np.take(sorted, picks, 0);

            if (target is not null)
            {
                // The writeback of the take's copy into out: an unsafe cast (U2 into U1 truncates, U into S encodes).
                np.copyto(target, result, "unsafe");
                return @out;
            }
            if (!keepdims)
                return result;
            if (result.ndim == 0)
                // NumPy's take returned a scalar, and _ureduce indexes it with a tuple: Python's str / bytes refusal.
                throw new TypeError(a.dtype.kind == 'S'
                    ? "byte indices must be integers or slices, not tuple"
                    : "string indices must be integers, not 'tuple'");
            // r[(Ellipsis,) + index_r]: a view with a new axis of length one at every reduced axis.
            var expand = new Slice[1 + nd];
            expand[0] = Slice.Ellipsis;
            for (int d = 0; d < nd; d++)
                expand[1 + d] = reduced is null || Array.IndexOf(reduced, d) >= 0 ? Slice.NewAxis : Slice.All;
            return result[expand];
        }

        /// <summary>
        ///     NumPy's virtual indexes (intp) of a take-path method for a line of <paramref name="n"/> elements, one per
        ///     quantile, computed as <c>_QuantileMethods[method]['get_virtual_index']</c> computes them in float64.
        /// </summary>
        /// <param name="n">The line's length (NumPy's <c>values_count</c>).</param>
        /// <param name="q">The quantiles in [0, 1].</param>
        /// <param name="method">A selection method: <see cref="QuantileMethod.Lower"/> also serves 'linear' with an integer
        ///     q, whose index <c>(n - 1) * q</c> is an integer already (the other selection methods keep their own rules,
        ///     which differ from it over an empty line).</param>
        /// <returns>The indexes; out of range only for an empty line (n = 0), where NumPy's take reports them.</returns>
        /// <exception cref="ArgumentOutOfRangeException">A method that interpolates (never passed: the caller refuses those
        ///     first).</exception>
        private static long[] VirtualIndexes(long n, double[] q, QuantileMethod method)
        {
            var indexes = new long[q.Length];
            for (int i = 0; i < q.Length; i++)
            {
                double qi = q[i];
                switch (method)
                {
                    case QuantileMethod.Lower:
                        // np.floor((n - 1) * quantiles).astype(np.intp)
                        indexes[i] = (long)Math.Floor((n - 1) * qi);
                        break;
                    case QuantileMethod.Higher:
                        indexes[i] = (long)Math.Ceiling((n - 1) * qi);
                        break;
                    case QuantileMethod.Nearest:
                        // np.around rounds half to even.
                        indexes[i] = (long)Math.Round((n - 1) * qi, MidpointRounding.ToEven);
                        break;
                    case QuantileMethod.InvertedCdf:
                        // _discrete_interpolation_to_boundaries(n * q - 1, gamma == 0): the previous index where the
                        // virtual index is integral, the next one otherwise, negatives clipped to 0.
                        indexes[i] = DiscreteBoundary(n * qi - 1, closestObservation: false);
                        break;
                    case QuantileMethod.ClosestObservation:
                        // ... with (n * q - 1 - 0.5) and the previous index only at an integral index whose floor is odd.
                        indexes[i] = DiscreteBoundary(n * qi - 1 - 0.5, closestObservation: true);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(method), method, "not a take-path method");
                }
            }
            return indexes;
        }

        /// <summary>
        ///     The neighbour index an interpolating method reads first over an EMPTY line: NumPy's
        ///     <c>_get_indexes(arr, virtual_indexes, 0)</c> for the first quantile, which <c>arr[previous_indexes]</c> then
        ///     reports (<c>index … is out of bounds for axis 0 with size 0</c>).
        /// </summary>
        /// <param name="method">An interpolating method (not a selection method; 'linear' with a float q).</param>
        /// <param name="q">The first quantile in C order, in [0, 1].</param>
        /// <returns>0 when the virtual index is negative, -1 when it is zero (<c>linear</c> / <c>midpoint</c> at q = 0,
        ///     <c>weibull</c> at q = 1).</returns>
        /// <remarks>
        ///     <c>get_virtual_index(0, q)</c> lies in [-1, 0] for every method and q in [0, 1]. <c>_get_indexes</c> first sets
        ///     every index with <c>virtual &gt;= n - 1 = -1</c> to -1, then every one with <c>virtual &lt; 0</c> to 0, so only
        ///     the sign of the virtual index decides. The expressions below are NumPy's own, operation for operation, so a
        ///     zero comes out exactly where NumPy's does (<c>-0.0</c> is not below 0).
        /// </remarks>
        internal static long EmptyAxisNeighbour(QuantileMethod method, double q)
        {
            const double n = 0;
            double virtualIndex = method switch
            {
                // 'linear': (n - 1) * quantiles; 'midpoint': 0.5 * (floor((n - 1) * q) + ceil((n - 1) * q)).
                QuantileMethod.Linear => (n - 1) * q,
                QuantileMethod.Midpoint => 0.5 * (Math.Floor((n - 1) * q) + Math.Ceiling((n - 1) * q)),
                QuantileMethod.AveragedInvertedCdf => n * q - 1,
                // _compute_virtual_index(n, quantiles, alpha, beta) = n * q + (alpha + q * (1 - alpha - beta)) - 1.
                QuantileMethod.InterpolatedInvertedCdf => ComputeVirtualIndex(n, q, 0, 1),
                QuantileMethod.Hazen => ComputeVirtualIndex(n, q, 0.5, 0.5),
                QuantileMethod.Weibull => ComputeVirtualIndex(n, q, 0, 0),
                QuantileMethod.MedianUnbiased => ComputeVirtualIndex(n, q, 1 / 3.0, 1 / 3.0),
                QuantileMethod.NormalUnbiased => ComputeVirtualIndex(n, q, 3 / 8.0, 3 / 8.0),
                _ => throw new ArgumentOutOfRangeException(nameof(method), method, "not an interpolating method"),
            };
            long previous = (long)Math.Floor(virtualIndex);
            if (virtualIndex >= n - 1)
                previous = -1;
            if (virtualIndex < 0)
                previous = 0;
            return previous;
        }

        /// <summary>NumPy's <c>_compute_virtual_index(n, quantiles, alpha, beta)</c>, in its operation order.</summary>
        /// <param name="n">The line's length.</param>
        /// <param name="q">The quantile.</param>
        /// <param name="alpha">The method's alpha.</param>
        /// <param name="beta">The method's beta.</param>
        /// <returns><c>n * q + (alpha + q * (1 - alpha - beta)) - 1</c>.</returns>
        private static double ComputeVirtualIndex(double n, double q, double alpha, double beta)
            => n * q + (alpha + q * (1 - alpha - beta)) - 1;

        /// <summary>
        ///     NumPy's <c>_discrete_interpolation_to_boundaries(index, gamma_condition_fun)</c>: <c>previous = floor(index)</c>,
        ///     <c>next = previous + 1</c>, <c>gamma = index - previous</c>; the previous index where the condition holds, the
        ///     next one otherwise, as intp, a negative result clipped to 0.
        /// </summary>
        /// <param name="index">The virtual index (float64).</param>
        /// <param name="closestObservation">The closest_observation condition (<c>gamma == 0 &amp; floor(index) % 2 == 1</c>)
        ///     instead of inverted_cdf's (<c>gamma == 0</c>).</param>
        /// <returns>The index.</returns>
        private static long DiscreteBoundary(double index, bool closestObservation)
        {
            double previous = Math.Floor(index);
            double gamma = index - previous;
            // Python's float % is a floored modulo: (-3.0) % 2 == 1.0.
            double mod = previous - 2 * Math.Floor(previous / 2);
            bool takePrevious = gamma == 0 && (!closestObservation || mod == 1);
            long result = (long)(takePrevious ? previous : previous + 1);
            return result < 0 ? 0 : result;
        }

        /// <summary>Whether two dimension lists are equal.</summary>
        /// <param name="a">The first.</param>
        /// <param name="b">The second.</param>
        /// <returns>True when they have the same length and values.</returns>
        private static bool SameDims(long[] a, long[] b)
        {
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }
    }
}
