using System;
using System.Collections.Generic;
using NumSharp.Backends;

// =============================================================================
// QuantileText.cs — np.quantile / np.percentile over text (S / U, NEP 55 StringDType): NumPy 2.4.2's take path
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
//   * StringDType differs in two places: an interpolating method (or 'linear' with a fractional q) reads the two
//     neighbours and calls _lerp, whose `b - a` subtracts their Python objects (NDVStringStatistics.QuantileInterpolate
//     runs that branch); and a 0-d take under keepdims is the element's own Python object (a str, or the NA object), so
//     _ureduce's tuple index raises that object's refusal.
//   * Text has no selection loop, so the partition is NumPy's generic quicksort of every line (StringDType's sort with its
//     NA rules: a None-like NA in a line of two or more elements raises there); the k-th string of a sorted
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
    ///     <c>np.quantile</c> / <c>np.percentile</c> over text — fixed-width <c>S</c> / <c>U</c> and NEP 55
    ///     <c>StringDType</c> — for the methods NumPy computes on text (the selection methods, and <c>linear</c> with an
    ///     integer q; over <c>StringDType</c> the interpolating methods too, which read the neighbours before they
    ///     refuse): see the file header.
    /// </summary>
    internal static class QuantileText
    {
        /// <summary>
        ///     NumPy's take path of <c>_quantile</c> over text, wrapped in <c>_ureduce</c>'s keepdims and out handling, with the
        ///     checks between them in NumPy's order: <c>_ureduce</c>'s prologue (<see cref="QuantileEngine.UReduce"/>: the
        ///     axes, the out index under keepdims, the merge of several axes), q's rank (<c>_quantile_ureduce_func</c>), the
        ///     method and the interpolating methods' refusal (<c>_quantile</c>), then the take.
        /// </summary>
        /// <param name="op">The NumPy function name (<c>quantile</c> or <c>percentile</c>), for the refusals' texts.</param>
        /// <param name="a">The text data (<c>S</c>, <c>U</c> or <c>StringDType</c>; only read).</param>
        /// <param name="q">The quantiles in [0, 1] (a percentile already divided by 100), flattened in C order.</param>
        /// <param name="qShape">q's shape: empty for a scalar q.</param>
        /// <param name="qIsInteger">True when q is an integer (or bool) array: NumPy's virtual index <c>(n - 1) * q</c> is then
        ///     an integer, and 'linear' takes the take path.</param>
        /// <param name="axes">The reduced axes as the caller gave them (the prologue normalizes them, NumPy's first check), or
        ///     null for every axis.</param>
        /// <param name="out">An optional output array, or null.</param>
        /// <param name="method">The method name.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="partitionRaises">True for a <c>numpy.char.chararray</c> operand of the user's call: NumPy's
        ///     <c>arr.partition(..., axis=0)</c> is then <c>chararray.partition(sep)</c>, which raises once the method is
        ///     resolved (see the exceptions).</param>
        /// <param name="qType">q's precision (<see cref="QuantileIndexes.PrecisionOf"/>): NumPy computes the virtual indexes
        ///     in a float16 / float32 q's own dtype, so they can round to the line's length (or to NaN / infinity), which the
        ///     partition refuses.</param>
        /// <returns>The selected strings, of shape <c>q.shape + kept dims</c> (a read-only 0-d array for a scalar q over every
        ///     axis: NumPy's scalar, or a <c>StringDType</c> element's Python object); <paramref name="out"/> itself when
        ///     given. A <c>StringDType</c> interpolating method returns only where NumPy's <c>_lerp</c> does: two NaN-like
        ///     NA neighbours give <c>np.float64(nan)</c>, written into a numeric <paramref name="out"/> by its own cast
        ///     rules.</returns>
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
        ///     <see cref="Backends.Strings.NDStringUfuncs.QuantileError"/>; over <c>StringDType</c> the Python subtraction of
        ///     the two neighbours, <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no <c>subtract</c> loop for a
        ///     non-scalar result; a text <paramref name="out"/> passes <c>_lerp</c>'s add and meets its subtract's
        ///     refusal; an integer / bool out the add's same_kind cast); <paramref name="out"/>'s dtype does not cast
        ///     safely to the data's (<c>Cannot cast array data from dtype('&lt;U5') to dtype('&lt;U2') according to the rule
        ///     'safe'</c>; <c>Cannot cast scalar from ...</c> for a 0-d out); or keepdims without <paramref name="out"/> over
        ///     a 0-d result (<c>string indices must be integers, not 'tuple'</c> for <c>U</c> and a <c>StringDType</c> str,
        ///     <c>byte indices must be integers or slices, not tuple</c> for <c>S</c>, <c>'float' object is not
        ///     subscriptable</c> / <c>'NoneType' object is not subscriptable</c> for a <c>StringDType</c> NA object: NumPy
        ///     indexes its scalar).</exception>
        /// <exception cref="AxisError">An axis is out of range (the prologue's first check).</exception>
        /// <exception cref="ValueError">An axis repeats (<c>repeated axis</c>, the prologue).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements holding a None-like NA (the sort's
        ///     <c>Cannot compare null that is not a nan-like value</c>), or the unsafe write-back into a numeric
        ///     <paramref name="out"/> that cannot parse a string (<c>could not convert string to float: 'a'</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="partitionRaises"/>, for every known method:
        ///     <c>chararray.partition() got an unexpected keyword argument 'axis'</c> — after the checks above it, before the
        ///     interpolating methods' refusal and the take.</exception>
        /// <exception cref="ValueError">The partition refuses a virtual index (NumPy's <c>kth(=k) out of bounds (n)</c>): a
        ///     float16 q over a line of about 2051 elements or more (a float32 q past 2^24) — before the interpolating
        ///     methods' refusal, whose subtraction NumPy reaches after the partition.</exception>
        /// <remarks>Scoped: the flattened or merged data, the sorted lines and the indexes are reclaimed; the result (or
        ///     the caller's <paramref name="out"/>) survives.</remarks>
        [NDScoped]
        internal static NDArray Compute(string op, NDArray a, double[] q, long[] qShape, bool qIsInteger, int[] axes,
            NDArray @out, string method, bool keepdims, bool partitionRaises = false, NPTypeCode qType = NPTypeCode.Double)
        {
            int nd = a.ndim;
            // _ureduce, the family's one prologue (QuantileEngine.UReduce): the axes normalized, keepdims' view of out (the
            // reduced axes taken at 0, every axis for axis=None), then the merge of several reduced axes into one last axis
            // behind the kept ones — `a.reshape(a.shape[:nkeep] + (-1,))`, whose -1 NumPy refuses to resolve when a kept axis
            // is 0 long ("cannot reshape array of size 0 into shape (0,newaxis)"), before q's rank is looked at.
            UReduced prologue = QuantileEngine.UReduce(a, axes, @out, keepdims);
            int[] reduced = prologue.Reduced;
            NDArray target = prologue.Target;

            // The data along one axis: _quantile_ureduce_func's flatten (axis=None), else the prologue's data along its
            // one axis (the reduced axis, or the merged last one).
            NDArray work;
            int axis;
            if (reduced is null)
            {
                work = a.flatten();
                axis = 0;
            }
            else
            {
                work = prologue.Data;
                axis = prologue.Axis.Value;
            }

            // _quantile_ureduce_func: `if q.ndim > 2: raise ValueError("q must be a scalar or 1d")`.
            if (qShape.Length > 2)
                throw new ValueError("q must be a scalar or 1d");

            // A numpy.char.chararray operand: _quantile resolves the method (an unknown one is its ValueError), then both its
            // routes call arr.partition(..., axis=0) on the chararray copy — chararray.partition(sep) takes no axis.
            if (partitionRaises)
            {
                QuantileEngine.ParseMethod(method);
                throw NDCharArrayBoundary.PartitionKeywordError();
            }

            // _quantile: the axis moved first; values_count = arr.shape[0]. 'linear' with an integer q indexes
            // (n - 1) * q itself (an integer, so its floor); every selection method keeps its own index rule, which for an
            // EMPTY line differs from linear's (inverted_cdf clips n * q - 1 = -1 to 0 where linear's (0 - 1) * 1 is -1).
            NDArray arr = np.moveaxis(work, axis, 0);
            long n = arr.Shape.dimensions[0];

            // _quantile: the method (an unknown name is QuantileAnswer's ValueError below), the virtual indexes in q's
            // precision, and `arr.partition(...)`, whose kth validation (partition_prep_kth_array; an array with no element
            // skips it) comes before the interpolating methods' subtraction — so it runs before their refusal. Text is not
            // inexact: the take path's list is the indexes alone, and a NaN virtual index stays INT64_MIN.
            long[] plan = null;
            if (QuantileEngine.TryParseMethod(method, out QuantileMethod known))
            {
                QuantileMethod planned = qIsInteger && known == QuantileMethod.Linear ? QuantileMethod.Lower : known;
                plan = Plan(planned, n, q, qType);
                // 'linear' with a uint64 q partitions at a uint64 kth array, which does not cast safely to intp: NumPy's
                // partition_prep_kth_array refuses its dtype before any value (an array with no element included).
                if (qIsInteger && known == QuantileMethod.Linear && qType == NPTypeCode.UInt64)
                    throw new TypeError("Partition index must be integer");
                if (n > 0 && a.size != 0)
                {
                    unsafe
                    {
                        fixed (long* planPtr = plan)
                        {
                            if (QuantileEngine.IsDiscrete(planned))
                                QuantileIndexes.CheckTakeKth(planPtr, q.Length, 2, n);
                            else
                                QuantileIndexes.CheckInterpolationKth(planPtr, 2 * q.Length, n);
                        }
                    }
                }
            }

            // _quantile: the interpolating methods' refusal (an empty reduced axis first: their arr[previous_indexes]
            // indexes it). A selection method — or 'linear' with an integer q — goes on; so does an interpolating method over
            // StringDType, whose answer depends on the neighbours' values.
            var refusal = Backends.Strings.NDStringUfuncs.QuantileAnswer(op, a, method, axes, qShape.Length == 0, qIsInteger, q);
            if (refusal is not null)
                throw refusal;
            QuantileMethod parsed = QuantileEngine.ParseMethod(method);

            bool vstring = Backends.Strings.NDStringUfuncs.IsVString(a);
            if (vstring && !(QuantileEngine.IsDiscrete(parsed) || (parsed == QuantileMethod.Linear && qIsInteger)))
                // StringDType's interpolating branch: the sort, the neighbours, and _lerp's Python subtraction of them.
                return Backends.Strings.NDVStringStatistics.QuantileInterpolate(arr, q, qShape, parsed, target, @out, keepdims,
                    nd, reduced, qType);
            // The take path's indexes: the plan's selected index of each quantile (in q's precision).
            var indexes = new long[q.Length];
            for (int i = 0; i < indexes.Length; i++)
                indexes[i] = plan[2 * i];

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

            // The partition (`arr.partition(virtual_indexes.ravel(), axis=0)`, before the take): text has no selection loop,
            // NumPy's generic quicksort sorts each line; the strings at the virtual indexes of a sorted line do not depend
            // on the order of equal strings. Over StringDType the sort applies the NA rules (a None-like NA in a line of two
            // or more elements raises here, before the take's checks).
            NDArray sorted = n > 1 ? np.sort(arr, 0) : arr;

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

            NDArray picks = qShape.Length == 0 ? NDArray.Scalar(indexes[0]) : np.array(indexes).reshape(Shape.Adopt((long[])qShape.Clone()));
            NDArray result = np.take(sorted, picks, 0);

            if (target is not null)
            {
                // The writeback of the take's copy into out: an unsafe cast (U2 into U1 truncates, U into S encodes; a
                // StringDType copy into a float64 out parses, `could not convert string to float: 'a'`).
                np.copyto(target, result, "unsafe");
                return @out;
            }
            if (!keepdims)
                // A 0-d take is NumPy's scalar (a np.str_ / np.bytes_, or a StringDType element's Python object): read-only,
                // as NumSharp's 0-d reduction results are.
                return result.MarkReductionScalar();
            if (result.ndim == 0)
            {
                // NumPy's take returned a scalar, and _ureduce indexes it with a tuple: Python's refusal for the object —
                // bytes, a str, or a StringDType element's own object (a str, or the NA object).
                if (vstring)
                    throw Backends.Strings.NDVStringStatistics.ScalarSubscriptError(result.Storage.GetAtIndex(0));
                throw new TypeError(a.dtype.kind == 'S'
                    ? "byte indices must be integers or slices, not tuple"
                    : "string indices must be integers, not 'tuple'");
            }
            // r[(Ellipsis,) + index_r]: a view with a new axis of length one at every reduced axis.
            var expand = new Slice[1 + nd];
            expand[0] = Slice.Ellipsis;
            for (int d = 0; d < nd; d++)
                expand[1 + d] = reduced is null || Array.IndexOf(reduced, d) >= 0 ? Slice.NewAxis : Slice.All;
            return result[expand];
        }

        /// <summary>
        ///     NumPy's index plan of a line of <paramref name="n"/> elements, in q's precision (see
        ///     <see cref="QuantileIndexes"/>): per quantile, two raw indexes — a selection method's index twice, or an
        ///     interpolating method's previous and next neighbour (intp, before the partition's negative adjustment).
        /// </summary>
        /// <param name="method">The method ('linear' with an integer q already mapped to 'lower').</param>
        /// <param name="n">The line's length (NumPy's <c>values_count</c>; 0 for an empty line).</param>
        /// <param name="q">The quantiles in [0, 1] (values of q's precision).</param>
        /// <param name="qType">q's precision.</param>
        /// <returns>The raw indexes, two per quantile: out of range only for an empty line (n = 0, where NumPy's take or
        ///     neighbour lookup reports them) or where q's rounding reaches n (the partition reports those).</returns>
        internal static long[] Plan(QuantileMethod method, long n, double[] q, NPTypeCode qType)
        {
            var raw = new long[2 * q.Length];
            var weight = new double[2 * q.Length];
            unsafe
            {
                fixed (double* qPtr = q)
                fixed (long* rawPtr = raw)
                fixed (double* weightPtr = weight)
                    // Text is not an inexact dtype: a NaN virtual index is not mapped to the last element.
                    Backends.Kernels.DirectILKernelGenerator.QuantilePlan(qType, method, n, qPtr, q.Length, false, rawPtr,
                        weightPtr);
            }
            return raw;
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
            double virtualIndex = VirtualIndex(method, 0, q);
            long previous = (long)Math.Floor(virtualIndex);
            if (virtualIndex >= n - 1)
                previous = -1;
            if (virtualIndex < 0)
                previous = 0;
            return previous;
        }

        /// <summary>
        ///     An interpolating method's <c>get_virtual_index(n, quantiles)</c> for one quantile, computed in float64 with
        ///     NumPy's own expressions, operation for operation, so an integral index comes out exactly where NumPy's does
        ///     (it decides which neighbours are read, and over <c>StringDType</c> the neighbours' Python types decide the
        ///     error text).
        /// </summary>
        /// <param name="method">An interpolating method (not a selection method).</param>
        /// <param name="n">The line's length (NumPy's <c>values_count</c>).</param>
        /// <param name="q">The quantile, in [0, 1].</param>
        /// <returns>The virtual index (fractional, possibly outside <c>[0, n - 1]</c>).</returns>
        /// <exception cref="ArgumentOutOfRangeException">A selection method (never passed: those take, they do not
        ///     interpolate).</exception>
        internal static double VirtualIndex(QuantileMethod method, long n, double q)
        {
            double count = n;
            return method switch
            {
                // 'linear': (n - 1) * quantiles; 'midpoint': 0.5 * (floor((n - 1) * q) + ceil((n - 1) * q)).
                QuantileMethod.Linear => (count - 1) * q,
                QuantileMethod.Midpoint => 0.5 * (Math.Floor((count - 1) * q) + Math.Ceiling((count - 1) * q)),
                QuantileMethod.AveragedInvertedCdf => count * q - 1,
                // _compute_virtual_index(n, quantiles, alpha, beta) = n * q + (alpha + q * (1 - alpha - beta)) - 1.
                QuantileMethod.InterpolatedInvertedCdf => ComputeVirtualIndex(count, q, 0, 1),
                QuantileMethod.Hazen => ComputeVirtualIndex(count, q, 0.5, 0.5),
                QuantileMethod.Weibull => ComputeVirtualIndex(count, q, 0, 0),
                QuantileMethod.MedianUnbiased => ComputeVirtualIndex(count, q, 1 / 3.0, 1 / 3.0),
                QuantileMethod.NormalUnbiased => ComputeVirtualIndex(count, q, 3 / 8.0, 3 / 8.0),
                _ => throw new ArgumentOutOfRangeException(nameof(method), method, "not an interpolating method"),
            };
        }

        /// <summary>NumPy's <c>_compute_virtual_index(n, quantiles, alpha, beta)</c>, in its operation order.</summary>
        /// <param name="n">The line's length.</param>
        /// <param name="q">The quantile.</param>
        /// <param name="alpha">The method's alpha.</param>
        /// <param name="beta">The method's beta.</param>
        /// <returns><c>n * q + (alpha + q * (1 - alpha - beta)) - 1</c>.</returns>
        private static double ComputeVirtualIndex(double n, double q, double alpha, double beta)
            => n * q + (alpha + q * (1 - alpha - beta)) - 1;

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
