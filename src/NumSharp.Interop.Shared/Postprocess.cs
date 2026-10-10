using System;

// ONE source file, compiled into every inference bridge that offers post-processing. Each bridge project links this
// file and defines the symbol that selects its namespace, so a fix lands once and every bridge gets it. The public
// type keeps the same name in each bridge; a file that imports two bridges names the one it wants with a using alias
// (using Post = NumSharp.Interop.OnnxRuntime.Postprocess;), because two types of one name are otherwise CS0104.
#if NUMSHARP_INTEROP_ONNXRUNTIME
namespace NumSharp.Interop.OnnxRuntime
#elif NUMSHARP_INTEROP_MLNET
namespace NumSharp.Interop.MLNet
#else
#error Postprocess.cs is compiled only into a bridge project that defines NUMSHARP_INTEROP_ONNXRUNTIME or NUMSHARP_INTEROP_MLNET.
#endif
{
    /// <summary>
    ///     The output side of inference as NumSharp compositions instead of hand-written loops: the softmax, argmax
    ///     and top-k that every classification post-step re-implements with <c>Math.Exp</c>,
    ///     <c>OrderByDescending().Take(k)</c> and a max-index loop. Nothing here is a new kernel: each method is a few
    ///     <c>np.*</c> calls over the <see cref="NDArray"/> that a model output became. The type is separate from the
    ///     conversion verbs so that those stay conversion-only.
    /// </summary>
    /// <remarks>
    ///     <para><b>Result dtypes follow NumPy's <c>exp</c>.</b> A float input keeps its precision (float32 logits give
    ///     a float32 softmax). A bool or integer input is computed in the float dtype NumPy's <c>exp</c> loop selects
    ///     for it: bool, int8 and uint8 in float16; int16, uint16 and char in float32; wider integers in float64.
    ///     float16 holds about three decimal digits, so cast narrow integers to float32 yourself when you need
    ///     more.</para>
    ///     <para><b>The integer input is converted BEFORE the arithmetic.</b> The subtraction of the maximum and the
    ///     negation would otherwise run in the integer dtype and wrap: <c>Softmax</c> of the uint8 values
    ///     <c>[1, 3]</c> computed <c>1 - 3 = 254</c> and returned <c>[NaN, 0]</c>, which is also what
    ///     <c>scipy.special.softmax</c> returns for that input. These methods return the mathematically correct
    ///     values instead.</para>
    /// </remarks>
    public static class Postprocess
    {
        /// <summary>
        ///     Numerically stable softmax along <paramref name="axis"/>: <c>exp(x - max) / sum(exp(x - max))</c>.
        ///     Subtracting the maximum cannot change the result and keeps <c>exp</c> from overflowing, so each slice
        ///     along the axis sums to 1 even for very large logits.
        /// </summary>
        /// <param name="logits">The scores; any layout. A bool or integer array is computed in its float dtype (see the remarks on the class).</param>
        /// <param name="axis">The axis the probabilities are normalized over; a negative value counts from the end.</param>
        /// <returns>A new array of the input's shape, in the input's float dtype (or its float tier for a bool or integer input); the caller disposes it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="logits"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is out of range for the array (NumSharp's <c>AxisError</c>).</exception>
        /// <exception cref="ArgumentException">The array has length 0 along <paramref name="axis"/>: a maximum of nothing has no value.</exception>
        /// <remarks>
        ///     A slice that holds <c>+inf</c> or NaN comes back as NaN, as <c>scipy.special.softmax</c> returns it:
        ///     <c>inf - inf</c> is NaN and it reaches the sum.
        /// </remarks>
        public static NDArray Softmax(NDArray logits, int axis = -1)
        {
            if (logits is null)
                throw new ArgumentNullException(nameof(logits));
            NDArray x = PromoteForExp(logits, out bool owned);
            try
            {
                using NDArray max = np.max(x, axis, keepdims: true);
                using NDArray shifted = x - max;
                using NDArray exp = np.exp(shifted);
                using NDArray sum = np.sum(exp, axis, keepdims: true);
                return exp / sum;
            }
            finally
            {
                // The float copy of an integer input is this method's own temporary; a float input is the caller's array.
                if (owned)
                    x.Dispose();
            }
        }

        /// <summary>
        ///     Log-softmax along <paramref name="axis"/>: <c>x - max - log(sum(exp(x - max)))</c>. Use it where a
        ///     cross-entropy or negative-log-likelihood step needs log-probabilities: <c>log(softmax(x))</c>
        ///     underflows to <c>-inf</c> for a small probability, and this form does not.
        /// </summary>
        /// <param name="logits">The scores; any layout. A bool or integer array is computed in its float dtype (see the remarks on the class).</param>
        /// <param name="axis">The axis the probabilities are normalized over; a negative value counts from the end.</param>
        /// <returns>A new array of the input's shape, in the input's float dtype (or its float tier for a bool or integer input); the caller disposes it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="logits"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is out of range for the array (NumSharp's <c>AxisError</c>).</exception>
        /// <exception cref="ArgumentException">The array has length 0 along <paramref name="axis"/>.</exception>
        /// <remarks>
        ///     A slice whose maximum is not finite follows <c>scipy.special.log_softmax</c>: the maximum is replaced
        ///     by 0 before the shift. So <c>[inf, 1]</c> gives <c>[NaN, -inf]</c> (the finite element is infinitely
        ///     less likely than the infinite one) where the plain formula would give NaN for both, and a slice that
        ///     holds NaN stays all NaN.
        /// </remarks>
        public static NDArray LogSoftmax(NDArray logits, int axis = -1)
        {
            if (logits is null)
                throw new ArgumentNullException(nameof(logits));
            NDArray x = PromoteForExp(logits, out bool owned);
            NDArray finiteMax = null;
            try
            {
                using NDArray rawMax = np.max(x, axis, keepdims: true);
                NDArray max = rawMax;
                if (HasNonFiniteValues(x.typecode))
                {
                    // SciPy's x_max[~isfinite(x_max)] = 0. A maximum of +inf would make the shift inf - inf = NaN for
                    // the infinite element AND -inf - ... for none: with 0 the finite elements keep their own value
                    // and end at -inf, the correct limit. The replacement array is built only when a slice needs it
                    // (the maximum has one element per slice, so the test is cheap).
                    using NDArray finite = np.isfinite(rawMax);
                    if (!np.all(finite))
                    {
                        using NDArray zeros = np.zeros_like(rawMax);
                        max = finiteMax = np.where(finite, rawMax, zeros);
                    }
                }

                using NDArray shifted = x - max;
                using NDArray exp = np.exp(shifted);
                using NDArray sum = np.sum(exp, axis, keepdims: true);
                using NDArray logSum = np.log(sum);
                return shifted - logSum;
            }
            finally
            {
                finiteMax?.Dispose();
                if (owned)
                    x.Dispose();
            }
        }

        /// <summary>
        ///     Element-wise logistic sigmoid <c>1 / (1 + exp(-x))</c>: the probability of a multi-label head, a
        ///     binary-classifier score or a detection confidence.
        /// </summary>
        /// <param name="x">The scores; any layout. A bool or integer array is computed in its float dtype (see the remarks on the class).</param>
        /// <returns>A new array of the input's shape, in the input's float dtype (or its float tier for a bool or integer input); the caller disposes it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>
        ///     A large negative input makes <c>exp(-x)</c> overflow to <c>+inf</c>, and <c>1 / inf</c> is 0: the
        ///     correct limit, with no exception. An unsigned or bool input used to be negated in its own dtype, where
        ///     <c>-1</c> wraps to the largest value, so every non-zero element came back as 0.
        /// </remarks>
        public static NDArray Sigmoid(NDArray x)
        {
            if (x is null)
                throw new ArgumentNullException(nameof(x));
            NDArray v = PromoteForExp(x, out bool owned);
            try
            {
                using NDArray negated = -v;
                using NDArray exp = np.exp(negated);
                using NDArray denominator = exp + 1;
                return 1 / denominator;
            }
            finally
            {
                if (owned)
                    v.Dispose();
            }
        }

        /// <summary>
        ///     Index of the maximum along <paramref name="axis"/> (int64), the first one on ties: the class id of a
        ///     classifier, the answer span ends of a question-answering head. The contract is that of ONNX
        ///     <c>ArgMax</c> with <c>select_last_index = 0</c>.
        /// </summary>
        /// <param name="scores">The scores; any layout and any dtype <c>np.argmax</c> takes.</param>
        /// <param name="axis">The axis to search along; a negative value counts from the end.</param>
        /// <param name="keepdims"><c>true</c> to keep <paramref name="axis"/> in the result with length 1.</param>
        /// <returns>A new int64 array of indices; the caller disposes it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="scores"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-ndim, ndim)</c> for an array of one or more dimensions.</exception>
        /// <remarks>
        ///     The axis is checked here and not left to <c>np.argmax</c>: an engine that wraps a too-negative axis
        ///     (<c>-3</c> on a 2-D array read as axis 1) returns indices along the wrong axis with no error, which a
        ///     caller cannot tell from a correct answer.
        /// </remarks>
        public static NDArray Argmax(NDArray scores, int axis = -1, bool keepdims = false)
        {
            if (scores is null)
                throw new ArgumentNullException(nameof(scores));
            int ndim = scores.ndim;
            // A 0-d array is left to np.argmax, which has NumPy's own rule for it (axis 0 and -1 are accepted).
            if (ndim > 0 && (axis < -ndim || axis >= ndim))
                throw new AxisError(axis, ndim);
            return np.argmax(scores, axis, keepdims);
        }

        /// <summary>
        ///     The <paramref name="k"/> largest (or smallest) values along <paramref name="axis"/> and their
        ///     indices, sorted — the ONNX <c>TopK</c> contract: descending for <paramref name="largest"/>,
        ///     ascending otherwise, and on equal values <b>the lower index comes first</b>. Output shape is the
        ///     input shape with <paramref name="axis"/> reduced to <paramref name="k"/>; indices are int64.
        /// </summary>
        /// <param name="scores">Scores, typically a <c>(batch, classes)</c> float array.</param>
        /// <param name="k">How many entries per slice; <c>1 ≤ k ≤ scores.shape[axis]</c>.</param>
        /// <param name="axis">Axis to rank along (negative counts from the end).</param>
        /// <param name="largest"><c>true</c> for the k largest, <c>false</c> for the k smallest.</param>
        /// <param name="sorted">Accepted for parity with ONNX <c>TopK</c>; the result is always sorted (an unsorted result is a valid sorted one).</param>
        /// <returns>The values (in the input's dtype) and their int64 indices, each a new array the caller disposes.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="scores"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="scores"/> is 0-d.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is out of range, or <paramref name="k"/> is not in <c>[1, scores.shape[axis]]</c>.</exception>
        public static (NDArray values, NDArray indices) TopK(NDArray scores, int k, int axis = -1, bool largest = true, bool sorted = true)
        {
            if (scores is null)
                throw new ArgumentNullException(nameof(scores));
            int ndim = scores.ndim;
            if (ndim == 0)
                throw new ArgumentException("TopK needs at least a 1-D array.", nameof(scores));
            int ax = axis < 0 ? axis + ndim : axis;
            if (ax < 0 || ax >= ndim)
                throw new ArgumentOutOfRangeException(nameof(axis), $"axis {axis} is out of bounds for array of dimension {ndim}");
            long n = scores.shape[ax];
            if (k < 1 || k > n)
                throw new ArgumentOutOfRangeException(nameof(k), $"k must be in [1, {n}] for an axis of length {n}, got {k}.");

            NDArray order;
            if (largest)
            {
                // Descending with ties keeping the LOWER index first == the reverse of a stable ascending argsort
                // taken over the axis-REVERSED array (whose ties come out higher-index-first), mapped back through
                // j -> n-1-j. A stable argsort of the negated values would need a signed, overflow-free negation for
                // every dtype; this form is exact for all of them.
                using NDArray reversed = np.flip(scores, ax);
                using NDArray ascendingOverReversed = np.argsort(reversed, ax);
                using NDArray mapped = (n - 1) - ascendingOverReversed;
                order = np.flip(mapped, ax);
            }
            else
            {
                order = np.argsort(scores, ax);   // stable: ties lower-index-first, ascending
            }

            using (order)
            {
                var window = new Slice[ndim];
                for (int i = 0; i < ndim; i++)
                    window[i] = i == ax ? new Slice(0, k) : Slice.All;
                using NDArray topView = order[window];
                NDArray indices = topView.copy();   // an owning, C-contiguous int64 array (not a view into `order`)
                NDArray values = np.take_along_axis(scores, indices, ax);
                return (values, indices);
            }
        }

        /// <summary>
        ///     The array the exponential-based methods compute on: the input itself for a float, decimal or complex
        ///     dtype, and a float copy for a bool or integer dtype, in the float dtype NumPy's <c>exp</c> loop selects
        ///     for it. Converting first is what keeps <c>x - max</c> and <c>-x</c> from wrapping in the integer dtype.
        /// </summary>
        /// <param name="x">The caller's array.</param>
        /// <param name="owned"><c>true</c> when the result is a new array that the caller of this helper must dispose; <c>false</c> when it is <paramref name="x"/> itself.</param>
        /// <returns><paramref name="x"/>, or its float copy.</returns>
        /// <remarks>
        ///     The tiers are NumPy's (<c>np.exp(np.int8(1)).dtype</c> is float16): every value of the narrow integer
        ///     dtypes is exact in its tier (float16 holds integers up to 2048, float32 up to 2^24). Only int64 and
        ///     uint64 values past 2^53 round, as they do in <c>np.exp</c> itself.
        /// </remarks>
        private static NDArray PromoteForExp(NDArray x, out bool owned)
        {
            NPTypeCode tier;
            switch (x.typecode)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                    tier = NPTypeCode.Half;
                    break;
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    tier = NPTypeCode.Single;
                    break;
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    tier = NPTypeCode.Double;
                    break;
                default:
                    // float16 / float32 / float64, decimal and complex compute in their own dtype: nothing wraps there.
                    owned = false;
                    return x;
            }

            owned = true;
            return x.astype(tier);
        }

        /// <summary>
        ///     Does the dtype have infinities and NaN that <c>np.isfinite</c> tests element by element? Only the three
        ///     IEEE float dtypes are asked: decimal has neither value, and a complex maximum is left as NumPy orders it.
        /// </summary>
        /// <param name="code">The dtype the computation runs in.</param>
        /// <returns><c>true</c> for float16, float32 and float64.</returns>
        private static bool HasNonFiniteValues(NPTypeCode code) =>
            code == NPTypeCode.Half || code == NPTypeCode.Single || code == NPTypeCode.Double;
    }
}
