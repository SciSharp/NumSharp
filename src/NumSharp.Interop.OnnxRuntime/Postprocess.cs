using System;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     The output side of inference, as NumSharp compositions instead of hand-written loops: the
    ///     softmax / argmax / top-k every classification tutorial re-implements with <c>Math.Exp</c>,
    ///     <c>OrderByDescending().Take(k)</c> and a <c>GetMaxValueIndex</c> loop. Nothing here is a new kernel —
    ///     each method is a few <c>np.*</c> calls over the NDArray a model output became
    ///     (<see cref="NDArrayOnnxInterop.ToNDArray(Microsoft.ML.OnnxRuntime.OrtValue)"/> /
    ///     <see cref="NDArrayOnnxInterop.AsNDArray(Microsoft.ML.OnnxRuntime.OrtValue, bool)"/>), kept in a
    ///     separate type so the conversion verbs stay conversion-only.
    ///
    ///     <para>Dtypes follow NumPy's ufunc rules: float inputs keep their precision (a float32 logit vector
    ///     gives a float32 softmax), integer inputs land on NumPy's float tier for <c>exp</c>
    ///     (int8 → float16, int16 → float32, int32+ → float64); feed float32/float64 logits for predictable
    ///     results. Values agree with ORT's own <c>Softmax</c> / <c>ArgMax</c> / <c>TopK</c> operators to floating
    ///     point tolerance (the tests run both on the same data).</para>
    /// </summary>
    public static class Postprocess
    {
        /// <summary>
        ///     Numerically stable softmax along <paramref name="axis"/>: <c>exp(x - max) / sum(exp(x - max))</c>,
        ///     the standard <c>scipy.special.softmax</c> formulation (subtracting the row max cannot change the
        ///     result and prevents overflow). Rows sum to 1.
        /// </summary>
        public static NDArray Softmax(NDArray logits, int axis = -1)
        {
            if (logits is null)
                throw new ArgumentNullException(nameof(logits));
            using NDArray max = np.max(logits, axis, keepdims: true);
            using NDArray shifted = logits - max;
            using NDArray exp = np.exp(shifted);
            using NDArray sum = np.sum(exp, axis, keepdims: true);
            return exp / sum;
        }

        /// <summary>
        ///     Log-softmax along <paramref name="axis"/>: <c>x - max - log(sum(exp(x - max)))</c> — what
        ///     cross-entropy / NLL post-processing wants, without the underflow of <c>log(softmax(x))</c>.
        /// </summary>
        public static NDArray LogSoftmax(NDArray logits, int axis = -1)
        {
            if (logits is null)
                throw new ArgumentNullException(nameof(logits));
            using NDArray max = np.max(logits, axis, keepdims: true);
            using NDArray shifted = logits - max;
            using NDArray exp = np.exp(shifted);
            using NDArray sum = np.sum(exp, axis, keepdims: true);
            using NDArray logSum = np.log(sum);
            return shifted - logSum;
        }

        /// <summary>Element-wise logistic sigmoid <c>1 / (1 + exp(-x))</c> — multi-label heads and detection confidences.</summary>
        public static NDArray Sigmoid(NDArray x)
        {
            if (x is null)
                throw new ArgumentNullException(nameof(x));
            using NDArray negated = -x;
            using NDArray exp = np.exp(negated);
            using NDArray denominator = exp + 1;
            return 1 / denominator;
        }

        /// <summary>
        ///     Index of the maximum along <paramref name="axis"/> (int64), first occurrence on ties — the same
        ///     contract as ONNX <c>ArgMax</c> with <c>select_last_index=0</c>: the class id of a classifier, the
        ///     answer span ends of a BERT QA head.
        /// </summary>
        public static NDArray Argmax(NDArray scores, int axis = -1, bool keepdims = false)
        {
            if (scores is null)
                throw new ArgumentNullException(nameof(scores));
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
    }
}
