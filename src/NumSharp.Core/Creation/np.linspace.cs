using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The result of <see cref="linspace(object, object, long, bool, bool, DType, int, string)"/>: NumPy's
        ///     <c>samples</c>-or-<c>(samples, step)</c> return as one type. It converts implicitly to <see cref="NDArray"/>
        ///     (the samples: <c>NDArray y = np.linspace(a, b, 5, retstep: false);</c>) and deconstructs into both
        ///     (<c>var (y, step) = np.linspace(0, 1, 5, retstep: true);</c>).
        /// </summary>
        /// <remarks>
        ///     Only the <c>object</c> overload, the one that takes <c>retstep</c>, returns it; every call without
        ///     <c>retstep</c> binds <see cref="linspace(NDArray, NDArray, long, bool, DType, int, string)"/>, which returns the
        ///     samples as an <see cref="NDArray"/>. Footgun: <c>var r = np.linspace(..., retstep: true)</c> is this struct, so
        ///     reach the array through <see cref="samples"/> (or deconstruct) before calling NDArray members on it.
        /// </remarks>
        public readonly struct LinspaceResult : INDArrayCarrier
        {
            /// <summary>Creates the result of one linspace call.</summary>
            /// <param name="samples">The samples.</param>
            /// <param name="step">NumPy's step when <c>retstep</c> was requested, otherwise null.</param>
            internal LinspaceResult(NDArray samples, NDArray step)
            {
                this.samples = samples;
                this.step = step;
            }

            /// <summary>The samples, exactly as the NDArray overload returns them.</summary>
            public NDArray samples { get; }

            /// <summary>
            ///     NumPy's step, <c>(stop - start) / div</c> in the computation dtype (div is <c>num - 1</c> with the endpoint,
            ///     <c>num</c> without): a 0-d array for scalar ends (NumPy's scalar, <c>np.float64(0.25)</c>), an array of the
            ///     ends' broadcast shape for array ends, and a 0-d NaN Python float when there is no step (no samples, or one
            ///     with the endpoint). Null when <c>retstep</c> was false.
            /// </summary>
            public NDArray step { get; }

            /// <summary>The samples: NumPy's bare-array return (<c>retstep=False</c>).</summary>
            /// <param name="result">The linspace result.</param>
            /// <returns><see cref="samples"/>.</returns>
            public static implicit operator NDArray(LinspaceResult result) => result.samples;

            /// <summary>Deconstructs NumPy's <c>(samples, step)</c> tuple: <c>var (y, step) = np.linspace(0, 1, 5, retstep: true);</c></summary>
            /// <param name="samples">Receives the samples.</param>
            /// <param name="step">Receives the step.</param>
            /// <exception cref="InvalidOperationException">The call had <c>retstep</c> false, so NumPy returned the samples
            ///     alone (unpacking them into two names would be Python's own error).</exception>
            public void Deconstruct(out NDArray samples, out NDArray step)
            {
                if (this.step is null)
                    throw new InvalidOperationException(
                        "np.linspace returned the samples alone (retstep=False); pass retstep: true to deconstruct (samples, step).");
                samples = this.samples;
                step = this.step;
            }

            /// <summary>Hands both arrays to an enclosing <see cref="NDScope"/> as results, so it does not reclaim them.</summary>
            /// <param name="scope">The scope the result leaves.</param>
            void INDArrayCarrier.YieldTo(NDScope scope)
            {
                scope.Returns(samples);
                scope.Returns(step);
            }
        }

        /// <summary>
        ///     Return evenly spaced numbers over a specified interval: <paramref name="num"/> samples from
        ///     <paramref name="start"/> to <paramref name="stop"/>, NumPy's <c>np.linspace</c>. The ends may be numbers or
        ///     arrays; array ends give one ramp per element, along a new axis placed by <paramref name="axis"/>.
        /// </summary>
        /// <param name="start">The starting value of the sequence: a number or an array. A C# number is a Python number,
        ///     WEAK under NEP 50 (it takes the other end's dtype); <c>NDArray.Scalar(0.1f)</c> or a float32 array is a NumPy
        ///     float32, so two of them compute in float32 (NumPy's bits, which differ from float64 samples cast to
        ///     float32).</param>
        /// <param name="stop">The end value of the sequence, unless <paramref name="endpoint"/> is false; then the sequence
        ///     is all but the last of <c>num + 1</c> evenly spaced samples, and the step changes. Broadcast against
        ///     <paramref name="start"/>.</param>
        /// <param name="num">Number of samples to generate. Default is 50. Must be non-negative.</param>
        /// <param name="endpoint">If true, <paramref name="stop"/> is the last sample (written exactly). Default is true.</param>
        /// <param name="dtype">The type of the output array: a C# <see cref="Type"/>, an <see cref="NPTypeCode"/>, a NumPy
        ///     dtype string (<c>"f4"</c>) or a <see cref="DType"/>. Null (the default) keeps the computation dtype,
        ///     <c>result_type(start, stop, 1.0)</c>: float64 for numbers and integer arrays, float32 for float32 ends, complex
        ///     for complex ends. An integer dtype floors the samples first; a text dtype formats them.</param>
        /// <param name="axis">The axis in the result to store the samples (relevant only for array ends): 0 puts them first,
        ///     -1 last.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>The samples, shape <c>(num, *broadcast(start, stop).shape)</c> with the sample axis moved to
        ///     <paramref name="axis"/>. A <c>numpy.matrix</c> end gives a matrix, as NumPy's <c>_array_converter.wrap</c>
        ///     does.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="start"/> or <paramref name="stop"/> is null.</exception>
        /// <exception cref="ValueError">A negative <paramref name="num"/> (<c>Number of samples, -1, must be
        ///     non-negative.</c>), checked first, as NumPy checks it; a <paramref name="device"/> other than <c>"cpu"</c>
        ///     (<c>Device not understood. Only "cpu" is allowed, but received: ...</c>).</exception>
        /// <exception cref="DTypePromotionError">Ends with no inexact common dtype (text ends).</exception>
        /// <exception cref="IncorrectShapeException">Array ends that do not broadcast (NumPy's ValueError text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range for the result (<c>destination: axis 3 is out of
        ///     bounds for array of dimension 2</c>).</exception>
        /// <exception cref="TypeError">An integer <paramref name="dtype"/> over complex ends.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        ///     <para>NumPy's <c>retstep=True</c> is the overload that takes <c>object</c> ends,
        ///     <see cref="linspace(object, object, long, bool, bool, DType, int, string)"/>: a call that names
        ///     <c>retstep</c> binds it, every other call binds this one.</para>
        ///     <para>Computed by the start operand's engine (<see cref="TensorEngine.Linspace(NDArray, NDArray, long, bool, DType, int, string)"/>).</para>
        /// </remarks>
        public static NDArray linspace(NDArray start, NDArray stop, long num = 50, bool endpoint = true, DType dtype = null,
            int axis = 0, string device = null)
        {
            // C#'s convention for a missing operand, before an engine is chosen through it.
            if (start is null)
                throw new ArgumentNullException(nameof(start));
            if (stop is null)
                throw new ArgumentNullException(nameof(stop));
            return start.TensorEngine.Linspace(start, stop, num, endpoint, dtype, axis, device);
        }

        /// <summary>
        ///     NumPy's whole <c>np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *,
        ///     device=None)</c>, <c>retstep</c> included: the samples, and with <paramref name="retstep"/> the step between
        ///     them. The ends are any array_like NumPy takes - a number, an <see cref="NDArray"/>, a C# array or an
        ///     <c>object[]</c> list.
        /// </summary>
        /// <param name="start">The starting value: a number (a weak Python number), a NumPy scalar or array, or a list.</param>
        /// <param name="stop">The end value (see <see cref="linspace(NDArray, NDArray, long, bool, DType, int, string)"/>).</param>
        /// <param name="num">Number of samples to generate. Default is 50. Must be non-negative.</param>
        /// <param name="endpoint">If true, <paramref name="stop"/> is the last sample. Default is true.</param>
        /// <param name="retstep">If true, the result also carries the step (<see cref="LinspaceResult.step"/>).</param>
        /// <param name="dtype">The type of the output array, or null for the computation dtype.</param>
        /// <param name="axis">The axis in the result to store the samples (array ends).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted.</param>
        /// <returns>The samples, and the step when <paramref name="retstep"/> is true; it converts implicitly to the samples'
        ///     <see cref="NDArray"/> and deconstructs into <c>(samples, step)</c>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="start"/> or <paramref name="stop"/> is null.</exception>
        /// <exception cref="ValueError">A negative <paramref name="num"/>, or a <paramref name="device"/> other than
        ///     <c>"cpu"</c>; a ragged list end.</exception>
        /// <exception cref="DTypePromotionError">Ends with no inexact common dtype (text ends).</exception>
        /// <exception cref="IncorrectShapeException">Ends that do not broadcast.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range for the result.</exception>
        /// <exception cref="TypeError">An integer <paramref name="dtype"/> over complex ends.</exception>
        /// <exception cref="NotSupportedException">An end of a type NumSharp has no array for.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        ///     <para>C# binds a call to this overload when it names <paramref name="retstep"/> (or passes it fifth,
        ///     <c>np.linspace(0, 1, 5, true, true)</c>), or when an end is typed <c>object</c>: an <see cref="NDArray"/> is a
        ///     better conversion target than <c>object</c>, so every other call binds the NDArray overload, under C# 12 and
        ///     13 alike.</para>
        ///     <para>The step is NumPy's <c>delta / div</c> computed with the public ufuncs, so a <c>numpy.matrix</c> end gives
        ///     a matrix step, as in NumPy; there is no step (a NaN) for no samples or one with the endpoint.</para>
        /// </remarks>
        public static LinspaceResult linspace(object start, object stop, long num = 50, bool endpoint = true, bool retstep = false,
            DType dtype = null, int axis = 0, string device = null)
        {
            if (start is null)
                throw new ArgumentNullException(nameof(start));
            if (stop is null)
                throw new ArgumentNullException(nameof(stop));

            // conv = _array_converter(start, stop): each end becomes the array the NDArray overload would receive for it (a
            // boxed C# number a weak Python number, a list NumPy's coercion of a sequence); an NDArray end is used as given.
            NDArray startArray = NDArray.FromPythonOperand(start);
            NDArray stopArray = null;
            try
            {
                stopArray = NDArray.FromPythonOperand(stop);
                // The samples come from the NDArray overload, whose woven entry applies NumPy's wrap to a subclass end.
                NDArray samples = linspace(startArray, stopArray, num, endpoint, dtype, axis, device);
                if (!retstep)
                    return new LinspaceResult(samples, null);
                NDArray step;
                try
                {
                    step = LinspaceStep(startArray, stopArray, num, endpoint);
                }
                catch
                {
                    samples.Dispose();
                    throw;
                }
                return new LinspaceResult(samples, step);
            }
            finally
            {
                // Only the arrays converted here are released: an NDArray end belongs to the caller.
                if (!ReferenceEquals(startArray, start))
                    startArray.Dispose();
                if (stopArray is not null && !ReferenceEquals(stopArray, stop))
                    stopArray.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's linspace step: <c>step = delta / div</c> with <c>delta = np.subtract(stop, start, dtype=dt)</c> when
        ///     <c>div</c> (<c>num - 1</c> with the endpoint, <c>num</c> without) is positive, else Python's <c>nan</c>. The
        ///     samples were computed (and every error raised) before this runs.
        /// </summary>
        /// <param name="start">The first end, as linspace received it.</param>
        /// <param name="stop">The second end.</param>
        /// <param name="num">The number of samples.</param>
        /// <param name="endpoint">Whether the last sample is the stop.</param>
        /// <returns>The step: 0-d for scalar ends, the broadcast shape for array ends, a weak 0-d NaN without a step.</returns>
        private static NDArray LinspaceStep(NDArray start, NDArray stop, long num, bool endpoint)
        {
            long div = endpoint ? num - 1 : num;
            if (div <= 0)
                return NDArray.PythonLiteral(double.NaN);
            DType dt = DefaultEngine.LinspaceComputeType(start, stop);
            // Public ufuncs, as NumPy's statements are: a subclass end keeps its class through them.
            using NDArray delta = subtract(stop, start, dtype: dt);
            using NDArray divisor = NDArray.PythonLiteral(div);
            return true_divide(delta, divisor);
        }
    }
}
