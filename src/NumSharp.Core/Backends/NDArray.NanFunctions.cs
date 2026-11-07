using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    // ---------------------------------------------------------------------------------------------------------------------
    // The array object's statements of numpy/lib/_nanfunctions_impl.py:
    //   * np.nanmax / np.nanmin's class test. NumPy's nanmax asks `type(a) is np.ndarray` and sends every other class down
    //     its branch "slow, but safe for subclasses of ndarray"; NumSharp asks the operand's class through C# inheritance:
    //     NanMax / NanMin below are virtual, an ndarray's is np.fmax.reduce / np.fmin.reduce (the engine's NanMax /
    //     NanMin), and NDMatrix / NDCharArray override them with NumPy's branch for their class.
    //   * _replace_nan (ReplaceNaN), whose dtype test (only an inexact array has a NaN mask) is the one decision of that
    //     branch, kept here below the facade (the np-layer dtype plan's ratchet R1). The engine's nan functions run their
    //     own copies over plain arrays (DefaultEngine's ReplaceNanForScan, the fused fills of the nan reductions).
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>_replace_nan(a, val)</c>: for an inexact array (float16, float32, float64, complex128) the mask
        ///     <c>np.isnan(a)</c>, a K-order copy (<c>np.array(a, subok=True, copy=True)</c>) and <c>np.copyto(copy, val,
        ///     where=mask)</c>; for any other dtype no mask and the array itself. Use it where a nan function's route needs
        ///     NumPy's replaced copy AND its mask (np.nanmax / np.nanmin of a numpy.matrix: <see cref="NDMatrix"/>'s
        ///     <see cref="NanMax"/> override); the engine's nan reductions fuse the replacement into their own passes
        ///     instead.
        /// </summary>
        /// <param name="a">The array; any class (the copy is the plain array the engine members make).</param>
        /// <param name="val">The replacement, a Python float (<c>-inf</c> for nanmax, <c>+inf</c> for nanmin), cast into
        ///     the copy under same_kind (a complex copy gets <c>val+0j</c>).</param>
        /// <param name="mask">The NaN mask (a new plain bool array of <paramref name="a"/>'s shape), or <c>null</c> for a
        ///     dtype that holds no NaN (the integers, bool, char, decimal, text): NumPy's <c>mask is None</c>.</param>
        /// <returns>The new replaced copy, or <paramref name="a"/> itself when <paramref name="mask"/> is <c>null</c>.
        ///     Footgun: the caller owns the copy and the mask (dispose them, or call from an <c>[NDScoped]</c>
        ///     member).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>NumPy also masks an object array (<c>np.not_equal(a, a)</c>); NumSharp has no object dtype.</remarks>
        internal static NDArray ReplaceNaN(NDArray a, double val, out NDArray mask)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    // mask = np.isnan(a); a = np.array(a, subok=True, copy=True); np.copyto(a, val, where=mask) - the
                    // engine's isnan and the array object's K-order copy, so a matrix operand gives plain arrays (the
                    // caller decides the class of what it builds from them).
                    mask = a.TensorEngine.IsNan(a);
                    NDArray copy = a.copy('K');
                    UnmanagedStorage.CopytoPythonScalar(copy, val, PyScalarKind.Float, NPY_CASTING.NPY_SAME_KIND_CASTING, mask);
                    return copy;
                default:
                    // Integer, bool, char, decimal and text arrays cannot hold a NaN: NumPy's mask is None.
                    mask = null;
                    return a;
            }
        }

        /// <summary>
        ///     np.nanmax for this array's class: NumPy's nanmax tests <c>type(a) is np.ndarray</c>, and a base ndarray is
        ///     <c>np.fmax.reduce(a, axis=axis, out=out, **kwargs)</c> - the engine's <see cref="TensorEngine.NanMax"/>, which
        ///     reproduces NumPy's fmax schedules bit for bit. A subclass overrides this member with NumPy's branch for every
        ///     other class (<see cref="NDMatrix"/>: <c>_replace_nan</c>, its own <c>max</c>, <c>np.nan</c> written back;
        ///     <see cref="NDCharArray"/>: <c>np.amax</c> of itself).
        /// </summary>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">NumPy's <c>out=</c>: the array that receives the result, or <c>null</c>.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="initial">NumPy's <c>initial=</c>, or <c>null</c>; required with <paramref name="where"/>.</param>
        /// <param name="where">NumPy's <c>where=</c> bool mask, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given (the caller's object, a matrix one included); otherwise a fresh
        ///     array, a read-only 0-d one for a flat reduction without keepdims.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="ArgumentException">An empty reduction without <paramref name="initial"/> (<c>zero-size array to
        ///     reduction operation fmax which has no identity</c>); a non-bool mask.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, a mask without
        ///     <paramref name="initial"/>, an <paramref name="initial"/> the loop cannot hold.</exception>
        /// <exception cref="OverflowException">An integer <paramref name="initial"/> past the loop's range.</exception>
        /// <exception cref="TypeError">Text, or a complex <paramref name="initial"/> for a real loop.</exception>
        /// <exception cref="IncorrectShapeException">The mask or <paramref name="out"/> does not broadcast.</exception>
        /// <remarks>
        ///     <c>[SubclassAware]</c>: np.nanmax hands its operand straight to this member, so the subclass weave leaves the
        ///     function alone and the override of the operand's class answers (C# virtual dispatch is NumPy's class test).
        ///     Footgun: an array subclass that does not override it is computed as a base ndarray, where NumPy would take its
        ///     subclass branch. The engine meets plain arrays only: a subclass <paramref name="out"/> or
        ///     <paramref name="where"/> beside this plain operand goes in as its plain view, and the reduction's hand-back of
        ///     that view returns the caller's object, as <c>np.fmax.reduce(a, out=m)</c> returns <c>m</c>.
        /// </remarks>
        [SubclassAware]
        internal virtual NDArray NanMax(int? axis, NDArray @out, bool keepdims, object initial, NDArray @where)
            => FmaxFminReduce(axis, @out, keepdims, initial, @where, isMax: true);

        /// <summary>
        ///     np.nanmin for this array's class: a base ndarray is <c>np.fmin.reduce(a, axis=axis, out=out, **kwargs)</c> (the
        ///     engine's <see cref="TensorEngine.NanMin"/>); a subclass overrides it with NumPy's branch for every other class,
        ///     as for <see cref="NanMax"/>.
        /// </summary>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">NumPy's <c>out=</c>: the array that receives the result, or <c>null</c>.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="initial">NumPy's <c>initial=</c>, or <c>null</c>; required with <paramref name="where"/>.</param>
        /// <param name="where">NumPy's <c>where=</c> bool mask, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh array, a read-only 0-d one for a flat reduction
        ///     without keepdims.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="ArgumentException">An empty reduction without <paramref name="initial"/> (<c>zero-size array to
        ///     reduction operation fmin which has no identity</c>); a non-bool mask.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, a mask without
        ///     <paramref name="initial"/>, an <paramref name="initial"/> the loop cannot hold.</exception>
        /// <exception cref="OverflowException">An integer <paramref name="initial"/> past the loop's range.</exception>
        /// <exception cref="TypeError">Text, or a complex <paramref name="initial"/> for a real loop.</exception>
        /// <exception cref="IncorrectShapeException">The mask or <paramref name="out"/> does not broadcast.</exception>
        /// <remarks><c>[SubclassAware]</c> for the reason of <see cref="NanMax"/>.</remarks>
        [SubclassAware]
        internal virtual NDArray NanMin(int? axis, NDArray @out, bool keepdims, object initial, NDArray @where)
            => FmaxFminReduce(axis, @out, keepdims, initial, @where, isMax: false);

        /// <summary>
        ///     The base ndarray's branch of <see cref="NanMax"/> / <see cref="NanMin"/>: <c>np.fmax.reduce</c> /
        ///     <c>np.fmin.reduce</c> of this array, the engine's whole call, over plain views of a subclass out= / where=.
        /// </summary>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="out">The caller's <c>out=</c>, or <c>null</c>.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="initial">The caller's <c>initial=</c>, or <c>null</c>.</param>
        /// <param name="where">The caller's <c>where=</c>, or <c>null</c>.</param>
        /// <param name="isMax"><c>true</c> for fmax (np.nanmax), <c>false</c> for fmin (np.nanmin).</param>
        /// <returns>The caller's <paramref name="out"/> when the reduction wrote into it, else the engine's result.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="ArgumentException">An empty reduction without <paramref name="initial"/>; a non-bool mask.</exception>
        /// <exception cref="ValueError">A read-only or misshapen out, a mask without an initial value, an initial value the
        ///     loop cannot hold.</exception>
        /// <exception cref="OverflowException">An integer initial value past the loop's range.</exception>
        /// <exception cref="TypeError">Text, or a complex initial value for a real loop.</exception>
        /// <exception cref="IncorrectShapeException">The mask or the out does not broadcast.</exception>
        private NDArray FmaxFminReduce(int? axis, NDArray @out, bool keepdims, object initial, NDArray @where, bool isMax)
        {
            // The engine computes on plain arrays: a matrix or chararray out= / where= goes in as its plain view, which the
            // lease releases after the call (a plain operand is handed through and never disposed).
            using var outLease = NDSubclass.LeasePlainView(@out);
            using var whereLease = NDSubclass.LeasePlainView(@where);
            NDArray result = isMax
                ? TensorEngine.NanMax(this, axis, keepdims, outLease.Array, initial, whereLease.Array)
                : TensorEngine.NanMin(this, axis, keepdims, outLease.Array, initial, whereLease.Array);
            // A ufunc reduction hands its out back: the caller's object, never the plain view the lease is about to release.
            return @out is not null && ReferenceEquals(result, outLease.Array) ? @out : result;
        }
    }
}
