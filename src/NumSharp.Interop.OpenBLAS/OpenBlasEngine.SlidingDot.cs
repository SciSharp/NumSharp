using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     The byte-parity dot behind <c>np.correlate</c> / <c>np.convolve</c> — NumSharp's
    ///     <see cref="Backends.ISlidingDotBackend"/> served by the SAME per-dtype <c>@name@_dot</c>
    ///     (<see cref="IBlasType{T}.Dot"/>) the product family uses.
    /// </summary>
    /// <remarks>
    ///     NumPy's <c>_pyarray_correlate</c> reduces every ramp position — and the middle whenever
    ///     <c>small_correlate</c> declines (a real kernel longer than 11, or any complex kernel) —
    ///     with its <c>dotfunc</c>, which for float32/float64/complex128 is the chunked, double-
    ///     accumulated cblas <c>?dot</c> (<c>?dotu</c> for complex) implemented in
    ///     <see cref="SingleBlas.Dot"/> / <see cref="DoubleBlas.Dot"/> / <see cref="ComplexBlas.Dot"/>.
    ///     Reusing exactly that here is what makes the sliding-dot family byte-identical to NumPy on
    ///     the long float kernels the managed reduction reorders — no new native code, just the same
    ///     primitive exposed through a second seam.
    /// </remarks>
    public static unsafe partial class OpenBlasEngine
    {
        /// <summary>
        ///     Whether the byte-parity sliding-dot is available for <paramref name="dtype"/> — the same
        ///     rule as the products (<see cref="IsSupported"/>): Single/Double whenever a CBLAS is
        ///     loaded, Complex only when the loaded library also exports the complex products.
        /// </summary>
        internal static bool SupportsSlidingDot(NPTypeCode dtype) => IsSupported(dtype);

        /// <summary>
        ///     NumPy's <c>@name@_dot</c> for the sliding-dot family:
        ///     <c>*result = Σ a[i·strideA]·b[i·strideB]</c>, summed the way NumPy sums it. UNCONJUGATED
        ///     for complex (<c>?dotu</c>) — <c>np.convolve</c> never conjugates, and <c>np.correlate</c>
        ///     has already conjugated its kernel before reaching the engine.
        /// </summary>
        internal static void SlidingDot(NPTypeCode dtype,
            void* a, long strideA, void* b, long strideB, void* result, long count)
        {
            switch (dtype)
            {
                case NPTypeCode.Single:
                    default(SingleBlas).Dot((float*)a, strideA, (float*)b, strideB, (float*)result, count);
                    break;
                case NPTypeCode.Double:
                    default(DoubleBlas).Dot((double*)a, strideA, (double*)b, strideB, (double*)result, count);
                    break;
                case NPTypeCode.Complex:
                    default(ComplexBlas).Dot((Complex*)a, strideA, (Complex*)b, strideB, (Complex*)result, count);
                    break;
                default:
                    throw new NotSupportedException(
                        $"OpenBlasEngine.SlidingDot: dtype {dtype} is not a cblas dot dtype " +
                        "(only Single/Double/Complex route through cblas ?dot in NumPy's dotfunc).");
            }
        }

        /// <summary>
        ///     The fully-overlapping MIDDLE of a sliding correlate as ONE call:
        ///     <c>result[i] = @name@_dot(a + i, 1, b, 1, n2)</c> for <c>i</c> in <c>[0, count)</c>.
        ///     The dtype is dispatched ONCE here, then the per-position loop calls the concrete
        ///     <c>@name@Blas.Dot</c> directly (no <see cref="Backends.ISlidingDotBackend"/> virtual
        ///     dispatch, no re-switch) — the hot path for <c>np.correlate</c>/<c>np.convolve</c>. The
        ///     inner <c>ops.Dot</c> is NumPy's <c>@name@_dot</c> exactly, so this is bit-identical to
        ///     the per-position <see cref="SlidingDot"/> route the ramps still use; only the dispatch
        ///     is hoisted out of the ~100K-iteration loop.
        /// </summary>
        internal static void SlidingDotBatch(NPTypeCode dtype,
            void* a, void* b, void* result, long count, long n2)
        {
            switch (dtype)
            {
                case NPTypeCode.Single:
                {
                    float* pa = (float*)a, pk = (float*)b, po = (float*)result;
                    var ops = default(SingleBlas);
                    for (long i = 0; i < count; i++)
                        ops.Dot(pa + i, 1, pk, 1, po + i, n2);
                    break;
                }
                case NPTypeCode.Double:
                {
                    double* pa = (double*)a, pk = (double*)b, po = (double*)result;
                    var ops = default(DoubleBlas);
                    for (long i = 0; i < count; i++)
                        ops.Dot(pa + i, 1, pk, 1, po + i, n2);
                    break;
                }
                case NPTypeCode.Complex:
                {
                    Complex* pa = (Complex*)a, pk = (Complex*)b, po = (Complex*)result;
                    var ops = default(ComplexBlas);
                    for (long i = 0; i < count; i++)
                        ops.Dot(pa + i, 1, pk, 1, po + i, n2);
                    break;
                }
                default:
                    throw new NotSupportedException(
                        $"OpenBlasEngine.SlidingDotBatch: dtype {dtype} is not a cblas dot dtype " +
                        "(only Single/Double/Complex route through cblas ?dot in NumPy's dotfunc).");
            }
        }
    }
}
