// TensorEngine's core file: the Blas seam, the Linalg and Fft sub-engines (numpy.linalg, LinalgEngine; numpy.fft,
// FftEngine) and NumSharp's own np.evaluate entry points. The members NumPy names live in partial files named after
// NumPy's sources:
// TensorEngine.Umath.cs (the ufuncs), TensorEngine.Methods.cs (the ndarray methods that compute),
// TensorEngine.Multiarray.cs (the C functions that compute), TensorEngine.Numeric.cs, TensorEngine.NanFunctions.cs,
// TensorEngine.UfuncLike.cs, TensorEngine.LinearAlgebra.cs, and TensorEngine.ArrayObject.cs (the array-object members
// shipped in 0.70.0). The rule that places a function is docs/plans/np-layer-dtype-logic.md, section 1 "Which O" and
// Appendix D.

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     An external BLAS this engine delegates its matrix products to, or null (the default)
        ///     to compute them with NumSharp's own managed SIMD kernels.
        /// </summary>
        /// <remarks>
        ///     NumSharp.Core is 100 % managed C# and never assigns this. It is set by an optional
        ///     package — <c>NumSharp.Interop.OpenBLAS</c> does it from a <c>[ModuleInitializer]</c>, so
        ///     referencing that package is the whole opt-in — and can be cleared again by assigning
        ///     null. A backend only answers for what it implements (see
        ///     <see cref="IBlasBackend"/>); everything else falls back to the managed kernels.
        ///     <para>
        ///     Engines are cached singletons (<see cref="BackendFactory"/>), so assigning this
        ///     affects every array that resolves to the engine — including ones already created.
        ///     </para>
        /// </remarks>
        public IBlasBackend Blas { get; set; }

        /// <summary>
        ///     numpy.linalg's functions for the arrays bound to this engine: every <c>np.linalg</c> function applies its
        ///     class boundary and calls one member of this sub-engine. <see cref="DefaultLinalgEngine"/> (NumPy's
        ///     <c>_linalg.py</c> statements over kernels that ask <see cref="Blas"/> first) until something else is
        ///     assigned.
        /// </summary>
        /// <value>The assigned engine, or the shared <see cref="DefaultLinalgEngine"/>; never null. Assigning null
        ///     restores the default.</value>
        /// <remarks>
        ///     <para>
        ///     <b>You rarely need this.</b> A LAPACK backend reaches the factorisations through <see cref="Blas"/> already
        ///     (the <c>Try*</c> members of <see cref="IBlasBackend"/>), which is how <c>NumSharp.Interop.OpenBLAS</c> serves
        ///     them. Assign a <see cref="DefaultLinalgEngine"/> subclass to compute a factorisation another way while NumPy's
        ///     argument checks, dtype rules and result layouts stay (override its <c>protected virtual</c> kernels), or a
        ///     <see cref="LinalgEngine"/> of your own to replace whole functions.
        ///     </para>
        ///     <para>
        ///     Engines are cached singletons (<see cref="BackendFactory"/>), so an assignment reaches every array bound to
        ///     the engine, existing ones included, from the next <c>np.linalg</c> call on: each call reads the property
        ///     once, so one already running finishes with the engine it read. Do not assign it while another thread runs
        ///     np.linalg functions whose results you compare: they may straddle the change.
        ///     </para>
        /// </remarks>
        public LinalgEngine Linalg
        {
            get => _linalg ?? DefaultLinalgEngine.Shared;
            set => _linalg = value;
        }

        /// <summary>The engine assigned to <see cref="Linalg"/>, or null for the shared default.</summary>
        private LinalgEngine _linalg;

        /// <summary>
        ///     numpy.fft's functions for the arrays bound to this engine: every <c>np.fft</c> transform and frequency helper
        ///     calls one member of this sub-engine (<c>fftfreq</c> / <c>rfftfreq</c>, which take no array, through the
        ///     default engine). <see cref="DefaultFftEngine"/> (NumPy's <c>_pocketfft.py</c> statements over kernels that run
        ///     NumSharp's managed pocketfft port) until something else is assigned.
        /// </summary>
        /// <value>The assigned engine, or the shared <see cref="DefaultFftEngine"/>; never null. Assigning null restores the
        ///     default.</value>
        /// <remarks>
        ///     <para>
        ///     Assign a <see cref="DefaultFftEngine"/> subclass to compute the transforms another way (MKL, FFTW, cuFFT)
        ///     while NumPy's length, norm, axis and <c>out=</c> checks and the N-D, real and Hermitian compositions stay
        ///     (override its <c>protected virtual</c> kernels), or an <see cref="FftEngine"/> of your own to replace whole
        ///     functions. <c>fftshift</c> and <c>ifftshift</c> are not members: they roll the array object.
        ///     </para>
        ///     <para>
        ///     Engines are cached singletons (<see cref="BackendFactory"/>), so an assignment reaches every array bound to
        ///     the engine, existing ones included, from the next <c>np.fft</c> call on: each call reads the property once,
        ///     so one already running finishes with the engine it read. Do not assign it while another thread runs np.fft
        ///     functions whose results you compare: they may straddle the change.
        ///     </para>
        /// </remarks>
        public FftEngine Fft
        {
            get => _fft ?? DefaultFftEngine.Shared;
            set => _fft = value;
        }

        /// <summary>The engine assigned to <see cref="Fft"/>, or null for the shared default.</summary>
        private FftEngine _fft;

        /// <summary>
        ///     Fused evaluation of an <see cref="NDExpr"/> tree in one iterator
        ///     pass (np.evaluate, roadmap Wave 6.1). Virtual with a
        ///     NotSupported default so alternative engines opt in explicitly.
        /// </summary>
        /// <param name="options">Resolved <c>dtype=</c>/<c>casting=</c>/<c>order=</c> keywords (plan P4.5); <c>default</c> = no override.</param>
        public virtual NDArray Evaluate(NDExpr expr, NDArray @out = null, in NDEvaluateOptions options = default)
            => throw new NotSupportedException($"{GetType().Name} does not support fused expression evaluation.");

        /// <summary>
        ///     Fused evaluation against an explicit operand list
        ///     (<see cref="NDExpr.Input"/> leaves reference operands by position).
        /// </summary>
        /// <param name="options">Resolved <c>dtype=</c>/<c>casting=</c>/<c>order=</c> keywords (plan P4.5); <c>default</c> = no override.</param>
        public virtual NDArray Evaluate(NDExpr expr, NDArray[] operands, NDArray @out = null, in NDEvaluateOptions options = default)
            => throw new NotSupportedException($"{GetType().Name} does not support fused expression evaluation.");

        /// <summary>
        ///     Fused evaluation of an already-compiled program (<see cref="CompiledExpression"/>):
        ///     the per-call work only — iteration shape, result allocation, iterator. The two
        ///     <c>Evaluate(NDExpr, …)</c> members above resolve the program from the tree's cache and
        ///     land here.
        /// </summary>
        /// <param name="options">Resolved <c>dtype=</c>/<c>casting=</c>/<c>order=</c> keywords (plan P4.5); <c>default</c> = no override.</param>
        internal virtual NDArray Evaluate(NDExprProgram program, NDArray[] operands, NDArray @out, in NDEvaluateOptions options = default)
            => throw new NotSupportedException($"{GetType().Name} does not support fused expression evaluation.");
    }
}
