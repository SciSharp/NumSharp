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
        ///     The engine an array binds when nothing else decides, and the engine of the np.* functions that take no array
        ///     (np.zeros, np.arange, np.linspace, np.fft.fftfreq, the window functions): every array created without an
        ///     engine of its own binds the engine this property returns at that moment. Assign an engine (a
        ///     <see cref="DefaultEngine"/> subclass that overrides some members, typically) to have every array created
        ///     from then on compute on it.
        /// </summary>
        /// <value>The assigned engine, or the built-in <see cref="DefaultEngine"/> (the instance
        ///     <c>BackendFactory.GetEngine&lt;DefaultEngine&gt;()</c> returns); never null. Assigning null restores the
        ///     built-in engine. <see cref="BackendFactory.GetEngine(BackendType)"/> returns this property.</value>
        /// <remarks>
        ///     <para>
        ///     An array binds its engine once, when it is created, so an assignment reaches the arrays created afterwards
        ///     and every array that already exists keeps the engine it bound (assign <see cref="NDArray.TensorEngine"/> to
        ///     move one). An engine binds the results it computes to itself and a view binds its source's engine, so the
        ///     arrays derived from an existing array stay on that array's engine after the default changes.
        ///     </para>
        ///     <para>
        ///     Footgun: <see cref="Blas"/>, <see cref="Linalg"/> and <see cref="Fft"/> belong to each engine instance, and a
        ///     new engine starts with none of what was assigned to the previous default. <c>NumSharp.Interop.OpenBLAS</c>
        ///     installs its backend on the engine that is the default when the package loads, so an engine assigned later
        ///     computes its matrix products with the managed kernels until its own <see cref="Blas"/> is set: copy it over
        ///     (<c>engine.Blas = TensorEngine.Default.Blas;</c>) before the assignment.
        ///     </para>
        ///     <para>
        ///     Process-wide and not synchronized: an array created on another thread while the property changes binds
        ///     either engine. Assign it at startup, or around single-threaded work (tests that assign it must not run in
        ///     parallel with tests that create arrays).
        ///     </para>
        /// </remarks>
        public static TensorEngine Default
        {
            get => _default ?? BackendFactory.BuiltIn;
            set => _default = value;
        }

        /// <summary>The engine assigned to <see cref="Default"/>, or null for the built-in <see cref="DefaultEngine"/>.</summary>
        private static TensorEngine _default;

        /// <summary>
        ///     A new array of <paramref name="dtype"/> and <paramref name="shape"/> bound to this engine: the one allocation
        ///     an engine makes its results with, so the result of a computation stays on the engine that computed it (and
        ///     the next operation on that result computes there too).
        /// </summary>
        /// <param name="dtype">The dtype descriptor of the elements (a <see cref="Type"/>, an <see cref="NPTypeCode"/> or a
        ///     NumPy dtype string converts implicitly).</param>
        /// <param name="shape">The dimensions. A view's layout (strides, offset, broadcast) is laid out again in C order and a
        ///     fresh dense or KEEPORDER layout is kept, as for every allocation (<see cref="Shape.ForFreshAllocation"/>).</param>
        /// <param name="fillZeros">True to zero the buffer; false leaves pool noise the caller must overwrite completely
        ///     before anyone reads it (a fixed-width <c>U</c> array is zeroed either way, NumPy's <c>NPY_NEEDS_INIT</c>).</param>
        /// <returns>The new array, owning its buffer, bound to this engine (<see cref="NDArray.TensorEngine"/> is
        ///     <c>this</c>).</returns>
        /// <exception cref="ValueError">A dimension is negative, or the byte count overflows.</exception>
        /// <remarks>
        ///     The storage comes from this engine's <see cref="GetStorage(DType)"/>, so an engine that overrides it
        ///     allocates its results through its own override. A result built any other way (<c>new NDArray(dtype,
        ///     shape)</c>, <c>NDArray.Scalar</c>, <c>np.zeros</c>) binds <see cref="Default"/> instead: an engine member that
        ///     builds its result so moves it off the engine that computed it whenever that engine is not the default (a
        ///     wrap of a buffer the engine filled names the engine in its initializer: <c>{ TensorEngine = this }</c>).
        /// </remarks>
        protected internal NDArray NewArray(DType dtype, Shape shape, bool fillZeros)
            => new NDArray(dtype, shape, fillZeros, this);

        /// <summary>
        ///     The typed form of <see cref="NewArray(DType, Shape, bool)"/>: a new <see cref="NDArray{T}"/> of
        ///     <paramref name="shape"/> bound to this engine.
        /// </summary>
        /// <typeparam name="T">The element type; its <see cref="NPTypeCode"/> is the dtype.</typeparam>
        /// <param name="shape">The dimensions (a view's layout is laid out again in C order, as for every allocation).</param>
        /// <param name="fillZeros">True to zero the buffer; false leaves pool noise the caller must overwrite completely.</param>
        /// <returns>The new typed array, owning its buffer, bound to this engine.</returns>
        /// <exception cref="ValueError">A dimension is negative, or the byte count overflows.</exception>
        protected internal NDArray<T> NewArray<T>(Shape shape, bool fillZeros) where T : unmanaged
            => new NDArray<T>(shape, fillZeros, this);

        /// <summary>
        ///     <see cref="NDArray.Scalar{T}(T)"/> bound to this engine: the 0-d value an engine member returns (a flat
        ///     reduction, a 0-d operation) or computes with stays on the engine that made it.
        /// </summary>
        /// <typeparam name="T">The element type; its <see cref="NPTypeCode"/> is the dtype.</typeparam>
        /// <param name="value">The value.</param>
        /// <returns>A new writeable 0-d array holding <paramref name="value"/>, bound to this engine.</returns>
        protected internal NDArray NewScalar<T>(T value) where T : unmanaged
            => Bind(NDArray.Scalar(value));

        /// <summary>
        ///     <see cref="NDArray.Scalar{T}(object)"/> bound to this engine: <paramref name="value"/> converted to
        ///     <typeparamref name="T"/> first.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="value">The value, converted with <c>Converts.ChangeType</c> unless it already is a
        ///     <typeparamref name="T"/>.</param>
        /// <returns>A new writeable 0-d array, bound to this engine.</returns>
        /// <exception cref="InvalidCastException"><paramref name="value"/> does not convert to <typeparamref name="T"/>.</exception>
        protected internal NDArray NewScalar<T>(object value) where T : unmanaged
            => Bind(NDArray.Scalar<T>(value));

        /// <summary>
        ///     <see cref="NDArray.Scalar(object)"/> bound to this engine: the dtype is the value's own (its C# type).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>A new writeable 0-d array, bound to this engine.</returns>
        /// <exception cref="NotSupportedException"><paramref name="value"/>'s type is no NumSharp dtype.</exception>
        protected internal NDArray NewScalar(object value)
            => Bind(NDArray.Scalar(value));

        /// <summary>
        ///     <see cref="NDArray.Scalar(object, DType)"/> bound to this engine: <paramref name="value"/> stored as
        ///     <paramref name="dtype"/>.
        /// </summary>
        /// <param name="value">The value (converted when it is not of <paramref name="dtype"/>).</param>
        /// <param name="dtype">The dtype descriptor.</param>
        /// <returns>A new writeable 0-d array, bound to this engine.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="ValueError">A non-text value into a <c>coerce=False</c> <c>StringDType</c>.</exception>
        protected internal NDArray NewScalar(object value, DType dtype)
            => Bind(NDArray.Scalar(value, dtype));

        /// <summary>
        ///     Moves a fresh array the engine made by another route (a scalar, a conversion) onto this engine, and returns it.
        /// </summary>
        /// <param name="fresh">An array no one else holds yet (its storage object moves with it).</param>
        /// <returns><paramref name="fresh"/>, now bound to this engine.</returns>
        private NDArray Bind(NDArray fresh)
        {
            // the property's setter writes both the array's and its storage's engine, as a construction with the engine
            // would have
            fresh.TensorEngine = this;
            return fresh;
        }

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
        ///     The property belongs to this engine instance, which every array bound to it shares, so assigning it
        ///     reaches those arrays, existing ones included. An engine assigned to <see cref="Default"/> later starts
        ///     with its own value (null unless set): copy this one over to keep an installed backend.
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
        ///     The property belongs to this engine instance, which every array bound to it shares, so an assignment reaches
        ///     every array bound to the engine, existing ones included, from the next <c>np.linalg</c> call on: each call
        ///     reads the property once, so one already running finishes with the engine it read. Do not assign it while
        ///     another thread runs np.linalg functions whose results you compare: they may straddle the change. An engine
        ///     assigned to <see cref="Default"/> later starts with the shared default here.
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
        ///     The property belongs to this engine instance, which every array bound to it shares, so an assignment reaches
        ///     every array bound to the engine, existing ones included, from the next <c>np.fft</c> call on: each call reads
        ///     the property once, so one already running finishes with the engine it read. Do not assign it while another
        ///     thread runs np.fft functions whose results you compare: they may straddle the change. An engine assigned to
        ///     <see cref="Default"/> later starts with the shared default here.
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
