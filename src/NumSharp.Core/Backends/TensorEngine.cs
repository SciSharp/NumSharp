// TensorEngine's core file: the Blas seam and NumSharp's own np.evaluate entry points. The members NumPy names
// live in partial files named after NumPy's sources: TensorEngine.Umath.cs (the ufuncs), TensorEngine.Methods.cs
// (the ndarray methods that compute), TensorEngine.Multiarray.cs (the C functions that compute),
// TensorEngine.Numeric.cs, TensorEngine.NanFunctions.cs, TensorEngine.UfuncLike.cs, TensorEngine.LinearAlgebra.cs,
// and TensorEngine.ArrayObject.cs (the array-object members shipped in 0.70.0). The rule that places a function is
// docs/plans/np-layer-dtype-logic.md, section 1 "Which O" and Appendix D.

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
