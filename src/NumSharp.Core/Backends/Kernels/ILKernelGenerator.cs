// =============================================================================
// ILKernelGenerator — IL-emitted per-chunk kernels driven by NDIter
// =============================================================================
//
// MODEL
// -----
// The iterator (NDIterRef) owns the loop. Kernels emitted here implement the
// inner-loop body and are called once per chunk:
//
//     unsafe delegate void NDInnerLoopFunc(
//         void** dataptrs,   // [nop] current operand pointers
//         long*  strides,    // [nop] per-operand byte stride for inner loop
//         long   count,      // number of elements to process this call
//         void*  auxdata);   // op-specific extras (e.g. axis index)
//
// The iterator's iternext function advances dataptrs between calls. Each kernel
// only knows how to process ONE chunk; it has no axis-coordinate / stride-walk
// logic of its own. This mirrors NumPy's PyUFuncGenericFunction contract
// (numpy/_core/include/numpy/ufuncobject.h).
//
// RELATIONSHIP TO DirectILKernelGenerator
// ---------------------------------------
// DirectILKernelGenerator (in ./Direct/) holds the legacy whole-array kernels
// that iterate the entire array themselves and are called once with shape/
// strides/iterSize. They bypass NDIter's iternext machinery.
//
// New work registers per-chunk kernels here. As each np.* function migrates
// to NDIter-driven execution, its DirectILKernelGenerator partial is
// retired and a new ILKernelGenerator partial takes over.
//
// Coexistence is intentional during the migration: both classes share
// VectorMethodCache, ScalarMethodCache, KernelOp enums, kernel key structs,
// and the broader Kernels/ namespace types. Only the kernel-driving model
// differs.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Generates per-chunk IL kernels for NDIter-driven execution.
    ///
    /// Kernels emitted here are called as the inner loop of an NDIter
    /// iteration — once per chunk, with dataptrs/strides/count provided by
    /// the iterator. The kernel does no axis or stride walking of its own.
    ///
    /// Add new kernel families in <c>ILKernelGenerator.&lt;Op&gt;.cs</c>
    /// partial files. See <see cref="DirectILKernelGenerator"/> for the
    /// legacy whole-array kernels currently being migrated to this model.
    /// </summary>
    public static partial class ILKernelGenerator
    {
        // Kernel families are added in partial files alongside this one.
    }
}
