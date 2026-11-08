using System;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>NPY_cast_info</c> (<c>dtype_transfer.h</c>): one resolved cast between two descriptors - the strided
    ///     loop a caller runs over its own walk, and the context it was resolved for. <see cref="DTypeCasting"/> hands it out
    ///     (<c>PyArray_GetDTypeTransferFunction</c>); the raw assignment, the iterator's buffer transfers and the reductions
    ///     over text call <see cref="func"/> once per strided run.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The members are the struct's fields - <c>func</c>, <c>context</c>, <c>descriptors</c> - under NumPy's names. Not
    ///     ported: <c>auxdata</c> (a NumSharp loop is a delegate that captures what it needs, so there is nothing to clone or
    ///     free with it). NumPy's <c>NPY_cast_info_xfree</c> has no counterpart either: the loop owns only managed state.
    ///     </para>
    ///     <para>
    ///     A loop may hold per-loop work space (a parse's scratch buffer), so one instance serves one walk at a time, as a
    ///     NumPy <c>NPY_cast_info</c> does; resolve one per concurrent walk.
    ///     </para>
    /// </remarks>
    internal sealed class NDCastInfo
    {
        /// <summary>Wraps a resolved loop with the context it was resolved for.</summary>
        /// <param name="func">The strided loop.</param>
        /// <param name="context">The method and the resolved descriptors.</param>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        internal NDCastInfo(NDInnerLoopFunc func, ArrayMethodContext context)
        {
            // The parameters carry the fields' names (NumPy's spelling), so every field is written through `this.`.
            this.func = func ?? throw new ArgumentNullException(nameof(func));
            this.context = context ?? throw new ArgumentNullException(nameof(context));
        }

        /// <summary>
        ///     The strided loop (NumPy's <c>func</c>): <c>dataptrs[0]</c> / <c>strides[0]</c> the source's first element and byte
        ///     stride, <c>dataptrs[1]</c> / <c>strides[1]</c> the destination's, <c>count</c> the run length; it raises a
        ///     conversion error at the element NumPy's loop does, the elements before it converted.
        /// </summary>
        internal NDInnerLoopFunc func { get; }

        /// <summary>The method and the resolved operand descriptors the loop runs on (NumPy's <c>context</c>).</summary>
        internal ArrayMethodContext context { get; }

        /// <summary>The resolved descriptors, [0] the source, [1] the destination (NumPy's <c>descriptors</c>).</summary>
        internal IReadOnlyList<DType> descriptors => context.descriptors;

        /// <summary>
        ///     Runs <see cref="func"/> over one strided run: the operand pointers and strides in the loop's order, on the stack.
        /// </summary>
        /// <param name="src">The run's first source element.</param>
        /// <param name="srcStride">The source's byte stride (any sign; 0 repeats one element).</param>
        /// <param name="dst">The run's first destination element.</param>
        /// <param name="dstStride">The destination's byte stride.</param>
        /// <param name="count">The run length (0 converts nothing).</param>
        /// <exception cref="Exception">Whatever the loop raises (a conversion error).</exception>
        internal unsafe void Run(byte* src, long srcStride, byte* dst, long dstStride, long count)
        {
            if (count <= 0)
                return;
            void** ptrs = stackalloc void*[2];
            long* strides = stackalloc long[2];
            ptrs[0] = src;
            ptrs[1] = dst;
            strides[0] = srcStride;
            strides[1] = dstStride;
            func(ptrs, strides, count, null);
        }
    }
}
