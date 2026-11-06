using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.WhereBytes.cs — np.where over elements of any byte width
// =============================================================================
//
// RESPONSIBILITY:
//   result[i] = cond[i] ? x[i] : y[i] for elements that are an opaque run of W bytes (a fixed-width string element:
//   W = chars for S, 4 * chars for U). The numeric where kernel (DirectILKernelGenerator.Where.cs) expands the bool
//   mask into SIMD lanes of a 1/2/4/8-byte element; a string element is wider than a lane, so this kernel selects a
//   whole element per condition byte instead: the source pointer is chosen branch-free and the W bytes move with ONE
//   cpblk whose size is an emit-time constant (the JIT unrolls it into vector moves for the common widths).
//
// KERNEL (DynamicMethod-emitted, one per width W):
//
//   * WhereBytesKernel(bool* cond, byte* x, byte* y, byte* result, long count)
//       every operand contiguous and element-aligned; count elements.
//   * WhereBytesStepKernel(bool* cond, long condStep, byte* x, long xStep, byte* y, long yStep, byte* result, long count)
//       the same select with each INPUT advancing by its own byte step: 0 for a one-element operand broadcast over the
//       result (np.where(c, names, "-")'s scalar branch), the element width for a contiguous one, a 1-D view's byte
//       stride for a strided or reversed one (names[::2]); the result is contiguous. Without it those calls took the
//       unit-view route: 2.0 ms for 100K <U6 elements against NumPy's 1.16 ms (0.58x), <S6 2.0 ms against 0.74 ms.
//
// WHY NOT THE UNIT VIEW: np.where(cond[..., None], xUnits, yUnits) broadcasts the condition over each element's units,
// which drives the general iterator path (28 ms for 1M <U8 elements, NumPy 15.6 ms); a masked np.copyto over the unit
// view still walks one row per element (12 ms). One element per condition byte is the shape of the problem.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     An IL-emitted element select for opaque elements of a fixed byte width (the width is baked into the kernel):
    ///     <c>result[i] = cond[i] ? x[i] : y[i]</c> over <paramref name="count"/> contiguous elements.
    /// </summary>
    /// <param name="cond">The condition, one byte per element (0 = false, anything else = true).</param>
    /// <param name="x">The values where the condition holds.</param>
    /// <param name="y">The values elsewhere.</param>
    /// <param name="result">The destination (must not overlap <paramref name="x"/> or <paramref name="y"/> partially).</param>
    /// <param name="count">The number of elements.</param>
    public unsafe delegate void WhereBytesKernel(bool* cond, byte* x, byte* y, byte* result, long count);

    /// <summary>
    ///     An IL-emitted element select for opaque elements of a fixed byte width whose INPUTS each advance by their own byte
    ///     step: <c>result[i] = cond[i·condStep] ? x[i·xStep] : y[i·yStep]</c> over <paramref name="count"/> elements written
    ///     contiguously. A step of 0 broadcasts a one-element operand; a negative step walks a reversed view.
    /// </summary>
    /// <param name="cond">The condition's first element, one byte per element (0 = false, anything else = true).</param>
    /// <param name="condStep">The byte step between condition elements (0, 1, or a 1-D view's stride).</param>
    /// <param name="x">The first value where the condition holds.</param>
    /// <param name="xStep">The byte step between <paramref name="x"/> elements (0, the width, or a 1-D view's stride).</param>
    /// <param name="y">The first value elsewhere.</param>
    /// <param name="yStep">The byte step between <paramref name="y"/> elements.</param>
    /// <param name="result">The contiguous destination (must not overlap <paramref name="x"/> or <paramref name="y"/>).</param>
    /// <param name="count">The number of elements.</param>
    public unsafe delegate void WhereBytesStepKernel(bool* cond, long condStep, byte* x, long xStep, byte* y, long yStep,
                                                     byte* result, long count);

    public static partial class DirectILKernelGenerator
    {
        /// <summary>The generated <see cref="WhereBytesKernel"/>s, one per element width in bytes.</summary>
        private static readonly ConcurrentDictionary<int, WhereBytesKernel> _whereBytesKernels = new();

        /// <summary>The generated <see cref="WhereBytesStepKernel"/>s, one per element width in bytes.</summary>
        private static readonly ConcurrentDictionary<int, WhereBytesStepKernel> _whereBytesStepKernels = new();

        /// <summary>
        ///     The <see cref="WhereBytesStepKernel"/> for elements of <paramref name="width"/> bytes, generated on first use and
        ///     cached: the select of <see cref="GetWhereBytesKernel"/> for inputs that are not all contiguous — a broadcast
        ///     one-element branch or condition, a strided or reversed 1-D view.
        /// </summary>
        /// <param name="width">The element width in bytes (at least 1).</param>
        /// <returns>The kernel, or null when IL generation is disabled (<see cref="Enabled"/>) or failed — the caller then
        ///     takes its general route.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="width"/> is not positive.</exception>
        public static WhereBytesStepKernel GetWhereBytesStepKernel(int width)
        {
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), width.ToString(CultureInfo.InvariantCulture), "An element is at least one byte wide.");
            if (!Enabled)
                return null;
            if (_whereBytesStepKernels.TryGetValue(width, out var cached))
                return cached;
            try
            {
                return _whereBytesStepKernels.GetOrAdd(width, GenerateWhereBytesStepKernelIL(width));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetWhereBytesStepKernel({width}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     Emits the step kernel. Pseudocode (<c>W</c> is a constant):
        ///     <code>
        ///     void WhereBytesStep(bool* cond, long cs, byte* x, long xs, byte* y, long ys, byte* result, long count) {
        ///         for (long i = 0; i &lt; count; i++) {
        ///             byte* src = *cond != 0 ? x : y;        // branch-free: y + (x - y) * (*cond != 0)
        ///             cpblk(result, src, W);
        ///             cond += cs; x += xs; y += ys; result += W;
        ///         }
        ///     }
        ///     </code>
        /// </summary>
        /// <param name="width">The element width in bytes.</param>
        /// <returns>The compiled kernel.</returns>
        /// <remarks>The select stays a pointer product, not a branch: the condition of a where is as unpredictable as
        ///     the data (half of a random mask mispredicts), and both candidate pointers are valid whichever is chosen.</remarks>
        private static unsafe WhereBytesStepKernel GenerateWhereBytesStepKernelIL(int width)
        {
            var dm = new DynamicMethod(
                name: $"IL_WhereBytesStep_{width}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(bool*), typeof(long), typeof(byte*), typeof(long), typeof(byte*), typeof(long),
                                        typeof(byte*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);
            var il = dm.GetILGenerator();

            var i = il.DeclareLocal(typeof(long));
            var src = il.DeclareLocal(typeof(byte*));
            var loop = il.DefineLabel();
            var done = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, i);

            il.MarkLabel(loop);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldarg_S, (byte)7);             // count
            il.Emit(OpCodes.Bge, done);

            // src = y + (x - y) * (*cond != 0)
            il.Emit(OpCodes.Ldarg_S, (byte)4);             // y
            il.Emit(OpCodes.Ldarg_2);                      // x
            il.Emit(OpCodes.Ldarg_S, (byte)4);             // y
            il.Emit(OpCodes.Sub);                          // x - y
            il.Emit(OpCodes.Ldarg_0);                      // cond
            il.Emit(OpCodes.Ldind_U1);                     // *cond
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Cgt_Un);                       // *cond != 0 → 1 / 0
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Mul);                          // (x - y) or 0
            il.Emit(OpCodes.Add);                          // x or y
            il.Emit(OpCodes.Stloc, src);

            // cpblk(result, src, W) — W is a constant, so the JIT unrolls it for the common widths.
            il.Emit(OpCodes.Ldarg_S, (byte)6);             // result
            il.Emit(OpCodes.Ldloc, src);
            il.Emit(OpCodes.Ldc_I4, width);
            il.Emit(OpCodes.Unaligned, (byte)1);
            il.Emit(OpCodes.Cpblk);

            // cond += condStep; x += xStep; y += yStep — each pointer argument followed by its step argument.
            foreach (var (ptr, step) in new (byte Ptr, byte Step)[] { (0, 1), (2, 3), (4, 5) })
            {
                il.Emit(OpCodes.Ldarg_S, ptr);
                il.Emit(OpCodes.Ldarg_S, step);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Starg_S, ptr);
            }
            // result += W (the destination is contiguous)
            il.Emit(OpCodes.Ldarg_S, (byte)6);
            il.Emit(OpCodes.Ldc_I4, width);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Starg_S, (byte)6);

            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, i);
            il.Emit(OpCodes.Br, loop);

            il.MarkLabel(done);
            il.Emit(OpCodes.Ret);
            return (WhereBytesStepKernel)dm.CreateDelegate(typeof(WhereBytesStepKernel));
        }

        /// <summary>
        ///     The <see cref="WhereBytesKernel"/> for elements of <paramref name="width"/> bytes, generated on first use and
        ///     cached.
        /// </summary>
        /// <param name="width">The element width in bytes (at least 1).</param>
        /// <returns>The kernel, or null when IL generation is disabled (<see cref="Enabled"/>) or failed — the caller then
        ///     takes its general route.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="width"/> is not positive.</exception>
        public static WhereBytesKernel GetWhereBytesKernel(int width)
        {
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), width.ToString(CultureInfo.InvariantCulture), "An element is at least one byte wide.");
            if (!Enabled)
                return null;
            if (_whereBytesKernels.TryGetValue(width, out var cached))
                return cached;
            try
            {
                return _whereBytesKernels.GetOrAdd(width, GenerateWhereBytesKernelIL(width));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetWhereBytesKernel({width}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     Emits the kernel. Pseudocode (<c>W</c> is a constant):
        ///     <code>
        ///     void WhereBytes(bool* cond, byte* x, byte* y, byte* result, long count) {
        ///         for (long i = 0; i &lt; count; i++) {
        ///             byte* src = cond[i] != 0 ? x : y;      // branch-free: y + (x - y) * (cond != 0)
        ///             cpblk(result, src, W);
        ///             x += W; y += W; result += W;
        ///         }
        ///     }
        ///     </code>
        /// </summary>
        /// <param name="width">The element width in bytes.</param>
        /// <returns>The compiled kernel.</returns>
        private static unsafe WhereBytesKernel GenerateWhereBytesKernelIL(int width)
        {
            var dm = new DynamicMethod(
                name: $"IL_WhereBytes_{width}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(bool*), typeof(byte*), typeof(byte*), typeof(byte*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);
            var il = dm.GetILGenerator();

            var i = il.DeclareLocal(typeof(long));
            var src = il.DeclareLocal(typeof(byte*));
            var loop = il.DefineLabel();
            var done = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, i);

            il.MarkLabel(loop);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Bge, done);

            // src = y + (x - y) * (cond[i] != 0): a pointer select with no branch on the (unpredictable) condition.
            il.Emit(OpCodes.Ldarg_2);                      // y
            il.Emit(OpCodes.Ldarg_1);                      // x
            il.Emit(OpCodes.Ldarg_2);                      // y
            il.Emit(OpCodes.Sub);                          // x - y
            il.Emit(OpCodes.Ldarg_0);                      // cond
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);                     // cond[i]
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Cgt_Un);                       // cond[i] != 0 → 1 / 0
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Mul);                          // (x - y) or 0
            il.Emit(OpCodes.Add);                          // x or y
            il.Emit(OpCodes.Stloc, src);

            // cpblk(result, src, W) — W is a constant, so the JIT unrolls it for the common widths.
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, src);
            il.Emit(OpCodes.Ldc_I4, width);
            il.Emit(OpCodes.Unaligned, (byte)1);
            il.Emit(OpCodes.Cpblk);

            // x += W; y += W; result += W; i++
            foreach (short arg in new short[] { 1, 2, 3 })
            {
                il.Emit(OpCodes.Ldarg_S, (byte)arg);
                il.Emit(OpCodes.Ldc_I4, width);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Starg_S, (byte)arg);
            }
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, i);
            il.Emit(OpCodes.Br, loop);

            il.MarkLabel(done);
            il.Emit(OpCodes.Ret);
            return (WhereBytesKernel)dm.CreateDelegate(typeof(WhereBytesKernel));
        }
    }
}
