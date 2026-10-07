using System;
using System.Collections.Concurrent;
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

    public static partial class DirectILKernelGenerator
    {
        /// <summary>The generated <see cref="WhereBytesKernel"/>s, one per element width in bytes.</summary>
        private static readonly ConcurrentDictionary<int, WhereBytesKernel> _whereBytesKernels = new();

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
                throw new ArgumentOutOfRangeException(nameof(width), width, "An element is at least one byte wide.");
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
