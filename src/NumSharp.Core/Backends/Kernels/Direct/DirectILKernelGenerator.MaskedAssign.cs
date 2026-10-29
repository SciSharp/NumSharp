using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.MaskedAssign.cs — np.copyto's where-masked assignment
// =============================================================================
//
// RESPONSIBILITY:
//   The element loop of NumPy's raw_array_wheremasked_assign_array and
//   raw_array_wheremasked_assign_scalar (numpy/_core/src/multiarray/array_assign_array.c,
//   array_assign_scalar.c): dst[i] = src[i] wherever mask[i] is nonzero, over operands the
//   caller has already broadcast, prepared (PyArray_PrepareThreeRawArrayIter's axis sort,
//   negative-stride flip and coalescing) and — for a cross-dtype copy — cast into the
//   destination's dtype. What is left is a BIT copy of one element width, so ONE kernel per
//   width (1 / 2 / 4 / 8 / 16 bytes) serves all 15 dtypes: no per-dtype code.
//
// STRUCTURE (two stages, the house "IL kernel calls a tidy C# helper" pattern):
//   * MaskedAssignKernel — an IL DynamicMethod per width: the outer walk. Axes arrive in C order
//     (the last axis is the inner run) with BYTE strides; it walks the outer axes with an odometer
//     of incremental byte offsets and calls the run stage once per inner run.
//   * MaskedAssignRun<T> / MaskedAssignRun16 — the inner run, chosen per run by its strides:
//       - a mask stride of 0 decides the whole run at once (NumPy's npy_memchr finds one run):
//         nothing, or an unmasked copy (memmove for a contiguous run, a fill for a broadcast
//         source, an element loop otherwise);
//       - a contiguous destination + contiguous mask + contiguous source: a SIMD blend
//         (ConditionalSelect between the source vector and the destination's own vector), 2x
//         unrolled; with a broadcast source (stride 0) the source vector is hoisted once;
//       - anything else, and every tail: a branch-free element select in the run's order.
//
// ORDER (why the SIMD paths are exact even when src and dst overlap):
//   The caller makes a temporary copy of an overlapping source unless the destination is 1-D with
//   both runs in the same direction — NumPy's own rule — so only a 1-D run can still overlap here.
//   After NumPy's flip (a negative destination stride becomes positive) and its backward rule
//   (src < dst with overlap: both walked from the end, strides negated), such a run is either
//   walked backward (negative strides → the element loop, element by element exactly like NumPy)
//   or forward with src >= dst, where a vector load of src[i..i+V) reads no element an earlier or the
//   same store wrote. A masked-off destination slot is written back with the value loaded from it in
//   the same step, so its final value is NumPy's too.
//
// SPEED: NumPy's masked copy splits every inner loop into runs of true mask values and calls the
//   cast function per run, so a random mask costs it ~3–4 ns per element. The blend touches each
//   element once at SIMD width whatever the mask holds.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Where-masked bit copy of one element width over prepared, broadcast operands (np.copyto's element loop):
    /// <c>dst[i] = src[i]</c> wherever <c>mask[i] != 0</c>.
    /// </summary>
    /// <param name="src">The source's first element (already cast to the destination's dtype).</param>
    /// <param name="dst">The destination's first element.</param>
    /// <param name="mask">The mask's first byte (a bool array; any nonzero byte is true).</param>
    /// <param name="srcStrides">Source BYTE strides, one per axis, C order (the last axis is the inner run); 0 broadcasts.</param>
    /// <param name="dstStrides">Destination BYTE strides, one per axis.</param>
    /// <param name="maskStrides">Mask BYTE strides, one per axis; 0 broadcasts.</param>
    /// <param name="shape">The extent of each axis; every extent must be at least 1.</param>
    /// <param name="ndim">The number of axes, at least 1 (a 0-d assignment passes one axis of extent 1).</param>
    public unsafe delegate void MaskedAssignKernel(
        byte* src, byte* dst, byte* mask,
        long* srcStrides, long* dstStrides, long* maskStrides,
        long* shape, int ndim);

    public static partial class DirectILKernelGenerator
    {
        /// <summary>The emitted outer walkers, one per element width (1, 2, 4, 8, 16 bytes).</summary>
        internal static readonly ConcurrentDictionary<int, MaskedAssignKernel> _maskedAssignCache = new();

        /// <summary>
        /// The masked-assign kernel for elements of <paramref name="elementSize"/> bytes, emitted on first use.
        /// </summary>
        /// <param name="elementSize">The element width in bytes: 1, 2, 4, 8 or 16 (the widths of the 15 dtypes).</param>
        /// <returns>The kernel, or <c>null</c> when IL generation is off (<see cref="Enabled"/>), the host has no dynamic
        /// code, the width is not one of the five, or the emission failed. A <c>null</c> result means "run
        /// <see cref="MaskedAssignFallback"/>", which computes the same bits through the same run stage.</returns>
        public static MaskedAssignKernel TryGetMaskedAssignKernel(int elementSize)
        {
            if (!Enabled || !RuntimeFeature.IsDynamicCodeSupported)
                return null;
            if (elementSize != 1 && elementSize != 2 && elementSize != 4 && elementSize != 8 && elementSize != 16)
                return null;
            if (_maskedAssignCache.TryGetValue(elementSize, out var cached))
                return cached;

            try
            {
                return _maskedAssignCache.GetOrAdd(elementSize, GenerateMaskedAssignKernel(elementSize));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGetMaskedAssignKernel({elementSize}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// The run stage for <paramref name="elementSize"/>-byte elements: <see cref="MaskedAssignRun{T}"/> over the
        /// unsigned integer of that width, or <see cref="MaskedAssignRun16"/> for 16-byte elements (complex, decimal).
        /// </summary>
        /// <param name="elementSize">1, 2, 4, 8 or 16.</param>
        /// <returns>The closed run method the emitted walker calls.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="elementSize"/> is not one of the five widths.</exception>
        private static MethodInfo MaskedAssignRunMethod(int elementSize)
        {
            if (elementSize == 16)
                return typeof(DirectILKernelGenerator).GetMethod(nameof(MaskedAssignRun16), BindingFlags.Static | BindingFlags.NonPublic)
                       ?? throw new MissingMethodException(nameof(DirectILKernelGenerator), nameof(MaskedAssignRun16));

            Type lane = elementSize switch
            {
                1 => typeof(byte),
                2 => typeof(ushort),
                4 => typeof(uint),
                8 => typeof(ulong),
                _ => throw new ArgumentOutOfRangeException(nameof(elementSize), elementSize, "element width must be 1, 2, 4, 8 or 16 bytes"),
            };
            var open = typeof(DirectILKernelGenerator).GetMethod(nameof(MaskedAssignRun), BindingFlags.Static | BindingFlags.NonPublic)
                       ?? throw new MissingMethodException(nameof(DirectILKernelGenerator), nameof(MaskedAssignRun));
            return open.MakeGenericMethod(lane);
        }

        /// <summary>
        /// Emits the outer walk. Pseudocode:
        /// <code>
        /// void MaskedAssign(byte* src, byte* dst, byte* mask, long* ss, long* ds, long* ms, long* shape, int ndim) {
        ///     int outer = ndim - 1;                                  // axes [0, outer) are walked, axis outer is the run
        ///     long n = shape[outer], sIn = ss[outer], dIn = ds[outer], mIn = ms[outer];
        ///     long* coords = stackalloc long[outer + 1];             // zeroed
        ///     long sOff = 0, dOff = 0, mOff = 0;
        ///     for (;;) {
        ///         Run(src + sOff, sIn, dst + dOff, dIn, mask + mOff, mIn, n);
        ///         int ax = outer - 1;
        ///         for (; ax &gt;= 0; ax--) {                               // the last outer axis moves fastest
        ///             coords[ax]++; sOff += ss[ax]; dOff += ds[ax]; mOff += ms[ax];
        ///             if (coords[ax] &lt; shape[ax]) break;
        ///             coords[ax] = 0; sOff -= ss[ax] * shape[ax]; dOff -= ds[ax] * shape[ax]; mOff -= ms[ax] * shape[ax];
        ///         }
        ///         if (ax &lt; 0) return;
        ///     }
        /// }
        /// </code>
        /// </summary>
        /// <param name="elementSize">The element width the walker's run stage copies.</param>
        /// <returns>The compiled walker.</returns>
        private static MaskedAssignKernel GenerateMaskedAssignKernel(int elementSize)
        {
            MethodInfo run = MaskedAssignRunMethod(elementSize);

            var dm = new DynamicMethod(
                name: $"IL_MaskedAssign_w{elementSize}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 src
                    typeof(byte*),  // 1 dst
                    typeof(byte*),  // 2 mask
                    typeof(long*),  // 3 srcStrides (bytes)
                    typeof(long*),  // 4 dstStrides (bytes)
                    typeof(long*),  // 5 maskStrides (bytes)
                    typeof(long*),  // 6 shape
                    typeof(int),    // 7 ndim
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locOuter = il.DeclareLocal(typeof(int));
            var locN = il.DeclareLocal(typeof(long));
            var locSIn = il.DeclareLocal(typeof(long));
            var locDIn = il.DeclareLocal(typeof(long));
            var locMIn = il.DeclareLocal(typeof(long));
            var locSOff = il.DeclareLocal(typeof(long));
            var locDOff = il.DeclareLocal(typeof(long));
            var locMOff = il.DeclareLocal(typeof(long));
            var locCoords = il.DeclareLocal(typeof(long*));
            var locAx = il.DeclareLocal(typeof(int));

            var lblRun = il.DefineLabel();
            var lblAdvance = il.DefineLabel();
            var lblRet = il.DefineLabel();

            // outer = ndim - 1
            il.Emit(OpCodes.Ldarg_S, (byte)7);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locOuter);

            // The inner run's extent and strides, read once.
            EmitLoadLongAt(il, 6, locOuter);
            il.Emit(OpCodes.Stloc, locN);
            EmitLoadLongAt(il, 3, locOuter);
            il.Emit(OpCodes.Stloc, locSIn);
            EmitLoadLongAt(il, 4, locOuter);
            il.Emit(OpCodes.Stloc, locDIn);
            EmitLoadLongAt(il, 5, locOuter);
            il.Emit(OpCodes.Stloc, locMIn);

            // coords = localloc((outer + 1) * 8), zeroed: one slot more than the walked axes keeps the size
            // positive for a single-run (outer == 0) call.
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Localloc);
            il.Emit(OpCodes.Stloc, locCoords);
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Initblk);

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locSOff);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locDOff);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locMOff);

            // ---- one inner run ----
            il.MarkLabel(lblRun);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locSOff);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locSIn);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locDOff);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locDIn);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locMOff);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locMIn);
            il.Emit(OpCodes.Ldloc, locN);
            il.Emit(OpCodes.Call, run);

            // ax = outer - 1; no outer axis → done
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locAx);

            // ---- odometer: the last outer axis moves fastest ----
            il.MarkLabel(lblAdvance);
            il.Emit(OpCodes.Ldloc, locAx);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Blt, lblRet);

            // coords[ax]++
            EmitCoordAddress(il, locCoords, locAx);
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stind_I8);

            // offsets += strides[ax]
            EmitOffsetAddStride(il, locSOff, 3, locAx);
            EmitOffsetAddStride(il, locDOff, 4, locAx);
            EmitOffsetAddStride(il, locMOff, 5, locAx);

            // if (coords[ax] < shape[ax]) goto run
            EmitCoordAddress(il, locCoords, locAx);
            il.Emit(OpCodes.Ldind_I8);
            EmitLoadLongAt(il, 6, locAx);
            il.Emit(OpCodes.Blt, lblRun);

            // coords[ax] = 0; offsets -= strides[ax] * shape[ax]
            EmitCoordAddress(il, locCoords, locAx);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stind_I8);
            EmitOffsetRewind(il, locSOff, 3, locAx);
            EmitOffsetRewind(il, locDOff, 4, locAx);
            EmitOffsetRewind(il, locMOff, 5, locAx);

            // ax--
            il.Emit(OpCodes.Ldloc, locAx);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locAx);
            il.Emit(OpCodes.Br, lblAdvance);

            il.MarkLabel(lblRet);
            il.Emit(OpCodes.Ret);

            return dm.CreateDelegate<MaskedAssignKernel>();
        }

        /// <summary>Pushes <c>((long*)arg)[index]</c>.</summary>
        /// <param name="il">The emitter.</param>
        /// <param name="argIndex">The <c>long*</c> argument.</param>
        /// <param name="index">The <c>int</c> local holding the axis.</param>
        private static void EmitLoadLongAt(ILGenerator il, byte argIndex, LocalBuilder index)
        {
            il.Emit(OpCodes.Ldarg_S, argIndex);
            il.Emit(OpCodes.Ldloc, index);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
        }

        /// <summary>Pushes <c>coords + ax</c> (a <c>long*</c>).</summary>
        /// <param name="il">The emitter.</param>
        /// <param name="coords">The <c>long*</c> local of the odometer.</param>
        /// <param name="ax">The <c>int</c> local holding the axis.</param>
        private static void EmitCoordAddress(ILGenerator il, LocalBuilder coords, LocalBuilder ax)
        {
            il.Emit(OpCodes.Ldloc, coords);
            il.Emit(OpCodes.Ldloc, ax);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
        }

        /// <summary>Emits <c>offset += strides[ax]</c>.</summary>
        /// <param name="il">The emitter.</param>
        /// <param name="offset">The <c>long</c> byte-offset local.</param>
        /// <param name="stridesArg">The <c>long*</c> strides argument.</param>
        /// <param name="ax">The <c>int</c> local holding the axis.</param>
        private static void EmitOffsetAddStride(ILGenerator il, LocalBuilder offset, byte stridesArg, LocalBuilder ax)
        {
            il.Emit(OpCodes.Ldloc, offset);
            EmitLoadLongAt(il, stridesArg, ax);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, offset);
        }

        /// <summary>Emits <c>offset -= strides[ax] * shape[ax]</c> (the odometer's carry back to coordinate 0).</summary>
        /// <param name="il">The emitter.</param>
        /// <param name="offset">The <c>long</c> byte-offset local.</param>
        /// <param name="stridesArg">The <c>long*</c> strides argument.</param>
        /// <param name="ax">The <c>int</c> local holding the axis.</param>
        private static void EmitOffsetRewind(ILGenerator il, LocalBuilder offset, byte stridesArg, LocalBuilder ax)
        {
            il.Emit(OpCodes.Ldloc, offset);
            EmitLoadLongAt(il, stridesArg, ax);
            EmitLoadLongAt(il, 6, ax);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, offset);
        }

        /// <summary>
        /// The same walk as the emitted kernel, in C#, for hosts where <see cref="TryGetMaskedAssignKernel"/> returns
        /// <c>null</c> (IL generation off, or no dynamic code, e.g. NativeAOT). It calls the same run stage, so the bits
        /// are the kernel's.
        /// </summary>
        /// <param name="elementSize">The element width: 1, 2, 4, 8 or 16 bytes.</param>
        /// <param name="src">See <see cref="MaskedAssignKernel"/>.</param>
        /// <param name="dst">See <see cref="MaskedAssignKernel"/>.</param>
        /// <param name="mask">See <see cref="MaskedAssignKernel"/>.</param>
        /// <param name="srcStrides">See <see cref="MaskedAssignKernel"/>.</param>
        /// <param name="dstStrides">See <see cref="MaskedAssignKernel"/>.</param>
        /// <param name="maskStrides">See <see cref="MaskedAssignKernel"/>.</param>
        /// <param name="shape">See <see cref="MaskedAssignKernel"/>.</param>
        /// <param name="ndim">See <see cref="MaskedAssignKernel"/>.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="elementSize"/> is not one of the five widths.</exception>
        internal static unsafe void MaskedAssignFallback(
            int elementSize, byte* src, byte* dst, byte* mask,
            long* srcStrides, long* dstStrides, long* maskStrides, long* shape, int ndim)
        {
            int outer = ndim - 1;
            long n = shape[outer], sIn = srcStrides[outer], dIn = dstStrides[outer], mIn = maskStrides[outer];
            long* coords = stackalloc long[outer + 1];
            for (int k = 0; k <= outer; k++)
                coords[k] = 0;
            long sOff = 0, dOff = 0, mOff = 0;
            for (;;)
            {
                switch (elementSize)
                {
                    case 1: MaskedAssignRun<byte>(src + sOff, sIn, dst + dOff, dIn, mask + mOff, mIn, n); break;
                    case 2: MaskedAssignRun<ushort>(src + sOff, sIn, dst + dOff, dIn, mask + mOff, mIn, n); break;
                    case 4: MaskedAssignRun<uint>(src + sOff, sIn, dst + dOff, dIn, mask + mOff, mIn, n); break;
                    case 8: MaskedAssignRun<ulong>(src + sOff, sIn, dst + dOff, dIn, mask + mOff, mIn, n); break;
                    case 16: MaskedAssignRun16(src + sOff, sIn, dst + dOff, dIn, mask + mOff, mIn, n); break;
                    default: throw new ArgumentOutOfRangeException(nameof(elementSize), elementSize, "element width must be 1, 2, 4, 8 or 16 bytes");
                }

                int ax = outer - 1;
                for (; ax >= 0; ax--)
                {
                    coords[ax]++;
                    sOff += srcStrides[ax];
                    dOff += dstStrides[ax];
                    mOff += maskStrides[ax];
                    if (coords[ax] < shape[ax])
                        break;
                    coords[ax] = 0;
                    sOff -= srcStrides[ax] * shape[ax];
                    dOff -= dstStrides[ax] * shape[ax];
                    mOff -= maskStrides[ax] * shape[ax];
                }

                if (ax < 0)
                    return;
            }
        }

        /// <summary>
        /// NumPy's unmasked 1-D <c>raw_array_assign_array</c> for two operands that MAY overlap: the same-dtype copy
        /// <c>dst[i] = src[i]</c>, i in [0, n), in NumPy's element order. That order is observable when the runs overlap
        /// with different strides: NumPy flips a negative destination stride (both operands walked from their last
        /// element), then walks backward when the source starts below the destination inside it, and otherwise
        /// forward — reading elements an earlier step may have written (<c>a[3:19:2] = a[6:14]</c> reads its own
        /// writes). Equal strides make that order the same as copying the source first.
        /// </summary>
        /// <param name="dst">The destination's first element.</param>
        /// <param name="dstStride">The destination's BYTE stride (any sign; not 0 — a writeable destination has distinct
        /// elements).</param>
        /// <param name="src">The source's first element.</param>
        /// <param name="srcStride">The source's BYTE stride, 0 for a broadcast element; same direction as
        /// <paramref name="dstStride"/> (the caller copies an opposite-direction overlapping source first, as NumPy
        /// does).</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="elementSize">The element width: 1, 2, 4, 8 or 16 bytes.</param>
        /// <remarks>
        /// Runs the masked-assign stages with a constant true mask, whose run stage copies a contiguous run with
        /// <see cref="Buffer.MemoryCopy(void*, void*, long, long)"/> (memmove: NumPy's contiguous copy is too) and any
        /// other run element by element in the order above.
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="elementSize"/> is not one of the five widths.</exception>
        internal static unsafe void AssignRaw1D(byte* dst, long dstStride, byte* src, long srcStride, long n, int elementSize)
        {
            if (n <= 0)
                return;

            // PyArray_PrepareTwoRawArrayIter, ndim 1: "Always make a positive stride for the first operand".
            if (dstStride < 0)
            {
                dst += dstStride * (n - 1);
                src += srcStride * (n - 1);
                dstStride = -dstStride;
                srcStride = -srcStride;
            }

            // raw_array_assign_array: "Overlap check for the 1D case" — walk from the end when the source starts
            // below the destination and runs into it.
            if (src < dst && src + n * srcStride > dst)
            {
                src += (n - 1) * srcStride;
                dst += (n - 1) * dstStride;
                srcStride = -srcStride;
                dstStride = -dstStride;
            }

            byte trueByte = 1;
            long one = n, s = srcStride, d = dstStride, m = 0;
            var kernel = TryGetMaskedAssignKernel(elementSize);
            if (kernel != null)
                kernel(src, dst, &trueByte, &s, &d, &m, &one, 1);
            else
                MaskedAssignFallback(elementSize, src, dst, &trueByte, &s, &d, &m, &one, 1);
        }

        // ---------------------------------------------------------------------------------------------------------
        // Run stage
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        /// One inner run of the masked assignment for a <typeparamref name="T"/>-wide element: <c>dst[i] = src[i]</c>
        /// for i in [0, n) wherever <c>mask[i] != 0</c>.
        /// </summary>
        /// <typeparam name="T">The unsigned integer of the element width (<see cref="byte"/>, <see cref="ushort"/>,
        /// <see cref="uint"/>, <see cref="ulong"/>): the copy is a bit copy, whatever the dtype.</typeparam>
        /// <param name="src">The run's first source element.</param>
        /// <param name="srcStride">The source byte stride (0 broadcasts one element; may be negative).</param>
        /// <param name="dst">The run's first destination element.</param>
        /// <param name="dstStride">The destination byte stride (may be negative after NumPy's backward rule).</param>
        /// <param name="mask">The run's first mask byte.</param>
        /// <param name="maskStride">The mask byte stride (0: one mask byte decides the whole run).</param>
        /// <param name="n">The number of elements in the run.</param>
        /// <remarks>
        /// Every element the select path visits is WRITTEN — a masked-off slot gets the value read from it in the same
        /// step — which is NumPy's result because no other step reads or writes that slot in between. The SIMD paths
        /// run only for a forward contiguous destination and mask; see the file header for why that order is NumPy's
        /// even for an overlapping 1-D run.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void MaskedAssignRun<T>(
            byte* src, long srcStride, byte* dst, long dstStride, byte* mask, long maskStride, long n)
            where T : unmanaged
        {
            if (maskStride == 0)
            {
                // A mask byte that does not move along the run: npy_memchr finds ONE run of length n — nothing to do,
                // or an unmasked copy of the whole run.
                if (*mask != 0)
                    MaskedAssignCopyRun<T>(src, srcStride, dst, dstStride, n);
                return;
            }

            long i = 0;
            if (dstStride == sizeof(T) && maskStride == 1)
            {
                if (srcStride == sizeof(T))
                    i = MaskedAssignBlendContiguous<T>(src, dst, mask, n);
                else if (srcStride == 0)
                    i = MaskedAssignBlendBroadcast<T>(*(T*)src, dst, mask, n);
            }

            // Element by element in the run's order: the strided runs, and the tails of the vector loops. A bit select
            // under an all-ones / all-zeros lane mask, never a branch: a random mask mispredicts a branch on a third of
            // the elements (~2.5 ns each, measured), the arithmetic costs a few cycles whatever the mask holds. Four
            // elements per pass off running pointers (no multiply per element); each element's store still precedes the
            // next element's loads, so an overlapping 1-D run is walked exactly in NumPy's order.
            byte* d = dst + i * dstStride;
            byte* s = src + i * srcStride;
            byte* mk = mask + i * maskStride;
            long d2 = 2 * dstStride, d3 = 3 * dstStride, d4 = 4 * dstStride;
            long s2 = 2 * srcStride, s3 = 3 * srcStride, s4 = 4 * srcStride;
            long m2 = 2 * maskStride, m3 = 3 * maskStride, m4 = 4 * maskStride;
            for (; i <= n - 4; i += 4)
            {
                MaskedAssignSelect<T>(d, s, mk[0]);
                MaskedAssignSelect<T>(d + dstStride, s + srcStride, mk[maskStride]);
                MaskedAssignSelect<T>(d + d2, s + s2, mk[m2]);
                MaskedAssignSelect<T>(d + d3, s + s3, mk[m3]);
                d += d4;
                s += s4;
                mk += m4;
            }
            for (; i < n; i++)
            {
                MaskedAssignSelect<T>(d, s, *mk);
                d += dstStride;
                s += srcStride;
                mk += maskStride;
            }
        }

        /// <summary>
        /// One element of the select loop: <c>*dst = *src</c> when <paramref name="maskByte"/> is nonzero, else
        /// <c>*dst</c> written back unchanged — a bit select, no branch.
        /// </summary>
        /// <typeparam name="T">The unsigned integer of the element width.</typeparam>
        /// <param name="dst">The destination element.</param>
        /// <param name="src">The source element (read before the store: it may be the destination itself).</param>
        /// <param name="maskByte">The element's mask byte (any nonzero byte is true).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void MaskedAssignSelect<T>(byte* dst, byte* src, byte maskByte) where T : unmanaged
        {
            ulong sel = MaskedAssignLaneMask(maskByte);
            if (sizeof(T) == 8)
            {
                ulong old = *(ulong*)dst;
                *(ulong*)dst = old ^ ((*(ulong*)src ^ old) & sel);
            }
            else if (sizeof(T) == 4)
            {
                uint old = *(uint*)dst;
                *(uint*)dst = old ^ ((*(uint*)src ^ old) & (uint)sel);
            }
            else if (sizeof(T) == 2)
            {
                ushort old = *(ushort*)dst;
                *(ushort*)dst = (ushort)(old ^ ((*(ushort*)src ^ old) & (ushort)sel));
            }
            else
            {
                byte old = *dst;
                *dst = (byte)(old ^ ((*src ^ old) & (byte)sel));
            }
        }

        /// <summary>
        /// The select mask of one mask byte without a compare-and-branch: all ones for any nonzero byte (a true
        /// <c>npy_bool</c>, canonical or not), zero for 0.
        /// </summary>
        /// <param name="b">The mask byte.</param>
        /// <returns><c>ulong.MaxValue</c> when <paramref name="b"/> is nonzero, else 0.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong MaskedAssignLaneMask(byte b) => (ulong)-(long)((b + 0xFFu) >> 8);

        /// <summary>
        /// An unmasked copy of one run (the run's mask is a single true byte): <c>dst[i] = src[i]</c> for i in [0, n).
        /// </summary>
        /// <typeparam name="T">The unsigned integer of the element width.</typeparam>
        /// <param name="src">The run's first source element.</param>
        /// <param name="srcStride">The source byte stride (0 fills one value).</param>
        /// <param name="dst">The run's first destination element.</param>
        /// <param name="dstStride">The destination byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <remarks>A forward contiguous run is one <see cref="Buffer.MemoryCopy(void*, void*, long, long)"/>, which
        /// is memmove (overlap-safe, NumPy's contiguous copy too); a negative stride — NumPy's backward walk — takes the
        /// element loop, which copies in exactly that order.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void MaskedAssignCopyRun<T>(byte* src, long srcStride, byte* dst, long dstStride, long n)
            where T : unmanaged
        {
            if (dstStride == sizeof(T) && srcStride == sizeof(T))
            {
                long bytes = n * sizeof(T);
                Buffer.MemoryCopy(src, dst, bytes, bytes);
                return;
            }

            if (srcStride == 0 && dstStride == sizeof(T))
            {
                // Span<T>.Fill is vectorized; its length is an int, so a longer run fills in int-sized pieces.
                T value = *(T*)src;
                T* p = (T*)dst;
                long left = n;
                while (left > 0)
                {
                    int chunk = left > int.MaxValue ? int.MaxValue : (int)left;
                    new Span<T>(p, chunk).Fill(value);
                    p += chunk;
                    left -= chunk;
                }
                return;
            }

            // Running pointers, element by element in the run's order (an overlapping backward walk reads each source
            // element before the step that could overwrite it, exactly as NumPy's loop does).
            for (long i = 0; i < n; i++)
            {
                *(T*)dst = *(T*)src;
                dst += dstStride;
                src += srcStride;
            }
        }

        /// <summary>Whether the run stage's vector paths can run for <typeparamref name="T"/> on this host.</summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <returns><c>true</c> when 256-bit vectors are accelerated and, for lanes wider than a byte, AVX2 provides the
        /// one-instruction mask widening (<c>vpmovzx</c>) — the condition every JIT folds to a constant.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool MaskedAssignVectorized<T>() where T : unmanaged
            => Vector256.IsHardwareAccelerated && (typeof(T) == typeof(byte) || Avx2.IsSupported);

        /// <summary>
        /// The lanes of a 256-bit <typeparamref name="T"/> vector whose mask byte is zero, as all-ones lanes (the
        /// "keep the destination" selector); the vector covers <c>Vector256&lt;T&gt;.Count</c> mask bytes from
        /// <paramref name="mask"/>, and no more are read.
        /// </summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <param name="mask">The first mask byte of the vector.</param>
        /// <returns>The selector: all ones where the mask byte is 0, zero where it is nonzero (any nonzero byte is true,
        /// as NumPy's <c>npy_bool</c> test is).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<T> MaskedAssignKeep256<T>(byte* mask) where T : unmanaged
        {
            if (typeof(T) == typeof(byte))
                return Vector256.Equals(Vector256.Load(mask), Vector256<byte>.Zero).As<byte, T>();
            if (typeof(T) == typeof(ushort))
                return Vector256.Equals(Avx2.ConvertToVector256Int16(Vector128.Load(mask)).AsUInt16(), Vector256<ushort>.Zero)
                    .As<ushort, T>();
            if (typeof(T) == typeof(uint))
                return Vector256.Equals(
                        Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(mask)).AsByte()).AsUInt32(),
                        Vector256<uint>.Zero)
                    .As<uint, T>();
            return Vector256.Equals(
                    Avx2.ConvertToVector256Int64(Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<uint>(mask)).AsByte()).AsUInt64(),
                    Vector256<ulong>.Zero)
                .As<ulong, T>();
        }

        /// <summary>
        /// The SIMD blend of a contiguous run: source, destination and mask all forward contiguous.
        /// </summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <param name="src">The first source element.</param>
        /// <param name="dst">The first destination element.</param>
        /// <param name="mask">The first mask byte.</param>
        /// <param name="n">The run's length.</param>
        /// <returns>How many leading elements were blended (a multiple of the lane count); the caller's element loop
        /// finishes the rest.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long MaskedAssignBlendContiguous<T>(byte* src, byte* dst, byte* mask, long n)
            where T : unmanaged
        {
            if (!MaskedAssignVectorized<T>())
                return 0;

            int lanes = Vector256<T>.Count;
            T* s = (T*)src;
            T* d = (T*)dst;
            long i = 0;
            // Two vectors per pass: the selects are independent, so the loads of the second overlap the first's.
            for (; i <= n - 2 * lanes; i += 2 * lanes)
            {
                Vector256<T> keep0 = MaskedAssignKeep256<T>(mask + i);
                Vector256<T> keep1 = MaskedAssignKeep256<T>(mask + i + lanes);
                Vector256<T> s0 = Vector256.Load(s + i);
                Vector256<T> s1 = Vector256.Load(s + i + lanes);
                Vector256<T> d0 = Vector256.Load(d + i);
                Vector256<T> d1 = Vector256.Load(d + i + lanes);
                Vector256.ConditionalSelect(keep0, d0, s0).Store(d + i);
                Vector256.ConditionalSelect(keep1, d1, s1).Store(d + i + lanes);
            }
            for (; i <= n - lanes; i += lanes)
            {
                Vector256<T> keep = MaskedAssignKeep256<T>(mask + i);
                Vector256.ConditionalSelect(keep, Vector256.Load(d + i), Vector256.Load(s + i)).Store(d + i);
            }
            return i;
        }

        /// <summary>
        /// The SIMD blend of a contiguous destination and mask with one broadcast source value.
        /// </summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <param name="value">The source value (read once: writing it back over its own slot, should the source lie
        /// inside the destination, leaves it unchanged).</param>
        /// <param name="dst">The first destination element.</param>
        /// <param name="mask">The first mask byte.</param>
        /// <param name="n">The run's length.</param>
        /// <returns>How many leading elements were blended; the caller's element loop finishes the rest.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long MaskedAssignBlendBroadcast<T>(T value, byte* dst, byte* mask, long n)
            where T : unmanaged
        {
            if (!MaskedAssignVectorized<T>())
                return 0;

            int lanes = Vector256<T>.Count;
            Vector256<T> v = Vector256.Create(value);
            T* d = (T*)dst;
            long i = 0;
            for (; i <= n - 2 * lanes; i += 2 * lanes)
            {
                Vector256<T> keep0 = MaskedAssignKeep256<T>(mask + i);
                Vector256<T> keep1 = MaskedAssignKeep256<T>(mask + i + lanes);
                Vector256.ConditionalSelect(keep0, Vector256.Load(d + i), v).Store(d + i);
                Vector256.ConditionalSelect(keep1, Vector256.Load(d + i + lanes), v).Store(d + i + lanes);
            }
            for (; i <= n - lanes; i += lanes)
            {
                Vector256<T> keep = MaskedAssignKeep256<T>(mask + i);
                Vector256.ConditionalSelect(keep, Vector256.Load(d + i), v).Store(d + i);
            }
            return i;
        }

        /// <summary>
        /// One inner run for 16-byte elements (complex128, decimal): the same contract as
        /// <see cref="MaskedAssignRun{T}"/>, each element moved as two 8-byte halves under one mask byte.
        /// </summary>
        /// <param name="src">The run's first source element.</param>
        /// <param name="srcStride">The source byte stride (0 broadcasts).</param>
        /// <param name="dst">The run's first destination element.</param>
        /// <param name="dstStride">The destination byte stride.</param>
        /// <param name="mask">The run's first mask byte.</param>
        /// <param name="maskStride">The mask byte stride (0: one byte decides the run).</param>
        /// <param name="n">The number of elements.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void MaskedAssignRun16(
            byte* src, long srcStride, byte* dst, long dstStride, byte* mask, long maskStride, long n)
        {
            if (maskStride == 0)
            {
                if (*mask == 0)
                    return;
                if (dstStride == 16 && srcStride == 16)
                {
                    long bytes = n * 16;
                    Buffer.MemoryCopy(src, dst, bytes, bytes);
                    return;
                }
                // Both halves read before either is written (the source element may be the destination element).
                for (long k = 0; k < n; k++)
                {
                    ulong lo = ((ulong*)src)[0], hi = ((ulong*)src)[1];
                    ((ulong*)dst)[0] = lo;
                    ((ulong*)dst)[1] = hi;
                    src += srcStride;
                    dst += dstStride;
                }
                return;
            }

            long i = 0;
            if (dstStride == 16 && maskStride == 1 && Avx2.IsSupported && Vector256.IsHardwareAccelerated)
            {
                // Two elements per 256-bit vector of four 8-byte lanes; each mask byte is duplicated onto its element's
                // two lanes before the widening (bytes m0 m0 m1 m1 → lanes m0 m0 m1 m1).
                ulong* d = (ulong*)dst;
                if (srcStride == 16)
                {
                    ulong* s = (ulong*)src;
                    for (; i <= n - 2; i += 2)
                    {
                        Vector256<ulong> keep = MaskedAssignKeepPairs256(mask + i);
                        Vector256.ConditionalSelect(keep, Vector256.Load(d + 2 * i), Vector256.Load(s + 2 * i)).Store(d + 2 * i);
                    }
                }
                else if (srcStride == 0)
                {
                    Vector256<ulong> v = Vector256.Create(((ulong*)src)[0], ((ulong*)src)[1], ((ulong*)src)[0], ((ulong*)src)[1]);
                    for (; i <= n - 2; i += 2)
                    {
                        Vector256<ulong> keep = MaskedAssignKeepPairs256(mask + i);
                        Vector256.ConditionalSelect(keep, Vector256.Load(d + 2 * i), v).Store(d + 2 * i);
                    }
                }
            }

            // The strided runs and the tail: a bit select of both halves under one lane mask (no branch to mispredict), off
            // running pointers. Both source halves are read before either destination half is written (the source may be
            // the destination element itself).
            byte* sp = src + i * srcStride;
            byte* dp = dst + i * dstStride;
            byte* mp = mask + i * maskStride;
            for (; i < n; i++)
            {
                ulong sel = MaskedAssignLaneMask(*mp);
                ulong slo = ((ulong*)sp)[0], shi = ((ulong*)sp)[1];
                ulong dlo = ((ulong*)dp)[0], dhi = ((ulong*)dp)[1];
                ((ulong*)dp)[0] = dlo ^ ((slo ^ dlo) & sel);
                ((ulong*)dp)[1] = dhi ^ ((shi ^ dhi) & sel);
                sp += srcStride;
                dp += dstStride;
                mp += maskStride;
            }
        }

        /// <summary>
        /// The "keep the destination" selector of two 16-byte elements: four 8-byte lanes, all ones where the element's
        /// mask byte is zero.
        /// </summary>
        /// <param name="mask">The two mask bytes.</param>
        /// <returns>The selector for one 256-bit vector of two elements.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<ulong> MaskedAssignKeepPairs256(byte* mask)
        {
            uint pairs = (uint)mask[0] * 0x0101u | ((uint)mask[1] * 0x0101u) << 16;
            return Vector256.Equals(
                Avx2.ConvertToVector256Int64(Vector128.CreateScalarUnsafe(pairs).AsByte()).AsUInt64(),
                Vector256<ulong>.Zero);
        }
    }
}
