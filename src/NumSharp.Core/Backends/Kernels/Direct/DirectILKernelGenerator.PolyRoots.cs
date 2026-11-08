using System;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.PolyRoots.cs — the one kernel numpy.polynomial's companion matrices add (plan U7)
// =============================================================================
//
// {p}companion builds an (n, n) matrix from a handful of NumPy statements (Polynomial/Package/NDPolyAlgebra.Roots.cs):
// np.zeros, a few diagonals assigned through `mat.reshape(-1)[k::n+1]`, helper float64 vectors such as
// `1. / np.sqrt(2 * np.arange(n) + 1)`, and one in-place update of the last column. Every element operation of those
// statements already has a house kernel — the binary ufunc loops (GetPolyHouseBinaryKernel), the dtype conversions
// (GetPolyCastKernel), np.sqrt's unary loop (GetUnaryKernel), np.multiply.accumulate (GetCumulativeKernel) and the
// strided diagonal store (GetDiagWriteKernel) — except the integer ramps np.arange makes, which this file adds.
//
// NumPy evaluates every ramp-derived helper in exact integer or float64 arithmetic (2*k + 1, -(k + 1), 2.0*k for k
// below 2^52), so the ramp is generated in int64 and converted by the house cast exactly where NumPy's ufunc loop
// converts its int64 operand — a value-for-value replay, not an approximation of one.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     NumPy's <c>np.arange</c> over int64 as a raw fill: <c>dst[k] = start + k*step</c> for k in [0, n), the running
    ///     value advanced by addition (no per-element multiply).
    /// </summary>
    /// <param name="dst">Element 0 of <paramref name="n"/> contiguous int64 slots, every one written.</param>
    /// <param name="n">Element count (≤ 0 writes nothing).</param>
    /// <param name="start">The first value.</param>
    /// <param name="step">The difference between consecutive values (any sign; the caller keeps the run inside int64).</param>
    public unsafe delegate void PolyRampKernel(long* dst, long n, long start, long step);

    public static partial class DirectILKernelGenerator
    {
        /// <summary>The cached ramp kernel (one for the process; a racing first call may compile twice — the same code).</summary>
        private static PolyRampKernel s_polyRamp;

        /// <summary>
        ///     The int64 ramp kernel (see <see cref="PolyRampKernel"/>): the <c>np.arange</c> statements of the companion
        ///     builders (<c>np.arange(1, n)</c>, <c>2 * np.arange(n) + 1</c>, <c>np.arange(n - 1, 0, -1)</c>) as one
        ///     running-sum loop, its values converted afterwards where NumPy's ufunc converts them.
        /// </summary>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="InvalidOperationException">IL generation is disabled (<see cref="Enabled"/> is false).</exception>
        /// <exception cref="PlatformNotSupportedException">The runtime cannot emit dynamic code (NativeAOT).</exception>
        internal static PolyRampKernel GetPolyRampKernel()
        {
            if (Volatile.Read(ref s_polyRamp) is { } fast)
                return fast;
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");
            var k = EmitPolyRamp();
            Volatile.Write(ref s_polyRamp, k);
            return k;
        }

        /// <summary>Emits <see cref="PolyRampKernel"/>.</summary>
        /// <returns>The kernel.</returns>
        private static unsafe PolyRampKernel EmitPolyRamp()
        {
            // args: 0 dst, 1 n, 2 start, 3 step
            var dm = new DynamicMethod("NDPolyRamp_Int64", typeof(void),
                new[] { typeof(long*), typeof(long), typeof(long), typeof(long) },
                typeof(DirectILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();
            var p = il.DeclareLocal(typeof(long*));
            var left = il.DeclareLocal(typeof(long));
            var v = il.DeclareLocal(typeof(long));
            var loop = il.DefineLabel();
            var done = il.DefineLabel();

            // p = dst; left = n; v = start
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, p);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, left);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Stloc, v);

            // while (left > 0) { *p = v; v += step; p++; left--; } — a SIGNED down-counter, so a count of zero or less
            // writes nothing (an end-pointer compare would wrap for a negative count and run off the buffer).
            il.MarkLabel(loop);
            il.Emit(OpCodes.Ldloc, left);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, done);
            il.Emit(OpCodes.Ldloc, p);
            il.Emit(OpCodes.Ldloc, v);
            il.Emit(OpCodes.Stind_I8);
            il.Emit(OpCodes.Ldloc, v);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, v);
            il.Emit(OpCodes.Ldloc, p);
            il.Emit(OpCodes.Ldc_I4, sizeof(long));
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, p);
            il.Emit(OpCodes.Ldloc, left);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, left);
            il.Emit(OpCodes.Br, loop);
            il.MarkLabel(done);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyRampKernel>();
        }
    }
}
