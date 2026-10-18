using System;
using System.Reflection;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.PolyChebPoints.cs — the Chebyshev node fills (plan U8, numpy-polynomial-remaining-compute.md §3.3)
// =============================================================================
//
// chebpts1, chebpts2 and chebgauss each build their nodes from an integer ramp and ONE trigonometric ufunc
// (numpy/polynomial/chebyshev.py, NumPy 2.4.2):
//
//   chebpts1(n):    np.sin(0.5*np.pi/n * np.arange(-n + 1, n + 1, 2))
//   chebpts2(n):    np.cos(np.linspace(-np.pi, 0, n))          linspace: arange(0, n) * (pi/(n-1)) + (-pi), y[-1] = 0
//   chebgauss(deg): np.cos(np.pi * np.arange(1, 2*deg, 2) / (2.0*deg))
//
// NumPy materializes the ramp, every intermediate and the result (three to five arrays); this file emits one fill per
// formula that writes the result and nothing else. The ramp values are exact integers in float64 (|k| < 2^53 for every
// count that can be allocated), so a running double counter (v += step) IS NumPy's int64 ramp converted to float64, and
// each element is NumPy's statement chain on that value in NumPy's order — the same IEEE operations, then Math.Sin /
// Math.Cos, which is the C runtime function NumPy's float64 sin / cos loops call on win-amd64 (and the house np.sin /
// np.cos kernels call). The fills are not unrolled: the C runtime call dominates each element (about 4.5 ns), and a
// wider body measured no faster (the arcsinh finding in CLAUDE.md).
//
// chebpts1's ramp is ANTISYMMETRIC (a[n-1-k] = -a[k]) and IEEE multiplication commutes with negation, so its argument
// pairs are exact negatives of each other; the C runtime's sin is an odd function bit for bit, so the fill computes the
// lower half and stores each result's negation in the mirrored slot — half the sin calls, the same bytes as one sin per
// element. Measured before relying on it: no exception on win-amd64's ucrtbase over every chebpts1 argument up to 5,000
// points (12.5M), 32M evenly spaced bit patterns of [0, pi/2] and 128M random values up to +-1e300, nor on glibc (WSL,
// 7.5M values, NumPy's own np.sin included). The unit test Nodes_AreNumPysStatementChain compares every element with
// Math.Sin of its own argument, so a host whose sin is not odd fails it instead of silently differing. chebpts2 and
// chebgauss have no such pairs: their arguments are rounded ON THE WAY to the mirror (fl(k*s) - pi, fl(pi*(2k + 1))/d),
// so no two of them are exact negatives or exact pi-complements.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     The formula a <see cref="PolyChebNodesKernel"/> fills with (one kernel per member).
    /// </summary>
    internal enum PolyChebNodes
    {
        /// <summary>
        ///     <c>chebpts1</c>: <c>dst[k] = sin(scale * v)</c> with <c>v = first + k*step</c> — NumPy's
        ///     <c>np.sin(c * np.arange(-n + 1, n + 1, 2))</c> for <c>first = 1 - n</c>, <c>step = 2</c>,
        ///     <c>scale = 0.5*np.pi/n</c> (the Python float times the int64 ramp converted to float64). The ramp MUST be
        ///     antisymmetric (<c>first = -(count - 1)*step/2</c>, as chebpts1's is): the fill computes the lower half and
        ///     stores <c>dst[count - 1 - k] = -dst[k]</c> (see the file header), so another ramp would get a wrong upper half.
        /// </summary>
        Pts1 = 0,

        /// <summary>
        ///     <c>chebpts2</c>: <c>dst[k] = cos(k*step + first)</c> — NumPy's linspace body
        ///     <c>(np.arange(0, n) * step) + start</c> under <c>np.cos</c>, for <c>first = -np.pi</c>,
        ///     <c>step = np.pi/(n - 1)</c>. The caller fills the first <c>n - 1</c> elements this way and stores the
        ///     endpoint itself (NumPy's separate <c>y[-1, ...] = stop</c>); the kernel's <c>scale</c> argument is unused.
        /// </summary>
        Pts2 = 1,

        /// <summary>
        ///     <c>chebgauss</c>: <c>dst[k] = cos((pi * v) / scale)</c> with <c>v = first + k*step</c> — NumPy's
        ///     <c>np.cos(np.pi * np.arange(1, 2*deg, 2) / (2.0*deg))</c> for <c>first = 1</c>, <c>step = 2</c>,
        ///     <c>scale = 2.0*deg</c>.
        /// </summary>
        Gauss = 2,
    }

    /// <summary>
    ///     One Chebyshev node fill (see <see cref="PolyChebNodes"/> for each formula): <paramref name="count"/> float64
    ///     results written contiguously from <paramref name="dst"/>, element <c>k</c> computed from the exact ramp value
    ///     <c>first + k*step</c> (or <c>k</c> itself, for <see cref="PolyChebNodes.Pts2"/>).
    /// </summary>
    /// <param name="dst">Element 0 of <paramref name="count"/> contiguous float64 slots, every one written.</param>
    /// <param name="count">Element count (zero or less writes nothing).</param>
    /// <param name="first">The ramp's first value (<see cref="PolyChebNodes.Pts1"/>, <see cref="PolyChebNodes.Gauss"/>) or
    ///     the added start (<see cref="PolyChebNodes.Pts2"/>).</param>
    /// <param name="step">The ramp's step, or linspace's step for <see cref="PolyChebNodes.Pts2"/>.</param>
    /// <param name="scale">The multiplier (<see cref="PolyChebNodes.Pts1"/>) or divisor (<see cref="PolyChebNodes.Gauss"/>).</param>
    /// <remarks>
    ///     The running ramp value is advanced by addition, which stays exact while every value is an integer below
    ///     <c>2^53</c> in magnitude — true for every count an allocation can hold (2^53 doubles are 64 PiB). A caller
    ///     passing a non-integral <paramref name="first"/> or <paramref name="step"/> would get the rounding of repeated
    ///     addition, not of <c>first + k*step</c>: this kernel is for NumPy's integer ramps only.
    /// </remarks>
    public unsafe delegate void PolyChebNodesKernel(double* dst, long count, double first, double step, double scale);

    public static partial class DirectILKernelGenerator
    {
        /// <summary>The cached node fills, indexed by <see cref="PolyChebNodes"/> (a racing first call may compile one twice — the same code).</summary>
        private static readonly PolyChebNodesKernel[] s_polyChebNodes = new PolyChebNodesKernel[3];

        /// <summary>
        ///     The node fill for <paramref name="kind"/> (see <see cref="PolyChebNodesKernel"/>), compiled on first use.
        /// </summary>
        /// <param name="kind">The formula.</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="InvalidOperationException">IL generation is disabled (<see cref="Enabled"/> is false).</exception>
        /// <exception cref="PlatformNotSupportedException">The runtime cannot emit dynamic code (NativeAOT).</exception>
        internal static PolyChebNodesKernel GetPolyChebNodesKernel(PolyChebNodes kind)
        {
            if (Volatile.Read(ref s_polyChebNodes[(int)kind]) is { } fast)
                return fast;
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");
            var k = EmitPolyChebNodes(kind);
            Volatile.Write(ref s_polyChebNodes[(int)kind], k);
            return k;
        }

        /// <summary>Emits the <see cref="PolyChebNodesKernel"/> for <paramref name="kind"/>.</summary>
        /// <param name="kind">The formula.</param>
        /// <returns>The kernel.</returns>
        private static unsafe PolyChebNodesKernel EmitPolyChebNodes(PolyChebNodes kind)
        {
            if (kind == PolyChebNodes.Pts1)
                return EmitPolyChebPts1Mirrored();

            // args: 0 dst, 1 count, 2 first, 3 step, 4 scale
            var dm = new DynamicMethod("NDPolyChebNodes_" + kind, typeof(void),
                new[] { typeof(double*), typeof(long), typeof(double), typeof(double), typeof(double) },
                typeof(DirectILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();
            var p = il.DeclareLocal(typeof(double*));
            var left = il.DeclareLocal(typeof(long));
            var v = il.DeclareLocal(typeof(double));
            var loop = il.DefineLabel();
            var done = il.DefineLabel();
            MethodInfo cos = ScalarMethodCache.MathFn1(typeof(double), "Cos");

            // p = dst; left = count; v = the first ramp value (the index 0 for Pts2, whose ramp is np.arange(0, n)).
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, p);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, left);
            if (kind == PolyChebNodes.Pts2)
                il.Emit(OpCodes.Ldc_R8, 0.0);
            else
                il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Stloc, v);

            // while (left > 0) — a SIGNED down-counter, so a count of zero or less writes nothing.
            il.MarkLabel(loop);
            il.Emit(OpCodes.Ldloc, left);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, done);

            il.Emit(OpCodes.Ldloc, p);
            if (kind == PolyChebNodes.Pts2)
            {
                // cos(v*step + first): linspace's `y *= step` then `y += start`, two roundings (no fused multiply-add —
                // NumPy runs them as separate in-place ufunc passes, and RyuJIT never contracts a separate mul and add).
                il.Emit(OpCodes.Ldloc, v);
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Add);
            }
            else
            {
                // cos((pi * v) / scale): `np.pi * np.arange(...)`, then the true division by the Python float 2.0*deg.
                il.Emit(OpCodes.Ldc_R8, Math.PI);
                il.Emit(OpCodes.Ldloc, v);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Ldarg_S, (byte)4);
                il.Emit(OpCodes.Div);
            }
            il.EmitCall(OpCodes.Call, cos, null);
            il.Emit(OpCodes.Stind_R8);

            // v += step (Gauss: the next ramp value) or v += 1 (Pts2: the next index) — exact integer arithmetic.
            il.Emit(OpCodes.Ldloc, v);
            if (kind == PolyChebNodes.Pts2)
                il.Emit(OpCodes.Ldc_R8, 1.0);
            else
                il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, v);

            // p++; left--
            il.Emit(OpCodes.Ldloc, p);
            il.Emit(OpCodes.Ldc_I4, sizeof(double));
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
            return dm.CreateDelegate<PolyChebNodesKernel>();
        }

        /// <summary>
        ///     Emits the <see cref="PolyChebNodes.Pts1"/> fill over an antisymmetric ramp: one <c>s = sin(scale * v)</c> per
        ///     PAIR of slots, stored at the low slot and negated at its mirror, then the middle element of an odd count
        ///     computed by its own sin (it is the ramp's zero, <c>sin(+0.0)</c>, never a negation).
        /// </summary>
        /// <returns>The kernel.</returns>
        /// <remarks>
        ///     Bit-identical to one <c>sin(scale * v)</c> per element because the mirrored argument is the exact negation
        ///     (<c>scale * -v == -(scale * v)</c> in IEEE arithmetic) and the C runtime's sin is odd bit for bit (measured, see
        ///     the file header). The IEEE <c>neg</c> flips the sign bit only, so it cannot round.
        /// </remarks>
        private static unsafe PolyChebNodesKernel EmitPolyChebPts1Mirrored()
        {
            // args: 0 dst, 1 count, 2 first, 3 step, 4 scale
            var dm = new DynamicMethod("NDPolyChebNodes_Pts1", typeof(void),
                new[] { typeof(double*), typeof(long), typeof(double), typeof(double), typeof(double) },
                typeof(DirectILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();
            var lo = il.DeclareLocal(typeof(double*));
            var hi = il.DeclareLocal(typeof(double*));
            var pairs = il.DeclareLocal(typeof(long));
            var v = il.DeclareLocal(typeof(double));
            var s = il.DeclareLocal(typeof(double));
            var loop = il.DefineLabel();
            var middle = il.DefineLabel();
            var done = il.DefineLabel();
            MethodInfo sin = ScalarMethodCache.MathFn1(typeof(double), "Sin");

            // A count of zero or less writes nothing (checked first: the odd-count middle below must not fire for -3).
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, done);

            // lo = dst; hi = dst + (count - 1); pairs = count >> 1; v = first (the ramp's most negative value)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, lo);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, (long)sizeof(double));
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, hi);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Shr);
            il.Emit(OpCodes.Stloc, pairs);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Stloc, v);

            // while (pairs > 0) { s = sin(scale * v); *lo = s; *hi = -s; v += step; lo++; hi--; pairs--; }
            il.MarkLabel(loop);
            il.Emit(OpCodes.Ldloc, pairs);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, middle);
            // sin(scale * v): NumPy's np.multiply(c, ramp) — the Python float first — then np.sin.
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldloc, v);
            il.Emit(OpCodes.Mul);
            il.EmitCall(OpCodes.Call, sin, null);
            il.Emit(OpCodes.Stloc, s);
            il.Emit(OpCodes.Ldloc, lo);
            il.Emit(OpCodes.Ldloc, s);
            il.Emit(OpCodes.Stind_R8);
            il.Emit(OpCodes.Ldloc, hi);
            il.Emit(OpCodes.Ldloc, s);
            il.Emit(OpCodes.Neg);
            il.Emit(OpCodes.Stind_R8);
            il.Emit(OpCodes.Ldloc, v);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, v);
            il.Emit(OpCodes.Ldloc, lo);
            il.Emit(OpCodes.Ldc_I4, sizeof(double));
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, lo);
            il.Emit(OpCodes.Ldloc, hi);
            il.Emit(OpCodes.Ldc_I4, sizeof(double));
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, hi);
            il.Emit(OpCodes.Ldloc, pairs);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, pairs);
            il.Emit(OpCodes.Br, loop);

            // An odd count's middle slot (lo == hi now): its own sin of the ramp's zero.
            il.MarkLabel(middle);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.And);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Beq, done);
            il.Emit(OpCodes.Ldloc, lo);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldloc, v);
            il.Emit(OpCodes.Mul);
            il.EmitCall(OpCodes.Call, sin, null);
            il.Emit(OpCodes.Stind_R8);

            il.MarkLabel(done);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyChebNodesKernel>();
        }
    }
}
