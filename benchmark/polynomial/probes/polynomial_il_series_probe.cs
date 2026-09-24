#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Optimize=true
// =============================================================================
// polynomial_il_series_probe.cs — the house-compliant engine for numpy.polynomial's SMALL-SERIES algebra
// (U1 add/sub/trim, U2 mul/div/pow/fromroots/conversions, U4 der/int), measured against NumPy 2.4.2.
//
// NumPy's small-series routines are Python loops over tiny arrays (6-51 coefficients): legmul(50, 50) runs
// ~50 iterations of legmulx/legadd/legsub, each allocating fresh arrays. Driving an NDIter per op is at or
// above NumPy's own per-op cost at these sizes, so the engine here is:
//   * dtype-generic IL PRIMITIVE kernels (trim length, scale, mul-div, in-place add/sub/neg, legmulx,
//     scalar divide) — every element loop is IL, emitted once per dtype through the house
//     EmitScalarOperation / EmitComparisonOperation, so NumPy's per-dtype op semantics (complex array-loop
//     multiply, float16 widen-compute-narrow, ...) come from the same emitters the ufuncs use;
//   * WHOLE-ROUTINE IL kernels where the routine is one flat recurrence (chebder: every loop is IL);
//   * dtype-AGNOSTIC orchestration (byte pointers + element size) that mirrors NumPy's Python line for
//     line — its only loops are NumPy's own Python-level loops over SERIES (e.g. legmul's i-loop), never
//     over elements;
//   * a bump ARENA for every intermediate series (one reused native block per call; production: one per
//     thread), so an intermediate costs a pointer bump instead of an NDArray.
// Every cell is byte-compared with NumPy's own output (written by polynomial_il_numpy.py, section S)
// before it is timed. The verdict and the tables live in docs/plans/numpy-polynomial.md §10.
//
// Usage (from this directory; generate data + NumPy timings first with the twin):
//   set NS_PROBE_AFFINITY=0xFF
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polynomial_il_numpy.py S > numpy_il_series_times.tsv
//   dotnet run -c Release polynomial_il_series_probe.cs -- data/il numpy_il_series_times.tsv
// Output TSV: cell, variant, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), byte check.
// AssemblyName=NumSharp.DotNetRunScript grants the InternalsVisibleTo the IL helpers need (the repo's
// Directory.Build.props signs the script with the matching key). PublishAot=false is load-bearing:
// without dynamic code there is no DynamicMethod.
// =============================================================================
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
#nullable disable

// Pin before anything is JIT-ed (several logical CPUs, so the tiered-JIT background compiler is not starved).
if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } affinity)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(affinity, 16);

Series.Data = args.Length > 0 ? args[0] : Path.Combine("data", "il");
if (args.Length > 1) Series.LoadNumPyTimes(args[1]);
Series.Run();

/// <summary>pu.trimseq length: index of the last non-zero coefficient + 1 (1 when all are zero).</summary>
unsafe delegate long TrimLenFn(byte* p, long n, byte* zero);
/// <summary>dst[q] = f(src[q], k0, k1) over n elements.</summary>
unsafe delegate void Map2Fn(byte* dst, byte* src, long n, byte* k0, byte* k1);
/// <summary>In-place dst[q] = dst[q] op src[q] (or a unary op on dst).</summary>
unsafe delegate void InPlaceFn(byte* dst, byte* src, long n);
/// <summary>A whole-routine kernel: dst from src (length n) with a constant pool.</summary>
unsafe delegate void RoutineFn(byte* dst, byte* src, long n, byte* pool);

/// <summary>
///     The probe body: the IL primitives, the arena, the NumPy-mirroring orchestration and the harness.
/// </summary>
static unsafe class Series
{
    /// <summary>Directory holding the .npy inputs and NumPy references written by the twin.</summary>
    public static string Data = Path.Combine("data", "il");

    // ======================================================================== IL primitives (dtype-generic)

    static readonly ConcurrentDictionary<string, Delegate> s_cache = new();

    /// <summary>Compiles (once per key) a DynamicMethod owned by the kernel generator's module, so the IL can
    /// call the house helpers it references.</summary>
    /// <typeparam name="TDel">The delegate type.</typeparam>
    /// <param name="key">Cache key (primitive + dtype).</param><param name="ret">Return type.</param>
    /// <param name="ps">Parameter types.</param><param name="body">IL body.</param><returns>The delegate.</returns>
    static TDel Emit<TDel>(string key, Type ret, Type[] ps, Action<ILGenerator> body) where TDel : Delegate
        => (TDel)s_cache.GetOrAdd(key, _ =>
        {
            var dm = new DynamicMethod("polyseries_" + key, ret, ps, typeof(DirectILKernelGenerator), skipVisibility: true);
            body(dm.GetILGenerator());
            return dm.CreateDelegate<TDel>();
        });

    /// <summary>Pushes <c>ptrArg + idx * sz</c>.</summary>
    /// <param name="il">Generator.</param><param name="ptrArg">The ldarg opcode of the base pointer.</param>
    /// <param name="idx">Index local.</param><param name="sz">Element size.</param>
    static void Elem(ILGenerator il, OpCode ptrArg, LocalBuilder idx, int sz)
    {
        il.Emit(ptrArg); il.Emit(OpCodes.Ldloc, idx); il.Emit(OpCodes.Ldc_I8, (long)sz); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
    }

    /// <summary>Emits <c>for (q = 0; q &lt; arg2; q++) body(q)</c>.</summary>
    /// <param name="il">Generator.</param><param name="body">Body given the index local.</param>
    static void ForN(ILGenerator il, Action<LocalBuilder> body)
    {
        var q = il.DeclareLocal(typeof(long)); var top = il.DefineLabel(); var done = il.DefineLabel();
        il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, q);
        il.MarkLabel(top);
        il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, done);
        body(q);
        il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, q);
        il.Emit(OpCodes.Br, top);
        il.MarkLabel(done);
    }

    /// <summary><c>pu.trimseq</c>'s length: scan down for the last coefficient that is not equal to zero
    /// (NumPy's <c>seq[i] != 0</c> — NaN counts as non-zero, -0.0 as zero), keep one when all are zero.</summary>
    /// <param name="t">Dtype.</param><returns>The kernel.</returns>
    static TrimLenFn TrimLen(NPTypeCode t) => Emit<TrimLenFn>($"trim_{t}", typeof(long), new[] { typeof(byte*), typeof(long), typeof(byte*) }, il =>
    {
        int sz = DirectILKernelGenerator.GetTypeSize(t);
        var i = il.DeclareLocal(typeof(long));
        var top = il.DefineLabel(); var found = il.DefineLabel(); var none = il.DefineLabel();
        il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, i);
        il.MarkLabel(top);
        il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Blt, none);
        Elem(il, OpCodes.Ldarg_0, i, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        il.Emit(OpCodes.Ldarg_2); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitComparisonOperation(il, ComparisonOp.NotEqual, t);
        il.Emit(OpCodes.Brtrue, found);
        il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, i);
        il.Emit(OpCodes.Br, top);
        il.MarkLabel(found); il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Ret);
        il.MarkLabel(none); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Ret);
    });

    /// <summary>
    ///     The two array expressions the recurrence bases use, each in NumPy's operand order:
    ///     mode 0 <c>k0 * src[q]</c> (<c>c[-i]*xs</c>: numpy scalar times array → the array loop);
    ///     mode 1 <c>(src[q] * k0) / k1</c> (<c>(c1*(nd-1))/nd</c>: two array ops with Python ints).
    /// </summary>
    /// <param name="t">Dtype.</param><param name="mode">0 or 1.</param><returns>The kernel.</returns>
    static Map2Fn Map(NPTypeCode t, int mode) => Emit<Map2Fn>($"map{mode}_{t}", typeof(void),
        new[] { typeof(byte*), typeof(byte*), typeof(long), typeof(byte*), typeof(byte*) }, il =>
    {
        int sz = DirectILKernelGenerator.GetTypeSize(t);
        ForN(il, q =>
        {
            Elem(il, OpCodes.Ldarg_0, q, sz);
            if (mode == 0)
            {
                il.Emit(OpCodes.Ldarg_3); DirectILKernelGenerator.EmitLoadIndirect(il, t);
                Elem(il, OpCodes.Ldarg_1, q, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
                DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, t);
            }
            else
            {
                Elem(il, OpCodes.Ldarg_1, q, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
                il.Emit(OpCodes.Ldarg_3); DirectILKernelGenerator.EmitLoadIndirect(il, t);
                DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, t);
                il.Emit(OpCodes.Ldarg, (short)4); DirectILKernelGenerator.EmitLoadIndirect(il, t);
                DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Divide, t);
            }
            DirectILKernelGenerator.EmitStoreIndirect(il, t);
        });
        il.Emit(OpCodes.Ret);
    });

    /// <summary>In-place <c>dst[q] = dst[q] + src[q]</c> (op 0), <c>- src[q]</c> (op 1) or <c>dst[q] = -dst[q]</c> (op 2)
    /// — <c>pu._add</c>/<c>pu._sub</c>'s <c>c1[:n] += c2</c> / <c>c2 = -c2</c>.</summary>
    /// <param name="t">Dtype.</param><param name="op">0 add, 1 subtract, 2 negate.</param><returns>The kernel.</returns>
    static InPlaceFn InPlace(NPTypeCode t, int op) => Emit<InPlaceFn>($"inplace{op}_{t}", typeof(void),
        new[] { typeof(byte*), typeof(byte*), typeof(long) }, il =>
    {
        int sz = DirectILKernelGenerator.GetTypeSize(t);
        ForN(il, q =>
        {
            Elem(il, OpCodes.Ldarg_0, q, sz);
            Elem(il, OpCodes.Ldarg_0, q, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
            if (op == 2) DirectILKernelGenerator.EmitUnaryScalarOperation(il, UnaryOp.Negate, t);
            else
            {
                Elem(il, OpCodes.Ldarg_1, q, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
                DirectILKernelGenerator.EmitScalarOperation(il, op == 0 ? BinaryOp.Add : BinaryOp.Subtract, t);
            }
            DirectILKernelGenerator.EmitStoreIndirect(il, t);
        });
        il.Emit(OpCodes.Ret);
    });

    /// <summary>
    ///     <c>legmulx</c>'s body for a non-zero series of length n: <c>prd[0] = c[0]*0; prd[1] = c[0];</c> then
    ///     for each i: <c>prd[i+1] = (c[i]*j)/s; prd[i-1] += (c[i]*i)/s</c> (j = i+1, s = i+j). The weak Python
    ///     ints come from the pool (<c>pool[v] = T(v)</c>).
    /// </summary>
    /// <param name="t">Dtype.</param><returns>The kernel.</returns>
    static RoutineFn LegMulx(NPTypeCode t) => Emit<RoutineFn>($"legmulx_{t}", typeof(void),
        new[] { typeof(byte*), typeof(byte*), typeof(long), typeof(byte*) }, il =>
    {
        int sz = DirectILKernelGenerator.GetTypeSize(t);
        var i = il.DeclareLocal(typeof(long)); var j = il.DeclareLocal(typeof(long)); var k = il.DeclareLocal(typeof(long));
        var s = il.DeclareLocal(typeof(long)); var zero = il.DeclareLocal(typeof(long));
        var top = il.DefineLabel(); var done = il.DefineLabel();
        il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, zero);
        il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Stloc, i);
        // prd[0] = c[0] * 0   (keeps NaN/inf/-0 exactly as NumPy's array op does)
        Elem(il, OpCodes.Ldarg_0, zero, sz);
        Elem(il, OpCodes.Ldarg_1, zero, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        Elem(il, OpCodes.Ldarg_3, zero, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, t);
        DirectILKernelGenerator.EmitStoreIndirect(il, t);
        // prd[1] = c[0]
        il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldc_I4, sz); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
        Elem(il, OpCodes.Ldarg_1, zero, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitStoreIndirect(il, t);
        il.MarkLabel(top);
        il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, done);
        il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, j);
        il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, k);
        il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, s);
        // prd[j] = (c[i] * j) / s
        Elem(il, OpCodes.Ldarg_0, j, sz);
        Elem(il, OpCodes.Ldarg_1, i, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        Elem(il, OpCodes.Ldarg_3, j, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, t);
        Elem(il, OpCodes.Ldarg_3, s, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Divide, t);
        DirectILKernelGenerator.EmitStoreIndirect(il, t);
        // prd[k] = prd[k] + (c[i] * i) / s
        Elem(il, OpCodes.Ldarg_0, k, sz);
        Elem(il, OpCodes.Ldarg_0, k, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        Elem(il, OpCodes.Ldarg_1, i, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        Elem(il, OpCodes.Ldarg_3, i, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, t);
        Elem(il, OpCodes.Ldarg_3, s, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Divide, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Add, t);
        DirectILKernelGenerator.EmitStoreIndirect(il, t);
        il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, i);
        il.Emit(OpCodes.Br, top);
        il.MarkLabel(done); il.Emit(OpCodes.Ret);
    });

    /// <summary>
    ///     <c>chebder</c> (m = 1, scl = 1) as ONE kernel over a mutable copy <c>c</c> of length n+1 — every loop
    ///     of the routine is IL: <c>c *= 1</c>; for j = n..3: <c>der[j-1] = (2*j)*c[j]; c[j-2] += (j*c[j])/(j-2)</c>;
    ///     then <c>der[1] = 4*c[2]</c> (n &gt; 1) and <c>der[0] = c[1]</c>.
    /// </summary>
    /// <param name="t">Dtype.</param><returns>The kernel.</returns>
    static RoutineFn ChebDer(NPTypeCode t) => Emit<RoutineFn>($"chebder_{t}", typeof(void),
        new[] { typeof(byte*), typeof(byte*), typeof(long), typeof(byte*) }, il =>
    {
        int sz = DirectILKernelGenerator.GetTypeSize(t);
        var j = il.DeclareLocal(typeof(long)); var q = il.DeclareLocal(typeof(long)); var tmp = il.DeclareLocal(typeof(long));
        var one = il.DeclareLocal(typeof(long));
        il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Stloc, one);
        // c *= scl (scl = 1): kept because NumPy performs it — the multiply quiets a signalling NaN.
        var l0 = il.DefineLabel(); var l0e = il.DefineLabel();
        il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, q);
        il.MarkLabel(l0); il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bgt, l0e);
        Elem(il, OpCodes.Ldarg_1, q, sz); Elem(il, OpCodes.Ldarg_1, q, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        Elem(il, OpCodes.Ldarg_3, one, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, t); DirectILKernelGenerator.EmitStoreIndirect(il, t);
        il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, q); il.Emit(OpCodes.Br, l0);
        il.MarkLabel(l0e);
        var top = il.DefineLabel(); var done = il.DefineLabel();
        il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Stloc, j);
        il.MarkLabel(top); il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Ble, done);
        // der[j-1] = (2*j) * c[j]
        il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, q);
        Elem(il, OpCodes.Ldarg_0, q, sz);
        il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Stloc, tmp);
        Elem(il, OpCodes.Ldarg_3, tmp, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        Elem(il, OpCodes.Ldarg_1, j, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, t);
        DirectILKernelGenerator.EmitStoreIndirect(il, t);
        // c[j-2] = c[j-2] + (j*c[j])/(j-2)
        il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, q);
        Elem(il, OpCodes.Ldarg_1, q, sz);
        Elem(il, OpCodes.Ldarg_1, q, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        Elem(il, OpCodes.Ldarg_3, j, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        Elem(il, OpCodes.Ldarg_1, j, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, t);
        Elem(il, OpCodes.Ldarg_3, q, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Divide, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Add, t);
        DirectILKernelGenerator.EmitStoreIndirect(il, t);
        il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, j); il.Emit(OpCodes.Br, top);
        il.MarkLabel(done);
        // if n > 1: der[1] = 4 * c[2]
        var skip = il.DefineLabel();
        il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Ble, skip);
        il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Stloc, q); Elem(il, OpCodes.Ldarg_0, q, sz);
        il.Emit(OpCodes.Ldc_I8, 4L); il.Emit(OpCodes.Stloc, tmp); Elem(il, OpCodes.Ldarg_3, tmp, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Stloc, tmp); Elem(il, OpCodes.Ldarg_1, tmp, sz); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, t); DirectILKernelGenerator.EmitStoreIndirect(il, t);
        il.MarkLabel(skip);
        // der[0] = c[1]
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I4, sz); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitStoreIndirect(il, t);
        il.Emit(OpCodes.Ret);
    });

    /// <summary><c>dst = a / b</c> for two numpy scalars (<c>_div</c>'s <c>q = rem[-1] / p[-1]</c>). For complex,
    /// NumPy's scalarmath divide calls the CDOUBLE_divide LOOP, so the house array-loop divide is exact here.</summary>
    /// <param name="t">Dtype.</param><returns>The kernel (src and n unused).</returns>
    static Map2Fn ScalarDiv(NPTypeCode t) => Emit<Map2Fn>($"sdiv_{t}", typeof(void),
        new[] { typeof(byte*), typeof(byte*), typeof(long), typeof(byte*), typeof(byte*) }, il =>
    {
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ldarg_3); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        il.Emit(OpCodes.Ldarg, (short)4); DirectILKernelGenerator.EmitLoadIndirect(il, t);
        DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Divide, t);
        DirectILKernelGenerator.EmitStoreIndirect(il, t);
        il.Emit(OpCodes.Ret);
    });

    // ======================================================================== arena + orchestration (dtype-agnostic)

    /// <summary>
    ///     Bump arena: every intermediate series is a slice of one native block; <see cref="Mark"/> /
    ///     <see cref="Release"/> give the stack discipline <c>_div</c> needs (each divisor product dies before
    ///     the next). One arena is reused across calls (production: one per thread), so a steady-state call
    ///     allocates nothing but its NDArray result.
    /// </summary>
    sealed class Arena : IDisposable
    {
        readonly byte* _base; readonly long _cap; long _top;

        /// <summary>Allocates the block.</summary><param name="bytes">Capacity.</param>
        public Arena(long bytes) { _base = (byte*)NativeMemory.Alloc((nuint)bytes); _cap = bytes; }

        /// <summary>Bumps out a 16-byte-aligned slice.</summary><param name="bytes">Size.</param><returns>The slice.</returns>
        /// <exception cref="OutOfMemoryException">The arena is exhausted (a real implementation grows it).</exception>
        public byte* Alloc(long bytes)
        {
            var p = _base + _top;
            _top += (bytes + 15) & ~15L;
            if (_top > _cap) throw new OutOfMemoryException("series arena exhausted");
            return p;
        }

        /// <summary>The current top (for <see cref="Release"/>).</summary>
        public long Mark => _top;

        /// <summary>Frees everything allocated after <paramref name="mark"/>.</summary><param name="mark">A previous <see cref="Mark"/>.</param>
        public void Release(long mark) => _top = mark;

        /// <summary>Frees the block.</summary>
        public void Dispose() => NativeMemory.Free(_base);
    }

    /// <summary>A series: pointer + length (dtype carried by the context).</summary>
    readonly struct Ser
    {
        /// <summary>First coefficient.</summary>
        public readonly byte* P;
        /// <summary>Coefficient count.</summary>
        public readonly long N;
        /// <summary>Wraps a series.</summary><param name="p">Pointer.</param><param name="n">Length.</param>
        public Ser(byte* p, long n) { P = p; N = n; }
    }

    /// <summary>One call's dtype-bound primitives, arena and weak-int pool (<c>Pool + v*Sz</c> holds T(v)).</summary>
    sealed class Ctx
    {
        /// <summary>The series dtype and its element size.</summary>
        public NPTypeCode T; public int Sz;
        /// <summary>Scratch.</summary>
        public Arena A;
        /// <summary>Weak Python ints 0..255 materialized in T; <see cref="Zero"/> is T(0).</summary>
        public byte* Pool, Zero;
        /// <summary>The dtype-bound IL primitives.</summary>
        public TrimLenFn Trim; public Map2Fn Scale, MulDiv, SDiv; public InPlaceFn Add, Sub, Neg; public RoutineFn Mulx;

        /// <summary>Copies the first n coefficients into the arena.</summary>
        /// <param name="s">Source.</param><param name="n">Count.</param><returns>The copy.</returns>
        public Ser Copy(Ser s, long n) { var p = A.Alloc(n * Sz); Buffer.MemoryCopy(s.P, p, n * Sz, n * Sz); return new Ser(p, n); }

        /// <summary><c>pu.as_series</c> for one series of the context dtype: a trimmed copy.</summary>
        /// <param name="s">Source.</param><returns>The copy.</returns>
        public Ser AsSeries(Ser s) => Copy(s, Trim(s.P, s.N, Zero));

        /// <summary>Address of T(v), a weak Python int.</summary>
        /// <param name="v">The int (0..255).</param><returns>Its address.</returns>
        public byte* Const(long v) => Pool + v * Sz;
    }

    /// <summary><c>pu._add</c>: as_series both, add the shorter into the longer, trimseq.</summary>
    /// <param name="x">Context.</param><param name="a">First.</param><param name="b">Second.</param><returns>The sum.</returns>
    static Ser AddS(Ctx x, Ser a, Ser b)
    {
        a = x.AsSeries(a); b = x.AsSeries(b);
        Ser ret;
        if (a.N > b.N) { x.Add(a.P, b.P, b.N); ret = a; } else { x.Add(b.P, a.P, a.N); ret = b; }
        return new Ser(ret.P, x.Trim(ret.P, ret.N, x.Zero));
    }

    /// <summary><c>pu._sub</c>, including its negate-then-add branch (the bits of <c>-c2 + c1</c> are not always
    /// those of <c>c1 - c2</c>: the sign of a zero result differs).</summary>
    /// <param name="x">Context.</param><param name="a">Minuend.</param><param name="b">Subtrahend.</param><returns>The difference.</returns>
    static Ser SubS(Ctx x, Ser a, Ser b)
    {
        a = x.AsSeries(a); b = x.AsSeries(b);
        Ser ret;
        if (a.N > b.N) { x.Sub(a.P, b.P, b.N); ret = a; } else { x.Neg(b.P, null, b.N); x.Add(b.P, a.P, a.N); ret = b; }
        return new Ser(ret.P, x.Trim(ret.P, ret.N, x.Zero));
    }

    /// <summary><c>s * v</c> (numpy scalar times array).</summary>
    /// <param name="x">Context.</param><param name="s">Scalar address.</param><param name="v">Series.</param><returns>A new series.</returns>
    static Ser Scale(Ctx x, byte* s, Ser v) { var r = new Ser(x.A.Alloc(v.N * x.Sz), v.N); x.Scale(r.P, v.P, v.N, s, null); return r; }

    /// <summary><c>(v * m) / d</c> with Python ints m, d.</summary>
    /// <param name="x">Context.</param><param name="v">Series.</param><param name="m">Multiplier.</param><param name="d">Divisor.</param><returns>A new series.</returns>
    static Ser MulDiv(Ctx x, Ser v, long m, long d) { var r = new Ser(x.A.Alloc(v.N * x.Sz), v.N); x.MulDiv(r.P, v.P, v.N, x.Const(m), x.Const(d)); return r; }

    /// <summary><c>legmulx</c>: as_series, the zero-series early return, else the IL body.</summary>
    /// <param name="x">Context.</param><param name="c">Series.</param><returns>x times the series.</returns>
    static Ser LegMulxS(Ctx x, Ser c)
    {
        c = x.AsSeries(c);
        if (c.N == 1 && IsZero(x, c.P)) return c;   // `if len(c) == 1 and c[0] == 0: return c`
        var prd = new Ser(x.A.Alloc((c.N + 1) * x.Sz), c.N + 1);
        x.Mulx(prd.P, c.P, c.N, x.Pool);
        return prd;
    }

    /// <summary>NumPy's <c>c[0] == 0</c> through the dtype's own comparison: the trim kernel over the pair
    /// <c>[0, value]</c> returns 2 exactly when <c>value != 0</c> (NaN counts as non-zero, ±0 as zero).</summary>
    /// <param name="x">Context.</param><param name="p">Coefficient address.</param><returns>True when the coefficient equals zero.</returns>
    static bool IsZero(Ctx x, byte* p)
    {
        byte* pair = stackalloc byte[32];
        Buffer.MemoryCopy(x.Zero, pair, x.Sz, x.Sz);
        Buffer.MemoryCopy(p, pair + x.Sz, x.Sz, x.Sz);
        return x.Trim(pair, 2, x.Zero) == 1;
    }

    /// <summary><c>legmul</c>, line for line (the smaller series drives the recurrence; <c>xs</c> is the other).</summary>
    /// <param name="x">Context.</param><param name="a">First series.</param><param name="b">Second series.</param><returns>The product.</returns>
    /// <remarks>NumPy's <c>len(c) == 1</c> branch sets <c>c1 = 0</c>, a Python int that flows through
    /// <c>as_series</c> → <c>common_type</c> and makes the RESULT float64 for float16/float32 inputs (probed).
    /// This probe runs len &gt; 2 only; the implementation must carry per-series dtypes for that branch.</remarks>
    static Ser LegMul(Ctx x, Ser a, Ser b)
    {
        a = x.AsSeries(a); b = x.AsSeries(b);
        Ser c, xs;
        if (a.N > b.N) { c = b; xs = a; } else { c = a; xs = b; }
        long len = c.N;
        Ser c0, c1;
        byte* E(long idx) => c.P + idx * x.Sz;
        if (len == 1) { c0 = Scale(x, E(0), xs); c1 = new Ser(x.Const(0), 1); }
        else if (len == 2) { c0 = Scale(x, E(0), xs); c1 = Scale(x, E(1), xs); }
        else
        {
            long nd = len;
            c0 = Scale(x, E(len - 2), xs);
            c1 = Scale(x, E(len - 1), xs);
            for (long i = 3; i <= len; i++)   // NumPy's Python loop over SERIES operations
            {
                var tmp = c0;
                nd = nd - 1;
                c0 = SubS(x, Scale(x, E(len - i), xs), MulDiv(x, c1, nd - 1, nd));
                c1 = AddS(x, tmp, MulDiv(x, LegMulxS(x, c1), 2 * nd - 1, nd));
            }
        }
        return AddS(x, c0, LegMulxS(x, c1));
    }

    /// <summary><c>pu._div(legmul, a, b)</c>: repeated subtraction of <c>legmul([0]*i + [1], b)</c>.</summary>
    /// <param name="x">Context.</param><param name="a">Dividend.</param><param name="b">Divisor.</param><returns>(quotient, remainder).</returns>
    static (Ser, Ser) LegDiv(Ctx x, Ser a, Ser b)
    {
        a = x.AsSeries(a); b = x.AsSeries(b);
        long lc1 = a.N, lc2 = b.N;
        var quo = new Ser(x.A.Alloc((lc1 - lc2 + 1) * x.Sz), lc1 - lc2 + 1);
        Ser rem = a;
        for (long i = lc1 - lc2; i >= 0; i--)
        {
            long mark = x.A.Mark;
            var basis = new Ser(x.A.Alloc((i + 1) * x.Sz), i + 1);   // [0]*i + [1]
            for (long q = 0; q <= i; q++) Buffer.MemoryCopy(x.Const(q == i ? 1 : 0), basis.P + q * x.Sz, x.Sz, x.Sz);
            var p = LegMul(x, basis, b);
            var qv = x.A.Alloc(x.Sz);
            x.SDiv(qv, null, 1, rem.P + (rem.N - 1) * x.Sz, p.P + (p.N - 1) * x.Sz);     // q = rem[-1] / p[-1]
            var qp = Scale(x, qv, new Ser(p.P, p.N - 1));                                 // q * p[:-1]
            var nr = x.Copy(new Ser(rem.P, rem.N - 1), rem.N - 1);                        // rem[:-1] - q*p[:-1]
            x.Sub(nr.P, qp.P, nr.N);
            Buffer.MemoryCopy(qv, quo.P + i * x.Sz, x.Sz, x.Sz);                          // quo[i] = q
            // Drop the divisor product: move the new remainder down to the mark (source lies above target).
            x.A.Release(mark);
            rem = x.Copy(nr, nr.N);
        }
        return (quo, new Ser(rem.P, x.Trim(rem.P, rem.N, x.Zero)));
    }

    /// <summary>Binds the dtype's primitives (compiled once per dtype, then cached).</summary>
    /// <param name="t">Dtype.</param><param name="a">Arena.</param><param name="pool">Weak-int pool of dtype t.</param><returns>The context.</returns>
    static Ctx NewCtx(NPTypeCode t, Arena a, NDArray pool)
    {
        var x = new Ctx { T = t, Sz = DirectILKernelGenerator.GetTypeSize(t), A = a };
        x.Pool = (byte*)pool.Storage.Address; x.Zero = x.Pool;
        x.Trim = TrimLen(t); x.Scale = Map(t, 0); x.MulDiv = Map(t, 1); x.SDiv = ScalarDiv(t);
        x.Add = InPlace(t, 0); x.Sub = InPlace(t, 1); x.Neg = InPlace(t, 2); x.Mulx = LegMulx(t);
        return x;
    }

    /// <summary>Copies an arena series out into an owning NDArray (the call's only managed allocation).</summary>
    /// <param name="t">Dtype.</param><param name="s">Series.</param><returns>The result array.</returns>
    static NDArray ToND(NPTypeCode t, Ser s)
    {
        var r = new NDArray(t, new Shape(s.N), false);
        Buffer.MemoryCopy(s.P, r.Storage.Address, s.N * r.dtypesize, s.N * r.dtypesize);
        return r;
    }

    /// <summary>A series view over an NDArray's contiguous buffer.</summary><param name="a">A 1-D contiguous array.</param><returns>The view.</returns>
    static Ser S(NDArray a) => new((byte*)a.Storage.Address + a.Shape.offset * a.dtypesize, a.size);

    // ======================================================================== harness

    static readonly Dictionary<string, double> s_numpy = new();

    /// <summary>Reads the twin's TSV (<c>cell&lt;TAB&gt;us</c>).</summary><param name="path">TSV path.</param>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    public static void LoadNumPyTimes(string path)
    {
        foreach (var line in File.ReadAllLines(path))
        {
            var p = line.Split('\t');
            if (p.Length == 2 && double.TryParse(p[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us)) s_numpy[p[0]] = us;
        }
    }

    /// <summary>Loads one .npy the twin wrote.</summary><param name="name">File stem.</param><returns>The array.</returns>
    /// <exception cref="FileNotFoundException">The twin has not been run.</exception>
    static NDArray L(string name) => np.load_npy(Path.Combine(Data, name + ".npy"));

    /// <summary>Disposes a result (an NDArray or a (q, r) tuple).</summary><param name="o">The result.</param>
    static void Rel(object o)
    {
        if (o is NDArray a) a.Dispose();
        else if (o is ValueTuple<NDArray, NDArray> t) { t.Item1.Dispose(); t.Item2.Dispose(); }
    }

    /// <summary>Best-of-7 microseconds per call after a 300 ms warm-up, each repeat sized to ~25 ms.</summary>
    /// <param name="f">The call.</param><returns>Microseconds.</returns>
    static double Bench(Func<object> f)
    {
        var sw = Stopwatch.StartNew(); int w = 0;
        while (sw.ElapsedMilliseconds < 300 || w < 3) { Rel(f()); w++; }
        sw.Restart(); Rel(f());
        int iters = Math.Max(1, (int)(25.0 / Math.Max(sw.Elapsed.TotalMilliseconds, 1e-4)));
        double best = double.MaxValue;
        for (int r = 0; r < 7; r++)
        {
            sw.Restart();
            for (int k = 0; k < iters; k++) Rel(f());
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds / iters);
        }
        return best * 1000.0;
    }

    /// <summary>Byte-compares one array with a reference (dtype, length, then every element's bytes).</summary>
    /// <param name="got">Result.</param><param name="refName">Reference stem.</param><returns><c>exact</c> or what differs.</returns>
    static string Same(NDArray got, string refName)
    {
        using var want = L(refName);
        if (got.typecode != want.typecode) return $"DTYPE {got.typecode} vs {want.typecode}";
        if (got.size != want.size) return $"SIZE {got.size} vs {want.size}";
        ReadOnlySpan<byte> a = got.Unsafe.ReadOnlyBytes(), w = want.Unsafe.ReadOnlyBytes();
        int isz = got.dtypesize; long bad = 0;
        for (int e = 0; e < a.Length; e += isz)
            if (!a.Slice(e, isz).SequenceEqual(w.Slice(e, isz))) bad++;
        return bad == 0 ? "exact" : $"{bad}/{got.size} differ";
    }

    /// <summary>Checks, times and prints one row.</summary>
    /// <param name="cell">Join label.</param><param name="f">The call.</param><param name="check">Byte check of one result.</param>
    static void Row(string cell, Func<object> f, Func<object, string> check)
    {
        var first = f(); string ok = check(first); Rel(first);
        double us = Bench(f);
        bool hasNp = s_numpy.TryGetValue(cell, out var npus);
        Console.WriteLine($"{cell}\tIL-prim+arena\t{us.ToString("F3", CultureInfo.InvariantCulture)}\t" +
                          $"{(hasNp ? npus.ToString("F3", CultureInfo.InvariantCulture) : "-")}\t" +
                          $"{(hasNp ? (npus / us).ToString("F2", CultureInfo.InvariantCulture) : "-")}\t{ok}");
    }

    /// <summary>Runs every series cell for every dtype the twin wrote references for.</summary>
    public static void Run()
    {
        Console.WriteLine("cell\tvariant\tns_us\tnumpy_us\tNPY/NS\tcheck");
        using var arena = new Arena(16 << 20);   // production: a [ThreadStatic] arena reused across calls
        var inputs = new Dictionary<string, (NDArray s10, NDArray s50, NPTypeCode t)>
        {
            ["f64"] = (L("s10"), L("s50"), NPTypeCode.Double),
            ["f32"] = (L("s10").astype(NPTypeCode.Single), L("s50").astype(NPTypeCode.Single), NPTypeCode.Single),
            ["f16"] = (L("s10").astype(NPTypeCode.Half), L("s50").astype(NPTypeCode.Half), NPTypeCode.Half),
            ["c128"] = (L("s10c"), L("s50c"), NPTypeCode.Complex),
        };
        var pools = inputs.Values.Select(v => v.t).Distinct().ToDictionary(t => t, t => np.arange(0.0, 256.0).astype(t));

        object Legmul(NDArray a, NDArray b, NPTypeCode t)
        {
            arena.Release(0);
            var x = NewCtx(t, arena, pools[t]);
            return ToND(t, LegMul(x, S(a), S(b)));
        }
        object Legdiv(NDArray a, NDArray b, NPTypeCode t)
        {
            arena.Release(0);
            var x = NewCtx(t, arena, pools[t]);
            var (q, r) = LegDiv(x, S(a), S(b));
            return (ToND(t, q), ToND(t, r));
        }
        object Chebder(NDArray c, NPTypeCode t)
        {
            long n = c.size - 1;
            var der = new NDArray(t, new Shape(n), false);
            arena.Release(0);
            var copy = arena.Alloc(c.size * c.dtypesize);                 // `c = np.array(c, ndmin=1, copy=True)`
            Buffer.MemoryCopy(S(c).P, copy, c.size * c.dtypesize, c.size * c.dtypesize);
            ChebDer(t)((byte*)der.Storage.Address, copy, n, (byte*)pools[t].Storage.Address);
            return der;
        }
        object Polyadd(NDArray a, NDArray b, NPTypeCode t)
        {
            arena.Release(0);
            var x = NewCtx(t, arena, pools[t]);
            return ToND(t, AddS(x, S(a), S(b)));
        }

        var s5 = L("s5");
        Row("S polyadd 5+10 f64", () => Polyadd(s5, inputs["f64"].s10, NPTypeCode.Double), o => Same((NDArray)o, "ref_S_polyadd_f64"));
        foreach (var (tag, (s10, s50, t)) in inputs)
        {
            Row($"S legmul 10x10 {tag}", () => Legmul(s10, s10, t), o => Same((NDArray)o, $"ref_S_legmul10_{tag}"));
            Row($"S chebder 50 {tag}", () => Chebder(s50, t), o => Same((NDArray)o, $"ref_S_chebder50_{tag}"));
        }
        var f64 = inputs["f64"]; var c128 = inputs["c128"];
        Row("S legmul 50x50 f64", () => Legmul(f64.s50, f64.s50, NPTypeCode.Double), o => Same((NDArray)o, "ref_S_legmul50_f64"));
        Row("S legdiv 50/10 f64", () => Legdiv(f64.s50, f64.s10, NPTypeCode.Double),
            o => { var (q, r) = ((NDArray, NDArray))o; return "q:" + Same(q, "ref_S_legdiv_f64_0") + " r:" + Same(r, "ref_S_legdiv_f64_1"); });
        Row("S legdiv 50/10 c128", () => Legdiv(c128.s50, c128.s10, NPTypeCode.Complex),
            o => { var (q, r) = ((NDArray, NDArray))o; return "q:" + Same(q, "ref_S_legdiv_c128_0") + " r:" + Same(r, "ref_S_legdiv_c128_1"); });
    }
}
