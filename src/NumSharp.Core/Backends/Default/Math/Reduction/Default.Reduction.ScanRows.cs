using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     The shortest contiguous block per scan step for which <see cref="TryScanRows"/> runs: below it a step is too
        ///     short to amortize a vector loop, and the per-position axis kernels (whose inner run IS that block) are as
        ///     fast. 32 elements is four 256-bit vectors of float64.
        /// </summary>
        private const long ScanRowsMinInner = 32;

        /// <summary>
        ///     np.cumsum / np.cumprod along an axis that is NOT innermost in memory, computed by whole rows: with the operand
        ///     dense in the output's memory order, each step of the scan axis is one contiguous block of positions, so
        ///     <c>out[j] = out[j-1] op in[j]</c> is a vector loop over that block, and the first block is copied (NumPy's
        ///     accumulate copies element 0). Every position still receives exactly NumPy's operations in NumPy's order — the
        ///     sequential scan, <c>out[j-1]</c> as the first operand — so the values are the per-position kernels' to the bit;
        ///     only the order ACROSS positions changes, from one strided walk per position to streaming rows.
        /// </summary>
        /// <param name="input">The accumulated array (any layout; only a dense one in the output's memory order qualifies).</param>
        /// <param name="axis">The scan axis, already normalized.</param>
        /// <param name="retType">The accumulator dtype; the route runs only when it is the input's own dtype (no cast).</param>
        /// <param name="layout">The output layout (<see cref="AccumulateOutputLayout"/>: NumPy's K order of the input).</param>
        /// <param name="multiply">True for np.cumprod's multiply, false for np.cumsum's add.</param>
        /// <param name="result">The fresh result laid out as <paramref name="layout"/> when the method returns true.</param>
        /// <returns>True when the row route computed the scan; false when the caller's per-position route must run (a cast,
        ///     a dtype without a vector loop, a reversed / stepped / broadcast input, or a block shorter than
        ///     <see cref="ScanRowsMinInner"/>, the innermost axis among them).</returns>
        /// <remarks>
        ///     <para>Why it exists: NumPy's own layout rule (parity QA-36) puts a permuted operand's scan axis wherever its
        ///     memory order says, often outermost, and the per-position walk along the outermost axis of a dense array reads
        ///     one element per cache line per step (np.cumsum of a (100, 100, 100) C array along axis 0: 2–3 ms that way, the
        ///     memory bandwidth's ~0.6 ms by rows; NumPy runs the strided walk, ~6.8 ms).</para>
        ///     <para>Types: the ten integer and float lanes <see cref="Vector{T}"/> has. float16, bool, char, decimal and
        ///     complex keep their per-position kernels (complex128's product follows NumPy's contracted loop_scalar form,
        ///     which a component-wise vector product would not).</para>
        /// </remarks>
        internal unsafe bool TryScanRows(NDArray input, int axis, NPTypeCode retType, Shape layout, bool multiply, out NDArray result)
        {
            result = null;
            var tc = input.GetTypeCode;
            if (tc != retType || !IsScanRowsType(tc) || input.ndim < 2 || layout.size < 2 * ScanRowsMinInner)
                return false;

            // The output's memory order, outermost axis first, and the block of positions after the scan axis in it.
            int[] perm = layout.DenseAxisOrder();
            int k = Array.IndexOf(perm, axis);
            long inner = 1;
            for (int i = k + 1; i < perm.Length; i++)
                inner *= layout.dimensions[perm[i]];
            if (inner < ScanRowsMinInner)
                return false;

            // The input must be dense in that same order: its transpose into it C-contiguous (a reversed axis, a step or a
            // broadcast fails this and keeps the per-position route).
            NDArray t = Transpose(input, perm);
            try
            {
                if (!t.Shape.IsContiguous)
                    return false;

                long n = layout.dimensions[axis];
                long outer = layout.size / (n * inner);
                // A fresh result in `layout` is dense with offset 0, so its buffer IS the C buffer of the transposed dims:
                // (outer, n, inner) indexes both operands alike.
                var ret = NewArray(retType, layout, false);
                byte* src = (byte*)t.Address + t.Shape.offset * t.dtypesize;
                byte* dst = (byte*)ret.Address;
                switch (tc)
                {
                    case NPTypeCode.Byte: ScanRows((byte*)src, (byte*)dst, outer, n, inner, multiply); break;
                    case NPTypeCode.SByte: ScanRows((sbyte*)src, (sbyte*)dst, outer, n, inner, multiply); break;
                    case NPTypeCode.Int16: ScanRows((short*)src, (short*)dst, outer, n, inner, multiply); break;
                    case NPTypeCode.UInt16: ScanRows((ushort*)src, (ushort*)dst, outer, n, inner, multiply); break;
                    case NPTypeCode.Int32: ScanRows((int*)src, (int*)dst, outer, n, inner, multiply); break;
                    case NPTypeCode.UInt32: ScanRows((uint*)src, (uint*)dst, outer, n, inner, multiply); break;
                    case NPTypeCode.Int64: ScanRows((long*)src, (long*)dst, outer, n, inner, multiply); break;
                    case NPTypeCode.UInt64: ScanRows((ulong*)src, (ulong*)dst, outer, n, inner, multiply); break;
                    case NPTypeCode.Single: ScanRows((float*)src, (float*)dst, outer, n, inner, multiply); break;
                    default: ScanRows((double*)src, (double*)dst, outer, n, inner, multiply); break;   // IsScanRowsType: Double
                }
                // The raw pointers do not root the input: keep it alive past the loop.
                GC.KeepAlive(input);
                result = ret;
                return true;
            }
            finally
            {
                // The transposed view only fed the pointers; the result is a fresh array.
                t.Dispose();
            }
        }

        /// <summary>The dtypes <see cref="TryScanRows"/> has a vector loop for: the <see cref="Vector{T}"/> lanes.</summary>
        /// <param name="tc">The input (= accumulator) dtype.</param>
        /// <returns>True for the eight integer widths, float32 and float64.</returns>
        private static bool IsScanRowsType(NPTypeCode tc) => tc is NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16
            or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64
            or NPTypeCode.Single or NPTypeCode.Double;

        /// <summary>
        ///     The row loop of <see cref="TryScanRows"/> over <c>outer</c> independent slabs of <c>n × inner</c> elements:
        ///     each slab's first row copied, every later row the running row combined with the input row.
        /// </summary>
        /// <typeparam name="T">The element type (input and output alike).</typeparam>
        /// <param name="src">The input's first element (dense, C order over (outer, n, inner)).</param>
        /// <param name="dst">The output's first element (the same order, a different buffer).</param>
        /// <param name="outer">The slab count (the axes before the scan axis in memory order).</param>
        /// <param name="n">The scan axis' extent.</param>
        /// <param name="inner">The positions per row (the axes after the scan axis), at least <see cref="ScanRowsMinInner"/>.</param>
        /// <param name="multiply">True to multiply, false to add.</param>
        /// <remarks>Integer arithmetic wraps, as NumPy's loops do (C# unchecked). The scalar tail runs the same expression
        ///     with the running value as the first operand.</remarks>
        private static unsafe void ScanRows<T>(T* src, T* dst, long outer, long n, long inner, bool multiply)
            where T : unmanaged, INumber<T>
        {
            long block = n * inner;
            int w = Vector<T>.Count;
            long rowBytes = inner * sizeof(T);
            for (long o = 0; o < outer; o++)
            {
                T* s = src + o * block;
                T* d = dst + o * block;
                // accumulate's first element is a copy (a -0.0 or a signalling NaN arrives as it is).
                Buffer.MemoryCopy(s, d, rowBytes, rowBytes);
                for (long j = 1; j < n; j++)
                {
                    T* prev = d + (j - 1) * inner;
                    T* cur = d + j * inner;
                    T* x = s + j * inner;
                    long p = 0;
                    if (Vector.IsHardwareAccelerated)
                    {
                        if (multiply)
                            for (; p <= inner - w; p += w)
                                Unsafe.WriteUnaligned(cur + p,
                                    Unsafe.ReadUnaligned<Vector<T>>(prev + p) * Unsafe.ReadUnaligned<Vector<T>>(x + p));
                        else
                            for (; p <= inner - w; p += w)
                                Unsafe.WriteUnaligned(cur + p,
                                    Unsafe.ReadUnaligned<Vector<T>>(prev + p) + Unsafe.ReadUnaligned<Vector<T>>(x + p));
                    }
                    for (; p < inner; p++)
                        cur[p] = multiply ? prev[p] * x[p] : prev[p] + x[p];
                }
            }
        }
    }
}
