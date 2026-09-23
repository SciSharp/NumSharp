/*
 * NumSharp
 * Copyright (C) 2018 Haiping Chen
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the Apache License 2.0 as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the Apache License 2.0
 * along with this program.  If not, see <http://www.apache.org/licenses/LICENSE-2.0/>.
 */

using System;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        public T[] ToArray<T>() where T : unmanaged
        {
            return Storage.ToArray<T>();
        }

        /// <summary>
        ///     Copies this array into a new .NET array of element type <typeparamref name="T"/>: a single-dimensional
        ///     <c>T[]</c> for a 0-d or 1-D array (a 0-d array yields a one-element array, since .NET has no rank-0
        ///     array), otherwise a rank-N <c>T[,…]</c> with the same lengths. Elements are written in logical C
        ///     (row-major) order for any memory layout.
        /// </summary>
        /// <typeparam name="T">
        ///     Element type of the result. When it differs from <see cref="dtype"/> the values are converted with
        ///     <see cref="astype(DType, bool)"/> semantics; it must be one of NumSharp's 15 dtypes.
        /// </typeparam>
        /// <returns>A freshly allocated managed array that shares no memory with this NDArray.</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not a NumSharp dtype.</exception>
        /// <exception cref="InvalidOperationException">A dimension exceeds <see cref="int.MaxValue"/> (managed arrays are int-indexed).</exception>
        /// <exception cref="TypeLoadException">The array has more than 32 dimensions, the .NET limit on array rank.</exception>
        /// <remarks>
        ///     One pass straight into the pinned result, lean where it is hot: the shape is read by reference, a large
        ///     1-D result is allocated uninitialized on the pinned-object heap, a contiguous same-dtype source is one
        ///     memcpy (one assignment for one element) from the logical start, and a one-element conversion is one inline
        ///     scalar conversion. Only the multi-element conversion setup is outlined ([NoInlining]
        ///     <see cref="ConvertInto{T}"/>), so it adds nothing to this frame. Rank ≥ 4 allocates through
        ///     <see cref="RankNAllocator{T}"/> (IL <c>newobj</c>, no reflection-driven activation).
        ///     The fill engine is the parent's: same dtype + non-contiguous goes through <see cref="CopyStrided{T}"/>
        ///     (row memcpy for unit inner stride, 64-column bands walked in 8-row strips of AVX register blocks for
        ///     transposed / F-ordered planes, AVX2 stride-2 deinterleave or gathers for plain strided rows of 4/8-byte
        ///     elements, an indexed scalar loop otherwise); a multi-element conversion is <see cref="ConvertInto{T}"/>
        ///     — astype's cast kernels, or <see cref="DecimalToDoubleKernel"/> for a contiguous decimal → double
        ///     where its probe proved it bit-identical.
        /// </remarks>
        public Array ToMuliDimArray<T>() where T : unmanaged
        {
            var target = InfoOf<T>.NPTypeCode;
            // astype semantics need a NumSharp dtype to cast into; Guid, nint, DateTime… have none.
            if (target == NPTypeCode.Empty)
                throw new NotSupportedException($"Unable to convert to {typeof(T).Name}[]: {typeof(T).Name} is not a NumSharp dtype.");

            var storage = Storage;
            ref readonly Shape shape = ref storage.ShapeReference;
            long[] dims = shape.dimensions;
            var source = storage.TypeCode;
            Array ret = dims.Length switch
            {
                0 => new T[1],
                // Every element is written before return, so a large 1-D result skips the GC's zero-fill, and on the
                // pinned-object heap the fixed below costs nothing (below ~2 KB the runtime zeroes regardless).
                1 => dims[0] >= 1024 ? GC.AllocateUninitializedArray<T>(ManagedLength(dims[0]), pinned: true) : new T[ManagedLength(dims[0])],
                2 => new T[ManagedLength(dims[0]), ManagedLength(dims[1])],
                3 => new T[ManagedLength(dims[0]), ManagedLength(dims[1]), ManagedLength(dims[2])],
                // Rank ≥ 4: a cached IL-generated allocator per (T, rank) — a direct newobj, no reflection-driven
                // activation per call (beyond rank 32 the runtime refuses the array type itself: TypeLoadException).
                _ => RankNAllocator<T>.Allocate(dims),
            };

            long count = shape.size;
            if (count == 0)
                return ret;

            unsafe
            {
                // Logical element 0 of ANY layout: base address + offset elements (the documented NDArray rule).
                byte* src = storage.Address + shape.offset * storage.DTypeSize;
                // Pinned only for the fill; nothing below retains the pointer.
                fixed (byte* p = &MemoryMarshal.GetArrayDataReference(ret))
                {
                    if (target != source)
                    {
                        if (count == 1)
                            // One value, inline: the scalar converter astype's copy core applies to a single element —
                            // behind a call it cost the tiniest conversion ~40 % (trial 17's held-out lesson).
                            NDIterCasting.ConvertValue(src, p, source, target);
                        else
                            ConvertInto((T*)p, target);
                    }
                    else if (count == 1)
                        *(T*)p = *(T*)src;
                    else if (shape.IsContiguous)
                    {
                        // One memcpy from the logical start — no helper re-validation on the hottest path.
                        long bytes = count * sizeof(T);
                        Buffer.MemoryCopy(src, p, bytes, bytes);
                    }
                    else
                        CopyStrided((T*)src, dims, shape.strides, (T*)p);
                }
            }
            return ret;
        }

        /// <summary>Narrows one NumSharp dimension to a managed-array length.</summary>
        /// <param name="d">A dimension length.</param>
        /// <returns><paramref name="d"/> as <see cref="int"/>.</returns>
        /// <exception cref="InvalidOperationException"><paramref name="d"/> exceeds <see cref="int.MaxValue"/> — .NET arrays are int-indexed.</exception>
        private static int ManagedLength(long d)
            => d <= int.MaxValue
                ? (int)d
                : throw new InvalidOperationException($"Dimension {d} exceeds int.MaxValue. Cannot convert to .NET multi-dimensional array.");

        /// <summary>
        ///     Writes a non-contiguous source's elements to a dense row-major destination, one plane (the two
        ///     innermost axes) at a time, with an odometer over the outer axes.
        /// </summary>
        /// <typeparam name="T">Element type (source and destination share it).</typeparam>
        /// <param name="src">Logical element 0 of the source.</param>
        /// <param name="dims">Source dimensions (all non-zero; caller handled empty arrays).</param>
        /// <param name="strides">Source strides in ELEMENTS (may be 0 for broadcast axes, negative for reversed ones).</param>
        /// <param name="dst">Destination with room for every element, written in C order.</param>
        private static unsafe void CopyStrided<T>(T* src, long[] dims, long[] strides, T* dst) where T : unmanaged
        {
            int nd = dims.Length;
            if (nd == 1)
            {
                long n = dims[0], s = strides[0];
                for (long i = 0; i < n; i++)
                    dst[i] = src[i * s];
                return;
            }

            long rows = dims[nd - 2], cols = dims[nd - 1];
            long rs = strides[nd - 2], cs = strides[nd - 1];
            int outerNd = nd - 2;
            long outer = 1;
            for (int i = 0; i < outerNd; i++)
                outer *= dims[i];
            long plane = rows * cols;
            long[] idx = outerNd > 0 ? new long[outerNd] : null;
            long srcOff = 0;
            for (long o = 0; o < outer; o++)
            {
                CopyPlane(src + srcOff, rows, cols, rs, cs, dst + o * plane);
                // Odometer over the outer axes (last outer axis fastest), tracking the source offset incrementally.
                for (int ax = outerNd - 1; ax >= 0; ax--)
                {
                    srcOff += strides[ax];
                    if (++idx[ax] < dims[ax])
                        break;
                    srcOff -= strides[ax] * dims[ax];
                    idx[ax] = 0;
                }
            }
        }

        /// <summary>Copies one rows×cols plane from a strided source into dense row-major memory.</summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="s">Plane origin in the source.</param>
        /// <param name="rows">Row count.</param>
        /// <param name="cols">Column count.</param>
        /// <param name="rs">Source stride between rows (elements).</param>
        /// <param name="cs">Source stride between columns (elements).</param>
        /// <param name="d">Plane origin in the destination (row stride = <paramref name="cols"/>).</param>
        /// <remarks>
        ///     Three regimes: unit column stride ⇒ each row is one memcpy; |row stride| &lt; |column stride|
        ///     (transposed / F-ordered) ⇒ <see cref="TransposeBands{T}"/> (64-column bands walked in 8-row
        ///     strips); anything else ⇒ a row loop, which for
        ///     4/8-byte elements under AVX2 deinterleaves contiguous vector loads when the column stride is 2 and
        ///     issues one hardware gather per 8 (4) elements for any other non-zero stride.
        /// </remarks>
        private static unsafe void CopyPlane<T>(T* s, long rows, long cols, long rs, long cs, T* d) where T : unmanaged
        {
            if (cs == 1)
            {
                long rowBytes = cols * sizeof(T);
                for (long r = 0; r < rows; r++)
                    Buffer.MemoryCopy(s + r * rs, d + r * cols, rowBytes, rowBytes);
                return;
            }

            if (Math.Abs(rs) < Math.Abs(cs))
            {
                TransposeBands(s, rows, cols, rs, cs, d);
                return;
            }

            // Plain strided rows of 4/8-byte elements under AVX2 (the element width is a JIT-time constant, so every
            // other width compiles this block away). A column stride of 2 — the ubiquitous [::2] — costs two
            // contiguous vector loads and a lane shuffle per block; any other non-zero stride small enough for
            // 32-bit gather indices (7·|cs| must fit in int) is one hardware gather per block. Stride 0 (a
            // broadcast column) keeps the plain loop.
            if ((sizeof(T) == 4 || sizeof(T) == 8) && System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            {
                if (cs == 2)
                {
                    if (sizeof(T) == 4)
                        DeinterleaveRows4((float*)s, rows, cols, rs, (float*)d);
                    else
                        DeinterleaveRows8((double*)s, rows, cols, rs, (double*)d);
                    return;
                }
                if (cs != 0 && Math.Abs(cs) < int.MaxValue / 8)
                {
                    if (sizeof(T) == 4)
                        GatherRows4((int*)s, rows, cols, rs, cs, (int*)d);
                    else
                        GatherRows8((long*)s, rows, cols, rs, cs, (long*)d);
                    return;
                }
            }

            for (long r = 0; r < rows; r++)
            {
                T* sr = s + r * rs;
                T* dr = d + r * cols;
                for (long c = 0; c < cols; c++)
                    dr[c] = sr[c * cs];
            }
        }

        /// <summary>
        ///     Copies stride-2 rows of 4-byte elements (the <c>[::2]</c> case) by deinterleaving pairs of contiguous
        ///     8-lane loads: 8 destination elements per 2 loads, 2 lane moves and 1 store.
        /// </summary>
        /// <param name="s">Plane origin in the source (column stride 2 elements).</param>
        /// <param name="rows">Row count.</param>
        /// <param name="cols">Column count (also the destination row stride).</param>
        /// <param name="rs">Source row stride in elements.</param>
        /// <param name="d">Plane origin in the dense row-major destination.</param>
        /// <remarks>
        ///     Lanes are typed <see cref="float"/> but only MOVED (vshufps 0x88 keeps the even lanes of both loads per
        ///     128-bit half, vpermpd 0xD8 restores their order), so every 32-bit pattern — NaN payloads, integers —
        ///     passes through bit-exactly. The block loop runs only while another needed element follows the block
        ///     (<c>c + 8 &lt; cols</c>), so the second load's last lane (element 2c+15) lies strictly between two
        ///     needed elements of the same row: it never reads past the view, even when the view ends at its
        ///     buffer's last element. The final block and any remainder go through the scalar tail.
        /// </remarks>
        private static unsafe void DeinterleaveRows4(float* s, long rows, long cols, long rs, float* d)
        {
            for (long r = 0; r < rows; r++)
            {
                float* sr = s + r * rs;
                float* dr = d + r * cols;
                long c = 0;
                for (; c + 8 < cols; c += 8)
                {
                    var v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(sr + 2 * c);
                    var v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(sr + 2 * c + 8);
                    // [a0 a2 b0 b2 | a4 a6 b4 b6] as four 64-bit pairs q0 q1 q2 q3 -> reorder to q0 q2 q1 q3.
                    var even = System.Runtime.Intrinsics.Vector256.AsDouble(System.Runtime.Intrinsics.X86.Avx.Shuffle(v0, v1, 0x88));
                    System.Runtime.Intrinsics.X86.Avx.Store(dr + c,
                        System.Runtime.Intrinsics.Vector256.AsSingle(System.Runtime.Intrinsics.X86.Avx2.Permute4x64(even, 0xD8)));
                }
                for (; c < cols; c++)
                    dr[c] = sr[2 * c];
            }
        }

        /// <summary>
        ///     Copies stride-2 rows of 8-byte elements (the <c>[::2]</c> case) by deinterleaving pairs of contiguous
        ///     4-lane loads: 4 destination elements per 2 loads, 2 lane moves and 1 store.
        /// </summary>
        /// <param name="s">Plane origin in the source (column stride 2 elements).</param>
        /// <param name="rows">Row count.</param>
        /// <param name="cols">Column count (also the destination row stride).</param>
        /// <param name="rs">Source row stride in elements.</param>
        /// <param name="d">Plane origin in the dense row-major destination.</param>
        /// <remarks>
        ///     vunpcklpd interleaves the even lanes of both loads per 128-bit half and vpermpd 0xD8 orders them —
        ///     lane moves only, bit-exact for every 64-bit pattern. As in <see cref="DeinterleaveRows4"/>, a block
        ///     runs only while a needed element follows it (<c>c + 4 &lt; cols</c>), so the loads never pass the
        ///     row's next needed element; the final block and remainder are scalar.
        /// </remarks>
        private static unsafe void DeinterleaveRows8(double* s, long rows, long cols, long rs, double* d)
        {
            for (long r = 0; r < rows; r++)
            {
                double* sr = s + r * rs;
                double* dr = d + r * cols;
                long c = 0;
                for (; c + 4 < cols; c += 4)
                {
                    var v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(sr + 2 * c);
                    var v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(sr + 2 * c + 4);
                    // [a0 b0 | a2 b2] -> [a0 a2 b0 b2].
                    var even = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v0, v1);
                    System.Runtime.Intrinsics.X86.Avx.Store(dr + c, System.Runtime.Intrinsics.X86.Avx2.Permute4x64(even, 0xD8));
                }
                for (; c < cols; c++)
                    dr[c] = sr[2 * c];
            }
        }

        /// <summary>Copies plain strided rows of 4-byte elements with AVX2 hardware gathers (8 elements per gather).</summary>
        /// <param name="s">Plane origin in the source.</param>
        /// <param name="rows">Row count.</param>
        /// <param name="cols">Column count (also the destination row stride).</param>
        /// <param name="rs">Source row stride in elements.</param>
        /// <param name="cs">Source column stride in elements (non-zero; 7·|cs| fits in int; may be negative).</param>
        /// <param name="d">Plane origin in the dense row-major destination.</param>
        /// <remarks>
        ///     A gather reads exactly the 8 needed elements (signed 32-bit indices 0, cs, …, 7·cs from the block's
        ///     first element), so it never touches memory outside the view; raw 32-bit patterns move unchanged. The
        ///     ragged remainder of each row is scalar.
        /// </remarks>
        private static unsafe void GatherRows4(int* s, long rows, long cols, long rs, long cs, int* d)
        {
            int c32 = (int)cs;
            var idx = System.Runtime.Intrinsics.Vector256.Create(0, c32, 2 * c32, 3 * c32, 4 * c32, 5 * c32, 6 * c32, 7 * c32);
            long cols8 = cols & ~7L;
            for (long r = 0; r < rows; r++)
            {
                int* sr = s + r * rs;
                int* dr = d + r * cols;
                long c = 0;
                for (; c < cols8; c += 8)
                    System.Runtime.Intrinsics.X86.Avx.Store(dr + c, System.Runtime.Intrinsics.X86.Avx2.GatherVector256(sr + c * cs, idx, 4));
                for (; c < cols; c++)
                    dr[c] = sr[c * cs];
            }
        }

        /// <summary>Copies plain strided rows of 8-byte elements with AVX2 hardware gathers (4 elements per gather).</summary>
        /// <param name="s">Plane origin in the source.</param>
        /// <param name="rows">Row count.</param>
        /// <param name="cols">Column count (also the destination row stride).</param>
        /// <param name="rs">Source row stride in elements.</param>
        /// <param name="cs">Source column stride in elements (non-zero; 3·|cs| fits in int; may be negative).</param>
        /// <param name="d">Plane origin in the dense row-major destination.</param>
        /// <remarks>
        ///     Reads exactly the 4 needed elements per gather (no out-of-view access) and moves raw 64-bit patterns,
        ///     so NaN payloads and integers pass bit-exactly; the ragged remainder of each row is scalar.
        /// </remarks>
        private static unsafe void GatherRows8(long* s, long rows, long cols, long rs, long cs, long* d)
        {
            int c32 = (int)cs;
            var idx = System.Runtime.Intrinsics.Vector128.Create(0, c32, 2 * c32, 3 * c32);
            long cols4 = cols & ~3L;
            for (long r = 0; r < rows; r++)
            {
                long* sr = s + r * rs;
                long* dr = d + r * cols;
                long c = 0;
                for (; c < cols4; c += 4)
                    System.Runtime.Intrinsics.X86.Avx.Store(dr + c, System.Runtime.Intrinsics.X86.Avx2.GatherVector256(sr + c * cs, idx, 8));
                for (; c < cols; c++)
                    dr[c] = sr[c * cs];
            }
        }

        /// <summary>
        ///     Copies one transpose-like plane (|row stride| &lt; |column stride| — transposed / F-ordered views) as
        ///     64-column bands walked down in 8-row strips: within a strip, each source column's 8 elements (one cache
        ///     line of 8-byte elements when the row stride is 1) go to the 8 destination rows at once.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="s">Plane origin in the source.</param>
        /// <param name="rows">Row count.</param>
        /// <param name="cols">Column count (also the destination row stride).</param>
        /// <param name="rs">Source row stride in elements (|rs| &lt; |cs|; may be 0 or negative).</param>
        /// <param name="cs">Source column stride in elements.</param>
        /// <param name="d">Plane origin in the dense row-major destination.</param>
        /// <remarks>
        ///     Why this shape rather than square tiles: a power-of-two destination pitch maps every destination row
        ///     onto the SAME L1 set (and a power-of-two column stride does the same to every source column), so a
        ///     32×32 tile keeps 32+ lines of one set live and thrashes its 12 ways — a square-tiled transpose ends up
        ///     L2-bound. Here each source line is consumed the moment it is loaded and only the 8 destination lines
        ///     being filled are live, which fits any set; the 64-column band bounds the pages one strip touches.
        ///     With element-contiguous source columns, 8-byte and 4-byte elements move as 8×4 / 8×8 AVX register
        ///     blocks (<see cref="TransposeBlock8x4"/>, <see cref="TransposeBlock8x8"/>).
        ///     Rows past the last whole strip are copied one row at a time.
        /// </remarks>
        private static unsafe void TransposeBands<T>(T* s, long rows, long cols, long rs, long cs, T* d) where T : unmanaged
        {
            const long W = 64;
            long rows8 = rows & ~7L;
            // Row offsets inside a strip: rows 4..7 are addressed from their own base, so three offsets (plus 0)
            // cover all eight rows and stay in registers.
            long o1 = cols, o2 = 2 * cols, o3 = 3 * cols;
            long q1 = rs, q2 = 2 * rs, q3 = 3 * rs;
            for (long c0 = 0; c0 < cols; c0 += W)
            {
                long w = Math.Min(W, cols - c0);
                for (long r0 = 0; r0 < rows8; r0 += 8)
                {
                    T* sa = s + r0 * rs + c0 * cs;
                    T* pa = d + r0 * cols + c0;
                    T* pb = pa + 4 * cols;
                    if (rs == 1)
                    {
                        long c = 0;
                        // Element-contiguous source columns: whole 8×4 (8-byte) / 8×8 (4-byte) blocks go through AVX
                        // register transposes (the element width is a JIT-time constant, so only one branch survives
                        // per T); the ragged columns and every other width take the scalar run below.
                        if (sizeof(T) == 8 && System.Runtime.Intrinsics.X86.Avx.IsSupported)
                        {
                            for (; c + 4 <= w; c += 4, sa += 4 * cs, pa += 4, pb += 4)
                                TransposeBlock8x4((double*)sa, cs, (double*)pa, (double*)pb, o1, o2, o3);
                        }
                        else if (sizeof(T) == 4 && System.Runtime.Intrinsics.X86.Avx.IsSupported)
                        {
                            for (; c + 8 <= w; c += 8, sa += 8 * cs, pa += 8, pb += 8)
                                TransposeBlock8x8((float*)sa, cs, (float*)pa, (float*)pb, o1, o2, o3);
                        }
                        // The 8 reads of one column are one run at constant displacements.
                        for (; c < w; c++, sa += cs, pa++, pb++)
                        {
                            pa[0] = sa[0]; pa[o1] = sa[1]; pa[o2] = sa[2]; pa[o3] = sa[3];
                            pb[0] = sa[4]; pb[o1] = sa[5]; pb[o2] = sa[6]; pb[o3] = sa[7];
                        }
                    }
                    else
                    {
                        T* sb = sa + 4 * rs;
                        for (long c = 0; c < w; c++, sa += cs, sb += cs, pa++, pb++)
                        {
                            pa[0] = sa[0]; pa[o1] = sa[q1]; pa[o2] = sa[q2]; pa[o3] = sa[q3];
                            pb[0] = sb[0]; pb[o1] = sb[q1]; pb[o2] = sb[q2]; pb[o3] = sb[q3];
                        }
                    }
                }
                for (long r = rows8; r < rows; r++)
                {
                    T* sp = s + r * rs + c0 * cs;
                    T* dr = d + r * cols + c0;
                    for (long c = 0; c < w; c++, sp += cs)
                        dr[c] = *sp;
                }
            }
        }

        /// <summary>
        ///     Transposes one 8-row × 4-column block of 8-byte elements (two 4×4 AVX register transposes) from
        ///     element-contiguous source columns into four-element destination row segments.
        /// </summary>
        /// <param name="s">Row 0 of the block's first source column (its 8 rows are contiguous; column j starts at s + j·cs).</param>
        /// <param name="cs">Source column stride in elements.</param>
        /// <param name="a">Destination of the block's row 0 (rows 1–3 at +o1 / +o2 / +o3).</param>
        /// <param name="b">Destination of the block's row 4 (rows 5–7 at +o1 / +o2 / +o3).</param>
        /// <param name="o1">One destination row pitch in elements.</param>
        /// <param name="o2">Two destination row pitches.</param>
        /// <param name="o3">Three destination row pitches.</param>
        /// <remarks>
        ///     vunpcklpd / vunpckhpd / vperm2f128 only MOVE lanes, so any 64-bit pattern (NaN payloads, integers)
        ///     is copied exactly although the lanes are typed <see cref="double"/>. Every load and store addresses an
        ///     element of the block, so nothing outside the view is touched.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void TransposeBlock8x4(double* s, long cs, double* a, double* b, long o1, long o2, long o3)
        {
            double* s1 = s + cs, s2 = s1 + cs, s3 = s2 + cs;
            // Rows 0..3: v_j holds column j; after the unpacks and 128-bit swaps u_i holds row i.
            var v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s);
            var v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s1);
            var v2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s2);
            var v3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s3);
            var t0 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v0, v1);
            var t1 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v0, v1);
            var t2 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v2, v3);
            var t3 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v2, v3);
            System.Runtime.Intrinsics.X86.Avx.Store(a, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x20));
            System.Runtime.Intrinsics.X86.Avx.Store(a + o1, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x20));
            System.Runtime.Intrinsics.X86.Avx.Store(a + o2, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x31));
            System.Runtime.Intrinsics.X86.Avx.Store(a + o3, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x31));
            // Rows 4..7: the second half of the same four column runs.
            v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 4);
            v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s1 + 4);
            v2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s2 + 4);
            v3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s3 + 4);
            t0 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v0, v1);
            t1 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v0, v1);
            t2 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v2, v3);
            t3 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v2, v3);
            System.Runtime.Intrinsics.X86.Avx.Store(b, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x20));
            System.Runtime.Intrinsics.X86.Avx.Store(b + o1, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x20));
            System.Runtime.Intrinsics.X86.Avx.Store(b + o2, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x31));
            System.Runtime.Intrinsics.X86.Avx.Store(b + o3, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x31));
        }

        /// <summary>
        ///     Transposes one 8×8 block of 4-byte elements (the standard unpack / shuffle / 128-bit-swap AVX sequence)
        ///     from element-contiguous source columns into eight-element destination row segments.
        /// </summary>
        /// <param name="s">Row 0 of the block's first source column (its 8 rows are contiguous; column j starts at s + j·cs).</param>
        /// <param name="cs">Source column stride in elements.</param>
        /// <param name="a">Destination of the block's row 0 (rows 1–3 at +o1 / +o2 / +o3).</param>
        /// <param name="b">Destination of the block's row 4 (rows 5–7 at +o1 / +o2 / +o3).</param>
        /// <param name="o1">One destination row pitch in elements.</param>
        /// <param name="o2">Two destination row pitches.</param>
        /// <param name="o3">Three destination row pitches.</param>
        /// <remarks>
        ///     vunpcklps / vunpckhps / vshufps / vperm2f128 only MOVE lanes: every 32-bit pattern is copied exactly.
        ///     Every load and store addresses an element of the block, so nothing outside the view is touched.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void TransposeBlock8x8(float* s, long cs, float* a, float* b, long o1, long o2, long o3)
        {
            // v_j holds column j (rows 0..7).
            var v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s);
            var v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + cs);
            var v2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 2 * cs);
            var v3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 3 * cs);
            var v4 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 4 * cs);
            var v5 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 5 * cs);
            var v6 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 6 * cs);
            var v7 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 7 * cs);
            var t0 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v0, v1);
            var t1 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v0, v1);
            var t2 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v2, v3);
            var t3 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v2, v3);
            var t4 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v4, v5);
            var t5 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v4, v5);
            var t6 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v6, v7);
            var t7 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v6, v7);
            // u_i: row i of columns 0..3 in the low half, row i+4 in the high half (columns 4..7 for u_{i+4}).
            var u0 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t0, t2, 0x44);
            var u1 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t0, t2, 0xEE);
            var u2 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t1, t3, 0x44);
            var u3 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t1, t3, 0xEE);
            var u4 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t4, t6, 0x44);
            var u5 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t4, t6, 0xEE);
            var u6 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t5, t7, 0x44);
            var u7 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t5, t7, 0xEE);
            System.Runtime.Intrinsics.X86.Avx.Store(a, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u0, u4, 0x20));
            System.Runtime.Intrinsics.X86.Avx.Store(a + o1, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u1, u5, 0x20));
            System.Runtime.Intrinsics.X86.Avx.Store(a + o2, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u2, u6, 0x20));
            System.Runtime.Intrinsics.X86.Avx.Store(a + o3, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u3, u7, 0x20));
            System.Runtime.Intrinsics.X86.Avx.Store(b, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u0, u4, 0x31));
            System.Runtime.Intrinsics.X86.Avx.Store(b + o1, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u1, u5, 0x31));
            System.Runtime.Intrinsics.X86.Avx.Store(b + o2, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u2, u6, 0x31));
            System.Runtime.Intrinsics.X86.Avx.Store(b + o3, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u3, u7, 0x31));
        }

        /// <summary>
        ///     Fills the pinned result with this array's values converted to <typeparamref name="T"/> — astype's
        ///     conversion, in one pass, for any layout.
        /// </summary>
        /// <typeparam name="T">Destination element type (differs from the dtype).</typeparam>
        /// <param name="dst">Pinned destination with room for every element, written in C order.</param>
        /// <param name="target">NumSharp type code of <typeparamref name="T"/>.</param>
        /// <remarks>
        ///     One element goes through the scalar converter astype's copy core applies to one value; a contiguous
        ///     decimal → double source through <see cref="DecimalToDoubleKernel"/> where its start-up probe proved it
        ///     bit-identical to this runtime; everything else through NumSharp's cast kernels, straight into a
        ///     non-owning wrapper over the result. [NoInlining] is the point of the helper: the wrapper construction
        ///     and the iterator call stay out of <see cref="ToMuliDimArray{T}"/>'s frame, so the tiny same-dtype calls
        ///     that never convert do not pay for their stack slots and prologue.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private unsafe void ConvertInto<T>(T* dst, NPTypeCode target) where T : unmanaged
        {
            var storage = Storage;
            ref readonly Shape shape = ref storage.ShapeReference;
            byte* src = storage.Address + shape.offset * storage.DTypeSize;
            long count = shape.size;
            if (count == 1)
                NDIterCasting.ConvertValue(src, dst, storage.TypeCode, target);
            else if (target == NPTypeCode.Double && storage.TypeCode == NPTypeCode.Decimal && shape.IsContiguous
                     && DecimalToDoubleKernel.MatchesRuntime)
                DecimalToDoubleKernel.Convert((decimal*)src, (double*)dst, count);
            else
                NDIter.Copy(new UnmanagedStorage(ArraySlice.Wrap<T>(dst, count), shape.Clean()), storage);
        }

        /// <summary>
        ///     Contiguous decimal → double conversion that reproduces the runtime's own <c>(double)decimal</c>
        ///     bit-for-bit, four lanes at a time: RN(RN(lo64) + hi32·2⁶⁴) / 10^scale, then the sign.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///         Why a replica and not a call: astype converts through <c>Converts.ToDouble(decimal)</c> =
        ///         <c>(double)value</c>, whose scalar arithmetic (a ulong→double conversion, a 2⁶⁴-scaled add, a
        ///         table division) costs several ns per element. The same arithmetic vectorizes exactly: each 32-bit
        ///         half becomes a double through the 2⁵² magic-number trick (exact), the halves recombine with ONE
        ///         rounding (= RN(lo64)), the hi32·2⁶⁴ term is exact, and <c>vdivpd</c> by the gathered power of ten
        ///         is the same IEEE division.
        ///     </para>
        ///     <para>
        ///         Why the probe: that arithmetic is the RUNTIME's, not the language's — .NET 8's ulong→double rounds
        ///         twice for values ≥ 2⁶³ (convert as signed, then add 2⁶⁴), so on .NET 8 the replica disagrees with
        ///         <c>(double)decimal</c> on ~1% of random mantissas (measured: 51,409 of 4,000,000), while on .NET 10
        ///         it matches every one. <see cref="MatchesRuntime"/> converts probe values chosen to discriminate
        ///         exactly that (mantissas ≥ 2⁶³, 96-bit mantissas, scale 28, negative zero) both ways once, and the
        ///         kernel is used only where all agree — a runtime whose conversion differs keeps astype's own path.
        ///     </para>
        ///     <para>
        ///         Invalid decimals (scale above 28 — unreachable through the decimal API, possible only in raw memory)
        ///         send their block to the scalar conversion, so they fail or convert exactly as astype would.
        ///     </para>
        /// </remarks>
        private static class DecimalToDoubleKernel
        {
            /// <summary>True when this process runs on AVX2 hardware and a runtime whose decimal → double
            /// conversion the kernel reproduces bit-for-bit; computed once, on first use.</summary>
            internal static readonly bool MatchesRuntime = Probe();

            /// <summary>Converts the probe set with the kernel and with <c>(double)decimal</c> and compares the bits.</summary>
            /// <returns>True when AVX2 is available and every probe converts identically.</returns>
            private static unsafe bool Probe()
            {
                if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                    return false;
                // 12 values = three whole SIMD blocks, so every probe goes through the vector path. The first two
                // are the .NET 8 counter-examples (mantissas ≥ 2^63 where a twice-rounded ulong→double differs).
                decimal[] probes =
                {
                    -15805.234742681957630m, -1124262054061824103.6m, 18446744073709551615m, 9223372036854775809m,
                    decimal.MaxValue, -7.9228162514264337593543950335m, 0.0000000000000000000000000001m,
                    new decimal(0, 0, 0, true, 5), 42.00m, -99.75m, 12345678901234567890.123456789m, 2.5m,
                };
                double* got = stackalloc double[12];
                fixed (decimal* src = probes)
                    Convert(src, got, 12);
                for (int i = 0; i < 12; i++)
                    if (BitConverter.DoubleToInt64Bits(got[i]) != BitConverter.DoubleToInt64Bits((double)probes[i]))
                        return false;
                return true;
            }

            /// <summary>Converts <paramref name="n"/> contiguous decimals to doubles.</summary>
            /// <param name="src">First source decimal.</param>
            /// <param name="dst">First destination double.</param>
            /// <param name="n">Element count.</param>
            /// <remarks>
            ///     Reads each decimal as its in-memory fields (flags = sign bit 31 and scale bits 16–23, then hi32,
            ///     then lo64 — the layout .NET has used since Core 3.0). The ragged tail, and any block holding a
            ///     scale above 28, uses the scalar <c>(double)decimal</c> conversion itself.
            /// </remarks>
            internal static unsafe void Convert(decimal* src, double* dst, long n)
            {
                long i = 0;
                if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                {
                    double* pow10 = stackalloc double[29]
                    {
                        1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14,
                        1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28,
                    };
                    var magic = System.Runtime.Intrinsics.Vector256.Create(0x4330000000000000UL);   // bit pattern of 2^52
                    var two52 = System.Runtime.Intrinsics.Vector256.Create(4503599627370496.0);
                    var low32 = System.Runtime.Intrinsics.Vector256.Create(0xFFFFFFFFUL);
                    var scaleMask = System.Runtime.Intrinsics.Vector256.Create(0xFFUL);
                    var signMask = System.Runtime.Intrinsics.Vector256.Create(0x80000000UL);
                    var maxScale = System.Runtime.Intrinsics.Vector256.Create(28L);
                    var two32 = System.Runtime.Intrinsics.Vector256.Create(4294967296.0);
                    var two64 = System.Runtime.Intrinsics.Vector256.Create(18446744073709551616.0);
                    for (; i + 4 <= n; i += 4)
                    {
                        // Two loads = four decimals as (head, lo64) pairs; head = flags | hi32 << 32.
                        var a = System.Runtime.Intrinsics.X86.Avx.LoadVector256((ulong*)(src + i));
                        var b = System.Runtime.Intrinsics.X86.Avx.LoadVector256((ulong*)(src + i + 2));
                        var head = System.Runtime.Intrinsics.X86.Avx2.Permute4x64(System.Runtime.Intrinsics.X86.Avx2.UnpackLow(a, b), 0xD8);
                        var lo = System.Runtime.Intrinsics.X86.Avx2.Permute4x64(System.Runtime.Intrinsics.X86.Avx2.UnpackHigh(a, b), 0xD8);
                        var scale = System.Runtime.Intrinsics.X86.Avx2.And(System.Runtime.Intrinsics.X86.Avx2.ShiftRightLogical(head, 16), scaleMask);
                        if (System.Runtime.Intrinsics.X86.Avx.MoveMask(System.Runtime.Intrinsics.Vector256.AsDouble(
                                System.Runtime.Intrinsics.X86.Avx2.CompareGreaterThan(System.Runtime.Intrinsics.Vector256.AsInt64(scale), maxScale))) != 0)
                        {
                            for (long k = i; k < i + 4; k++)
                                dst[k] = (double)src[k];
                            continue;
                        }
                        // lo64 → double with ONE rounding: each 32-bit half is exact via the 2^52 trick, the high half
                        // times 2^32 is exact, and the add rounds once — RN(lo64), as the runtime's conversion.
                        var loLo = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.Vector256.AsDouble(
                            System.Runtime.Intrinsics.X86.Avx2.Or(System.Runtime.Intrinsics.X86.Avx2.And(lo, low32), magic)), two52);
                        var loHi = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.Vector256.AsDouble(
                            System.Runtime.Intrinsics.X86.Avx2.Or(System.Runtime.Intrinsics.X86.Avx2.ShiftRightLogical(lo, 32), magic)), two52);
                        var low = System.Runtime.Intrinsics.X86.Avx.Add(System.Runtime.Intrinsics.X86.Avx.Multiply(loHi, two32), loLo);
                        // + hi32·2^64 (exact product, one rounding in the add), then the IEEE division by 10^scale.
                        var hi = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.Vector256.AsDouble(
                            System.Runtime.Intrinsics.X86.Avx2.Or(System.Runtime.Intrinsics.X86.Avx2.ShiftRightLogical(head, 32), magic)), two52);
                        var sum = System.Runtime.Intrinsics.X86.Avx.Add(low, System.Runtime.Intrinsics.X86.Avx.Multiply(hi, two64));
                        var q = System.Runtime.Intrinsics.X86.Avx.Divide(sum,
                            System.Runtime.Intrinsics.X86.Avx2.GatherVector256(pow10, System.Runtime.Intrinsics.Vector256.AsInt64(scale), 8));
                        // The runtime negates the quotient for a negative decimal: flip the sign bit (negative zero too).
                        var sign = System.Runtime.Intrinsics.X86.Avx2.ShiftLeftLogical(System.Runtime.Intrinsics.X86.Avx2.And(head, signMask), 32);
                        System.Runtime.Intrinsics.X86.Avx.Store(dst + i, System.Runtime.Intrinsics.Vector256.AsDouble(
                            System.Runtime.Intrinsics.X86.Avx2.Xor(System.Runtime.Intrinsics.Vector256.AsUInt64(q), sign)));
                    }
                }
                for (; i < n; i++)
                    dst[i] = (double)src[i];
            }
        }

        /// <summary>
        ///     Allocates rank ≥ 4 results through IL generated once per (T, rank): a direct <c>newobj</c> of the rank-N
        ///     array constructor over the NumSharp dimensions — no reflection-driven activation per call.
        /// </summary>
        /// <typeparam name="T">Element type of the result.</typeparam>
        /// <remarks>
        ///     <para>
        ///         Why not <see cref="System.Array.CreateInstance(Type, int[])"/> on every call: it validates the element
        ///         type through the runtime's type system and needs a freshly allocated <c>int[]</c> of lengths (plus a
        ///         conversion pass) each time — measured ~215–240 ns for a small rank-4..8 array against ~37–52 ns for the
        ///         emitted <c>newobj</c>. The emitted body narrows each dimension through <see cref="ManagedLength"/> (same
        ///         check, same exception, same left-to-right order) and passes the lengths straight to the constructor, so
        ///         nothing but the result is allocated.
        ///     </para>
        ///     <para>
        ///         Each rank's delegate is built on first use and cached in a per-T table; a race only builds the same
        ///         delegate twice (reference stores are atomic). Ranks above 32 — where the runtime refuses the array type
        ///         with a <see cref="TypeLoadException"/> — and hosts without dynamic code (NativeAOT) keep
        ///         <see cref="System.Array.CreateInstance(Type, int[])"/>, whose behavior this reproduces everywhere else.
        ///     </para>
        /// </remarks>
        private static class RankNAllocator<T> where T : unmanaged
        {
            /// <summary>One cached allocator per rank (index = rank, 4..32); null until first use.</summary>
            private static readonly Func<long[], Array>[] s_byRank = new Func<long[], Array>[33];

            /// <summary>Allocates a zero-filled <c>T[,…]</c> with the given lengths.</summary>
            /// <param name="dims">NumSharp dimensions; the length is the rank (at least 4).</param>
            /// <returns>The new array.</returns>
            /// <exception cref="InvalidOperationException">A dimension exceeds <see cref="int.MaxValue"/>.</exception>
            /// <exception cref="TypeLoadException">The rank exceeds 32, the .NET array rank limit.</exception>
            internal static Array Allocate(long[] dims)
            {
                int rank = dims.Length;
                if (rank > 32 || !System.Runtime.CompilerServices.RuntimeFeature.IsDynamicCodeSupported)
                    return System.Array.CreateInstance(typeof(T), System.Array.ConvertAll(dims, ManagedLength));
                var allocate = s_byRank[rank] ??= Build(rank);
                return allocate(dims);
            }

            /// <summary>Emits the allocator for one rank.</summary>
            /// <param name="rank">Array rank, 4..32.</param>
            /// <returns>A delegate that narrows each dimension and constructs <c>T[,…]</c> with <c>newobj</c>.</returns>
            private static Func<long[], Array> Build(int rank)
            {
                var lengths = new Type[rank];
                for (int i = 0; i < rank; i++)
                    lengths[i] = typeof(int);
                var ctor = typeof(T).MakeArrayType(rank).GetConstructor(lengths);
                var narrow = typeof(NDArray).GetMethod(nameof(ManagedLength),
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
                // Owned by NDArray, so the body may call the private ManagedLength with no run-time visibility check.
                var method = new System.Reflection.Emit.DynamicMethod("ToMuliDimArray_NewArray" + rank, typeof(Array),
                    new[] { typeof(long[]) }, typeof(NDArray), skipVisibility: true);
                var il = method.GetILGenerator();
                for (int i = 0; i < rank; i++)
                {
                    il.Emit(System.Reflection.Emit.OpCodes.Ldarg_0);
                    il.Emit(System.Reflection.Emit.OpCodes.Ldc_I4, i);
                    il.Emit(System.Reflection.Emit.OpCodes.Ldelem_I8);
                    il.Emit(System.Reflection.Emit.OpCodes.Call, narrow);
                }
                il.Emit(System.Reflection.Emit.OpCodes.Newobj, ctor);
                il.Emit(System.Reflection.Emit.OpCodes.Ret);
                return (Func<long[], Array>)method.CreateDelegate(typeof(Func<long[], Array>));
            }
        }


    }

}
