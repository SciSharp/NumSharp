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
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Leaf size, in bytes, from which a leaf is allocated uninitialized. It mirrors the runtime's own
        ///     cut-over inside <see cref="GC.AllocateUninitializedArray{T}(int, bool)"/>. Below it, zeroing is folded
        ///     into the pre-cleared allocation context and a plain <c>new T[n]</c> is the faster path (the JIT's
        ///     inline allocation helper, no call). Above it, the GC's clearing pass would write every byte that the
        ///     fill then overwrites.
        /// </summary>
        private const int UninitializedLeafBytes = 2048;

        /// <summary>
        ///     Test hook: when set on the calling thread, <see cref="ToJaggedArray{T}"/> builds rank ≥ 3 results with its
        ///     no-dynamic-code fallback (a recursive descent over reflection-created levels) and
        ///     <see cref="ToMuliDimArray{T}"/> allocates rank ≥ 4 results with
        ///     <see cref="System.Array.CreateInstance(Type, int[])"/> — exactly what a runtime without dynamic code
        ///     (NativeAOT, the interpreter) runs. A JIT-compiled test process never takes those paths on its own, so
        ///     this is the only way a test can prove them equal to the emitted ones.
        /// </summary>
        /// <remarks>
        ///     Thread-static, so a test that sets it re-routes only its own calls, never another thread's; production
        ///     code never sets it. A test resets it in a <c>finally</c>.
        /// </remarks>
        [ThreadStatic]
        internal static bool ForceNoDynamicCodeFallbacks;

        /// <summary>
        ///     Copies this array into a fresh .NET jagged array (<c>T[]</c> for 1-D, <c>T[][]</c> for 2-D, …,
        ///     one nesting level per dimension, any rank), reading the logical elements in C order whatever
        ///     the memory layout (C/F-contiguous, transposed, strided, reversed, offset or broadcast views).
        /// </summary>
        /// <typeparam name="T">
        ///     Element type of the result. When it differs from this array's dtype, the values are converted
        ///     exactly as <see cref="astype(DType, bool)"/> converts them. A type that is not a NumSharp dtype is
        ///     refused the same way <c>astype</c> refuses it.
        /// </typeparam>
        /// <returns>
        ///     A new jagged array, statically typed as <see cref="Array"/>; cast it to the nested type
        ///     (e.g. <c>(double[][])</c>). Every sub-array is a distinct object, and nothing is shared with
        ///     this array.
        /// </returns>
        /// <exception cref="InvalidOperationException">This array is 0-d (a scalar has no jagged form), or a
        ///     dimension exceeds <see cref="int.MaxValue"/> (managed arrays are int-indexed).</exception>
        /// <exception cref="NotSupportedException">T is not one of NumSharp's dtypes (raised by
        ///     <c>astype</c> itself, so the exception is exactly the one <c>astype</c> raises).</exception>
        /// <remarks>
        ///     <para>
        ///     The work is allocation-shaped: the only per-element cost is one read and one write (a conversion
        ///     when T differs from the dtype), and no temporary array is ever allocated. Each leaf's fixed cost is
        ///     kept to its allocation:
        ///     </para>
        ///     <list type="bullet">
        ///         <item><description>
        ///         Small leaves are created inline in one row loop (allocate, copy or gather, typed store), with no
        ///         helper call per row. Such a call measured about 1.1 ns per leaf (20⁴ int16: 8 000 leaves of 20).
        ///         </description></item>
        ///         <item><description>
        ///         A block's big leaves are allocated first, all of them, and filled in a second pass. That keeps
        ///         the allocator's slow path, which every big leaf takes, hot; see <see cref="FillBlock{T}"/>.
        ///         </description></item>
        ///         <item><description>
        ///         The row's size class (uninitialized or zeroed allocation) and access pattern (block copy or
        ///         gather) are decided once per block, outside that loop.
        ///         </description></item>
        ///         <item><description>
        ///         Leaves of <see cref="UninitializedLeafBytes"/> or more skip the GC's zeroing pass.
        ///         </description></item>
        ///         <item><description>
        ///         A block whose rows are adjacent source columns (a transposed or F-contiguous 2-D block) of
        ///         4-, 8- or 16-byte elements is filled in <see cref="TransposeTileRows"/> ×
        ///         <see cref="TransposeTileColumns"/> tiles by AVX register transposes (<see cref="TransposeBand4"/>
        ///         for 8-byte elements, <see cref="TransposeBand8"/> for 4-byte ones, <see cref="TransposeBand2"/> for
        ///         complex128 and decimal), instead of one strided gather per element. Big and small leaves alike:
        ///         small ones from <see cref="SmallTransposeMinBytes"/>, a tile row's leaves allocated just before they
        ///         are filled; see <see cref="TransposeRows{T}"/>.
        ///         </description></item>
        ///         <item><description>
        ///         A reversed row (stride −1) is copied by a fused AVX2 reverse (4- and 8-byte elements) or a block
        ///         copy plus the BCL's vectorized in-place reverse (other widths), and a broadcast row (stride 0) is
        ///         one <see cref="Span{T}.Fill"/>; see <see cref="GatherRow{T}"/>.
        ///         </description></item>
        ///         <item><description>
        ///         When T differs from the dtype, every source row is converted straight into its leaf by
        ///         NumSharp's own cast kernels (<see cref="RowConverter{T}"/>), in the order <c>astype</c>'s copy
        ///         core uses. The whole-array conversion temp, and with it one extra write and one extra read of
        ///         all the converted data, is never created.
        ///         </description></item>
        ///     </list>
        ///     <para>
        ///     The levels above the blocks (rank ≥ 3) are built by an IL-emitted, rank-specialised method: one per
        ///     (T, rank, exact-or-converted), compiled on first use and cached. Each level is allocated with its
        ///     exact nested type (<c>newarr</c>) and stored with a typed <c>stelem</c>, so there is no per-call
        ///     <c>MakeArrayType</c>, <c>Array.CreateInstance</c> or covariant store. Runtimes that cannot compile
        ///     dynamic code fall back to a recursive descent with reflection-created levels and the same results.
        ///     The pointer follows the source's own strides, so every layout is read in place.
        ///     </para>
        /// </remarks>
        public unsafe Array ToJaggedArray<T>() where T : unmanaged
        {
            if (ndim == 0)
                throw new InvalidOperationException("A 0-d array has no jagged form; read the scalar with GetValue or item().");

            NPTypeCode target = InfoOf<T>.NPTypeCode;
            if (target != typecode)
            {
                // A T with no NumSharp dtype has no conversion. Let astype refuse it, so the exception is exactly
                // astype's; the bits of a foreign type are never reinterpreted.
                if (!IsStorageDType(target))
                {
                    using (astype(typeof(T)))
                    {
                    }

                    throw new NotSupportedException($"{typeof(T).Name} is not a NumSharp dtype.");
                }

                return ToJaggedArrayConverted<T>(target);
            }

            try
            {
                Shape shape = Shape;
                long[] dims = shape.dimensions;
                long[] strides = shape.strides;
                int nd = dims.Length;
                ThrowIfAnyDimensionExceedsInt(dims);

                // Logical element 0 of any view: buffer base + element offset (strided/offset views keep their
                // start in Shape.offset; contiguous slices were re-based into Address with offset 0).
                T* first = (T*)Storage.Address + shape.offset;
                if (nd == 1)
                    return NewLeaf(first, (int)dims[0], strides[0]);
                if (nd == 2)
                    return FillBlock(first, (int)dims[0], strides[0], (int)dims[1], strides[1]);

                // NativeAOT / interpreter: emitting IL there is either impossible or slower than reflection. (The test
                // hook takes the same fallback on a JIT runtime, so the fallback is covered by tests.)
                if (!RuntimeFeature.IsDynamicCodeCompiled || ForceNoDynamicCodeFallbacks)
                    return JaggedLevel(first, dims, strides, LevelElementTypes<T>(nd), 0);

                BlockLevelsBuilder<T> build = BlockLevelsBuilders<T>.ByRank.GetOrAdd(nd, static rank => (BlockLevelsBuilder<T>)EmitBlockLevelsBuilder<T>(rank, converted: false));
                return build(first, dims, strides);
            }
            finally
            {
                // The raw pointer is only valid while the source buffer is alive.
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        ///     The T ≠ dtype path: builds the jagged result by converting every source row straight into its leaf.
        /// </summary>
        /// <typeparam name="T">Element type of the result (a NumSharp dtype other than this array's).</typeparam>
        /// <param name="target">T's type code.</param>
        /// <returns>The converted jagged array.</returns>
        /// <exception cref="InvalidOperationException">A dimension exceeds <see cref="int.MaxValue"/>.</exception>
        /// <remarks>
        ///     <para>
        ///     The conversion follows <c>astype</c>'s copy core (the NDIter copy) decision by decision, so every
        ///     value is converted by the same code astype would use.
        ///     </para>
        ///     <list type="bullet">
        ///         <item><description>
        ///         A source that is one value broadcast everywhere (every stride 0) is converted ONCE with
        ///         <see cref="NDIterCasting.ConvertValue"/> and replicated, exactly as the copy core's
        ///         scalar-broadcast fast path does.
        ///         </description></item>
        ///         <item><description>
        ///         Otherwise each row uses the contiguous cast kernel when its elements are adjacent. The copy core
        ///         uses that kernel for a contiguous whole, and its strided kernel runs the same SIMD body on any
        ///         unit-stride inner axis. Other rows use the strided cast kernel, and a pair with no IL kernel
        ///         takes the iterator's per-element cast.
        ///         </description></item>
        ///     </list>
        ///     <para>
        ///     Casts are element-wise and stateless, so converting row by row gives the same bits as converting
        ///     the whole array at once.
        ///     </para>
        /// </remarks>
        private unsafe Array ToJaggedArrayConverted<T>(NPTypeCode target) where T : unmanaged
        {
            try
            {
                Shape shape = Shape;
                long[] dims = shape.dimensions;
                long[] strides = shape.strides;
                int nd = dims.Length;
                ThrowIfAnyDimensionExceedsInt(dims);

                int itemSize = InfoOf.GetSize(typecode);
                byte* first = (byte*)Storage.Address + shape.offset * itemSize;

                RowConverter<T> converter;
                if (shape.IsScalarBroadcast && shape.size > 0)
                {
                    T value;
                    NDIterCasting.ConvertValue(first, &value, typecode, target);
                    converter = new RowConverter<T>(value);
                }
                else
                {
                    converter = new RowConverter<T>(typecode, target);
                }

                if (nd == 1)
                    return NewLeafConverted(first, (int)dims[0], strides[0], converter);
                if (nd == 2)
                    return FillBlockConverted(first, (int)dims[0], strides[0] * itemSize, (int)dims[1], strides[1], converter);

                // No dynamic code (or the test hook): the reflection-built fallback, as in the exact path.
                if (!RuntimeFeature.IsDynamicCodeCompiled || ForceNoDynamicCodeFallbacks)
                    return JaggedLevelConverted(first, dims, strides, itemSize, LevelElementTypes<T>(nd), 0, converter);

                ConvertedBuilder<T> build = ConvertedBuilders<T>.ByRank.GetOrAdd(nd, static rank => (ConvertedBuilder<T>)EmitBlockLevelsBuilder<T>(rank, converted: true));
                return build(first, dims, strides, itemSize, converter);
            }
            finally
            {
                // The raw pointer is only valid while the source buffer is alive.
                GC.KeepAlive(this);
            }
        }

        /// <summary>Throws when a dimension cannot be a .NET array length.</summary>
        /// <param name="dims">Source dimensions.</param>
        /// <exception cref="InvalidOperationException">A dimension exceeds <see cref="int.MaxValue"/>.</exception>
        private static void ThrowIfAnyDimensionExceedsInt(long[] dims)
        {
            for (int d = 0; d < dims.Length; d++)
                if (dims[d] > int.MaxValue)
                    throw new InvalidOperationException($"Dimension {d} has length {dims[d]}, which exceeds int.MaxValue ({int.MaxValue}); .NET arrays are int-indexed.");
        }

        /// <summary>Whether <paramref name="code"/> is one of the 15 dtypes NumSharp can store and convert to.</summary>
        /// <param name="code">A type code, possibly <see cref="NPTypeCode.Empty"/> for a foreign type.</param>
        /// <returns>True for the 15 storage dtypes.</returns>
        private static bool IsStorageDType(NPTypeCode code)
        {
            switch (code)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal:
                case NPTypeCode.Complex:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     Converts source rows of one (source dtype → T) pair into leaves, with the kernels resolved once per
        ///     call. It can also replicate one pre-converted value, for a fully broadcast source.
        /// </summary>
        /// <typeparam name="T">Element type of the leaves.</typeparam>
        /// <remarks>
        ///     Immutable after construction and created per call, so it carries no shared state. The kernel
        ///     lookups hit NumSharp's own kernel caches, so constructing one costs two dictionary reads.
        /// </remarks>
        private sealed unsafe class RowConverter<T> where T : unmanaged
        {
            /// <summary>Source dtype.</summary>
            private readonly NPTypeCode _source;

            /// <summary>Target dtype (T's).</summary>
            private readonly NPTypeCode _target;

            /// <summary>Contiguous cast kernel, or null when the pair has none.</summary>
            private readonly DirectILKernelGenerator.CastKernel _contiguous;

            /// <summary>Strided cast kernel, or null when the pair has none (e.g. a Boolean source).</summary>
            private readonly DirectILKernelGenerator.StridedCastKernel _strided;

            /// <summary>True when every element is <see cref="_value"/> (fully broadcast source).</summary>
            private readonly bool _fill;

            /// <summary>The single converted value of a fully broadcast source.</summary>
            private readonly T _value;

            /// <summary>Creates a converter for rows of <paramref name="source"/> elements.</summary>
            /// <param name="source">Source dtype.</param>
            /// <param name="target">T's dtype.</param>
            public RowConverter(NPTypeCode source, NPTypeCode target)
            {
                _source = source;
                _target = target;
                _contiguous = DirectILKernelGenerator.TryGetCastKernel(source, target);
                _strided = DirectILKernelGenerator.TryGetStridedCastKernel(source, target);
            }

            /// <summary>Creates a converter that writes <paramref name="value"/> everywhere.</summary>
            /// <param name="value">The value, already converted exactly as astype converts a broadcast scalar.</param>
            public RowConverter(T value)
            {
                _fill = true;
                _value = value;
            }

            /// <summary>
            ///     Writes <paramref name="count"/> converted elements to <paramref name="dst"/> from the source row at
            ///     <paramref name="src"/>.
            /// </summary>
            /// <param name="src">Address of the row's first source element.</param>
            /// <param name="srcStride">Source element stride along the row (any sign; 0 = broadcast).</param>
            /// <param name="dst">Address of the (pinned) leaf's first element.</param>
            /// <param name="count">Row length.</param>
            public void Convert(byte* src, long srcStride, T* dst, int count)
            {
                if (count == 0)
                    return;
                if (_fill)
                {
                    new Span<T>(dst, count).Fill(_value);
                    return;
                }

                if (srcStride == 1 && _contiguous != null)
                {
                    _contiguous(src, dst, count);
                    return;
                }

                // One-dimensional strided descriptors, in elements, for the strided kernel / iterator cast.
                long srcStrides = srcStride;
                long dstStrides = 1;
                long shape = count;
                if (_strided != null)
                {
                    _strided(src, dst, &srcStrides, &dstStrides, &shape, 1);
                    return;
                }

                NDIterCasting.CopyStridedToStridedWithCast(src, &srcStrides, _source, dst, &dstStrides, _target, &shape, 1, count);
            }
        }

        /// <summary>
        ///     Allocates the single leaf of a 1-D result and fills it from the source at <paramref name="p"/>. A leaf
        ///     of <see cref="UninitializedLeafBytes"/> or more is allocated uninitialized: every element is written
        ///     below, so the GC's clearing pass would only add a full extra write of the leaf.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="p">Address of the row's first element.</param>
        /// <param name="n">Row length.</param>
        /// <param name="stride">Element stride along the row (1 = adjacent, 0 = broadcast, negative = reversed).</param>
        /// <returns>The new leaf, every element written.</returns>
        private static unsafe T[] NewLeaf<T>(T* p, int n, long stride) where T : unmanaged
        {
            T[] leaf = (long)n * sizeof(T) >= UninitializedLeafBytes ? GC.AllocateUninitializedArray<T>(n) : new T[n];
            if (stride == 1)
            {
                // Adjacent elements: one vectorized block copy.
                new ReadOnlySpan<T>(p, n).CopyTo(leaf);
            }
            else
            {
                // Strided, reversed or broadcast row: GatherRow picks a fill, a reverse copy or an element gather.
                GatherRow(p, stride, ref MemoryMarshal.GetArrayDataReference(leaf), n);
            }

            return leaf;
        }

        /// <summary>
        ///     Allocates the single leaf of a converted 1-D result and converts the source row into it.
        /// </summary>
        /// <typeparam name="T">Element type of the result.</typeparam>
        /// <param name="p">Address of the row's first source element.</param>
        /// <param name="n">Row length.</param>
        /// <param name="stride">Source element stride along the row.</param>
        /// <param name="converter">The call's converter.</param>
        /// <returns>The new leaf, every element written.</returns>
        private static unsafe T[] NewLeafConverted<T>(byte* p, int n, long stride, RowConverter<T> converter) where T : unmanaged
        {
            T[] leaf = (long)n * sizeof(T) >= UninitializedLeafBytes ? GC.AllocateUninitializedArray<T>(n) : new T[n];

            // The kernels write through a raw pointer, so the leaf is pinned for the call.
            fixed (T* d = leaf)
                converter.Convert(p, stride, d, n);
            return leaf;
        }

        /// <summary>
        ///     Builds one <c>T[][]</c> block: <paramref name="rows"/> leaves, each read from the source starting
        ///     <paramref name="rowStep"/> elements after the previous one. Small leaves are created inline in one
        ///     row loop (small transposed leaves a tile row at a time, see <see cref="TransposeRows{T}"/>). Leaves of
        ///     <see cref="UninitializedLeafBytes"/> or more are allocated first, all of them, and filled in a second
        ///     pass. The size class and the access pattern are decided once, outside the loops.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="p">Address of the block's first element.</param>
        /// <param name="rows">Number of rows (the second-to-last dimension).</param>
        /// <param name="rowStep">Element stride between consecutive rows (0 = broadcast, negative = reversed).</param>
        /// <param name="rowLength">Row length (the last dimension).</param>
        /// <param name="rowStride">Element stride along a row.</param>
        /// <returns>The typed block, every element of every leaf written.</returns>
        /// <remarks>
        ///     <para>
        ///     Allocation order. A big leaf is uninitialized, so creating it touches none of its data. It also does
        ///     not fit the thread's 8 KB allocation context, so every big leaf takes the allocator's slow path.
        ///     Allocating all of a block's big leaves back to back keeps that slow path hot in the instruction
        ///     cache and the branch predictor, and leaves the fill pass as a pure copy or gather loop. The split
        ///     costs nothing: each line's first touch happens in the fill either way (a 1000 × 1000 float64
        ///     copy measured 5 % faster). Small leaves stay interleaved on purpose: measured at every block size,
        ///     the split lost for them even on a 1.3 KB, cache-resident block (+8–36 %), because the second pass
        ///     itself (every leaf reference re-loaded, a second loop) costs more than batching the allocations saves.
        ///     </para>
        ///     <para>
        ///     Stores use <c>block[r]</c> with <c>r &lt; block.Length</c>, which gives the JIT three things: the
        ///     bounds check folds away, the exact types (a fresh <c>T[]</c> into a fresh <c>T[][]</c>) remove the
        ///     covariant store check, and the unchecked write barrier suffices because the target is a heap array.
        ///     The block itself must be zeroed: it holds references, and the GC scans it.
        ///     </para>
        ///     <para>
        ///     Transposed blocks (leaves whose rows are adjacent source columns, of an element width
        ///     <see cref="CanTransposeRows{T}"/> accepts) are filled in tiles by register transposes. Big 8-byte leaves
        ///     use the tile loop inline in pass 2, which stays there on purpose (see its comment). Every other case goes
        ///     through <see cref="TransposeRows{T}"/>: other big leaves after pass 1, and small ones (from
        ///     <see cref="SmallTransposeMinBytes"/>) allocated a tile row at a time. The rows the tiles leave over
        ///     (fewer than one sub-band) go through <see cref="GatherRow{T}"/>.
        ///     </para>
        /// </remarks>
        private static unsafe T[][] FillBlock<T>(T* p, int rows, long rowStep, int rowLength, long rowStride) where T : unmanaged
        {
            var block = new T[rows][];
            if ((long)rowLength * sizeof(T) >= UninitializedLeafBytes)
            {
                // Pass 1: every big leaf of the block, uninitialized (its data is fully written in pass 2).
                for (int r = 0; r < block.Length; r++)
                    block[r] = GC.AllocateUninitializedArray<T>(rowLength);

                // Pass 2: fill each leaf with a block copy (adjacent row elements) or a gather (anything else).
                if (rowStride == 1)
                {
                    for (int r = 0; r < block.Length; r++)
                    {
                        new ReadOnlySpan<T>(p, rowLength).CopyTo(block[r]);
                        p += rowStep;
                    }
                }
                else
                {
                    int r = 0;

                    // Rows that are adjacent source columns (row step 1) of 8-byte elements: tiles of
                    // TransposeTileRows leaves × TransposeTileColumns positions, walked in row-of-tiles order. Each
                    // tile is 8 sub-bands of 4 leaves; position i of leaves r..r+3 is the 32-byte run
                    // p[r + i·rowStride .. +3], and a 4×4 register transpose turns four such runs into four 32-byte
                    // stores. sizeof(T) is a JIT-time constant, so other widths compile this away.
                    // This loop stays inline here instead of moving into TransposeRows, which serves every other
                    // transposed case. Dynamic PGO inlines TransposeBand4 into whichever method holds the loop. Routed
                    // through the helper, the float64 1000×1000 transpose ran at ≈400 µs in some processes and
                    // ≈455–490 µs in others; inline here it was steady at ≈400 µs (measured across processes).
                    if (sizeof(T) == 8 && rowStep == 1 && Avx.IsSupported)
                    {
                        for (; r + TransposeTileRows <= block.Length; r += TransposeTileRows)
                        {
                            for (int j0 = 0; j0 < rowLength; j0 += TransposeTileColumns)
                            {
                                int count = Math.Min(TransposeTileColumns, rowLength - j0);
                                double* tile = (double*)p + j0 * rowStride;
                                for (int k = 0; k < TransposeTileRows; k += 4)
                                {
                                    TransposeBand4(
                                        tile + k, rowStride, count,
                                        ref LeafAt(block[r + k], j0),
                                        ref LeafAt(block[r + k + 1], j0),
                                        ref LeafAt(block[r + k + 2], j0),
                                        ref LeafAt(block[r + k + 3], j0));
                                }
                            }

                            p += TransposeTileRows;
                        }

                        // Fewer than TransposeTileRows rows left: 4-row bands over the full row length.
                        for (; r + 4 <= block.Length; r += 4)
                        {
                            TransposeBand4(
                                (double*)p, rowStride, rowLength,
                                ref LeafAt(block[r], 0),
                                ref LeafAt(block[r + 1], 0),
                                ref LeafAt(block[r + 2], 0),
                                ref LeafAt(block[r + 3], 0));
                            p += 4;
                        }
                    }
                    else if (rowStep == 1 && CanTransposeRows<T>(rowLength))
                    {
                        // The other widths the register transposes handle: the shared tile helper, over the leaves
                        // pass 1 allocated. It fills whole sub-bands only and returns how many rows it covered; the
                        // row step is 1, so the pointer moves by the same count.
                        r = TransposeRows(p, block, rowLength, rowStride, allocate: false);
                        p += r;
                    }

                    // Every other strided row (and the rows a transposed block leaves over): fill, reverse copy or
                    // gather.
                    for (; r < block.Length; r++)
                    {
                        GatherRow(p, rowStride, ref MemoryMarshal.GetArrayDataReference(block[r]), rowLength);
                        p += rowStep;
                    }
                }
            }
            else if (rowStride == 1)
            {
                // Small adjacent rows: allocate, copy and store in one pass while the zeroed lines are in L1.
                for (int r = 0; r < block.Length; r++)
                {
                    var leaf = new T[rowLength];
                    new ReadOnlySpan<T>(p, rowLength).CopyTo(leaf);
                    block[r] = leaf;
                    p += rowStep;
                }
            }
            else
            {
                int r = 0;

                // Small transposed leaves at least SmallTransposeMinBytes long: the element gather would issue one
                // scalar load and one store per element, so they take the same tiled transposes, each tile row's
                // leaves allocated (zeroed, small) right before the tiles fill them.
                if (rowStep == 1 && CanTransposeRows<T>(rowLength) && (long)rowLength * sizeof(T) >= SmallTransposeMinBytes)
                {
                    r = TransposeRows(p, block, rowLength, rowStride, allocate: true);
                    p += r;
                }

                // Small strided, reversed or broadcast rows (and the rows a transposed block leaves over): allocate,
                // fill / reverse copy / gather, store, one pass.
                for (; r < block.Length; r++)
                {
                    var leaf = new T[rowLength];
                    GatherRow(p, rowStride, ref MemoryMarshal.GetArrayDataReference(leaf), rowLength);
                    block[r] = leaf;
                    p += rowStep;
                }
            }

            return block;
        }

        /// <summary>
        ///     Builds one converted <c>T[][]</c> block: <paramref name="rows"/> leaves, each converted from the source
        ///     row starting <paramref name="rowStepBytes"/> bytes after the previous one. The allocation policy is
        ///     the same as <see cref="FillBlock{T}"/>'s.
        /// </summary>
        /// <typeparam name="T">Element type of the result.</typeparam>
        /// <param name="p">Address of the block's first source element.</param>
        /// <param name="rows">Number of rows.</param>
        /// <param name="rowStepBytes">Byte stride between consecutive source rows.</param>
        /// <param name="rowLength">Row length.</param>
        /// <param name="rowStride">Source element stride along a row.</param>
        /// <param name="converter">The call's converter.</param>
        /// <returns>The typed block, every element of every leaf written.</returns>
        /// <remarks>Each leaf is pinned only for its own kernel call; the kernels write through raw pointers.</remarks>
        private static unsafe T[][] FillBlockConverted<T>(byte* p, int rows, long rowStepBytes, int rowLength, long rowStride, RowConverter<T> converter) where T : unmanaged
        {
            var block = new T[rows][];
            if ((long)rowLength * sizeof(T) >= UninitializedLeafBytes)
            {
                for (int r = 0; r < block.Length; r++)
                    block[r] = GC.AllocateUninitializedArray<T>(rowLength);

                for (int r = 0; r < block.Length; r++)
                {
                    fixed (T* d = block[r])
                        converter.Convert(p, rowStride, d, rowLength);
                    p += rowStepBytes;
                }
            }
            else
            {
                for (int r = 0; r < block.Length; r++)
                {
                    var leaf = new T[rowLength];
                    fixed (T* d = leaf)
                        converter.Convert(p, rowStride, d, rowLength);
                    block[r] = leaf;
                    p += rowStepBytes;
                }
            }

            return block;
        }

        /// <summary>
        ///     Leaves per transposed tile: 8 sub-bands of the 4×4 kernel (8-byte elements), 4 of the 8×8 kernel
        ///     (4-byte) or 16 of the 2×2 kernel (16-byte). A tile row keeps 32 sequential write streams, and each
        ///     source row contributes 32 elements (32 × 8 bytes = 4 lines, or 5 when unaligned), all consumed while
        ///     the tile is L1-resident.
        /// </summary>
        private const int TransposeTileRows = 32;

        /// <summary>
        ///     Positions per transposed tile. For 8-byte elements, 64 positions × 4 lines = 256 source lines (16 KB)
        ///     plus 32 × 64 × 8 bytes = 16 KB of stores fit a 48 KB L1D together; 4-byte tiles are half that, and
        ///     16-byte tiles take half as many positions to stay at the 8-byte footprint.
        /// </summary>
        private const int TransposeTileColumns = 64;

        /// <summary>
        ///     Shortest small leaf, in bytes, that the tiled transpose takes over from the element gather: two
        ///     iterations of the widest 4- or 8-byte kernel step (8 × 8-byte or 16 × 4-byte elements). Below that the
        ///     kernel's per-call cost is not repaid, so shorter transposed rows stay on the gather. 16-byte leaves
        ///     never take the small path: they tile only from <see cref="WideTransposeMinLength"/> positions.
        /// </summary>
        private const int SmallTransposeMinBytes = 64;

        /// <summary>
        ///     Shortest leaf, in 16-byte elements (complex128, decimal), that takes the tiled transpose. The element
        ///     gather already moves one whole 16-byte lane per load and store, so the tiles pay off only once a leaf's
        ///     column walk outgrows the L1 and the TLB (one page per position once a source row spans 4 KB).
        ///     Measured: complex128 transposes ran 2.06× faster at 2000 positions (30.2 → 14.7 ms),
        ///     level at 700 (≈390 µs either way) and 9–17 % slower at 60 and 500 positions.
        /// </summary>
        private const int WideTransposeMinLength = 1024;

        /// <summary>
        ///     A byref to position <paramref name="position"/> of an 8-byte-element leaf, viewed as <see cref="double"/>
        ///     lanes for <see cref="TransposeBand4"/>. Pure reinterpretation: no copy, no pinning.
        /// </summary>
        /// <typeparam name="T">The leaf's element type (8 bytes wide; callers check).</typeparam>
        /// <param name="leaf">The leaf.</param>
        /// <param name="position">Element index inside the leaf (in range; callers guarantee it).</param>
        /// <returns>A GC-tracked reference to that element.</returns>
        private static ref double LeafAt<T>(T[] leaf, int position) where T : unmanaged
            => ref System.Runtime.CompilerServices.Unsafe.Add(
                ref System.Runtime.CompilerServices.Unsafe.As<T, double>(ref MemoryMarshal.GetArrayDataReference(leaf)), position);

        /// <summary>
        ///     Fills positions <c>0 .. length − 1</c> of four leaves from a transposed source with AVX 4×4 register
        ///     transposes: position i of leaf k (k = 0..3) is <c>src[k + i·rowStride]</c>. Four 32-byte loads (positions
        ///     i..i+3, each holding the four leaves' values) become four 32-byte stores (four positions of one leaf).
        /// </summary>
        /// <param name="src">Address of position 0 of leaf 0 (the four leaves' values at a position are adjacent).</param>
        /// <param name="rowStride">Element stride between consecutive positions (any sign; the source's column stride).</param>
        /// <param name="length">Number of positions to fill in each leaf.</param>
        /// <param name="d0">Position 0 of leaf 0 (a GC-tracked reference, so the leaf needs no pinning).</param>
        /// <param name="d1">Position 0 of leaf 1.</param>
        /// <param name="d2">Position 0 of leaf 2.</param>
        /// <param name="d3">Position 0 of leaf 3.</param>
        /// <remarks>
        ///     <para>
        ///     Callers check <see cref="Avx.IsSupported"/> and an 8-byte element size. The elements are handled as
        ///     <see cref="double"/> lanes, but only through loads, <c>vunpcklpd</c>/<c>vunpckhpd</c>,
        ///     <c>vperm2f128</c> and stores, none of which inspects or alters a value. So any 8-byte bit pattern
        ///     (int64, uint64, NaN payloads, signalling NaNs) is copied exactly.
        ///     </para>
        ///     <para>
        ///     The leaves are written through byrefs (<c>Vector256.StoreUnsafe(ref, offset)</c> and
        ///     <c>Unsafe.Add</c>), which the GC tracks and updates if it moves a leaf mid-call, so no leaf is pinned.
        ///     The source is NumSharp's unmanaged buffer, which never moves.
        ///     </para>
        /// </remarks>
        private static unsafe void TransposeBand4(double* src, long rowStride, int length, ref double d0, ref double d1, ref double d2, ref double d3)
        {
            nuint n = (nuint)length;
            nuint i = 0;
            long step4 = 4 * rowStride;
            for (; i + 4 <= n; i += 4)
            {
                // a..e hold positions i..i+3; lane k of each is leaf k's value there.
                Vector256<double> a = Avx.LoadVector256(src);
                Vector256<double> b = Avx.LoadVector256(src + rowStride);
                Vector256<double> c = Avx.LoadVector256(src + 2 * rowStride);
                Vector256<double> e = Avx.LoadVector256(src + 3 * rowStride);

                // In-lane interleave, then swap 128-bit halves: t0 = a0 b0 a2 b2, t1 = a1 b1 a3 b3,
                // t2 = c0 e0 c2 e2, t3 = c1 e1 c3 e3 → leaf 0 = a0 b0 c0 e0 (low halves of t0, t2), leaf 1 = a1 b1 c1 e1,
                // leaf 2 = a2 b2 c2 e2 (high halves of t0, t2), leaf 3 = a3 b3 c3 e3.
                Vector256<double> t0 = Avx.UnpackLow(a, b);
                Vector256<double> t1 = Avx.UnpackHigh(a, b);
                Vector256<double> t2 = Avx.UnpackLow(c, e);
                Vector256<double> t3 = Avx.UnpackHigh(c, e);
                Avx.Permute2x128(t0, t2, 0x20).StoreUnsafe(ref d0, i);
                Avx.Permute2x128(t1, t3, 0x20).StoreUnsafe(ref d1, i);
                Avx.Permute2x128(t0, t2, 0x31).StoreUnsafe(ref d2, i);
                Avx.Permute2x128(t1, t3, 0x31).StoreUnsafe(ref d3, i);
                src += step4;
            }

            // The last length % 4 positions: element by element.
            for (; i < n; i++)
            {
                System.Runtime.CompilerServices.Unsafe.Add(ref d0, i) = src[0];
                System.Runtime.CompilerServices.Unsafe.Add(ref d1, i) = src[1];
                System.Runtime.CompilerServices.Unsafe.Add(ref d2, i) = src[2];
                System.Runtime.CompilerServices.Unsafe.Add(ref d3, i) = src[3];
                src += rowStride;
            }
        }

        /// <summary>
        ///     Whether <see cref="TransposeRows{T}"/> should fill a transposed block of T whose leaves hold
        ///     <paramref name="rowLength"/> elements, on this CPU: 4- and 8-byte elements at any length, 16-byte
        ///     elements (complex128, decimal) from <see cref="WideTransposeMinLength"/> positions, all with AVX. The
        ///     width tests are JIT-time constants per T, so for other widths a caller's check folds away.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="rowLength">Leaf length (only 16-byte elements look at it).</param>
        /// <returns>True when <see cref="TransposeRows{T}"/> should fill the block.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool CanTransposeRows<T>(int rowLength) where T : unmanaged
            => Avx.IsSupported && (sizeof(T) == 8 || sizeof(T) == 4 || (sizeof(T) == 16 && rowLength >= WideTransposeMinLength));


        /// <summary>
        ///     Fills the leading leaves of a transposed block (rows that are adjacent source columns, row step 1) with
        ///     AVX register transposes (<see cref="TransposeBand2"/>, <see cref="TransposeBand4"/> or <see cref="TransposeBand8"/>) in <see cref="TransposeTileRows"/> ×
        ///     <see cref="TransposeTileColumns"/> tiles, and returns how many leaves it filled. It fills whole
        ///     sub-bands only (2 leaves for 16-byte elements, 4 for 8-byte, 8 for 4-byte), so fewer than one sub-band of rows is left to the caller's gather. Big
        ///     8-byte leaves are the one transposed case it never sees: <see cref="FillBlock{T}"/> keeps that loop
        ///     inline (see the comment there).
        /// </summary>
        /// <typeparam name="T">Element type; <see cref="CanTransposeRows{T}"/> must be true (callers check).</typeparam>
        /// <param name="p">Address of the block's first element: position i of row r is element <c>r + i·rowStride</c>.</param>
        /// <param name="block">The block. With <paramref name="allocate"/> false, every leaf must already exist.</param>
        /// <param name="rowLength">Leaf length.</param>
        /// <param name="rowStride">Element stride between consecutive positions of a row (any sign).</param>
        /// <param name="allocate">
        ///     True for small leaves: each tile row's leaves (or a tail band's) are allocated, zeroed, right before the
        ///     tiles fill them, so each leaf is written while its lines are still in cache. False for big leaves,
        ///     which <see cref="FillBlock{T}"/> allocated uninitialized in its pass 1.
        /// </param>
        /// <returns>
        ///     The number of leaves filled, counted from row 0 (a multiple of the sub-band width); the caller
        ///     continues with that row, at <c>p + returned</c>.
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     Tiles are walked in row-of-tiles order. A tile of 8-byte elements reads 64 source rows × 32 columns
        ///     (256 lines, 16 KB), which stay in L1 across its 8 sub-bands, so every source line is fully used while
        ///     it is resident; 4-byte tiles read half the bytes. 16-byte tiles take <see cref="TransposeTileColumns"/> / 2 positions, which keeps the 8-byte
        ///     footprint. The 32 leaves of a tile row keep 32
        ///     sequential write streams that continue from one tile to the next.
        ///     </para>
        ///     <para>
        ///     The tiled transposes need no scratch and write each leaf position exactly once. Rows past the last
        ///     whole tile row go through full-length bands of one sub-band each.
        ///     </para>
        ///     <para>
        ///     Never inlined. It runs once per block, so a call costs nothing measurable. Keeping its body out of
        ///     <see cref="FillBlock{T}"/> also keeps FillBlock's own code, including the inline 8-byte loop's layout,
        ///     independent of this method's size.
        ///     </para>
        /// </remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe int TransposeRows<T>(T* p, T[][] block, int rowLength, long rowStride, bool allocate) where T : unmanaged
        {
            int r = 0;
            if (sizeof(T) == 8)
            {
                // Each tile is 8 sub-bands of 4 leaves; position i of leaves r..r+3 is the 32-byte run starting at
                // element r + i·rowStride, and a 4×4 register transpose turns four such runs into four 32-byte stores.
                for (; r + TransposeTileRows <= block.Length; r += TransposeTileRows)
                {
                    if (allocate)
                        AllocateLeaves(block, r, TransposeTileRows, rowLength);

                    for (int j0 = 0; j0 < rowLength; j0 += TransposeTileColumns)
                    {
                        int count = Math.Min(TransposeTileColumns, rowLength - j0);
                        double* tile = (double*)(p + r) + j0 * rowStride;
                        for (int k = 0; k < TransposeTileRows; k += 4)
                        {
                            TransposeBand4(
                                tile + k, rowStride, count,
                                ref LeafAt(block[r + k], j0),
                                ref LeafAt(block[r + k + 1], j0),
                                ref LeafAt(block[r + k + 2], j0),
                                ref LeafAt(block[r + k + 3], j0));
                        }
                    }
                }

                // Fewer than TransposeTileRows rows left: 4-row bands over the full row length.
                for (; r + 4 <= block.Length; r += 4)
                {
                    if (allocate)
                        AllocateLeaves(block, r, 4, rowLength);

                    TransposeBand4(
                        (double*)(p + r), rowStride, rowLength,
                        ref LeafAt(block[r], 0),
                        ref LeafAt(block[r + 1], 0),
                        ref LeafAt(block[r + 2], 0),
                        ref LeafAt(block[r + 3], 0));
                }
            }
            else if (sizeof(T) == 4)
            {
                // 4-byte elements: the same tiles, each sub-band 8 leaves wide. Position i of leaves r..r+7 is the
                // 32-byte run starting at element r + i·rowStride, and an 8×8 register transpose turns eight such runs
                // into eight 32-byte stores, one per leaf.
                for (; r + TransposeTileRows <= block.Length; r += TransposeTileRows)
                {
                    if (allocate)
                        AllocateLeaves(block, r, TransposeTileRows, rowLength);

                    for (int j0 = 0; j0 < rowLength; j0 += TransposeTileColumns)
                    {
                        int count = Math.Min(TransposeTileColumns, rowLength - j0);
                        float* tile = (float*)(p + r) + j0 * rowStride;
                        for (int k = 0; k < TransposeTileRows; k += 8)
                        {
                            TransposeBand8(
                                tile + k, rowStride, count,
                                ref LeafAtSingle(block[r + k], j0),
                                ref LeafAtSingle(block[r + k + 1], j0),
                                ref LeafAtSingle(block[r + k + 2], j0),
                                ref LeafAtSingle(block[r + k + 3], j0),
                                ref LeafAtSingle(block[r + k + 4], j0),
                                ref LeafAtSingle(block[r + k + 5], j0),
                                ref LeafAtSingle(block[r + k + 6], j0),
                                ref LeafAtSingle(block[r + k + 7], j0));
                        }
                    }
                }

                // Fewer than TransposeTileRows rows left: 8-row bands over the full row length.
                for (; r + 8 <= block.Length; r += 8)
                {
                    if (allocate)
                        AllocateLeaves(block, r, 8, rowLength);

                    TransposeBand8(
                        (float*)(p + r), rowStride, rowLength,
                        ref LeafAtSingle(block[r], 0),
                        ref LeafAtSingle(block[r + 1], 0),
                        ref LeafAtSingle(block[r + 2], 0),
                        ref LeafAtSingle(block[r + 3], 0),
                        ref LeafAtSingle(block[r + 4], 0),
                        ref LeafAtSingle(block[r + 5], 0),
                        ref LeafAtSingle(block[r + 6], 0),
                        ref LeafAtSingle(block[r + 7], 0));
                }
            }
            else if (sizeof(T) == 16)
            {
                // 16-byte elements (complex128, decimal): each sub-band is 2 leaves wide. Position i of leaves
                // r, r+1 is the 32-byte run starting at element r + i·rowStride, and a 2×2 transpose of 128-bit
                // halves turns two such runs into two 32-byte stores. Tiles take half as many positions, so a tile
                // moves as many bytes as an 8-byte one. Reached only for leaves of WideTransposeMinLength positions or
                // more (CanTransposeRows): shorter 16-byte leaves are faster on the gather.
                for (; r + TransposeTileRows <= block.Length; r += TransposeTileRows)
                {
                    if (allocate)
                        AllocateLeaves(block, r, TransposeTileRows, rowLength);

                    for (int j0 = 0; j0 < rowLength; j0 += TransposeTileColumns / 2)
                    {
                        int count = Math.Min(TransposeTileColumns / 2, rowLength - j0);
                        double* tile = (double*)(p + r + j0 * rowStride);
                        for (int k = 0; k < TransposeTileRows; k += 2)
                        {
                            TransposeBand2(
                                tile + 2 * k, rowStride, count,
                                ref LeafAtPair(block[r + k], j0),
                                ref LeafAtPair(block[r + k + 1], j0));
                        }
                    }
                }

                // Fewer than TransposeTileRows rows left: 2-row bands over the full row length.
                for (; r + 2 <= block.Length; r += 2)
                {
                    if (allocate)
                        AllocateLeaves(block, r, 2, rowLength);

                    TransposeBand2(
                        (double*)(p + r), rowStride, rowLength,
                        ref LeafAtPair(block[r], 0),
                        ref LeafAtPair(block[r + 1], 0));
                }
            }

            return r;
        }

        /// <summary>
        ///     Allocates leaves <paramref name="first"/> .. <c>first + count − 1</c> of a block as zeroed arrays. Used
        ///     only for small leaves, whose zeroing is folded into the pre-cleared allocation context.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="block">The block (the slots are overwritten).</param>
        /// <param name="first">First row to allocate.</param>
        /// <param name="count">Number of rows (callers keep <c>first + count</c> within the block).</param>
        /// <param name="rowLength">Leaf length.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AllocateLeaves<T>(T[][] block, int first, int count, int rowLength) where T : unmanaged
        {
            for (int k = first; k < first + count; k++)
                block[k] = new T[rowLength];
        }

        /// <summary>
        ///     A byref to position <paramref name="position"/> of a 4-byte-element leaf, viewed as <see cref="float"/>
        ///     lanes for <see cref="TransposeBand8"/>. Pure reinterpretation: no copy, no pinning.
        /// </summary>
        /// <typeparam name="T">The leaf's element type (4 bytes wide; callers check).</typeparam>
        /// <param name="leaf">The leaf.</param>
        /// <param name="position">Element index inside the leaf (in range; callers guarantee it).</param>
        /// <returns>A GC-tracked reference to that element.</returns>
        private static ref float LeafAtSingle<T>(T[] leaf, int position) where T : unmanaged
            => ref System.Runtime.CompilerServices.Unsafe.Add(
                ref System.Runtime.CompilerServices.Unsafe.As<T, float>(ref MemoryMarshal.GetArrayDataReference(leaf)), position);

        /// <summary>
        ///     Fills positions <c>0 .. length − 1</c> of eight leaves from a transposed source of 4-byte elements with
        ///     AVX 8×8 register transposes: position i of leaf k (k = 0..7) is <c>src[k + i·rowStride]</c>. Eight
        ///     32-byte loads (positions i..i+7, each holding the eight leaves' values) become eight 32-byte stores.
        /// </summary>
        /// <param name="src">Address of position 0 of leaf 0 (the eight leaves' values at a position are adjacent).</param>
        /// <param name="rowStride">Element stride between consecutive positions (any sign).</param>
        /// <param name="length">Number of positions to fill in each leaf.</param>
        /// <param name="d0">Position 0 of leaf 0 (a GC-tracked reference).</param>
        /// <param name="d1">Position 0 of leaf 1.</param>
        /// <param name="d2">Position 0 of leaf 2.</param>
        /// <param name="d3">Position 0 of leaf 3.</param>
        /// <param name="d4">Position 0 of leaf 4.</param>
        /// <param name="d5">Position 0 of leaf 5.</param>
        /// <param name="d6">Position 0 of leaf 6.</param>
        /// <param name="d7">Position 0 of leaf 7.</param>
        /// <remarks>
        ///     The classic three-stage 8×8 transpose: <c>vunpcklps</c>/<c>vunpckhps</c> pair up positions,
        ///     <c>vshufps</c> 0x44/0xEE gather four positions of one leaf per 128-bit lane, and <c>vperm2f128</c>
        ///     joins the two lanes. Only bits move, so float NaN payloads, signalling NaNs and int32/uint32 values are
        ///     copied exactly. Callers check <see cref="Avx.IsSupported"/> and a 4-byte element size.
        /// </remarks>
        private static unsafe void TransposeBand8(float* src, long rowStride, int length, ref float d0, ref float d1, ref float d2, ref float d3, ref float d4, ref float d5, ref float d6, ref float d7)
        {
            nuint n = (nuint)length;
            nuint i = 0;
            long step8 = 8 * rowStride;
            for (; i + 8 <= n; i += 8)
            {
                // r0..r7 hold positions i..i+7; lane k of each is leaf k's value there.
                Vector256<float> r0 = Avx.LoadVector256(src);
                Vector256<float> r1 = Avx.LoadVector256(src + rowStride);
                Vector256<float> r2 = Avx.LoadVector256(src + 2 * rowStride);
                Vector256<float> r3 = Avx.LoadVector256(src + 3 * rowStride);
                Vector256<float> r4 = Avx.LoadVector256(src + 4 * rowStride);
                Vector256<float> r5 = Avx.LoadVector256(src + 5 * rowStride);
                Vector256<float> r6 = Avx.LoadVector256(src + 6 * rowStride);
                Vector256<float> r7 = Avx.LoadVector256(src + 7 * rowStride);

                // Stage 1: pairs of positions per leaf pair within each 128-bit lane.
                Vector256<float> t0 = Avx.UnpackLow(r0, r1);
                Vector256<float> t1 = Avx.UnpackHigh(r0, r1);
                Vector256<float> t2 = Avx.UnpackLow(r2, r3);
                Vector256<float> t3 = Avx.UnpackHigh(r2, r3);
                Vector256<float> t4 = Avx.UnpackLow(r4, r5);
                Vector256<float> t5 = Avx.UnpackHigh(r4, r5);
                Vector256<float> t6 = Avx.UnpackLow(r6, r7);
                Vector256<float> t7 = Avx.UnpackHigh(r6, r7);

                // Stage 2: four positions of one leaf per lane (leaves k | k+4 for positions 0..3 or 4..7).
                Vector256<float> u0 = Avx.Shuffle(t0, t2, 0x44);
                Vector256<float> u1 = Avx.Shuffle(t0, t2, 0xEE);
                Vector256<float> u2 = Avx.Shuffle(t1, t3, 0x44);
                Vector256<float> u3 = Avx.Shuffle(t1, t3, 0xEE);
                Vector256<float> u4 = Avx.Shuffle(t4, t6, 0x44);
                Vector256<float> u5 = Avx.Shuffle(t4, t6, 0xEE);
                Vector256<float> u6 = Avx.Shuffle(t5, t7, 0x44);
                Vector256<float> u7 = Avx.Shuffle(t5, t7, 0xEE);

                // Stage 3: join lanes: low halves are leaves 0..3, high halves leaves 4..7.
                Avx.Permute2x128(u0, u4, 0x20).StoreUnsafe(ref d0, i);
                Avx.Permute2x128(u1, u5, 0x20).StoreUnsafe(ref d1, i);
                Avx.Permute2x128(u2, u6, 0x20).StoreUnsafe(ref d2, i);
                Avx.Permute2x128(u3, u7, 0x20).StoreUnsafe(ref d3, i);
                Avx.Permute2x128(u0, u4, 0x31).StoreUnsafe(ref d4, i);
                Avx.Permute2x128(u1, u5, 0x31).StoreUnsafe(ref d5, i);
                Avx.Permute2x128(u2, u6, 0x31).StoreUnsafe(ref d6, i);
                Avx.Permute2x128(u3, u7, 0x31).StoreUnsafe(ref d7, i);
                src += step8;
            }

            // The last length % 8 positions: element by element.
            for (; i < n; i++)
            {
                System.Runtime.CompilerServices.Unsafe.Add(ref d0, i) = src[0];
                System.Runtime.CompilerServices.Unsafe.Add(ref d1, i) = src[1];
                System.Runtime.CompilerServices.Unsafe.Add(ref d2, i) = src[2];
                System.Runtime.CompilerServices.Unsafe.Add(ref d3, i) = src[3];
                System.Runtime.CompilerServices.Unsafe.Add(ref d4, i) = src[4];
                System.Runtime.CompilerServices.Unsafe.Add(ref d5, i) = src[5];
                System.Runtime.CompilerServices.Unsafe.Add(ref d6, i) = src[6];
                System.Runtime.CompilerServices.Unsafe.Add(ref d7, i) = src[7];
                src += rowStride;
            }
        }

        /// <summary>
        ///     A byref to position <paramref name="position"/> of a 16-byte-element leaf, viewed as <see cref="double"/>
        ///     lanes (two per element) for <see cref="TransposeBand2"/>. Pure reinterpretation: no copy, no pinning.
        /// </summary>
        /// <typeparam name="T">The leaf's element type (16 bytes wide; callers check).</typeparam>
        /// <param name="leaf">The leaf.</param>
        /// <param name="position">Element index inside the leaf (in range; callers guarantee it).</param>
        /// <returns>A GC-tracked reference to that element's first 8 bytes.</returns>
        private static ref double LeafAtPair<T>(T[] leaf, int position) where T : unmanaged
            => ref System.Runtime.CompilerServices.Unsafe.Add(
                ref System.Runtime.CompilerServices.Unsafe.As<T, double>(ref MemoryMarshal.GetArrayDataReference(leaf)), (nint)position * 2);

        /// <summary>
        ///     Fills positions <c>0 .. length − 1</c> of two leaves from a transposed source of 16-byte elements
        ///     (complex128, decimal): position i of leaf k (k = 0, 1) is source element <c>k + i·rowStride</c>. Two
        ///     32-byte loads (positions i and i+1, each holding both leaves' elements) become two 32-byte stores
        ///     (positions i, i+1 of one leaf) through one 128-bit-half swap each.
        /// </summary>
        /// <param name="src">Address of position 0 of leaf 0, viewed as doubles (an element is two lanes).</param>
        /// <param name="rowStride">Stride between consecutive positions, in 16-byte elements (any sign).</param>
        /// <param name="length">Number of positions to fill in each leaf.</param>
        /// <param name="d0">Position 0 of leaf 0, viewed as doubles (a GC-tracked reference).</param>
        /// <param name="d1">Position 0 of leaf 1.</param>
        /// <remarks>
        ///     <c>vperm2f128</c> only moves 128-bit halves, and each element is exactly one half, so any 16-byte
        ///     pattern (a complex's two doubles incl. NaN payloads, a decimal's four ints) is copied exactly; the odd
        ///     last position moves as 8-byte halves, which are bit copies too. Callers check
        ///     <see cref="Avx.IsSupported"/> and a 16-byte element size.
        /// </remarks>
        private static unsafe void TransposeBand2(double* src, long rowStride, int length, ref double d0, ref double d1)
        {
            nuint n = (nuint)length;
            nuint i = 0;

            // Doubles between consecutive positions (an element is two doubles).
            long step = 2 * rowStride;
            for (; i + 2 <= n; i += 2)
            {
                // a = position i (leaf 0 | leaf 1), b = position i + 1 (leaf 0 | leaf 1).
                Vector256<double> a = Avx.LoadVector256(src);
                Vector256<double> b = Avx.LoadVector256(src + step);
                Avx.Permute2x128(a, b, 0x20).StoreUnsafe(ref d0, 2 * i);
                Avx.Permute2x128(a, b, 0x31).StoreUnsafe(ref d1, 2 * i);
                src += 2 * step;
            }

            // An odd length's last position: each leaf's element as two 8-byte halves.
            if (i < n)
            {
                System.Runtime.CompilerServices.Unsafe.Add(ref d0, 2 * i) = src[0];
                System.Runtime.CompilerServices.Unsafe.Add(ref d0, 2 * i + 1) = src[1];
                System.Runtime.CompilerServices.Unsafe.Add(ref d1, 2 * i) = src[2];
                System.Runtime.CompilerServices.Unsafe.Add(ref d1, 2 * i + 1) = src[3];
            }
        }

        /// <summary>
        ///     Fills one leaf from a source row that is not unit-stride, picking the cheapest exact copy for the stride:
        ///     a broadcast row (stride 0) is one fill, a reversed row (stride −1) a vectorized reverse copy, and any
        ///     other stride an element-by-element gather.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="p">
        ///     Address of the row's first element (position 0 of the leaf). For an EMPTY row it need not address a valid
        ///     element — an empty view's offset may point one past its buffer (an <c>np.split</c> child at the end) or
        ///     anywhere at all (an empty slice keeps its parent's offset over a fresh zero-length buffer) — so nothing is
        ///     read when <paramref name="n"/> is 0.
        /// </param>
        /// <param name="stride">Element stride along the row (never 1; callers block-copy those rows).</param>
        /// <param name="d">Position 0 of the leaf (a GC-tracked reference, so the leaf needs no pinning).</param>
        /// <param name="n">Row length.</param>
        /// <remarks>
        ///     Inlined into its three callers (the 1-D leaf, big-leaf blocks, small-leaf blocks), so a plain strided
        ///     row still runs the same tight gather loop as before, with no call per row.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void GatherRow<T>(T* p, long stride, ref T d, int n) where T : unmanaged
        {
            if (stride == 0)
            {
                // A broadcast row holds one value n times. The value is read only when there is a position to fill:
                // an empty row's pointer is not an element (an empty slice reports stride 0, and its offset may lie far
                // outside the zero-length buffer it owns — reading it access-violated).
                if (n != 0)
                    MemoryMarshal.CreateSpan(ref d, n).Fill(*p);
                return;
            }

            if (stride == -1)
            {
                ReverseCopy(p, ref d, n);
                return;
            }

            // Running source pointer and a ref into the leaf: no bounds checks, no index multiply.
            for (int i = 0; i < n; i++)
            {
                System.Runtime.CompilerServices.Unsafe.Add(ref d, i) = *p;
                p += stride;
            }
        }

        /// <summary>
        ///     Copies a reversed row: position i of the leaf receives <c>p[−i]</c>, for i = 0 .. n − 1. The source row
        ///     occupies the contiguous range <c>p[−(n − 1)] .. p[0]</c>, walked downwards.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="p">Address of the row's first element (its HIGHEST address).</param>
        /// <param name="d">Position 0 of the leaf.</param>
        /// <param name="n">Row length.</param>
        /// <remarks>
        ///     <para>
        ///     With AVX2, 8- and 4-byte elements take one fused pass: a 32-byte load ending at the current source
        ///     position, a cross-lane permute that reverses the lanes (<c>vpermpd 0x1B</c> / <c>vpermps</c> with
        ///     indices 7..0) and a 32-byte store. That is as many loads and stores as a straight copy, where the
        ///     element gather issued one scalar load and store per element. The permutes only move bits, so NaN
        ///     payloads, signalling NaNs and integer values are copied exactly.
        ///     </para>
        ///     <para>
        ///     Every other case (other widths, or no AVX2) takes two passes: one block copy of the contiguous range,
        ///     then <see cref="MemoryExtensions.Reverse{T}(Span{T})"/> in place, which the BCL vectorizes for 1-, 2-,
        ///     4- and 8-byte elements and swaps pairwise for wider ones. The second pass runs over a leaf that is
        ///     already in cache.
        ///     </para>
        /// </remarks>
        private static unsafe void ReverseCopy<T>(T* p, ref T d, int n) where T : unmanaged
        {
            int i = 0;
            if (sizeof(T) == 8 && Avx2.IsSupported)
            {
                ref double dd = ref System.Runtime.CompilerServices.Unsafe.As<T, double>(ref d);
                double* s = (double*)p - 3;
                for (; i + 4 <= n; i += 4)
                {
                    // Source p[−i−3 .. −i] ascending → leaf positions i .. i+3 need it descending.
                    Avx2.Permute4x64(Avx.LoadVector256(s), 0x1B).StoreUnsafe(ref dd, (nuint)i);
                    s -= 4;
                }
            }
            else if (sizeof(T) == 4 && Avx2.IsSupported)
            {
                ref float df = ref System.Runtime.CompilerServices.Unsafe.As<T, float>(ref d);
                float* s = (float*)p - 7;
                Vector256<int> descending = Vector256.Create(7, 6, 5, 4, 3, 2, 1, 0);
                for (; i + 8 <= n; i += 8)
                {
                    Avx2.PermuteVar8x32(Avx.LoadVector256(s), descending).StoreUnsafe(ref df, (nuint)i);
                    s -= 8;
                }
            }
            else
            {
                Span<T> leaf = MemoryMarshal.CreateSpan(ref d, n);
                new ReadOnlySpan<T>(p - (n - 1), n).CopyTo(leaf);
                leaf.Reverse();
                return;
            }

            // The last n % 4 (8-byte) or n % 8 (4-byte) positions.
            for (; i < n; i++)
                System.Runtime.CompilerServices.Unsafe.Add(ref d, i) = p[-i];
        }

        /// <summary>
        ///     Signature of an emitted builder for the levels above the <c>T[][]</c> blocks of one rank (T = dtype).
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="p">Address of the logical element 0.</param>
        /// <param name="dims">Source dimensions (every entry already checked to fit an int).</param>
        /// <param name="strides">Source element strides.</param>
        /// <returns>The level-0 array.</returns>
        private unsafe delegate Array BlockLevelsBuilder<T>(T* p, long[] dims, long[] strides) where T : unmanaged;

        /// <summary>
        ///     Signature of an emitted builder for the levels above the converted <c>T[][]</c> blocks of one rank.
        /// </summary>
        /// <typeparam name="T">Element type of the result.</typeparam>
        /// <param name="p">Address of the logical source element 0.</param>
        /// <param name="dims">Source dimensions (every entry already checked to fit an int).</param>
        /// <param name="strides">Source element strides.</param>
        /// <param name="itemSize">Source element size in bytes (scales the strides into byte steps).</param>
        /// <param name="converter">The call's converter.</param>
        /// <returns>The level-0 array.</returns>
        private unsafe delegate Array ConvertedBuilder<T>(byte* p, long[] dims, long[] strides, long itemSize, RowConverter<T> converter) where T : unmanaged;

        /// <summary>
        ///     Compiled exact-type builders per rank for one element type. The entries are immutable compiled code,
        ///     a pure function of (T, rank): no per-call data is ever stored, so the cache is thread-safe and a
        ///     duplicate compile under a race is harmless.
        /// </summary>
        /// <typeparam name="T">Element type of the results the builders produce.</typeparam>
        private static class BlockLevelsBuilders<T> where T : unmanaged
        {
            /// <summary>Builder per source rank (≥ 3).</summary>
            internal static readonly ConcurrentDictionary<int, BlockLevelsBuilder<T>> ByRank = new ConcurrentDictionary<int, BlockLevelsBuilder<T>>();
        }

        /// <summary>
        ///     Compiled converting builders per rank for one result element type (the source dtype is a run-time
        ///     argument: its item size and converter are passed in, so one builder serves every source dtype).
        ///     Same thread-safety argument as <see cref="BlockLevelsBuilders{T}"/>.
        /// </summary>
        /// <typeparam name="T">Element type of the results the builders produce.</typeparam>
        private static class ConvertedBuilders<T> where T : unmanaged
        {
            /// <summary>Builder per source rank (≥ 3).</summary>
            internal static readonly ConcurrentDictionary<int, ConvertedBuilder<T>> ByRank = new ConcurrentDictionary<int, ConvertedBuilder<T>>();
        }

        /// <summary>Element types of the result's levels: level d holds T nested (nd − 1 − d) levels.</summary>
        /// <typeparam name="T">Leaf element type.</typeparam>
        /// <param name="nd">Rank.</param>
        /// <returns>The per-level element types, outermost level first.</returns>
        private static Type[] LevelElementTypes<T>(int nd)
        {
            var elementTypes = new Type[nd];
            Type t = typeof(T);
            for (int d = nd - 1; d >= 0; d--)
            {
                elementTypes[d] = t;
                t = t.MakeArrayType();
            }

            return elementTypes;
        }

        /// <summary>
        ///     Emits the builder for one rank: nested counted loops over the block levels 0 .. ndim − 3. Each has its
        ///     own source pointer, advanced by the level's byte stride. The innermost level calls
        ///     <see cref="FillBlock{T}"/> (exact) or <see cref="FillBlockConverted{T}"/> (converted) for every
        ///     position.
        /// </summary>
        /// <typeparam name="T">Element type of the result.</typeparam>
        /// <param name="nd">Source rank (≥ 3).</param>
        /// <param name="converted">
        ///     False for the exact builder (<see cref="BlockLevelsBuilder{T}"/>: <c>T*</c> source, byte steps
        ///     scaled by <c>sizeof(T)</c>). True for the converting one (<see cref="ConvertedBuilder{T}"/>:
        ///     <c>byte*</c> source, byte steps scaled by the run-time item size, the converter passed through).
        /// </param>
        /// <returns>The compiled builder, of the delegate type selected by <paramref name="converted"/>.</returns>
        /// <exception cref="InvalidOperationException">The fill method cannot be found (a refactoring broke the
        ///     reflection lookup: a programming error, surfaced at the first rank-≥3 call).</exception>
        /// <remarks>
        ///     Zero-length axes need no special case: a level of length 0 is an empty <c>newarr</c> whose loop
        ///     runs zero times, which is exactly the required structure (the source is never read).
        /// </remarks>
        private static unsafe Delegate EmitBlockLevelsBuilder<T>(int nd, bool converted) where T : unmanaged
        {
            int levels = nd - 2;
            Type[] elementTypes = LevelElementTypes<T>(nd);
            Type pointerType = converted ? typeof(byte*) : typeof(T).MakePointerType();
            Type[] parameters = converted
                ? new[] { pointerType, typeof(long[]), typeof(long[]), typeof(long), typeof(RowConverter<T>) }
                : new[] { pointerType, typeof(long[]), typeof(long[]) };

            // Owner NDArray + skipVisibility: the body calls the private fill methods.
            var method = new DynamicMethod(
                converted ? $"ToJaggedArrayConverted_{typeof(T).Name}_{nd}D" : $"ToJaggedArray_{typeof(T).Name}_{nd}D",
                typeof(Array),
                parameters,
                typeof(NDArray),
                skipVisibility: true);
            ILGenerator il = method.GetILGenerator();
            MethodInfo fill = typeof(NDArray)
                .GetMethod(converted ? nameof(FillBlockConverted) : nameof(FillBlock), BindingFlags.NonPublic | BindingFlags.Static)
                ?.MakeGenericMethod(typeof(T))
                ?? throw new InvalidOperationException("The jagged block filler was not found for the emitted builder.");

            var locals = new BlockLevelLocals(levels);
            for (int d = 0; d < levels; d++)
            {
                locals.Count[d] = il.DeclareLocal(typeof(int));
                locals.Step[d] = il.DeclareLocal(typeof(IntPtr));
                locals.Pointer[d] = il.DeclareLocal(pointerType);
                locals.Index[d] = il.DeclareLocal(typeof(int));
                locals.Array[d] = il.DeclareLocal(elementTypes[d].MakeArrayType());

                // Hoisted once per call: count[d] = (int)dims[d]; step[d] = (nint)(strides[d] * itemSize).
                EmitLoadLong(il, OpCodes.Ldarg_1, d, true, locals.Count[d]);
                EmitLoadByteStep<T>(il, d, converted, locals.Step[d]);
            }

            // The block shape, the same for every block: rows = dims[nd-2], rowStep = strides[nd-2] (elements for
            // the exact filler, bytes for the converting one), rowLength = dims[nd-1], rowStride = strides[nd-1].
            locals.Rows = il.DeclareLocal(typeof(int));
            locals.RowStep = il.DeclareLocal(typeof(long));
            locals.RowLength = il.DeclareLocal(typeof(int));
            locals.RowStride = il.DeclareLocal(typeof(long));
            EmitLoadLong(il, OpCodes.Ldarg_1, nd - 2, true, locals.Rows);
            if (converted)
            {
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldc_I4, nd - 2);
                il.Emit(OpCodes.Ldelem_I8);
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Stloc, locals.RowStep);
            }
            else
            {
                EmitLoadLong(il, OpCodes.Ldarg_2, nd - 2, false, locals.RowStep);
            }

            EmitLoadLong(il, OpCodes.Ldarg_1, nd - 1, true, locals.RowLength);
            EmitLoadLong(il, OpCodes.Ldarg_2, nd - 1, false, locals.RowStride);

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, locals.Pointer[0]);
            EmitBlockLevel(il, 0, levels, elementTypes, locals, fill, converted);
            il.Emit(OpCodes.Ldloc, locals.Array[0]);
            il.Emit(OpCodes.Ret);

            return converted
                ? method.CreateDelegate(typeof(ConvertedBuilder<T>))
                : method.CreateDelegate(typeof(BlockLevelsBuilder<T>));
        }

        /// <summary>Emits <c>local = narrow ? (int)array[index] : array[index]</c> for a <c>long[]</c> argument.</summary>
        /// <param name="il">Target IL stream.</param>
        /// <param name="loadArray">The <c>ldarg</c> opcode that pushes the array.</param>
        /// <param name="index">Element index.</param>
        /// <param name="narrow">True to convert the element to int (a dimension already checked to fit).</param>
        /// <param name="local">Destination local.</param>
        private static void EmitLoadLong(ILGenerator il, OpCode loadArray, int index, bool narrow, LocalBuilder local)
        {
            il.Emit(loadArray);
            il.Emit(OpCodes.Ldc_I4, index);
            il.Emit(OpCodes.Ldelem_I8);
            if (narrow)
                il.Emit(OpCodes.Conv_I4);
            il.Emit(OpCodes.Stloc, local);
        }

        /// <summary>
        ///     Emits <c>local = (nint)(strides[index] * itemSize)</c>: an element stride as a byte step. The item size
        ///     is the constant <c>sizeof(T)</c> for the exact builder and the <c>itemSize</c> argument for the
        ///     converting one.
        /// </summary>
        /// <typeparam name="T">Element type (the exact builder's scale).</typeparam>
        /// <param name="il">Target IL stream.</param>
        /// <param name="index">Axis whose stride is loaded.</param>
        /// <param name="converted">True to scale by the run-time item size argument.</param>
        /// <param name="local">Destination native-int local.</param>
        private static unsafe void EmitLoadByteStep<T>(ILGenerator il, int index, bool converted, LocalBuilder local) where T : unmanaged
        {
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I4, index);
            il.Emit(OpCodes.Ldelem_I8);
            if (converted)
                il.Emit(OpCodes.Ldarg_3);
            else
                il.Emit(OpCodes.Ldc_I8, (long)sizeof(T));
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Stloc, local);
        }

        /// <summary>The IL locals of one emitted builder: one slot per block level plus the block shape.</summary>
        private sealed class BlockLevelLocals
        {
            /// <summary>Allocates the per-level slot arrays.</summary>
            /// <param name="levels">Number of block levels.</param>
            public BlockLevelLocals(int levels)
            {
                Count = new LocalBuilder[levels];
                Step = new LocalBuilder[levels];
                Pointer = new LocalBuilder[levels];
                Index = new LocalBuilder[levels];
                Array = new LocalBuilder[levels];
            }

            /// <summary>Level length (int).</summary>
            public LocalBuilder[] Count { get; }

            /// <summary>Level byte stride (native int).</summary>
            public LocalBuilder[] Step { get; }

            /// <summary>Source position of the level's current element.</summary>
            public LocalBuilder[] Pointer { get; }

            /// <summary>Loop index (int).</summary>
            public LocalBuilder[] Index { get; }

            /// <summary>The level's array, typed exactly.</summary>
            public LocalBuilder[] Array { get; }

            /// <summary>Rows per block (int).</summary>
            public LocalBuilder Rows { get; set; }

            /// <summary>Stride between rows (long): elements for the exact filler, bytes for the converting one.</summary>
            public LocalBuilder RowStep { get; set; }

            /// <summary>Row length (int).</summary>
            public LocalBuilder RowLength { get; set; }

            /// <summary>Element stride along a row (long).</summary>
            public LocalBuilder RowStride { get; set; }
        }

        /// <summary>
        ///     Emits one block level: allocate the level's array, loop over its positions (recursing into the next
        ///     level, or calling the block filler at the innermost one), store each child, and step the level pointer.
        /// </summary>
        /// <param name="il">Target IL stream.</param>
        /// <param name="d">Level being emitted.</param>
        /// <param name="levels">Number of block levels (ndim − 2).</param>
        /// <param name="elementTypes">Per-level element types.</param>
        /// <param name="locals">The builder's locals.</param>
        /// <param name="fill">The closed block filler (<see cref="FillBlock{T}"/> or <see cref="FillBlockConverted{T}"/>).</param>
        /// <param name="converted">True to pass the converter argument (<c>ldarg.s 4</c>) to the filler.</param>
        private static void EmitBlockLevel(ILGenerator il, int d, int levels, Type[] elementTypes, BlockLevelLocals locals, MethodInfo fill, bool converted)
        {
            Label body = il.DefineLabel();
            Label check = il.DefineLabel();

            il.Emit(OpCodes.Ldloc, locals.Count[d]);
            il.Emit(OpCodes.Newarr, elementTypes[d]);
            il.Emit(OpCodes.Stloc, locals.Array[d]);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locals.Index[d]);
            il.Emit(OpCodes.Br, check);

            il.MarkLabel(body);
            if (d + 1 < levels)
            {
                // The child level starts at this level's current position.
                il.Emit(OpCodes.Ldloc, locals.Pointer[d]);
                il.Emit(OpCodes.Stloc, locals.Pointer[d + 1]);
                EmitBlockLevel(il, d + 1, levels, elementTypes, locals, fill, converted);
                il.Emit(OpCodes.Ldloc, locals.Array[d]);
                il.Emit(OpCodes.Ldloc, locals.Index[d]);
                il.Emit(OpCodes.Ldloc, locals.Array[d + 1]);
                il.Emit(OpCodes.Stelem, elementTypes[d]);
            }
            else
            {
                il.Emit(OpCodes.Ldloc, locals.Array[d]);
                il.Emit(OpCodes.Ldloc, locals.Index[d]);
                il.Emit(OpCodes.Ldloc, locals.Pointer[d]);
                il.Emit(OpCodes.Ldloc, locals.Rows);
                il.Emit(OpCodes.Ldloc, locals.RowStep);
                il.Emit(OpCodes.Ldloc, locals.RowLength);
                il.Emit(OpCodes.Ldloc, locals.RowStride);
                if (converted)
                    il.Emit(OpCodes.Ldarg_S, (byte)4);
                il.Emit(OpCodes.Call, fill);
                il.Emit(OpCodes.Stelem, elementTypes[d]);
            }

            il.Emit(OpCodes.Ldloc, locals.Pointer[d]);
            il.Emit(OpCodes.Ldloc, locals.Step[d]);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locals.Pointer[d]);
            il.Emit(OpCodes.Ldloc, locals.Index[d]);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locals.Index[d]);

            il.MarkLabel(check);
            il.Emit(OpCodes.Ldloc, locals.Index[d]);
            il.Emit(OpCodes.Ldloc, locals.Count[d]);
            il.Emit(OpCodes.Blt, body);
        }

        /// <summary>
        ///     No-dynamic-code fallback: builds the jagged sub-array for one level of a recursive descent,
        ///     reading from the source at <paramref name="p"/>.
        /// </summary>
        /// <typeparam name="T">Element type (already the source's dtype).</typeparam>
        /// <param name="p">Address of the first element of this sub-array in the source.</param>
        /// <param name="dims">Source dimensions (validated to fit in <see cref="int"/>).</param>
        /// <param name="strides">Source strides in ELEMENTS (may be 0 for broadcast axes or negative for reversed ones).</param>
        /// <param name="elementTypes">Per-level element type of the result arrays.</param>
        /// <param name="level">The dimension this call materializes (≤ ndim − 2).</param>
        /// <returns>A typed <c>T[][]</c> block at level ndim − 2, else an array of the next level's arrays.</returns>
        private static unsafe Array JaggedLevel<T>(T* p, long[] dims, long[] strides, Type[] elementTypes, int level) where T : unmanaged
        {
            int n = (int)dims[level];
            long stride = strides[level];
            if (level == dims.Length - 2)
                return FillBlock(p, n, stride, (int)dims[level + 1], strides[level + 1]);

            // An array of arrays has reference-type elements, so it can be filled through its object[] view
            // (array covariance); System.Array is spelled out because inside NDArray the bare name `Array`
            // binds to the NDArray.Array property in expression context.
            var children = (object[])System.Array.CreateInstance(elementTypes[level], n);
            for (int i = 0; i < n; i++)
                children[i] = JaggedLevel(p + i * stride, dims, strides, elementTypes, level + 1);
            return children;
        }

        /// <summary>
        ///     No-dynamic-code fallback for the converted path: the same recursive descent over a byte pointer.
        /// </summary>
        /// <typeparam name="T">Element type of the result.</typeparam>
        /// <param name="p">Address of the first source element of this sub-array.</param>
        /// <param name="dims">Source dimensions (validated to fit in <see cref="int"/>).</param>
        /// <param name="strides">Source strides in ELEMENTS.</param>
        /// <param name="itemSize">Source element size in bytes.</param>
        /// <param name="elementTypes">Per-level element type of the result arrays.</param>
        /// <param name="level">The dimension this call materializes (≤ ndim − 2).</param>
        /// <param name="converter">The call's converter.</param>
        /// <returns>A typed <c>T[][]</c> block at level ndim − 2, else an array of the next level's arrays.</returns>
        private static unsafe Array JaggedLevelConverted<T>(byte* p, long[] dims, long[] strides, int itemSize, Type[] elementTypes, int level, RowConverter<T> converter) where T : unmanaged
        {
            int n = (int)dims[level];
            long stepBytes = strides[level] * itemSize;
            if (level == dims.Length - 2)
                return FillBlockConverted(p, n, stepBytes, (int)dims[level + 1], strides[level + 1], converter);

            var children = (object[])System.Array.CreateInstance(elementTypes[level], n);
            for (int i = 0; i < n; i++)
                children[i] = JaggedLevelConverted(p + i * stepBytes, dims, strides, itemSize, elementTypes, level + 1, converter);
            return children;
        }
    }
}
