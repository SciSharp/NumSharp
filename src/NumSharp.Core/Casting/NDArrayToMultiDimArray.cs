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
        /// <summary>
        ///     Copies this array into a new single-dimensional .NET array of element type <typeparamref name="T"/>, in
        ///     logical C (row-major) order for any memory layout and any rank. A 0-d array yields a one-element array.
        /// </summary>
        /// <typeparam name="T">
        ///     Element type of the result. When it differs from <see cref="dtype"/>, the values are converted exactly as
        ///     <see cref="astype(DType, bool)"/> converts them; it must be one of NumSharp's 15 dtypes.
        /// </typeparam>
        /// <returns>
        ///     A freshly allocated array of <see cref="size"/> elements that shares no memory with this NDArray (an empty
        ///     result is the shared, immutable <see cref="System.Array.Empty{T}"/> instance).
        /// </returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not a NumSharp dtype.</exception>
        /// <exception cref="InvalidOperationException">The array holds more than <see cref="int.MaxValue"/> elements, which no .NET array can index.</exception>
        /// <remarks>
        ///     <para>
        ///     One pass, straight into the result: <see cref="flat"/> + a block copy is the two-pass form of this method
        ///     (materialize a C-order image, then copy it); here the pinned result itself is the destination and no
        ///     intermediate buffer exists. A C-contiguous source (any offset) is the block copy alone.
        ///     </para>
        ///     <para>
        ///     Dedicated paths for everything the plain copy does not cover. Exact copies: one element (a direct copy), a
        ///     scalar broadcast (one fill), and every other non-contiguous layout through <see cref="ToArrayDedicated{T}"/>,
        ///     a layout-specific fill engine that replaced the iterator copy — per layout class it measured faster than the
        ///     iterator's generic walk (transposes by register blocks or 64-column bands — a large float64-width one
        ///     streamed into the result by whole 64-byte lines where its row pitch allows it —, strided rows by AVX2
        ///     deinterleave / gathers, reversed rows by a vector reverse, permuted views transposed over their unit-stride
        ///     axis; a reversed run of at least a mebibyte streamed into the result by whole 64-byte lines on non-temporal
        ///     stores, assembled in registers — or, for a plane of rows shorter than 16 KB, packed through an L1-resident
        ///     scratch; a rank-2 view is canonicalized in registers, <see cref="ToArrayPlane2D{T}"/>). Conversions: a
        ///     contiguous source of one of 24 common pairs through this file's AVX2 converters
        ///     (<see cref="ToArrayConvertVectors{T}"/>), any other through the generator's cast kernel for
        ///     the pair in one call (or the bit-exact decimal → double vector kernel; a result of at least a mebibyte of
        ///     one of seven common pairs converted in registers straight into whole 64-byte lines on non-temporal
        ///     stores, <see cref="ToArrayConvertLines{T}"/>); a pair without a kernel, or any
        ///     other layout, up to
        ///     <see cref="ToArrayGatherMaxElements"/> elements through a scalar odometer converting element by element with
        ///     astype's scalar converter; larger ones keep astype's copy core (the iterator's cast copy).
        ///     </para>
        ///     <para>
        ///     A large-object result (from <see cref="ToArrayLargeObjectBytes"/>) is fresh from the OS and faults in on
        ///     first touch. A fill that writes the C-order result out of address order (a transpose) would take those
        ///     faults inside its walk, each evicting the walk's cache working set; such results have their pages faulted
        ///     in first, in address order, by a tight pre-pass (<see cref="ToArrayTouchPages"/>) — measured on a 1000×1000
        ///     float64 transpose: ≈ 1.9 ms → ≈ 1.3 ms. A sequential fill skips it (there it measured +8 %).
        ///     </para>
        ///     <para>
        ///     Four frames, each shaped for the fixed cost of the calls it serves. This one holds only the hot case — a
        ///     C-contiguous source of the same dtype: one allocation and one block copy (a result of at most 64 bytes
        ///     copied inline, <see cref="ToArrayCopySmall"/>) — behind one type-code compare, so a tiny contiguous call
        ///     pays for nothing else. Every other same-dtype case leaves for
        ///     <see cref="ToArrayExact{T}"/>, every conversion (a T with no NumSharp dtype included) for
        ///     <see cref="ToArrayConverted{T}"/>, each ONE [NoInlining] frame. The conversion frame runs only the common
        ///     small conversion — a C-contiguous source whose pair already has a remembered cast kernel: one allocation,
        ///     one kernel call — and hands everything else to <see cref="ToArrayConvertedGeneral{T}"/>, which reads its
        ///     target from <see cref="InfoOf{T}"/> (a JIT-time constant there, so the inlined scalar converter folds to one
        ///     arm) and fills the per-(T, source dtype) kernel cache (<see cref="ToArrayCastKernel{T}"/>) the conversion
        ///     frame reads.
        ///     </para>
        /// </remarks>
        public unsafe T[] ToArray<T>() where T : unmanaged
        {
            var storage = Storage;
            // T ≠ dtype — a T that is no NumSharp dtype at all included — leaves through the conversion frame, whole.
            if (InfoOf<T>.NPTypeCode != storage.TypeCode)
                return ToArrayConverted<T>();

            ref readonly Shape shape = ref storage.ShapeReference;
            long count = shape.size;
            // The hot case: 1 … int.MaxValue elements, C-contiguous from the logical start. One unsigned compare checks
            // both bounds (count − 1 wraps to a huge value for an empty array).
            if (shape.IsContiguous && (ulong)(count - 1) < int.MaxValue)
            {
                T[] ret = ToArrayAllocate<T>((int)count);
                T* src = (T*)storage.Address + shape.offset;
                // Published fact (42): a result of at most 64 bytes is copied inline — at most four overlapping 16-byte
                // moves instead of the block copy's length check, call, size dispatch and return. The element count is
                // tested against a JIT-time constant and the byte count formed only inside the branch: a value the JIT
                // could share with the allocation's own size test would stay live across the allocation call and cost
                // this frame a sixth saved register on every call.
                if ((ulong)count <= (ulong)(64 / sizeof(T)))
                    ToArrayCopySmall((byte*)src, ref System.Runtime.CompilerServices.Unsafe.As<T, byte>(ref MemoryMarshal.GetArrayDataReference(ret)), (int)count * sizeof(T));
                else
                    new ReadOnlySpan<T>(src, (int)count).CopyTo(ret);
                // The raw source pointer is only valid while this array (and so its buffer) is alive.
                GC.KeepAlive(this);
                return ret;
            }

            return ToArrayExact<T>();
        }

        /// <summary>
        ///     The same-dtype copy of everything but <see cref="ToArray{T}"/>'s contiguous hot case: empty and oversized
        ///     arrays, one element, a single-value broadcast, and every other non-contiguous layout
        ///     (<see cref="ToArrayDedicated{T}"/>).
        /// </summary>
        /// <typeparam name="T">Element type (the dtype's).</typeparam>
        /// <returns>The copy (the shared empty array for a size-0 source).</returns>
        /// <exception cref="InvalidOperationException">The array holds more than <see cref="int.MaxValue"/> elements.</exception>
        /// <remarks>
        ///     A view of 1- or 2-byte elements whose innermost walk is a stride-2 or reversed run
        ///     (<see cref="ToArraySubwordIterSuits"/>) is copied by NumSharp's iterator, whose SIMD subword kernels beat this
        ///     file's element-by-element walks there (charter R8, trial 33's probes). Otherwise a rank-2 view goes to
        ///     <see cref="ToArrayPlane2D{T}"/> (published fact (43): canonicalized in registers, o6 −6.5 %), every other
        ///     rank to <see cref="ToArrayDedicated{T}"/>.
        ///     [NoInlining]: none of this frame's setup (the pin, the layout tests) lands in the hot frame. The oversize
        ///     refusal is thrown by <see cref="ToArrayThrowTooLarge"/>: a message formatted here would keep a
        ///     <see cref="System.Runtime.CompilerServices.DefaultInterpolatedStringHandler"/> in this frame, which the
        ///     prologue zeroes on every call — a cost every small non-contiguous copy (a 20 × 30 transpose: ≈ 84 ns in
        ///     all) would pay for a throw it never takes.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private unsafe T[] ToArrayExact<T>() where T : unmanaged
        {
            var storage = Storage;
            ref readonly Shape shape = ref storage.ShapeReference;
            long count = shape.size;
            if (count > int.MaxValue)
                ToArrayThrowTooLarge(count);
            if (count == 0)
                return System.Array.Empty<T>();

            int n = (int)count;
            T[] ret = ToArrayAllocate<T>(n);
            // Pinned only for the fill; nothing below retains the pointer.
            fixed (T* d = ret)
            {
                // Logical element 0 of ANY layout: base address + offset elements.
                T* first = (T*)storage.Address + shape.offset;
                if (n == 1)
                    *d = *first;
                else if (shape.IsScalarBroadcast)
                    new Span<T>(d, n).Fill(*first);
                // The element width is a JIT-time constant: for 4-, 8- and 16-byte elements this test compiles away.
                else if ((sizeof(T) == 1 || sizeof(T) == 2) && n >= ToArraySubwordIterMinElements
                         && ToArraySubwordIterSuits(shape.dimensions, shape.strides, (long)n * sizeof(T)))
                    // NumSharp's own copy (the spelling ConvertInto uses): its iterator walks the view with the SIMD
                    // subword kernels — a vpackus deinterleave for stride 2, a vpshufb reverse for stride −1 — straight into
                    // a non-owning wrapper over the pinned result.
                    NDIter.Copy(new UnmanagedStorage(ArraySlice.Wrap<T>(d, n), shape.Clean()), storage);
                else if (shape.dimensions.Length == 2)
                    ToArrayPlane2D(first, shape.dimensions[0], shape.dimensions[1], shape.strides[0], shape.strides[1], d);
                else
                    ToArrayDedicated(first, shape.dimensions, shape.strides, d);
            }

            // The raw source pointer is only valid while this array (and so its buffer) is alive.
            GC.KeepAlive(this);
            return ret;
        }

        /// <summary>
        ///     Fewest elements from which a 1- or 2-byte stride-2 / reversed view is copied by NumSharp's iterator
        ///     (<see cref="ToArraySubwordIterSuits"/>): its setup costs ≈ 190 ns, which its SIMD subword kernels repay over
        ///     this file's element walk from ≈ 1 024 elements on (a tie there: u8 [::2] 215 vs 239 ns, i16 236 vs 286 ns);
        ///     at 2 048 they win (u8 425 vs 288 ns, i16 459 vs 356 ns). Trial 33's probe, <c>t33/r8_cross_probe.cs</c>.
        /// </summary>
        private const int ToArraySubwordIterMinElements = 2048;

        /// <summary>
        ///     Whether a view of 1- or 2-byte elements is one NumSharp's SIMD subword copy kernels serve better than this
        ///     file's walk: its innermost non-unit axis has stride 2 (any size) or −1 (below
        ///     <see cref="ToArrayStreamPlaneBytes"/>, where the register-streamed reverse takes over), and every other
        ///     non-unit axis steps farther than that axis, so the iterator's innermost loop is exactly that run.
        /// </summary>
        /// <param name="dims">The view's dimensions.</param>
        /// <param name="strides">The view's element strides.</param>
        /// <param name="bytes">The result's size in bytes.</param>
        /// <returns>True when the iterator's subword kernels should copy the view.</returns>
        /// <remarks>
        ///     Charter R8's measured answer (trial 33, the harness's regime, NDIter.Copy against P0061's engine): u8 [::2]
        ///     64 K elements 2.35 vs 12.4 µs, u8 [::-1] 64 K 1.67 vs 12.1 µs, i16 [::2] 32 K 2.33 vs 6.15 µs, f16 [::2] 32 K
        ///     2.2 vs 7.6 µs, i16 [::2, ::2] 256 × 128 2.69 vs 7.86 µs, u8 [::2] 1 M 161 vs 279 µs — this file walks those
        ///     element by element (its vector deinterleaves and reverses exist for 4- and 8-byte elements only). The same
        ///     probe measured the iterator 1.3–5.4× SLOWER on every scored same-dtype layout of wider elements, so the route
        ///     is gated on the element width and on exactly these walks. A broadcast axis (stride 0) is never "farther", so
        ///     broadcast views stay on the file's fills.
        /// </remarks>
        private static bool ToArraySubwordIterSuits(long[] dims, long[] strides, long bytes)
        {
            // The innermost axis that the walk actually steps along (an extent-1 axis adds no offset).
            int last = dims.Length - 1;
            while (last >= 0 && dims[last] == 1)
                last--;
            if (last < 0)
                return false;
            long cs = strides[last];
            if (cs != 2 && !(cs == -1 && bytes < ToArrayStreamPlaneBytes))
                return false;
            long step = cs < 0 ? -cs : cs;
            for (int a = 0; a < last; a++)
                if (dims[a] != 1 && Math.Abs(strides[a]) <= step)
                    return false;
            return true;
        }

        /// <summary>
        ///     Throws the refusal of an array too large for a .NET array: the one place the message is formatted, so no
        ///     caller's frame carries the formatting state.
        /// </summary>
        /// <param name="count">The array's element count (more than <see cref="int.MaxValue"/>).</param>
        /// <exception cref="InvalidOperationException">Always.</exception>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static void ToArrayThrowTooLarge(long count)
            => throw new InvalidOperationException($"Array size {count} exceeds int.MaxValue; a .NET array cannot hold it.");

        /// <summary>
        ///     The T ≠ dtype path of <see cref="ToArray{T}"/>. Runs the common small conversion itself — a C-contiguous
        ///     source of 2 elements up to a result just under <see cref="ToArrayStreamPlaneBytes"/> whose (source dtype, T)
        ///     pair this file's AVX2 converters cover from one destination vector on (<see cref="ToArrayConvertVectors{T}"/>),
        ///     or whose pair already has a remembered cast kernel: one allocation, one converter or kernel call — and hands
        ///     every other conversion, whole, to <see cref="ToArrayConvertedGeneral{T}"/>.
        /// </summary>
        /// <typeparam name="T">Element type of the result.</typeparam>
        /// <returns>The converted array (the shared empty array for a size-0 source).</returns>
        /// <exception cref="NotSupportedException">
        ///     T is not a NumSharp dtype (raised by <see cref="ToArrayConvertedGeneral{T}"/>, before anything is read or
        ///     allocated).
        /// </exception>
        /// <exception cref="InvalidOperationException">
        ///     The array holds more than <see cref="int.MaxValue"/> elements (raised by
        ///     <see cref="ToArrayConvertedGeneral{T}"/>).
        /// </exception>
        /// <remarks>
        ///     <para>
        ///     Why a frame of its own: a small conversion (int16 3 × 4 → double[] ≈ 10 ns) is dominated by fixed cost, and
        ///     the general frame's fixed cost is its size. Its string-formatted throw keeps a
        ///     <see cref="System.Runtime.CompilerServices.DefaultInterpolatedStringHandler"/> in the frame, which the prologue
        ///     zeroes — 176 bytes — on every call; it saves all eight callee-saved registers, finds the source's element
        ///     size through a jump table, and calls out for the kernel lookup. Measured on that conversion in the harness's
        ///     regime: allocation + the kernel call alone 7.0–7.4 ns, the general frame's whole call 10.55 ns. This frame
        ///     holds none of it: the lookup is two loads and two compares, the element size is read only for a source at a
        ///     non-zero offset, and it formats no exception message.
        ///     </para>
        ///     <para>
        ///     It changes no result: every case it declines — one element, an empty or oversized array, a result of at
        ///     least <see cref="ToArrayStreamPlaneBytes"/> (the streamed pairs), decimal → double (the bit-identical vector
        ///     kernel), any non-contiguous layout, the generator switched off
        ///     (<see cref="NumSharp.Backends.Kernels.DirectILKernelGenerator.Enabled"/>), a pair without a kernel or not
        ///     looked up yet — runs the general frame exactly as before, and the case it takes is the general frame's own
        ///     contiguous-kernel branch (the same slot, the same kernel, the same call). A pair's first call fills its slot
        ///     through the general frame, so from the second call on it stays here.
        ///     </para>
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private unsafe T[] ToArrayConverted<T>() where T : unmanaged
        {
            var storage = Storage;
            ref readonly Shape shape = ref storage.ShapeReference;
            long count = shape.size;
            NPTypeCode source = storage.TypeCode;
            // T's dtype is a JIT-time constant: for a T with no dtype the whole test folds to false, and for
            // every T but double the decimal clause (the bit-identical vector kernel's case) folds away.
            // 2 ≤ count < the streaming size in one unsigned compare (count − 2 wraps for 0 and 1); the bound
            // keeps count far below int.MaxValue.
            if (InfoOf<T>.NPTypeCode != NPTypeCode.Empty
                && !(InfoOf<T>.NPTypeCode == NPTypeCode.Double && source == NPTypeCode.Decimal)
                && shape.IsContiguous
                && (ulong)(count - 2) < (ulong)(ToArrayStreamPlaneBytes / sizeof(T) - 2))
            {
                // Published fact (40): a pair this file's AVX2 converters cover, from one destination vector on, is
                // converted by them — no slot, no delegate. The coverage test is an inlined bit test decided BEFORE the
                // allocation, so a declined pair allocates once, in whichever frame converts it.
                //
                // Each branch forms its source pointer BEFORE the allocation call and keeps its own copy of the
                // allocate-and-convert sequence: then only the pointer, the count and the converter survive the call —
                // not the storage and its shape as well — and the tier-1 frame saves 6 callee-saved registers instead of
                // 8 (trial 34's JIT probe: `push` × 6, `sub rsp, 40`, against `push` × 8, `sub rsp, 56` when one shared
                // sequence served both branches). The unmanaged source never moves, so a pointer formed before the call
                // stays valid; `this` is kept alive to the end.
                if (count >= 32 / sizeof(T) && ToArrayConvertVectorsCovers<T>(source))
                {
                    // Logical element 0: base address + offset elements. A source at offset 0 (the usual case) skips the
                    // element-size lookup, a jump table over the type code.
                    byte* first = storage.Address;
                    if (shape.offset != 0)
                        first += shape.offset * storage.DTypeSize;
                    int n = (int)count;
                    T[] ret = ToArrayAllocate<T>(n);
                    // n ≥ 2, so the array's data reference is its element 0 (no null / empty test); pinned for the call
                    // only.
                    fixed (T* d = &MemoryMarshal.GetArrayDataReference(ret))
                        ToArrayConvertVectors(first, d, n, source);
                    // The raw source pointer is only valid while this array (and its buffer) is alive.
                    GC.KeepAlive(this);
                    return ret;
                }
                // Otherwise the remembered kernel. Slots only ever go from null to a kernel or to the no-kernel marker,
                // so one read decides: a pair not looked up yet (null) or without a kernel (the marker) takes the
                // general path, which fills the slot on a pair's first call.
                if (NumSharp.Backends.Kernels.DirectILKernelGenerator.Enabled)
                {
                    var kernel = ToArrayKernelSlots<T>.BySource[(int)source];
                    if (kernel != null && !ReferenceEquals(kernel, ToArrayKernelSlots<T>.None))
                    {
                        byte* first = storage.Address;
                        if (shape.offset != 0)
                            first += shape.offset * storage.DTypeSize;
                        int n = (int)count;
                        T[] ret = ToArrayAllocate<T>(n);
                        fixed (T* d = &MemoryMarshal.GetArrayDataReference(ret))
                            kernel(first, d, n);
                        GC.KeepAlive(this);
                        return ret;
                    }
                }
            }

            return ToArrayConvertedGeneral<T>();
        }

        /// <summary>
        ///     The general frame of the T ≠ dtype path: every conversion the lean frame (<see cref="ToArrayConverted{T}"/>)
        ///     declines, whole, in one frame — converts every element straight into the result with astype's conversions,
        ///     without an astype temporary.
        /// </summary>
        /// <typeparam name="T">Element type of the result.</typeparam>
        /// <returns>The converted array (the shared empty array for a size-0 source).</returns>
        /// <exception cref="NotSupportedException">T is not a NumSharp dtype (refused before anything is read or allocated).</exception>
        /// <exception cref="InvalidOperationException">The array holds more than <see cref="int.MaxValue"/> elements.</exception>
        /// <remarks>
        ///     <para>
        ///     Decisions, cheapest first: one element → the scalar converter; a contiguous source → decimal → double through
        ///     the bit-identical <see cref="DecimalToDoubleKernel"/> where its probe allows it, else — for a result of at
        ///     least <see cref="ToArrayStreamPlaneBytes"/> of one of seven common pairs — converted in registers straight
        ///     into whole lines on non-temporal stores (<see cref="ToArrayConvertLines{T}"/>), else — one of 24 common
        ///     pairs, at least one 32-byte destination vector — this file's AVX2 converters
        ///     (<see cref="ToArrayConvertVectors{T}"/>), else the generator's cast
        ///     kernel for the pair (cached, <see cref="ToArrayCastKernel{T}"/>) in one call; a pair without a kernel, or any
        ///     other layout, up to <see cref="ToArrayGatherMaxElements"/> elements → the scalar odometer
        ///     (<see cref="ToArrayGatherConvert{T}"/>); anything larger → astype's cast copy through the iterator
        ///     (<see cref="ConvertInto{T}"/>), after an address-order pre-fault when a non-contiguous source would fill a
        ///     large fresh result in its own memory order.
        ///     </para>
        ///     <para>
        ///     Why one frame, and why <c>target</c> is read here rather than passed in: the scalar converter
        ///     (<see cref="NDIterCasting.ConvertValue"/>) inlines, and with the target a JIT-time constant of the
        ///     instantiation its dtype switch folds to one arm. [NoInlining] keeps all of it out of the hot frame and out
        ///     of the lean conversion frame.
        ///     </para>
        ///     <para>
        ///     The lean frame's case — a contiguous source of 2 elements up to a result under
        ///     <see cref="ToArrayStreamPlaneBytes"/> whose pair has a kernel — is this frame's contiguous-kernel branch
        ///     too: it arrives here on a pair's first call (its slot not filled yet), and fills the slot the lean frame
        ///     reads from then on.
        ///     </para>
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private unsafe T[] ToArrayConvertedGeneral<T>() where T : unmanaged
        {
            NPTypeCode target = InfoOf<T>.NPTypeCode;
            // astype semantics need a NumSharp dtype to cast into; Guid, nint, DateTime… have none. Refused before
            // anything is read or allocated.
            if (target == NPTypeCode.Empty)
                throw new NotSupportedException($"Unable to convert to {typeof(T).Name}[]: {typeof(T).Name} is not a NumSharp dtype.");

            var storage = Storage;
            ref readonly Shape shape = ref storage.ShapeReference;
            long count = shape.size;
            if (count > int.MaxValue)
                throw new InvalidOperationException($"Array size {count} exceeds int.MaxValue; a .NET array cannot hold it.");
            if (count == 0)
                return System.Array.Empty<T>();

            int n = (int)count;
            T[] ret = ToArrayAllocate<T>(n);
            NPTypeCode source = storage.TypeCode;
            // Logical element 0 of ANY layout: base address + offset elements.
            byte* first = storage.Address + shape.offset * storage.DTypeSize;
            // The kernels write through raw pointers: the result is pinned for the conversion only.
            fixed (T* d = ret)
            {
                if (n == 1)
                    NDIterCasting.ConvertValue(first, d, source, target);
                else
                {
                    bool contiguous = shape.IsContiguous;
                    NumSharp.Backends.Kernels.DirectILKernelGenerator.CastKernel kernel;
                    // target is a JIT-time constant per T, so the decimal test folds away for every T but double.
                    if (contiguous && target == NPTypeCode.Double && source == NPTypeCode.Decimal && DecimalToDoubleKernel.MatchesRuntime)
                        DecimalToDoubleKernel.Convert((decimal*)first, (double*)d, n);
                    else if (contiguous && (long)n * sizeof(T) >= ToArrayStreamPlaneBytes && ToArrayConvertLines(first, d, n, source))
                    {
                        // A common pair, at least a mebibyte: converted in registers straight into whole lines — done.
                    }
                    else if (contiguous && ToArrayConvertVectors(first, d, n, source))
                    {
                        // One of the 24 common pairs, at least one destination vector: converted by the AVX2 loop — done.
                    }
                    else if (contiguous && (kernel = ToArrayCastKernel<T>(source)) != null)
                        kernel(first, d, n);
                    else if (n <= ToArrayGatherMaxElements)
                        // Small conversions of any layout: element by element through the scalar odometer, no iterator.
                        ToArrayGatherConvert(first, storage.DTypeSize, shape.dimensions, shape.strides, d, source);
                    else
                    {
                        // A non-contiguous source converts through the iterator in its own memory order: fault the
                        // fresh large-object result in address order first.
                        if (!contiguous && (long)n * sizeof(T) >= ToArrayLargeObjectBytes)
                            ToArrayTouchPages((byte*)d, (long)n * sizeof(T));
                        ConvertInto(d, target);
                    }
                }
            }

            // The raw source pointer is only valid while this array (and so its buffer) is alive.
            GC.KeepAlive(this);
            return ret;
        }

        /// <summary>
        ///     The generator's contiguous cast kernel from <paramref name="source"/> to T's dtype, looked up once per
        ///     (T, source) and remembered: <see cref="NumSharp.Backends.Kernels.DirectILKernelGenerator.TryGetCastKernel"/>
        ///     walks a chain of specialized lookups and a concurrent dictionary on every call (≈ 6 ns), which a small
        ///     conversion would pay in full.
        /// </summary>
        /// <typeparam name="T">Destination element type (a NumSharp dtype).</typeparam>
        /// <param name="source">Source dtype.</param>
        /// <returns>The kernel, or null when the generator is switched off or has no kernel for the pair.</returns>
        /// <remarks>
        ///     Answers only while <see cref="NumSharp.Backends.Kernels.DirectILKernelGenerator.Enabled"/> is set: that switch
        ///     is public, and a kernel remembered before it was turned off must not keep running after. A pair without a
        ///     kernel is remembered too (<see cref="ToArrayKernelSlots{T}.None"/>) so it is not probed again. The slots are
        ///     written without a lock: two threads resolving one slot at once store the same kernel, and a reference write
        ///     is atomic.
        /// </remarks>
        private static NumSharp.Backends.Kernels.DirectILKernelGenerator.CastKernel ToArrayCastKernel<T>(NPTypeCode source) where T : unmanaged
        {
            if (!NumSharp.Backends.Kernels.DirectILKernelGenerator.Enabled)
                return null;
            var slots = ToArrayKernelSlots<T>.BySource;
            var kernel = slots[(int)source];
            if (kernel == null)
                slots[(int)source] = kernel = NumSharp.Backends.Kernels.DirectILKernelGenerator.TryGetCastKernel(source, InfoOf<T>.NPTypeCode)
                                              ?? ToArrayKernelSlots<T>.None;
            return ReferenceEquals(kernel, ToArrayKernelSlots<T>.None) ? null : kernel;
        }

        /// <summary>
        ///     The remembered contiguous cast kernels into T, one slot per source type code: null = not looked up yet,
        ///     <see cref="None"/> = looked up, no kernel.
        /// </summary>
        /// <typeparam name="T">Destination element type.</typeparam>
        private static unsafe class ToArrayKernelSlots<T> where T : unmanaged
        {
            /// <summary>The slots, indexed by source <see cref="NPTypeCode"/> (sized to the largest code, Complex).</summary>
            internal static readonly NumSharp.Backends.Kernels.DirectILKernelGenerator.CastKernel[] BySource =
                new NumSharp.Backends.Kernels.DirectILKernelGenerator.CastKernel[(int)NPTypeCode.Complex + 1];

            /// <summary>The "no kernel for this pair" marker; compared by reference, never invoked.</summary>
            internal static readonly NumSharp.Backends.Kernels.DirectILKernelGenerator.CastKernel None = Unreachable;

            /// <summary>The marker's target, which nothing calls.</summary>
            /// <param name="src">Unused.</param>
            /// <param name="dst">Unused.</param>
            /// <param name="count">Unused.</param>
            /// <exception cref="InvalidOperationException">Always: reaching it would mean the marker was invoked.</exception>
            private static void Unreachable(void* src, void* dst, long count)
                => throw new InvalidOperationException("The no-kernel marker is never invoked.");
        }

        /// <summary>
        ///     Converts a large contiguous run of one of seven common pairs — float64 → float32, float32 → float64,
        ///     int64 → int32, int32 → int64, int32 → float64, int32 → float32 and uint8 → float32 — in registers, straight
        ///     into whole 64-byte destination lines on non-temporal stores (<see cref="ToArrayConvertLinesCore{TFrom, TTo}"/>);
        ///     declines every other pair without writing anything.
        /// </summary>
        /// <typeparam name="T">Destination element type (a NumSharp dtype).</typeparam>
        /// <param name="s">The source's logical element 0: a C-contiguous run of <paramref name="n"/> elements.</param>
        /// <param name="d">The destination (pinned), with room for <paramref name="n"/> elements.</param>
        /// <param name="n">Element count (≥ 1).</param>
        /// <param name="source">Source dtype (≠ T's).</param>
        /// <returns>
        ///     True when the pair is one of the seven and the whole run has been converted; false — nothing written — for
        ///     every other pair, without AVX2, or for a destination not aligned to its element size. On false the caller
        ///     converts through the generator's cast kernel as before.
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     Why: the generator's contiguous kernels store one 16-byte vector after another, so every fresh line of a
        ///     large result is read for ownership before it is written — traffic that result, past every cache level it
        ///     could stay in, never repays. Converted in registers and written by whole lines with non-temporal stores, the
        ///     lines are never read. Measured on a 1 000 000-element float64 → float32 conversion (probe in the harness's
        ///     regime: pinned CPU, a full compacting GC before every call, anchor-like neighbour calls): 624–744 µs in the
        ///     kernel's shape, 536–548 µs this way, 498–501 µs with the page-ahead prefetch. The same streaming through an
        ///     L1 scratch measured only 6–13 % in that
        ///     probe and under 1 % in the harness (the published fact (23)): each chunk's conversion and its stream-out
        ///     then alternate, where here the loads of the next line overlap the stores of the last.
        ///     </para>
        ///     <para>
        ///     Every vector conversion is bit-identical to the scalar converter astype applies
        ///     (<see cref="NDIterCasting.ConvertValue"/>): per element the same IEEE operation under the default rounding —
        ///     vcvtpd2ps / vcvtps2pd / vcvtdq2ps / vcvtdq2pd round or widen exactly as the scalar casts do, NaN payloads
        ///     included —, the int64 → int32 narrowing keeps the low 32 bits (the scalar converter's modular wrap), and the
        ///     widenings are exact. The partial first and last lines go through that scalar converter itself.
        ///     </para>
        ///     <para>
        ///     The caller routes only results of at least <see cref="ToArrayStreamPlaneBytes"/>: below it a result can still
        ///     be cache-resident for its caller, and streaming it past the caches would only move the cost to the caller's
        ///     first read. A managed <c>T[]</c> is aligned to its element size on every 64-bit runtime; the alignment test
        ///     keeps the aligned line stores safe where it is not. [NoInlining]: none of this lands in the conversion
        ///     frame, which stays lean (the published fact (25)).
        ///     </para>
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static unsafe bool ToArrayConvertLines<T>(byte* s, T* d, long n, NPTypeCode source) where T : unmanaged
        {
            if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported || ((ulong)d & (ulong)(sizeof(T) - 1)) != 0)
                return false;
            // T is a JIT-time constant: each instantiation keeps only its own arm.
            if (typeof(T) == typeof(float))
            {
                if (source == NPTypeCode.Double)
                    ToArrayConvertLinesCore((double*)s, (float*)d, n);
                else if (source == NPTypeCode.Int32)
                    ToArrayConvertLinesCore((int*)s, (float*)d, n);
                else if (source == NPTypeCode.Byte)
                    ToArrayConvertLinesCore(s, (float*)d, n);
                else
                    return false;
                return true;
            }
            if (typeof(T) == typeof(double))
            {
                if (source == NPTypeCode.Single)
                    ToArrayConvertLinesCore((float*)s, (double*)d, n);
                else if (source == NPTypeCode.Int32)
                    ToArrayConvertLinesCore((int*)s, (double*)d, n);
                else
                    return false;
                return true;
            }
            if (typeof(T) == typeof(int) && source == NPTypeCode.Int64)
            {
                ToArrayConvertLinesCore((long*)s, (int*)d, n);
                return true;
            }
            if (typeof(T) == typeof(long) && source == NPTypeCode.Int32)
            {
                ToArrayConvertLinesCore((int*)s, (long*)d, n);
                return true;
            }
            return false;
        }

        /// <summary>
        ///     How far ahead of its lines <see cref="ToArrayConvertLinesCore{TFrom, TTo}"/> prefetches the source: one
        ///     4 KB page. The hardware stream prefetcher does not prefetch across a 4 KB page boundary, so without it each
        ///     new source page starts with demand misses. Measured on a 1 000 000-element float64 → float32 conversion
        ///     (probe, harness regime) against no prefetch: 2 KB ahead −5 %, 4 KB and 8 KB ahead −7 … −8 % alike — one
        ///     page is the shortest distance that covers the crossing.
        /// </summary>
        private const int ToArrayConvertPrefetchBytes = 4096;

        /// <summary>
        ///     How far ahead of its reads a large walker that does not convert prefetches its source: the reversed-run
        ///     streamers (<see cref="ToArrayReverseStreamed{T}"/>, <see cref="ToArrayReverseStreamedHalf16"/> — one page
        ///     BELOW their position, since the run walks down) and the stride-2 deinterleaves
        ///     (<see cref="DeinterleaveRows4"/>, <see cref="DeinterleaveRows8"/> — one page above). One 4 KB page, for
        ///     <see cref="ToArrayConvertPrefetchBytes"/>'s reason: the hardware stream prefetcher stops at every page
        ///     boundary, so each new source page would otherwise start with demand misses.
        /// </summary>
        /// <remarks>
        ///     A prefetch is a hint: it never faults (an address past the source, even an unmapped one, is dropped) and
        ///     changes no result. A source already in cache — a stride-2 view converted over and over — pays one prefetch per
        ///     64 source bytes and gains nothing; that is the price of a size-blind rule, small next to the loop's own work.
        /// </remarks>
        private const int ToArrayWalkPrefetchBytes = 4096;

        /// <summary>
        ///     The line driver of <see cref="ToArrayConvertLines{T}"/> for one pair: the scalar converter up to the
        ///     destination's first 64-byte line boundary, one <see cref="ToArrayConvertLine{TFrom, TTo}"/> per whole line,
        ///     the scalar converter again for the last partial line, then one store fence; the source is prefetched one
        ///     page ahead of the lines (<see cref="ToArrayConvertPrefetchBytes"/>).
        /// </summary>
        /// <typeparam name="TFrom">Source element type.</typeparam>
        /// <typeparam name="TTo">Destination element type (4 or 8 bytes).</typeparam>
        /// <param name="s">The source run (any alignment).</param>
        /// <param name="d">The destination (pinned; aligned to its element size).</param>
        /// <param name="n">Element count (≥ 1).</param>
        /// <remarks>
        ///     Every whole line receives exactly two aligned 32-byte non-temporal stores; the partial lines at both ends —
        ///     shared with the array's header and with whatever follows the array — receive ordinary stores, so no line gets
        ///     both kinds. Non-temporal stores are weakly ordered against every other store: the one sfence at the end makes
        ///     them globally visible before the result can be published to another thread. Both ends convert through
        ///     <see cref="NDIterCasting.ConvertValue"/> with the type codes read from <see cref="InfoOf{T}"/> — JIT-time
        ///     constants here, so its dtype switches fold to the one conversion. Reads exactly the run's elements.
        /// </remarks>
        private static unsafe void ToArrayConvertLinesCore<TFrom, TTo>(TFrom* s, TTo* d, long n) where TFrom : unmanaged where TTo : unmanaged
        {
            NPTypeCode from = InfoOf<TFrom>.NPTypeCode, to = InfoOf<TTo>.NPTypeCode;
            long perLine = 64 / sizeof(TTo);
            // Elements before the destination's first line boundary (the whole run when it ends first): d is aligned to
            // its element size, so the byte distance divides exactly.
            long lead = Math.Min((long)((64 - ((ulong)d & 63)) & 63) / sizeof(TTo), n);
            long i = 0;
            for (; i < lead; i++)
                NDIterCasting.ConvertValue(s + i, d + i, from, to);
            // [lead, body) is a whole number of lines.
            long body = i + ((n - i) & ~(perLine - 1));
            for (; i < body; i += perLine)
            {
                // The source one page ahead (reading past the run's end is harmless: a prefetch never faults).
                byte* ahead = (byte*)(s + i) + ToArrayConvertPrefetchBytes;
                System.Runtime.Intrinsics.X86.Sse.Prefetch0(ahead);
                // A line of an 8 → 4-byte pair consumes two source lines: both are prefetched.
                if (sizeof(TFrom) > sizeof(TTo))
                    System.Runtime.Intrinsics.X86.Sse.Prefetch0(ahead + 64);
                ToArrayConvertLine(s + i, d + i);
            }
            for (; i < n; i++)
                NDIterCasting.ConvertValue(s + i, d + i, from, to);
            // Make the weakly-ordered non-temporal stores visible before anything that could publish the result.
            System.Runtime.Intrinsics.X86.Sse.StoreFence();
        }

        /// <summary>
        ///     Converts the elements of one whole 64-byte destination line (64 / sizeof(TTo) of them) in registers and
        ///     writes the line with two aligned 32-byte non-temporal stores.
        /// </summary>
        /// <typeparam name="TFrom">Source element type.</typeparam>
        /// <typeparam name="TTo">Destination element type.</typeparam>
        /// <param name="s">The line's first source element (any alignment); exactly 64 / sizeof(TTo) elements are read.</param>
        /// <param name="d">The line (64-byte aligned).</param>
        /// <exception cref="InvalidOperationException">A pair outside the seven (unreachable: only <see cref="ToArrayConvertLines{T}"/> instantiates it, with the seven).</exception>
        /// <remarks>
        ///     One arm per pair; the type tests are JIT-time constants, so every instantiation compiles to its own arm alone
        ///     and inlines into the driver's loop. Loads never reach past the line's own source elements (the uint8 arm
        ///     reads its 16 bytes as two 8-byte zero-extending loads; the 4 → 8-byte arms load 16 bytes per 32-byte store).
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void ToArrayConvertLine<TFrom, TTo>(TFrom* s, TTo* d) where TFrom : unmanaged where TTo : unmanaged
        {
            if (typeof(TFrom) == typeof(double) && typeof(TTo) == typeof(float))
            {
                // vcvtpd2ps on each 4-lane half: the scalar (float)double, lane by lane.
                var lo = System.Runtime.Intrinsics.Vector256.Narrow(System.Runtime.Intrinsics.X86.Avx.LoadVector256((double*)s), System.Runtime.Intrinsics.X86.Avx.LoadVector256((double*)s + 4));
                var hi = System.Runtime.Intrinsics.Vector256.Narrow(System.Runtime.Intrinsics.X86.Avx.LoadVector256((double*)s + 8), System.Runtime.Intrinsics.X86.Avx.LoadVector256((double*)s + 12));
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((float*)d, lo);
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((float*)d + 8, hi);
            }
            else if (typeof(TFrom) == typeof(long) && typeof(TTo) == typeof(int))
            {
                // Keeps the low 32 bits of every lane: the scalar converter's unchecked (modular) narrowing.
                var lo = System.Runtime.Intrinsics.Vector256.Narrow(System.Runtime.Intrinsics.X86.Avx.LoadVector256((long*)s), System.Runtime.Intrinsics.X86.Avx.LoadVector256((long*)s + 4));
                var hi = System.Runtime.Intrinsics.Vector256.Narrow(System.Runtime.Intrinsics.X86.Avx.LoadVector256((long*)s + 8), System.Runtime.Intrinsics.X86.Avx.LoadVector256((long*)s + 12));
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((int*)d, lo);
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((int*)d + 8, hi);
            }
            else if (typeof(TFrom) == typeof(int) && typeof(TTo) == typeof(float))
            {
                // vcvtdq2ps: round-to-nearest-even, as the scalar int → float conversion.
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((float*)d, System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Single(System.Runtime.Intrinsics.X86.Avx.LoadVector256((int*)s)));
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((float*)d + 8, System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Single(System.Runtime.Intrinsics.X86.Avx.LoadVector256((int*)s + 8)));
            }
            else if (typeof(TFrom) == typeof(byte) && typeof(TTo) == typeof(float))
            {
                // vpmovzxbd (8 bytes → 8 exact int32 lanes), then vcvtdq2ps: exact for 0…255.
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((float*)d, System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Single(System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32((byte*)s)));
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((float*)d + 8, System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Single(System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32((byte*)s + 8)));
            }
            else if (typeof(TFrom) == typeof(float) && typeof(TTo) == typeof(double))
            {
                // vcvtps2pd: exact widening; a signalling NaN is quieted exactly as the scalar (double)float quiets it.
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((double*)d, System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse.LoadVector128((float*)s)));
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((double*)d + 4, System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse.LoadVector128((float*)s + 4)));
            }
            else if (typeof(TFrom) == typeof(int) && typeof(TTo) == typeof(double))
            {
                // vcvtdq2pd: exact.
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((double*)d, System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse2.LoadVector128((int*)s)));
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((double*)d + 4, System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse2.LoadVector128((int*)s + 4)));
            }
            else if (typeof(TFrom) == typeof(int) && typeof(TTo) == typeof(long))
            {
                // vpmovsxdq: sign extension, exact.
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((long*)d, System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int64((int*)s));
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((long*)d + 4, System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int64((int*)s + 4));
            }
            else
                throw new InvalidOperationException($"No line conversion from {typeof(TFrom).Name} to {typeof(TTo).Name}.");
        }

        /// <summary>
        ///     Whether this file's AVX2 converters (<see cref="ToArrayConvertVectors{T}"/>) cover the pair (source → T) on
        ///     this host — decided before anything is allocated.
        /// </summary>
        /// <typeparam name="T">Destination element type.</typeparam>
        /// <param name="source">Source dtype.</param>
        /// <returns>True for the 24 pairs under AVX2; false otherwise.</returns>
        /// <remarks>
        ///     T and AVX2 support are JIT-time constants, so each instantiation is one bit test over the source code. It is
        ///     the single list of covered pairs: <see cref="ToArrayConvertVectors{T}"/> consults it before its dispatch, and
        ///     its dispatch covers exactly these sources (a disagreement throws there, loudly, rather than leave a result
        ///     unconverted).
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static bool ToArrayConvertVectorsCovers<T>(NPTypeCode source) where T : unmanaged
        {
            if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                return false;
            if (typeof(T) == typeof(double))
                return source is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                    or NPTypeCode.Int32 or NPTypeCode.Single;
            if (typeof(T) == typeof(float))
                return source is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                    or NPTypeCode.Int32 or NPTypeCode.Double;
            if (typeof(T) == typeof(long))
                return source is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                    or NPTypeCode.Int32 or NPTypeCode.UInt32;
            if (typeof(T) == typeof(int))
                return source is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                    or NPTypeCode.Int64;
            return false;
        }

        /// <summary>
        ///     Converts a contiguous run of one of 24 common pairs with 32-byte AVX2 vectors on ordinary stores (published
        ///     fact (40)); declines every other pair, a run shorter than one destination vector, and a host without AVX2,
        ///     writing nothing.
        /// </summary>
        /// <typeparam name="T">Destination element type (a NumSharp dtype).</typeparam>
        /// <param name="s">The source's logical element 0: a C-contiguous run of <paramref name="n"/> elements.</param>
        /// <param name="d">The destination (pinned; any alignment), with room for <paramref name="n"/> elements.</param>
        /// <param name="n">Element count (≥ 1).</param>
        /// <param name="source">Source dtype (≠ T's).</param>
        /// <returns>True when the pair is covered and the whole run converted; false — nothing written — otherwise.</returns>
        /// <exception cref="InvalidOperationException">
        ///     <see cref="ToArrayConvertVectorsCovers{T}"/> accepted a pair this dispatch does not convert (unreachable: the
        ///     two list the same 24 pairs; the throw makes a future disagreement loud instead of a silent unconverted result).
        /// </exception>
        /// <remarks>
        ///     <para>
        ///     The pairs: into float64 from int8, uint8, int16, uint16, char, int32 and float32; into float32 from int8,
        ///     uint8, int16, uint16, char, int32 and float64; into int64 from int8, uint8, int16, uint16, char, int32 and
        ///     uint32; into int32 from int8, uint8, int16, uint16, char and int64. Each lane is the answer of the scalar
        ///     converter astype applies (<see cref="NDIterCasting.ConvertValue"/>): the widenings are exact (sign or zero
        ///     extension, then vcvtdq2pd, or vcvtdq2ps for integers of at most 16 bits); int32 → float32 and float64 →
        ///     float32 round to nearest-even as the scalar conversions do; float32 → float64 quiets a signalling NaN as
        ///     the scalar widening does; int64 → int32 keeps the low 32 bits (the scalar converter's modular narrowing);
        ///     a char reads as its uint16 code unit, as the scalar converter reads it.
        ///     </para>
        ///     <para>
        ///     Why (published fact (40), E P0065): the generator's contiguous kernels for these pairs pay a delegate call
        ///     and a generic prologue, and the size-changing integer pairs run far below vector speed — per-pair loops in the
        ///     file measured 1.3–10× faster than one kernel call, and o8 (int16 → float64, 12 elements) −35 % in the
        ///     harness. The run is covered by whole 32-byte destination vectors, the last one ending exactly at the run's
        ///     end and overlapping its predecessor: it rewrites a few elements with the values they already hold (the source
        ///     and the destination never overlap — the result is a fresh array) and reads only elements of the run. A run
        ///     shorter than one vector is declined: there the kernel's call measured as fast as plain casts. [NoInlining]
        ///     keeps the pair switch out of the conversion frames.
        ///     </para>
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static unsafe bool ToArrayConvertVectors<T>(byte* s, T* d, long n, NPTypeCode source) where T : unmanaged
        {
            if (n < 32 / sizeof(T) || !ToArrayConvertVectorsCovers<T>(source))
                return false;
            if (typeof(T) == typeof(double))
            {
                switch (source)
                {
                    case NPTypeCode.SByte: ToArrayConvertVectorsCore((sbyte*)s, (double*)d, n); return true;
                    case NPTypeCode.Byte: ToArrayConvertVectorsCore(s, (double*)d, n); return true;
                    case NPTypeCode.Int16: ToArrayConvertVectorsCore((short*)s, (double*)d, n); return true;
                    case NPTypeCode.UInt16:
                    case NPTypeCode.Char: ToArrayConvertVectorsCore((ushort*)s, (double*)d, n); return true;
                    case NPTypeCode.Int32: ToArrayConvertVectorsCore((int*)s, (double*)d, n); return true;
                    case NPTypeCode.Single: ToArrayConvertVectorsCore((float*)s, (double*)d, n); return true;
                }
            }
            else if (typeof(T) == typeof(float))
            {
                switch (source)
                {
                    case NPTypeCode.SByte: ToArrayConvertVectorsCore((sbyte*)s, (float*)d, n); return true;
                    case NPTypeCode.Byte: ToArrayConvertVectorsCore(s, (float*)d, n); return true;
                    case NPTypeCode.Int16: ToArrayConvertVectorsCore((short*)s, (float*)d, n); return true;
                    case NPTypeCode.UInt16:
                    case NPTypeCode.Char: ToArrayConvertVectorsCore((ushort*)s, (float*)d, n); return true;
                    case NPTypeCode.Int32: ToArrayConvertVectorsCore((int*)s, (float*)d, n); return true;
                    case NPTypeCode.Double: ToArrayConvertVectorsCore((double*)s, (float*)d, n); return true;
                }
            }
            else if (typeof(T) == typeof(long))
            {
                switch (source)
                {
                    case NPTypeCode.SByte: ToArrayConvertVectorsCore((sbyte*)s, (long*)d, n); return true;
                    case NPTypeCode.Byte: ToArrayConvertVectorsCore(s, (long*)d, n); return true;
                    case NPTypeCode.Int16: ToArrayConvertVectorsCore((short*)s, (long*)d, n); return true;
                    case NPTypeCode.UInt16:
                    case NPTypeCode.Char: ToArrayConvertVectorsCore((ushort*)s, (long*)d, n); return true;
                    case NPTypeCode.Int32: ToArrayConvertVectorsCore((int*)s, (long*)d, n); return true;
                    case NPTypeCode.UInt32: ToArrayConvertVectorsCore((uint*)s, (long*)d, n); return true;
                }
            }
            else if (typeof(T) == typeof(int))
            {
                switch (source)
                {
                    case NPTypeCode.SByte: ToArrayConvertVectorsCore((sbyte*)s, (int*)d, n); return true;
                    case NPTypeCode.Byte: ToArrayConvertVectorsCore(s, (int*)d, n); return true;
                    case NPTypeCode.Int16: ToArrayConvertVectorsCore((short*)s, (int*)d, n); return true;
                    case NPTypeCode.UInt16:
                    case NPTypeCode.Char: ToArrayConvertVectorsCore((ushort*)s, (int*)d, n); return true;
                    case NPTypeCode.Int64: ToArrayConvertVectorsCore((long*)s, (int*)d, n); return true;
                }
            }
            ToArrayThrowVectorCoverage(source, typeof(T));
            return false;
        }

        /// <summary>Throws the (unreachable) disagreement between the converters' coverage list and their dispatch.</summary>
        /// <param name="source">The source dtype the list accepted.</param>
        /// <param name="target">The destination element type.</param>
        /// <exception cref="InvalidOperationException">Always.</exception>
        /// <remarks>Out of line so the dispatcher's frame carries no message formatting.</remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static void ToArrayThrowVectorCoverage(NPTypeCode source, Type target)
            => throw new InvalidOperationException($"The AVX2 converters list {source} → {target.Name} as covered but do not convert it.");

        /// <summary>
        ///     The loop of <see cref="ToArrayConvertVectors{T}"/> for one pair: whole 32-byte destination vectors, the last
        ///     one ending exactly at the run's end.
        /// </summary>
        /// <typeparam name="TFrom">Source element type.</typeparam>
        /// <typeparam name="TTo">Destination element type (4 or 8 bytes).</typeparam>
        /// <param name="s">The source run (any alignment).</param>
        /// <param name="d">The destination (any alignment; never overlapping the source).</param>
        /// <param name="n">Element count: at least one destination vector (32 / sizeof(TTo) elements).</param>
        /// <remarks>
        ///     The ragged end is one more vector ending at the run's end, overlapping the last whole one: its lanes rewrite
        ///     elements with the values they already hold, and its loads read elements of the run only.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void ToArrayConvertVectorsCore<TFrom, TTo>(TFrom* s, TTo* d, long n) where TFrom : unmanaged where TTo : unmanaged
        {
            long w = 32 / sizeof(TTo);
            long k = 0;
            for (; k + w <= n; k += w)
                ToArrayConvertVector(s + k, d + k);
            if (k < n)
                ToArrayConvertVector(s + n - w, d + n - w);
        }

        /// <summary>Converts one 32-byte destination vector (32 / sizeof(TTo) elements) of one pair.</summary>
        /// <typeparam name="TFrom">Source element type.</typeparam>
        /// <typeparam name="TTo">Destination element type.</typeparam>
        /// <param name="s">The vector's first source element; exactly 32 / sizeof(TTo) elements are read.</param>
        /// <param name="d">The vector's first destination element (any alignment).</param>
        /// <exception cref="InvalidOperationException">A pair outside the 24 (unreachable: only the dispatcher instantiates it, with the 24).</exception>
        /// <remarks>
        ///     One arm per pair; the type tests are JIT-time constants, so each instantiation compiles to its own arm and
        ///     inlines into the loop. The widening loads read exactly the vector's source bytes (vpmovsx / vpmovzx from
        ///     memory: 4, 8 or 16 bytes), the narrowing ones two 32-byte loads.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void ToArrayConvertVector<TFrom, TTo>(TFrom* s, TTo* d) where TFrom : unmanaged where TTo : unmanaged
        {
            if (typeof(TTo) == typeof(double))
            {
                System.Runtime.Intrinsics.Vector256<double> v;
                if (typeof(TFrom) == typeof(float))
                    // vcvtps2pd: exact; a signalling NaN is quieted as the scalar widening quiets it.
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse.LoadVector128((float*)s));
                else if (typeof(TFrom) == typeof(int))
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse2.LoadVector128((int*)s));
                else if (typeof(TFrom) == typeof(short))
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse41.ConvertToVector128Int32((short*)s));
                else if (typeof(TFrom) == typeof(ushort))
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse41.ConvertToVector128Int32((ushort*)s));
                else if (typeof(TFrom) == typeof(sbyte))
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse41.ConvertToVector128Int32((sbyte*)s));
                else if (typeof(TFrom) == typeof(byte))
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse41.ConvertToVector128Int32((byte*)s));
                else
                    throw new InvalidOperationException($"No vector conversion from {typeof(TFrom).Name} to {typeof(TTo).Name}.");
                System.Runtime.Intrinsics.X86.Avx.Store((double*)d, v);
            }
            else if (typeof(TTo) == typeof(float))
            {
                System.Runtime.Intrinsics.Vector256<float> v;
                if (typeof(TFrom) == typeof(double))
                    // vcvtpd2ps on each half: round to nearest-even, lane by lane, as the scalar (float)double.
                    v = System.Runtime.Intrinsics.Vector256.Narrow(System.Runtime.Intrinsics.X86.Avx.LoadVector256((double*)s),
                        System.Runtime.Intrinsics.X86.Avx.LoadVector256((double*)s + 4));
                else if (typeof(TFrom) == typeof(int))
                    // vcvtdq2ps: round to nearest-even, as the scalar int → float conversion.
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Single(System.Runtime.Intrinsics.X86.Avx.LoadVector256((int*)s));
                else if (typeof(TFrom) == typeof(short))
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Single(System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32((short*)s));
                else if (typeof(TFrom) == typeof(ushort))
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Single(System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32((ushort*)s));
                else if (typeof(TFrom) == typeof(sbyte))
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Single(System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32((sbyte*)s));
                else if (typeof(TFrom) == typeof(byte))
                    v = System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Single(System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32((byte*)s));
                else
                    throw new InvalidOperationException($"No vector conversion from {typeof(TFrom).Name} to {typeof(TTo).Name}.");
                System.Runtime.Intrinsics.X86.Avx.Store((float*)d, v);
            }
            else if (typeof(TTo) == typeof(long))
            {
                System.Runtime.Intrinsics.Vector256<long> v;
                if (typeof(TFrom) == typeof(int))
                    v = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int64((int*)s);
                else if (typeof(TFrom) == typeof(uint))
                    v = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int64((uint*)s);
                else if (typeof(TFrom) == typeof(short))
                    v = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int64((short*)s);
                else if (typeof(TFrom) == typeof(ushort))
                    v = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int64((ushort*)s);
                else if (typeof(TFrom) == typeof(sbyte))
                    v = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int64((sbyte*)s);
                else if (typeof(TFrom) == typeof(byte))
                    v = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int64((byte*)s);
                else
                    throw new InvalidOperationException($"No vector conversion from {typeof(TFrom).Name} to {typeof(TTo).Name}.");
                System.Runtime.Intrinsics.X86.Avx.Store((long*)d, v);
            }
            else if (typeof(TTo) == typeof(int))
            {
                System.Runtime.Intrinsics.Vector256<int> v;
                if (typeof(TFrom) == typeof(long))
                    // Keeps the low 32 bits of every lane: the scalar converter's modular narrowing.
                    v = System.Runtime.Intrinsics.Vector256.Narrow(System.Runtime.Intrinsics.X86.Avx.LoadVector256((long*)s),
                        System.Runtime.Intrinsics.X86.Avx.LoadVector256((long*)s + 4));
                else if (typeof(TFrom) == typeof(short))
                    v = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32((short*)s);
                else if (typeof(TFrom) == typeof(ushort))
                    v = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32((ushort*)s);
                else if (typeof(TFrom) == typeof(sbyte))
                    v = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32((sbyte*)s);
                else if (typeof(TFrom) == typeof(byte))
                    v = System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32((byte*)s);
                else
                    throw new InvalidOperationException($"No vector conversion from {typeof(TFrom).Name} to {typeof(TTo).Name}.");
                System.Runtime.Intrinsics.X86.Avx.Store((int*)d, v);
            }
            else
                throw new InvalidOperationException($"No vector conversion from {typeof(TFrom).Name} to {typeof(TTo).Name}.");
        }

        /// <summary>
        ///     Largest conversion (T ≠ dtype) handled by the scalar odometer of <see cref="ToArrayGatherConvert{T}"/>
        ///     instead of the iterator's cast copy. The iterator's construction and dispatch are a fixed cost of roughly
        ///     a microsecond, which per-element conversion repays only below about a thousand elements; above it the
        ///     iterator's coalesced, kernel-driven cast copy wins.
        /// </summary>
        private const int ToArrayGatherMaxElements = 1024;

        /// <summary>
        ///     Largest transposed plane (rows × columns × element size) that <see cref="ToArrayPlaneFill{T}"/> transposes
        ///     with <see cref="ToArrayRowsTranspose{T}"/>'s register blocks. Past it the plane goes to
        ///     <see cref="TransposeBands{T}"/>, whose 64-column bands bound the pages one 8-row strip touches — a concern
        ///     only once a plane spans many pages; below it the bands' extra bookkeeping is pure overhead and the
        ///     register blocks finish ragged row counts (4–7 rows) that the bands would copy one row at a time.
        /// </summary>
        private const int ToArraySmallPlaneBytes = 32 * 1024;

        /// <summary>
        ///     Byte size from which a result skips the GC's zeroing. Below it the JIT's inline allocation helper plus the
        ///     zeroing is cheaper than <see cref="GC.AllocateUninitializedArray{T}(int, bool)"/>, which leaves the inline
        ///     path for a runtime call: a 2.4 KB result measured 10–21 % slower through it end-to-end.
        /// </summary>
        private const int ToArrayUninitializedBytes = 8192;

        /// <summary>
        ///     The runtime's large-object threshold: a result at least this big is a large object whose pages come fresh
        ///     from the OS after a collection and fault in on first touch.
        /// </summary>
        private const int ToArrayLargeObjectBytes = 85_000;

        /// <summary>
        ///     Allocates a result the copy will overwrite completely: uninitialized from
        ///     <see cref="ToArrayUninitializedBytes"/>, zeroed below it. Never on the pinned-object heap: a repeated
        ///     mid-size pinned result costs a fresh pinned-heap commit each time (measured 12–16 µs per 64–82 KB result,
        ///     against ~1 µs on the small-object heap), and the copy only needs the result pinned for its own duration.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="n">Element count (≥ 1).</param>
        /// <returns>The new array; its contents are unspecified when uninitialized.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe T[] ToArrayAllocate<T>(int n) where T : unmanaged
            => (long)n * sizeof(T) >= ToArrayUninitializedBytes ? GC.AllocateUninitializedArray<T>(n) : new T[n];

        /// <summary>
        ///     Copies 1 … 64 bytes with at most four overlapping moves (published fact (42)): the same-dtype hot path's copy
        ///     of a tiny contiguous result, inlined into <see cref="ToArray{T}"/>'s frame.
        /// </summary>
        /// <param name="s">Source (any alignment); exactly <paramref name="bytes"/> bytes are read.</param>
        /// <param name="d">Destination (the result's first byte; any alignment); exactly <paramref name="bytes"/> bytes are written.</param>
        /// <param name="bytes">Byte count, 1 … 64.</param>
        /// <remarks>
        ///     Four byte-count ranges: 16 … 64 bytes take two (≤ 32) or four 16-byte moves — [0, 16) and [bytes − 16, bytes),
        ///     plus [16, 32) and [bytes − 32, bytes − 16) past 32 — whose overlap covers every byte exactly with its own
        ///     value; 8 … 15 bytes two 8-byte moves, 4 … 7 two 4-byte, 2 … 3 two 2-byte, 1 byte one. All loads precede the
        ///     stores of each range, and the source and the fresh result never overlap. The destination is a managed
        ///     reference, so nothing is pinned and the GC may move the result between calls. Why: F's trial-29 measurement,
        ///     o4 (5 × int64) −9 % in the harness against the block copy, whose call and size dispatch are o4's fixed cost.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void ToArrayCopySmall(byte* s, ref byte d, int bytes)
        {
            if (bytes >= 16)
            {
                var a = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Runtime.Intrinsics.Vector128<byte>>(s);
                var b = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Runtime.Intrinsics.Vector128<byte>>(s + bytes - 16);
                if (bytes > 32)
                {
                    var c = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Runtime.Intrinsics.Vector128<byte>>(s + 16);
                    var e = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Runtime.Intrinsics.Vector128<byte>>(s + bytes - 32);
                    System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref System.Runtime.CompilerServices.Unsafe.Add(ref d, 16), c);
                    System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref System.Runtime.CompilerServices.Unsafe.Add(ref d, bytes - 32), e);
                }
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref d, a);
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref System.Runtime.CompilerServices.Unsafe.Add(ref d, bytes - 16), b);
            }
            else if (bytes >= 8)
            {
                ulong a = *(ulong*)s, b = *(ulong*)(s + bytes - 8);
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref d, a);
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref System.Runtime.CompilerServices.Unsafe.Add(ref d, bytes - 8), b);
            }
            else if (bytes >= 4)
            {
                uint a = *(uint*)s, b = *(uint*)(s + bytes - 4);
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref d, a);
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref System.Runtime.CompilerServices.Unsafe.Add(ref d, bytes - 4), b);
            }
            else if (bytes >= 2)
            {
                ushort a = *(ushort*)s, b = *(ushort*)(s + bytes - 2);
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref d, a);
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref System.Runtime.CompilerServices.Unsafe.Add(ref d, bytes - 2), b);
            }
            else
                d = *s;
        }

        /// <summary>
        ///     Faults in every 4 KB page of a fresh result in address order by storing one zero byte per page (and at the
        ///     last byte, whose page the stride can skip).
        /// </summary>
        /// <param name="start">First byte of the result (pinned).</param>
        /// <param name="bytes">Result length in bytes (≥ 1).</param>
        /// <remarks>
        ///     The copy that follows overwrites every byte, so the stores change no result; they only decide the order in
        ///     which the OS maps the pages, and they leave the most recently touched ones in cache.
        /// </remarks>
        private static unsafe void ToArrayTouchPages(byte* start, long bytes)
        {
            byte* end = start + bytes;
            for (byte* q = start; q < end; q += 4096)
                *q = 0;
            end[-1] = 0;
        }

        /// <summary>
        ///     Converts a small view of any layout in C order with a scalar odometer — the innermost axis a plain strided
        ///     loop, the outer axes a carry chain that moves a row pointer — one <see cref="NDIterCasting.ConvertValue"/>
        ///     per element: the scalar converter astype's copy core applies to single values, bit-exact with NumPy over
        ///     the full cast matrix. No iterator is built.
        /// </summary>
        /// <typeparam name="T">Destination element type.</typeparam>
        /// <param name="p">Logical element 0 of the view.</param>
        /// <param name="itemSize">Source element size in bytes.</param>
        /// <param name="dims">The view's dimensions (rank ≥ 1, no zero extent, at least two elements).</param>
        /// <param name="strides">The view's element strides (any sign; 0 for a broadcast axis).</param>
        /// <param name="d">Destination (pinned), with room for every element.</param>
        /// <param name="source">Source dtype (≠ T's).</param>
        /// <remarks>
        ///     The iterator's construction costs about 0.2 µs (measured on a 12-element int16 → float64 conversion), more
        ///     than a thousand-element view's conversions take here. The carry chain ends exactly when the last row has
        ///     been written: the destination cursor reaching the element count is the loop's exit, checked before a
        ///     carry, so the odometer never steps past the outermost axis. The target dtype is read from
        ///     <see cref="InfoOf{T}"/> here rather than passed in: a JIT-time constant of the instantiation, it lets the
        ///     inlined scalar converter's dtype switch fold to one arm (the published frame fact, measured 12.8 ns against
        ///     31–34 ns for a small conversion behind a target parameter).
        /// </remarks>
        private static unsafe void ToArrayGatherConvert<T>(byte* p, int itemSize, long[] dims, long[] strides, T* d, NPTypeCode source) where T : unmanaged
        {
            NPTypeCode target = InfoOf<T>.NPTypeCode;
            int last = dims.Length - 1;
            long cols = dims[last], cs = strides[last] * itemSize;
            long total = 1;
            for (int k = 0; k <= last; k++)
                total *= dims[k];
            T* end = d + total;
            // One coordinate per outer axis; the innermost axis is the row loop and needs none.
            long* coord = stackalloc long[last + 1];
            for (int k = 0; k <= last; k++)
                coord[k] = 0;
            byte* row = p;
            while (true)
            {
                byte* s = row;
                for (long j = 0; j < cols; j++, s += cs)
                    NDIterCasting.ConvertValue(s, d++, source, target);
                if (d == end)
                    return;
                // Carry: advance the innermost outer axis, rewinding every axis that wraps (byte offsets).
                int ax = last - 1;
                while (true)
                {
                    row += strides[ax] * itemSize;
                    if (++coord[ax] < dims[ax])
                        break;
                    row -= strides[ax] * dims[ax] * itemSize;
                    coord[ax] = 0;
                    ax--;
                }
            }
        }

        /// <summary>
        ///     The exact copy (T = dtype) of every non-contiguous layout that is not a single-value broadcast: a
        ///     layout-specific fill engine writing the result in C order, dispatched once on the view's normalized shape.
        /// </summary>
        /// <typeparam name="T">Element type (the dtype).</typeparam>
        /// <param name="p">Logical element 0 of the view.</param>
        /// <param name="dims">The view's dimensions (rank ≥ 1, no zero extent, at least two elements).</param>
        /// <param name="strides">The view's element strides (any sign; 0 for a broadcast axis).</param>
        /// <param name="d">Destination (pinned), with room for every element, written in C order.</param>
        /// <remarks>
        ///     <para>
        ///     The shape is normalized first, as the iterator coalesces it: extent-1 axes are dropped (they add no
        ///     offset) and an axis is merged into its outer neighbour when that neighbour steps exactly over one whole run
        ///     of it, so a stepped <c>[..., ::2]</c> view becomes ONE long strided line and a view that is contiguous in
        ///     all but its flags becomes one block copy. Then three shapes of work, in this order: a 1-D (normalized)
        ///     view is one strided line (<see cref="ToArrayLine{T}"/>); a rank ≥ 3 view of 4/8-byte elements whose two
        ///     innermost axes are both strided but whose unit-stride axis lies further out (a permuted view such as
        ///     <c>np.swapaxes(a, 0, -1)</c>) is transposed over that outer axis (<see cref="ToArrayTransposeOuterAxis{T}"/>
        ///     — measured ≈ 1.6× faster than plane by plane, because every source read is then a contiguous run);
        ///     everything else walks the view plane by plane (the two innermost axes) with an odometer over the outer
        ///     axes (<see cref="ToArrayPlaneFill{T}"/>).
        ///     </para>
        ///     <para>
        ///     A transposing fill writes the result out of address order. When the result is a large object (fresh pages
        ///     that fault in on first touch), those pages are faulted in first, in address order
        ///     (<see cref="ToArrayTouchPages"/>), so the faults do not land inside the transpose and evict its cache
        ///     working set. Sequential fills (row copies, strided rows, reversed lines) skip it: before an in-order writer
        ///     the pre-pass is pure cost (measured +8 %).
        ///     </para>
        ///     <para>
        ///     [NoInlining] keeps the dispatch, its stack arrays and the odometer out of <see cref="ToArray{T}"/>'s frame,
        ///     whose tiny contiguous calls would otherwise pay for their slots and prologue. [SkipLocalsInit]: every stack
        ///     array here is written before it is read — the normalized axes up to the rank found, the odometer's
        ///     coordinates by the explicit zeroing loop — so the runtime's zeroing of each stackalloc (and of the frame's
        ///     locals) on every call observes nothing; a small view (a 20 × 30 transpose, ≈ 84 ns in all) pays it on
        ///     every call.
        ///     </para>
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        [System.Runtime.CompilerServices.SkipLocalsInit]
        private static unsafe void ToArrayDedicated<T>(T* p, long[] dims, long[] strides, T* d) where T : unmanaged
        {
            int rank = dims.Length;
            // The normalized axes (at most `rank` of them), outermost first.
            long* ext = stackalloc long[rank];
            long* str = stackalloc long[rank];
            int nd = 0;
            long total = 1;
            for (int a = 0; a < rank; a++)
            {
                long e = dims[a], s = strides[a];
                total *= e;
                if (e == 1)
                    continue;
                // The kept axis before this one steps over exactly one whole run of this axis: offsets p·S + a·s with
                // S = e·s are (p·e + a)·s, one axis of extent E·e and stride s (also for negative and zero strides).
                if (nd > 0 && str[nd - 1] == s * e)
                {
                    ext[nd - 1] *= e;
                    str[nd - 1] = s;
                }
                else
                {
                    ext[nd] = e;
                    str[nd] = s;
                    nd++;
                }
            }

            // At least two elements, so at least one axis survives; one axis is a single strided line.
            if (nd == 1)
            {
                ToArrayLine(p, ext[0], str[0], d);
                return;
            }

            long bytes = total * sizeof(T);
            bool large = bytes >= ToArrayLargeObjectBytes;
            long rs = str[nd - 2], cs = str[nd - 1];

            // A permuted view: neither innermost axis is unit-stride (a column stride of 0 / ±1 is a fill, a reverse or a
            // row copy, which the plane walk does in address order), but an outer axis is, with at least one 4-row
            // register block of extent. The element width is a JIT-time constant, so every other width compiles this
            // block away.
            if (nd >= 3 && (sizeof(T) == 4 || sizeof(T) == 8) && System.Runtime.Intrinsics.X86.Avx.IsSupported
                && rs != 1 && cs != 0 && cs != 1 && cs != -1)
            {
                for (int u = 0; u < nd - 2; u++)
                {
                    if (str[u] == 1 && ext[u] >= 4)
                    {
                        if (large)
                            ToArrayTouchPages((byte*)d, bytes);
                        ToArrayTransposeOuterAxis(p, ext, str, nd, u, total, d);
                        return;
                    }
                }
            }

            // Plane by plane. The planes land in address order; only a transposing plane fill (|row stride| < |column
            // stride|, the column stride not a fill / reverse / row copy) writes out of order within its plane.
            if (large && cs != 1 && cs != 0 && cs != -1 && Math.Abs(rs) < Math.Abs(cs))
                ToArrayTouchPages((byte*)d, bytes);

            long rows = ext[nd - 2], cols = ext[nd - 1];
            long plane = rows * cols;
            long planes = total / plane;
            int outerNd = nd - 2;
            // One coordinate per outer axis (one spare slot, so a 2-D view never asks for a zero-length block).
            long* idx = stackalloc long[outerNd + 1];
            for (int k = 0; k <= outerNd; k++)
                idx[k] = 0;
            long srcOff = 0;
            T* dp = d;
            for (long o = 0; o < planes; o++, dp += plane)
            {
                ToArrayPlaneFill(p + srcOff, rows, cols, rs, cs, dp);
                // Odometer over the outer axes (last outer axis fastest), tracking the source offset incrementally;
                // after the last plane every axis wraps back and the plane count ends the loop.
                for (int ax = outerNd - 1; ax >= 0; ax--)
                {
                    srcOff += str[ax];
                    if (++idx[ax] < ext[ax])
                        break;
                    srcOff -= str[ax] * ext[ax];
                    idx[ax] = 0;
                }
            }
        }

        /// <summary>
        ///     The exact copy of a rank-2 view (published fact (43)): the rank-2 decisions of <see cref="ToArrayDedicated{T}"/>
        ///     — drop an extent-1 axis, merge the two axes when the row stride steps over one whole row, else one plane,
        ///     pre-faulted when it is a large transposing fill — taken in registers, then straight to the line or plane kernel.
        /// </summary>
        /// <typeparam name="T">Element type (the dtype).</typeparam>
        /// <param name="p">Logical element 0 of the view.</param>
        /// <param name="rows">Row count (≥ 1).</param>
        /// <param name="cols">Column count (≥ 1; rows · cols ≥ 2).</param>
        /// <param name="rs">Row stride in elements (any sign; 0 for a broadcast axis).</param>
        /// <param name="cs">Column stride in elements (any sign; 0 for a broadcast axis).</param>
        /// <param name="d">Destination (pinned), with room for rows · cols elements, written in C order.</param>
        /// <remarks>
        ///     Why a path of its own: the general canonicalizer allocates three stack arrays and walks a plane odometer that
        ///     a rank-2 view never needs; G's trial-30 probe measured o6 (a 20 × 30 transpose) −8 % without them, and the
        ///     harness −6.5 % in INSIGHT 12's window. Every decision is the general canonicalizer's own, so every result is
        ///     identical: an extent-1 axis makes the view one line along the other axis; a row stride equal to the column
        ///     stride × the column count makes it one line of rows · cols elements (also for negative and zero strides); a
        ///     plane whose fill transposes (|row stride| &lt; |column stride|, the column stride not a fill / reverse / row
        ///     copy) into a large-object result has its fresh pages faulted in first, in address order.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static unsafe void ToArrayPlane2D<T>(T* p, long rows, long cols, long rs, long cs, T* d) where T : unmanaged
        {
            if (rows == 1)
            {
                ToArrayLine(p, cols, cs, d);
                return;
            }
            if (cols == 1)
            {
                ToArrayLine(p, rows, rs, d);
                return;
            }
            if (rs == cs * cols)
            {
                ToArrayLine(p, rows * cols, cs, d);
                return;
            }
            long bytes = rows * cols * sizeof(T);
            if (bytes >= ToArrayLargeObjectBytes && cs != 1 && cs != 0 && cs != -1 && Math.Abs(rs) < Math.Abs(cs))
                ToArrayTouchPages((byte*)d, bytes);
            ToArrayPlaneFill(p, rows, cols, rs, cs, d);
        }

        /// <summary>Copies a 1-D strided view (a line) into dense memory, in order.</summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="s">Logical element 0 of the line.</param>
        /// <param name="n">Element count (≥ 1, ≤ <see cref="int.MaxValue"/>).</param>
        /// <param name="stride">Element stride (any sign; 0 repeats one element).</param>
        /// <param name="d">Destination (pinned), with room for <paramref name="n"/> elements.</param>
        /// <remarks>
        ///     Stride 1 is one block copy and stride 0 one fill (reachable when normalization finds a view contiguous or
        ///     broadcast in all but its flags). A reversed line (stride −1) goes through <see cref="ToArrayReverse{T}"/>,
        ///     or, from <see cref="ToArrayStreamPlaneBytes"/> (1 MiB) on, straight into the result by whole lines assembled
        ///     in registers (<see cref="ToArrayReverseStreamed{T}"/>, under AVX2; a full reversal of any rank normalizes to
        ///     one line), else, under AVX alone, through the L1 scratch of <see cref="ToArrayReverseRowsStreamed{T}"/>.
        ///     For 4/8-byte elements under AVX2 a stride of 2 deinterleaves contiguous vector loads
        ///     (<see cref="DeinterleaveRows4"/> / <see cref="DeinterleaveRows8"/>, as one row) and any other stride small
        ///     enough for 32-bit gather indices is one hardware gather per vector (<see cref="GatherRows4"/> /
        ///     <see cref="GatherRows8"/>); everything else is a plain pointer walk. All of these write in address order.
        /// </remarks>
        private static unsafe void ToArrayLine<T>(T* s, long n, long stride, T* d) where T : unmanaged
        {
            if (stride == 1)
            {
                long bytes = n * sizeof(T);
                Buffer.MemoryCopy(s, d, bytes, bytes);
                return;
            }
            if (stride == 0)
            {
                new Span<T>(d, (int)n).Fill(*s);
                return;
            }
            if (stride == -1)
            {
                // A reversed run that large lands past every cache level its result could stay in: stream it — by
                // lines assembled in registers under AVX2 (fact (28)), else through the L1 scratch.
                if (n * sizeof(T) >= ToArrayStreamPlaneBytes && System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                    ToArrayReverseStreamed(s, n, d);
                else if (n * sizeof(T) >= ToArrayStreamPlaneBytes && System.Runtime.Intrinsics.X86.Avx.IsSupported)
                    ToArrayReverseRowsStreamed(s, 1, n, 0, d);
                else
                    ToArrayReverse(s, n, d);
                return;
            }
            if ((sizeof(T) == 4 || sizeof(T) == 8) && System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            {
                if (stride == 2)
                {
                    if (sizeof(T) == 4)
                        DeinterleaveRows4((float*)s, 1, n, 0, (float*)d);
                    else
                        DeinterleaveRows8((double*)s, 1, n, 0, (double*)d);
                    return;
                }
                if (Math.Abs(stride) < int.MaxValue / 8)
                {
                    if (sizeof(T) == 4)
                        GatherRows4((int*)s, 1, n, 0, stride, (int*)d);
                    else
                        GatherRows8((long*)s, 1, n, 0, stride, (long*)d);
                    return;
                }
            }
            for (long i = 0; i < n; i++, s += stride)
                d[i] = *s;
        }

        /// <summary>Copies a reversed run (element i at <c>s - i</c>) into dense memory, in order.</summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="s">The run's logical element 0 (the highest address it reads).</param>
        /// <param name="n">Element count (≥ 1).</param>
        /// <param name="d">Destination, with room for <paramref name="n"/> elements.</param>
        /// <remarks>
        ///     Under AVX2, 4-byte elements move as one vpermps-reversed 8-lane load per 8 results and 8-byte elements as
        ///     one vpermpd-reversed 4-lane load per 4 — lane moves only, so every bit pattern passes through unchanged.
        ///     A block loads <c>s[-i-7] … s[-i]</c> (resp. <c>s[-i-3] … s[-i]</c>), all inside the run while
        ///     <c>i + 8 ≤ n</c> (resp. 4), so nothing outside the view is read; the remainder is element by element.
        ///     Every other width is element by element: for 16-byte elements that measured ≈ 4 % faster than a fused
        ///     32-byte lane-swap reverse (whose loads straddle cache lines at a 16-byte offset), while for 4-byte
        ///     elements the vector reverse measured ≈ 8 % faster than the element loop.
        /// </remarks>
        private static unsafe void ToArrayReverse<T>(T* s, long n, T* d) where T : unmanaged
        {
            long i = 0;
            if (sizeof(T) == 4 && System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            {
                var rev = System.Runtime.Intrinsics.Vector256.Create(7, 6, 5, 4, 3, 2, 1, 0);
                for (; i + 8 <= n; i += 8)
                    System.Runtime.Intrinsics.X86.Avx.Store((float*)(d + i),
                        System.Runtime.Intrinsics.X86.Avx2.PermuteVar8x32(System.Runtime.Intrinsics.X86.Avx.LoadVector256((float*)(s - i - 7)), rev));
            }
            else if (sizeof(T) == 8 && System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            {
                for (; i + 4 <= n; i += 4)
                    System.Runtime.Intrinsics.X86.Avx.Store((double*)(d + i),
                        System.Runtime.Intrinsics.X86.Avx2.Permute4x64(System.Runtime.Intrinsics.X86.Avx.LoadVector256((double*)(s - i - 3)), 0x1B));
            }
            for (; i < n; i++)
                d[i] = s[-i];
        }

        /// <summary>
        ///     Size in bytes of the L1-resident scratch that <see cref="ToArrayReverseRowsStreamed{T}"/> reverses into
        ///     before streaming it out: a whole number of 64-byte lines and of every element size, well inside a 32–48 KB
        ///     L1 data cache next to the source lines the reverse reads, and large enough that the per-chunk bookkeeping
        ///     (one reverse call per row piece, one stream-out call) is spread over 256 lines (the published 16 KB
        ///     finding: a 16 KB chunk measured ≈ 2 % faster than 8 KB on a 4 MB complex reverse). It is also the row length
        ///     from which <see cref="ToArrayPlaneFill{T}"/> streams reversed rows straight from registers instead
        ///     (<see cref="ToArrayReverseStreamed{T}"/>): a row that long fills a chunk by itself, so packing buys it nothing.
        /// </summary>
        private const int ToArrayReverseChunkBytes = 16 * 1024;

        /// <summary>
        ///     Reverses the rows of a plane — row r's element c at <c>s[r·rs − c]</c> — into dense row-major memory through
        ///     an L1-resident scratch, writing the destination by whole 64-byte lines with non-temporal stores: the
        ///     large-reversed-run path of <see cref="ToArrayPlaneFill{T}"/> for rows shorter than a chunk, and of
        ///     <see cref="ToArrayLine{T}"/> / <see cref="ToArrayReverseStreamed{T}"/> where the register path is unavailable
        ///     (no AVX2, or a destination off its element grid).
        /// </summary>
        /// <typeparam name="T">Element type (any width; elements are moved, never interpreted).</typeparam>
        /// <param name="s">Row 0's logical element 0 (the highest address that row reads).</param>
        /// <param name="rows">Row count (≥ 1).</param>
        /// <param name="cols">Row length in elements (≥ 1); also the destination row pitch.</param>
        /// <param name="rs">Source row stride in elements (any sign; ignored when <paramref name="rows"/> is 1).</param>
        /// <param name="d">Destination of row 0, element 0 (pinned; any alignment), with room for rows·cols elements.</param>
        /// <remarks>
        ///     <para>
        ///     Why a scratch: a reversed row's natural writer stores upward while its loads walk downward, and each fresh
        ///     destination line is read for ownership before it is written — traffic the result, past every cache level it
        ///     could stay in, never repays. Streaming the destination needs whole lines, which short rows of arbitrary
        ///     width and phase cannot fill on their own (long rows and whole runs are assembled in registers instead,
        ///     <see cref="ToArrayReverseStreamed{T}"/>); so the rows are reversed (with the ordinary
        ///     <see cref="ToArrayReverse{T}"/>, vector lane moves for 4/8-byte elements) into an L1-resident chunk, and the
        ///     chunk is copied out by <see cref="ToArrayStreamOut"/>, whose whole lines go out as non-temporal stores.
        ///     Measured: a 4 MB complex reverse streamed this way ran 5–9 % faster than the ordinary reverse, while the
        ///     same streaming of a contiguous copy gained under 1 % (the block copy already writes at the memory floor),
        ///     and so did a conversion streamed through such a scratch — its chunks alternate converting and streaming;
        ///     the common conversion pairs are streamed straight from registers instead (<see cref="ToArrayConvertLines{T}"/>).
        ///     </para>
        ///     <para>
        ///     The chunks follow the destination, not the rows: rows are packed back to back (a row longer than the room
        ///     left is split across chunks), so a plane of short rows streams exactly like one long run. The first chunk is
        ///     cut short at the destination's first line boundary when that boundary falls between elements, so every later
        ///     chunk starts on a line and only the run's first and last lines are partial; a 16-byte element run 8 bytes
        ///     off a 16-byte boundary cannot be cut there, and its chunk seams are partial lines on ordinary stores.
        ///     </para>
        ///     <para>
        ///     Non-temporal stores are weakly ordered against every other store: ONE sfence after the last chunk makes all
        ///     of them globally visible before the result is handed out (this thread already sees its own stores in order,
        ///     so the chunks need none between them). The scratch is written before it is read — every chunk's elements
        ///     are reversed into it before the stream-out reads exactly those bytes — so [SkipLocalsInit] only skips a
        ///     16 KB zeroing that no read would observe. [NoInlining]: the stack scratch stays out of every caller's frame.
        ///     Requires AVX (the callers check). Every source read addresses an element of a row; nothing outside the
        ///     destination's rows·cols elements is written.
        ///     </para>
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        [System.Runtime.CompilerServices.SkipLocalsInit]
        private static unsafe void ToArrayReverseRowsStreamed<T>(T* s, long rows, long cols, long rs, T* d) where T : unmanaged
        {
            byte* raw = stackalloc byte[ToArrayReverseChunkBytes + 64];
            // A line-aligned scratch: the stream-out reads it by 32-byte vectors, none of which then splits a line.
            T* scratch = (T*)(((ulong)raw + 63) & ~63UL);
            long chunk = ToArrayReverseChunkBytes / sizeof(T);
            // The first chunk ends at the destination's first line boundary when whole elements reach it, so every
            // later chunk begins on a line; otherwise (16-byte elements 8 bytes off) every chunk is a full one.
            long leadBytes = (long)((64 - ((ulong)d & 63)) & 63);
            long cap = leadBytes != 0 && leadBytes % sizeof(T) == 0 ? leadBytes / sizeof(T) : chunk;
            long filled = 0;
            T* outp = d;
            for (long r = 0; r < rows; r++)
            {
                T* src = s + r * rs;
                long left = cols;
                while (left > 0)
                {
                    // The part of this row that fits the room left in the chunk, reversed onto its end.
                    long m = Math.Min(left, cap - filled);
                    ToArrayReverse(src, m, scratch + filled);
                    src -= m;
                    left -= m;
                    filled += m;
                    if (filled == cap)
                    {
                        ToArrayStreamOut((byte*)scratch, (byte*)outp, filled * sizeof(T));
                        outp += filled;
                        filled = 0;
                        cap = chunk;
                    }
                }
            }

            // The last, partial chunk.
            if (filled > 0)
                ToArrayStreamOut((byte*)scratch, (byte*)outp, filled * sizeof(T));
            // Make the weakly-ordered non-temporal stores visible before anything that could publish the result.
            System.Runtime.Intrinsics.X86.Sse.StoreFence();
        }

        /// <summary>
        ///     Copies <paramref name="bytes"/> bytes with every destination 64-byte line that lies wholly inside the range
        ///     written by two back-to-back non-temporal 32-byte stores, so no fresh destination line is read for ownership;
        ///     issues NO fence (the caller fences once after its last call).
        /// </summary>
        /// <param name="s">Source (any alignment; read with unaligned loads). Must not overlap the destination.</param>
        /// <param name="d">Destination (pinned; any alignment — the line grid is found from it).</param>
        /// <param name="bytes">Byte count (≥ 1).</param>
        /// <remarks>
        ///     The partial lines at both ends — shared with whatever lies before and after the range (the previous or next
        ///     chunk of the same result, the array's header, the next object) — are copied with ordinary stores, so no line
        ///     receives both kinds of store from one call. Requires AVX (the caller checks).
        /// </remarks>
        private static unsafe void ToArrayStreamOut(byte* s, byte* d, long bytes)
        {
            // Bytes before the destination's first line boundary (the whole range when it ends first).
            long head = Math.Min((long)((64 - ((ulong)d & 63)) & 63), bytes);
            Buffer.MemoryCopy(s, d, head, head);
            long i = head;
            // [head, body) is a whole number of lines.
            long body = i + ((bytes - i) & ~63L);
            for (; i < body; i += 64)
            {
                var lo = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + i);
                var hi = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + i + 32);
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(d + i, lo);
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(d + i + 32, hi);
            }

            // The last partial line, on ordinary stores.
            long tail = bytes - i;
            Buffer.MemoryCopy(s + i, d + i, tail, tail);
        }

        /// <summary>
        ///     Writes the reversed run <c>s[0], s[−1], …, s[−(n−1)]</c> straight into the result by whole 64-byte lines:
        ///     per line two 32-byte loads, an in-register lane reverse (<see cref="ToArrayReverseLanes{T}"/>) and two aligned
        ///     non-temporal stores — the published fact (28): lines assembled in registers ran ≈ 11 % faster on a 4 MB
        ///     complex reverse than the same lines copied out of the L1 scratch of <see cref="ToArrayReverseRowsStreamed{T}"/>.
        /// </summary>
        /// <typeparam name="T">Element type (1, 2, 4, 8 or 16 bytes; elements are moved, never interpreted).</typeparam>
        /// <param name="s">The run's logical element 0 (the highest address it reads).</param>
        /// <param name="n">Element count (≥ 1).</param>
        /// <param name="d">Destination (pinned), with room for <paramref name="n"/> elements.</param>
        /// <remarks>
        ///     <para>
        ///     The elements before the destination's first line boundary and after its last one go through
        ///     <see cref="ToArrayReverse{T}"/> on ordinary stores, so the partial lines — shared with whatever lies around
        ///     the run — never receive a non-temporal store. A whole line takes the next 64 / sizeof(T) run elements, whose
        ///     source is the 64 bytes just below the current position, so every load addresses elements of the run only. One
        ///     sfence at the end makes the weakly-ordered stores visible before the result can be published to another thread.
        ///     The source is prefetched one page further down the run than each line reads
        ///     (<see cref="ToArrayWalkPrefetchBytes"/>): the hardware prefetcher does not cross the page boundaries a
        ///     downward walk meets every 64 lines.
        ///     </para>
        ///     <para>
        ///     A 16-byte element whose destination sits 8 bytes off the 16-byte grid — a large-object <c>Complex[]</c>'s data
        ///     usually does — cannot fill whole lines element by element: <see cref="ToArrayReverseStreamedHalf16"/>. A
        ///     destination off its element grid altogether (possible only on a 32-bit runtime) goes through the scratch
        ///     streamer, which takes any phase. Requires AVX2 (the callers check). [NoInlining]: the callers keep their
        ///     frames.
        ///     </para>
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static unsafe void ToArrayReverseStreamed<T>(T* s, long n, T* d) where T : unmanaged
        {
            if (sizeof(T) == 16 && ((ulong)d & 15) == 8)
            {
                ToArrayReverseStreamedHalf16((byte*)s, n, (byte*)d);
                return;
            }
            if (((ulong)d & (ulong)(sizeof(T) - 1)) != 0)
            {
                ToArrayReverseRowsStreamed(s, 1, n, 0, d);
                return;
            }
            long perLine = 64 / sizeof(T), half = 32 / sizeof(T);
            // Elements before the destination's first line boundary (the whole run when it ends first): d is aligned to
            // its element size, so the byte distance divides exactly.
            long lead = Math.Min((long)((64 - ((ulong)d & 63)) & 63) / sizeof(T), n);
            if (lead > 0)
                ToArrayReverse(s, lead, d);
            long i = lead;
            // [lead, body) is a whole number of lines.
            long body = i + ((n - i) & ~(perLine - 1));
            for (; i < body; i += perLine)
            {
                // The source one page further down the run, which this loop reaches 64 lines later (a prefetch never
                // faults, so a page below the run's start is harmless).
                System.Runtime.Intrinsics.X86.Sse.Prefetch0((byte*)(s - i - (perLine - 1)) - ToArrayWalkPrefetchBytes);
                // Run elements i … i+half−1 are the 32 bytes ENDING at element s[−i]; loaded upward, then lane-reversed.
                var lo = System.Runtime.Intrinsics.X86.Avx.LoadVector256((byte*)(s - i - (half - 1)));
                var hi = System.Runtime.Intrinsics.X86.Avx.LoadVector256((byte*)(s - i - (perLine - 1)));
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((byte*)(d + i), ToArrayReverseLanes<T>(lo));
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((byte*)(d + i + half), ToArrayReverseLanes<T>(hi));
            }
            if (i < n)
                ToArrayReverse(s - i, n - i, d + i);
            // Make the weakly-ordered non-temporal stores visible before anything that could publish the result.
            System.Runtime.Intrinsics.X86.Sse.StoreFence();
        }

        /// <summary>Reverses the order of the T elements inside one 32-byte vector, with lane moves only.</summary>
        /// <typeparam name="T">Element type: 1, 2, 4, 8 or 16 bytes.</typeparam>
        /// <param name="v">32 bytes holding 32 / sizeof(T) elements.</param>
        /// <returns>The same elements, last first; every bit pattern passes through unchanged.</returns>
        /// <remarks>
        ///     16-byte: the two 128-bit halves swapped (vpermq 0x4E); 8-byte: vpermq 0x1B; 4-byte: vpermd 7 … 0; 2- and
        ///     1-byte: vpshufb reverses each 128-bit half, vpermq 0x4E then swaps the halves. The width is a JIT-time
        ///     constant, so one arm survives per instantiation. Requires AVX2.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe System.Runtime.Intrinsics.Vector256<byte> ToArrayReverseLanes<T>(System.Runtime.Intrinsics.Vector256<byte> v) where T : unmanaged
        {
            if (sizeof(T) == 16)
                return System.Runtime.Intrinsics.Vector256.AsByte(System.Runtime.Intrinsics.X86.Avx2.Permute4x64(System.Runtime.Intrinsics.Vector256.AsUInt64(v), 0x4E));
            if (sizeof(T) == 8)
                return System.Runtime.Intrinsics.Vector256.AsByte(System.Runtime.Intrinsics.X86.Avx2.Permute4x64(System.Runtime.Intrinsics.Vector256.AsUInt64(v), 0x1B));
            if (sizeof(T) == 4)
                return System.Runtime.Intrinsics.Vector256.AsByte(System.Runtime.Intrinsics.X86.Avx2.PermuteVar8x32(System.Runtime.Intrinsics.Vector256.AsInt32(v),
                    System.Runtime.Intrinsics.Vector256.Create(7, 6, 5, 4, 3, 2, 1, 0)));
            var inHalf = sizeof(T) == 2
                ? System.Runtime.Intrinsics.Vector256.Create((byte)14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1)
                : System.Runtime.Intrinsics.Vector256.Create((byte)15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            return System.Runtime.Intrinsics.Vector256.AsByte(System.Runtime.Intrinsics.X86.Avx2.Permute4x64(
                System.Runtime.Intrinsics.Vector256.AsUInt64(System.Runtime.Intrinsics.X86.Avx2.Shuffle(v, inHalf)), 0x4E));
        }

        /// <summary>
        ///     The 16-byte case of <see cref="ToArrayReverseStreamed{T}"/> for a destination 8 bytes off the 16-byte grid,
        ///     written by WHOLE 64-byte lines: each line boundary then starts at the high half of some element j, so the line
        ///     holds the qwords [hi j, lo j+1, hi j+1, lo j+2, hi j+2, lo j+3, hi j+3, lo j+4], which three overlapping
        ///     32-byte loads, four vpermq and two vpblendd assemble into two aligned 32-byte non-temporal stores. Measured in
        ///     a harness-regime probe on a 4 MB complex reverse: 420–425 µs against 440 µs for the published slot path (one
        ///     16-byte non-temporal store per straddling slot) and 483 µs for the L1 scratch.
        /// </summary>
        /// <param name="s">The run's logical element 0 (the highest address it reads).</param>
        /// <param name="n">Element count (≥ 1).</param>
        /// <param name="d">Destination (pinned; 8 mod 16), with room for <paramref name="n"/> 16-byte elements.</param>
        /// <remarks>
        ///     <para>
        ///     With q0 … q9 the ten qwords from element j+4's low half upward in the source (the run walks down, so element
        ///     j + 4 lies lowest), the line is [q9 q6 q7 q4 | q5 q2 q3 q0]: X = q6…q9, Y = q2…q5 and Z = q0…q3 are loaded,
        ///     vpermq 0x13 takes [3 0 1 0] of X and of Y, vpermq 0x80 / 0x00 bring Y's q4 / Z's q0 into lane 3, and vpblendd
        ///     0xC0 puts them there. Every load addresses elements j … j+4 of the run (the loop runs while j + 4 ≤ n − 1).
        ///     </para>
        ///     <para>
        ///     The bytes before the first line boundary (whole elements, then element j's low half) and after the last line
        ///     (element j's high half, then whole elements) go on ordinary stores, so no line receives both kinds. A run with
        ///     no whole line is copied element by element (no non-temporal store, no fence). One sfence after the lines.
        ///     Each line prefetches the source one page further down the run (<see cref="ToArrayWalkPrefetchBytes"/>).
        ///     Requires AVX2 (the callers check).
        ///     </para>
        /// </remarks>
        private static unsafe void ToArrayReverseStreamedHalf16(byte* s, long n, byte* d)
        {
            // The first line boundary lies 8, 24, 40 or 56 bytes in; element j's high half starts there.
            long lead = (long)((64 - ((ulong)d & 63)) & 63);
            long j = (lead - 8) / 16;
            if (j + 4 > n - 1)
            {
                for (long k = 0; k < n; k++)
                    System.Runtime.Intrinsics.Vector128.Store(System.Runtime.Intrinsics.Vector128.Load(s - 16 * k), d + 16 * k);
                return;
            }
            for (long k = 0; k < j; k++)
                System.Runtime.Intrinsics.Vector128.Store(System.Runtime.Intrinsics.Vector128.Load(s - 16 * k), d + 16 * k);
            // Element j's low half, ending at the line boundary.
            *(ulong*)(d + 16 * j) = *(ulong*)(s - 16 * j);
            for (; j + 4 <= n - 1; j += 4)
            {
                byte* q0 = s - 16 * (j + 4);
                // The source one page further down the run (a prefetch never faults).
                System.Runtime.Intrinsics.X86.Sse.Prefetch0(q0 - ToArrayWalkPrefetchBytes);
                var x = System.Runtime.Intrinsics.X86.Avx.LoadVector256((ulong*)(q0 + 48));
                var y = System.Runtime.Intrinsics.X86.Avx.LoadVector256((ulong*)(q0 + 16));
                var z = System.Runtime.Intrinsics.X86.Avx.LoadVector256((ulong*)q0);
                var lo = System.Runtime.Intrinsics.X86.Avx2.Blend(
                    System.Runtime.Intrinsics.Vector256.AsUInt32(System.Runtime.Intrinsics.X86.Avx2.Permute4x64(x, 0x13)),
                    System.Runtime.Intrinsics.Vector256.AsUInt32(System.Runtime.Intrinsics.X86.Avx2.Permute4x64(y, 0x80)), 0xC0);
                var hi = System.Runtime.Intrinsics.X86.Avx2.Blend(
                    System.Runtime.Intrinsics.Vector256.AsUInt32(System.Runtime.Intrinsics.X86.Avx2.Permute4x64(y, 0x13)),
                    System.Runtime.Intrinsics.Vector256.AsUInt32(System.Runtime.Intrinsics.X86.Avx2.Permute4x64(z, 0x00)), 0xC0);
                byte* o = d + 16 * j + 8;
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((uint*)o, lo);
                System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal((uint*)(o + 32), hi);
            }
            // Element j's high half, then the whole elements after the last line.
            *(ulong*)(d + 16 * j + 8) = *(ulong*)(s - 16 * j + 8);
            for (long k = j + 1; k < n; k++)
                System.Runtime.Intrinsics.Vector128.Store(System.Runtime.Intrinsics.Vector128.Load(s - 16 * k), d + 16 * k);
            // Make the weakly-ordered non-temporal stores visible before anything that could publish the result.
            System.Runtime.Intrinsics.X86.Sse.StoreFence();
        }

        /// <summary>Copies one rows × cols plane of a strided view into dense row-major memory.</summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="s">Plane origin in the source.</param>
        /// <param name="rows">Row count (≥ 1).</param>
        /// <param name="cols">Column count (≥ 1, ≤ <see cref="int.MaxValue"/>; also the destination row stride).</param>
        /// <param name="rs">Source row stride in elements (any sign; 0 repeats a row).</param>
        /// <param name="cs">Source column stride in elements (any sign; 0 repeats an element).</param>
        /// <param name="d">Plane origin in the destination.</param>
        /// <remarks>
        ///     Unit column stride: one block copy per row. Column stride 0 (a broadcast innermost axis): one fill per row.
        ///     Column stride −1 (a reversed innermost axis): <see cref="ToArrayReverse{T}"/> per row, or, for a plane of
        ///     at least <see cref="ToArrayStreamPlaneBytes"/> under AVX, streamed: rows of at least
        ///     <see cref="ToArrayReverseChunkBytes"/> one by one straight from registers (<see cref="ToArrayReverseStreamed{T}"/>,
        ///     under AVX2), shorter rows all through <see cref="ToArrayReverseRowsStreamed{T}"/> — the plane's destination is
        ///     one dense run however short its rows are, so they are packed into the L1 scratch and streamed out by whole
        ///     lines together. A small
        ///     (≤ <see cref="ToArraySmallPlaneBytes"/>) transposed plane of 4/8-byte elements with element-contiguous
        ///     source columns (row stride 1): <see cref="ToArrayRowsTranspose{T}"/>'s register blocks, which also finish
        ///     4–7 leftover rows and 4–7 leftover columns in registers. A large (≥ <see cref="ToArrayStreamPlaneBytes"/>)
        ///     transposed plane of 8-byte elements with element-contiguous source columns streams its destination with
        ///     non-temporal stores behind the caller's pre-fault: by WHOLE 64-byte lines
        ///     (<see cref="ToArrayStreamLines8"/>) when the column count is a multiple of 8 — every row then has the same
        ///     line phase — else, for a multiple of 4, by 32-byte segments (<see cref="ToArrayStreamTranspose8"/>).
        ///     Everything else (other large transposes, strided rows, other widths): <see cref="CopyPlane{T}"/> —
        ///     64-column transposing bands, AVX2 deinterleave / gathers, or the indexed loop.
        /// </remarks>
        private static unsafe void ToArrayPlaneFill<T>(T* s, long rows, long cols, long rs, long cs, T* d) where T : unmanaged
        {
            if (cs == 1)
            {
                long rowBytes = cols * sizeof(T);
                for (long r = 0; r < rows; r++)
                    Buffer.MemoryCopy(s + r * rs, d + r * cols, rowBytes, rowBytes);
                return;
            }
            if (cs == 0)
            {
                for (long r = 0; r < rows; r++)
                    new Span<T>(d + r * cols, (int)cols).Fill(s[r * rs]);
                return;
            }
            if (cs == -1)
            {
                // The plane's rows land back to back, so a large plane is one sequential destination run: stream it.
                if (rows * cols * sizeof(T) >= ToArrayStreamPlaneBytes && System.Runtime.Intrinsics.X86.Avx.IsSupported)
                {
                    // A row at least a scratch chunk long fills whole lines of its own: straight from registers. Shorter
                    // rows would leave a partial line at every row seam, so they are packed through the scratch.
                    if (cols * sizeof(T) >= ToArrayReverseChunkBytes && System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                    {
                        for (long r = 0; r < rows; r++)
                            ToArrayReverseStreamed(s + r * rs, cols, d + r * cols);
                    }
                    else
                        ToArrayReverseRowsStreamed(s, rows, cols, rs, d);
                    return;
                }
                for (long r = 0; r < rows; r++)
                    ToArrayReverse(s + r * rs, cols, d + r * cols);
                return;
            }
            if (rs == 1 && rows >= 4 && (sizeof(T) == 4 || sizeof(T) == 8) && System.Runtime.Intrinsics.X86.Avx.IsSupported
                && rows * cols * sizeof(T) <= ToArraySmallPlaneBytes)
            {
                ToArrayRowsTranspose(s, rows, cols, cs, d, cols);
                return;
            }
            // A LARGE transposed plane of 8-byte elements (row stride 1, so every source column is contiguous): streamed
            // into the destination with non-temporal stores — the caller has already faulted the fresh result in, in
            // address order. The element width is a JIT-time constant, so every other width compiles this test away.
            if (sizeof(T) == 8 && rs == 1 && rows >= 8 && cols >= ToArrayStreamMinCols && (cols & 3) == 0
                && System.Runtime.Intrinsics.X86.Avx.IsSupported && rows * cols * 8 >= ToArrayStreamPlaneBytes
                && ((nuint)d & 7) == 0)
            {
                // A row pitch that is a whole number of 64-byte lines gives every row one line phase, so whole lines can
                // be streamed; a pitch of 4 mod 8 columns alternates the phase row to row and keeps the 32-byte grid.
                if ((cols & 7) == 0)
                    ToArrayStreamLines8((double*)s, rows, cols, cs, (double*)d);
                else
                    ToArrayStreamTranspose8((double*)s, rows, cols, cs, (double*)d);
                return;
            }
            CopyPlane(s, rows, cols, rs, cs, d);
        }

        /// <summary>
        ///     Plane size in bytes from which <see cref="ToArrayPlaneFill{T}"/> streams a transposed plane of 8-byte elements
        ///     (<see cref="ToArrayStreamLines8"/> / <see cref="ToArrayStreamTranspose8"/>): past every cache level the result
        ///     could stay in, the read-for-ownership of each fresh destination line is pure traffic.
        /// </summary>
        private const long ToArrayStreamPlaneBytes = 1L << 20;

        /// <summary>
        ///     Fewest columns for the streamed transposes: every row keeps up to 4 (32-byte grid) or 7 (line grid) columns
        ///     that share a line with the neighbouring row on ordinary stores, a large share of a narrow row, so a narrow
        ///     plane stays on the ordinary bands.
        /// </summary>
        private const long ToArrayStreamMinCols = 32;

        /// <summary>
        ///     Band width, in columns, of <see cref="ToArrayStreamLines8"/>: two whole 64-byte lines per destination row
        ///     per band. A band walks every row of the plane before the next band starts, so its source working set is one
        ///     column run per band column (8 KB each for 1 000 rows): 16 columns keep that at 16 runs — a quarter of what a
        ///     64-column band streams through L1 at once — while each row still receives two back-to-back lines.
        /// </summary>
        private const long ToArrayStreamLineBandColumns = 16;

        /// <summary>
        ///     Transposes a large plane of 8-byte elements whose source columns are contiguous (row stride 1) and whose
        ///     column count is a multiple of 8 into dense row-major memory, writing every destination 64-byte line that
        ///     belongs to one row WHOLE with two back-to-back non-temporal 32-byte stores, so that no fresh destination
        ///     line is ever read for ownership or left half-written in a write-combining buffer.
        /// </summary>
        /// <param name="s">Source of row 0, column 0; row r, column c is <c>s[r + c·cs]</c>.</param>
        /// <param name="rows">Row count (≥ 8).</param>
        /// <param name="cols">Column count: a multiple of 8, at least <see cref="ToArrayStreamMinCols"/>; also the destination row pitch.</param>
        /// <param name="cs">Source column stride in elements (any sign).</param>
        /// <param name="d">Destination of row 0, column 0 (pinned, 8-byte aligned).</param>
        /// <remarks>
        ///     <para>
        ///     The line grid: a pitch of <c>cols·8</c> bytes with cols a multiple of 8 is a whole number of 64-byte lines,
        ///     so every row has row 0's line phase. <c>lead</c> (0–7) columns precede row 0's first line boundary; from
        ///     there, 8-column groups up to <c>end</c> are whole lines owned by one row. The lead columns of row r share a
        ///     line with the tail columns (after <c>end</c>) of row r − 1 (or with bytes before the result for row 0), so
        ///     both go on ordinary stores, as do all columns of the rows after the last whole 8-row strip — no line ever
        ///     receives both kinds of store.
        ///     </para>
        ///     <para>
        ///     The walk: bands of <see cref="ToArrayStreamLineBandColumns"/> columns, each walked down the whole plane in
        ///     8-row strips of two 4×8 register blocks (<see cref="ToArrayStreamBlock4x8"/>). vmovntpd needs 32-byte
        ///     aligned addresses; a line boundary is one. Non-temporal stores are weakly ordered against every other store,
        ///     so the closing sfence makes them globally visible before the result is handed out. Lanes are only moved —
        ///     every 64-bit pattern is copied exactly — and every access addresses an element of the plane.
        ///     </para>
        /// </remarks>
        private static unsafe void ToArrayStreamLines8(double* s, long rows, long cols, long cs, double* d)
        {
            // Columns before row 0's first 64-byte boundary: an 8-byte-aligned destination sits 0–7 elements short of one.
            long lead = (long)(((64 - ((ulong)d & 63)) & 63) >> 3);
            // [lead, end) is a whole number of 8-column line groups; the rest of each row is its tail.
            long end = lead + ((cols - lead) & ~7L);
            long strips = rows & ~7L;
            long o1 = cols, o2 = 2 * cols, o3 = 3 * cols;
            for (long c0 = lead; c0 < end; c0 += ToArrayStreamLineBandColumns)
            {
                long c1 = Math.Min(c0 + ToArrayStreamLineBandColumns, end);
                for (long r0 = 0; r0 < strips; r0 += 8)
                {
                    double* src = s + r0 + c0 * cs;
                    double* top = d + r0 * cols + c0;
                    for (long c = c0; c < c1; c += 8, src += 8 * cs, top += 8)
                    {
                        // Rows r0..r0+3, then rows r0+4..r0+7 (the second half of the same eight column runs).
                        ToArrayStreamBlock4x8(src, cs, top, o1, o2, o3);
                        ToArrayStreamBlock4x8(src + 4, cs, top + 4 * cols, o1, o2, o3);
                    }
                }
            }

            // The strip rows' lead and tail columns — lines shared with the neighbouring row — on ordinary stores.
            for (long r = 0; r < strips; r++)
            {
                double* sr = s + r;
                double* dr = d + r * cols;
                for (long c = 0; c < lead; c++)
                    dr[c] = sr[c * cs];
                for (long c = end; c < cols; c++)
                    dr[c] = sr[c * cs];
            }

            // Rows after the last whole strip, element by element.
            for (long r = strips; r < rows; r++)
            {
                double* sp = s + r;
                double* dr = d + r * cols;
                for (long c = 0; c < cols; c++, sp += cs)
                    dr[c] = *sp;
            }

            // Make the weakly-ordered non-temporal stores visible before anything that could publish the result.
            System.Runtime.Intrinsics.X86.Sse.StoreFence();
        }

        /// <summary>
        ///     Transposes one 4-row × 8-column block of 8-byte elements from contiguous source columns into four whole
        ///     64-byte destination lines, each written by two back-to-back non-temporal stores.
        /// </summary>
        /// <param name="s">Row 0 of the block's first source column (4 contiguous rows; column j at s + j·cs).</param>
        /// <param name="cs">Source column stride in elements.</param>
        /// <param name="a">Destination of the block's row 0 — a 64-byte line boundary (rows 1–3 at +o1 / +o2 / +o3).</param>
        /// <param name="o1">One destination row pitch in elements (a multiple of 8).</param>
        /// <param name="o2">Two destination row pitches.</param>
        /// <param name="o3">Three destination row pitches.</param>
        /// <remarks>
        ///     Columns 0–3 and 4–7 are each a 4×4 transpose (vunpcklpd / vunpckhpd, then vperm2f128 selecting 128-bit
        ///     halves) — lane moves only, so every 64-bit pattern is copied exactly. Row k's line is then its columns 0–3
        ///     (low 32 bytes) followed at once by its columns 4–7 (high 32 bytes), so a line's two halves reach the
        ///     write-combining buffer back to back. The caller guarantees the alignment (a misaligned vmovntpd faults)
        ///     and fences after the last block.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void ToArrayStreamBlock4x8(double* s, long cs, double* a, long o1, long o2, long o3)
        {
            double* s1 = s + cs, s2 = s1 + cs, s3 = s2 + cs, s4 = s3 + cs, s5 = s4 + cs, s6 = s5 + cs, s7 = s6 + cs;
            // Columns 0–3: after the unpacks and the 128-bit selections each vector is one row's low half.
            var v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s);
            var v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s1);
            var v2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s2);
            var v3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s3);
            var t0 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v0, v1);
            var t1 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v0, v1);
            var t2 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v2, v3);
            var t3 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v2, v3);
            // Columns 4–7: each row's high half.
            var w0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s4);
            var w1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s5);
            var w2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s6);
            var w3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s7);
            var u0 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(w0, w1);
            var u1 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(w0, w1);
            var u2 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(w2, w3);
            var u3 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(w2, w3);
            // One whole line per row: low half, then high half.
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x20));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a + 4, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u0, u2, 0x20));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a + o1, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x20));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a + o1 + 4, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u1, u3, 0x20));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a + o2, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x31));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a + o2 + 4, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u0, u2, 0x31));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a + o3, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x31));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a + o3 + 4, System.Runtime.Intrinsics.X86.Avx.Permute2x128(u1, u3, 0x31));
        }

        /// <summary>
        ///     Transposes a large plane of 8-byte elements whose source columns are contiguous (row stride 1) into dense
        ///     row-major memory, writing 32-byte row segments with NON-TEMPORAL stores (vmovntpd) so that no fresh
        ///     destination line is first read for ownership: 64-column bands of 8-row strips of 8×4 register blocks
        ///     (<see cref="ToArrayStreamBlock8x4"/>), the unaligned columns and the rows past the last strip on ordinary
        ///     stores, and one store fence at the end. The dispatcher sends it the column counts of 4 mod 8, whose row
        ///     pitch alternates the 64-byte line phase row to row; a multiple of 8 goes to <see cref="ToArrayStreamLines8"/>.
        /// </summary>
        /// <param name="s">Source of row 0, column 0; row r, column c is <c>s[r + c·cs]</c>.</param>
        /// <param name="rows">Row count (≥ 8).</param>
        /// <param name="cols">Column count: a multiple of 4, at least <see cref="ToArrayStreamMinCols"/>; also the destination row pitch.</param>
        /// <param name="cs">Source column stride in elements (any sign).</param>
        /// <param name="d">Destination of row 0, column 0 (pinned, 8-byte aligned).</param>
        /// <remarks>
        ///     <para>
        ///     vmovntpd needs 32-byte-aligned addresses. An 8-byte-aligned destination reaches a 32-byte boundary after 0–3
        ///     columns (the peel); the row pitch, cols·8 bytes with cols a multiple of 4, is itself a multiple of 32 bytes,
        ///     so every row has that same phase, and from the peel on every 4-column group of every row starts on a
        ///     boundary. The peel, the ≤ 3 columns after the last whole group and the rows after the last whole 8-row strip
        ///     take ordinary stores, after the streamed part.
        ///     </para>
        ///     <para>
        ///     Non-temporal stores are weakly ordered against every other store, so the closing sfence makes them globally
        ///     visible before the result is handed out. Lanes are only moved — every 64-bit pattern is copied exactly — and
        ///     every access addresses an element of the plane.
        ///     </para>
        /// </remarks>
        private static unsafe void ToArrayStreamTranspose8(double* s, long rows, long cols, long cs, double* d)
        {
            const long Band = 64;
            // Columns before row 0's first 32-byte boundary: 0 / 3 / 2 / 1 for a destination 0 / 8 / 16 / 24 bytes past one.
            long peel = (long)((32 - ((ulong)d & 31)) & 31) >> 3;
            // [peel, aligned) is a whole number of 4-column groups; the rest of each row is its tail.
            long aligned = peel + ((cols - peel) & ~3L);
            long strips = rows & ~7L;
            long o1 = cols, o2 = 2 * cols, o3 = 3 * cols;
            for (long c0 = peel; c0 < aligned; c0 += Band)
            {
                long c1 = Math.Min(c0 + Band, aligned);
                for (long r0 = 0; r0 < strips; r0 += 8)
                {
                    double* src = s + r0 + c0 * cs;
                    double* top = d + r0 * cols + c0;
                    double* bottom = top + 4 * cols;
                    for (long c = c0; c < c1; c += 4, src += 4 * cs, top += 4, bottom += 4)
                        ToArrayStreamBlock8x4(src, cs, top, bottom, o1, o2, o3);
                }
            }

            // The strip rows' peel and tail columns, row by row (8 consecutive rows share each source line).
            for (long r = 0; r < strips; r++)
            {
                double* sr = s + r;
                double* dr = d + r * cols;
                for (long c = 0; c < peel; c++)
                    dr[c] = sr[c * cs];
                for (long c = aligned; c < cols; c++)
                    dr[c] = sr[c * cs];
            }

            // Rows after the last whole strip, element by element.
            for (long r = strips; r < rows; r++)
            {
                double* sp = s + r;
                double* dr = d + r * cols;
                for (long c = 0; c < cols; c++, sp += cs)
                    dr[c] = *sp;
            }

            // Make the weakly-ordered non-temporal stores visible before anything that could publish the result.
            System.Runtime.Intrinsics.X86.Sse.StoreFence();
        }

        /// <summary>
        ///     Transposes one 8-row × 4-column block of 8-byte elements from contiguous source columns into eight
        ///     32-byte-aligned destination row segments with non-temporal stores.
        /// </summary>
        /// <param name="s">Row 0 of the block's first source column (8 contiguous rows; column j at s + j·cs).</param>
        /// <param name="cs">Source column stride in elements.</param>
        /// <param name="a">Destination of the block's row 0, 32-byte aligned (rows 1–3 at +o1 / +o2 / +o3).</param>
        /// <param name="b">Destination of the block's row 4, 32-byte aligned (rows 5–7 at +o1 / +o2 / +o3).</param>
        /// <param name="o1">One destination row pitch in elements (a multiple of 4).</param>
        /// <param name="o2">Two destination row pitches.</param>
        /// <param name="o3">Three destination row pitches.</param>
        /// <remarks>
        ///     vunpcklpd / vunpckhpd / vperm2f128 only move lanes, so every 64-bit pattern is copied exactly. The caller
        ///     guarantees the alignment (a misaligned vmovntpd faults) and fences after the last block.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void ToArrayStreamBlock8x4(double* s, long cs, double* a, double* b, long o1, long o2, long o3)
        {
            double* s1 = s + cs, s2 = s1 + cs, s3 = s2 + cs;
            // Rows 0..3 of the four columns: after the unpacks and the 128-bit swaps each vector is one row.
            var v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s);
            var v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s1);
            var v2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s2);
            var v3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s3);
            var t0 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v0, v1);
            var t1 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v0, v1);
            var t2 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v2, v3);
            var t3 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v2, v3);
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x20));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a + o1, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x20));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a + o2, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x31));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(a + o3, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x31));
            // Rows 4..7: the second half of the same four column runs.
            v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 4);
            v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s1 + 4);
            v2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s2 + 4);
            v3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s3 + 4);
            t0 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v0, v1);
            t1 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v0, v1);
            t2 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v2, v3);
            t3 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v2, v3);
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(b, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x20));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(b + o1, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x20));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(b + o2, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x31));
            System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(b + o3, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x31));
        }

        /// <summary>
        ///     Copies a permuted view by transposing it over its unit-stride OUTER axis <paramref name="u"/>: for every
        ///     position of the remaining axes (all but <paramref name="u"/> and the innermost one), the
        ///     extent(u) × extent(innermost) block — contiguous runs along <paramref name="u"/> in the source, contiguous
        ///     runs along the innermost axis in the result — goes through <see cref="ToArrayRowsTranspose{T}"/>'s register
        ///     blocks with the result's pitch for axis <paramref name="u"/>.
        /// </summary>
        /// <typeparam name="T">Element type: 4 or 8 bytes (the caller checked it, and AVX).</typeparam>
        /// <param name="p">Logical element 0 of the view.</param>
        /// <param name="dims">The normalized extents (rank ≥ 3, all ≥ 2).</param>
        /// <param name="strides">The normalized element strides; <c>strides[u] == 1</c>.</param>
        /// <param name="nd">The normalized rank.</param>
        /// <param name="u">The unit-stride outer axis (0 ≤ u ≤ rank − 3, extent ≥ 4).</param>
        /// <param name="total">The view's element count.</param>
        /// <param name="d">Destination (pinned), with room for every element, written in C order.</param>
        /// <remarks>
        ///     Result coordinates are C-order, so axis a advances the destination by the product of the extents after it
        ///     (its pitch); the odometer tracks the source and destination offsets of the remaining axes incrementally.
        ///     Each block's rows are the <paramref name="u"/>-slices (a pitch apart in the result) and its columns the
        ///     innermost axis (adjacent in the result), so the fill is out of address order: the caller pre-faults a
        ///     large result. [SkipLocalsInit]: the pitches, the odometer's axes and its coordinates are all written before
        ///     they are read, so the per-call zeroing of the three stack arrays observes nothing.
        /// </remarks>
        [System.Runtime.CompilerServices.SkipLocalsInit]
        private static unsafe void ToArrayTransposeOuterAxis<T>(T* p, long* dims, long* strides, int nd, int u, long total, T* d) where T : unmanaged
        {
            int last = nd - 1;
            long rows = dims[u], cols = dims[last], cs = strides[last];
            // C-order pitch of every axis: the product of the extents after it.
            long* pitch = stackalloc long[nd];
            pitch[last] = 1;
            for (int a = last - 1; a >= 0; a--)
                pitch[a] = pitch[a + 1] * dims[a + 1];
            // The odometer runs over every axis except u and the innermost one, the fastest axis last.
            int m = nd - 2;
            int* axes = stackalloc int[m];
            long* idx = stackalloc long[m];
            for (int a = 0, k = 0; a < last; a++)
            {
                if (a == u)
                    continue;
                axes[k] = a;
                idx[k] = 0;
                k++;
            }
            long blocks = total / (rows * cols);
            long pu = pitch[u];
            long srcOff = 0, dstOff = 0;
            for (long q = 0; q < blocks; q++)
            {
                ToArrayRowsTranspose(p + srcOff, rows, cols, cs, d + dstOff, pu);
                // After the last block every axis wraps back and the block count ends the loop.
                for (int k = m - 1; k >= 0; k--)
                {
                    int a = axes[k];
                    srcOff += strides[a];
                    dstOff += pitch[a];
                    if (++idx[k] < dims[a])
                        break;
                    srcOff -= strides[a] * dims[a];
                    dstOff -= pitch[a] * dims[a];
                    idx[k] = 0;
                }
            }
        }

        /// <summary>
        ///     Transposes a rows × cols block whose source rows are element-contiguous (row stride 1) and whose columns
        ///     lie <paramref name="cs"/> apart into destination rows <paramref name="pitch"/> apart (columns adjacent),
        ///     with AVX register blocks for 4/8-byte elements.
        /// </summary>
        /// <typeparam name="T">Element type (register blocks for 4/8 bytes under AVX, which the callers check; any other width is copied element by element).</typeparam>
        /// <param name="s">Row 0 of the block's column 0 in the source.</param>
        /// <param name="rows">Row count (≥ 1).</param>
        /// <param name="cols">Column count (≥ 1).</param>
        /// <param name="cs">Source column stride in elements (any sign).</param>
        /// <param name="d">Row 0 of the block in the destination.</param>
        /// <param name="pitch">Destination row pitch in elements.</param>
        /// <remarks>
        ///     <para>
        ///     8-row bands: 8-byte elements as 8×4 blocks (<see cref="TransposeBlock8x4"/>); 4-byte elements as 8×8 blocks
        ///     (<see cref="TransposeBlock8x8"/>), or, for a block of 4–7 columns, 8×4 blocks (<see cref="ToArrayBlock8x4F"/>).
        ///     Then 4-row bands for 4–7 leftover rows (<see cref="ToArrayBlock4x4D"/> / <see cref="ToArrayBlock4x4F"/>).
        ///     </para>
        ///     <para>
        ///     Under AVX the tails are register blocks as well: the last columns of a band that no whole block covers (1–7
        ///     for 4-byte 8-row bands, 1–3 otherwise) are ONE more block ending at the block's last column, and 1–3 rows
        ///     left under the bands (the block at least 4 rows tall) one more 4-row band ending at its last row. Such a
        ///     block starts inside cells already written and rewrites them with the values they already hold — harmless,
        ///     because the destination never overlaps the source — in place of element-by-element tails. Why: o6 (a 20×30
        ///     int32 transpose) had 40 element moves in its tails (two columns of two 8-row bands, two of the 4-row band),
        ///     now 3 register blocks (trial 35's probe, <c>t35/block_probe2.cs</c>: −3 %); the 4–7-column / 4–7-row tails
        ///     as 8×4 / 4×4 blocks had already measured −13 % on the same transpose (P0017). Only a block narrower or
        ///     shorter than 4 (the view itself that small), or a run without AVX, goes element by element.
        ///     </para>
        ///     <para>
        ///     Every block reads whole column runs of its own rows and writes whole row runs of its own columns, so nothing
        ///     outside the block is read or written, and lanes are only moved, so every bit pattern passes through.
        ///     </para>
        /// </remarks>
        private static unsafe void ToArrayRowsTranspose<T>(T* s, long rows, long cols, long cs, T* d, long pitch) where T : unmanaged
        {
            long o1 = pitch, o2 = 2 * pitch, o3 = 3 * pitch;
            long r = 0;
            for (; r + 8 <= rows; r += 8)
            {
                T* sa = s + r;
                T* pa = d + r * pitch;
                T* pb = pa + 4 * pitch;
                long c = 0;
                // The element width is a JIT-time constant: one branch survives per T.
                if (sizeof(T) == 8 && System.Runtime.Intrinsics.X86.Avx.IsSupported)
                {
                    for (; c + 4 <= cols; c += 4, sa += 4 * cs, pa += 4, pb += 4)
                        TransposeBlock8x4((double*)sa, cs, (double*)pa, (double*)pb, o1, o2, o3);
                    // The last 1–3 columns: one more 8×4 block ending at the last column. It starts `back` (−3 … −1)
                    // columns before the first column left, inside the cells the previous block wrote.
                    if (c < cols && cols >= 4)
                    {
                        long back = cols - 4 - c;
                        TransposeBlock8x4((double*)(sa + back * cs), cs, (double*)(pa + back), (double*)(pb + back), o1, o2, o3);
                        c = cols;
                    }
                }
                else if (sizeof(T) == 4 && System.Runtime.Intrinsics.X86.Avx.IsSupported)
                {
                    for (; c + 8 <= cols; c += 8, sa += 8 * cs, pa += 8, pb += 8)
                        TransposeBlock8x8((float*)sa, cs, (float*)pa, (float*)pb, o1, o2, o3);
                    if (c < cols && cols >= 8)
                    {
                        // The last 1–7 columns: one more 8×8 block ending at the last column (`back` is −7 … −1).
                        long back = cols - 8 - c;
                        TransposeBlock8x8((float*)(sa + back * cs), cs, (float*)(pa + back), (float*)(pb + back), o1, o2, o3);
                        c = cols;
                    }
                    else if (c < cols && cols >= 4)
                    {
                        // 4–7 columns in all (no 8×8 block ran, so c is 0): one 8×4 block from column 0 and, for 5–7
                        // columns, one more ending at the last column.
                        ToArrayBlock8x4F((float*)sa, cs, (float*)pa, (float*)pb, o1, o2, o3);
                        long back = cols - 4;
                        if (back > 0)
                            ToArrayBlock8x4F((float*)(sa + back * cs), cs, (float*)(pa + back), (float*)(pb + back), o1, o2, o3);
                        c = cols;
                    }
                }
                // Columns no register block covers (fewer than 4 in all, or no AVX): the column's 8 reads are one run at
                // constant displacements.
                for (; c < cols; c++, sa += cs, pa++, pb++)
                {
                    pa[0] = sa[0]; pa[o1] = sa[1]; pa[o2] = sa[2]; pa[o3] = sa[3];
                    pb[0] = sa[4]; pb[o1] = sa[5]; pb[o2] = sa[6]; pb[o3] = sa[7];
                }
            }
            // 4-row bands: one for 4–7 rows left under the 8-row bands; then, when 1–3 rows are left of a block at least
            // 4 rows tall and the bands are register blocks (a JIT-time constant), one more band ending at the last row,
            // rewriting up to 3 rows with the values they already hold. Each pass moves r forward by at least one row or
            // to the end, so the loop ends.
            while (r < rows && (r + 4 <= rows
                                || (rows >= 4 && (sizeof(T) == 4 || sizeof(T) == 8) && System.Runtime.Intrinsics.X86.Avx.IsSupported)))
            {
                if (r + 4 > rows)
                    r = rows - 4;
                T* sa = s + r;
                T* pa = d + r * pitch;
                long c = 0;
                if (sizeof(T) == 8 && System.Runtime.Intrinsics.X86.Avx.IsSupported)
                {
                    for (; c + 4 <= cols; c += 4, sa += 4 * cs, pa += 4)
                        ToArrayBlock4x4D((double*)sa, cs, (double*)pa, o1, o2, o3);
                    // The last 1–3 columns: one more 4×4 block ending at the last column.
                    if (c < cols && cols >= 4)
                    {
                        long back = cols - 4 - c;
                        ToArrayBlock4x4D((double*)(sa + back * cs), cs, (double*)(pa + back), o1, o2, o3);
                        c = cols;
                    }
                }
                else if (sizeof(T) == 4 && System.Runtime.Intrinsics.X86.Avx.IsSupported)
                {
                    for (; c + 4 <= cols; c += 4, sa += 4 * cs, pa += 4)
                        ToArrayBlock4x4F((float*)sa, cs, (float*)pa, o1, o2, o3);
                    if (c < cols && cols >= 4)
                    {
                        long back = cols - 4 - c;
                        ToArrayBlock4x4F((float*)(sa + back * cs), cs, (float*)(pa + back), o1, o2, o3);
                        c = cols;
                    }
                }
                for (; c < cols; c++, sa += cs, pa++)
                {
                    pa[0] = sa[0]; pa[o1] = sa[1]; pa[o2] = sa[2]; pa[o3] = sa[3];
                }
                r += 4;
            }
            // Rows no band covers (a block fewer than 4 rows tall, or no AVX): element by element.
            for (; r < rows; r++)
            {
                T* sp = s + r;
                T* dr = d + r * pitch;
                for (long c = 0; c < cols; c++, sp += cs)
                    dr[c] = *sp;
            }
        }

        /// <summary>
        ///     Transposes one 8-row × 4-column block of 4-byte elements from element-contiguous source columns into
        ///     four-element destination row segments: one 8-lane load per column, eight 4-lane stores.
        /// </summary>
        /// <param name="s">Row 0 of the block's first source column (8 contiguous rows; column j at s + j·cs).</param>
        /// <param name="cs">Source column stride in elements.</param>
        /// <param name="a">Destination of the block's row 0 (rows 1–3 at +o1 / +o2 / +o3).</param>
        /// <param name="b">Destination of the block's row 4 (rows 5–7 at +o1 / +o2 / +o3).</param>
        /// <param name="o1">One destination row pitch in elements.</param>
        /// <param name="o2">Two destination row pitches.</param>
        /// <param name="o3">Three destination row pitches.</param>
        /// <remarks>
        ///     vunpcklps / vunpckhps interleave the column pairs per 128-bit half; vshufps 0x44 / 0xEE then collect row i
        ///     (columns 0–3) in the low half and row i + 4 in the high half — lane moves only, so every 32-bit pattern
        ///     passes through exactly. Every load and store addresses an element of the block.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void ToArrayBlock8x4F(float* s, long cs, float* a, float* b, long o1, long o2, long o3)
        {
            var v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s);
            var v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + cs);
            var v2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 2 * cs);
            var v3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 3 * cs);
            var t0 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v0, v1);
            var t1 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v0, v1);
            var t2 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v2, v3);
            var t3 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v2, v3);
            // u_i: row i in the low half, row i + 4 in the high half.
            var u0 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t0, t2, 0x44);
            var u1 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t0, t2, 0xEE);
            var u2 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t1, t3, 0x44);
            var u3 = System.Runtime.Intrinsics.X86.Avx.Shuffle(t1, t3, 0xEE);
            System.Runtime.Intrinsics.X86.Sse.Store(a, System.Runtime.Intrinsics.Vector256.GetLower(u0));
            System.Runtime.Intrinsics.X86.Sse.Store(a + o1, System.Runtime.Intrinsics.Vector256.GetLower(u1));
            System.Runtime.Intrinsics.X86.Sse.Store(a + o2, System.Runtime.Intrinsics.Vector256.GetLower(u2));
            System.Runtime.Intrinsics.X86.Sse.Store(a + o3, System.Runtime.Intrinsics.Vector256.GetLower(u3));
            System.Runtime.Intrinsics.X86.Sse.Store(b, System.Runtime.Intrinsics.X86.Avx.ExtractVector128(u0, 1));
            System.Runtime.Intrinsics.X86.Sse.Store(b + o1, System.Runtime.Intrinsics.X86.Avx.ExtractVector128(u1, 1));
            System.Runtime.Intrinsics.X86.Sse.Store(b + o2, System.Runtime.Intrinsics.X86.Avx.ExtractVector128(u2, 1));
            System.Runtime.Intrinsics.X86.Sse.Store(b + o3, System.Runtime.Intrinsics.X86.Avx.ExtractVector128(u3, 1));
        }

        /// <summary>
        ///     Transposes one 4×4 block of 4-byte elements (the SSE unpack / shuffle transpose) from element-contiguous
        ///     source columns into four-element destination row segments.
        /// </summary>
        /// <param name="s">Row 0 of the block's first source column (4 contiguous rows; column j at s + j·cs).</param>
        /// <param name="cs">Source column stride in elements.</param>
        /// <param name="a">Destination of the block's row 0 (rows 1–3 at +o1 / +o2 / +o3).</param>
        /// <param name="o1">One destination row pitch in elements.</param>
        /// <param name="o2">Two destination row pitches.</param>
        /// <param name="o3">Three destination row pitches.</param>
        /// <remarks>
        ///     unpcklps / unpckhps / shufps only move lanes: every 32-bit pattern is copied exactly, and every load and
        ///     store addresses an element of the block.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void ToArrayBlock4x4F(float* s, long cs, float* a, long o1, long o2, long o3)
        {
            var v0 = System.Runtime.Intrinsics.X86.Sse.LoadVector128(s);
            var v1 = System.Runtime.Intrinsics.X86.Sse.LoadVector128(s + cs);
            var v2 = System.Runtime.Intrinsics.X86.Sse.LoadVector128(s + 2 * cs);
            var v3 = System.Runtime.Intrinsics.X86.Sse.LoadVector128(s + 3 * cs);
            var t0 = System.Runtime.Intrinsics.X86.Sse.UnpackLow(v0, v1);
            var t1 = System.Runtime.Intrinsics.X86.Sse.UnpackHigh(v0, v1);
            var t2 = System.Runtime.Intrinsics.X86.Sse.UnpackLow(v2, v3);
            var t3 = System.Runtime.Intrinsics.X86.Sse.UnpackHigh(v2, v3);
            System.Runtime.Intrinsics.X86.Sse.Store(a, System.Runtime.Intrinsics.X86.Sse.Shuffle(t0, t2, 0x44));
            System.Runtime.Intrinsics.X86.Sse.Store(a + o1, System.Runtime.Intrinsics.X86.Sse.Shuffle(t0, t2, 0xEE));
            System.Runtime.Intrinsics.X86.Sse.Store(a + o2, System.Runtime.Intrinsics.X86.Sse.Shuffle(t1, t3, 0x44));
            System.Runtime.Intrinsics.X86.Sse.Store(a + o3, System.Runtime.Intrinsics.X86.Sse.Shuffle(t1, t3, 0xEE));
        }

        /// <summary>
        ///     Transposes one 4×4 block of 8-byte elements (the AVX unpack / 128-bit-swap transpose) from
        ///     element-contiguous source columns into four-element destination row segments.
        /// </summary>
        /// <param name="s">Row 0 of the block's first source column (4 contiguous rows; column j at s + j·cs).</param>
        /// <param name="cs">Source column stride in elements.</param>
        /// <param name="a">Destination of the block's row 0 (rows 1–3 at +o1 / +o2 / +o3).</param>
        /// <param name="o1">One destination row pitch in elements.</param>
        /// <param name="o2">Two destination row pitches.</param>
        /// <param name="o3">Three destination row pitches.</param>
        /// <remarks>
        ///     vunpcklpd / vunpckhpd / vperm2f128 only move lanes, so any 64-bit pattern (NaN payloads, integers) is
        ///     copied exactly although the lanes are typed <see cref="double"/>; every load and store addresses an element
        ///     of the block.
        /// </remarks>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static unsafe void ToArrayBlock4x4D(double* s, long cs, double* a, long o1, long o2, long o3)
        {
            var v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s);
            var v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + cs);
            var v2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 2 * cs);
            var v3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 3 * cs);
            var t0 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v0, v1);
            var t1 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v0, v1);
            var t2 = System.Runtime.Intrinsics.X86.Avx.UnpackLow(v2, v3);
            var t3 = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(v2, v3);
            System.Runtime.Intrinsics.X86.Avx.Store(a, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x20));
            System.Runtime.Intrinsics.X86.Avx.Store(a + o1, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x20));
            System.Runtime.Intrinsics.X86.Avx.Store(a + o2, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t0, t2, 0x31));
            System.Runtime.Intrinsics.X86.Avx.Store(a + o3, System.Runtime.Intrinsics.X86.Avx.Permute2x128(t1, t3, 0x31));
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
        ///     buffer's last element. The final block and any remainder go through the scalar tail. Each block prefetches the
        ///     source one page ahead (<see cref="ToArrayWalkPrefetchBytes"/>) — a hint that reads nothing and never faults.
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
                    // The source one page ahead: a block reads 64 bytes, so every source line is prefetched once.
                    System.Runtime.Intrinsics.X86.Sse.Prefetch0((byte*)(sr + 2 * c) + ToArrayWalkPrefetchBytes);
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
        ///     row's next needed element; the final block and remainder are scalar. Each block prefetches the source one page
        ///     ahead (<see cref="ToArrayWalkPrefetchBytes"/>) — a hint that reads nothing and never faults.
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
                    // The source one page ahead: a block reads 64 bytes, so every source line is prefetched once.
                    System.Runtime.Intrinsics.X86.Sse.Prefetch0((byte*)(sr + 2 * c) + ToArrayWalkPrefetchBytes);
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
