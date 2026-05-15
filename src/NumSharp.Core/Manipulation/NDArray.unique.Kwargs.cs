using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NDArray
    {
        // ============================================================
        //  FLAT (no axis) — sort + mask path (NumPy algorithm)
        //
        //  Pipeline:
        //      1. Allocate keys[n] + perm[n], copy data, init perm = 0..n-1
        //      2. Array.Sort(keys, perm) with NaN-aware comparer for floats
        //      3. Build mask: mask[i] = (i==0) || keys[i] != keys[i-1]
        //      4. For equal_nan=true and float dtypes: collapse trailing NaN run
        //      5. Emit values/index/inverse/counts from mask + keys + perm
        //
        //  Matches numpy/lib/_arraysetops_impl.py::_unique1d exactly.
        // ============================================================

        private NDArray[] uniqueFlatKwargs(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            switch (typecode)
            {
                case NPTypeCode.Boolean: return uniqueFlatSorted<bool>(return_index, return_inverse, return_counts);
                case NPTypeCode.Byte: return uniqueFlatSorted<byte>(return_index, return_inverse, return_counts);
                case NPTypeCode.SByte: return uniqueFlatSorted<sbyte>(return_index, return_inverse, return_counts);
                case NPTypeCode.Int16: return uniqueFlatSorted<short>(return_index, return_inverse, return_counts);
                case NPTypeCode.UInt16: return uniqueFlatSorted<ushort>(return_index, return_inverse, return_counts);
                case NPTypeCode.Int32: return uniqueFlatSorted<int>(return_index, return_inverse, return_counts);
                case NPTypeCode.UInt32: return uniqueFlatSorted<uint>(return_index, return_inverse, return_counts);
                case NPTypeCode.Int64: return uniqueFlatSorted<long>(return_index, return_inverse, return_counts);
                case NPTypeCode.UInt64: return uniqueFlatSorted<ulong>(return_index, return_inverse, return_counts);
                case NPTypeCode.Char: return uniqueFlatSorted<char>(return_index, return_inverse, return_counts);
                case NPTypeCode.Decimal: return uniqueFlatSorted<decimal>(return_index, return_inverse, return_counts);
                case NPTypeCode.Half: return uniqueFlatSortedHalf(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Single: return uniqueFlatSortedFloat(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Double: return uniqueFlatSortedDouble(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Complex: return uniqueFlatSortedComplex(return_index, return_inverse, return_counts, equal_nan);
                default: throw new NotSupportedException();
            }
        }

        // ----- Generic path for non-NaN-capable types -----

        private unsafe NDArray[] uniqueFlatSorted<T>(bool return_index, bool return_inverse, bool return_counts)
            where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            long n = this.size;
            if (n == 0) return BuildEmptyResults<T>(return_index, return_inverse, return_counts);
            CheckSortableLength(n);

            var (keys, perm) = ExtractKeysAndPerm<T>(n);
            // No comparer → uses Comparer<T>.Default which delegates to IComparable<T>.
            // Inlines well in the JIT for primitive types; no delegate dispatch.
            System.Array.Sort(keys, perm);

            return BuildSortedResults<T>(keys, perm, n, return_index, return_inverse, return_counts, firstNaN: -1);
        }

        // ----- NaN-aware float paths -----
        //
        //  Strategy: do an O(n) partition pass to push NaN values to the end of the
        //  array, then call default Array.Sort on the non-NaN prefix only. This
        //  eliminates the Comparer<T> delegate overhead which previously doubled
        //  sort time (custom NaN comparer: ~22 ms / 100K doubles, default sort:
        //  ~11 ms; partition cost: ~0.5 ms; net win is ~2× on float types).

        private unsafe NDArray[] uniqueFlatSortedDouble(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long n = this.size;
            if (n == 0) return BuildEmptyResults<double>(return_index, return_inverse, return_counts);
            CheckSortableLength(n);

            var (keys, perm) = ExtractKeysAndPerm<double>(n);
            long firstNaN = PartitionNaN_Double(keys, perm, n);
            System.Array.Sort(keys, perm, 0, (int)firstNaN);
            StabilizeNaNTail(perm, firstNaN, n);

            return BuildMaskAndEmit<double>(keys, perm, n, firstNaN, equal_nan,
                                             return_index, return_inverse, return_counts);
        }

        private unsafe NDArray[] uniqueFlatSortedFloat(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long n = this.size;
            if (n == 0) return BuildEmptyResults<float>(return_index, return_inverse, return_counts);
            CheckSortableLength(n);

            var (keys, perm) = ExtractKeysAndPerm<float>(n);
            long firstNaN = PartitionNaN_Float(keys, perm, n);
            System.Array.Sort(keys, perm, 0, (int)firstNaN);
            StabilizeNaNTail(perm, firstNaN, n);

            return BuildMaskAndEmit<float>(keys, perm, n, firstNaN, equal_nan,
                                            return_index, return_inverse, return_counts);
        }

        private unsafe NDArray[] uniqueFlatSortedHalf(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long n = this.size;
            if (n == 0) return BuildEmptyResults<Half>(return_index, return_inverse, return_counts);
            CheckSortableLength(n);

            var (keys, perm) = ExtractKeysAndPerm<Half>(n);
            long firstNaN = PartitionNaN_Half(keys, perm, n);
            System.Array.Sort(keys, perm, 0, (int)firstNaN);
            StabilizeNaNTail(perm, firstNaN, n);

            return BuildMaskAndEmit<Half>(keys, perm, n, firstNaN, equal_nan,
                                           return_index, return_inverse, return_counts);
        }

        private unsafe NDArray[] uniqueFlatSortedComplex(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long n = this.size;
            if (n == 0) return BuildEmptyResults<Complex>(return_index, return_inverse, return_counts);
            CheckSortableLength(n);

            var (keys, perm) = ExtractKeysAndPerm<Complex>(n);
            long firstNaN = PartitionNaN_Complex(keys, perm, n);
            // Complex doesn't implement IComparable<Complex>; non-NaN portion needs lex comparer.
            // No NaN-handling inside since partition already moved them out.
            System.Array.Sort(keys, perm, 0, (int)firstNaN,
                Comparer<Complex>.Create((x, y) =>
                {
                    int c = x.Real.CompareTo(y.Real);
                    return c != 0 ? c : x.Imaginary.CompareTo(y.Imaginary);
                }));
            StabilizeNaNTail(perm, firstNaN, n);

            return BuildMaskAndEmit<Complex>(keys, perm, n, firstNaN, equal_nan,
                                              return_index, return_inverse, return_counts);
        }

        /// <summary>
        ///     After unstable partition + sort, the NaN tail's <paramref name="perm"/> entries
        ///     are in arbitrary order. NumPy's stable mergesort path preserves original input
        ///     order for NaN entries; we recover the same semantics by sorting the perm tail
        ///     ascending (the keys in that range are all NaN/NaN-component and order-irrelevant).
        ///     Cost: O(k log k) on the NaN-count, negligible vs the main sort.
        /// </summary>
        private static void StabilizeNaNTail(long[] perm, long firstNaN, long n)
        {
            if (firstNaN >= n - 1) return;
            System.Array.Sort(perm, (int)firstNaN, (int)(n - firstNaN));
        }

        // ----- Partition helpers (NaN to end via two-pointer swap) -----

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long PartitionNaN_Double(double[] keys, long[] perm, long n)
        {
            long hi = n;
            long i = 0;
            while (i < hi)
            {
                if (double.IsNaN(keys[i]))
                {
                    hi--;
                    (keys[i], keys[hi]) = (keys[hi], keys[i]);
                    (perm[i], perm[hi]) = (perm[hi], perm[i]);
                }
                else i++;
            }
            return hi;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long PartitionNaN_Float(float[] keys, long[] perm, long n)
        {
            long hi = n;
            long i = 0;
            while (i < hi)
            {
                if (float.IsNaN(keys[i]))
                {
                    hi--;
                    (keys[i], keys[hi]) = (keys[hi], keys[i]);
                    (perm[i], perm[hi]) = (perm[hi], perm[i]);
                }
                else i++;
            }
            return hi;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long PartitionNaN_Half(Half[] keys, long[] perm, long n)
        {
            long hi = n;
            long i = 0;
            while (i < hi)
            {
                if (Half.IsNaN(keys[i]))
                {
                    hi--;
                    (keys[i], keys[hi]) = (keys[hi], keys[i]);
                    (perm[i], perm[hi]) = (perm[hi], perm[i]);
                }
                else i++;
            }
            return hi;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long PartitionNaN_Complex(Complex[] keys, long[] perm, long n)
        {
            long hi = n;
            long i = 0;
            while (i < hi)
            {
                Complex c = keys[i];
                if (double.IsNaN(c.Real) || double.IsNaN(c.Imaginary))
                {
                    hi--;
                    (keys[i], keys[hi]) = (keys[hi], keys[i]);
                    (perm[i], perm[hi]) = (perm[hi], perm[i]);
                }
                else i++;
            }
            return hi;
        }

        /// <summary>
        ///     Float-aware mask + emit pipeline. After PartitionNaN+Sort, keys[0..firstNaN-1]
        ///     is sorted ascending, keys[firstNaN..n-1] is a contiguous NaN run (arbitrary
        ///     order, but all "equal" under equal_nan=true).
        /// </summary>
        private unsafe NDArray[] BuildMaskAndEmit<T>(
            T[] keys, long[] perm, long n, long firstNaN, bool equal_nan,
            bool return_index, bool return_inverse, bool return_counts) where T : unmanaged
        {
            var mask = new bool[n];
            mask[0] = true;
            long uniqueCount = 1;

            // Mask the non-NaN prefix using IEEE != via .Equals semantics.
            // For Float/Double/Half/Complex: .Equals matches IEEE equality on non-NaN values,
            // so this is equivalent to operator != without dispatching through it generically.
            for (long i = 1; i < firstNaN; i++)
            {
                if (!keys[i].Equals(keys[i - 1])) { mask[i] = true; uniqueCount++; }
            }

            // NaN run starts at firstNaN. mask[firstNaN]=true (transition from non-NaN to NaN, or first elem).
            // For equal_nan=false: every NaN is unique → mask all-true in NaN run.
            // For equal_nan=true: only one NaN representative → mask[firstNaN]=true, rest false.
            if (firstNaN < n)
            {
                mask[firstNaN] = true;
                if (firstNaN > 0) uniqueCount++; // we'd already counted index 0; this is a new transition
                if (equal_nan)
                {
                    // Single NaN representative; nothing else to add
                }
                else
                {
                    for (long i = firstNaN + 1; i < n; i++)
                    {
                        mask[i] = true;
                        uniqueCount++;
                    }
                }
            }

            return EmitOutputs(keys, perm, mask, n, uniqueCount, return_index, return_inverse, return_counts);
        }

        // ----- Helpers -----

        /// <summary>
        ///     The sort+mask kwargs path uses managed T[] arrays, which are bounded by
        ///     <see cref="Array.MaxLength"/> (~2.1B elements). For larger inputs, the
        ///     caller should fall through to the hash-based <see cref="unique()"/> path
        ///     (which uses NumSharp's <c>Hashset&lt;T&gt;</c> with long indexing).
        ///     In practice, n &gt; int.MaxValue requires &gt;16 GB of memory for doubles,
        ///     so this is a theoretical limit, not a practical one.
        /// </summary>
        private static void CheckSortableLength(long n)
        {
            if (n > System.Array.MaxLength)
                throw new NotSupportedException(
                    $"np.unique with return_index/inverse/counts requires the input " +
                    $"to fit in a managed T[] array (max {System.Array.MaxLength} elements). " +
                    $"Got n={n}. Use the no-kwargs np.unique(a) for the hash-based path instead.");
        }

        private unsafe (T[] keys, long[] perm) ExtractKeysAndPerm<T>(long n) where T : unmanaged
        {
            var keys = new T[n];
            var perm = new long[n];

            if (Shape.IsContiguous)
            {
                T* src = (T*)this.Address;
                fixed (T* dst = keys)
                {
                    long byteCount = n * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
                    Buffer.MemoryCopy(src, dst, byteCount, byteCount);
                }
            }
            else
            {
                var flat = this.flat;
                T* src = (T*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < n; i++) keys[i] = src[getOffset(i)];
            }

            for (long i = 0; i < n; i++) perm[i] = i;
            return (keys, perm);
        }

        private static long FindFirstNaN_Double(double[] keys, long n)
        {
            long lo = 0, hi = n - 1;
            while (lo < hi)
            {
                long mid = lo + (hi - lo) / 2;
                if (double.IsNaN(keys[mid])) hi = mid;
                else lo = mid + 1;
            }
            return lo;
        }

        private static long FindFirstNaN_Float(float[] keys, long n)
        {
            long lo = 0, hi = n - 1;
            while (lo < hi)
            {
                long mid = lo + (hi - lo) / 2;
                if (float.IsNaN(keys[mid])) hi = mid;
                else lo = mid + 1;
            }
            return lo;
        }

        /// <summary>
        ///     For non-NaN-capable types: mask[i] = !keys[i].Equals(keys[i-1]) (with mask[0]=true).
        ///     Float/complex paths handle their own mask construction inline using IEEE != to
        ///     preserve equal_nan=false semantics, then call EmitOutputs directly.
        ///
        ///     Index uses min-perm-within-run for first-occurrence semantics (Array.Sort is
        ///     unstable in .NET; we recover stability cheaply in a single pass).
        /// </summary>
        private unsafe NDArray[] BuildSortedResults<T>(
            T[] keys, long[] perm, long n,
            bool return_index, bool return_inverse, bool return_counts,
            long firstNaN) where T : unmanaged, IEquatable<T>
        {
            var mask = new bool[n];
            mask[0] = true;
            long uniqueCount = 1;
            for (long i = 1; i < n; i++)
            {
                if (!keys[i].Equals(keys[i - 1])) { mask[i] = true; uniqueCount++; }
            }

            // BuildSortedResults is only used for non-NaN-capable types (firstNaN always -1).
            // Float/Complex paths use BuildMaskAndEmit instead.
            _ = firstNaN;
            return EmitOutputs(keys, perm, mask, n, uniqueCount, return_index, return_inverse, return_counts);
        }

        private unsafe NDArray[] EmitOutputs<T>(
            T[] keys, long[] perm, bool[] mask, long n, long uniqueCount,
            bool return_index, bool return_inverse, bool return_counts) where T : unmanaged
        {
            int outCount = 1 + (return_index ? 1 : 0) + (return_inverse ? 1 : 0) + (return_counts ? 1 : 0);
            var results = new NDArray[outCount];

            // values
            var valuesBlock = new UnmanagedMemoryBlock<T>(uniqueCount);
            var valuesSlice = new ArraySlice<T>(valuesBlock);
            T* vDst = valuesBlock.Address;
            long vIdx = 0;
            for (long i = 0; i < n; i++)
                if (mask[i]) vDst[vIdx++] = keys[i];
            results[0] = new NDArray(valuesSlice, Shape.Vector(uniqueCount));
            int outPos = 1;

            // index — min(perm) within each run of equal keys (and across collapsed-NaN runs)
            if (return_index)
            {
                var idxBlock = new UnmanagedMemoryBlock<long>(uniqueCount);
                var idxSlice = new ArraySlice<long>(idxBlock);
                long* iDst = idxBlock.Address;
                long oIdx = 0;
                long currentMin = perm[0];
                for (long i = 1; i < n; i++)
                {
                    if (mask[i])
                    {
                        iDst[oIdx++] = currentMin;
                        currentMin = perm[i];
                    }
                    else if (perm[i] < currentMin)
                    {
                        currentMin = perm[i];
                    }
                }
                iDst[oIdx++] = currentMin;
                results[outPos++] = new NDArray(idxSlice, Shape.Vector(uniqueCount));
            }

            // inverse — inv[perm[i]] = cumsum(mask)[i] - 1; reshape to input shape when ndim > 1
            if (return_inverse)
            {
                var invBlock = new UnmanagedMemoryBlock<long>(n);
                var invSlice = new ArraySlice<long>(invBlock);
                long* invDst = invBlock.Address;
                long rank = -1;
                for (long i = 0; i < n; i++)
                {
                    if (mask[i]) rank++;
                    invDst[perm[i]] = rank;
                }
                var invShape = ndim <= 1 ? Shape.Vector(n) : new Shape(Storage.Shape.Dimensions);
                results[outPos++] = new NDArray(invSlice, invShape);
            }

            // counts — distance between consecutive mask=true positions
            if (return_counts)
            {
                var cntBlock = new UnmanagedMemoryBlock<long>(uniqueCount);
                var cntSlice = new ArraySlice<long>(cntBlock);
                long* cDst = cntBlock.Address;
                long prevPos = 0;
                long cIdx = 0;
                for (long i = 1; i < n; i++)
                {
                    if (mask[i])
                    {
                        cDst[cIdx++] = i - prevPos;
                        prevPos = i;
                    }
                }
                cDst[cIdx++] = n - prevPos;
                results[outPos++] = new NDArray(cntSlice, Shape.Vector(uniqueCount));
            }

            return results;
        }

        private NDArray[] BuildEmptyResults<T>(bool return_index, bool return_inverse, bool return_counts) where T : unmanaged
        {
            int outCount = 1 + (return_index ? 1 : 0) + (return_inverse ? 1 : 0) + (return_counts ? 1 : 0);
            var results = new NDArray[outCount];
            results[0] = new NDArray(new ArraySlice<T>(new UnmanagedMemoryBlock<T>(0)), Shape.Vector(0));
            int pos = 1;
            if (return_index)
                results[pos++] = new NDArray(new ArraySlice<long>(new UnmanagedMemoryBlock<long>(0)), Shape.Vector(0));
            if (return_inverse)
                results[pos++] = new NDArray(new ArraySlice<long>(new UnmanagedMemoryBlock<long>(0)), Shape.Vector(0));
            if (return_counts)
                results[pos++] = new NDArray(new ArraySlice<long>(new UnmanagedMemoryBlock<long>(0)), Shape.Vector(0));
            return results;
        }

        // ============================================================
        //  AXIS-AWARE unique (slab comparison — unchanged)
        // ============================================================

        private NDArray[] uniqueAxisKwargs(int axis, bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            var moved = axis == 0 ? this : np.moveaxis(this, axis, 0);
            long n = moved.Shape.Dimensions[0];

            long slabSize = 1;
            for (int d = 1; d < moved.ndim; d++) slabSize *= moved.Shape.Dimensions[d];

            var orig = new int[n];
            for (int i = 0; i < n; i++) orig[i] = i;

            var movedCopy = moved.Shape.IsContiguous ? moved : moved.copy();
            System.Array.Sort(orig, (a, b) => CompareSlabs(movedCopy, a, b, slabSize));

            var sortedKeepIdx = new List<int>();
            var sortedFirstOrig = new List<long>();
            var sortedCounts = new List<long>();
            int prev = -1;
            for (int i = 0; i < n; i++)
            {
                int cur = orig[i];
                if (prev == -1 || !SlabsEqual(movedCopy, prev, cur, slabSize, equal_nan))
                {
                    sortedKeepIdx.Add(cur);
                    sortedFirstOrig.Add(cur);
                    sortedCounts.Add(1);
                }
                else
                {
                    long lastIdx = sortedFirstOrig.Count - 1;
                    if (cur < sortedFirstOrig[(int)lastIdx])
                        sortedFirstOrig[(int)lastIdx] = cur;
                    sortedCounts[(int)lastIdx]++;
                }
                prev = cur;
            }

            int outN = sortedKeepIdx.Count;

            var resultShapeDims = new long[moved.ndim];
            resultShapeDims[0] = outN;
            for (int d = 1; d < moved.ndim; d++)
                resultShapeDims[d] = moved.Shape.Dimensions[d];

            var values = GatherSlabs(movedCopy, sortedKeepIdx.ToArray(), slabSize, new Shape(resultShapeDims));

            if (axis != 0)
                values = np.moveaxis(values, 0, axis);

            var results = new List<NDArray> { values };

            if (return_index)
            {
                var idxBlock = new UnmanagedMemoryBlock<long>(outN);
                var idxSlice = new ArraySlice<long>(idxBlock);
                unsafe
                {
                    long* p = idxBlock.Address;
                    for (int i = 0; i < outN; i++)
                        p[i] = sortedFirstOrig[i];
                }
                results.Add(new NDArray(idxSlice, Shape.Vector(outN)));
            }

            if (return_inverse)
            {
                var invBlock = new UnmanagedMemoryBlock<long>(n);
                var invSlice = new ArraySlice<long>(invBlock);
                unsafe
                {
                    long* p = invBlock.Address;
                    int keptSlot = -1;
                    int prevSorted = -1;
                    for (int i = 0; i < n; i++)
                    {
                        int cur = orig[i];
                        if (prevSorted == -1 || !SlabsEqual(movedCopy, prevSorted, cur, slabSize, equal_nan))
                            keptSlot++;
                        p[cur] = keptSlot;
                        prevSorted = cur;
                    }
                }
                results.Add(new NDArray(invSlice, Shape.Vector(n)));
            }

            if (return_counts)
            {
                var cntBlock = new UnmanagedMemoryBlock<long>(outN);
                var cntSlice = new ArraySlice<long>(cntBlock);
                unsafe
                {
                    long* p = cntBlock.Address;
                    for (int i = 0; i < outN; i++)
                        p[i] = sortedCounts[i];
                }
                results.Add(new NDArray(cntSlice, Shape.Vector(outN)));
            }

            return results.ToArray();
        }

        private static int CompareSlabs(NDArray src, int a, int b, long slabSize)
        {
            switch (src.typecode)
            {
                case NPTypeCode.Boolean: return CompareSlabsT<bool>(src, a, b, slabSize);
                case NPTypeCode.Byte: return CompareSlabsT<byte>(src, a, b, slabSize);
                case NPTypeCode.SByte: return CompareSlabsT<sbyte>(src, a, b, slabSize);
                case NPTypeCode.Int16: return CompareSlabsT<short>(src, a, b, slabSize);
                case NPTypeCode.UInt16: return CompareSlabsT<ushort>(src, a, b, slabSize);
                case NPTypeCode.Int32: return CompareSlabsT<int>(src, a, b, slabSize);
                case NPTypeCode.UInt32: return CompareSlabsT<uint>(src, a, b, slabSize);
                case NPTypeCode.Int64: return CompareSlabsT<long>(src, a, b, slabSize);
                case NPTypeCode.UInt64: return CompareSlabsT<ulong>(src, a, b, slabSize);
                case NPTypeCode.Char: return CompareSlabsT<char>(src, a, b, slabSize);
                case NPTypeCode.Half: return CompareSlabsHalf(src, a, b, slabSize);
                case NPTypeCode.Single: return CompareSlabsFloat(src, a, b, slabSize);
                case NPTypeCode.Double: return CompareSlabsDouble(src, a, b, slabSize);
                case NPTypeCode.Decimal: return CompareSlabsT<decimal>(src, a, b, slabSize);
                case NPTypeCode.Complex: return CompareSlabsComplex(src, a, b, slabSize);
                default: throw new NotSupportedException();
            }
        }

        private static unsafe int CompareSlabsT<T>(NDArray src, int a, int b, long slabSize)
            where T : unmanaged, IComparable<T>
        {
            T* ptr = (T*)src.Address;
            long aBase = a * slabSize;
            long bBase = b * slabSize;
            for (long k = 0; k < slabSize; k++)
            {
                int c = ptr[aBase + k].CompareTo(ptr[bBase + k]);
                if (c != 0) return c;
            }
            return 0;
        }

        private static unsafe int CompareSlabsDouble(NDArray src, int a, int b, long slabSize)
        {
            double* ptr = (double*)src.Address;
            long aBase = a * slabSize;
            long bBase = b * slabSize;
            var cmp = NaNAwareDoubleComparer.Instance;
            for (long k = 0; k < slabSize; k++)
            {
                int c = cmp.Compare(ptr[aBase + k], ptr[bBase + k]);
                if (c != 0) return c;
            }
            return 0;
        }

        private static unsafe int CompareSlabsFloat(NDArray src, int a, int b, long slabSize)
        {
            float* ptr = (float*)src.Address;
            long aBase = a * slabSize;
            long bBase = b * slabSize;
            var cmp = NaNAwareSingleComparer.Instance;
            for (long k = 0; k < slabSize; k++)
            {
                int c = cmp.Compare(ptr[aBase + k], ptr[bBase + k]);
                if (c != 0) return c;
            }
            return 0;
        }

        private static unsafe int CompareSlabsHalf(NDArray src, int a, int b, long slabSize)
        {
            Half* ptr = (Half*)src.Address;
            long aBase = a * slabSize;
            long bBase = b * slabSize;
            for (long k = 0; k < slabSize; k++)
            {
                Half x = ptr[aBase + k], y = ptr[bBase + k];
                int c;
                if (Half.IsNaN(x) && Half.IsNaN(y)) c = 0;
                else if (Half.IsNaN(x)) c = 1;
                else if (Half.IsNaN(y)) c = -1;
                else c = x.CompareTo(y);
                if (c != 0) return c;
            }
            return 0;
        }

        private static unsafe int CompareSlabsComplex(NDArray src, int a, int b, long slabSize)
        {
            Complex* ptr = (Complex*)src.Address;
            long aBase = a * slabSize;
            long bBase = b * slabSize;
            var cmp = NaNAwareComplexComparer.Instance;
            for (long k = 0; k < slabSize; k++)
            {
                int c = cmp.Compare(ptr[aBase + k], ptr[bBase + k]);
                if (c != 0) return c;
            }
            return 0;
        }

        private static unsafe bool SlabsEqual(NDArray src, int a, int b, long slabSize, bool equal_nan)
        {
            switch (src.typecode)
            {
                case NPTypeCode.Boolean: return SlabsEqualT<bool>(src, a, b, slabSize);
                case NPTypeCode.Byte: return SlabsEqualT<byte>(src, a, b, slabSize);
                case NPTypeCode.SByte: return SlabsEqualT<sbyte>(src, a, b, slabSize);
                case NPTypeCode.Int16: return SlabsEqualT<short>(src, a, b, slabSize);
                case NPTypeCode.UInt16: return SlabsEqualT<ushort>(src, a, b, slabSize);
                case NPTypeCode.Int32: return SlabsEqualT<int>(src, a, b, slabSize);
                case NPTypeCode.UInt32: return SlabsEqualT<uint>(src, a, b, slabSize);
                case NPTypeCode.Int64: return SlabsEqualT<long>(src, a, b, slabSize);
                case NPTypeCode.UInt64: return SlabsEqualT<ulong>(src, a, b, slabSize);
                case NPTypeCode.Char: return SlabsEqualT<char>(src, a, b, slabSize);
                case NPTypeCode.Decimal: return SlabsEqualT<decimal>(src, a, b, slabSize);
                case NPTypeCode.Half:
                {
                    Half* ptr = (Half*)src.Address;
                    long aBase = a * slabSize, bBase = b * slabSize;
                    for (long k = 0; k < slabSize; k++)
                    {
                        Half x = ptr[aBase + k], y = ptr[bBase + k];
                        if (equal_nan && Half.IsNaN(x) && Half.IsNaN(y)) continue;
                        if (!x.Equals(y)) return false;
                    }
                    return true;
                }
                case NPTypeCode.Single:
                {
                    float* ptr = (float*)src.Address;
                    long aBase = a * slabSize, bBase = b * slabSize;
                    for (long k = 0; k < slabSize; k++)
                    {
                        float x = ptr[aBase + k], y = ptr[bBase + k];
                        if (equal_nan && float.IsNaN(x) && float.IsNaN(y)) continue;
                        if (!x.Equals(y)) return false;
                    }
                    return true;
                }
                case NPTypeCode.Double:
                {
                    double* ptr = (double*)src.Address;
                    long aBase = a * slabSize, bBase = b * slabSize;
                    for (long k = 0; k < slabSize; k++)
                    {
                        double x = ptr[aBase + k], y = ptr[bBase + k];
                        if (equal_nan && double.IsNaN(x) && double.IsNaN(y)) continue;
                        if (!x.Equals(y)) return false;
                    }
                    return true;
                }
                case NPTypeCode.Complex:
                {
                    Complex* ptr = (Complex*)src.Address;
                    long aBase = a * slabSize, bBase = b * slabSize;
                    for (long k = 0; k < slabSize; k++)
                    {
                        Complex x = ptr[aBase + k], y = ptr[bBase + k];
                        bool xNan = double.IsNaN(x.Real) || double.IsNaN(x.Imaginary);
                        bool yNan = double.IsNaN(y.Real) || double.IsNaN(y.Imaginary);
                        if (equal_nan && xNan && yNan) continue;
                        if (!x.Equals(y)) return false;
                    }
                    return true;
                }
                default: throw new NotSupportedException();
            }
        }

        private static unsafe bool SlabsEqualT<T>(NDArray src, int a, int b, long slabSize)
            where T : unmanaged, IEquatable<T>
        {
            T* ptr = (T*)src.Address;
            long aBase = a * slabSize, bBase = b * slabSize;
            for (long k = 0; k < slabSize; k++)
            {
                if (!ptr[aBase + k].Equals(ptr[bBase + k])) return false;
            }
            return true;
        }

        private static unsafe NDArray GatherSlabs(NDArray src, int[] indices, long slabSize, Shape outShape)
        {
            switch (src.typecode)
            {
                case NPTypeCode.Boolean: return GatherSlabsT<bool>(src, indices, slabSize, outShape);
                case NPTypeCode.Byte: return GatherSlabsT<byte>(src, indices, slabSize, outShape);
                case NPTypeCode.SByte: return GatherSlabsT<sbyte>(src, indices, slabSize, outShape);
                case NPTypeCode.Int16: return GatherSlabsT<short>(src, indices, slabSize, outShape);
                case NPTypeCode.UInt16: return GatherSlabsT<ushort>(src, indices, slabSize, outShape);
                case NPTypeCode.Int32: return GatherSlabsT<int>(src, indices, slabSize, outShape);
                case NPTypeCode.UInt32: return GatherSlabsT<uint>(src, indices, slabSize, outShape);
                case NPTypeCode.Int64: return GatherSlabsT<long>(src, indices, slabSize, outShape);
                case NPTypeCode.UInt64: return GatherSlabsT<ulong>(src, indices, slabSize, outShape);
                case NPTypeCode.Char: return GatherSlabsT<char>(src, indices, slabSize, outShape);
                case NPTypeCode.Half: return GatherSlabsT<Half>(src, indices, slabSize, outShape);
                case NPTypeCode.Single: return GatherSlabsT<float>(src, indices, slabSize, outShape);
                case NPTypeCode.Double: return GatherSlabsT<double>(src, indices, slabSize, outShape);
                case NPTypeCode.Decimal: return GatherSlabsT<decimal>(src, indices, slabSize, outShape);
                case NPTypeCode.Complex: return GatherSlabsT<Complex>(src, indices, slabSize, outShape);
                default: throw new NotSupportedException();
            }
        }

        private static unsafe NDArray GatherSlabsT<T>(NDArray src, int[] indices, long slabSize, Shape outShape)
            where T : unmanaged
        {
            long outN = indices.Length;
            var block = new UnmanagedMemoryBlock<T>(outN * slabSize);
            var slice = new ArraySlice<T>(block);
            T* dst = block.Address;
            T* srcPtr = (T*)src.Address;
            for (long i = 0; i < outN; i++)
            {
                long srcBase = (long)indices[i] * slabSize;
                long dstBase = i * slabSize;
                for (long k = 0; k < slabSize; k++)
                    dst[dstBase + k] = srcPtr[srcBase + k];
            }
            return new NDArray(slice, outShape);
        }
    }
}
