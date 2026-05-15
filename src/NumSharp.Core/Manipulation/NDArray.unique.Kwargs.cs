using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NDArray
    {
        // ============================================================
        //  FLAT (no axis) — keyword-argument unique implementation
        // ============================================================

        private NDArray[] uniqueFlatKwargs(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            switch (typecode)
            {
                case NPTypeCode.Boolean: return uniqueFlatKwargs<bool>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Byte: return uniqueFlatKwargs<byte>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.SByte: return uniqueFlatKwargs<sbyte>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Int16: return uniqueFlatKwargs<short>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.UInt16: return uniqueFlatKwargs<ushort>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Int32: return uniqueFlatKwargs<int>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.UInt32: return uniqueFlatKwargs<uint>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Int64: return uniqueFlatKwargs<long>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.UInt64: return uniqueFlatKwargs<ulong>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Char: return uniqueFlatKwargs<char>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Half: return uniqueFlatKwargsHalf(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Single: return uniqueFlatKwargsFloat(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Double: return uniqueFlatKwargsDouble(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Decimal: return uniqueFlatKwargs<decimal>(return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Complex: return uniqueFlatKwargsComplex(return_index, return_inverse, return_counts, equal_nan);
                default: throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     Generic flat unique with kwargs for non-NaN-capable types (integers, bool, char, decimal).
        ///     <paramref name="equal_nan"/> is irrelevant for these types.
        /// </summary>
        private unsafe NDArray[] uniqueFlatKwargs<T>(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
            where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            long len = this.size;

            // Walk the array, building unique values + first indices + counts
            var dict = new Dictionary<T, long>();     // value → position in parallel lists
            var values = new List<T>();
            var firstIndices = new List<long>();
            var counts = new List<long>();

            // Helper to read i-th element from flattened view
            if (Shape.IsContiguous)
            {
                var src = (T*)this.Address;
                for (long i = 0; i < len; i++)
                {
                    var v = src[i];
                    if (dict.TryGetValue(v, out long entry))
                    {
                        counts[(int)entry]++;
                    }
                    else
                    {
                        dict[v] = values.Count;
                        values.Add(v);
                        firstIndices.Add(i);
                        counts.Add(1);
                    }
                }
            }
            else
            {
                var flat = this.flat;
                var src = (T*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < len; i++)
                {
                    var v = src[getOffset(i)];
                    if (dict.TryGetValue(v, out long entry))
                    {
                        counts[(int)entry]++;
                    }
                    else
                    {
                        dict[v] = values.Count;
                        values.Add(v);
                        firstIndices.Add(i);
                        counts.Add(1);
                    }
                }
            }

            // Sort unique values, build permutation
            int n = values.Count;
            var perm = new int[n];
            for (int i = 0; i < n; i++) perm[i] = i;

            // perm[i] = original index in values[] that becomes position i in sorted output
            // Sort perm by the value at values[perm[i]]
            var valuesArr = values.ToArray();
            System.Array.Sort(perm, (a, b) => valuesArr[a].CompareTo(valuesArr[b]));

            return BuildResults<T>(perm, valuesArr, firstIndices, counts, return_index, return_inverse, return_counts);
        }

        // ----- NaN-aware overloads for floating types -----

        private unsafe NDArray[] uniqueFlatKwargsDouble(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long len = this.size;
            var dict = new Dictionary<double, long>();
            var values = new List<double>();
            var firstIndices = new List<long>();
            var counts = new List<long>();

            Action<double, long> addValue = (v, i) =>
            {
                if (double.IsNaN(v))
                {
                    if (equal_nan && dict.TryGetValue(v, out long e))
                    {
                        counts[(int)e]++;
                        return;
                    }
                    if (equal_nan)
                    {
                        dict[v] = values.Count;
                        values.Add(v);
                        firstIndices.Add(i);
                        counts.Add(1);
                        return;
                    }
                    // equal_nan=false → each NaN is unique, bypass dict
                    values.Add(v);
                    firstIndices.Add(i);
                    counts.Add(1);
                    return;
                }
                if (dict.TryGetValue(v, out long entry))
                {
                    counts[(int)entry]++;
                }
                else
                {
                    dict[v] = values.Count;
                    values.Add(v);
                    firstIndices.Add(i);
                    counts.Add(1);
                }
            };

            if (Shape.IsContiguous)
            {
                var src = (double*)this.Address;
                for (long i = 0; i < len; i++)
                    addValue(src[i], i);
            }
            else
            {
                var flat = this.flat;
                var src = (double*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < len; i++)
                    addValue(src[getOffset(i)], i);
            }

            int n = values.Count;
            var perm = new int[n];
            for (int i = 0; i < n; i++) perm[i] = i;
            var valuesArr = values.ToArray();
            System.Array.Sort(perm, (a, b) => NaNAwareDoubleComparer.Instance.Compare(valuesArr[a], valuesArr[b]));

            return BuildResults<double>(perm, valuesArr, firstIndices, counts, return_index, return_inverse, return_counts);
        }

        private unsafe NDArray[] uniqueFlatKwargsFloat(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long len = this.size;
            var dict = new Dictionary<float, long>();
            var values = new List<float>();
            var firstIndices = new List<long>();
            var counts = new List<long>();

            Action<float, long> addValue = (v, i) =>
            {
                if (float.IsNaN(v))
                {
                    if (equal_nan && dict.TryGetValue(v, out long e))
                    {
                        counts[(int)e]++;
                        return;
                    }
                    if (equal_nan)
                    {
                        dict[v] = values.Count;
                        values.Add(v);
                        firstIndices.Add(i);
                        counts.Add(1);
                        return;
                    }
                    values.Add(v);
                    firstIndices.Add(i);
                    counts.Add(1);
                    return;
                }
                if (dict.TryGetValue(v, out long entry))
                {
                    counts[(int)entry]++;
                }
                else
                {
                    dict[v] = values.Count;
                    values.Add(v);
                    firstIndices.Add(i);
                    counts.Add(1);
                }
            };

            if (Shape.IsContiguous)
            {
                var src = (float*)this.Address;
                for (long i = 0; i < len; i++)
                    addValue(src[i], i);
            }
            else
            {
                var flat = this.flat;
                var src = (float*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < len; i++)
                    addValue(src[getOffset(i)], i);
            }

            int n = values.Count;
            var perm = new int[n];
            for (int i = 0; i < n; i++) perm[i] = i;
            var valuesArr = values.ToArray();
            System.Array.Sort(perm, (a, b) => NaNAwareSingleComparer.Instance.Compare(valuesArr[a], valuesArr[b]));

            return BuildResults<float>(perm, valuesArr, firstIndices, counts, return_index, return_inverse, return_counts);
        }

        private unsafe NDArray[] uniqueFlatKwargsHalf(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long len = this.size;
            var dict = new Dictionary<Half, long>();
            var values = new List<Half>();
            var firstIndices = new List<long>();
            var counts = new List<long>();

            Action<Half, long> addValue = (v, i) =>
            {
                if (Half.IsNaN(v))
                {
                    if (equal_nan && dict.TryGetValue(v, out long e))
                    {
                        counts[(int)e]++;
                        return;
                    }
                    if (equal_nan)
                    {
                        dict[v] = values.Count;
                        values.Add(v);
                        firstIndices.Add(i);
                        counts.Add(1);
                        return;
                    }
                    values.Add(v);
                    firstIndices.Add(i);
                    counts.Add(1);
                    return;
                }
                if (dict.TryGetValue(v, out long entry))
                {
                    counts[(int)entry]++;
                }
                else
                {
                    dict[v] = values.Count;
                    values.Add(v);
                    firstIndices.Add(i);
                    counts.Add(1);
                }
            };

            if (Shape.IsContiguous)
            {
                var src = (Half*)this.Address;
                for (long i = 0; i < len; i++)
                    addValue(src[i], i);
            }
            else
            {
                var flat = this.flat;
                var src = (Half*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < len; i++)
                    addValue(src[getOffset(i)], i);
            }

            int n = values.Count;
            var perm = new int[n];
            for (int i = 0; i < n; i++) perm[i] = i;
            var valuesArr = values.ToArray();
            // Half implements IComparable<Half>; use it for non-NaN. NaN handled separately.
            System.Array.Sort(perm, (a, b) =>
            {
                Half x = valuesArr[a], y = valuesArr[b];
                if (Half.IsNaN(x) && Half.IsNaN(y)) return 0;
                if (Half.IsNaN(x)) return 1;
                if (Half.IsNaN(y)) return -1;
                return x.CompareTo(y);
            });

            return BuildResults<Half>(perm, valuesArr, firstIndices, counts, return_index, return_inverse, return_counts);
        }

        private unsafe NDArray[] uniqueFlatKwargsComplex(bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long len = this.size;
            var dict = new Dictionary<Complex, long>();
            var values = new List<Complex>();
            var firstIndices = new List<long>();
            var counts = new List<long>();

            Action<Complex, long> addValue = (v, i) =>
            {
                bool isNan = double.IsNaN(v.Real) || double.IsNaN(v.Imaginary);
                if (isNan)
                {
                    if (equal_nan && dict.TryGetValue(v, out long e))
                    {
                        counts[(int)e]++;
                        return;
                    }
                    if (equal_nan)
                    {
                        dict[v] = values.Count;
                        values.Add(v);
                        firstIndices.Add(i);
                        counts.Add(1);
                        return;
                    }
                    values.Add(v);
                    firstIndices.Add(i);
                    counts.Add(1);
                    return;
                }
                if (dict.TryGetValue(v, out long entry))
                {
                    counts[(int)entry]++;
                }
                else
                {
                    dict[v] = values.Count;
                    values.Add(v);
                    firstIndices.Add(i);
                    counts.Add(1);
                }
            };

            if (Shape.IsContiguous)
            {
                var src = (Complex*)this.Address;
                for (long i = 0; i < len; i++)
                    addValue(src[i], i);
            }
            else
            {
                var flat = this.flat;
                var src = (Complex*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < len; i++)
                    addValue(src[getOffset(i)], i);
            }

            int n = values.Count;
            var perm = new int[n];
            for (int i = 0; i < n; i++) perm[i] = i;
            var valuesArr = values.ToArray();
            System.Array.Sort(perm, (a, b) => NaNAwareComplexComparer.Instance.Compare(valuesArr[a], valuesArr[b]));

            return BuildResults<Complex>(perm, valuesArr, firstIndices, counts, return_index, return_inverse, return_counts);
        }

        /// <summary>
        ///     Builds the result tuple of [values, index?, inverse?, counts?] from sorted permutation.
        /// </summary>
        private unsafe NDArray[] BuildResults<T>(
            int[] perm,
            T[] valuesArr,
            List<long> firstIndices,
            List<long> counts,
            bool return_index,
            bool return_inverse,
            bool return_counts) where T : unmanaged
        {
            int n = perm.Length;

            // values array (sorted)
            var valuesBlock = new UnmanagedMemoryBlock<T>(n);
            var valuesSlice = new ArraySlice<T>(valuesBlock);
            T* vAddr = valuesBlock.Address;
            for (int i = 0; i < n; i++)
                vAddr[i] = valuesArr[perm[i]];
            var valuesNd = new NDArray(valuesSlice, Shape.Vector(n));

            int outCount = 1
                + (return_index ? 1 : 0)
                + (return_inverse ? 1 : 0)
                + (return_counts ? 1 : 0);
            var results = new NDArray[outCount];
            results[0] = valuesNd;
            int idx = 1;

            // inverse-perm: invPerm[origIndex] = sortedIndex
            int[] invPerm = null;
            if (return_inverse)
            {
                invPerm = new int[n];
                for (int i = 0; i < n; i++) invPerm[perm[i]] = i;
            }

            if (return_index)
            {
                var idxBlock = new UnmanagedMemoryBlock<long>(n);
                var idxSlice = new ArraySlice<long>(idxBlock);
                long* idxAddr = idxBlock.Address;
                for (int i = 0; i < n; i++)
                    idxAddr[i] = firstIndices[perm[i]];
                results[idx++] = new NDArray(idxSlice, Shape.Vector(n));
            }

            if (return_inverse)
            {
                long len = this.size;
                var invBlock = new UnmanagedMemoryBlock<long>(len);
                var invSlice = new ArraySlice<long>(invBlock);
                long* invAddr = invBlock.Address;

                // For each element in original, find its position in sorted unique
                // We need to re-walk the array and map each value back. Use the same dict logic.
                FillInverse<T>(invAddr, len, valuesArr, invPerm);

                // Per NumPy 2.x: return_inverse keeps the original input shape (not flattened)
                var invShape = ndim <= 1 ? Shape.Vector(len) : new Shape(Storage.Shape.Dimensions);
                results[idx++] = new NDArray(invSlice, invShape);
            }

            if (return_counts)
            {
                var cntBlock = new UnmanagedMemoryBlock<long>(n);
                var cntSlice = new ArraySlice<long>(cntBlock);
                long* cAddr = cntBlock.Address;
                for (int i = 0; i < n; i++)
                    cAddr[i] = counts[perm[i]];
                results[idx++] = new NDArray(cntSlice, Shape.Vector(n));
            }

            return results;
        }

        /// <summary>
        ///     Walks the input array and writes the inverse-index for each element into <paramref name="invAddr"/>.
        ///     Uses a dictionary built from <paramref name="valuesArr"/> + <paramref name="invPerm"/>
        ///     so each lookup yields the sorted rank.
        /// </summary>
        private unsafe void FillInverse<T>(long* invAddr, long len, T[] valuesArr, int[] invPerm)
            where T : unmanaged
        {
            // Build a lookup: value → sorted-rank.
            // Use Dictionary which works for all unmanaged equatable types we use here
            // (default equality on float/double/half/complex treats NaN==NaN).
            var lookup = new Dictionary<T, int>(EqualityComparer<T>.Default);
            for (int j = 0; j < valuesArr.Length; j++)
                lookup[valuesArr[j]] = invPerm[j];

            // For equal_nan=false, multiple NaN entries exist in valuesArr but lookup collapses them.
            // We instead walk in original order and assign by encounter index for NaN values.
            // Approach: track per-NaN-occurrence-index via separate counter.
            int nanCounter = 0;
            int[] nanPositions = null; // sorted-rank for each NaN occurrence in original order

            // Pre-compute NaN positions: which entries in valuesArr are NaN, in original order
            bool tIsFloat = typeof(T) == typeof(double) || typeof(T) == typeof(float)
                            || typeof(T) == typeof(Half) || typeof(T) == typeof(Complex);
            if (tIsFloat)
            {
                var nanRanks = new List<int>();
                for (int j = 0; j < valuesArr.Length; j++)
                {
                    if (IsValueNaN(valuesArr[j]))
                        nanRanks.Add(invPerm[j]);
                }
                nanPositions = nanRanks.ToArray();
            }

            if (Shape.IsContiguous)
            {
                var src = (T*)this.Address;
                for (long i = 0; i < len; i++)
                {
                    var v = src[i];
                    if (tIsFloat && IsValueNaN(v))
                    {
                        // For equal_nan=true: dict will have one NaN → use lookup
                        // For equal_nan=false: each NaN gets next slot in nanPositions
                        if (nanPositions.Length == 1)
                        {
                            invAddr[i] = nanPositions[0];
                        }
                        else
                        {
                            invAddr[i] = nanPositions[nanCounter++];
                        }
                    }
                    else
                    {
                        invAddr[i] = lookup[v];
                    }
                }
            }
            else
            {
                var flat = this.flat;
                var src = (T*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < len; i++)
                {
                    var v = src[getOffset(i)];
                    if (tIsFloat && IsValueNaN(v))
                    {
                        if (nanPositions.Length == 1)
                        {
                            invAddr[i] = nanPositions[0];
                        }
                        else
                        {
                            invAddr[i] = nanPositions[nanCounter++];
                        }
                    }
                    else
                    {
                        invAddr[i] = lookup[v];
                    }
                }
            }
        }

        private static bool IsValueNaN<T>(T v) where T : unmanaged
        {
            if (typeof(T) == typeof(double)) return double.IsNaN((double)(object)v);
            if (typeof(T) == typeof(float)) return float.IsNaN((float)(object)v);
            if (typeof(T) == typeof(Half)) return Half.IsNaN((Half)(object)v);
            if (typeof(T) == typeof(Complex))
            {
                var c = (Complex)(object)v;
                return double.IsNaN(c.Real) || double.IsNaN(c.Imaginary);
            }
            return false;
        }

        // ============================================================
        //  AXIS-AWARE unique
        // ============================================================

        /// <summary>
        ///     Unique along an axis: each sub-array along <paramref name="axis"/> is treated as
        ///     a record. Records are compared element-wise (NumPy semantics). Returns sorted
        ///     unique records along that axis, plus optional index/inverse/counts (1-D arrays).
        /// </summary>
        private NDArray[] uniqueAxisKwargs(int axis, bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            // Strategy: bring axis to position 0, then iterate slabs along axis 0
            // and compare element-wise with stable lex order.
            var moved = axis == 0 ? this : np.moveaxis(this, axis, 0);

            long n = moved.Shape.Dimensions[0]; // count along target axis

            // For each i in [0..n): record indices of equal earlier slabs and lex-rank for sorting
            // Slab shape = moved.shape[1..]
            // Element count per slab:
            long slabSize = 1;
            for (int d = 1; d < moved.ndim; d++) slabSize *= moved.Shape.Dimensions[d];

            // Build list of (slab-index, sort-key-array) using copyData; sort by lex.
            // To keep memory bounded we store original index and a Func<long, int> comparator.
            var orig = new int[n];
            for (int i = 0; i < n; i++) orig[i] = i;

            // Sort orig stably by lex comparison of slabs.
            // Use moved as contiguous after moveaxis? moveaxis returns a view (non-contig).
            // Materialize a contiguous copy for predictable slab access.
            var movedCopy = moved.Shape.IsContiguous ? moved : moved.copy();

            // Sort by lex compare on slabs
            System.Array.Sort(orig, (a, b) => CompareSlabs(movedCopy, a, b, slabSize));

            // Walk sorted list, dedup adjacent slabs that compare equal under equal_nan semantics
            var sortedKeepIdx = new List<int>();
            var sortedFirstOrig = new List<long>(); // first orig-index that maps to this kept slab
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
                    // Replace firstOrig if cur is smaller (first occurrence)
                    long lastIdx = sortedFirstOrig.Count - 1;
                    if (cur < sortedFirstOrig[(int)lastIdx])
                        sortedFirstOrig[(int)lastIdx] = cur;
                    sortedCounts[(int)lastIdx]++;
                }
                prev = cur;
            }

            int outN = sortedKeepIdx.Count;

            // Build values: gather slabs by sortedKeepIdx from movedCopy
            // Output shape: same as input but axis dim is outN, with that axis at position 0
            var resultShapeDims = new long[moved.ndim];
            resultShapeDims[0] = outN;
            for (int d = 1; d < moved.ndim; d++)
                resultShapeDims[d] = moved.Shape.Dimensions[d];

            var values = GatherSlabs(movedCopy, sortedKeepIdx.ToArray(), slabSize, new Shape(resultShapeDims));

            // Move axis back to original position
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
                // For each original index 0..n-1, find which sorted-kept slot it maps to.
                var invBlock = new UnmanagedMemoryBlock<long>(n);
                var invSlice = new ArraySlice<long>(invBlock);
                unsafe
                {
                    long* p = invBlock.Address;
                    // Build lookup: orig-index → sorted-kept slot
                    // Walk sorted list; assign each contiguous-equal block the same kept slot
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

        /// <summary>
        ///     Lexicographic compare of two slabs (rows at index <paramref name="a"/> and <paramref name="b"/>
        ///     along axis 0 of <paramref name="src"/>). NaN sorts to end.
        /// </summary>
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

        /// <summary>Element-wise slab equality, with <paramref name="equal_nan"/> semantics for floats.</summary>
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

        /// <summary>Build a contiguous NDArray of the gathered slabs in <paramref name="indices"/> order.</summary>
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
