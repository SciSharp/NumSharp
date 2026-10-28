using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Sorting;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // ============================================================
        //  FLAT (no axis) — sort + mask path (NumPy algorithm)
        //
        //  NumPy's _unique1d (numpy/lib/_arraysetops_impl.py):
        //      if return_index or return_inverse: perm = ar.argsort(kind=...); aux = ar[perm]
        //      else:                              ar.sort(); aux = ar
        //      mask = run starts (aux[1:] != aux[:-1]); equal_nan makes the trailing NaN run one run
        //      ret = aux[mask], perm[mask], inverse (cumsum(mask) - 1 scattered through perm), counts (run lengths)
        //
        //  Routes here:
        //      float16 / float32 / float64 / complex128 with an index or an inverse (and every float16 call, complex128
        //          counts): those statements over NumPy's stable argsort order (uniqueFlatArgsorted).
        //      float32 / float64 values only or counts only: the NaNs partitioned to the tail, the prefix radix-sorted,
        //          NumPy's sort NaN written; complex values only: NumPy's hash path (a high-cardinality line, and above
        //          2^28 elements every line, gets the same answer from the stable argsort).
        //      the integer family, char, bool: the hash paths for values or counts alone (Default.Unique.Hash.cs), a
        //          stable radix argsort of keys and permutation for an index or an inverse; decimal: the BCL sort.
        //      above Array.MaxLength elements: the long-indexed fallback (uniqueFlatSortedLong*).
        // ============================================================

        /// <summary>
        ///     The flat (<c>axis=None</c>, or a 1-D input) numeric unique: dispatches on the operand's dtype to the
        ///     sort + mask pipeline of its family (the integer-family generic path, or the NaN-aware Half / Single /
        ///     Double / Complex paths, which are the only ones <paramref name="equal_nan"/> reaches).
        /// </summary>
        /// <param name="a">The numeric operand (any layout; only read).</param>
        /// <param name="return_index">Also return each unique value's first-occurrence index (flat, C order).</param>
        /// <param name="return_inverse">Also return the indices into the values that rebuild the flattened input.</param>
        /// <param name="return_counts">Also return each unique value's occurrence count.</param>
        /// <param name="equal_nan">For the float and complex dtypes: true collapses every NaN into one value, false keeps
        ///     each NaN distinct. Ignored by the other dtypes, which hold no NaN.</param>
        /// <returns>[values, index?, inverse?, counts?] in NumPy's order: the values sorted in <paramref name="a"/>'s dtype,
        ///     the index outputs int64. The inverse has the input's shape for an N-D input; the caller
        ///     (<see cref="UniqueCompute"/>) reshapes it to the input's shape in every case, as NumPy does.</returns>
        /// <exception cref="NotSupportedException">A dtype outside the 15 numeric ones (a text array reaching this route is a
        ///     bug in the caller, which routes text first).</exception>
        private static NDArray[] uniqueFlatKwargs(NDArray a, bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Boolean: return uniqueFlatSorted<bool>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.Byte: return uniqueFlatSorted<byte>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.SByte: return uniqueFlatSorted<sbyte>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.Int16: return uniqueFlatSorted<short>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.UInt16: return uniqueFlatSorted<ushort>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.Int32: return uniqueFlatSorted<int>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.UInt32: return uniqueFlatSorted<uint>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.Int64: return uniqueFlatSorted<long>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.UInt64: return uniqueFlatSorted<ulong>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.Char: return uniqueFlatSorted<char>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.Decimal: return uniqueFlatSorted<decimal>(a, return_index, return_inverse, return_counts);
                case NPTypeCode.Half: return uniqueFlatSortedHalf(a, return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Single: return uniqueFlatSortedFloat(a, return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Double: return uniqueFlatSortedDouble(a, return_index, return_inverse, return_counts, equal_nan);
                case NPTypeCode.Complex: return uniqueFlatSortedComplex(a, return_index, return_inverse, return_counts, equal_nan);
                default: throw new NotSupportedException();
            }
        }

        // ----- Generic path for non-NaN-capable types -----

        /// <summary>
        ///     The flat unique of a dtype without NaN (bool, the integers, char, decimal): the values-only route
        ///     (<see cref="uniqueValuesOnly{T}"/>), the counts-only hash route, or the keys + permutation sort followed by the
        ///     mask and the outputs. The radix argsort is stable, so the first occurrence of each value comes out of the
        ///     permutation directly; decimal (no monotonic radix key) sorts with the BCL comparer, whose instability the
        ///     emit's min-permutation scan undoes.
        /// </summary>
        /// <typeparam name="T">The operand's element type.</typeparam>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="uniqueFlatKwargs"/>).</returns>
        /// <remarks>
        ///     The counts-only hash route is taken only up to 2^28 elements: the hash path's own large-array bail-out
        ///     re-enters np.unique's flat path, which would come back here. Above <see cref="Array.MaxLength"/> elements
        ///     the long-indexed fallback runs (managed arrays cannot hold the keys).
        /// </remarks>
        private static unsafe NDArray[] uniqueFlatSorted<T>(NDArray a, bool return_index, bool return_inverse, bool return_counts)
            where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            long n = a.size;
            if (n == 0) return BuildEmptyResults<T>(return_index, return_inverse, return_counts);

            // Values-only fast path: skip perm allocation + parallel sort when caller doesn't
            // need index/inverse/counts. Saves ~8MB allocation and ~10ms sort overhead on 1M elems.
            if (!return_index && !return_inverse && !return_counts)
                return new[] { uniqueValuesOnly<T>(a, n) };

            // Counts-only (no index/inverse): the perm is unused (counts are run-lengths of the
            // sorted values). Integer-family → the hash path (fast low/med, radix bail at high card),
            // same routing as np.unique_counts; Decimal keeps the BCL perm path below.
            if (return_counts && !return_index && !return_inverse && typeof(T) != typeof(decimal) && n <= (1L << 28))
            {
                var (cv, cc) = UniqueHashSortedInt<T>(a, wantCounts: true);
                return new[] { cv, cc };
            }

            if (!IsManagedSortableLength(n))
                return uniqueFlatSortedLong<T>(a, n, return_index, return_inverse, return_counts, firstNaN: -1);

            var (keys, perm) = ExtractKeysAndPerm<T>(a, n);
            // Radix argsort (LSD, comparison-free) replaces the BCL introsort for every radix-able
            // dtype; Decimal (no monotonic key) keeps Comparer<T>.Default. Radix is stable so ties
            // resolve ascending-index (first occurrence) — which the min-perm-within-run emit also
            // recovers, so the outputs are byte-identical to the BCL path.
            if (typeof(T) != typeof(decimal)) RadixArgSortKeysPerm(keys, perm, (int)n);
            else System.Array.Sort(keys, perm);

            return BuildSortedResults<T>(a, keys, perm, n, return_index, return_inverse, return_counts, firstNaN: -1);
        }

        /// <summary>
        ///     Values-only path for non-float types: sort + dedup-emit, no perm tracking. The integer family (bool and
        ///     char included) takes the open-addressing hash path up to 2^28 elements (it bails to a radix sort itself at
        ///     high cardinality); decimal sorts with the BCL comparer, because its bit patterns are not a bijection with
        ///     its values (<c>1.0m</c> and <c>1.00m</c> are equal).
        /// </summary>
        /// <typeparam name="T">The operand's element type.</typeparam>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="n">Its element count (positive).</param>
        /// <returns>The sorted distinct values, a fresh 1-D array of <paramref name="a"/>'s dtype.</returns>
        private static unsafe NDArray uniqueValuesOnly<T>(NDArray a, long n)
            where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            // Integer-family values-only: the open-addressing hash path (Default.Unique.Hash.cs)
            // beats any sort at low/medium cardinality and bails to radix at high cardinality —
            // mirroring NumPy's own hash selection for these dtypes, but bailing to radix where
            // NumPy's std::unordered_set thrashes (int32 2M-distinct: NumPy 1540ms → us ~40ms). Its
            // own n>2^28 guard falls back to the sort path, so route there only within that bound
            // (else the fallback would re-enter here and recurse). Decimal has no monotonic radix key
            // and no bitwise-hash (1.0m != 1.00m by bits) → BCL sort.
            if (typeof(T) != typeof(decimal) && n <= (1L << 28))
                return UniqueHashSortedInt<T>(a, wantCounts: false).values;

            if (!IsManagedSortableLength(n))
                return uniqueFlatSortedLong<T>(a, n, false, false, false, firstNaN: -1)[0];

            var keys = ExtractKeysOnly<T>(a, n);
            if (typeof(T) != typeof(decimal)) RadixSortValues(keys, (int)n);
            else System.Array.Sort(keys);
            return EmitValuesOnly<T>(keys, n);
        }

        // ----- NaN-aware float paths -----
        //
        //  Strategy (float32 / float64 values only, counts only): do an O(n) partition pass to push NaN values
        //  to the end of the array, then sort the non-NaN prefix only, with the radix (RadixSortValues,
        //  Default.Unique.Radix.cs; it was the default Array.Sort, and before that a custom NaN comparer). The
        //  partition eliminated the Comparer<T> delegate overhead which had doubled
        //  sort time (custom NaN comparer: ~22 ms / 100K doubles, default sort:
        //  ~11 ms; partition cost: ~0.5 ms; net win is ~2× on float types).
        //  An index or an inverse takes uniqueFlatArgsorted instead (NumPy's argsort and gather), and so does every
        //  float16 call (its stable radix argsort is faster than a comparison sort of float16 keys).

        /// <summary>
        ///     The flat unique of a float64 operand, by NumPy's two routes: a values-only or counts-only call is NumPy's
        ///     in-place sort (the NaNs partitioned to the tail and written as NumPy's sort NaN, the non-NaN prefix
        ///     radix-sorted, no permutation); a call with an index or an inverse is NumPy's argsort and gather
        ///     (<see cref="uniqueFlatArgsorted{T}"/>).
        /// </summary>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="equal_nan">True: every NaN is one value (with an index or an inverse the first NaN in input
        ///     order, its bits kept; on the values-only and counts-only routes NumPy's sort NaN,
        ///     <see cref="WriteSortNaNs(double[], long, long)"/>); false: each NaN is a value of its own.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="uniqueFlatKwargs"/>).</returns>
        /// <remarks>
        ///     On the values-only and counts-only routes the radix key orders <c>-0.0</c> below <c>+0.0</c>, so a run of
        ///     zeros starts with <c>-0.0</c> whenever the input holds one, and that is the zero the values keep (the run
        ///     itself is one value: the mask compares with IEEE equality). NumPy's in-place sort orders equal keys its own
        ///     way and may keep <c>+0.0</c>. With an index or an inverse the values keep the zero NumPy's stable mergesort
        ///     keeps, the first in input order (NumPy's quicksort argsort, an inverse without an index, may keep the other).
        /// </remarks>
        private static unsafe NDArray[] uniqueFlatSortedDouble(NDArray a, bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long n = a.size;
            if (n == 0) return BuildEmptyResults<double>(return_index, return_inverse, return_counts);

            // Values-only fast path: NumPy's in-place sort needs no permutation (no argsort, no gather)
            // when the caller doesn't need index/inverse/counts. Saves ~8MB allocation and ~10ms sort
            // overhead on 1M doubles. The mask/emit step also simplifies — no min-perm tracking.
            if (!return_index && !return_inverse && !return_counts)
                return new[] { uniqueValuesOnlyDouble(a, n, equal_nan) };

            // Counts-only: skip the perm argsort — counts are run-lengths of the sorted values, which
            // the emit derives from the mask alone (the perm is only read for index/inverse).
            if (return_counts && !return_index && !return_inverse && IsManagedSortableLength(n))
            {
                var kc = ExtractKeysOnly<double>(a, n);
                long fn = PartitionNaN_DoubleKeysOnly(kc, n);
                RadixSortValues(kc, (int)fn);
                WriteSortNaNs(kc, fn, n);   // NumPy's in-place ar.sort() rewrites every NaN (see WriteSortNaNs)
                return EmitValuesAndCountsFloat<double>(kc, n, fn, equal_nan);
            }

            if (!IsManagedSortableLength(n))
                return uniqueFlatSortedLongFloat<double>(a, n, equal_nan, return_index, return_inverse, return_counts);

            // An index or an inverse: NumPy's argsort-and-gather statements (uniqueFlatArgsorted).
            return uniqueFlatArgsorted<double>(a, n, return_index, return_inverse, return_counts, equal_nan);
        }

        /// <summary>
        /// Values-only optimized path: extract → partition NaN → sort prefix → dedup-emit.
        /// No perm array (skips ~8MB alloc + ~10ms parallel-sort overhead on 1M doubles).
        /// </summary>
        /// <param name="a">The float64 operand (any layout; only read).</param>
        /// <param name="n">Its element count (positive).</param>
        /// <param name="equal_nan">True keeps one NaN, false every NaN.</param>
        /// <returns>The sorted distinct values (NaNs last), a fresh 1-D float64 array. With two or more elements every NaN
        ///     is NumPy's sort NaN <c>0x7FF8000000000000</c> (NumPy sorts the array in place first).</returns>
        private static unsafe NDArray uniqueValuesOnlyDouble(NDArray a, long n, bool equal_nan)
        {
            if (!IsManagedSortableLength(n))
            {
                // Fall through to long path; it does its own values-only optimization via flags.
                return uniqueFlatSortedLongFloat<double>(a, n, equal_nan, false, false, false)[0];
            }

            var keys = ExtractKeysOnly<double>(a, n);
            long firstNaN = PartitionNaN_DoubleKeysOnly(keys, n);
            RadixSortValues(keys, (int)firstNaN);   // radix on the non-NaN prefix (NaN stays at the tail)
            WriteSortNaNs(keys, firstNaN, n);       // NumPy's in-place ar.sort() rewrites every NaN (see WriteSortNaNs)
            return EmitValuesOnlyFloat<double>(keys, n, firstNaN, equal_nan);
        }

        /// <summary>
        ///     The flat unique of a float32 operand: <see cref="uniqueFlatSortedDouble"/>'s routes at single precision
        ///     (NumPy's in-place sort for values or counts alone, NumPy's argsort and gather for an index or an inverse).
        /// </summary>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="equal_nan">True: every NaN is one value; false: each NaN is a value of its own.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="uniqueFlatKwargs"/>).</returns>
        private static unsafe NDArray[] uniqueFlatSortedFloat(NDArray a, bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long n = a.size;
            if (n == 0) return BuildEmptyResults<float>(return_index, return_inverse, return_counts);

            if (!return_index && !return_inverse && !return_counts)
                return new[] { uniqueValuesOnlyFloat(a, n, equal_nan) };

            // Counts-only: skip the perm argsort (see uniqueFlatSortedDouble).
            if (return_counts && !return_index && !return_inverse && IsManagedSortableLength(n))
            {
                var kc = ExtractKeysOnly<float>(a, n);
                long fn = PartitionNaN_FloatKeysOnly(kc, n);
                RadixSortValues(kc, (int)fn);
                WriteSortNaNs(kc, fn, n);   // NumPy's in-place ar.sort() rewrites every NaN (see WriteSortNaNs)
                return EmitValuesAndCountsFloat<float>(kc, n, fn, equal_nan);
            }

            if (!IsManagedSortableLength(n))
                return uniqueFlatSortedLongFloat<float>(a, n, equal_nan, return_index, return_inverse, return_counts);

            // An index or an inverse: NumPy's argsort-and-gather statements (uniqueFlatArgsorted).
            return uniqueFlatArgsorted<float>(a, n, return_index, return_inverse, return_counts, equal_nan);
        }

        /// <summary>
        ///     Values-only float32 path: extract, partition the NaNs to the tail, radix-sort the prefix, dedup-emit; no
        ///     permutation.
        /// </summary>
        /// <param name="a">The float32 operand (any layout; only read).</param>
        /// <param name="n">Its element count (positive).</param>
        /// <param name="equal_nan">True keeps one NaN, false every NaN.</param>
        /// <returns>The sorted distinct values (NaNs last), a fresh 1-D float32 array. With two or more elements every NaN
        ///     is NumPy's sort NaN <c>0x7FC00000</c> (NumPy sorts the array in place first).</returns>
        private static unsafe NDArray uniqueValuesOnlyFloat(NDArray a, long n, bool equal_nan)
        {
            if (!IsManagedSortableLength(n))
                return uniqueFlatSortedLongFloat<float>(a, n, equal_nan, false, false, false)[0];

            var keys = ExtractKeysOnly<float>(a, n);
            long firstNaN = PartitionNaN_FloatKeysOnly(keys, n);
            RadixSortValues(keys, (int)firstNaN);   // radix on the non-NaN prefix (NaN stays at the tail)
            WriteSortNaNs(keys, firstNaN, n);       // NumPy's in-place ar.sort() rewrites every NaN (see WriteSortNaNs)
            return EmitValuesOnlyFloat<float>(keys, n, firstNaN, equal_nan);
        }

        /// <summary>
        ///     The flat unique of a float16 operand: every call is NumPy's argsort and gather
        ///     (<see cref="uniqueFlatArgsorted{T}"/>), over the stable order; for values or counts alone that is the answer
        ///     of NumPy's in-place float16 sort where that sort keeps equal values in input order (see
        ///     <see cref="uniqueValuesOnlyHalf"/>).
        /// </summary>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="equal_nan">True: every NaN is one value; false: each NaN is a value of its own.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="uniqueFlatKwargs"/>).</returns>
        private static unsafe NDArray[] uniqueFlatSortedHalf(NDArray a, bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long n = a.size;
            if (n == 0) return BuildEmptyResults<Half>(return_index, return_inverse, return_counts);

            if (!return_index && !return_inverse && !return_counts)
                return new[] { uniqueValuesOnlyHalf(a, n, equal_nan) };

            if (!IsManagedSortableLength(n))
                return uniqueFlatSortedLongFloat<Half>(a, n, equal_nan, return_index, return_inverse, return_counts);

            // An index, an inverse or counts: NumPy's argsort-and-gather statements (uniqueFlatArgsorted). NumPy sorts in
            // place for counts alone; its float16 sort keeps the bits, so only the order of equal values could differ.
            return uniqueFlatArgsorted<Half>(a, n, return_index, return_inverse, return_counts, equal_nan);
        }

        /// <summary>
        ///     Values-only float16 path: the argsort statements without outputs (<see cref="uniqueFlatArgsorted{T}"/>). NumPy
        ///     runs <c>ar.sort()</c> here, and on a host without AVX-512 its float16 sort is its scalar quicksort: the bits
        ///     kept, and below 17 elements an insertion sort that keeps equal values in input order, which is the stable
        ///     order's answer (the first zero of a run of zeros, the first NaN in input order); longer lines may differ in
        ///     which of two equal values represents them. The stable radix argsort is also about six times faster than the
        ///     comparison sort of the float16 keys this replaced (100,000 elements: 1.5 ms, was 8.9 ms; NumPy 8.5 ms).
        /// </summary>
        /// <param name="a">The float16 operand (any layout; only read).</param>
        /// <param name="n">Its element count (positive).</param>
        /// <param name="equal_nan">True keeps one NaN, false every NaN.</param>
        /// <returns>The sorted distinct values (NaNs last), a fresh 1-D float16 array.</returns>
        private static unsafe NDArray uniqueValuesOnlyHalf(NDArray a, long n, bool equal_nan)
        {
            if (!IsManagedSortableLength(n))
                return uniqueFlatSortedLongFloat<Half>(a, n, equal_nan, false, false, false)[0];

            return uniqueFlatArgsorted<Half>(a, n, false, false, false, equal_nan)[0];
        }

        /// <summary>
        ///     The flat unique of a complex128 operand. A values-only call is NumPy's hash path whatever
        ///     <paramref name="equal_nan"/> is (each equality class keeps its first element in C order; under
        ///     <paramref name="equal_nan"/> every value with a NaN part is one class, otherwise each is a value of its own,
        ///     sorted last in NumPy's complex order); every other call is NumPy's argsort and gather
        ///     (<see cref="uniqueFlatArgsorted{T}"/>), whose argsort orders the NaN classes as NumPy's complex sort does.
        /// </summary>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="equal_nan">True: every value with a NaN part is one value; false: each is a value of its own.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="uniqueFlatKwargs"/>).</returns>
        private static unsafe NDArray[] uniqueFlatSortedComplex(NDArray a, bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            long n = a.size;
            if (n == 0) return BuildEmptyResults<Complex>(return_index, return_inverse, return_counts);

            // Values only: NumPy's _unique1d takes its C++ hash path for complex (unique.cpp) whatever equal_nan is, and
            // sorts the result. Each equality class is represented by its FIRST element in C order (+0.0 and -0.0 one
            // class); under equal_nan all values with a NaN part are one class (np.unique([nan+0j, 1j, 1+nanj]) is
            // [1j, nan+0j]), otherwise each is a value, in NumPy's complex order (np.unique([1+nanj, nan+2j],
            // equal_nan=False) puts R+nanj first). Above 2^28 elements the hash bails to this method, so those take
            // uniqueValuesOnlyComplex, the same answer from the stable argsort.
            if (!return_index && !return_inverse && !return_counts)
                return new[] { n <= (1L << 28)
                    ? UniqueHashSortedComplex(a, wantCounts: false, equalNan: equal_nan).values
                    : uniqueValuesOnlyComplex(a, n, equal_nan) };

            if (!IsManagedSortableLength(n))
                return uniqueFlatSortedLongComplex(a, n, equal_nan, return_index, return_inverse, return_counts);

            // An index, an inverse or counts: NumPy's argsort-and-gather statements (uniqueFlatArgsorted), whose argsort
            // orders the NaN classes as NumPy's complex sort does. NumPy sorts in place for counts alone; its complex sort
            // keeps the bits, so only the order of equal values could differ.
            return uniqueFlatArgsorted<Complex>(a, n, return_index, return_inverse, return_counts, equal_nan);
        }

        /// <summary>
        ///     The values-only complex128 route above 2^28 elements, where the hash path bails: the hash path's answer from
        ///     the stable argsort (<see cref="ArgsortedHashAnswer"/>: <see cref="uniqueFlatArgsorted{T}"/> without outputs).
        ///     The sorted distinct values, each class represented by its first element in input order (the stable order's
        ///     run start); the values with a NaN part sorted last in NumPy's complex order, each a value without
        ///     <paramref name="equal_nan"/> and one value with it, represented by the first of them in C order as NumPy's
        ///     hash set keeps it (<see cref="KeepFirstNaNPartInCOrder"/>).
        /// </summary>
        /// <param name="a">The complex128 operand (any layout; only read).</param>
        /// <param name="n">Its element count (positive).</param>
        /// <param name="equal_nan">True keeps one value with a NaN part, false every one.</param>
        /// <returns>The sorted distinct values (those with a NaN part last), a fresh 1-D complex128 array.</returns>
        private static unsafe NDArray uniqueValuesOnlyComplex(NDArray a, long n, bool equal_nan)
        {
            if (!IsManagedSortableLength(n))
                return uniqueFlatSortedLongComplex(a, n, equal_nan, false, false, false)[0];

            return ArgsortedHashAnswer(a, n, wantCounts: false, equalNan: equal_nan).values;
        }

        /// <summary>
        ///     NumPy's hash-path representative of a collapsed NaN class: when the sorted distinct values end with a value
        ///     that has a NaN part (the run <c>equal_nan</c> collapsed, represented by the first such value in sorted order),
        ///     overwrites it with the first such value of <paramref name="a"/> in C order, the one NumPy's hash set keeps
        ///     (it inserts in C order, keeps the first of an equality class, and makes every value with a NaN part one
        ///     class under <c>equal_nan</c>).
        /// </summary>
        /// <param name="values">The sorted distinct values: a fresh, writeable, C-contiguous 1-D complex128 array.</param>
        /// <param name="a">The operand the values came from (any layout; only read).</param>
        private static unsafe void KeepFirstNaNPartInCOrder(NDArray values, NDArray a)
        {
            long m = values.size;
            Complex* v = (Complex*)values.Address;   // fresh and C-contiguous: its first element is at its base
            if (m == 0 || !IsNaNKey(v[m - 1]))
                return;
            long n = a.size;
            if (a.Shape.IsContiguous)
            {
                // Base address PLUS the shape's offset: a contiguous view may keep the base with a non-zero offset.
                Complex* src = (Complex*)a.Address + a.Shape.offset;
                for (long i = 0; i < n; i++)
                    if (IsNaNKey(src[i])) { v[m - 1] = src[i]; return; }
            }
            else
            {
                using var flat = a.flat;   // the C-order flattening (a copy for this layout)
                Complex* src = (Complex*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < n; i++)
                {
                    Complex z = src[getOffset(i)];
                    if (IsNaNKey(z)) { v[m - 1] = z; return; }
                }
            }
        }

        /// <summary>
        ///     NumPy's index and inverse routes of <c>_unique1d</c> over float16, float32, float64 and complex128, statement
        ///     for statement: <c>perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')</c>,
        ///     <c>aux = ar[perm]</c>, the run starts <c>aux[1:] != aux[:-1]</c> (IEEE: a NaN differs from everything), the
        ///     <c>equal_nan</c> rule (one run from the first NaN in sorted order), then the outputs with the index
        ///     <c>perm[mask]</c>.
        /// </summary>
        /// <typeparam name="T">Half, float, double or Complex (the operand's element type).</typeparam>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="n">Its element count (positive, at most <see cref="Array.MaxLength"/>).</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="equal_nan">True: the NaNs (sorted last; for complex, every value with a NaN part) are one run from
        ///     the first of them in sorted order; false: each is a run of its own.</param>
        /// <param name="values">The elements of <paramref name="a"/> in C order when the caller extracted them already (the
        ///     complex hash path handing a line over); null extracts them here. Only read.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="EmitOutputs{T}"/>): each value bitwise
        ///     <c>ar[index]</c>.</returns>
        /// <remarks>
        ///     <para>
        ///     NumSharp's numeric argsort (<c>AxisSort.ArgSort</c>) is one stable order for every kind, NumPy's mergesort
        ///     order: NaNs last in input order, <c>-0.0</c> equal to <c>+0.0</c>, complex values as NumPy's complex sort
        ///     orders them (the others lexicographically, then <c>R+nanj</c> by R, <c>nan+Rj</c> by R, <c>nan+nanj</c>). So
        ///     on the return_index route (NumPy's mergesort) every output is NumPy's: a zero's sign, a NaN's payload and the
        ///     NaN class that represents a collapsed run included. NumPy's quicksort argsort (an inverse without an index)
        ///     orders equal keys its own way, which only changes the bits of a run's value (which zero, which NaN payload).
        ///     </para>
        ///     <para>
        ///     Do not swap the argsort for the values-only routes' NaN partition plus a sort of the prefix carrying a
        ///     permutation: the partition is unstable and the radix key orders <c>-0.0</c> below <c>+0.0</c>, so the
        ///     values, the index and the inverse leave NumPy's answer (a run of zeros keeps <c>-0.0</c> whatever the input
        ///     order, a collapsed NaN run's value can belong to another NaN than its index) and complex NaN classes come
        ///     out in partition order.
        ///     </para>
        /// </remarks>
        private static unsafe NDArray[] uniqueFlatArgsorted<T>(NDArray a, long n, bool return_index, bool return_inverse,
            bool return_counts, bool equal_nan, T[] values = null) where T : unmanaged
        {
            // ar = np.asanyarray(ar).flatten(): the elements in C order, bits kept.
            values ??= ExtractKeysOnly<T>(a, n);
            // perm = ar.argsort(kind=...): one stable order, NumPy's mergesort order, for every kind (see remarks).
            var perm = new long[n];
            using (var order = AxisSort.ArgSort(a, null, NPY_SORTKIND.NPY_STABLESORT))
            {
                long* p = (long*)order.Address;   // a fresh C-contiguous int64 array: its first element is at its base
                for (long i = 0; i < n; i++) perm[i] = p[i];
            }
            // aux = ar[perm]: a gather, so every value keeps its bits.
            var aux = new T[n];
            for (long i = 0; i < n; i++) aux[i] = values[perm[i]];

            // mask[:1] = True; mask[1:] = aux[1:] != aux[:-1]
            var mask = new bool[n];
            mask[0] = true;
            long uniqueCount = 1;
            for (long i = 1; i < n; i++)
                if (IeeeNotEqual(aux[i], aux[i - 1])) { mask[i] = true; uniqueCount++; }

            // equal_nan and np.isnan(aux[-1]): aux_firstnan = the first NaN (sorted last, so a backward scan finds it);
            // mask[aux_firstnan] = True; mask[aux_firstnan + 1:] = False. NumPy's prefix statement
            // `mask[1:aux_firstnan] = aux[1:aux_firstnan] != aux[:aux_firstnan - 1]` is what the loop above computed.
            if (equal_nan && IsNaNKey(aux[n - 1]))
            {
                long firstNaN = n - 1;
                while (firstNaN > 0 && IsNaNKey(aux[firstNaN - 1])) firstNaN--;
                // Always set already (a NaN differs from the value before it), kept as NumPy's statement.
                if (!mask[firstNaN]) { mask[firstNaN] = true; uniqueCount++; }
                for (long i = firstNaN + 1; i < n; i++)
                    if (mask[i]) { mask[i] = false; uniqueCount--; }
            }

            return EmitOutputs(a, aux, perm, mask, n, uniqueCount, return_index, return_inverse, return_counts,
                               indexAtRunStart: true);
        }

        /// <summary>
        ///     NumPy's <c>!=</c> of two float or complex values: IEEE inequality, so a NaN differs from every value (itself
        ///     included) and <c>-0.0</c> equals <c>+0.0</c>; a complex pair differs when either part does.
        /// </summary>
        /// <typeparam name="T">Half, float, double or Complex; the <c>typeof</c> tests are JIT constants.</typeparam>
        /// <param name="x">The first value.</param>
        /// <param name="y">The second value.</param>
        /// <returns>True when the values differ.</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is another type (a bug in the caller).</exception>
        [MethodImpl(OptimizeAndInline)]
        private static bool IeeeNotEqual<T>(T x, T y) where T : unmanaged
        {
            if (typeof(T) == typeof(double))
                return System.Runtime.CompilerServices.Unsafe.As<T, double>(ref x) != System.Runtime.CompilerServices.Unsafe.As<T, double>(ref y);
            if (typeof(T) == typeof(float))
                return System.Runtime.CompilerServices.Unsafe.As<T, float>(ref x) != System.Runtime.CompilerServices.Unsafe.As<T, float>(ref y);
            if (typeof(T) == typeof(Half))
                return System.Runtime.CompilerServices.Unsafe.As<T, Half>(ref x) != System.Runtime.CompilerServices.Unsafe.As<T, Half>(ref y);
            if (typeof(T) == typeof(Complex))
            {
                var cx = System.Runtime.CompilerServices.Unsafe.As<T, Complex>(ref x);
                var cy = System.Runtime.CompilerServices.Unsafe.As<T, Complex>(ref y);
                return cx.Real != cy.Real || cx.Imaginary != cy.Imaginary;
            }
            throw new NotSupportedException($"IeeeNotEqual: {typeof(T)} is not a float or complex type.");
        }

        /// <summary>The NaN NumPy's float64 sort writes: the positive quiet NaN <c>0x7FF8000000000000</c>.</summary>
        private static readonly double NumPySortNaN64 = BitConverter.UInt64BitsToDouble(0x7FF8000000000000UL);

        /// <summary>The NaN NumPy's float32 sort writes: the positive quiet NaN <c>0x7FC00000</c>.</summary>
        private static readonly float NumPySortNaN32 = BitConverter.UInt32BitsToSingle(0x7FC00000U);

        /// <summary>
        ///     NumPy's in-place sort of a float64 array's NaNs: <c>_unique1d</c>'s values-only and counts-only routes run
        ///     <c>ar.sort()</c> (neither an index nor an inverse is asked), and NumPy's default float sort (x86-simd-sort)
        ///     rewrites every NaN of a line longer than one element — negative, signalling or payload-bearing — with the
        ///     positive quiet NaN. So <c>np.unique([-nan, 1.0])</c> is <c>[1.0, nan]</c> with the bits
        ///     <c>0x7FF8000000000000</c>, while a one-element <c>np.unique([-nan])</c> keeps <c>0xFFF8…</c> (a line of one
        ///     element is not sorted). The index and inverse routes argsort and gather, so they keep every NaN's bits.
        /// </summary>
        /// <param name="keys">The partitioned keys; the NaN tail <c>[firstNaN, n)</c> is overwritten in place.</param>
        /// <param name="firstNaN">The index where the NaN tail starts.</param>
        /// <param name="n">The number of keys: one element keeps its NaN.</param>
        /// <remarks><c>np.sort</c> writes the same NaN on its default kind (<c>AxisSort.SortLineF64</c>).</remarks>
        private static void WriteSortNaNs(double[] keys, long firstNaN, long n)
        {
            if (n <= 1) return;
            double nan = NumPySortNaN64;
            for (long i = firstNaN; i < n; i++) keys[i] = nan;
        }

        /// <summary>
        ///     <see cref="WriteSortNaNs(double[], long, long)"/> for float32: every NaN of a line longer than one element
        ///     becomes <c>0x7FC00000</c>.
        /// </summary>
        /// <param name="keys">The partitioned keys; the NaN tail <c>[firstNaN, n)</c> is overwritten in place.</param>
        /// <param name="firstNaN">The index where the NaN tail starts.</param>
        /// <param name="n">The number of keys: one element keeps its NaN.</param>
        private static void WriteSortNaNs(float[] keys, long firstNaN, long n)
        {
            if (n <= 1) return;
            float nan = NumPySortNaN32;
            for (long i = firstNaN; i < n; i++) keys[i] = nan;
        }

        // ----- Helpers -----

        /// <summary>
        ///     Returns true when n fits in a managed T[] (n ≤ Array.MaxLength). When false,
        ///     the caller routes to the unmanaged long-indexed fallback
        ///     (<see cref="uniqueFlatSortedLong{T}"/>) which is slower but supports any size.
        /// </summary>
        /// <param name="n">The element count.</param>
        /// <returns>True when a managed key array of <paramref name="n"/> elements can be allocated.</returns>
        private static bool IsManagedSortableLength(long n) => n <= System.Array.MaxLength;

        /// <summary>
        ///     Copies the operand's elements into a managed key array in C order and builds the identity permutation
        ///     beside it: the sort inputs of every flat path that returns an index or an inverse. A C-contiguous operand is
        ///     one block copy from its first element (its base address plus its offset); any other layout is read element
        ///     by element through its strides.
        /// </summary>
        /// <typeparam name="T">The operand's element type (the caller dispatched on its dtype).</typeparam>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="n">Its element count; must not exceed <see cref="Array.MaxLength"/>
        ///     (see <see cref="IsManagedSortableLength"/>).</param>
        /// <returns>The keys (C order) and the permutation <c>0..n-1</c>.</returns>
        private static unsafe (T[] keys, long[] perm) ExtractKeysAndPerm<T>(NDArray a, long n) where T : unmanaged
        {
            var keys = new T[n];
            var perm = new long[n];

            if (a.Shape.IsContiguous)
            {
                // Base address PLUS the shape's offset: a contiguous view may keep the base with a non-zero offset.
                T* src = (T*)a.Address + a.Shape.offset;
                fixed (T* dst = keys)
                {
                    long byteCount = n * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
                    Buffer.MemoryCopy(src, dst, byteCount, byteCount);
                }
            }
            else
            {
                var flat = a.flat;
                T* src = (T*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < n; i++) keys[i] = src[getOffset(i)];
            }

            for (long i = 0; i < n; i++) perm[i] = i;
            return (keys, perm);
        }

        /// <summary>
        /// Values-only extract — skips the perm array allocation+fill (saves ~8N bytes and ~N writes).
        /// </summary>
        /// <typeparam name="T">The operand's element type (the caller dispatched on its dtype).</typeparam>
        /// <param name="a">The operand (any layout; only read): a C-contiguous one is one block copy from its first element
        ///     (base address plus offset), any other is read through its strides.</param>
        /// <param name="n">Its element count; must not exceed <see cref="Array.MaxLength"/>.</param>
        /// <returns>The elements in C order, a fresh managed array.</returns>
        private static unsafe T[] ExtractKeysOnly<T>(NDArray a, long n) where T : unmanaged
        {
            var keys = new T[n];
            if (a.Shape.IsContiguous)
            {
                // Base address PLUS the shape's offset: a contiguous view may keep the base with a non-zero offset.
                T* src = (T*)a.Address + a.Shape.offset;
                fixed (T* dst = keys)
                {
                    long byteCount = n * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
                    Buffer.MemoryCopy(src, dst, byteCount, byteCount);
                }
            }
            else
            {
                var flat = a.flat;
                T* src = (T*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < n; i++) keys[i] = src[getOffset(i)];
            }
            return keys;
        }

        // ----- Partition-only helpers (no perm tracking) — used by the values-only and counts-only paths -----

        /// <summary>
        ///     Moves every NaN of <paramref name="keys"/><c>[0..n)</c> to the tail by a two-pointer swap: the NaN step of
        ///     the values-only and counts-only routes, which run NumPy's in-place sort and carry no permutation.
        ///     Unstable: the values swapped forward lose their order, which the sort of the prefix restores, and the NaNs
        ///     keep the partition's order, which on these routes only decides which NaN represents a collapsed run
        ///     (float32 and float64 tails are then written as NumPy's sort NaN).
        /// </summary>
        /// <param name="keys">The keys, partitioned in place (each NaN keeps its bits).</param>
        /// <param name="n">The number of keys.</param>
        /// <returns>The index of the first NaN (<paramref name="n"/> when there is none).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static long PartitionNaN_DoubleKeysOnly(double[] keys, long n)
        {
            long hi = n;
            long i = 0;
            while (i < hi)
            {
                if (double.IsNaN(keys[i])) { hi--; (keys[i], keys[hi]) = (keys[hi], keys[i]); }
                else i++;
            }
            return hi;
        }

        /// <summary>
        ///     The values-only NaN partition for float32 keys (see <see cref="PartitionNaN_DoubleKeysOnly"/>).
        /// </summary>
        /// <param name="keys">The keys, partitioned in place.</param>
        /// <param name="n">The number of keys.</param>
        /// <returns>The index of the first NaN (<paramref name="n"/> when there is none).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static long PartitionNaN_FloatKeysOnly(float[] keys, long n)
        {
            long hi = n;
            long i = 0;
            while (i < hi)
            {
                if (float.IsNaN(keys[i])) { hi--; (keys[i], keys[hi]) = (keys[hi], keys[i]); }
                else i++;
            }
            return hi;
        }

        /// <summary>
        /// Values-only emit for non-NaN-capable types: single dedup-scan over sorted keys.
        /// </summary>
        /// <typeparam name="T">The key type (no NaN: bool, the integers, char, decimal).</typeparam>
        /// <param name="keys">The sorted keys.</param>
        /// <param name="n">The number of keys; must be positive (the first key is always kept).</param>
        /// <returns>The first key of each run of equal keys, a fresh 1-D array.</returns>
        private static unsafe NDArray EmitValuesOnly<T>(T[] keys, long n) where T : unmanaged, IEquatable<T>
        {
            // First pass: count uniques.
            long uniqueCount = 1;
            for (long i = 1; i < n; i++)
                if (!keys[i].Equals(keys[i - 1])) uniqueCount++;

            // Second pass: emit.
            var valuesBlock = new UnmanagedMemoryBlock<T>(uniqueCount);
            var valuesSlice = new ArraySlice<T>(valuesBlock);
            T* vDst = valuesBlock.Address;
            vDst[0] = keys[0];
            long vIdx = 1;
            for (long i = 1; i < n; i++)
                if (!keys[i].Equals(keys[i - 1])) vDst[vIdx++] = keys[i];

            return new NDArray(valuesSlice, Shape.Vector(uniqueCount));
        }

        /// <summary>
        /// Values-only emit for float types (Double/Single). Handles the NaN tail:
        /// equal_nan=true → one NaN representative; equal_nan=false → every NaN unique.
        /// </summary>
        /// <typeparam name="T">The key type (float or double).</typeparam>
        /// <param name="keys">The sorted non-NaN prefix followed by the NaN tail (partition order).</param>
        /// <param name="n">The number of keys.</param>
        /// <param name="firstNaN">The index where the NaN tail starts (<paramref name="n"/> without NaN).</param>
        /// <param name="equal_nan">True keeps the tail's first NaN (its bits as they are), false every NaN.</param>
        /// <returns>The distinct values, NaNs last, a fresh 1-D array.</returns>
        private static unsafe NDArray EmitValuesOnlyFloat<T>(T[] keys, long n, long firstNaN, bool equal_nan)
            where T : unmanaged, IEquatable<T>
        {
            // Count uniques in non-NaN prefix.
            long uniqueCount = firstNaN > 0 ? 1 : 0;
            for (long i = 1; i < firstNaN; i++)
                if (!keys[i].Equals(keys[i - 1])) uniqueCount++;

            // Add NaN entries: 1 if equal_nan, else (n - firstNaN).
            long nanCount = n - firstNaN;
            if (nanCount > 0)
                uniqueCount += equal_nan ? 1 : nanCount;

            var valuesBlock = new UnmanagedMemoryBlock<T>(uniqueCount);
            var valuesSlice = new ArraySlice<T>(valuesBlock);
            T* vDst = valuesBlock.Address;
            long vIdx = 0;

            if (firstNaN > 0)
            {
                vDst[vIdx++] = keys[0];
                for (long i = 1; i < firstNaN; i++)
                    if (!keys[i].Equals(keys[i - 1])) vDst[vIdx++] = keys[i];
            }

            if (nanCount > 0)
            {
                if (equal_nan)
                {
                    vDst[vIdx++] = keys[firstNaN];
                }
                else
                {
                    for (long i = firstNaN; i < n; i++) vDst[vIdx++] = keys[i];
                }
            }

            return new NDArray(valuesSlice, Shape.Vector(uniqueCount));
        }

        /// <summary>
        ///     Counts-only emit for the float paths (Double/Single): NumPy's counts-only route, which sorts in place and
        ///     takes no argsort. Mirrors <see cref="EmitValuesOnlyFloat{T}"/> for the values AND
        ///     <see cref="EmitOutputs{T}"/>'s counts branch (run-length between mask=true positions), with NO perm (counts
        ///     are run-lengths of the sorted values). <paramref name="keys"/>[0..firstNaN) is radix-sorted ascending;
        ///     [firstNaN..n) is the NaN run (NumPy's sort NaN for two or more elements,
        ///     <see cref="WriteSortNaNs(double[], long, long)"/>).
        /// </summary>
        /// <typeparam name="T">The key type (float or double).</typeparam>
        /// <param name="keys">The sorted non-NaN prefix followed by the NaN tail.</param>
        /// <param name="n">The number of keys.</param>
        /// <param name="firstNaN">The index where the NaN tail starts (<paramref name="n"/> without NaN).</param>
        /// <param name="equal_nan">True: one NaN value counted as the whole tail; false: each NaN counted once.</param>
        /// <returns>[values, counts]: fresh 1-D arrays, the counts int64.</returns>
        private static unsafe NDArray[] EmitValuesAndCountsFloat<T>(T[] keys, long n, long firstNaN, bool equal_nan)
            where T : unmanaged, IEquatable<T>
        {
            long uniqueCount = firstNaN > 0 ? 1 : 0;
            for (long i = 1; i < firstNaN; i++)
                if (!keys[i].Equals(keys[i - 1])) uniqueCount++;
            long nanCount = n - firstNaN;
            if (nanCount > 0) uniqueCount += equal_nan ? 1 : nanCount;

            var vblock = new UnmanagedMemoryBlock<T>(uniqueCount);
            var cblock = new UnmanagedMemoryBlock<long>(uniqueCount);
            T* vp = vblock.Address;
            long* cp = cblock.Address;
            long vi = 0;

            if (firstNaN > 0)
            {
                vp[vi] = keys[0];
                long runStart = 0;
                for (long i = 1; i < firstNaN; i++)
                {
                    if (!keys[i].Equals(keys[i - 1]))
                    {
                        cp[vi++] = i - runStart;   // count of the run that just ended
                        vp[vi] = keys[i];
                        runStart = i;
                    }
                }
                cp[vi++] = firstNaN - runStart;    // last prefix run (up to the NaN boundary)
            }

            if (nanCount > 0)
            {
                if (equal_nan)
                {
                    vp[vi] = keys[firstNaN]; cp[vi] = nanCount; vi++;
                }
                else
                {
                    for (long i = firstNaN; i < n; i++) { vp[vi] = keys[i]; cp[vi] = 1; vi++; }
                }
            }

            return new[]
            {
                new NDArray(new ArraySlice<T>(vblock), Shape.Vector(uniqueCount)),
                new NDArray(new ArraySlice<long>(cblock), Shape.Vector(uniqueCount)),
            };
        }

        /// <summary>
        ///     For non-NaN-capable types: mask[i] = !keys[i].Equals(keys[i-1]) (with mask[0]=true).
        ///     Float/complex paths handle their own mask construction inline using IEEE != to
        ///     preserve equal_nan=false semantics, then call EmitOutputs directly.
        ///
        ///     Index uses min-perm-within-run for first-occurrence semantics (Array.Sort is
        ///     unstable in .NET; we recover stability cheaply in a single pass).
        /// </summary>
        /// <typeparam name="T">The key type (no NaN).</typeparam>
        /// <param name="a">The operand (its rank decides the inverse's shape in <see cref="EmitOutputs{T}"/>).</param>
        /// <param name="keys">The sorted keys.</param>
        /// <param name="perm">The permutation that sorted them.</param>
        /// <param name="n">The number of keys (positive).</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="firstNaN">Unused (always -1 here: these types hold no NaN); kept so the call sites read like the
        ///     float paths'.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="EmitOutputs{T}"/>).</returns>
        private static unsafe NDArray[] BuildSortedResults<T>(
            NDArray a, T[] keys, long[] perm, long n,
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
            // The float / complex paths build their own masks (uniqueFlatArgsorted, EmitValuesAndCountsFloat).
            _ = firstNaN;
            return EmitOutputs(a, keys, perm, mask, n, uniqueCount, return_index, return_inverse, return_counts);
        }

        /// <summary>
        ///     Builds NumPy's outputs from the sorted keys, their permutation and the run-start mask: the values
        ///     <c>aux[mask]</c>, the index (<c>perm[mask]</c> with <paramref name="indexAtRunStart"/>; otherwise the smallest
        ///     permutation entry of each run: the first occurrence, whether or not the sort was stable), the inverse
        ///     <c>inv[perm[i]] = cumsum(mask)[i] - 1</c> and the counts (the run lengths).
        /// </summary>
        /// <typeparam name="T">The key type.</typeparam>
        /// <param name="a">The operand: for more than one dimension the inverse takes its dimensions.</param>
        /// <param name="keys">The sorted keys.</param>
        /// <param name="perm">The permutation that sorted them.</param>
        /// <param name="mask">True where a run of equal keys starts (<c>mask[0]</c> is true).</param>
        /// <param name="n">The number of keys (positive).</param>
        /// <param name="uniqueCount">The number of true mask entries.</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="indexAtRunStart">True: each index is the permutation entry at its run's start, NumPy's
        ///     <c>perm[mask]</c> (for a permutation that is NumPy's argsort: a run collapsed by <c>equal_nan</c> holds
        ///     different NaN values, and NumPy's index is the first of them in sorted order). False (the routes whose sort
        ///     is unstable): the smallest permutation entry of each run, the first occurrence.</param>
        /// <returns>[values, index?, inverse?, counts?]: fresh arrays, the index outputs int64.</returns>
        private static unsafe NDArray[] EmitOutputs<T>(
            NDArray a, T[] keys, long[] perm, bool[] mask, long n, long uniqueCount,
            bool return_index, bool return_inverse, bool return_counts, bool indexAtRunStart = false) where T : unmanaged
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

            // index — perm[mask] (NumPy's statement, for NumPy's argsort), else min(perm) within each run of equal keys
            // (and across collapsed-NaN runs: the first occurrence whatever order an unstable sort left)
            if (return_index && indexAtRunStart)
            {
                var idxBlock = new UnmanagedMemoryBlock<long>(uniqueCount);
                long* iDst = idxBlock.Address;
                long oIdx = 0;
                for (long i = 0; i < n; i++)
                    if (mask[i]) iDst[oIdx++] = perm[i];
                results[outPos++] = new NDArray(new ArraySlice<long>(idxBlock), Shape.Vector(uniqueCount));
            }
            else if (return_index)
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
                var invShape = a.ndim <= 1 ? Shape.Vector(n) : Shape.Adopt(a.Storage.Shape.dimensions);
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

        /// <summary>
        ///     The outputs of an empty operand: an empty values array of the operand's element type and empty int64 index
        ///     outputs, one per requested flag.
        /// </summary>
        /// <typeparam name="T">The operand's element type.</typeparam>
        /// <param name="return_index">Also return an empty index.</param>
        /// <param name="return_inverse">Also return an empty inverse (1-D: <see cref="UniqueCompute"/> reshapes it to the
        ///     input's shape, e.g. <c>(0, 3)</c>).</param>
        /// <param name="return_counts">Also return empty counts.</param>
        /// <returns>[values, index?, inverse?, counts?], every array of shape <c>(0,)</c>.</returns>
        private static NDArray[] BuildEmptyResults<T>(bool return_index, bool return_inverse, bool return_counts) where T : unmanaged
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

        /// <summary>
        ///     NumPy's row path of <c>np.unique</c> over a numeric operand (an axis of an N-D input): the axis is moved
        ///     first, each slab (every other axis flattened, C order) is one row, the rows are sorted lexicographically
        ///     with a stable tie-break by input index (NumPy's mergesort of the structured rows), and the distinct rows
        ///     are gathered and moved back to the axis.
        /// </summary>
        /// <param name="a">The operand (two or more dimensions; only read).</param>
        /// <param name="axis">The normalized axis.</param>
        /// <param name="return_index">Also return the first row of each distinct row.</param>
        /// <param name="return_inverse">Also return the reconstruction indices (1-D, as NumPy returns them with an
        ///     axis).</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="equal_nan">Ignored: NumPy's structured rows are not of a float kind, so its NaN branch never runs
        ///     and every NaN row is distinct.</param>
        /// <returns>[values, index?, inverse?, counts?]: the values a view of the gathered rows moved back to the axis
        ///     (<c>flags.owndata</c> False for every axis, as NumPy's <c>reshape_uniq</c> returns), the index outputs fresh
        ///     int64 arrays.</returns>
        /// <exception cref="NotSupportedException">A dtype outside the 15 numeric ones (text takes its own route).</exception>
        /// <remarks>
        ///     <para>
        ///     Rows compare with IEEE <c>&lt;</c> / <c>==</c>, so <c>-0.0</c> and <c>+0.0</c> are equal (the surviving
        ///     row's zeros follow input order) and NaN rows are equal for ordering but distinct for the mask.
        ///     </para>
        ///     <para>
        ///     A numpy.matrix operand is computed as a plain array (the region below). An operand already C-contiguous
        ///     after the move (a view at an offset included: an np.split child) is read in place, NumPy's
        ///     <c>np.ascontiguousarray</c>; the slab helpers read it from its first element, its base address plus its
        ///     offset.
        ///     </para>
        /// </remarks>
        private static NDArray[] uniqueAxisKwargs(NDArray a, int axis, bool return_index, bool return_inverse, bool return_counts, bool equal_nan)
        {
            // equal_nan has NO EFFECT on the axis path — NumPy consolidates each slab into a
            // structured scalar whose dtype.kind is 'V', so _unique1d's equal_nan branch (guarded by
            // kind in "cfmM") is skipped and the mask is a plain structured `!=`, under which every
            // NaN slab is DISTINCT (nan != nan) regardless of equal_nan. Accepted for API symmetry.
            _ = equal_nan;
            using var plain = NDMatrixBoundary.PlainFor(a);   // the member keeps the plain-array meaning for a matrix object
            var moved = axis == 0 ? a : np.moveaxis(a, axis, 0);
            long n = moved.Shape.dimensions[0];

            long slabSize = 1;
            for (int d = 1; d < moved.ndim; d++) slabSize *= moved.Shape.dimensions[d];

            var orig = new int[n];
            for (int i = 0; i < n; i++) orig[i] = i;

            var movedCopy = moved.Shape.IsContiguous ? moved : moved.copy();
            // Stable tie-break by original index: identical slabs (in particular NaN slabs, which
            // compare EQUAL for ordering but are kept DISTINCT by the != mask below) keep input
            // order — reproducing NumPy's mergesort argsort in the axis path.
            System.Array.Sort(orig, (x, y) =>
            {
                int c = CompareSlabs(movedCopy, x, y, slabSize);
                return c != 0 ? c : x.CompareTo(y);
            });

            var sortedKeepIdx = new List<int>();
            var sortedFirstOrig = new List<long>();
            var sortedCounts = new List<long>();
            int prev = -1;
            for (int i = 0; i < n; i++)
            {
                int cur = orig[i];
                if (prev == -1 || !SlabsEqual(movedCopy, prev, cur, slabSize))
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
                resultShapeDims[d] = moved.Shape.dimensions[d];

            var values = GatherSlabs(movedCopy, sortedKeepIdx.ToArray(), slabSize, Shape.Adopt(resultShapeDims));

            // NumPy's reshape_uniq: `np.moveaxis(uniq, 0, axis)` for every axis, axis 0 included, so the values are always
            // a VIEW of the gathered rows' buffer (flags.owndata False). The gathered owner is left to the caller's scope
            // (np.unique's boundary scope), its buffer kept alive by the view.
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
                        if (prevSorted == -1 || !SlabsEqual(movedCopy, prevSorted, cur, slabSize))
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
        ///     The lexicographic order of two slabs (rows) of a C-contiguous array, dispatched on its dtype: the integer
        ///     family and decimal by <c>CompareTo</c>, the float and complex dtypes by IEEE compares with NaN last.
        /// </summary>
        /// <param name="src">The C-contiguous array whose rows are compared, read from its first element (base address plus
        ///     offset).</param>
        /// <param name="a">The first row's index.</param>
        /// <param name="b">The second row's index.</param>
        /// <param name="slabSize">The elements per row.</param>
        /// <returns>Negative, zero or positive as row <paramref name="a"/> orders before, with or after row
        ///     <paramref name="b"/>.</returns>
        /// <exception cref="NotSupportedException">A dtype outside the 15 numeric ones.</exception>
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

        /// <summary>
        ///     The lexicographic order of two rows of a dtype without NaN, element by element with <c>CompareTo</c>.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="src">The C-contiguous array, read from its first element (base address plus offset).</param>
        /// <param name="a">The first row's index.</param>
        /// <param name="b">The second row's index.</param>
        /// <param name="slabSize">The elements per row.</param>
        /// <returns>The first nonzero element comparison, or 0 for equal rows.</returns>
        private static unsafe int CompareSlabsT<T>(NDArray src, int a, int b, long slabSize)
            where T : unmanaged, IComparable<T>
        {
            T* ptr = (T*)src.Address + src.Shape.offset;
            long aBase = a * slabSize;
            long bBase = b * slabSize;
            for (long k = 0; k < slabSize; k++)
            {
                int c = ptr[aBase + k].CompareTo(ptr[bBase + k]);
                if (c != 0) return c;
            }
            return 0;
        }

        // The CompareSlabs* float/complex orderings use IEEE compares (`<`/`>`), NOT CompareTo, so
        // that -0.0 and +0.0 compare EQUAL (CompareTo would order -0.0 < +0.0) — matching NumPy's
        // structured sort, where the surviving zero's sign follows input order. NaN sorts to the end
        // (all NaN equivalent); the stable tie-break then keeps identical NaN slabs in input order.

        /// <summary>
        ///     The lexicographic order of two float64 rows: IEEE compares (<c>-0.0</c> equals <c>+0.0</c>), a NaN after
        ///     every number, two NaNs equal.
        /// </summary>
        /// <param name="src">The C-contiguous float64 array, read from its first element (base address plus offset).</param>
        /// <param name="a">The first row's index.</param>
        /// <param name="b">The second row's index.</param>
        /// <param name="slabSize">The elements per row.</param>
        /// <returns>Negative, zero or positive as row <paramref name="a"/> orders before, with or after row
        ///     <paramref name="b"/>.</returns>
        private static unsafe int CompareSlabsDouble(NDArray src, int a, int b, long slabSize)
        {
            double* ptr = (double*)src.Address + src.Shape.offset;
            long aBase = a * slabSize, bBase = b * slabSize;
            for (long k = 0; k < slabSize; k++)
            {
                double x = ptr[aBase + k], y = ptr[bBase + k];
                bool xn = double.IsNaN(x), yn = double.IsNaN(y);
                if (xn || yn) { if (xn && yn) continue; return xn ? 1 : -1; }
                if (x < y) return -1;
                if (x > y) return 1;
            }
            return 0;
        }

        /// <summary>
        ///     <see cref="CompareSlabsDouble"/> for float32 rows.
        /// </summary>
        /// <param name="src">The C-contiguous float32 array, read from its first element (base address plus offset).</param>
        /// <param name="a">The first row's index.</param>
        /// <param name="b">The second row's index.</param>
        /// <param name="slabSize">The elements per row.</param>
        /// <returns>Negative, zero or positive as row <paramref name="a"/> orders before, with or after row
        ///     <paramref name="b"/>.</returns>
        private static unsafe int CompareSlabsFloat(NDArray src, int a, int b, long slabSize)
        {
            float* ptr = (float*)src.Address + src.Shape.offset;
            long aBase = a * slabSize, bBase = b * slabSize;
            for (long k = 0; k < slabSize; k++)
            {
                float x = ptr[aBase + k], y = ptr[bBase + k];
                bool xn = float.IsNaN(x), yn = float.IsNaN(y);
                if (xn || yn) { if (xn && yn) continue; return xn ? 1 : -1; }
                if (x < y) return -1;
                if (x > y) return 1;
            }
            return 0;
        }

        /// <summary>
        ///     <see cref="CompareSlabsDouble"/> for float16 rows.
        /// </summary>
        /// <param name="src">The C-contiguous float16 array, read from its first element (base address plus offset).</param>
        /// <param name="a">The first row's index.</param>
        /// <param name="b">The second row's index.</param>
        /// <param name="slabSize">The elements per row.</param>
        /// <returns>Negative, zero or positive as row <paramref name="a"/> orders before, with or after row
        ///     <paramref name="b"/>.</returns>
        private static unsafe int CompareSlabsHalf(NDArray src, int a, int b, long slabSize)
        {
            Half* ptr = (Half*)src.Address + src.Shape.offset;
            long aBase = a * slabSize, bBase = b * slabSize;
            for (long k = 0; k < slabSize; k++)
            {
                Half x = ptr[aBase + k], y = ptr[bBase + k];
                bool xn = Half.IsNaN(x), yn = Half.IsNaN(y);
                if (xn || yn) { if (xn && yn) continue; return xn ? 1 : -1; }
                if (x < y) return -1;
                if (x > y) return 1;
            }
            return 0;
        }

        /// <summary>
        ///     The lexicographic order of two complex128 rows: each element by real part, then imaginary part, with IEEE
        ///     compares; an element with a NaN part after every other, two such elements equal.
        /// </summary>
        /// <param name="src">The C-contiguous complex128 array, read from its first element (base address plus offset).</param>
        /// <param name="a">The first row's index.</param>
        /// <param name="b">The second row's index.</param>
        /// <param name="slabSize">The elements per row.</param>
        /// <returns>Negative, zero or positive as row <paramref name="a"/> orders before, with or after row
        ///     <paramref name="b"/>.</returns>
        private static unsafe int CompareSlabsComplex(NDArray src, int a, int b, long slabSize)
        {
            Complex* ptr = (Complex*)src.Address + src.Shape.offset;
            long aBase = a * slabSize, bBase = b * slabSize;
            for (long k = 0; k < slabSize; k++)
            {
                Complex x = ptr[aBase + k], y = ptr[bBase + k];
                bool xn = double.IsNaN(x.Real) || double.IsNaN(x.Imaginary);
                bool yn = double.IsNaN(y.Real) || double.IsNaN(y.Imaginary);
                if (xn || yn) { if (xn && yn) continue; return xn ? 1 : -1; }
                if (x.Real < y.Real) return -1;
                if (x.Real > y.Real) return 1;
                if (x.Imaginary < y.Imaginary) return -1;
                if (x.Imaginary > y.Imaginary) return 1;
            }
            return 0;
        }

        // Slab EQUALITY uses IEEE `==` (via `!=`), matching NumPy's structured `!=` mask: NaN != NaN,
        // so every NaN slab is DISTINCT (independent of equal_nan — see uniqueAxisKwargs), while
        // -0.0 == +0.0 collapses signed-zero slabs. Integer/decimal/char have no NaN/-0 so the
        // generic .Equals path (SlabsEqualT) is exact for them.

        /// <summary>
        ///     Whether two rows are equal by NumPy's structured <c>!=</c>: element-wise IEEE equality, so a row with a NaN
        ///     never equals any row (itself included) and <c>-0.0</c> equals <c>+0.0</c>.
        /// </summary>
        /// <param name="src">The C-contiguous array, read from its first element (base address plus offset).</param>
        /// <param name="a">The first row's index.</param>
        /// <param name="b">The second row's index.</param>
        /// <param name="slabSize">The elements per row.</param>
        /// <returns>True when every element pair is equal.</returns>
        /// <exception cref="NotSupportedException">A dtype outside the 15 numeric ones.</exception>
        private static unsafe bool SlabsEqual(NDArray src, int a, int b, long slabSize)
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
                    Half* ptr = (Half*)src.Address + src.Shape.offset;
                    long aBase = a * slabSize, bBase = b * slabSize;
                    for (long k = 0; k < slabSize; k++)
                        if (ptr[aBase + k] != ptr[bBase + k]) return false;
                    return true;
                }
                case NPTypeCode.Single:
                {
                    float* ptr = (float*)src.Address + src.Shape.offset;
                    long aBase = a * slabSize, bBase = b * slabSize;
                    for (long k = 0; k < slabSize; k++)
                        if (ptr[aBase + k] != ptr[bBase + k]) return false;
                    return true;
                }
                case NPTypeCode.Double:
                {
                    double* ptr = (double*)src.Address + src.Shape.offset;
                    long aBase = a * slabSize, bBase = b * slabSize;
                    for (long k = 0; k < slabSize; k++)
                        if (ptr[aBase + k] != ptr[bBase + k]) return false;
                    return true;
                }
                case NPTypeCode.Complex:
                {
                    Complex* ptr = (Complex*)src.Address + src.Shape.offset;
                    long aBase = a * slabSize, bBase = b * slabSize;
                    for (long k = 0; k < slabSize; k++)
                    {
                        Complex x = ptr[aBase + k], y = ptr[bBase + k];
                        if (x.Real != y.Real || x.Imaginary != y.Imaginary) return false;
                    }
                    return true;
                }
                default: throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     Row equality for a dtype without NaN or signed zeros (bool, the integers, char, decimal): element-wise
        ///     <c>Equals</c>, which for decimal is value equality (<c>1.0m</c> equals <c>1.00m</c>).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="src">The C-contiguous array, read from its first element (base address plus offset).</param>
        /// <param name="a">The first row's index.</param>
        /// <param name="b">The second row's index.</param>
        /// <param name="slabSize">The elements per row.</param>
        /// <returns>True when every element pair is equal.</returns>
        private static unsafe bool SlabsEqualT<T>(NDArray src, int a, int b, long slabSize)
            where T : unmanaged, IEquatable<T>
        {
            T* ptr = (T*)src.Address + src.Shape.offset;
            long aBase = a * slabSize, bBase = b * slabSize;
            for (long k = 0; k < slabSize; k++)
            {
                if (!ptr[aBase + k].Equals(ptr[bBase + k])) return false;
            }
            return true;
        }

        /// <summary>
        ///     Gathers the chosen rows of a C-contiguous array into a fresh array, dispatched on its dtype.
        /// </summary>
        /// <param name="src">The C-contiguous array, read from its first element (base address plus offset).</param>
        /// <param name="indices">The rows to copy, in output order.</param>
        /// <param name="slabSize">The elements per row.</param>
        /// <param name="outShape">The result's shape (<c>indices.Length</c> rows of the source's other dimensions).</param>
        /// <returns>A fresh C-contiguous array of <paramref name="src"/>'s dtype.</returns>
        /// <exception cref="NotSupportedException">A dtype outside the 15 numeric ones.</exception>
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

        /// <summary>
        ///     Copies the rows <paramref name="indices"/> of a C-contiguous array, element by element, into a fresh buffer.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="src">The C-contiguous array, read from its first element (base address plus offset).</param>
        /// <param name="indices">The rows to copy, in output order.</param>
        /// <param name="slabSize">The elements per row.</param>
        /// <param name="outShape">The result's shape.</param>
        /// <returns>A fresh array that owns its buffer.</returns>
        private static unsafe NDArray GatherSlabsT<T>(NDArray src, int[] indices, long slabSize, Shape outShape)
            where T : unmanaged
        {
            long outN = indices.Length;
            var block = new UnmanagedMemoryBlock<T>(outN * slabSize);
            var slice = new ArraySlice<T>(block);
            T* dst = block.Address;
            T* srcPtr = (T*)src.Address + src.Shape.offset;
            for (long i = 0; i < outN; i++)
            {
                long srcBase = (long)indices[i] * slabSize;
                long dstBase = i * slabSize;
                for (long k = 0; k < slabSize; k++)
                    dst[dstBase + k] = srcPtr[srcBase + k];
            }
            return new NDArray(slice, outShape);
        }

        // ============================================================
        //  LONG-INDEXED FALLBACK (n > Array.MaxLength ~ 2.1B)
        //
        //  Uses UnmanagedMemoryBlock<KeyPerm<T>> + LongIntroSort. Packs key+perm
        //  into a 16-byte struct so we can reuse the existing single-array sort
        //  utility without writing a parallel-array sort.
        //
        //  Trade-offs vs the managed fast path:
        //  - ~30-50% slower (.NET's introsort > our LongIntroSort port)
        //  - 2× memory for the keys+perm pair (16 bytes packed vs 8+8 separate;
        //    same total but worse alignment in cache)
        //  - Worth it: this path is only reached when n > Array.MaxLength,
        //    which requires 16+ GB just for the input array.
        // ============================================================

        /// <summary>
        ///     A key and its input position, sorted as one 16-byte record by the long-indexed fallback (one array to sort
        ///     where the managed path co-sorts two).
        /// </summary>
        /// <typeparam name="T">The key type.</typeparam>
        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        private struct KeyPerm<T> : IComparable<KeyPerm<T>>
            where T : unmanaged, IComparable<T>
        {
            /// <summary>The element's value.</summary>
            public T Key;

            /// <summary>The element's flat position in the input (C order).</summary>
            public long Perm;

            /// <summary>Orders by key only, so equal keys keep whatever order the (unstable) introsort leaves.</summary>
            /// <param name="other">The record to compare with.</param>
            /// <returns>The keys' <c>CompareTo</c>.</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public int CompareTo(KeyPerm<T> other) => Key.CompareTo(other.Key);
        }

        /// <summary>
        ///     Long-indexed unique for non-NaN-capable types (or types where the
        ///     <c>Comparer&lt;T&gt;.Default</c> works correctly: bool/byte/short/int/long/decimal/char/etc).
        /// </summary>
        /// <typeparam name="T">The operand's element type.</typeparam>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="n">Its element count.</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="firstNaN">-1 to sort every record; otherwise the length of an already partitioned non-NaN prefix,
        ///     the only part sorted.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="EmitOutputsLong{T}"/>).</returns>
        /// <remarks>Reached only above <see cref="Array.MaxLength"/> elements (and by reflection from the tests). The
        ///     record block is not disposed: its buffer returns through the block's finalizer.</remarks>
        private static unsafe NDArray[] uniqueFlatSortedLong<T>(NDArray a, long n,
            bool return_index, bool return_inverse, bool return_counts,
            long firstNaN) where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            var block = new UnmanagedMemoryBlock<KeyPerm<T>>(n);
            KeyPerm<T>* kp = block.Address;
            PopulateKeyPerm(a, kp, n);

            if (firstNaN < 0)
            {
                // Sort entire range with default compare
                Utilities.LongIntroSort.Sort(kp, n);
            }
            else
            {
                // Sort only the non-NaN prefix (caller already partitioned)
                Utilities.LongIntroSort.Sort(kp, firstNaN);
            }

            return BuildMaskAndEmitLong<T>(a, kp, n, firstNaN, equal_nan: true,
                                            return_index, return_inverse, return_counts);
        }

        /// <summary>
        ///     Long-indexed unique for NaN-capable types Single/Double/Half (not Complex), by the managed routes' split: a
        ///     call with an index or an inverse, and every float16 call, is NumPy's argsort and gather
        ///     (<see cref="uniqueFlatArgsortedLong{T}"/>); a float32 / float64 values-only or counts-only call is NumPy's
        ///     in-place sort, here the NaNs partitioned to the tail, the non-NaN portion sorted with the default compare,
        ///     the NaN tail's records put back in input order, and the tail written as NumPy's sort NaN.
        /// </summary>
        /// <typeparam name="T">The operand's element type (Half, float or double).</typeparam>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="n">Its element count.</param>
        /// <param name="equal_nan">True: every NaN is one value; false: each NaN is a value of its own.</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="EmitOutputsLong{T}"/>).</returns>
        /// <remarks>Reached only above <see cref="Array.MaxLength"/> elements (and by reflection from the tests).</remarks>
        private static unsafe NDArray[] uniqueFlatSortedLongFloat<T>(NDArray a, long n, bool equal_nan,
            bool return_index, bool return_inverse, bool return_counts)
            where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            // An index, an inverse, or float16: the statements the managed route runs (uniqueFlatArgsorted).
            if (return_index || return_inverse || typeof(T) == typeof(Half))
                return uniqueFlatArgsortedLong<T>(a, n, equal_nan, return_index, return_inverse, return_counts);

            var block = new UnmanagedMemoryBlock<KeyPerm<T>>(n);
            KeyPerm<T>* kp = block.Address;
            PopulateKeyPerm(a, kp, n);

            long firstNaN = PartitionNaN_Long<T>(kp, n);
            Utilities.LongIntroSort.Sort(kp, firstNaN);
            StabilizeNaNTailLong<T>(kp, firstNaN, n);
            // NumPy's values-only and counts-only routes sort in place, which rewrites every float32 / float64 NaN
            // (WriteSortNaNs); the index and inverse routes argsort and keep the bits (they returned above, so the test
            // always holds here).
            if (!return_index && !return_inverse)
                WriteSortNaNsLong(kp, firstNaN, n);

            return BuildMaskAndEmitLong<T>(a, kp, n, firstNaN, equal_nan,
                                            return_index, return_inverse, return_counts);
        }

        /// <summary>
        ///     <see cref="WriteSortNaNs(double[], long, long)"/> for the long-indexed records: a float32 or float64 NaN tail
        ///     of a line longer than one element becomes NumPy's sort NaN; a float16 tail is left as it is (NumPy's float16
        ///     sort keeps the bits).
        /// </summary>
        /// <typeparam name="T">The key type (Half, float or double); only float and double are rewritten.</typeparam>
        /// <param name="kp">The records; the keys of the tail <c>[firstNaN, n)</c> are overwritten in place.</param>
        /// <param name="firstNaN">The index where the NaN tail starts.</param>
        /// <param name="n">The number of records: one record keeps its NaN.</param>
        private static unsafe void WriteSortNaNsLong<T>(KeyPerm<T>* kp, long firstNaN, long n) where T : unmanaged, IComparable<T>
        {
            if (n <= 1) return;
            // The typeof tests are JIT constants: one branch survives per instantiation.
            if (typeof(T) == typeof(double))
            {
                double nan = NumPySortNaN64;
                for (long i = firstNaN; i < n; i++) kp[i].Key = System.Runtime.CompilerServices.Unsafe.As<double, T>(ref nan);
            }
            else if (typeof(T) == typeof(float))
            {
                float nan = NumPySortNaN32;
                for (long i = firstNaN; i < n; i++) kp[i].Key = System.Runtime.CompilerServices.Unsafe.As<float, T>(ref nan);
            }
        }

        /// <summary>
        ///     The long-indexed form of <see cref="uniqueFlatArgsorted{T}"/> for float16 / float32 / float64: NumPy's
        ///     argsort-and-gather statements over unmanaged records. <c>perm = ar.argsort()</c> is NumSharp's stable argsort
        ///     (its radix scratch is unmanaged, so a line of any length), <c>aux = ar[perm]</c> fills the records (every
        ///     value keeps its bits), the NaNs sorted last are the tail, and <see cref="BuildMaskAndEmitLong{T}"/> builds the
        ///     outputs.
        /// </summary>
        /// <typeparam name="T">Half, float or double (the operand's element type).</typeparam>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="n">Its element count (positive).</param>
        /// <param name="equal_nan">True: the NaNs are one run from the first of them in sorted order; false: a run each.</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="EmitOutputsLong{T}"/>).</returns>
        /// <remarks>
        ///     <see cref="EmitOutputsLong{T}"/>'s index is each run's smallest position. Under the stable order that is the
        ///     position at the run's start, NumPy's <c>perm[mask]</c>: equal values keep their input order, and so do the
        ///     NaNs of a collapsed run. Reached only above <see cref="Array.MaxLength"/> elements (and by reflection from the
        ///     tests); the record block is not disposed, its buffer returns through the block's finalizer.
        /// </remarks>
        private static unsafe NDArray[] uniqueFlatArgsortedLong<T>(NDArray a, long n, bool equal_nan,
            bool return_index, bool return_inverse, bool return_counts)
            where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            var block = new UnmanagedMemoryBlock<KeyPerm<T>>(n);
            KeyPerm<T>* kp = block.Address;
            using (var order = AxisSort.ArgSort(a, null, NPY_SORTKIND.NPY_STABLESORT))
            {
                long* perm = (long*)order.Address;   // a fresh C-contiguous int64 array: its first element is at its base
                if (a.Shape.IsContiguous)
                {
                    // Base address PLUS the shape's offset: a contiguous view may keep the base with a non-zero offset.
                    T* src = (T*)a.Address + a.Shape.offset;
                    for (long j = 0; j < n; j++) { long p = perm[j]; kp[j].Key = src[p]; kp[j].Perm = p; }
                }
                else
                {
                    // perm indexes the C-order flattening; a.flat is that flattening (a copy for this layout).
                    using var flat = a.flat;
                    T* src = (T*)flat.Address;
                    Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                    for (long j = 0; j < n; j++) { long p = perm[j]; kp[j].Key = src[getOffset(p)]; kp[j].Perm = p; }
                }
            }
            long firstNaN = n;
            while (firstNaN > 0 && IsNaNKey(kp[firstNaN - 1].Key)) firstNaN--;
            return BuildMaskAndEmitLong<T>(a, kp, n, firstNaN, equal_nan, return_index, return_inverse, return_counts);
        }

        /// <summary>
        ///     The long-indexed form of <see cref="uniqueFlatArgsorted{T}"/> for complex128: the stable argsort (NumPy's
        ///     complex order: the NaN classes <c>R+nanj</c>, <c>nan+Rj</c>, <c>nan+nanj</c> after the other values), the
        ///     gather into <see cref="ComplexKeyPerm"/> records, and <see cref="BuildMaskAndEmitLongComplex"/> with the
        ///     index at each run's start (NumPy's <c>perm[mask]</c>: a collapsed run of values with a NaN part is
        ///     represented by the first of them in sorted order, not by the first in input order).
        /// </summary>
        /// <param name="a">The complex128 operand (any layout; only read).</param>
        /// <param name="n">Its element count (positive).</param>
        /// <param name="equal_nan">True: the values with a NaN part are one run; false: a run each.</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="BuildMaskAndEmitLongComplex"/>).</returns>
        /// <remarks>Reached only above <see cref="Array.MaxLength"/> elements (and by reflection from the tests); the record
        ///     block is not disposed, its buffer returns through the block's finalizer.</remarks>
        private static unsafe NDArray[] uniqueFlatArgsortedLongComplex(NDArray a, long n, bool equal_nan,
            bool return_index, bool return_inverse, bool return_counts)
        {
            var block = new UnmanagedMemoryBlock<ComplexKeyPerm>(n);
            ComplexKeyPerm* kp = block.Address;
            using (var order = AxisSort.ArgSort(a, null, NPY_SORTKIND.NPY_STABLESORT))
            {
                long* perm = (long*)order.Address;
                if (a.Shape.IsContiguous)
                {
                    Complex* src = (Complex*)a.Address + a.Shape.offset;
                    for (long j = 0; j < n; j++) { long p = perm[j]; kp[j].Key = src[p]; kp[j].Perm = p; }
                }
                else
                {
                    using var flat = a.flat;
                    Complex* src = (Complex*)flat.Address;
                    Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                    for (long j = 0; j < n; j++) { long p = perm[j]; kp[j].Key = src[getOffset(p)]; kp[j].Perm = p; }
                }
            }
            long firstNaN = n;
            while (firstNaN > 0 && IsNaNKey(kp[firstNaN - 1].Key)) firstNaN--;
            return BuildMaskAndEmitLongComplex(a, kp, n, firstNaN, equal_nan, return_index, return_inverse, return_counts);
        }

        /// <summary>
        ///     Long-indexed unique for Complex, the managed routes' answers over unmanaged records: a call with an index, an
        ///     inverse or counts is NumPy's argsort and gather (<see cref="uniqueFlatArgsortedLongComplex"/>); a values-only
        ///     call is the hash path's answer from the same stable argsort, a collapsed NaN class represented by its first
        ///     value in C order (as <see cref="uniqueValuesOnlyComplex"/> gives it).
        /// </summary>
        /// <param name="a">The complex128 operand (any layout; only read).</param>
        /// <param name="n">Its element count.</param>
        /// <param name="equal_nan">True: every value with a NaN part is one value; false: each is a value of its own.</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="BuildMaskAndEmitLongComplex"/>).</returns>
        /// <remarks>Reached only above <see cref="Array.MaxLength"/> elements (and by reflection from the tests).</remarks>
        private static unsafe NDArray[] uniqueFlatSortedLongComplex(NDArray a, long n, bool equal_nan,
            bool return_index, bool return_inverse, bool return_counts)
        {
            // An index, an inverse or counts: the statements the managed route runs (uniqueFlatArgsorted).
            if (return_index || return_inverse || return_counts)
                return uniqueFlatArgsortedLongComplex(a, n, equal_nan, return_index, return_inverse, return_counts);

            // Values only: the hash path's answer (see uniqueValuesOnlyComplex).
            var r = uniqueFlatArgsortedLongComplex(a, n, equal_nan, false, false, false);
            if (equal_nan)
                KeepFirstNaNPartInCOrder(r[0], a);
            return r;
        }

        /// <summary>A complex128 key and its input position: the long-indexed complex path's sort record.</summary>
        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        private struct ComplexKeyPerm
        {
            /// <summary>The element's value.</summary>
            public Complex Key;

            /// <summary>The element's flat position in the input (C order).</summary>
            public long Perm;
        }

        /// <summary>
        ///     The mask and the outputs of the long-indexed complex path (<see cref="BuildMaskAndEmitLong{T}"/> and
        ///     <see cref="EmitOutputsLong{T}"/> in one, reading <see cref="ComplexKeyPerm"/> records).
        /// </summary>
        /// <param name="a">The operand: for more than one dimension the inverse takes its dimensions.</param>
        /// <param name="kp">The records: the sorted prefix, then the tail of values with a NaN part.</param>
        /// <param name="n">The number of records (positive).</param>
        /// <param name="firstNaN">The index where the NaN tail starts.</param>
        /// <param name="equal_nan">True starts one run at the tail; false a run at every tail record.</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <returns>[values, index?, inverse?, counts?]: fresh arrays, the index outputs int64. Each index is the position
        ///     at its run's start, NumPy's <c>perm[mask]</c>: the records are in NumPy's argsort order, and a collapsed run
        ///     of values with a NaN part, which holds different values, is represented by the first of them in that
        ///     order.</returns>
        /// <remarks>The mask block is not disposed: its buffer returns through the block's finalizer.</remarks>
        private static unsafe NDArray[] BuildMaskAndEmitLongComplex(
            NDArray a, ComplexKeyPerm* kp, long n, long firstNaN, bool equal_nan,
            bool return_index, bool return_inverse, bool return_counts)
        {
            long nanStart = firstNaN;

            var maskBlock = new UnmanagedMemoryBlock<byte>(n);
            byte* mask = maskBlock.Address;
            for (long i = 0; i < n; i++) mask[i] = 0;
            mask[0] = 1;
            long uniqueCount = 1;

            for (long i = 1; i < nanStart; i++)
            {
                if (!kp[i].Key.Equals(kp[i - 1].Key)) { mask[i] = 1; uniqueCount++; }
            }

            if (nanStart < n)
            {
                mask[nanStart] = 1;
                if (nanStart > 0) uniqueCount++;
                if (!equal_nan)
                {
                    for (long i = nanStart + 1; i < n; i++)
                    {
                        mask[i] = 1;
                        uniqueCount++;
                    }
                }
            }

            // Inline emit (same pattern as EmitOutputsLong but reads from ComplexKeyPerm)
            int outCount = 1 + (return_index ? 1 : 0) + (return_inverse ? 1 : 0) + (return_counts ? 1 : 0);
            var results = new NDArray[outCount];

            var valuesBlock = new UnmanagedMemoryBlock<Complex>(uniqueCount);
            var valuesSlice = new ArraySlice<Complex>(valuesBlock);
            Complex* vDst = valuesBlock.Address;
            long vIdx = 0;
            for (long i = 0; i < n; i++) if (mask[i] != 0) vDst[vIdx++] = kp[i].Key;
            results[0] = new NDArray(valuesSlice, Shape.Vector(uniqueCount));
            int outPos = 1;

            if (return_index)
            {
                // perm[mask]: the position of each run's first record.
                var idxBlock = new UnmanagedMemoryBlock<long>(uniqueCount);
                long* iDst = idxBlock.Address;
                long oIdx = 0;
                for (long i = 0; i < n; i++)
                    if (mask[i] != 0) iDst[oIdx++] = kp[i].Perm;
                results[outPos++] = new NDArray(new ArraySlice<long>(idxBlock), Shape.Vector(uniqueCount));
            }

            if (return_inverse)
            {
                var invBlock = new UnmanagedMemoryBlock<long>(n);
                var invSlice = new ArraySlice<long>(invBlock);
                long* invDst = invBlock.Address;
                long rank = -1;
                for (long i = 0; i < n; i++)
                {
                    if (mask[i] != 0) rank++;
                    invDst[kp[i].Perm] = rank;
                }
                var invShape = a.ndim <= 1 ? Shape.Vector(n) : Shape.Adopt(a.Storage.Shape.dimensions);
                results[outPos++] = new NDArray(invSlice, invShape);
            }

            if (return_counts)
            {
                var cntBlock = new UnmanagedMemoryBlock<long>(uniqueCount);
                var cntSlice = new ArraySlice<long>(cntBlock);
                long* cDst = cntBlock.Address;
                long prevPos = 0, cIdx = 0;
                for (long i = 1; i < n; i++)
                {
                    if (mask[i] != 0) { cDst[cIdx++] = i - prevPos; prevPos = i; }
                }
                cDst[cIdx++] = n - prevPos;
                results[outPos++] = new NDArray(cntSlice, Shape.Vector(uniqueCount));
            }

            return results;
        }

        /// <summary>
        ///     Fills the long-indexed sort records with the operand's elements in C order and their positions: a
        ///     C-contiguous operand from its first element (base address plus offset), any other through its strides.
        /// </summary>
        /// <typeparam name="T">The operand's element type.</typeparam>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="kp">The record block to fill (<paramref name="n"/> records).</param>
        /// <param name="n">The element count.</param>
        private static unsafe void PopulateKeyPerm<T>(NDArray a, KeyPerm<T>* kp, long n) where T : unmanaged, IComparable<T>
        {
            if (a.Shape.IsContiguous)
            {
                // Base address PLUS the shape's offset: a contiguous view may keep the base with a non-zero offset.
                T* src = (T*)a.Address + a.Shape.offset;
                for (long i = 0; i < n; i++) { kp[i].Key = src[i]; kp[i].Perm = i; }
            }
            else
            {
                var flat = a.flat;
                T* src = (T*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < n; i++) { kp[i].Key = src[getOffset(i)]; kp[i].Perm = i; }
            }
        }

        /// <summary>
        ///     NumPy's <c>np.isnan</c> of a key (a NaN part, for complex): the <c>typeof(T)</c> tests are JIT constants, so
        ///     one branch survives per instantiation, and a type without NaN folds to false. Shared by the long-indexed
        ///     path and <see cref="uniqueFlatArgsorted{T}"/>.
        /// </summary>
        /// <typeparam name="T">The key type.</typeparam>
        /// <param name="key">The key.</param>
        /// <returns>True for a NaN key.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe bool IsNaNKey<T>(T key) where T : unmanaged
        {
            if (typeof(T) == typeof(double)) return double.IsNaN((double)(object)key);
            if (typeof(T) == typeof(float)) return float.IsNaN((float)(object)key);
            if (typeof(T) == typeof(Half)) return Half.IsNaN((Half)(object)key);
            if (typeof(T) == typeof(Complex))
            {
                var c = (Complex)(object)key;
                return double.IsNaN(c.Real) || double.IsNaN(c.Imaginary);
            }
            return false;
        }

        /// <summary>
        ///     Moves every NaN record to the tail of the long-indexed record block (two-pointer swap, unstable).
        /// </summary>
        /// <typeparam name="T">The key type.</typeparam>
        /// <param name="kp">The records, partitioned in place.</param>
        /// <param name="n">The number of records.</param>
        /// <returns>The index of the first NaN record (<paramref name="n"/> when there is none).</returns>
        private static unsafe long PartitionNaN_Long<T>(KeyPerm<T>* kp, long n)
            where T : unmanaged, IComparable<T>
        {
            long hi = n;
            long i = 0;
            while (i < hi)
            {
                if (IsNaNKey(kp[i].Key))
                {
                    hi--;
                    (kp[i], kp[hi]) = (kp[hi], kp[i]);
                }
                else i++;
            }
            return hi;
        }

        /// <summary>
        ///     Sort the NaN-tail's perm ascending (keys are all NaN, order irrelevant)
        ///     to match NumPy's stable mergesort semantics. We sort the KeyPerm structs
        ///     by Perm using LongIntroSort with a comparer.
        /// </summary>
        /// <typeparam name="T">The key type.</typeparam>
        /// <param name="kp">The records; the tail from <paramref name="firstNaN"/> is sorted by position in place.</param>
        /// <param name="firstNaN">The index where the NaN tail starts.</param>
        /// <param name="n">The number of records; a tail of fewer than two records is left as it is.</param>
        private static unsafe void StabilizeNaNTailLong<T>(KeyPerm<T>* kp, long firstNaN, long n)
            where T : unmanaged, IComparable<T>
        {
            if (firstNaN >= n - 1) return;
            Utilities.LongIntroSort.Sort(kp + firstNaN, n - firstNaN,
                (x, y) => x.Perm.CompareTo(y.Perm));
        }

        /// <summary>
        ///     Long-indexed mask + emit: the run starts of the sorted records (IEEE equality on the non-NaN prefix; the NaN
        ///     tail one run under <paramref name="equal_nan"/>, else a run per record) in an
        ///     UnmanagedMemoryBlock&lt;byte&gt;, then <see cref="EmitOutputsLong{T}"/>.
        /// </summary>
        /// <typeparam name="T">The key type.</typeparam>
        /// <param name="a">The operand: for more than one dimension the inverse takes its dimensions.</param>
        /// <param name="kp">The sorted records (a NaN tail after <paramref name="firstNaN"/> when it is not -1).</param>
        /// <param name="n">The number of records (positive).</param>
        /// <param name="firstNaN">The index where the NaN tail starts, or -1 for a type without NaN.</param>
        /// <param name="equal_nan">True starts one run at the tail; false a run at every tail record.</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <returns>[values, index?, inverse?, counts?] (see <see cref="EmitOutputsLong{T}"/>).</returns>
        /// <remarks>The mask block is not disposed: its buffer returns through the block's finalizer.</remarks>
        private static unsafe NDArray[] BuildMaskAndEmitLong<T>(
            NDArray a, KeyPerm<T>* kp, long n, long firstNaN, bool equal_nan,
            bool return_index, bool return_inverse, bool return_counts)
            where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            // For non-float (firstNaN == -1), treat as "no NaN section at all" → firstNaN = n.
            long nanStart = firstNaN >= 0 ? firstNaN : n;

            var maskBlock = new UnmanagedMemoryBlock<byte>(n);
            byte* mask = maskBlock.Address;
            // Zero-init isn't guaranteed; clear explicitly.
            for (long i = 0; i < n; i++) mask[i] = 0;
            mask[0] = 1;
            long uniqueCount = 1;

            for (long i = 1; i < nanStart; i++)
            {
                if (!kp[i].Key.Equals(kp[i - 1].Key)) { mask[i] = 1; uniqueCount++; }
            }

            if (nanStart < n)
            {
                mask[nanStart] = 1;
                if (nanStart > 0) uniqueCount++;
                if (!equal_nan)
                {
                    for (long i = nanStart + 1; i < n; i++)
                    {
                        mask[i] = 1;
                        uniqueCount++;
                    }
                }
            }

            return EmitOutputsLong<T>(a, kp, mask, n, uniqueCount, return_index, return_inverse, return_counts);
        }

        /// <summary>
        ///     The long-indexed form of <see cref="EmitOutputs{T}"/>: the values, the index (smallest position of each run),
        ///     the inverse and the counts, read from the sorted records and the byte mask.
        /// </summary>
        /// <typeparam name="T">The key type.</typeparam>
        /// <param name="a">The operand: for more than one dimension the inverse takes its dimensions.</param>
        /// <param name="kp">The sorted records.</param>
        /// <param name="mask">Nonzero where a run starts (<c>mask[0]</c> nonzero).</param>
        /// <param name="n">The number of records (positive).</param>
        /// <param name="uniqueCount">The number of nonzero mask entries.</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <returns>[values, index?, inverse?, counts?]: fresh arrays, the index outputs int64.</returns>
        private static unsafe NDArray[] EmitOutputsLong<T>(
            NDArray a, KeyPerm<T>* kp, byte* mask, long n, long uniqueCount,
            bool return_index, bool return_inverse, bool return_counts)
            where T : unmanaged, IComparable<T>
        {
            int outCount = 1 + (return_index ? 1 : 0) + (return_inverse ? 1 : 0) + (return_counts ? 1 : 0);
            var results = new NDArray[outCount];

            // values
            var valuesBlock = new UnmanagedMemoryBlock<T>(uniqueCount);
            var valuesSlice = new ArraySlice<T>(valuesBlock);
            T* vDst = valuesBlock.Address;
            long vIdx = 0;
            for (long i = 0; i < n; i++)
                if (mask[i] != 0) vDst[vIdx++] = kp[i].Key;
            results[0] = new NDArray(valuesSlice, Shape.Vector(uniqueCount));
            int outPos = 1;

            // index — min(perm) within each run of equal keys
            if (return_index)
            {
                var idxBlock = new UnmanagedMemoryBlock<long>(uniqueCount);
                var idxSlice = new ArraySlice<long>(idxBlock);
                long* iDst = idxBlock.Address;
                long oIdx = 0;
                long currentMin = kp[0].Perm;
                for (long i = 1; i < n; i++)
                {
                    if (mask[i] != 0)
                    {
                        iDst[oIdx++] = currentMin;
                        currentMin = kp[i].Perm;
                    }
                    else if (kp[i].Perm < currentMin)
                    {
                        currentMin = kp[i].Perm;
                    }
                }
                iDst[oIdx++] = currentMin;
                results[outPos++] = new NDArray(idxSlice, Shape.Vector(uniqueCount));
            }

            // inverse — inv[perm[i]] = cumsum(mask)[i] - 1
            if (return_inverse)
            {
                var invBlock = new UnmanagedMemoryBlock<long>(n);
                var invSlice = new ArraySlice<long>(invBlock);
                long* invDst = invBlock.Address;
                long rank = -1;
                for (long i = 0; i < n; i++)
                {
                    if (mask[i] != 0) rank++;
                    invDst[kp[i].Perm] = rank;
                }
                var invShape = a.ndim <= 1 ? Shape.Vector(n) : Shape.Adopt(a.Storage.Shape.dimensions);
                results[outPos++] = new NDArray(invSlice, invShape);
            }

            // counts
            if (return_counts)
            {
                var cntBlock = new UnmanagedMemoryBlock<long>(uniqueCount);
                var cntSlice = new ArraySlice<long>(cntBlock);
                long* cDst = cntBlock.Address;
                long prevPos = 0;
                long cIdx = 0;
                for (long i = 1; i < n; i++)
                {
                    if (mask[i] != 0)
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
    }
}
