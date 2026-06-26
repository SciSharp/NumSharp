using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Result of a memory-overlap query. Matches NumPy's <c>mem_overlap_t</c>
    /// (mem_overlap.h:15-21).
    /// </summary>
    public enum MemOverlap
    {
        /// <summary>No solution exists — the arrays provably do not share memory.</summary>
        No = 0,

        /// <summary>A solution was found — the arrays share at least one byte.</summary>
        Yes = 1,

        /// <summary>max_work exceeded — undecided, treat as "may share".</summary>
        TooHard = -1,

        /// <summary>Algorithm failed due to integer overflow — treat as "may share".</summary>
        Overflow = -2,

        /// <summary>Invalid input.</summary>
        Error = -3,
    }

    /// <summary>
    /// Solver for array memory-overlap queries: bounded Diophantine equations
    /// with positive coefficients, solved by GCD-pruned depth-first search.
    ///
    /// Faithful port of NumPy's <c>numpy/_core/src/common/mem_overlap.c</c>
    /// (Pauli Virtanen, 3-clause BSD). Asking whether two strided arrays
    /// overlap is equivalent to asking whether
    ///
    ///     sum(stride_a[i] * x_a[i]) - sum(stride_b[i] * x_b[i]) == base_b - base_a
    ///     with 0 &lt;= x[i] &lt; shape[i]
    ///
    /// has an integer solution; itemsize contributes one extra term with
    /// stride 1. Negative strides are mapped positive by the variable change
    /// x → shape-1-x, so the problem becomes a bounded Diophantine equation
    /// with positive coefficients. The general problem is NP-hard, so the
    /// amount of work is capped by <c>maxWork</c>:
    ///
    ///   maxWork == 0  → bounds check only (NumPy NPY_MAY_SHARE_BOUNDS)
    ///   maxWork == -1 → exact solution    (NumPy NPY_MAY_SHARE_EXACT)
    ///   maxWork &gt; 0 → at most maxWork solution candidates are examined
    ///
    /// NumPy's 128-bit helper arithmetic (npy_extint128.h) maps to
    /// <see cref="Int128"/>; products of two int64 magnitudes stay below
    /// 2^126 so intermediate Int128 arithmetic cannot overflow — only the
    /// final narrowing back to 64-bit is range-checked, as in NumPy.
    /// </summary>
    internal static unsafe class NpyMemOverlap
    {
        /// <summary>One term of the Diophantine problem: coefficient A (&gt; 0) and upper bound Ub.</summary>
        internal struct DiophantineTerm
        {
            public long A;
            public long Ub;
        }

        // =====================================================================
        // Public entry points
        // =====================================================================

        /// <summary>
        /// Determine whether two arrays may share memory.
        /// Port of <c>solve_may_share_memory</c> (mem_overlap.c:757).
        /// </summary>
        /// <param name="a">First array (any layout, including views and broadcasts).</param>
        /// <param name="b">Second array.</param>
        /// <param name="maxWork">Work cap: 0 = bounds only, -1 = exact, &gt;0 = candidate cap.</param>
        public static MemOverlap SolveMayShareMemory(NDArray a, NDArray b, long maxWork)
        {
            GetMemoryExtents(a, out ulong start1, out ulong end1);
            GetMemoryExtents(b, out ulong start2, out ulong end2);

            if (!(start1 < end2 && start2 < end1 && start1 < end1 && start2 < end2))
            {
                // Memory extents don't overlap (also covers empty arrays).
                return MemOverlap.No;
            }

            if (maxWork == 0)
            {
                // Bounds check only — too much work to decide exactly.
                return MemOverlap.TooHard;
            }

            // Convert to a Diophantine problem with positive coefficients.
            // RHS: pick the smaller of the two equivalent formulations
            // (mem_overlap.c:782-810).
            ulong uintpRhs = Math.Min(end2 - 1 - start1, end1 - 1 - start2);
            if (uintpRhs > long.MaxValue)
                return MemOverlap.Overflow;
            long rhs = (long)uintpRhs;

            int maxTerms = a.ndim + b.ndim + 2;
            Span<DiophantineTerm> terms = maxTerms <= 32
                ? stackalloc DiophantineTerm[32]
                : new DiophantineTerm[maxTerms];

            int nterms = 0;
            if (StridesToTerms(a, terms, ref nterms, skipEmpty: true))
                return MemOverlap.Overflow;
            if (StridesToTerms(b, terms, ref nterms, skipEmpty: true))
                return MemOverlap.Overflow;

            int itemsizeA = a.GetTypeCode.SizeOf();
            int itemsizeB = b.GetTypeCode.SizeOf();
            if (itemsizeA > 1)
            {
                terms[nterms].A = 1;
                terms[nterms].Ub = itemsizeA - 1;
                nterms++;
            }
            if (itemsizeB > 1)
            {
                terms[nterms].A = 1;
                terms[nterms].Ub = itemsizeB - 1;
                nterms++;
            }

            if (DiophantineSimplify(ref nterms, terms, rhs))
                return MemOverlap.Overflow;

            Span<long> x = nterms <= 32 ? stackalloc long[32] : new long[nterms];
            return SolveDiophantine(nterms, terms, rhs, maxWork, requireUbNontrivial: false, x);
        }

        /// <summary>
        /// Determine whether an array can map two different logical indices to
        /// the same memory address (e.g. a stride-0 broadcast on a non-unit
        /// dimension). Port of <c>solve_may_have_internal_overlap</c>
        /// (mem_overlap.c:848).
        /// </summary>
        public static MemOverlap SolveMayHaveInternalOverlap(NDArray a, long maxWork)
        {
            if (a.Shape.IsContiguous)
            {
                // Quick case (PyArray_ISCONTIGUOUS).
                return MemOverlap.No;
            }

            int maxTerms = a.ndim + 1;
            Span<DiophantineTerm> terms = maxTerms <= 32
                ? stackalloc DiophantineTerm[32]
                : new DiophantineTerm[maxTerms];

            // All dims participate (skipEmpty: false) — stride-0 dims decide YES below.
            int nterms = 0;
            if (StridesToTerms(a, terms, ref nterms, skipEmpty: false))
                return MemOverlap.Overflow;

            int itemsize = a.GetTypeCode.SizeOf();
            if (itemsize > 1)
            {
                terms[nterms].A = 1;
                terms[nterms].Ub = itemsize - 1;
                nterms++;
            }

            // Remove zero coefficients and empty terms (mem_overlap.c:893-910).
            int k = 0;
            for (int j = 0; j < nterms; j++)
            {
                if (terms[j].Ub == 0)
                    continue;
                if (terms[j].Ub < 0)
                    return MemOverlap.No;
                if (terms[j].A == 0)
                    return MemOverlap.Yes; // stride-0 on a non-unit dimension
                if (k != j)
                    terms[k] = terms[j];
                k++;
            }
            nterms = k;

            // Double bounds to get the internal-overlap problem.
            for (int j = 0; j < nterms; j++)
                terms[j].Ub *= 2;

            // Sort by descending coefficient; diophantine_simplify must NOT be
            // called here because it may change the decision problem.
            SortTermsDescending(terms, nterms);

            Span<long> x = nterms <= 32 ? stackalloc long[32] : new long[nterms];
            return SolveDiophantine(nterms, terms, b: -1, maxWork, requireUbNontrivial: true, x);
        }

        // =====================================================================
        // Array → problem conversion
        // =====================================================================

        /// <summary>
        /// Half-open byte range [start, end) occupied by the array's elements.
        /// Port of <c>offset_bounds_from_strides</c> + <c>get_array_memory_extents</c>
        /// (mem_overlap.c:659-710). An empty array yields start == end.
        /// </summary>
        internal static void GetMemoryExtents(NDArray arr, out ulong start, out ulong end)
        {
            var shape = arr.Shape;
            int itemsize = arr.GetTypeCode.SizeOf();
            // First-element address of the view: buffer base + element offset.
            ulong baseAddr = (ulong)arr.Address + (ulong)((long)shape.offset * itemsize);

            long lower = 0, upper = 0;
            int nd = shape.NDim;
            var dims = shape.dimensions;
            var strides = shape.strides;
            for (int i = 0; i < nd; i++)
            {
                if (dims[i] == 0)
                {
                    // Zero-size array occupies no bytes.
                    start = 0;
                    end = 0;
                    return;
                }
                long maxAxisOffset = (long)strides[i] * itemsize * (dims[i] - 1);
                if (maxAxisOffset > 0)
                    upper += maxAxisOffset;
                else
                    lower += maxAxisOffset;
            }
            upper += itemsize; // half-open
            start = baseAddr + (ulong)lower;
            end = baseAddr + (ulong)upper;
        }

        /// <summary>
        /// Bounds-only overlap test for two storages — the cheap NPY_MAY_SHARE_BOUNDS
        /// check <c>PyArray_AssignArray</c> uses (<c>arrays_overlap</c>). Conservative:
        /// returns true whenever the two byte extents intersect, without solving exact
        /// stride collisions. Operates directly on <see cref="UnmanagedStorage"/> so it
        /// allocates nothing and has NO reference-counting side effects (wrapping a
        /// storage in a throwaway <see cref="NDArray"/> would retain its buffer).
        /// </summary>
        internal static bool StoragesMayShareMemory(UnmanagedStorage a, UnmanagedStorage b)
        {
            StorageExtent(a, out ulong s1, out ulong e1);
            StorageExtent(b, out ulong s2, out ulong e2);
            // Non-empty and intersecting (matches the extent test in SolveMayShareMemory).
            return s1 < e2 && s2 < e1 && s1 < e1 && s2 < e2;
        }

        private static void StorageExtent(UnmanagedStorage st, out ulong start, out ulong end)
        {
            var shape = st.Shape;
            int itemsize = st.TypeCode.SizeOf();
            ulong baseAddr = (ulong)st.Address + (ulong)((long)shape.offset * itemsize);

            long lower = 0, upper = 0;
            int nd = (int)shape.NDim;
            var dims = shape.dimensions;
            var strides = shape.strides;
            for (int i = 0; i < nd; i++)
            {
                if (dims[i] == 0)
                {
                    start = 0;
                    end = 0;
                    return;
                }
                long maxAxisOffset = (long)strides[i] * itemsize * (dims[i] - 1);
                if (maxAxisOffset > 0)
                    upper += maxAxisOffset;
                else
                    lower += maxAxisOffset;
            }
            upper += itemsize; // half-open
            start = baseAddr + (ulong)lower;
            end = baseAddr + (ulong)upper;
        }

        /// <summary>
        /// Append one positive-coefficient term per dimension (|byte stride|, dim-1).
        /// Port of <c>strides_to_terms</c> (mem_overlap.c:713). Returns true on
        /// integer overflow.
        /// </summary>
        private static bool StridesToTerms(NDArray arr, Span<DiophantineTerm> terms, ref int nterms, bool skipEmpty)
        {
            var shape = arr.Shape;
            int itemsize = arr.GetTypeCode.SizeOf();
            int nd = shape.NDim;
            var dims = shape.dimensions;
            var strides = shape.strides;

            for (int i = 0; i < nd; i++)
            {
                long byteStride = (long)strides[i] * itemsize;
                if (skipEmpty)
                {
                    if (dims[i] <= 1 || byteStride == 0)
                        continue;
                }

                long a = byteStride < 0 ? -byteStride : byteStride;
                if (a < 0)
                    return true; // |long.MinValue| overflow

                terms[nterms].A = a;
                terms[nterms].Ub = dims[i] - 1;
                nterms++;
            }
            return false;
        }

        // =====================================================================
        // Diophantine machinery (mem_overlap.c:202-655)
        // =====================================================================

        /// <summary>
        /// Extended Euclid: solves gamma*a1 + epsilon*a2 == gcd(a1, a2) with
        /// |gamma| &lt; a2/gcd, |epsilon| &lt; a1/gcd (mem_overlap.c:208).
        /// </summary>
        private static void Euclid(long a1, long a2, out long aGcd, out long gamma, out long epsilon)
        {
            long gamma1 = 1, gamma2 = 0, epsilon1 = 0, epsilon2 = 1, r;

            while (true)
            {
                if (a2 > 0)
                {
                    r = a1 / a2;
                    a1 -= r * a2;
                    gamma1 -= r * gamma2;
                    epsilon1 -= r * epsilon2;
                }
                else
                {
                    aGcd = a1;
                    gamma = gamma1;
                    epsilon = epsilon1;
                    return;
                }

                if (a1 > 0)
                {
                    r = a2 / a1;
                    a2 -= r * a1;
                    gamma2 -= r * gamma1;
                    epsilon2 -= r * epsilon1;
                }
                else
                {
                    aGcd = a2;
                    gamma = gamma2;
                    epsilon = epsilon2;
                    return;
                }
            }
        }

        /// <summary>Precompute GCD chain and transformed bounds (mem_overlap.c:256). Returns true on overflow.</summary>
        private static bool DiophantinePrecompute(int n,
            ReadOnlySpan<DiophantineTerm> E, Span<DiophantineTerm> Ep,
            Span<long> Gamma, Span<long> Epsilon)
        {
            bool overflow = false;

            Euclid(E[0].A, E[1].A, out long aGcd, out long gamma, out long epsilon);
            Ep[0].A = aGcd;
            Gamma[0] = gamma;
            Epsilon[0] = epsilon;

            if (n > 2)
            {
                long c1 = E[0].A / aGcd;
                long c2 = E[1].A / aGcd;
                Ep[0].Ub = SafeAdd(SafeMul(E[0].Ub, c1, ref overflow),
                                   SafeMul(E[1].Ub, c2, ref overflow), ref overflow);
                if (overflow)
                    return true;
            }

            for (int j = 2; j < n; j++)
            {
                Euclid(Ep[j - 2].A, E[j].A, out aGcd, out gamma, out epsilon);
                Ep[j - 1].A = aGcd;
                Gamma[j - 1] = gamma;
                Epsilon[j - 1] = epsilon;

                if (j < n - 1)
                {
                    long c1 = Ep[j - 2].A / aGcd;
                    long c2 = E[j].A / aGcd;
                    Ep[j - 1].Ub = SafeAdd(SafeMul(c1, Ep[j - 2].Ub, ref overflow),
                                           SafeMul(c2, E[j].Ub, ref overflow), ref overflow);
                    if (overflow)
                        return true;
                }
            }
            return false;
        }

        /// <summary>Depth-first bounded Euclid search (mem_overlap.c:312).</summary>
        private static MemOverlap DiophantineDfs(int n, int v,
            ReadOnlySpan<DiophantineTerm> E, ReadOnlySpan<DiophantineTerm> Ep,
            ReadOnlySpan<long> Gamma, ReadOnlySpan<long> Epsilon,
            long b, long maxWork, bool requireUbNontrivial, Span<long> x, ref long count)
        {
            bool overflow = false;

            if (maxWork >= 0 && count >= maxWork)
                return MemOverlap.TooHard;

            // Fetch precomputed values for the reduced problem.
            long a1, u1;
            if (v == 1)
            {
                a1 = E[0].A;
                u1 = E[0].Ub;
            }
            else
            {
                a1 = Ep[v - 2].A;
                u1 = Ep[v - 2].Ub;
            }

            long a2 = E[v].A;
            long u2 = E[v].Ub;

            long aGcd = Ep[v - 1].A;
            long gamma = Gamma[v - 1];
            long epsilon = Epsilon[v - 1];

            // Generate the set of allowed solutions.
            long c = b / aGcd;
            long r = b % aGcd;
            if (r != 0)
            {
                count++;
                return MemOverlap.No;
            }

            long c1 = a2 / aGcd;
            long c2 = a1 / aGcd;

            // Solutions: x1 = gamma*c + c1*t, x2 = epsilon*c - c2*t for integer t
            // with 0 <= x1 <= u1 and 0 <= x2 <= u2. Intermediates need 128-bit.
            Int128 x10 = (Int128)gamma * c;
            Int128 x20 = (Int128)epsilon * c;

            Int128 tL1 = CeilDiv128(-x10, c1);
            Int128 tL2 = CeilDiv128(x20 - u2, c2);
            Int128 tU1 = FloorDiv128((Int128)u1 - x10, c1);
            Int128 tU2 = FloorDiv128(x20, c2);

            if (tL2 > tL1) tL1 = tL2;
            if (tU1 > tU2) tU1 = tU2;

            if (tL1 > tU1)
            {
                count++;
                return MemOverlap.No;
            }

            long tl = To64(tL1, ref overflow);
            long tu = To64(tU1, ref overflow);

            x10 += (Int128)c1 * tl;
            x20 -= (Int128)c2 * tl;

            tu = SafeSub(tu, tl, ref overflow);
            tl = 0;
            long x1 = To64(x10, ref overflow);
            long x2 = To64(x20, ref overflow);

            if (overflow)
                return MemOverlap.Overflow;

            if (v == 1)
            {
                // Base case.
                if (tu >= tl)
                {
                    x[0] = x1 + c1 * tl;
                    x[1] = x2 - c2 * tl;
                    if (requireUbNontrivial)
                    {
                        bool isUbTrivial = true;
                        for (int j = 0; j < n; j++)
                        {
                            if (x[j] != E[j].Ub / 2)
                            {
                                isUbTrivial = false;
                                break;
                            }
                        }
                        if (isUbTrivial)
                        {
                            // Ignore the 'trivial' solution.
                            count++;
                            return MemOverlap.No;
                        }
                    }
                    return MemOverlap.Yes;
                }
                count++;
                return MemOverlap.No;
            }

            // Recurse over all candidates.
            for (long t = tl; t <= tu; t++)
            {
                x[v] = x2 - c2 * t;

                long b2 = SafeSub(b, SafeMul(a2, x[v], ref overflow), ref overflow);
                if (overflow)
                    return MemOverlap.Overflow;

                var res = DiophantineDfs(n, v - 1, E, Ep, Gamma, Epsilon,
                                         b2, maxWork, requireUbNontrivial, x, ref count);
                if (res != MemOverlap.No)
                    return res;
            }
            count++;
            return MemOverlap.No;
        }

        /// <summary>
        /// Solve the bounded Diophantine equation
        /// <c>A[0] x[0] + ... + A[n-1] x[n-1] == b</c> with <c>0 ≤ x[i] ≤ U[i]</c>.
        /// Port of <c>solve_diophantine</c> (mem_overlap.c:482).
        ///
        /// When <paramref name="requireUbNontrivial"/> is set, looks for solutions
        /// to b = sum(A[i]*U[i]/2) other than the trivial x[i] = U[i]/2 (the
        /// internal-overlap formulation); <paramref name="b"/> is ignored.
        /// </summary>
        internal static MemOverlap SolveDiophantine(int n, ReadOnlySpan<DiophantineTerm> E, long b,
            long maxWork, bool requireUbNontrivial, Span<long> x)
        {
            for (int j = 0; j < n; j++)
            {
                if (E[j].A <= 0)
                    return MemOverlap.Error;
                if (E[j].Ub < 0)
                    return MemOverlap.No;
            }

            if (requireUbNontrivial)
            {
                long ubSum = 0;
                bool overflow = false;
                for (int j = 0; j < n; j++)
                {
                    if (E[j].Ub % 2 != 0)
                        return MemOverlap.Error;
                    ubSum = SafeAdd(ubSum, SafeMul(E[j].A, E[j].Ub / 2, ref overflow), ref overflow);
                }
                if (overflow)
                    return MemOverlap.Error;
                b = ubSum;
            }

            if (b < 0)
                return MemOverlap.No;

            if (n == 0)
            {
                if (requireUbNontrivial)
                    return MemOverlap.No; // only the trivial solution exists
                return b == 0 ? MemOverlap.Yes : MemOverlap.No;
            }

            if (n == 1)
            {
                if (requireUbNontrivial)
                    return MemOverlap.No; // only the trivial solution exists
                if (b % E[0].A == 0)
                {
                    x[0] = b / E[0].A;
                    if (x[0] >= 0 && x[0] <= E[0].Ub)
                        return MemOverlap.Yes;
                }
                return MemOverlap.No;
            }

            Span<DiophantineTerm> Ep = n <= 32 ? stackalloc DiophantineTerm[32] : new DiophantineTerm[n];
            Span<long> Gamma = n <= 32 ? stackalloc long[32] : new long[n];
            Span<long> Epsilon = n <= 32 ? stackalloc long[32] : new long[n];

            if (DiophantinePrecompute(n, E, Ep, Gamma, Epsilon))
                return MemOverlap.Overflow;

            long count = 0;
            return DiophantineDfs(n, n - 1, E, Ep, Gamma, Epsilon, b, maxWork,
                                  requireUbNontrivial, x, ref count);
        }

        /// <summary>
        /// Simplify the decision problem: sort by descending coefficient, merge
        /// identical coefficients, trim bounds, drop fixed variables. The
        /// feasible/infeasible answer is preserved. Port of
        /// <c>diophantine_simplify</c> (mem_overlap.c:596). Returns true on overflow.
        /// </summary>
        internal static bool DiophantineSimplify(ref int n, Span<DiophantineTerm> E, long b)
        {
            bool overflow = false;

            // Skip obviously infeasible cases.
            for (int j = 0; j < n; j++)
            {
                if (E[j].Ub < 0)
                    return false;
            }
            if (b < 0)
                return false;

            SortTermsDescending(E, n);

            // Combine identical coefficients.
            int m = n;
            int i = 0;
            for (int j = 1; j < m; j++)
            {
                if (E[i].A == E[j].A)
                {
                    E[i].Ub = SafeAdd(E[i].Ub, E[j].Ub, ref overflow);
                    n--;
                }
                else
                {
                    i++;
                    if (i != j)
                        E[i] = E[j];
                }
            }

            // Trim bounds and remove unnecessary variables.
            m = n;
            i = 0;
            for (int j = 0; j < m; j++)
            {
                E[j].Ub = Math.Min(E[j].Ub, b / E[j].A);
                if (E[j].Ub == 0)
                {
                    // If the problem is feasible at all, x[j] = 0 here.
                    n--;
                }
                else
                {
                    if (i != j)
                        E[i] = E[j];
                    i++;
                }
            }

            return overflow;
        }

        // =====================================================================
        // Helpers
        // =====================================================================

        /// <summary>Insertion sort by descending coefficient (n is small; avoids comparer allocations).</summary>
        private static void SortTermsDescending(Span<DiophantineTerm> E, int n)
        {
            for (int i = 1; i < n; i++)
            {
                var key = E[i];
                int j = i - 1;
                while (j >= 0 && E[j].A < key.A)
                {
                    E[j + 1] = E[j];
                    j--;
                }
                E[j + 1] = key;
            }
        }

        private static long SafeAdd(long a, long b, ref bool overflow)
        {
            Int128 r = (Int128)a + b;
            if (r > long.MaxValue || r < long.MinValue)
                overflow = true;
            return unchecked((long)r);
        }

        private static long SafeSub(long a, long b, ref bool overflow)
        {
            Int128 r = (Int128)a - b;
            if (r > long.MaxValue || r < long.MinValue)
                overflow = true;
            return unchecked((long)r);
        }

        private static long SafeMul(long a, long b, ref bool overflow)
        {
            Int128 r = (Int128)a * b;
            if (r > long.MaxValue || r < long.MinValue)
                overflow = true;
            return unchecked((long)r);
        }

        private static long To64(Int128 v, ref bool overflow)
        {
            if (v > long.MaxValue || v < long.MinValue)
                overflow = true;
            return unchecked((long)v);
        }

        /// <summary>Floor division of a 128-bit value by a positive 64-bit divisor.</summary>
        private static Int128 FloorDiv128(Int128 a, long b)
        {
            Int128 q = a / b;
            if (a % b != 0 && a < 0)
                q--;
            return q;
        }

        /// <summary>Ceiling division of a 128-bit value by a positive 64-bit divisor.</summary>
        private static Int128 CeilDiv128(Int128 a, long b)
        {
            Int128 q = a / b;
            if (a % b != 0 && a > 0)
                q++;
            return q;
        }
    }
}
