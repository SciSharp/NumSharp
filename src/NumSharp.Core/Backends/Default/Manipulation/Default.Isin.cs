using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using SysUnsafe = System.Runtime.CompilerServices.Unsafe;

// =============================================================================
// Default.Isin.cs - DefaultEngine's np.isin, after the facade's class boundary (numpy/lib/_arraysetops_impl.py, isin / _isin)
// =============================================================================
//
// WHAT NUMPY DOES
//   isin(element, test_elements) is `_isin(asarray(element), test_elements, ...).reshape(element.shape)`. _isin ravels both
//   operands, checks kind, and answers on one of three paths: the TABLE method (both arrays integer or bool, kind None or
//   'table', the set's range at most six times the two sizes or kind='table'), the LOOP path (len(ar2) < 10 *
//   len(ar1) ** 0.145, or an operand with objects: StringDType) and the SORT path (np.unique of each operand, one
//   concatenate, a stable argsort and the adjacent-equality step).
//
// HOW THIS FILE ANSWERS
//   StringDType and fixed-width text run NumPy's own statements (IsinVString in Default.Isin.VString.cs, IsinText here).
//   Every other dtype pair takes NumPy's path decision (NumPyPath). The table method and the loop path run as one-pass
//   kernels (TableMembership, LoopMembership), whose answer is NumPy's: the exact membership. The sort path runs NumPy's
//   statements where they are not the exact membership (SortPathFlat: a lossy concatenation, assume_unique). Everything
//   else, and each pair a kernel declines, computes the exact membership in result_type(ar1, ar2) by a hash set, a
//   counting table or a sorted search (ComputeMembership), an exact uint64 / signed pair included.
//
// DefaultEngine's override of TensorEngine.Isin (TensorEngine.ArraySetOps.cs), moved from the public np.isin by the
// np-layer dtype plan's P2 into the internal operation class IsinEngine, and made these DefaultEngine partials by its
// phase PE (E1: NumPy writes isin in Python, and such a function is a TensorEngine member that DefaultEngine implements).
// NumPy's statements call np.ravel, np.concatenate, np.argsort, np.equal, np.searchsorted, np.take, ...; here they are
// this engine's or the operand's engine members, the array object's (the ravel, the allocations, the concatenation, the
// take) and the dtype layer's promotion. Its private helpers keep their names: none is shared with another
// DefaultEngine member, so no overload set changed.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.Isin"/>: <c>isin(element, test_elements, assume_unique, invert,
        ///     kind)</c>, the <c>kind</c> check, then the <c>StringDType</c> route, the text route or the numeric
        ///     membership, the answer shaped like <paramref name="element"/>.
        /// </summary>
        /// <param name="element">The tested values (not null; the result has their shape).</param>
        /// <param name="test_elements">The set (not null; flattened).</param>
        /// <param name="assume_unique">NumPy's speed hint: where NumPy takes its sort path, the <c>np.unique</c> calls are
        ///     skipped, so repeated values get NumPy's answer for them (an element is flagged when an equal value follows it
        ///     in the elements or lies in the set); the table method and the loop path ignore it.</param>
        /// <param name="invert">Whether the answer is <c>element not in test_elements</c>.</param>
        /// <param name="kind">The algorithm name: null (automatic), <c>"sort"</c> or <c>"table"</c>.</param>
        /// <returns>A bool array of <paramref name="element"/>'s shape: a view of a fresh 1-D answer, as NumPy's reshape
        ///     returns it.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/> (<c>Invalid kind: 'x'. Please use None, 'sort' or
        ///     'table'.</c>), <c>kind='table'</c> for an array that is not boolean or integer, or <c>kind='table'</c> over a
        ///     set whose range is <c>2**63 - 1</c> or more (<c>Maximum allowed dimension exceeded</c>).</exception>
        /// <exception cref="InvalidOperationException"><c>kind='table'</c> with a set whose range exceeds its dtype's maximum
        ///     (NumPy's RuntimeError text).</exception>
        /// <exception cref="UnicodeDecodeError">The text route's sort path decodes <c>S</c> text holding a byte that is not
        ///     ASCII beside <c>U</c> text.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> set whose NA object is a sequence, held by an NA cell
        ///     (see <see cref="IsinVString"/>).</exception>
        /// <remarks>A public entry point since the plan's phase PE, so it owns its scope (it was <c>[NDScopedCovered]</c>
        ///     under np.isin's): the ravels, the promoted casts and the sort / table / hash work arrays are released here,
        ///     the answer's view escapes.</remarks>
        [NDScoped]
        public override NDArray Isin(NDArray element, NDArray test_elements, bool assume_unique = false, bool invert = false,
            string kind = null)
        {
            if (kind != null && kind != "sort" && kind != "table")
                throw new ValueError($"Invalid kind: '{kind}'. Please use None, 'sort' or 'table'.");
            // StringDType (tested first: IsString refuses it): its descriptor has objects, so NumPy always takes the loop
            // path, folded here into set lookups with NumPy's NA answers (IsinVString).
            if (Backends.Strings.NDStringUfuncs.IsVString(element) || Backends.Strings.NDStringUfuncs.IsVString(test_elements))
                return IsinVString(element, test_elements, invert, kind);
            // Text: NumPy's _isin statements over the text kernels (IsinText): its loop path for a small set, its sort path
            // otherwise — each with NumPy's own answer, also where the two paths disagree on a text / number pair.
            if (Backends.Strings.NDStringUfuncs.IsString(element) || Backends.Strings.NDStringUfuncs.IsString(test_elements))
                return IsinText(element, test_elements, assume_unique, invert, kind);

            // Result is shaped like element in C-order. Build the shape from the DIMENSIONS (a fresh
            // C-contiguous shape) — reusing element.Shape would carry a view's strides and scramble the
            // contiguous membership buffer on reshape (transposed/strided/negative-stride element).
            Shape outShape = Shape.Adopt(element.Shape.dimensions);
            NDArray ar1 = NDArray.ravel(element, 'C');        // C-order flatten (NumPy: np.asarray(element).ravel())
            NDArray ar2 = NDArray.ravel(test_elements, 'C');  // test values are always flattened

            // kind='table' is only defined for boolean/integer arrays (NumPy's dtype.kind in ("u", "i", "b"): a char array,
            // NumPy's uint16, is one), and can overflow for wide signed ranges.
            if (kind == "table")
            {
                if (!(IsNumPyIntegerKind(ar1.typecode) && IsNumPyIntegerKind(ar2.typecode)))
                    throw new ValueError("The 'table' method is only supported for boolean or integer arrays. "
                        + "Please select 'sort' or None for kind.");
                if (ar2.size > 0 && !TableRangeIsSafe(ar2))
                    throw new InvalidOperationException(
                        "You have specified kind='table', but the range of values in `ar2` or `ar1` exceed the "
                        + "maximum integer of the datatype. Please set `kind` to None or 'sort'.");
                // NumPy then allocates its table, np.zeros(ar2_range + 1, dtype=bool) (np.ones under invert), whatever the
                // element holds (an empty element too): a length past npy_intp is that call's shape conversion's ValueError.
                // A length NumPy can address but not allocate (8 EiB) is its MemoryError, which NumSharp does not reproduce:
                // the membership below needs no table (the documented divergence).
                if (ar2.size > 0 && TableRange(ar2) >= long.MaxValue)
                    throw new ValueError("Maximum allowed dimension exceeded");
            }

            // Empty element -> empty boolean result shaped like element (value irrelevant). Every NumPy path ends in
            // `_isin(...).reshape(element.shape)`, so even this answer is a VIEW of a fresh 1-D array (flags.owndata
            // False); allocating outShape directly returned an owner.
            if (ar1.size == 0)
                return new NDArray(NPTypeCode.Boolean, new Shape(ar1.size), true).reshape(outShape);

            // Empty test set -> nothing is a member (all invert); a view of the 1-D answer, as above.
            if (ar2.size == 0)
                return NDArray.full(new Shape(ar1.size), invert, NPTypeCode.Boolean).reshape(outShape);

            // NumPy's _isin answers on one of three paths (the table method, the loop path, the sort path; NumPyPath), and
            // each gives the exact membership, except NumPy's SORT path in two cases: its np.concatenate casts both unique
            // arrays to their common dtype, where two 64-bit integers past 2**53 can become one float64, and under
            // assume_unique it skips the np.unique calls, so repeated values meet as neighbours. Its adjacent-equality step
            // then flags the first of two equal neighbours as a member. Where NumPy takes the sort path in either case its
            // statements run (SortPathFlat); the answer is a view of the 1-D flags, as NumPy's reshape returns it.
            IsinPath path = NumPyPath(ar1, ar2, kind, out long tableMin, out long tableRange);
            if (path == IsinPath.Sort && (assume_unique || !ConcatenationIsLossless(ar1, ar2)))
            {
                using NDArray flat = SortPathFlat(ar1, ar2, assume_unique, invert);
                return flat.reshape(outShape);
            }

            // The table method and the loop path run as one pass each (NumPy's statements fused, the same exact
            // membership): NumPy's table masks, compaction and fancy indexing, or its one comparison pass per set value,
            // cost more than the pass. Each declines (null) the pairs it does not take: they and NumPy's lossless sort path
            // keep ComputeMembership, whose hash set outruns NumPy's sort.
            using NDArray fused = path switch
            {
                IsinPath.Table => TableMembership(ar1, ar2, tableMin, tableRange, invert),
                IsinPath.Loop => LoopMembership(ar1, ar2, invert),
                _ => null,
            };
            if (fused is not null)
                return fused.reshape(outShape);

            // Every other case: NumPy's answer is the exact membership in result_type(ar1, ar2) (the loop path's
            // comparisons, exact for an int64 / uint64 pair too), which ComputeMembership computes without NumPy's
            // per-value passes or sort.
            NDArray member = ComputeMembership(ar1, ar2);
            NDArray result = invert ? LogicalNot(member, null, null, null) : member;
            return result.reshape(outShape);
        }

        /// <summary>
        ///     NumPy's <c>_isin</c> when an operand holds strings (the <c>kind</c> name already validated), statement by
        ///     statement: the loop path for a small set, the sort path otherwise, each over NumSharp's text kernels.
        /// </summary>
        /// <param name="element">The tested values (the result has its shape).</param>
        /// <param name="test_elements">The set (flattened).</param>
        /// <param name="assume_unique">NumPy's speed hint: the sort path skips its two <c>np.unique</c> calls and takes the
        ///     first <c>len(ar1)</c> flags (the answer is then NumPy's, also for inputs that are not unique).</param>
        /// <param name="invert">Whether the answer is <c>not in</c>.</param>
        /// <param name="kind">The validated <c>kind</c>: <c>"table"</c> refuses every string array; <c>"sort"</c> and null
        ///     take the same paths (the table method needs integer arrays).</param>
        /// <returns>A bool array of <paramref name="element"/>'s shape: a view of a fresh 1-D answer, as NumPy's reshape
        ///     returns it.</returns>
        /// <exception cref="ValueError"><c>kind='table'</c> (the method needs boolean or integer arrays).</exception>
        /// <exception cref="UnicodeDecodeError">The sort path concatenates <c>S</c> text with <c>U</c> text, decoding the bytes
        ///     as ASCII: a byte that is not ASCII fails, in the order of the unique values (the concatenation's order).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's statements, with <c>ar1</c> / <c>ar2</c> the two flattened arrays (no <c>StringDType</c>, no object
        ///     array, so <c>contains_object</c> is false):
        ///     <list type="bullet">
        ///       <item><description><c>len(ar2) &lt; 10 * len(ar1) ** 0.145</c> takes the LOOP path: <c>mask |= (ar1 == a)</c>
        ///       for each <c>a</c> of <c>ar2</c> (<c>&amp;= (ar1 != a)</c> under invert). Two strings of one class compare
        ///       their text; a pair <c>equal</c> has no loop for (text against a number, bytes against str) answers
        ///       all-False / all-True, so it leaves the mask as it was (<c>np.isin(['1', '2'], [1, 2])</c> is
        ///       <c>[False, False]</c>, probed).</description></item>
        ///       <item><description>otherwise the SORT path: <c>np.unique</c> of each array (with <c>ar1</c>'s inverse),
        ///       <c>np.concatenate</c> — which formats numbers into text and decodes bytes beside str, so the same pair
        ///       compares EQUAL there (<c>np.isin(['1'], range(20))</c> is <c>[True]</c>) —, a stable argsort of the joined
        ///       values, the flags of adjacent equal values scattered back in the order, and the inverse's
        ///       gather.</description></item>
        ///     </list>
        ///     An empty <c>ar1</c> takes the sort path (<c>len(ar2) &lt; 10 * 0 ** 0.145</c> is false): its concatenation
        ///     still decodes <c>ar2</c>, and the answer is empty.
        ///     </para>
        /// </remarks>
        private static NDArray IsinText(NDArray element, NDArray test_elements, bool assume_unique, bool invert, string kind)
        {
            // The table method exists only for boolean / integer arrays; NumPy checks before it reads a value.
            if (kind == "table")
                throw new ValueError("The 'table' method is only supported for boolean or integer arrays. "
                    + "Please select 'sort' or None for kind.");
            var dims = (long[])element.Shape.dimensions.Clone();
            Shape shape = dims.Length == 0 ? Shape.NewScalar() : Shape.Adopt(dims);
            // NumPy reshapes _isin's 1-D answer to element's shape: the result is a view of it (flags.owndata False). The
            // view holds its own reference to the flat buffer.
            using NDArray flat = IsinTextFlat(element, test_elements, assume_unique, invert);
            return flat.reshape(shape);
        }

        /// <summary>
        ///     NumPy's <c>_isin</c> over its two flattened operands when one holds strings (see <see cref="IsinText"/>): the
        ///     1-D answer, one flag per element of <paramref name="element"/>.
        /// </summary>
        /// <param name="element">The tested values (flattened here, C order).</param>
        /// <param name="test_elements">The set (flattened here).</param>
        /// <param name="assume_unique">Whether NumPy's sort path skips its <c>np.unique</c> calls.</param>
        /// <param name="invert">Whether the answer is <c>not in</c>.</param>
        /// <returns>A fresh 1-D bool array.</returns>
        /// <exception cref="UnicodeDecodeError">The sort path decodes <c>S</c> text holding a byte that is not ASCII.</exception>
        /// <remarks>Scoped: the ravels, the unique arrays, the joined values, the order and the flags are reclaimed; only
        ///     the answer survives.</remarks>
        [NDScoped]
        private static NDArray IsinTextFlat(NDArray element, NDArray test_elements, bool assume_unique, bool invert)
        {
            NDArray ar1 = NDArray.ravel(element, 'C');
            NDArray ar2 = NDArray.ravel(test_elements, 'C');
            long n1 = ar1.size, n2 = ar2.size;

            // Python evaluates 10 * len(ar1) ** 0.145 in double precision, as Math.Pow does.
            if (n2 < 10.0 * Math.Pow(n1, 0.145))
            {
                // The loop path: one comparison of ar1 with each value of ar2, OR-ed (AND-ed under invert) into the mask.
                // A pair the comparison has no loop for answers all-False (all-True for !=), which leaves the mask as it
                // is, so only a pair that compares text runs the kernel.
                NDArray mask = NDArray.full(Shape.Vector(n1), invert, NPTypeCode.Boolean);
                if (n1 == 0 || !Backends.Strings.NDStringUfuncs.ComparesText(ar1.dtype, ar2.dtype))
                    return mask;
                for (long i = 0; i < n2; i++)
                {
                    using NDArray a = ar2[i.ToString()];
                    using NDArray hit = invert ? ar1.TensorEngine.NotEqual(ar1, a, null, null, null) : ar1.TensorEngine.Compare(ar1, a, null, null, null);
                    if (invert)
                        mask.TensorEngine.LogicalAnd(mask, hit, null, mask, null);
                    else
                        mask.TensorEngine.LogicalOr(mask, hit, null, mask, null);
                }
                return mask;
            }

            // The sort path, shared with the numeric route (SortPathFlat).
            return SortPathFlat(ar1, ar2, assume_unique, invert);
        }

        /// <summary>
        ///     NumPy's sort path of <c>_isin</c> over two flattened operands: <c>np.unique</c> of each (unless
        ///     <paramref name="assume_unique"/>), one <c>np.concatenate</c> in their common dtype, a stable argsort, the
        ///     adjacent-equality flags scattered back in the sorted order, and the inverse's gather (or the first
        ///     <c>len(ar1)</c> flags).
        /// </summary>
        /// <param name="ar1">The flattened tested values.</param>
        /// <param name="ar2">The flattened set.</param>
        /// <param name="assume_unique">Whether the <c>np.unique</c> calls are skipped: the answer is then NumPy's for inputs
        ///     that are not unique too (repeated values meet as neighbours).</param>
        /// <param name="invert">Whether the answer is <c>not in</c>.</param>
        /// <returns>The 1-D answer, one flag per element of <paramref name="ar1"/>: a fresh gather, or under
        ///     <paramref name="assume_unique"/> a view of the flags (NumPy's slice).</returns>
        /// <exception cref="UnicodeDecodeError">The concatenation decodes <c>S</c> text holding a byte that is not ASCII
        ///     beside <c>U</c> text.</exception>
        /// <remarks>
        ///     NumPy's answer, which is the exact membership only when the concatenation keeps every value apart: its cast to
        ///     the common dtype can merge two values (two 64-bit integers past 2**53 in float64, a number formatted into the
        ///     text it equals), and "my successor in the sorted order equals me" then flags the first of them. Scoped: the
        ///     unique arrays, the joined values, the order and the flags are reclaimed; only the answer survives.
        /// </remarks>
        [NDScoped]
        private static NDArray SortPathFlat(NDArray ar1, NDArray ar2, bool assume_unique, bool invert)
        {
            // Without assume_unique each array is deduplicated (ar1 with its inverse, to map the flags back).
            NDArray revIdx = null;
            if (!assume_unique)
            {
                var u1 = ar1.unique(return_index: false, return_inverse: true);
                ar1 = u1[0];
                revIdx = u1[1];
                ar2 = ar2.unique();
            }
            // ar = concatenate((ar1, ar2)) casts both to their common dtype: it formats numbers into text and decodes bytes
            // beside str, and makes float64 of a 64-bit integer beside a float or (uint64 beside a signed integer) beside
            // the other sign (NumPy's promotion).
            NDArray ar = NDArray.concatenate(new[] { ar1, ar2 }, 0, null, null, "same_kind");   // np.concatenate((ar1, ar2))
            // "We need this to be a stable sort, so always use 'mergesort'": ar1's values come before ar2's equal ones.
            NDArray order = ar.TensorEngine.ArgSort(ar, -1, "mergesort", null, null);
            NDArray sar = ar[order];
            NDArray boolAr = invert ? sar["1:"] != sar[":-1"] : sar["1:"] == sar[":-1"];
            // np.array([invert]) is np.array's statement for a C# array of one bool: a one-element array over a copy.
            NDArray flag = NDArray.concatenate(
                new[] { boolAr, new NDArray(Unmanaged.ArraySlice.FromArray(new[] { invert }, true), Shape.Vector(1)) },
                0, null, null, "same_kind");
            var ret = new NDArray(NPTypeCode.Boolean, Shape.Vector(ar.size), false);
            ret[order] = flag;
            // assume_unique keeps the first len(ar1) flags (NumPy's slice: a view); otherwise the inverse maps each original
            // element to its flag (a fresh gather).
            return assume_unique ? ret[$":{ar1.size}"] : ret[revIdx];
        }

        /// <summary>The path NumPy's <c>_isin</c> answers two numeric operands on (see <see cref="NumPyPath"/>).</summary>
        private enum IsinPath
        {
            /// <summary>The table method: a bool table over the set's range, read at each element (the exact
            ///     membership).</summary>
            Table,

            /// <summary>The loop path: <c>mask |= ar1 == a</c> for each set value <c>a</c> (the exact membership in the
            ///     comparisons' dtypes; exact for an int64 / uint64 pair).</summary>
            Loop,

            /// <summary>The sort path: <c>np.unique</c> of each operand, one concatenate in their common dtype, a stable
            ///     argsort and the adjacent-equality flags (<see cref="SortPathFlat"/>).</summary>
            Sort,
        }

        /// <summary>
        ///     The path NumPy's <c>_isin</c> answers two flattened numeric operands on (its <c>kind</c> errors already raised):
        ///     the table method for an integer or bool pair with <c>kind</c> None or <c>'table'</c> whose set range is safe for
        ///     the set's dtype and at most <c>6 * (len(ar1) + len(ar2))</c> (or <c>kind='table'</c>), else the loop path while
        ///     <c>len(ar2) &lt; 10 * len(ar1) ** 0.145</c>, else the sort path.
        /// </summary>
        /// <param name="ar1">The flattened tested values, not empty.</param>
        /// <param name="ar2">The flattened set, not empty.</param>
        /// <param name="kind">The validated <c>kind</c>: null, <c>"sort"</c> (no table method) or <c>"table"</c>.</param>
        /// <param name="tableMin">For the table method within NumPy's memory constraint, the set's minimum (NumPy's
        ///     <c>ar2_min</c>; a uint64 minimum past <c>long.MaxValue</c> as its 64 bits); otherwise 0.</param>
        /// <param name="tableRange">For the table method within NumPy's memory constraint, the set's range
        ///     (<c>ar2_max - ar2_min</c>); otherwise -1 (a <c>kind='table'</c> table past the constraint, which NumPy
        ///     allocates and <see cref="TableMembership"/> declines: the documented divergence).</param>
        /// <returns>NumPy's path.</returns>
        /// <remarks>Reads the set's bounds (two reductions) only for an integer or bool pair.</remarks>
        private static IsinPath NumPyPath(NDArray ar1, NDArray ar2, string kind, out long tableMin, out long tableRange)
        {
            tableMin = 0;
            tableRange = -1;
            long n1 = ar1.size, n2 = ar2.size;
            if (kind != "sort" && IsNumPyIntegerKind(ar1.typecode) && IsNumPyIntegerKind(ar2.typecode))
            {
                // The table method's two constraints, on the range as a Python int (kind='table' with an unsafe range has
                // raised already, so it reaches the table here).
                var (min, max) = TableBounds(ar2);
                BigInteger range = max - min;
                bool belowMemory = range <= 6 * ((BigInteger)n1 + n2);
                if (RangeFitsDtype(range, ar2.typecode) && (belowMemory || kind == "table"))
                {
                    if (belowMemory)
                    {
                        // The table kernel's offsets are 64-bit arithmetic modulo 2**64, so a uint64 set's minimum past
                        // long.MaxValue rides as its bits (TableMembership).
                        tableMin = min <= long.MaxValue ? (long)min : unchecked((long)(ulong)min);
                        tableRange = (long)range;
                    }
                    return IsinPath.Table;
                }
            }
            // Python evaluates 10 * len(ar1) ** 0.145 in double precision, as Math.Pow does.
            return n2 < 10.0 * Math.Pow(n1, 0.145) ? IsinPath.Loop : IsinPath.Sort;
        }

        /// <summary>
        ///     NumPy's table method in one pass: the set scattered into a bool table over <c>[min, min + range]</c> (NumPy's
        ///     <c>isin_helper_ar</c>: zeros with the set marked True, or under <paramref name="invert"/> ones with the set
        ///     marked False), then each element's flag is its table entry when it lies in that range and
        ///     <paramref name="invert"/> otherwise (NumPy's outgoing array).
        /// </summary>
        /// <param name="ar1">The flattened tested values (bool, an integer dtype or char).</param>
        /// <param name="ar2">The flattened set (bool, an integer dtype or char).</param>
        /// <param name="min">The set's minimum (<see cref="NumPyPath"/>'s <c>tableMin</c>: a uint64 minimum past
        ///     <c>long.MaxValue</c> as its 64 bits).</param>
        /// <param name="range">The set's range, or -1 for no table (<see cref="NumPyPath"/>'s <c>tableRange</c>).</param>
        /// <param name="invert">Whether the answer is <c>not in</c>.</param>
        /// <returns>A fresh 1-D bool array, one flag per element; null to decline (no table, or a uint64 operand beside a
        ///     signed one, whose 64 bits read as one number in one operand and another in the other: the membership routes
        ///     answer that pair exactly).</returns>
        /// <remarks>
        ///     Exact like NumPy's statements: an element is read as its 64 bits and tested by
        ///     <c>(ulong)(v - min) &lt;= range</c>, 64-bit arithmetic modulo 2**64. Without a uint64 / signed pair every value
        ///     lies in one span of 2**64 numbers (<c>[-2**63, 2**63)</c> when an operand is signed, <c>[0, 2**64)</c>
        ///     otherwise; bool reads as 0 / 1, char as its code), and <c>min + range</c> is the set's maximum, so
        ///     <c>v - min</c> lands in <c>[0, range]</c> exactly when <c>v</c> lies in <c>[min, min + range]</c>. Reads each
        ///     operand once, where NumPy's statements make two comparisons, an AND, a compaction, a subtraction, a fancy
        ///     gather and a masked store. Elements of one byte, and of two bytes from <see cref="DomainTableMinElements"/>
        ///     on, read a table over their dtype's whole domain instead (<see cref="DomainTableMembership"/>): the same
        ///     flags, one load per element. The tables are released before returning; a contiguous copy of an operand is
        ///     reclaimed by the caller's scope.
        /// </remarks>
        private static unsafe NDArray TableMembership(NDArray ar1, NDArray ar2, long min, long range, bool invert)
        {
            if (range < 0 || UInt64BesideSigned(ar1.typecode, ar2.typecode))
                return null;
            ar1 = ContiguousRun(ar1);
            ar2 = ContiguousRun(ar2);
            void* s = (byte*)ar2.Address + ar2.Shape.offset * ar2.dtypesize;
            long m = ar2.size;
            bool mark = !invert;
            if (ar1.dtypesize == 1 || (ar1.dtypesize == 2 && ar1.size >= DomainTableMinElements))
                return DomainTableMembership(ar1, ar2.typecode, s, m, mark, invert);

            // NumPy's table: np.zeros with the set marked 1, or under invert np.ones with the set marked 0, so the probe reads
            // the flag itself. The table is scratch, released here: the using also keeps it reachable while its raw pointer
            // is read.
            using NDArray table = NDArray.full(new Shape(range + 1), invert, NPTypeCode.Boolean);
            bool* tb = (bool*)table.Address + table.Shape.offset;
            switch (ar2.typecode)
            {
                case NPTypeCode.Boolean: ScatterSet((byte*)s, m, min, mark, tb); break;
                case NPTypeCode.Byte: ScatterSet((byte*)s, m, min, mark, tb); break;
                case NPTypeCode.SByte: ScatterSet((sbyte*)s, m, min, mark, tb); break;
                case NPTypeCode.Int16: ScatterSet((short*)s, m, min, mark, tb); break;
                case NPTypeCode.UInt16: ScatterSet((ushort*)s, m, min, mark, tb); break;
                case NPTypeCode.Char: ScatterSet((ushort*)s, m, min, mark, tb); break;
                case NPTypeCode.Int32: ScatterSet((int*)s, m, min, mark, tb); break;
                case NPTypeCode.UInt32: ScatterSet((uint*)s, m, min, mark, tb); break;
                case NPTypeCode.Int64: ScatterSet((long*)s, m, min, mark, tb); break;
                case NPTypeCode.UInt64: ScatterSet((ulong*)s, m, min, mark, tb); break;
                default: return null;
            }

            long n = ar1.size;
            var result = new NDArray(NPTypeCode.Boolean, Shape.Vector(n), false);
            bool* r = (bool*)result.Address;
            void* e = (byte*)ar1.Address + ar1.Shape.offset * ar1.dtypesize;
            ulong urange = (ulong)range;
            switch (ar1.typecode)
            {
                // One-byte elements (bool, int8, uint8) took the domain table above; char reads as its uint16 value.
                case NPTypeCode.Int16: TableProbe((short*)e, n, tb, min, urange, invert, r); break;
                case NPTypeCode.UInt16: TableProbe((ushort*)e, n, tb, min, urange, invert, r); break;
                case NPTypeCode.Char: TableProbe((ushort*)e, n, tb, min, urange, invert, r); break;
                case NPTypeCode.Int32: TableProbe((int*)e, n, tb, min, urange, invert, r); break;
                case NPTypeCode.UInt32: TableProbe((uint*)e, n, tb, min, urange, invert, r); break;
                case NPTypeCode.Int64: TableProbe((long*)e, n, tb, min, urange, invert, r); break;
                case NPTypeCode.UInt64: TableProbe((ulong*)e, n, tb, min, urange, invert, r); break;
                default:
                    result.Dispose();
                    return null;
            }
            return result;
        }

        /// <summary>
        ///     The elements of two bytes from which <see cref="DomainTableMembership"/> replaces the range table: filling its
        ///     65,536 flags costs about 1.3 us, which the saved range test (about 0.23 ns per element) repays from about
        ///     5,900 elements (measured over uint16 at 1K / 10K / 100K elements: 1.72 / 3.86 / 27.4 us against the range
        ///     probe's 0.49 / 4.87 / 49.0). One-byte elements fill 256 flags, which always pays.
        /// </summary>
        private const long DomainTableMinElements = 8192;

        /// <summary>
        ///     NumPy's table method over elements of one or two bytes: a table over the element dtype's whole domain (256 or
        ///     65,536 flags, indexed by the element's bits) holds the outgoing flag everywhere and the members' flag at every
        ///     set value of the domain, so each element's flag is one load, without the range test.
        /// </summary>
        /// <param name="ar1">The flattened tested values, contiguous: bool, int8, uint8, int16, uint16 or char.</param>
        /// <param name="setType">The set's dtype (bool, an integer dtype or char; a uint64 set beside a signed element was
        ///     declined by the caller).</param>
        /// <param name="s">The set's first element (contiguous).</param>
        /// <param name="m">The set's length.</param>
        /// <param name="mark">The flag of a member: True, or False under invert.</param>
        /// <param name="invert">Whether the answer is <c>not in</c>: the flag of every value the set does not hold.</param>
        /// <returns>A fresh 1-D bool array, one flag per element; null to decline (a set dtype the switch does not
        ///     read).</returns>
        /// <remarks>
        ///     The answer is NumPy's table answer: an element value the set holds is a member, every other value is not (a
        ///     set value outside the element dtype's domain names no element and is skipped). A negative int8 / int16 is
        ///     indexed by its two's complement bits, the bits the probe reads. Measured over 100K elements: uint8 27.4 us
        ///     and uint16 27.4 us, against the range probe's 48.8 and 49.0. The table is released before returning.
        /// </remarks>
        private static unsafe NDArray DomainTableMembership(NDArray ar1, NPTypeCode setType, void* s, long m, bool mark,
            bool invert)
        {
            int width = ar1.dtypesize;
            int size = width == 1 ? 256 : 65536;
            // The element dtype's domain [lo, hi]: its values are exactly the 256 or 65,536 bit patterns.
            long lo = ar1.typecode switch
            {
                NPTypeCode.SByte => sbyte.MinValue,
                NPTypeCode.Int16 => short.MinValue,
                _ => 0,
            };
            long hi = lo + size - 1;
            // Scratch, released here: the using also keeps it reachable while its raw pointer is read.
            using NDArray domain = NDArray.full(new Shape(size), invert, NPTypeCode.Boolean);
            bool* table = (bool*)domain.Address + domain.Shape.offset;
            switch (setType)
            {
                case NPTypeCode.Boolean: ScatterDomain((byte*)s, m, lo, hi, mark, table); break;
                case NPTypeCode.Byte: ScatterDomain((byte*)s, m, lo, hi, mark, table); break;
                case NPTypeCode.SByte: ScatterDomain((sbyte*)s, m, lo, hi, mark, table); break;
                case NPTypeCode.Int16: ScatterDomain((short*)s, m, lo, hi, mark, table); break;
                case NPTypeCode.UInt16: ScatterDomain((ushort*)s, m, lo, hi, mark, table); break;
                case NPTypeCode.Char: ScatterDomain((ushort*)s, m, lo, hi, mark, table); break;
                case NPTypeCode.Int32: ScatterDomain((int*)s, m, lo, hi, mark, table); break;
                case NPTypeCode.UInt32: ScatterDomain((uint*)s, m, lo, hi, mark, table); break;
                case NPTypeCode.Int64: ScatterDomain((long*)s, m, lo, hi, mark, table); break;
                case NPTypeCode.UInt64: ScatterDomain((ulong*)s, m, lo, hi, mark, table); break;
                default: return null;
            }
            long n = ar1.size;
            var result = new NDArray(NPTypeCode.Boolean, Shape.Vector(n), false);
            void* e = (byte*)ar1.Address + ar1.Shape.offset * width;
            if (width == 1)
                DomainProbe((byte*)e, n, table, (bool*)result.Address);
            else
                DomainProbe((ushort*)e, n, table, (bool*)result.Address);
            return result;
        }

        /// <summary>Marks every set value of the element domain <c>[lo, hi]</c> in the domain table, at the value's bits
        ///     (<c>v &amp; (hi - lo)</c>: a negative value's two's complement).</summary>
        /// <typeparam name="T">The set's element type (an integer type; bool read as byte, char as ushort).</typeparam>
        /// <param name="s">The set's first element (contiguous).</param>
        /// <param name="m">The set's length.</param>
        /// <param name="lo">The element dtype's smallest value.</param>
        /// <param name="hi">The element dtype's largest value (<c>lo + 255</c> or <c>lo + 65535</c>).</param>
        /// <param name="mark">The flag of a member: True, or False under invert.</param>
        /// <param name="table">The domain table, <c>hi - lo + 1</c> entries.</param>
        /// <remarks>A uint64 set value past <c>long.MaxValue</c> reads as negative bits, below every unsigned element
        ///     domain: skipped, as it names no element.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ScatterDomain<T>(T* s, long m, long lo, long hi, bool mark, bool* table)
            where T : unmanaged
        {
            long bits = hi - lo;
            for (long i = 0; i < m; i++)
            {
                long v = AsLong(s + i);
                if (v >= lo && v <= hi)
                    table[v & bits] = mark;
            }
        }

        /// <summary>Each element's flag from the domain table: one load at the element's bits.</summary>
        /// <typeparam name="TBits">byte (one-byte elements) or ushort (two-byte elements): the elements read as their
        ///     bits.</typeparam>
        /// <param name="e">The first element (contiguous).</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="table">The domain table (256 or 65,536 flags).</param>
        /// <param name="r">The 1-D bool output, one flag per element.</param>
        /// <remarks>Four elements per step: their loads are independent of each other, and the loop's compare and branch
        ///     run once per four. Optimized at its first call (see <see cref="TableProbe"/>).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DomainProbe<TBits>(TBits* e, long n, bool* table, bool* r) where TBits : unmanaged
        {
            long i = 0;
            for (; i + 4 <= n; i += 4)
            {
                r[i] = table[ElementBits(e + i)];
                r[i + 1] = table[ElementBits(e + i + 1)];
                r[i + 2] = table[ElementBits(e + i + 2)];
                r[i + 3] = table[ElementBits(e + i + 3)];
            }
            for (; i < n; i++)
                r[i] = table[ElementBits(e + i)];
        }

        /// <summary>An element of one or two bytes as its bits, the domain table's index.</summary>
        /// <typeparam name="TBits">byte or ushort.</typeparam>
        /// <param name="p">The element's address.</param>
        /// <returns>Its bits, zero-extended.</returns>
        [MethodImpl(OptimizeAndInline)]
        private static unsafe nuint ElementBits<TBits>(TBits* p) where TBits : unmanaged
        {
            if (typeof(TBits) == typeof(byte)) return *(byte*)p;
            return *(ushort*)p;
        }

        /// <summary>Marks every set value in the table: <c>isin_helper_ar[ar2 - ar2_min] = 1</c> (<c>= 0</c> under
        ///     invert).</summary>
        /// <typeparam name="T">The set's element type (an integer type; bool read as byte, char as ushort).</typeparam>
        /// <param name="s">The set's first element (contiguous).</param>
        /// <param name="m">The set's length.</param>
        /// <param name="min">The set's minimum, so every offset lies in the table.</param>
        /// <param name="mark">The flag of a member: True, or False under invert.</param>
        /// <param name="table">The table, <c>max - min + 1</c> entries.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ScatterSet<T>(T* s, long m, long min, bool mark, bool* table) where T : unmanaged
        {
            for (long i = 0; i < m; i++)
                table[AsLong(s + i) - min] = mark;
        }

        /// <summary>Each element's flag: its table entry when <c>v - min</c> lies in <c>[0, range]</c>, else
        ///     <paramref name="invert"/> (NumPy's outgoing array: np.zeros, or np.ones under invert).</summary>
        /// <typeparam name="T">The elements' type: an integer type of two bytes or more (char read as ushort).</typeparam>
        /// <param name="e">The first element (contiguous).</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="table">The set's table (inverted under <paramref name="invert"/>).</param>
        /// <param name="min">The set's minimum.</param>
        /// <param name="range">The set's range: the table has <c>range + 1</c> entries.</param>
        /// <param name="invert">Whether the flags are <c>not in</c>: the flag of an element outside the range.</param>
        /// <param name="r">The 1-D bool output, one flag per element.</param>
        /// <remarks>Optimized at its first call and not inlined (as the other kernels): it runs once per np.isin call,
        ///     which a short program never tiers up. The inverted table keeps an XOR out of the loop (it cost 14% at 100K
        ///     elements, measured), and four elements per step 10% more. An AVX2 gather of the table bytes measured 1.7x
        ///     faster over 100K int32 elements and is not taken: Intel's GDS microcode mitigation slows gathers on the 6th
        ///     to 11th generation cores, so it could lose to this loop there.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void TableProbe<T>(T* e, long n, bool* table, long min, ulong range, bool invert, bool* r)
            where T : unmanaged
        {
            // A value below min wraps to an offset past the range (the range is small), so one compare decides both sides;
            // the select reads the table only inside it.
            long i = 0;
            for (; i + 4 <= n; i += 4)
            {
                ulong o0 = unchecked((ulong)(AsLong(e + i) - min));
                ulong o1 = unchecked((ulong)(AsLong(e + i + 1) - min));
                ulong o2 = unchecked((ulong)(AsLong(e + i + 2) - min));
                ulong o3 = unchecked((ulong)(AsLong(e + i + 3) - min));
                r[i] = o0 <= range ? table[o0] : invert;
                r[i + 1] = o1 <= range ? table[o1] : invert;
                r[i + 2] = o2 <= range ? table[o2] : invert;
                r[i + 3] = o3 <= range ? table[o3] : invert;
            }
            for (; i < n; i++)
            {
                ulong offset = unchecked((ulong)(AsLong(e + i) - min));
                r[i] = offset <= range ? table[offset] : invert;
            }
        }

        /// <summary>An integer element as 64 bits: its value for every integer type but uint64, whose values past
        ///     <c>long.MaxValue</c> come back as their bits (the table's arithmetic is modulo 2**64, see
        ///     <see cref="TableMembership"/>), read through its pointer.</summary>
        /// <typeparam name="T">byte, sbyte, short, ushort, int, uint, long or ulong.</typeparam>
        /// <param name="p">The element's address.</param>
        /// <returns>Its value (or bits).</returns>
        /// <remarks>A pointer, not the value: <c>Unsafe.As</c> of a value's <c>ref</c> makes the JIT keep it in memory.</remarks>
        [MethodImpl(OptimizeAndInline)]
        private static unsafe long AsLong<T>(T* p) where T : unmanaged
        {
            if (typeof(T) == typeof(byte)) return *(byte*)p;
            if (typeof(T) == typeof(sbyte)) return *(sbyte*)p;
            if (typeof(T) == typeof(short)) return *(short*)p;
            if (typeof(T) == typeof(ushort)) return *(ushort*)p;
            if (typeof(T) == typeof(int)) return *(int*)p;
            if (typeof(T) == typeof(uint)) return *(uint*)p;
            if (typeof(T) == typeof(ulong)) return unchecked((long)*(ulong*)p);
            return *(long*)p;
        }

        /// <summary>Whether a uint64 operand meets a signed one (int8, int16, int32 or int64): their 64 bits are not one
        ///     number line, so the table kernel declines the pair.</summary>
        /// <param name="a">One operand's dtype.</param>
        /// <param name="b">The other's.</param>
        /// <returns>True for a uint64 / signed pair, either way round.</returns>
        private static bool UInt64BesideSigned(NPTypeCode a, NPTypeCode b)
        {
            static bool Signed(NPTypeCode t) => t is NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64;
            return (a == NPTypeCode.UInt64 && Signed(b)) || (b == NPTypeCode.UInt64 && Signed(a));
        }

        /// <summary>
        ///     The most set values per vector lane the one-pass loop kernel compares an element with (a float64 set of 36
        ///     values, a float32 set of 72, a uint8 set of 288): past it the membership routes' per-element cost (2 to 3 ns
        ///     over 100K elements) is below the kernel's, which grows by about a quarter of a nanosecond per vector
        ///     comparison, <c>m / lanes</c> of them per element. Measured over 6 dtypes and 10 set sizes, the crossover lies
        ///     between 8.3 values per lane (int32) and 11 (float32), so 9 keeps every set size within 10% of the faster
        ///     route.
        /// </summary>
        private const int LoopKernelMaxValuesPerLane = 9;

        /// <summary>
        ///     NumPy's loop path in one pass: each element compared with every set value in vector registers, OR-ed (the flag
        ///     XOR <paramref name="invert"/>, NumPy's AND of <c>!=</c>), both operands in <c>result_type(ar1, ar2)</c> as
        ///     NumPy's comparisons promote them.
        /// </summary>
        /// <param name="ar1">The flattened tested values.</param>
        /// <param name="ar2">The flattened set (NumPy's loop path: few values).</param>
        /// <param name="invert">Whether the answer is <c>not in</c>.</param>
        /// <returns>A fresh 1-D bool array, one flag per element; null to decline: a host without 256-bit vectors, more than
        ///     <see cref="LoopKernelMaxValuesPerLane"/> values per vector lane of the common dtype, a common dtype without
        ///     vector equality (float16, complex128, decimal), or an int64 / uint64 pair, which NumPy's comparison loops
        ///     answer exactly where the common dtype is NEP 50's float64 (the membership routes' exact uint64 route answers
        ///     it).</returns>
        /// <remarks>Equality is IEEE's: NaN is never a member, -0.0 equals 0.0 (NumPy's == and the hash set agree). The casts
        ///     are NDArrays, reclaimed by the caller's scope.</remarks>
        private static unsafe NDArray LoopMembership(NDArray ar1, NDArray ar2, bool invert)
        {
            // Without 256-bit vectors (arm64, an x86 host without AVX2) the kernel would compare one element with one value
            // at a time, m times per element: the membership routes' constant cost per element is lower.
            if (!Vector256.IsHardwareAccelerated)
                return null;
            NPTypeCode t = DTypePromotion.PromoteTypes(ar1.typecode, ar2.typecode);
            if (t == NPTypeCode.Double && IsIntegerOrBool(ar1.typecode) && IsIntegerOrBool(ar2.typecode))
                return null;
            if (t is NPTypeCode.Half or NPTypeCode.Complex or NPTypeCode.Decimal)
                return null;
            // The kernel's cost per element grows with m / lanes (one vector comparison per value per vector of elements).
            if (ar2.size > LoopKernelMaxValuesPerLane * (32 / t.SizeOf()))
                return null;
            NDArray e = ContiguousRun(ar1.typecode == t ? ar1 : ar1.astype(t));
            NDArray s = ContiguousRun(ar2.typecode == t ? ar2 : ar2.astype(t));
            long n = e.size;
            int m = (int)s.size;
            var result = new NDArray(NPTypeCode.Boolean, Shape.Vector(n), false);
            bool* r = (bool*)result.Address;
            void* ep = (byte*)e.Address + e.Shape.offset * e.dtypesize;
            void* sp = (byte*)s.Address + s.Shape.offset * s.dtypesize;
            switch (t)
            {
                // bool compares as its bytes 0 / 1, char as its uint16 value.
                case NPTypeCode.Boolean: LoopProbe((byte*)ep, n, (byte*)sp, m, invert, r); break;
                case NPTypeCode.Byte: LoopProbe((byte*)ep, n, (byte*)sp, m, invert, r); break;
                case NPTypeCode.SByte: LoopProbe((sbyte*)ep, n, (sbyte*)sp, m, invert, r); break;
                case NPTypeCode.Int16: LoopProbe((short*)ep, n, (short*)sp, m, invert, r); break;
                case NPTypeCode.UInt16: LoopProbe((ushort*)ep, n, (ushort*)sp, m, invert, r); break;
                case NPTypeCode.Char: LoopProbe((ushort*)ep, n, (ushort*)sp, m, invert, r); break;
                case NPTypeCode.Int32: LoopProbe((int*)ep, n, (int*)sp, m, invert, r); break;
                case NPTypeCode.UInt32: LoopProbe((uint*)ep, n, (uint*)sp, m, invert, r); break;
                case NPTypeCode.Int64: LoopProbe((long*)ep, n, (long*)sp, m, invert, r); break;
                case NPTypeCode.UInt64: LoopProbe((ulong*)ep, n, (ulong*)sp, m, invert, r); break;
                case NPTypeCode.Single: LoopProbe((float*)ep, n, (float*)sp, m, invert, r); break;
                case NPTypeCode.Double: LoopProbe((double*)ep, n, (double*)sp, m, invert, r); break;
                default:
                    result.Dispose();
                    return null;
            }
            return result;
        }

        /// <summary>
        ///     The loop kernel: for each vector of elements, the OR of its equality with each broadcast set value, then one
        ///     flag per lane; four vectors per step, then one, then a scalar tail (and a scalar loop without 256-bit vectors,
        ///     which its caller does not send here).
        /// </summary>
        /// <typeparam name="T">A primitive numeric type with vector equality.</typeparam>
        /// <param name="e">The first element (contiguous, in the common dtype).</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="s">The first set value (contiguous, in the common dtype).</param>
        /// <param name="m">The number of set values, at most <see cref="LoopKernelMaxValuesPerLane"/> per vector lane.</param>
        /// <param name="invert">Whether the flags are <c>not in</c>.</param>
        /// <param name="r">The 1-D bool output, one flag per element.</param>
        /// <remarks>Optimized at its first call (see <see cref="TableProbe"/>). The set values sit on the stack, one vector
        ///     each: at most 9 KB (288 one-byte values).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void LoopProbe<T>(T* e, long n, T* s, int m, bool invert, bool* r) where T : unmanaged
        {
            long i = 0;
            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
            {
                // The set values broadcast once (m is small: the stack holds them).
                Vector256<T>* values = stackalloc Vector256<T>[m];
                for (int j = 0; j < m; j++)
                    values[j] = Vector256.Create(s[j]);
                int lanes = Vector256<T>.Count;
                // Four element vectors per step: each set value is loaded once for all four, and their four OR chains run
                // side by side. One vector per step waits on its own chain: 2.2x slower over float64 at 16 values
                // (measured; two vectors per step 1.6x).
                for (; i + 4 * lanes <= n; i += 4 * lanes)
                {
                    var v0 = Vector256.Load(e + i);
                    var v1 = Vector256.Load(e + i + lanes);
                    var v2 = Vector256.Load(e + i + 2 * lanes);
                    var v3 = Vector256.Load(e + i + 3 * lanes);
                    var h0 = Vector256<T>.Zero;
                    var h1 = Vector256<T>.Zero;
                    var h2 = Vector256<T>.Zero;
                    var h3 = Vector256<T>.Zero;
                    for (int j = 0; j < m; j++)
                    {
                        var c = values[j];
                        h0 |= Vector256.Equals(v0, c);
                        h1 |= Vector256.Equals(v1, c);
                        h2 |= Vector256.Equals(v2, c);
                        h3 |= Vector256.Equals(v3, c);
                    }
                    StoreFlags(h0, invert, r + i);
                    StoreFlags(h1, invert, r + i + lanes);
                    StoreFlags(h2, invert, r + i + 2 * lanes);
                    StoreFlags(h3, invert, r + i + 3 * lanes);
                }
                for (; i + lanes <= n; i += lanes)
                {
                    var v = Vector256.Load(e + i);
                    var hit = Vector256<T>.Zero;
                    for (int j = 0; j < m; j++)
                        hit |= Vector256.Equals(v, values[j]);
                    StoreFlags(hit, invert, r + i);
                }
            }
            for (; i < n; i++)
            {
                bool hit = false;
                for (int j = 0; j < m; j++)
                    hit |= ScalarEquals(e[i], s[j]);
                r[i] = hit != invert;
            }
        }

        /// <summary>One vector's flags as bytes 0 / 1: True where a lane compared equal (False under
        ///     <paramref name="invert"/>).</summary>
        /// <typeparam name="T">The lanes' type.</typeparam>
        /// <param name="hit">The OR of the lane equalities: each lane all ones or zero.</param>
        /// <param name="invert">Whether the flags are <c>not in</c>.</param>
        /// <param name="r">Where the <c>Vector256&lt;T&gt;.Count</c> flags go.</param>
        [MethodImpl(OptimizeAndInline)]
        private static unsafe void StoreFlags<T>(Vector256<T> hit, bool invert, bool* r) where T : unmanaged
        {
            if (sizeof(T) == 1)
            {
                // One-byte lanes are all-ones or zero: their low bit is the flag itself.
                var flags = hit.AsByte() & Vector256.Create((byte)1);
                if (invert)
                    flags ^= Vector256.Create((byte)1);
                flags.Store((byte*)r);
                return;
            }
            uint bits = hit.ExtractMostSignificantBits();
            if (invert)
                bits = ~bits;
            // The lanes' flags as bytes 0 / 1, eight lanes per bit spread (one PDEP with BMI2; the comparison kernels'
            // helper, which reads the low eight bits). Four lanes keep the low four bytes of the spread.
            int lanes = Vector256<T>.Count;
            if (lanes == 4)
                SysUnsafe.WriteUnaligned(r, (uint)DirectILKernelGenerator.SpreadBitsToBytes(bits));
            else if (lanes == 8)
                SysUnsafe.WriteUnaligned(r, DirectILKernelGenerator.SpreadBitsToBytes(bits));
            else
            {
                SysUnsafe.WriteUnaligned(r, DirectILKernelGenerator.SpreadBitsToBytes(bits));
                SysUnsafe.WriteUnaligned(r + 8, DirectILKernelGenerator.SpreadBitsToBytes(bits >> 8));
            }
        }

        /// <summary>IEEE equality of two primitive numbers (NaN is unequal to itself, -0.0 equals 0.0), which
        ///     <c>EqualityComparer&lt;T&gt;</c> is not for floats.</summary>
        /// <typeparam name="T">A primitive numeric type.</typeparam>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>True when NumPy's <c>==</c> is.</returns>
        [MethodImpl(OptimizeAndInline)]
        private static bool ScalarEquals<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(double)) return SysUnsafe.As<T, double>(ref a) == SysUnsafe.As<T, double>(ref b);
            if (typeof(T) == typeof(float)) return SysUnsafe.As<T, float>(ref a) == SysUnsafe.As<T, float>(ref b);
            if (typeof(T) == typeof(byte)) return SysUnsafe.As<T, byte>(ref a) == SysUnsafe.As<T, byte>(ref b);
            if (typeof(T) == typeof(sbyte)) return SysUnsafe.As<T, sbyte>(ref a) == SysUnsafe.As<T, sbyte>(ref b);
            if (typeof(T) == typeof(short)) return SysUnsafe.As<T, short>(ref a) == SysUnsafe.As<T, short>(ref b);
            if (typeof(T) == typeof(ushort)) return SysUnsafe.As<T, ushort>(ref a) == SysUnsafe.As<T, ushort>(ref b);
            if (typeof(T) == typeof(int)) return SysUnsafe.As<T, int>(ref a) == SysUnsafe.As<T, int>(ref b);
            if (typeof(T) == typeof(uint)) return SysUnsafe.As<T, uint>(ref a) == SysUnsafe.As<T, uint>(ref b);
            if (typeof(T) == typeof(long)) return SysUnsafe.As<T, long>(ref a) == SysUnsafe.As<T, long>(ref b);
            return SysUnsafe.As<T, ulong>(ref a) == SysUnsafe.As<T, ulong>(ref b);
        }

        /// <summary>An operand as one contiguous run the kernels read from <c>Address + offset</c>: itself when C-contiguous
        ///     (np.ravel's results are, a view at an offset included), else a C-ordered copy.</summary>
        /// <param name="a">A flattened operand.</param>
        /// <returns>A C-contiguous array with the same values.</returns>
        private static NDArray ContiguousRun(NDArray a) => a.Shape.IsContiguous ? a : NDArray.ascontiguousarray(a, null);

        /// <summary>
        ///     NumPy's <c>ar2_range &lt;= np.iinfo(ar2.dtype).max</c>, the table method's overflow constraint (bool read as
        ///     NumPy's uint8, char as its uint16).
        /// </summary>
        /// <param name="range">The set's range (<see cref="TableRange"/>).</param>
        /// <param name="tc">The set's dtype: bool, an integer dtype or char.</param>
        /// <returns>True when the range fits the dtype's maximum.</returns>
        private static bool RangeFitsDtype(BigInteger range, NPTypeCode tc) => range <= tc switch
        {
            NPTypeCode.Boolean or NPTypeCode.Byte => (BigInteger)byte.MaxValue,
            NPTypeCode.SByte => (BigInteger)sbyte.MaxValue,
            NPTypeCode.Int16 => (BigInteger)short.MaxValue,
            NPTypeCode.UInt16 or NPTypeCode.Char => (BigInteger)ushort.MaxValue,
            NPTypeCode.Int32 => (BigInteger)int.MaxValue,
            NPTypeCode.UInt32 => (BigInteger)uint.MaxValue,
            NPTypeCode.Int64 => (BigInteger)long.MaxValue,
            _ => (BigInteger)ulong.MaxValue,
        };

        /// <summary>
        ///     Whether NumPy's <c>np.concatenate((ar1, ar2))</c> keeps every value of both operands apart. It casts both to
        ///     <c>result_type(ar1, ar2)</c>, which can merge values here only as float64 or complex128, for a 64-bit integer
        ///     past 2**53 (uint64 beside a signed integer is NEP 50's float64 too): two such integers can round to one
        ///     double.
        /// </summary>
        /// <param name="ar1">The flattened tested values.</param>
        /// <param name="ar2">The flattened set.</param>
        /// <returns>True when the cast is exact for every value present. Conservative: a 64-bit integer beyond 2**53 in
        ///     magnitude counts as lossy (NumPy's statements then run, which are right either way).</returns>
        private static bool ConcatenationIsLossless(NDArray ar1, NDArray ar2)
        {
            NPTypeCode t = DTypePromotion.PromoteTypes(ar1.typecode, ar2.typecode);
            if (t != NPTypeCode.Double && t != NPTypeCode.Complex)
                return true;
            return WithinDoublePrecision(ar1) && WithinDoublePrecision(ar2);
        }

        /// <summary>
        ///     Whether every value of a 64-bit integer array lies in <c>[-2**53, 2**53]</c>, where a double holds each one
        ///     exactly; true for every other dtype and for an empty array.
        /// </summary>
        /// <param name="a">A flattened operand.</param>
        /// <returns>True when the array's values survive a cast to float64 apart.</returns>
        /// <remarks>The reductions are reclaimed here.</remarks>
        private static bool WithinDoublePrecision(NDArray a)
        {
            const long exact = 1L << 53;
            if (a.size == 0)
                return true;
            switch (a.typecode)
            {
                case NPTypeCode.Int64:
                {
                    using NDArray min = a.TensorEngine.ReduceAMin(a, null, false, null, null, null, null), max = a.TensorEngine.ReduceAMax(a, null, false, null, null, null, null);
                    return Convert.ToInt64(min.GetAtIndex(0)) >= -exact && Convert.ToInt64(max.GetAtIndex(0)) <= exact;
                }
                case NPTypeCode.UInt64:
                {
                    using NDArray max = a.TensorEngine.ReduceAMax(a, null, false, null, null, null, null);
                    return (ulong)max.GetAtIndex(0) <= (ulong)exact;
                }
                default:
                    return true;
            }
        }

        /// <summary>
        ///     Membership of every value in <paramref name="ar1"/> against the set of values in <paramref name="ar2"/>,
        ///     both already flattened. Comparison happens in <c>result_type(ar1, ar2)</c> — matching NumPy, whose
        ///     brute/sort methods concatenate/compare in the promoted dtype.
        /// </summary>
        private static NDArray ComputeMembership(NDArray ar1, NDArray ar2)
        {
            NPTypeCode t = DTypePromotion.PromoteTypes(ar1.typecode, ar2.typecode);

            // Exact-integer guard: among integer/bool pairs, ONLY uint64-with-signed promotes to Double
            // (NEP50), which loses precision past 2^53 and would report two distinct integers as equal.
            // Membership between integers must be exact (as NumPy's table method is), so route that pair
            // through the uint64 domain with a sign filter rather than compare as float64.
            if (t == NPTypeCode.Double && IsIntegerOrBool(ar1.typecode) && IsIntegerOrBool(ar2.typecode))
                return MembershipUInt64Signed(ar1, ar2);

            NDArray elemT = ar1.typecode == t ? ar1 : ar1.astype(t);
            NDArray testT = ar2.typecode == t ? ar2 : ar2.astype(t);

            // Integer counting-table fast path (NumPy's 'table' method): when the value RANGE of the
            // test set is modest relative to the operand sizes, a direct lookup gather beats the
            // O(n log m) sort/search. uint64 is excluded (its values don't fit the int64 offset math);
            // its own kind is exact via SortedSearchMembership.
            if (IsIntegerOrBool(t) && t != NPTypeCode.UInt64)
            {
                NDArray table = TryTableMembership(elemT, testT);
                if (table is not null)
                    return table;
            }

            return SortedSearchMembership(elemT, testT);
        }

        /// <summary>
        ///     NumPy's counting-table membership: build a boolean lookup over [min(test), max(test)],
        ///     then gather one bool per element. Returns null (declining) when the range exceeds
        ///     6×(|element|+|test|) — NumPy's own below-memory heuristic — leaving the sort path to run.
        ///     Both operands are the same exact integer dtype (not uint64), so int64 offset math is exact.
        /// </summary>
        [NDScopedCovered] // only reached via ComputeMembership, under the [NDScoped] isin scope
        private static NDArray TryTableMembership(NDArray elemT, NDArray testT)
        {
            // Booleans have no arithmetic; NumPy promotes them to uint8 for the table (the result is
            // still the bool membership mask).
            if (elemT.typecode == NPTypeCode.Boolean) elemT = elemT.astype(NPTypeCode.Byte);
            if (testT.typecode == NPTypeCode.Boolean) testT = testT.astype(NPTypeCode.Byte);

            NDArray minS = testT.TensorEngine.ReduceAMin(testT, null, false, null, null, null, null);                          // 0-d, dtype of the operands
            NDArray maxS = testT.TensorEngine.ReduceAMax(testT, null, false, null, null, null, null);
            long minI = Convert.ToInt64(minS.GetAtIndex(0));
            long maxI = Convert.ToInt64(maxS.GetAtIndex(0));
            ulong urange = unchecked((ulong)(maxI - minI));
            if (urange > 6UL * (ulong)(elemT.size + testT.size))
                return null;                                       // range too wide -> sort path
            long range = (long)urange;

            NDArray table = new NDArray(NPTypeCode.Boolean, new Shape(range + 1), true);
            table[testT - minS] = (NDArray)true;                   // scatter: mark each present value

            // Stay in the operands' own dtype (no widening of the large element array). Out-of-range
            // elements clamp to a valid slot via take(mode='clip') and are then forced False by the
            // range mask regardless of the (possibly wrapped) offset — so no explicit bounds check.
            NDArray offset = elemT - minS;
            NDArray gathered = UnmanagedStorage.TakeFrom(table, offset, null, null, ClipModes.ClipModeClip);   // np.take(..., mode='clip')
            NDArray inRange = (elemT >= minS) & (elemT <= maxS);
            return gathered & inRange;
        }

        /// <summary>
        ///     Core membership over two operands ALREADY in the same dtype: sort the test set, then a
        ///     searchsorted + equality probe answers membership. Duplicates are harmless (side='left' finds
        ///     the first equal slot); NaN never equals itself, so a NaN element is never a member.
        /// </summary>
        [NDScopedCovered] // only reached via ComputeMembership / MembershipUInt64Signed, under the [NDScoped] isin scope
        private static NDArray SortedSearchMembership(NDArray elemT, NDArray testT)
        {
            // Hash-set membership (O(n+m)) replaces the sort + n·searchsorted below for hashable
            // dtypes. That searchsorted — n binary searches over the m-element sorted test set — was
            // the isin bottleneck (1M float64: ~317ms of a ~213ms call; NumPy's in1d is itself
            // sort-based here, so hashing BEATS it). Integer bit patterns are exact; floats normalize
            // -0.0→+0.0 and exclude NaN (NaN never equals itself — never a member). Complex/Half/
            // Decimal keep the sort path (Decimal needs value-equality: 1.0m == 1.00m by value, not
            // bits). Returns null when it declines (huge test set) so the sort path runs.
            NDArray hashed = TryHashMembership(elemT, testT);
            if (hashed is not null)
                return hashed;

            NDArray u = testT.TensorEngine.Sort(testT, -1, null, null, null);
            long m = u.size;
            if (m == 0)
                return new NDArray(NPTypeCode.Boolean, new Shape(elemT.size), true);

            NDArray idx = u.TensorEngine.SearchSorted(u, elemT, "left", null);   // int64, shape == elemT.shape (1-D)
            NDArray clamped = idx.TensorEngine.Minimum(idx, (NDArray)(m - 1), null, null, null);   // stay in-bounds for the gather
            NDArray gathered = UnmanagedStorage.TakeFrom(u, clamped, null, null, ClipModes.ClipModeRaise);   // np.take(u, clamped)
            NDArray inRange = idx < (NDArray)m;                     // idx == m => greater than all => not found
            return (gathered == elemT) & inRange;
        }

        /// <summary>
        ///     Open-addressing hash-set membership: build a table of <paramref name="testT"/>'s
        ///     (normalized) bit patterns, then probe every element of <paramref name="elemT"/> — O(n+m)
        ///     vs the sort path's O(m log m + n log m). Handles the hashable dtypes (integer-family +
        ///     char + float32/float64); declines (returns null) for Complex/Half/Decimal and for a test
        ///     set too large to index, leaving <see cref="SortedSearchMembership"/> to run.
        /// </summary>
        private static NDArray TryHashMembership(NDArray elemT, NDArray testT)
        {
            // elemT and testT are the SAME dtype here (both cast to result_type upstream).
            if (testT.size > (1L << 29))   // ~512M distinct-candidates: fall back rather than over-allocate the table
                return null;
            switch (elemT.typecode)
            {
                case NPTypeCode.Boolean: return HashMembership<bool>(elemT, testT, floatSem: false);
                case NPTypeCode.Byte: return HashMembership<byte>(elemT, testT, floatSem: false);
                case NPTypeCode.SByte: return HashMembership<sbyte>(elemT, testT, floatSem: false);
                case NPTypeCode.Int16: return HashMembership<short>(elemT, testT, floatSem: false);
                case NPTypeCode.UInt16: return HashMembership<ushort>(elemT, testT, floatSem: false);
                case NPTypeCode.Char: return HashMembership<char>(elemT, testT, floatSem: false);
                case NPTypeCode.Int32: return HashMembership<int>(elemT, testT, floatSem: false);
                case NPTypeCode.UInt32: return HashMembership<uint>(elemT, testT, floatSem: false);
                case NPTypeCode.Int64: return HashMembership<long>(elemT, testT, floatSem: false);
                case NPTypeCode.UInt64: return HashMembership<ulong>(elemT, testT, floatSem: false);
                case NPTypeCode.Single: return HashMembership<float>(elemT, testT, floatSem: true);
                case NPTypeCode.Double: return HashMembership<double>(elemT, testT, floatSem: true);
                default: return null;   // Complex / Half / Decimal → sort path
            }
        }

        private static unsafe NDArray HashMembership<T>(NDArray elemT, NDArray testT, bool floatSem)
            where T : unmanaged
        {
            // `.Address` ignores Shape.offset, so a sliced/broadcast view must be materialized first
            // (the upstream astype already produces a fresh contiguous array when the dtype changed).
            if (elemT.Shape.IsSliced || elemT.Shape.IsBroadcasted) elemT = elemT.copy();
            if (testT.Shape.IsSliced || testT.Shape.IsBroadcasted) testT = testT.copy();
            long m = testT.size, n = elemT.size;

            // Pre-size to load factor ≤ 0.5 (next pow2 ≥ 2·m) so no growth/rehash is needed.
            long want = m * 2 + 1;
            int cap = 1024;
            while (cap < want) cap <<= 1;
            int mask = cap - 1;
            var keys = new ulong[cap];
            var used = new bool[cap];

            T* tp = (T*)testT.Address;
            for (long i = 0; i < m; i++)
            {
                T v = tp[i];
                if (floatSem && IsNaNFloat(v)) continue;   // NaN is never a member
                ulong k = KeyBits(v, floatSem);
                int h = (int)(Splitmix(k) & (ulong)mask);
                while (used[h]) { if (keys[h] == k) goto inserted; h = (h + 1) & mask; }
                used[h] = true; keys[h] = k;
                inserted: ;
            }

            var result = new NDArray(NPTypeCode.Boolean, new Shape(n), false);
            bool* rp = (bool*)result.Address;
            T* ep = (T*)elemT.Address;
            for (long i = 0; i < n; i++)
            {
                T v = ep[i];
                if (floatSem && IsNaNFloat(v)) { rp[i] = false; continue; }
                ulong k = KeyBits(v, floatSem);
                int h = (int)(Splitmix(k) & (ulong)mask);
                bool found = false;
                while (used[h]) { if (keys[h] == k) { found = true; break; } h = (h + 1) & mask; }
                rp[i] = found;
            }
            return result;
        }

        /// <summary>Normalized bit key: raw zero-extended bits for integers; for floats, -0.0→+0.0
        /// collapses to 0 so signed zeros share a bucket (NaN is filtered by the caller).</summary>
        [MethodImpl(OptimizeAndInline)]
        private static ulong KeyBits<T>(T v, bool floatSem) where T : unmanaged
        {
            if (floatSem)
            {
                if (typeof(T) == typeof(double)) { double d = SysUnsafe.As<T, double>(ref v); return d == 0.0 ? 0UL : BitConverter.DoubleToUInt64Bits(d); }
                if (typeof(T) == typeof(float)) { float f = SysUnsafe.As<T, float>(ref v); return f == 0f ? 0UL : BitConverter.SingleToUInt32Bits(f); }
            }
            int sz = SysUnsafe.SizeOf<T>();
            if (sz == 1) return SysUnsafe.As<T, byte>(ref v);
            if (sz == 2) return SysUnsafe.As<T, ushort>(ref v);
            if (sz == 4) return SysUnsafe.As<T, uint>(ref v);
            return SysUnsafe.As<T, ulong>(ref v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsNaNFloat<T>(T v) where T : unmanaged
        {
            if (typeof(T) == typeof(double)) return double.IsNaN(SysUnsafe.As<T, double>(ref v));
            if (typeof(T) == typeof(float)) return float.IsNaN(SysUnsafe.As<T, float>(ref v));
            return false;
        }

        /// <summary>splitmix64 finalizer — avalanches every bit into the low bits used as the table index.</summary>
        [MethodImpl(OptimizeAndInline)]
        private static ulong Splitmix(ulong k)
        {
            k = (k ^ (k >> 30)) * 0xBF58476D1CE4E5B9UL;
            k = (k ^ (k >> 27)) * 0x94D049BB133111EBUL;
            return k ^ (k >> 31);
        }

        /// <summary>
        ///     Exact membership when one operand is uint64 and the other is a signed integer. Equality can
        ///     only hold in the overlap [0, 2^63-1]: a negative signed value never equals a uint64, and a
        ///     uint64 above int64.max never equals a signed value. Reduce to a uint64 comparison and mask
        ///     the signed operand's negatives out.
        /// </summary>
        [NDScopedCovered] // only reached via ComputeMembership, under the [NDScoped] isin scope
        private static NDArray MembershipUInt64Signed(NDArray ar1, NDArray ar2)
        {
            if (ar1.typecode == NPTypeCode.UInt64)
            {
                // element uint64, test signed: only test's non-negative values can match anything.
                NDArray testNonNeg = ar2[ar2 >= (NDArray)0L].astype(NPTypeCode.UInt64);
                return SortedSearchMembership(ar1, testNonNeg);
            }
            // element signed, test uint64: reinterpret element as uint64 for the probe, then exclude
            // negatives (whose reinterpreted huge value could otherwise collide with a real uint64 test value).
            NDArray raw = SortedSearchMembership(ar1.astype(NPTypeCode.UInt64), ar2);
            return raw & (ar1 >= (NDArray)0L);
        }

        /// <summary>
        ///     Whether a dtype takes the membership routes' integer arithmetic: bool and the eight integer dtypes. Char is not
        ///     one here: its arithmetic is not uint16's, so a char pair stays off the counting table and on the exact hash
        ///     route (for NumPy's table-method test, where char counts, see <see cref="IsNumPyIntegerKind"/>).
        /// </summary>
        /// <param name="tc">An operand's dtype (or the dtype both are cast to).</param>
        /// <returns>True for bool and the integer dtypes.</returns>
        private static bool IsIntegerOrBool(NPTypeCode tc)
        {
            switch (tc)
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
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     NumPy's <c>ar.dtype.kind in ("u", "i", "b")</c>, the test of the arrays its table method takes: bool, the
        ///     integer dtypes, and NumSharp's char, which stands for NumPy's uint16 (the proxy np.searchsorted's
        ///     <c>StringDType</c> route reads too).
        /// </summary>
        /// <param name="tc">An operand's dtype.</param>
        /// <returns>True for bool, the integer dtypes and char.</returns>
        /// <remarks>A test of kinds only: the membership itself keeps <see cref="IsIntegerOrBool"/>'s routes, which answer a
        ///     char operand exactly.</remarks>
        private static bool IsNumPyIntegerKind(NPTypeCode tc) => tc == NPTypeCode.Char || IsIntegerOrBool(tc);

        /// <summary>
        ///     NumPy's <c>ar2_range = int(np.max(ar2)) - int(np.min(ar2))</c> of the table method, exactly (a Python int): a
        ///     bool set reads as NumPy's uint8 0 / 1, a char set as its uint16 values, a uint64 set past 2**63 as itself.
        /// </summary>
        /// <param name="ar2">The flattened set: bool, an integer dtype or char, not empty.</param>
        /// <returns>The range, never negative.</returns>
        private static BigInteger TableRange(NDArray ar2)
        {
            var (min, max) = TableBounds(ar2);
            return max - min;
        }

        /// <summary>
        ///     NumPy's <c>int(np.min(ar2))</c> and <c>int(np.max(ar2))</c> of the table method, exactly (Python ints), read as
        ///     <see cref="TableRange"/> reads them.
        /// </summary>
        /// <param name="ar2">The flattened set: bool, an integer dtype or char, not empty.</param>
        /// <returns>The set's minimum and maximum.</returns>
        /// <remarks>The two reductions are reclaimed here; the caller's scope covers nothing of them.</remarks>
        private static (BigInteger Min, BigInteger Max) TableBounds(NDArray ar2)
        {
            using NDArray min = ar2.TensorEngine.ReduceAMin(ar2, null, false, null, null, null, null), max = ar2.TensorEngine.ReduceAMax(ar2, null, false, null, null, null, null);
            return (IsinPythonInt(min.GetAtIndex(0)), IsinPythonInt(max.GetAtIndex(0)));
        }

        /// <summary>
        ///     An integer element as the Python int NumPy's <c>int()</c> makes of it: bool as 0 / 1, char as its code (the
        ///     uint16 value), uint64 without a sign, every other integer through <c>long</c>.
        /// </summary>
        /// <param name="value">A boxed element of a bool, integer or char array.</param>
        /// <returns>Its value.</returns>
        /// <exception cref="InvalidCastException">A boxed value of another kind (no caller passes one).</exception>
        private static BigInteger IsinPythonInt(object value) => value switch
        {
            bool b => b ? BigInteger.One : BigInteger.Zero,
            char c => new BigInteger((int)c),
            ulong u => new BigInteger(u),
            _ => new BigInteger(Convert.ToInt64(value)),
        };

        // range = max(ar2) - min(ar2); safe iff it fits the positive span of ar2's dtype (NumPy's
        // ar2_range <= iinfo(ar2.dtype).max). Only signed integer test dtypes can overflow: for unsigned
        // (and bool, treated as uint8) the range is always <= the type's max by construction.
        private static bool TableRangeIsSafe(NDArray ar2)
        {
            switch (ar2.typecode)
            {
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.Int32:
                case NPTypeCode.Int64:
                    long mn = Convert.ToInt64(ar2.TensorEngine.ReduceAMin(ar2, null, false, null, null, null, null).GetAtIndex(0));
                    long mx = Convert.ToInt64(ar2.TensorEngine.ReduceAMax(ar2, null, false, null, null, null, null).GetAtIndex(0));
                    ulong range = unchecked((ulong)(mx - mn));      // two's-complement distance, exact for all signed
                    ulong typeMax = ar2.typecode switch
                    {
                        NPTypeCode.SByte => (ulong)sbyte.MaxValue,
                        NPTypeCode.Int16 => (ulong)short.MaxValue,
                        NPTypeCode.Int32 => (ulong)int.MaxValue,
                        _ => (ulong)long.MaxValue,
                    };
                    return range <= typeMax;
                default:
                    return true;    // unsigned / bool: range always fits
            }
        }
    }
}
