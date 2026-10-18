using System;
using System.Collections.Generic;
using System.IO;

namespace NumSharp.Tests.Fuzz
{
    internal static partial class LeakCatalogue
    {
        /// <summary>
        ///     The entries of <see cref="NDListArray"/>'s members that compare, sort, reduce, compute and assign lists (L-B /
        ///     L-C): every result is a new list array (its own heap or a new reference on the shared one) or a NumSharp array,
        ///     which the harness disposes; every intermediate an entry makes is disposed inside it, so a member that strands an
        ///     id array, a heap or a converted operand shows as an escaped pool buffer.
        /// </summary>
        /// <param name="l">The entry list.</param>
        private static void AddListArrayCompute(List<LeakCase> l)
        {
            // A fresh split result per entry (released inside it): the (4,) row r and the (2, 3) grid of the unit tests.
            static (NDArray Text, NDListArray Lists) Row()
            {
                var u = np.array(new[] { "a b", "c", "", "d e f" });
                return (u, np.@char.split(u));
            }
            static (NDArray Text, NDListArray Lists) Grid()
            {
                var u = np.array(new[,] { { "a b", "c", "x y" }, { "", "d e f", "g" } });
                return (u, np.@char.split(u));
            }

            // ---- comparisons: list arrays, a reversed view, a list stand-in, an ndarray, a str, a reflected value ----
            AddListComparison(l, "NDListArray.op_Equality", (a, b) => a == b, (a, x) => a == x, (x, a) => x == a, (a, s) => a == s, false);
            AddListComparison(l, "NDListArray.op_Inequality", (a, b) => a != b, (a, x) => a != x, (x, a) => x != a, (a, s) => a != s, false);
            AddListComparison(l, "NDListArray.op_LessThan", (a, b) => a < b, (a, x) => a < x, (x, a) => x < a, (a, s) => a < s, true);
            AddListComparison(l, "NDListArray.op_LessThanOrEqual", (a, b) => a <= b, (a, x) => a <= x, (x, a) => x <= a, (a, s) => a <= s, true);
            AddListComparison(l, "NDListArray.op_GreaterThan", (a, b) => a > b, (a, x) => a > x, (x, a) => x > a, (a, s) => a > s, true);
            AddListComparison(l, "NDListArray.op_GreaterThanOrEqual", (a, b) => a >= b, (a, x) => a >= x, (x, a) => x >= a, (a, s) => a >= s, true);
            E(l, "NDListArray.Equals", "the same object and another list array", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var other = lists.copy())
                    return Box(lists.Equals(lists) && !lists.Equals(other));
            });
            E(l, "NDListArray.GetHashCode", "the identity hash", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                    return Box(lists.GetHashCode());
            });

            // ---- + and * ----
            E(l, "NDListArray.op_Addition", "two list arrays, a stand-in, an ndarray (TypeError), a str, None", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var rev = lists["::-1"])
                using (var head = lists[":2"])
                using (var first = lists["0"])
                using (var ints = np.array(new long[] { 1, 2, 3, 4 }))
                {
                    using var col = lists.reshape(4, 1);
                    using var outer = lists + col;
                    Swallow<TypeError>(() => _ = head + first);
                    Swallow<TypeError>(() => _ = lists + ints);
                    Swallow<TypeError>(() => _ = "x" + lists);
                    Swallow<TypeError>(() => _ = lists + (object)null);
                    return lists + rev;
                }
            });
            E(l, "NDListArray.op_Multiply", "an int, counts, a reflected int, the float / list / MemoryError paths", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var counts = np.array(new long[] { 0, 1, 2, 3 }))
                using (var one = lists["1:2"])
                using (var big = NDArray.Scalar(1L << 62))
                {
                    using var twice = 2 * lists;
                    using var each = lists * counts;
                    Swallow<TypeError>(() => _ = lists * 1.5);
                    Swallow<TypeError>(() => _ = lists * lists);
                    Swallow<OutOfMemoryException>(() => _ = one * big);
                    return lists * 2;
                }
            });

            // ---- truth / conversions ----
            E(l, "NDListArray.all", "flat, along an axis, keepdims", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                {
                    using var flat = lists.all();
                    using var kept = lists.all(1, keepdims: true);
                    return lists.all(0);
                }
            });
            E(l, "NDListArray.any", "flat, along an axis", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                {
                    using var flat = lists.any();
                    return lists.any(-1);
                }
            });
            E(l, "NDListArray.nonzero", "the index arrays of a 2-d result; the 0-d refusal", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var zt = np.array("a b"))
                using (var z = np.@char.split(zt))
                {
                    Swallow<ValueError>(() => z.nonzero());
                    var nz = lists.nonzero();
                    nz[1].Dispose();
                    return nz[0];
                }
            });
            E(l, "NDListArray.astype", "bool in F order, StringDType reprs, the int / float / casting refusals", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                {
                    using var texts = lists.astype(np.dtypes.StringDType());
                    Swallow<TypeError>(() => lists.astype(np.dtype("int64")));
                    Swallow<ValueError>(() => lists.astype(np.dtype("float64")));
                    Swallow<TypeError>(() => lists.astype(np.dtype("bool"), casting: "safe"));
                    return lists.astype(np.dtype("bool"), 'F');
                }
            });

            // ---- sort family ----
            E(l, "NDListArray.sort", "in place along each axis, stable, the read-only error", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                {
                    lists.sort(0);
                    lists.sort(kind: "stable");
                    using var ro = lists.copy();
                    ro.setflags(write: false);
                    Swallow<ValueError>(() => ro.sort());
                    return lists;
                }
            });
            E(l, "NDListArray.argsort", "the default and stable kinds, flattened, a 0-d array", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var zt = np.array("a b"))
                using (var z = np.@char.split(zt))
                {
                    using var stable = lists.argsort(kind: "stable");
                    using var flat = lists.argsort(null);
                    using var single = z.argsort();
                    return lists.argsort(0);
                }
            });
            E(l, "NDListArray.partition", "one kth, several, the bounds error", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var kths = np.array(new long[] { 0, 3 }))
                {
                    lists.partition(1);
                    lists.partition(kths);
                    Swallow<ValueError>(() => lists.partition(9));
                    return lists;
                }
            });
            E(l, "NDListArray.argpartition", "one kth, several, the bounds error", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                {
                    using var several = lists.argpartition(new long[] { 0, 2 });
                    Swallow<ValueError>(() => lists.argpartition(9));
                    return lists.argpartition(1);
                }
            });
            E(l, "NDListArray.searchsorted", "lists, a stand-in (its items), a sorter, a str (TypeError)", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var order = lists.argsort())
                using (var first = lists["2"])
                {
                    using var sorted = lists.searchsorted(lists, sorter: order);
                    using var items = lists.searchsorted(first);
                    Swallow<TypeError>(() => lists.searchsorted("x"));
                    return lists.searchsorted(lists, "right");
                }
            });
            E(l, "NDListArray.argmax", "flat, along an axis with keepdims, the empty error", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var et = np.array(new string[0]))
                using (var empty = np.@char.split(et))
                {
                    using var kept = lists.argmax(1, keepdims: true);
                    Swallow<ValueError>(() => empty.argmax());
                    return lists.argmax();
                }
            });
            E(l, "NDListArray.argmin", "flat and along an axis", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                {
                    using var flat = lists.argmin();
                    return lists.argmin(0);
                }
            });

            // ---- reductions ----
            E(l, "NDListArray.max", "flat (a list), an axis, initial, where + initial, out=, the empty error", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var mask = np.array(new[] { true, false, true }))
                using (var et = np.array(new string[0]))
                using (var empty = np.@char.split(et))
                using (var row = lists["0"])
                using (var o = row.copy())
                {
                    using var flat = lists.max();
                    using var initial = lists.max(0, initial: new[] { "m" });
                    using var masked = lists.max(where: mask, initial: new[] { "zz" });
                    _ = lists.max(0, o);
                    Swallow<ValueError>(() => empty.max());
                    return lists.max(1, keepdims: true);
                }
            });
            E(l, "NDListArray.min", "flat and along an axis", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                {
                    using var flat = lists.min();
                    return lists.min(0);
                }
            });
            E(l, "NDListArray.sum", "concatenations: flat, an axis, initial, dtype=bool, the where error", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var mask = np.array(new[] { true, false, true }))
                {
                    using var flat = lists.sum();
                    using var initial = lists.sum(initial: new[] { "z" });
                    using var truth = lists.sum(null, np.dtype("bool"));
                    Swallow<ValueError>(() => lists.sum(where: mask));
                    return lists.sum(0);
                }
            });
            E(l, "NDListArray.prod", "one element (the list), an int initial, the list * list error", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var one = lists["1:2"])
                {
                    using var single = one.prod();
                    Swallow<TypeError>(() => lists.prod());
                    return one.prod(initial: 2);
                }
            });
            E(l, "NDListArray.cumsum", "flattened, along each axis, into out=", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var o = lists.copy())
                {
                    using var flat = lists.cumsum();
                    _ = lists.cumsum(1, o);
                    return lists.cumsum(0);
                }
            });
            E(l, "NDListArray.cumprod", "one element per line, the list * list error", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var top = lists[":1"])
                {
                    Swallow<TypeError>(() => lists.cumprod());
                    return top.cumprod(0);
                }
            });
            E(l, "NDListArray.trace", "the main and an offset diagonal's concatenation, the rank error", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var flat = lists.ravel())
                {
                    using var main = lists.trace();
                    Swallow<ValueError>(() => flat.trace());
                    return lists.trace(1);
                }
            });
            E(l, "NDListArray.clip", "list bounds, a stand-in bound (its items), the str and unary errors", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var lo = lists["1:2"])
                using (var hi = lists["3:4"])
                using (var empty = lists["2"])
                {
                    using var low = lists.clip(lo);
                    Swallow<IncorrectShapeException>(() => lists.clip(empty));
                    Swallow<TypeError>(() => lists.clip("x", "y"));
                    Swallow<TypeError>(() => lists.clip());
                    return lists.clip(lo, hi);
                }
            });
            E(l, "NDListArray.mean", "an empty array (NaN), an empty list (an empty float64), dtype=bool, the TypeError", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var et = np.array(new string[0]))
                using (var empty = np.@char.split(et))
                using (var blank = lists["2:3"])
                {
                    using var nan = empty.mean();
                    using var none = blank.mean();
                    Swallow<TypeError>(() => lists.mean());
                    return lists.mean(null, np.dtype("bool"));
                }
            });
            E(l, "NDListArray.var", "the TypeError, the empty ZeroDivisionError, a 0-d empty list (0.0)", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var et = np.array(new string[0]))
                using (var empty = np.@char.split(et))
                using (var zt = np.array(""))
                using (var z = np.@char.split(zt))
                {
                    Swallow<TypeError>(() => lists.var());
                    Swallow<DivideByZeroException>(() => empty.var());
                    return z.var();
                }
            });
            E(l, "NDListArray.std", "the TypeError and a 0-d empty list", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var zt = np.array(""))
                using (var z = np.@char.split(zt))
                {
                    Swallow<TypeError>(() => lists.std());
                    return z.std();
                }
            });

            // ---- dot / round / conj ----
            E(l, "NDListArray.dot", "counts (a list), a 2-d grid, a scalar, out=, the alignment and list errors", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var w = np.array(new long[] { 1, 2, 0 }))
                using (var bad = np.array(new long[] { 1, 2 }))
                using (var column = lists[":, 0"])
                using (var o = column.copy())
                using (var t = lists.T)
                {
                    using var scalar = lists.dot(2);
                    _ = lists.dot(w, o);
                    Swallow<IncorrectShapeException>(() => lists.dot(bad));
                    Swallow<TypeError>(() => lists.dot(t));
                    return lists.dot(w);
                }
            });
            E(l, "NDListArray.round", "an empty array, the TypeError", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var et = np.array(new string[0]))
                using (var empty = np.@char.split(et))
                {
                    Swallow<TypeError>(() => lists.round(2));
                    return empty.round();
                }
            });
            E(l, "NDListArray.conj", "an empty array, the TypeError", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var et = np.array(new string[0]))
                using (var empty = np.@char.split(et))
                {
                    Swallow<TypeError>(() => lists.conj());
                    return empty.conj();
                }
            });
            E(l, "NDListArray.conjugate", "an empty array into out=", f =>
            {
                using var et = np.array(new string[0]);
                using var empty = np.@char.split(et);
                using var o = empty.copy();
                return Box(ReferenceEquals(empty.conjugate(o), o));
            });

            // ---- assignment ----
            E(l, "NDListArray.fill", "a list stand-in, a C# list (a new list in the shared store), the read-only error", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var first = lists["3"])
                {
                    lists.fill(first);
                    lists.fill(new[] { "z", "w" });
                    using var ro = lists.copy();
                    ro.setflags(write: false);
                    Swallow<ValueError>(() => ro.fill(new[] { "x" }));
                    return lists;
                }
            });
            E(l, "NDListArray.put", "cycling values, clip, lists of another store, the bounds error", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var ot = np.array(new[] { "q r", "s" }))
                using (var other = np.@char.split(ot))
                using (var indices = np.array(new long[] { 0, 1, 2 }))
                using (var far = np.array(new long[] { 10 }))
                using (var values = lists["1:3"])
                {
                    lists.put(indices, values);
                    lists.put(far, other, "clip");
                    Swallow<IndexError>(() => lists.put(far, values));
                    return lists;
                }
            });
            E(l, "NDListArray.resize", "a shrink in place, the growth refusal, the view refusal", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var view = lists["0"])
                {
                    Swallow<IncorrectShapeException>(() => view.resize(6));
                    lists.resize(new Shape(3, 1), false);
                    Swallow<NotSupportedException>(() => lists.resize(new Shape(9), false));
                    return lists.copy();
                }
            });
            E(l, "NDListArray.setflags", "write off and on again", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                {
                    lists.setflags(write: false);
                    lists.setflags(write: true);
                    return Box(lists.flags.writeable);
                }
            });
            E(l, "NDListArray.byteswap", "a copy in A order, in place (the array itself), the read-only error", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var t = lists.T)
                {
                    _ = lists.byteswap(true);
                    using var ro = lists.copy();
                    ro.setflags(write: false);
                    Swallow<ValueError>(() => ro.byteswap(true));
                    return t.byteswap();
                }
            });
            T(l, "NDListArray.getfield", "raises by design: an object array has no field NumSharp can spell (NumPy's _getfield_is_safe)", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                    return lists.getfield(np.dtype("int64"));
            });
            T(l, "NDListArray.setfield", "raises by design: the writeability, then getfield's refusal", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                {
                    lists.setfield(1, np.dtype("int64"));
                    return null;
                }
            });
            T(l, "NDListArray.choose", "raises by design: an object array does not cast to intp safely", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                    return lists.choose(new[] { 0 });
            });
            T(l, "NDListArray.tobytes", "raises by design: an object array's bytes are PyObject pointers", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                    return lists.tobytes();
            });
            E(l, "NDListArray.tofile", "text mode with the default and a padded format; the binary refusal", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var stream = new MemoryStream())
                {
                    lists.tofile(stream, "|");
                    lists.tofile(stream, "|", "%-12s");
                    Swallow<OSError>(() => lists.tofile(stream));
                    return Box(stream.Length);
                }
            });
            E(l, "NDListArray.view", "a view as another dtype (the TypeError)", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                {
                    Swallow<TypeError>(() => lists.view(np.dtype("int64")));
                    return lists.view();
                }
            });
            E(l, "NDListArray.take", "into out= of the same store and of another store", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var ot = np.array(new[] { "q r", "s" }))
                using (var other = np.@char.split(ot))
                using (var indices = np.array(new long[] { 3, 0 }))
                using (var head = lists[":2"])
                using (var own = head.copy())
                {
                    _ = lists.take(indices, null, own);
                    _ = lists.take(indices, null, other);
                    return lists.take(indices);
                }
            });
            E(l, "NDListArray.compress", "into out=", f =>
            {
                var (u, lists) = Row();
                using (u)
                using (lists)
                using (var condition = np.array(new[] { true, false, true }))
                using (var head = lists[":2"])
                using (var o = head.copy())
                {
                    _ = lists.compress(condition, null, o);
                    return lists.compress(condition);
                }
            });
            E(l, "NDListArray.Item", "element and sub-array assignment through the indexer, the sequence error", f =>
            {
                var (u, lists) = Grid();
                using (u)
                using (lists)
                using (var ot = np.array(new[] { "q r", "s", "t" }))
                using (var other = np.@char.split(ot))
                using (var first = other["0"])
                {
                    lists["0, 1"] = first;
                    lists[":, :"] = other;
                    // A list into a 3-long sub-array is the sequence of its 2 items: NumPy's broadcast error.
                    Swallow<ValueError>(() => lists["0, :"] = first);
                    return lists.copy();
                }
            });
        }

        /// <summary>
        ///     The entry of one list-array comparison operator: two list arrays (broadcast), a reversed view, a list stand-in
        ///     (its items), an ndarray, a str, and a reflected value; an ordering raises against the ndarray, the str and the
        ///     items.
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <param name="api">The operator's inventory id.</param>
        /// <param name="lists">The operator over two list arrays.</param>
        /// <param name="value">The operator with a value on the right.</param>
        /// <param name="reflected">The operator with a value on the left.</param>
        /// <param name="str">The operator with a C# string on the right.</param>
        /// <param name="ordering">True for an ordering (the non-list operands raise).</param>
        private static void AddListComparison(List<LeakCase> l, string api, Func<NDListArray, NDListArray, NDArray> lists,
            Func<NDListArray, object, NDArray> value, Func<object, NDListArray, NDArray> reflected, Func<NDListArray, string, NDArray> str,
            bool ordering)
        {
            E(l, api, "list arrays, a reversed view, a stand-in, an ndarray, a str, a reflected value", f =>
            {
                var u = np.array(new[] { "a b", "c", "", "d e f" });
                using (u)
                using (var r = np.@char.split(u))
                using (var rev = r["::-1"])
                using (var col = r.reshape(4, 1))
                using (var head = r[":2"])
                using (var first = r["0"])
                using (var ints = np.array(new long[] { 1, 2, 3, 4 }))
                {
                    using var outer = lists(r, col);
                    if (ordering)
                    {
                        Swallow<TypeError>(() => lists(head, first));
                        Swallow<TypeError>(() => value(r, ints));
                        Swallow<TypeError>(() => str(r, "x"));
                        Swallow<TypeError>(() => reflected(1, r));
                    }
                    else
                    {
                        using var items = lists(head, first);
                        using var numbers = value(r, ints);
                        using var text = str(r, "x");
                        using var left = reflected(1, r);
                    }
                    return lists(r, rev);
                }
            });
        }

        /// <summary>Runs an action that must throw <typeparamref name="TException"/> (an error path of an entry).</summary>
        /// <typeparam name="TException">The exception expected.</typeparam>
        /// <param name="action">The call.</param>
        /// <exception cref="InvalidOperationException">The call did not throw.</exception>
        private static void Swallow<TException>(Action action) where TException : Exception
        {
            try
            {
                action();
            }
            catch (TException)
            {
                return;
            }
            throw new InvalidOperationException($"the entry's error path did not raise {typeof(TException).Name}");
        }
    }
}
