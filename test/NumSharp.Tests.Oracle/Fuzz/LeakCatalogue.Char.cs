using System;
using System.Collections.Generic;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Catalogue entries for <c>numpy.char.chararray</c> as data (the np-char plan's phase 2): the module's
    ///     <c>chararray</c> / <c>array</c> / <c>asarray</c> and every member <see cref="NDCharArray"/> declares. The string
    ///     oracle's section <c>strchararray</c> replays them through its own harness, outside the corpus sweep, so each is
    ///     measured here: the routes that allocate (a str filler, the <c>tolist()</c> route of numbers, the casts, the
    ///     element stand-ins, the gathered results a chararray wraps as a view) and the refusals (the finalize ValueError,
    ///     Python's ZeroDivisionError), whose temporaries must be reclaimed too.
    /// </summary>
    /// <remarks>
    ///     Every entry builds its string arrays itself and disposes them before it returns (the fixture holds none): a
    ///     chararray view holds a counted reference on its base's buffer, so the result the harness disposes keeps what it
    ///     needs alive and nothing else is left to the finalizer.
    /// </remarks>
    internal static partial class LeakCatalogue
    {
        /// <summary>The <c>np.char</c> constructors and NDCharArray's declared members.</summary>
        /// <param name="l">The entry list.</param>
        private static void AddCharArray(List<LeakCase> l)
        {
            // ---- np.char.chararray / array / asarray ----
            E(l, "np.char.chararray", "no buffer (U, F order); a str filler", f =>
            {
                using var empty = np.@char.chararray(new Shape(2, 3), 2, unicode: true, order: 'F');
                return np.@char.chararray(new Shape(3), 2, unicode: true, buffer: "ab");
            });
            E(l, "np.char.chararray", "bytes, bytearray and ndarray buffers; the too-small refusal", f =>
            {
                using var ro = np.@char.chararray(new Shape(2), 2, buffer: new ReadOnlyMemory<byte>(new byte[] { 97, 98, 99, 100 }));
                using var rw = np.@char.chararray(new Shape(2), 1, buffer: new byte[] { 97, 98, 99, 100 }, offset: 3, strides: new long[] { -2 });
                try
                {
                    np.@char.chararray(new Shape(3), 6, buffer: new byte[] { 97, 98 });
                }
                catch (TypeError)
                {
                }
                using var src = np.array(new[] { "ab", "cd" });
                return np.@char.chararray(new Shape(2), 1, unicode: true, buffer: src);
            });
            E(l, "np.char.array", "a str (filled), bytes (read-only buffer), the empty str's ZeroDivisionError", f =>
            {
                using var text = np.@char.array("abcdef", 2);
                try
                {
                    np.@char.array("");
                }
                catch (DivideByZeroException)
                {
                }
                return np.@char.array(new byte[] { 97, 98, 99, 100 }, 2);
            });
            E(l, "np.char.array", "lists of str and of numbers (np.asarray first)", f =>
            {
                using var words = np.@char.array(new[] { "ab", "c" });
                return np.@char.array(new object[] { 1L, 22L, "x" });
            });
            E(l, "np.char.array", "a U array: a view, a cast to another width, an order (np.asarray)", f =>
            {
                using var u = np.array(new[] { "ab", "c" });
                using var view = np.@char.array(u, copy: false);
                using var wide = np.@char.array(u, 5);
                return np.@char.array(u, order: 'F');
            });
            E(l, "np.char.array", "numbers through tolist(), itemsize 0 (unsized), StringDType, a Python scalar", f =>
            {
                using var d = np.array(new[] { 0.1, 2.5 });
                using var text = np.@char.array(d);
                using var unsized = np.@char.array(d, 0);
                using var t = np.array(new[] { "ab", "c" }, np.dtypes.StringDType());
                using var fromT = np.@char.array(t);
                return np.@char.array(5L, unicode: true);
            });
            E(l, "np.char.asarray", "an S chararray (itself) and a U one (cast)", f =>
            {
                using var s = np.@char.array(new object[] { new byte[] { 97, 98 }, new byte[] { 99 } });
                bool same = ReferenceEquals(np.@char.asarray(s), s);
                using var u = np.@char.array(new[] { "ab", "c" });
                return same ? np.@char.asarray(u) : null;
            });

            // ---- NDCharArray: element access and iteration ----
            E(l, "NDCharArray.Item", "get: the stripped stand-in, a view, an index array, a mask, typed index arrays", f =>
            {
                using var c = np.@char.array(new[] { "ab ", "c\t", "d" });
                using var mask = np.array(new[] { true, false, true });
                using var typedMask = mask.MakeGeneric<bool>();
                using var idx = np.array(new[] { 2, 0 });
                using var typedIdx = idx.MakeGeneric<int>();
                using var picks = np.array(new long[] { 2, 0 });
                return new object[]
                {
                    c[0], c["1:"], c[new object[] { picks }], c[typedMask], c[typedIdx], c[new object[] { true }],
                    c[Slice.Index(1)],
                };
            });
            E(l, "NDCharArray.Item", "set: an element and a slice of an owned chararray", f =>
            {
                // The values are held, not written as literals: an implicit string -> NDCharArray conversion at the call
                // site is the CALLER's temporary, which no library fix can dispose (the gate would count it).
                var c = np.@char.array(new[] { "ab", "c", "d" });
                using NDCharArray x = "x";
                using NDCharArray yz = "yz";
                c[0] = x;
                c["1:"] = yz;
                return c;
            });
            E(l, "NDCharArray.__getitem__", "an int, a long, a slice string, int[] coordinates", f =>
            {
                using var c = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                using var row = c.__getitem__(0);
                using var col = c.__getitem__("::-1");
                using var element = c.__getitem__(1, 0);
                return c.__getitem__(1L);
            });
            E(l, "NDCharArray.GetEnumerator", "stripped elements of a 1-d chararray, rows of a 2-d one", f =>
            {
                using var c = np.@char.array(new[] { "ab ", "c" });
                long chars = 0;
                foreach (var item in c)
                    chars += ((string)item).Length;
                using var m = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                foreach (var row in m)
                {
                    chars += ((NDArray)row).size;
                    ((NDArray)row).Dispose();
                }

                return Box(chars);
            });
            E(l, "NDCharArray.__iter__", "rows of a 2-d chararray", f =>
            {
                using var m = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                var it = m.__iter__();
                long rows = 0;
                while (it.MoveNext())
                {
                    ((NDArray)it.Current).Dispose();
                    rows++;
                }

                return Box(rows);
            });

            // ---- NDCharArray: conversions and copies ----
            E(l, "NDCharArray.astype", "to another width, digits to int8, a non-string target's finalize ValueError", f =>
            {
                using var c = np.@char.array(new[] { "1", "-5" });
                using var wide = c.astype(np.dtype("U5"));
                try
                {
                    c.astype(np.float64);
                }
                catch (ValueError)
                {
                }
                return c.astype(np.int8, true, 'C', "unsafe");
            });
            E(l, "NDCharArray.copy", "C and F", f =>
            {
                using var c = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                using var cc = c.copy();
                return c.copy('F');
            });
            E(l, "NDCharArray.view", "the same dtype, uint32 code points, the plain class, the chararray class", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                using var same = c.view();
                using var codes = c.view(np.uint32);
                using var plain = c.view(typeof(NDArray));
                return c.view((DType)null, typeof(NDCharArray));
            });
            E(l, "NDCharArray.byteswap", "a U chararray", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                return c.byteswap();
            });
            E(l, "NDCharArray.getfield", "the second character of each element", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                return c.getfield(np.dtype("U1"), 4);
            });
            E(l, "NDCharArray.to_device", "cpu returns the chararray itself", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                return Box(ReferenceEquals(c.to_device("cpu"), c));
            });
            E(l, "NDCharArray.Clone", "an owning copy", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                return c.Clone();
            });

            // ---- NDCharArray: shape and axes ----
            E(l, "NDCharArray.T", "a 2-d chararray", f =>
            {
                using var c = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                return c.T;
            });
            E(l, "NDCharArray.mT", "a 2-d chararray", f =>
            {
                using var c = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                return c.mT;
            });
            E(l, "NDCharArray.transpose", "reversed and given axes", f =>
            {
                using var c = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                using var t = c.transpose();
                return c.transpose(new[] { 1, 0 });
            });
            E(l, "NDCharArray.swapaxes", "0 and -1", f =>
            {
                using var c = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                return c.swapaxes(0, -1);
            });
            E(l, "NDCharArray.reshape", "every overload: long[], int[], Shape, Shape + order, ref Shape, a span", f =>
            {
                using var c = np.@char.array(new[] { "a", "b", "c", "d" });
                using var a = c.reshape(2, 2);
                using var b = c.reshape(new[] { 4 });
                using var d = c.reshape(new Shape(4, 1));
                using var e = c.reshape(new Shape(2, 2), 'F');
                var shape = new Shape(1, 4);
                using var g = c.reshape(ref shape);
                return c.reshape((ReadOnlySpan<long>)new long[] { 2, 2 });
            });
            E(l, "NDCharArray.ravel", "a view of a contiguous chararray, a copy in F order", f =>
            {
                using var c = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                using var v = c.ravel();
                return c.ravel('F');
            });
            E(l, "NDCharArray.flatten", "C and F", f =>
            {
                using var c = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                using var flat = c.flatten();
                return c.flatten('F');
            });
            E(l, "NDCharArray.squeeze", "itself without a length-1 axis, a view with one", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                bool same = ReferenceEquals(c.squeeze(), c);
                using var col = c.reshape(2, 1);
                return same ? col.squeeze() : null;
            });
            E(l, "NDCharArray.diagonal", "a 2-d chararray", f =>
            {
                using var c = np.@char.array(new[,] { { "a ", "b" }, { "c", "d" } });
                return c.diagonal();
            });

            // ---- NDCharArray: selection ----
            E(l, "NDCharArray.take", "an index array, one index (the unstripped stand-in)", f =>
            {
                using var c = np.@char.array(new[] { "ab ", "c" });
                using var one = c.take(0L);
                using var picks = np.array(new long[] { 1, 0 });
                return c.take(picks);
            });
            E(l, "NDCharArray.compress", "a bool condition", f =>
            {
                using var c = np.@char.array(new[] { "ab ", "c" });
                using var cond = np.array(new[] { true, false });
                return c.compress(cond);
            });
            E(l, "NDCharArray.repeat", "an int, a long, a count array along an axis", f =>
            {
                using var c = np.@char.array(new[] { "ab ", "c" });
                using var a = c.repeat(2);
                using var b = c.repeat(2L);
                using var counts = np.array(new long[] { 1, 2 });
                return c.repeat(counts, 0);
            });

            // ---- NDCharArray: complex parts and conversions ----
            E(l, "NDCharArray.real", "get: the chararray itself; set: stores into it", f =>
            {
                var c = np.@char.array(new[] { "ab", "c" });
                bool same = ReferenceEquals(c.real, c);
                using var value = np.@char.array(new[] { "x", "y" });
                c.real = value;
                return same ? c : null;
            });
            E(l, "NDCharArray.imag", "a read-only chararray of empty strings", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                return c.imag;
            });
            E(l, "NDCharArray.op_Implicit", "a C# string and a byte[] as 0-d stand-ins", f =>
            {
                NDCharArray text = "ab ";
                NDCharArray bytes = new byte[] { 97, 32 };
                text.Dispose();
                return bytes;
            });

            // ---- NDCharArray: the phase-3 operators and members (section strcharops replays them outside the sweep) ----
            // Each comparison entry takes the stripping route, the reflected route of a plain array on the left, a C# string,
            // Python's fallback for an S / U mix of two chararrays and the C# null literal; the errors are measured too.
            AddCharArrayComparison(l, "NDCharArray.op_Equality", (a, b) => a == b, (a, b) => a == b, a => a == "ab",
                                   (a, b) => a == b, a => a == (object)null, ordering: false);
            AddCharArrayComparison(l, "NDCharArray.op_Inequality", (a, b) => a != b, (a, b) => a != b, a => a != "ab",
                                   (a, b) => a != b, a => a != (object)null, ordering: false);
            AddCharArrayComparison(l, "NDCharArray.op_LessThan", (a, b) => a < b, (a, b) => a < b, a => a < "ab",
                                   (a, b) => a < b, a => a < (object)null, ordering: true);
            AddCharArrayComparison(l, "NDCharArray.op_LessThanOrEqual", (a, b) => a <= b, (a, b) => a <= b, a => a <= "ab",
                                   (a, b) => a <= b, a => a <= (object)null, ordering: true);
            AddCharArrayComparison(l, "NDCharArray.op_GreaterThan", (a, b) => a > b, (a, b) => a > b, a => a > "ab",
                                   (a, b) => a > b, a => a > (object)null, ordering: true);
            AddCharArrayComparison(l, "NDCharArray.op_GreaterThanOrEqual", (a, b) => a >= b, (a, b) => a >= b, a => a >= "ab",
                                   (a, b) => a >= b, a => a >= (object)null, ordering: true);
            E(l, "NDCharArray.op_Addition", "chararray + str / chararray / plain, str + chararray, the None and number errors", f =>
            {
                using var c = np.@char.array(new[] { "ab  ", "b" });
                using var u = np.array(new[] { "x", "yz" });
                using var withStr = c + "x";
                using var reflected = "x" + c;
                using var both = c + c;
                using var plain = u + c;
                try
                {
                    _ = c + (object)null;   // a short chararray: the trivial loop's copy into object, the finalize error
                }
                catch (ValueError)
                {
                }
                try
                {
                    _ = c + (object)5L;   // no add loop for (<U4, int64): the ufunc's TypeError
                }
                catch (TypeError)
                {
                }
                return c + u;
            });
            E(l, "NDCharArray.op_Addition", "a StringDType operand: short (the finalize error), 2-d (the refusal)", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                using var t = np.array(new[] { "q", "r" }, np.dtypes.StringDType());
                try
                {
                    _ = c + t;
                }
                catch (ValueError)
                {
                }
                // The 1-d bases are held: each 2-d view counts a reference on its base's buffer.
                using var flatChars = np.@char.array(new[] { "ab", "c", "d", "e" });
                using var flatText = np.array(new[] { "q", "r", "s", "t" }, np.dtypes.StringDType());
                using var c2 = flatChars.reshape(2, 2);
                using var t2 = flatText.reshape(2, 2);
                try
                {
                    _ = c2 + t2;   // NumPy's plain StringDType result, which the operator's NDCharArray type cannot hold
                }
                catch (NotSupportedException)
                {
                }
                return null;
            });
            E(l, "NDCharArray.op_Multiply", "an int, an int array, int * chararray, the non-integer and None errors", f =>
            {
                using var c = np.@char.array(new[] { "ab ", "c" });
                using var counts = np.array(new[] { 2L, 0L });
                using var byArray = c * counts;
                using var reflected = 3 * c;
                try
                {
                    _ = c * 2.5;
                }
                catch (ValueError)
                {
                }
                try
                {
                    _ = c * (object)null;
                }
                catch (ValueError)
                {
                }
                return c * 2;
            });
            E(l, "NDCharArray.Equals", "against a chararray holding the same text, against another object", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                using var same = np.@char.array(new[] { "ab", "c" });
                bool equal = c.Equals((object)same);
                return Box(equal && !c.Equals("ab"));
            });
            T(l, "NDCharArray.GetHashCode", "raises by design (an array is mutable, so unhashable, as in NumPy)", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                return Box(c.GetHashCode());
            });
            E(l, "NDCharArray.argmax", "the finalize ValueError without out=, the indices into out=", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c", "b" });
                try
                {
                    c.argmax();
                }
                catch (ValueError)
                {
                }
                try
                {
                    c.argmax(0, keepdims: true);
                }
                catch (ValueError)
                {
                }
                var @out = np.zeros(Shape.Scalar, np.int64);
                return c.argmax(null, @out);
            });
            E(l, "NDCharArray.argmin", "the finalize ValueError without out=, the indices into out=", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c", "b" });
                try
                {
                    c.argmin();
                }
                catch (ValueError)
                {
                }
                var @out = np.zeros(Shape.Scalar, np.int64);
                return c.argmin(null, @out);
            });
            T(l, "NDCharArray.argpartition", "raises by design (NumPy creates the int64 index array as the subclass: the finalize ValueError)", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c", "b" });
                return c.argpartition(0);
            });
            T(l, "NDCharArray.partition", "raises by design (the sort partition: chararray.partition takes only a separator, NumPy's TypeError)", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c", "b" });
                c.partition(0, -1);
                return null;
            });
            T(l, "NDCharArray.partition", "the string partition (np.char.partition): the next phase of np.char", f =>
            {
                using var c = np.@char.array(new[] { "a b", "c" });
                return c.partition(" ");
            });
        }

        /// <summary>
        ///     The entries of one comparison operator: the stripping route (a chararray on each side), the reflected route (a
        ///     plain array on the left), a C# string, Python's fallback for an S / U mix of two chararrays (a 0-d bool for
        ///     <c>==</c> / <c>!=</c>, NumPy's TypeError for an ordering), and the C# <c>null</c> literal (a reference test for
        ///     <c>==</c> / <c>!=</c>, the TypeError of an ordering against None).
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <param name="api">The operator's inventory id.</param>
        /// <param name="chars">The operator over two chararrays.</param>
        /// <param name="reflected">The operator with a plain array on the left.</param>
        /// <param name="str">The operator against a C# string.</param>
        /// <param name="mixed">The operator over a <c>U</c> and an <c>S</c> chararray.</param>
        /// <param name="none">The operator against the C# null literal.</param>
        /// <param name="ordering">True for <c>&lt;</c>, <c>&lt;=</c>, <c>&gt;</c>, <c>&gt;=</c>: the mix and null routes raise.</param>
        private static void AddCharArrayComparison(List<LeakCase> l, string api, Func<NDCharArray, NDCharArray, NDArray> chars,
            Func<NDArray, NDCharArray, NDArray> reflected, Func<NDCharArray, NDArray> str,
            Func<NDCharArray, NDCharArray, NDArray> mixed, Func<NDCharArray, NDArray> none, bool ordering)
        {
            E(l, api, "two chararrays, a plain array on the left, a str, an S / U mix, null", f =>
            {
                using var c = np.@char.array(new[] { "ab  ", "b" });
                using var u = np.array(new[] { "ab", "b " });
                using var s = np.@char.array(new object[] { new byte[] { 97, 98 }, new byte[] { 98 } });
                using var left = reflected(u, c);
                using var text = str(c);
                if (ordering)
                {
                    try
                    {
                        mixed(c, s);
                    }
                    catch (TypeError)
                    {
                    }
                    try
                    {
                        none(c);
                    }
                    catch (TypeError)
                    {
                    }
                }
                else
                {
                    using var fallback = mixed(c, s);
                    using var reference = none(c);
                }
                return chars(c, c);
            });
        }
    }
}
