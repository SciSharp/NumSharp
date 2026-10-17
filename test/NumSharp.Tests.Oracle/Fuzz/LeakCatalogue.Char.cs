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
            E(l, "NDCharArray.partition", "the string partition (np.strings.partition, stacked on a new last axis): U, S, a str and an array separator", f =>
            {
                using var c = np.@char.array(new[] { "a b", "c" });
                using var s = np.@char.array(new object[] { new byte[] { 97, 44, 98 }, new byte[] { 99 } });
                using var bySep = s.partition(new byte[] { 44 });
                using var seps = np.array(new[] { " ", "c" });
                using var byArray = c.partition(seps);
                return c.partition(" ");
            });
            // ---- numpy.char's string functions and the chararray members over them (C4 / C5: section strcharmeth replays
            // them outside the sweep) ----
            // Each entry takes the U, S and StringDType routes, a chararray operand (the parts and results come back as
            // chararray views: their wrappers must be reclaimed too), an argument array, a C# value and a refusal.
            E(l, "np.char.partition", "U / S / StringDType, a chararray input, a separator array, the empty-separator error", f =>
            {
                using var u = np.array(new[] { "a b", "c" });
                using var s = np.array(new object[] { new byte[] { 97, 45, 98 }, new byte[] { 99 } });
                using var t = np.array(new[] { "x y", "z" }, np.dtypes.StringDType());
                using var tsep = np.array(new[] { " " }, np.dtypes.StringDType());
                using var c = np.@char.array(new[] { "p q", "r" });
                using var seps = np.array(new[] { " ", "c" });
                using var bytes = np.@char.partition(s, new byte[] { 45 });
                using var vstring = np.@char.partition(t, tsep);
                using var chars = np.@char.partition(c, " ");
                using var bySeps = np.@char.partition(u, seps);
                try
                {
                    np.@char.partition(u, "");
                }
                catch (ValueError)
                {
                }
                return np.@char.partition(u, " ");
            });
            E(l, "np.char.rpartition", "U / StringDType, a chararray input, the no-loop error of a str separator beside StringDType", f =>
            {
                using var u = np.array(new[] { "a b c", "d" });
                using var t = np.array(new[] { "x y z" }, np.dtypes.StringDType());
                using var tsep = np.array(new[] { " " }, np.dtypes.StringDType());
                using var c = np.@char.array(new[] { "p q r" });
                using var vstring = np.@char.rpartition(t, tsep);
                using var chars = np.@char.rpartition(c, " ");
                try
                {
                    np.@char.rpartition(t, " ");
                }
                catch (TypeError)
                {
                }
                return np.@char.rpartition(u, " ");
            });
            AddCharPad(l, "np.char.center", (a, w, fill) => np.@char.center(a, w, fill));
            AddCharPad(l, "np.char.ljust", (a, w, fill) => np.@char.ljust(a, w, fill));
            AddCharPad(l, "np.char.rjust", (a, w, fill) => np.@char.rjust(a, w, fill));
            E(l, "np.char.zfill", "U / S / StringDType, a chararray input, a width array, a float width (TypeError)", f =>
            {
                using var u = np.array(new[] { "-7", "12" });
                using var s = np.array(new object[] { new byte[] { 43, 51 } });
                using var t = np.array(new[] { "-12", "x" }, np.dtypes.StringDType());
                using var c = np.@char.array(new[] { "5", "-" });
                using var widths = np.array(new long[] { 3, 5 });
                using var bytes = np.@char.zfill(s, 4);
                using var vstring = np.@char.zfill(t, 5);
                using var chars = np.@char.zfill(c, 4);
                using var byWidths = np.@char.zfill(u, widths);
                try
                {
                    np.@char.zfill(u, 2.5);
                }
                catch (TypeError)
                {
                }
                return np.@char.zfill(u, 6);
            });
            E(l, "np.char.expandtabs", "U / S / StringDType, a chararray input, a tab-size array, a float tab size (the cast error)", f =>
            {
                using var u = np.array(new[] { "a\tb", "\tc" });
                using var s = np.array(new object[] { new byte[] { 97, 9, 98 } });
                using var t = np.array(new[] { "x\ty" }, np.dtypes.StringDType());
                using var c = np.@char.array(new[] { "p\tq" });
                using var tabs = np.array(new long[] { 2, 4 });
                using var bytes = np.@char.expandtabs(s);
                using var vstring = np.@char.expandtabs(t, 3);
                using var chars = np.@char.expandtabs(c, 2);
                using var byTabs = np.@char.expandtabs(u, tabs);
                try
                {
                    np.@char.expandtabs(u, 2.0);
                }
                catch (ArgumentException)
                {
                }
                return np.@char.expandtabs(u);
            });
            E(l, "NDCharArray.rpartition", "U (a copy) and S (a view of the stacked parts), a separator array", f =>
            {
                using var c = np.@char.array(new[] { "a b c", "d" });
                using var s = np.@char.array(new object[] { new byte[] { 97, 44, 98 }, new byte[] { 99 } });
                using var bySep = s.rpartition(new byte[] { 44 });
                using var seps = np.array(new[] { " ", "d" });
                using var byArray = c.rpartition(seps);
                return c.rpartition(" ");
            });
            AddCharArrayPad(l, "NDCharArray.center", (c, w, fill) => c.center(w, fill));
            AddCharArrayPad(l, "NDCharArray.ljust", (c, w, fill) => c.ljust(w, fill));
            AddCharArrayPad(l, "NDCharArray.rjust", (c, w, fill) => c.rjust(w, fill));
            E(l, "NDCharArray.zfill", "U (a copy) and S (a view), a width array, a text width (TypeError)", f =>
            {
                using var c = np.@char.array(new[] { "-7", "+3" });
                using var s = np.@char.array(new object[] { new byte[] { 45, 49 } });
                using var widths = np.array(new long[] { 4, 2 });
                using var bytes = s.zfill(5);
                using var byWidths = c.zfill(widths);
                try
                {
                    c.zfill("3");
                }
                catch (TypeError)
                {
                }
                return c.zfill(6);
            });
            E(l, "NDCharArray.expandtabs", "U (a copy) and S (a view), the default and a tab size", f =>
            {
                using var c = np.@char.array(new[] { "a\tb", "c" });
                using var s = np.@char.array(new object[] { new byte[] { 9, 97 } });
                using var bytes = s.expandtabs();
                using var bySize = c.expandtabs(4);
                return c.expandtabs();
            });
            // The length and search functions (numpy.strings' own) and the chararray search members: plain int64 / bool
            // results, so a chararray operand leaves no chararray view behind.
            E(l, "np.char.str_len", "U / S / StringDType, a chararray input, out= with where=, a number (TypeError)", f =>
            {
                using var u = np.array(new[] { "ab", "cde" });
                using var s = np.array(new object[] { new byte[] { 97, 98 } });
                using var t = np.array(new[] { "xy", "z" }, np.dtypes.StringDType());
                using var c = np.@char.array(new[] { "p  ", "qr" });
                using var o = np.zeros(new Shape(2), np.int32);
                using var mask = np.array(new[] { true, false });
                using var bytes = np.@char.str_len(s);
                using var vstring = np.@char.str_len(t);
                using var chars = np.@char.str_len(c);
                np.@char.str_len(u, o, mask);
                try
                {
                    np.@char.str_len(5);
                }
                catch (TypeError)
                {
                }
                return np.@char.str_len(u);
            });
            AddCharSearch(l, "np.char.count", (a, sub, start, end) => np.@char.count(a, sub, start, end));
            AddCharSearch(l, "np.char.find", (a, sub, start, end) => np.@char.find(a, sub, start, end));
            AddCharSearch(l, "np.char.rfind", (a, sub, start, end) => np.@char.rfind(a, sub, start, end));
            AddCharSearch(l, "np.char.index", (a, sub, start, end) => np.@char.index(a, sub, start, end));
            AddCharSearch(l, "np.char.rindex", (a, sub, start, end) => np.@char.rindex(a, sub, start, end));
            AddCharSearch(l, "np.char.startswith", (a, prefix, start, end) => np.@char.startswith(a, prefix, start, end));
            AddCharSearch(l, "np.char.endswith", (a, suffix, start, end) => np.@char.endswith(a, suffix, start, end));
            AddCharArraySearch(l, "NDCharArray.count", (c, sub, start, end) => c.count(sub, start, end));
            AddCharArraySearch(l, "NDCharArray.find", (c, sub, start, end) => c.find(sub, start, end));
            AddCharArraySearch(l, "NDCharArray.rfind", (c, sub, start, end) => c.rfind(sub, start, end));
            AddCharArraySearch(l, "NDCharArray.index", (c, sub, start, end) => c.index(sub, start, end));
            AddCharArraySearch(l, "NDCharArray.rindex", (c, sub, start, end) => c.rindex(sub, start, end));
            AddCharArraySearch(l, "NDCharArray.startswith", (c, prefix, start, end) => c.startswith(prefix, start, end));
            AddCharArraySearch(l, "NDCharArray.endswith", (c, suffix, start, end) => c.endswith(suffix, start, end));
            // The strips and replace (numpy.strings' own) and their chararray members (rule 2: the function's chararray view;
            // a plain StringDType result is refused by the member, which releases it).
            AddCharStrip(l, "np.char.strip", (a, chars) => np.@char.strip(a, chars));
            AddCharStrip(l, "np.char.lstrip", (a, chars) => np.@char.lstrip(a, chars));
            AddCharStrip(l, "np.char.rstrip", (a, chars) => np.@char.rstrip(a, chars));
            AddCharArrayStrip(l, "NDCharArray.strip", (c, chars) => c.strip(chars));
            AddCharArrayStrip(l, "NDCharArray.lstrip", (c, chars) => c.lstrip(chars));
            AddCharArrayStrip(l, "NDCharArray.rstrip", (c, chars) => c.rstrip(chars));
            E(l, "np.char.replace", "U / S / StringDType, a chararray input (a view) and a chararray old beside a plain text (plain), old and count arrays, a Python bytes old (the cast); a StringDType old beside a short (ValueError) and a 2-d chararray (a plain StringDType result); a float count (TypeError)", f =>
            {
                using var u = np.array(new[] { "a b", "bab" });
                using var s = np.array(new object[] { new byte[] { 97, 98 } });
                using var t = np.array(new[] { "xy", "yx" }, np.dtypes.StringDType());
                using var c = np.@char.array(new[] { "ab ", "b a" });
                using var flat = np.@char.array(new[] { "ab", "ba", "a ", " b" });
                using var c2 = flat.reshape(2, 2);
                using var olds = np.array(new[] { "a", "b" });
                using var counts = np.array(new long[] { 1, -1 });
                using var told = np.array(new[] { "a" }, np.dtypes.StringDType());
                using var cold = np.@char.array(new[] { "a" });
                using var bytes = np.@char.replace(s, new byte[] { 97 }, new byte[] { 65, 65 });
                using var vstring = np.@char.replace(t, "x", "__", 1);
                using var chars = np.@char.replace(c, " ", "_");
                using var charOld = np.@char.replace(u, cold, "x");
                using var byOlds = np.@char.replace(u, olds, "Z", counts);
                using var bytesOld = np.@char.replace(u, new byte[] { 98 }, "q");
                using var plainT = np.@char.replace(c2, told, "q");
                try
                {
                    np.@char.replace(c, told, "q");
                }
                catch (ValueError)
                {
                }
                try
                {
                    np.@char.replace(u, "a", "b", 1.5);
                }
                catch (TypeError)
                {
                }
                return np.@char.replace(u, " ", "");
            });
            E(l, "NDCharArray.replace", "U and S receivers, a count, old and count arrays, an element stand-in and Python bytes as old (the cast); a StringDType old beside a short (ValueError) and a 2-d receiver (NotSupportedException, the plain result released); a float count (TypeError)", f =>
            {
                using var c = np.@char.array(new[] { "a b", "bab" });
                using var s = np.@char.array(new object[] { new byte[] { 97, 98 }, new byte[] { 98, 98 } });
                using var flat = np.@char.array(new[] { "ab", "ba", "a ", " b" });
                using var c2 = (NDCharArray)flat.reshape(2, 2);
                using var olds = np.array(new[] { "a", "b" });
                using var counts = np.array(new long[] { 1, -1 });
                using var told = np.array(new[] { "a" }, np.dtypes.StringDType());
                using var element = c[1];
                using var bytes = s.replace(new byte[] { 98 }, new byte[] { 65 }, 1);
                using var byOlds = c.replace(olds, "Z", counts);
                using var byElement = c.replace(element, "Q");
                using var bytesOld = c.replace(new byte[] { 98 }, "q");
                try
                {
                    c.replace(told, "q");
                }
                catch (ValueError)
                {
                }
                try
                {
                    c2.replace(told, "q");
                }
                catch (NotSupportedException)
                {
                }
                try
                {
                    c.replace("a", "b", 1.5);
                }
                catch (TypeError)
                {
                }
                return c.replace(" ", "");
            });
            // The nine predicates (numpy.strings' ufuncs, re-exported) and their chararray members: plain bool results, so a
            // chararray operand leaves no chararray view behind; a chararray out= gives a new view of it.
            AddCharPredicate(l, "np.char.isalnum", (x, o, w, d) => np.@char.isalnum(x, o, w, d), hasBytesLoop: true);
            AddCharPredicate(l, "np.char.isalpha", (x, o, w, d) => np.@char.isalpha(x, o, w, d), hasBytesLoop: true);
            AddCharPredicate(l, "np.char.isdecimal", (x, o, w, d) => np.@char.isdecimal(x, o, w, d), hasBytesLoop: false);
            AddCharPredicate(l, "np.char.isdigit", (x, o, w, d) => np.@char.isdigit(x, o, w, d), hasBytesLoop: true);
            AddCharPredicate(l, "np.char.islower", (x, o, w, d) => np.@char.islower(x, o, w, d), hasBytesLoop: true);
            AddCharPredicate(l, "np.char.isnumeric", (x, o, w, d) => np.@char.isnumeric(x, o, w, d), hasBytesLoop: false);
            AddCharPredicate(l, "np.char.isspace", (x, o, w, d) => np.@char.isspace(x, o, w, d), hasBytesLoop: true);
            AddCharPredicate(l, "np.char.istitle", (x, o, w, d) => np.@char.istitle(x, o, w, d), hasBytesLoop: true);
            AddCharPredicate(l, "np.char.isupper", (x, o, w, d) => np.@char.isupper(x, o, w, d), hasBytesLoop: true);
            AddCharArrayPredicate(l, "NDCharArray.isalnum", c => c.isalnum(), hasBytesLoop: true);
            AddCharArrayPredicate(l, "NDCharArray.isalpha", c => c.isalpha(), hasBytesLoop: true);
            AddCharArrayPredicate(l, "NDCharArray.isdecimal", c => c.isdecimal(), hasBytesLoop: false);
            AddCharArrayPredicate(l, "NDCharArray.isdigit", c => c.isdigit(), hasBytesLoop: true);
            AddCharArrayPredicate(l, "NDCharArray.islower", c => c.islower(), hasBytesLoop: true);
            AddCharArrayPredicate(l, "NDCharArray.isnumeric", c => c.isnumeric(), hasBytesLoop: false);
            AddCharArrayPredicate(l, "NDCharArray.isspace", c => c.isspace(), hasBytesLoop: true);
            AddCharArrayPredicate(l, "NDCharArray.istitle", c => c.istitle(), hasBytesLoop: true);
            AddCharArrayPredicate(l, "NDCharArray.isupper", c => c.isupper(), hasBytesLoop: true);
            // ---- numpy.char's object path (phase 4: section strcharobj replays it outside the sweep) ----
            // Each entry takes the S, U and StringDType text routes, a separator / limit / keepends operand (an array, a
            // Python scalar, an object list holding None), the element errors (whose piece store and readers must be
            // released), and the results' own arrays (an NDListArray owns four or five, the harness disposes it).
            E(l, "np.char.split", "S / U / StringDType, whitespace and a separator, an object-list separator, an error", f =>
            {
                using var u = np.array(new[] { "a b  c", " d " });
                using var s = np.array(new object[] { new byte[] { 97, 32, 98 }, new byte[] { 99 } });
                using var t = np.array(new[] { "a b", "c,d" }, np.dtypes.StringDType());
                using var byWords = np.@char.split(u);
                using var bytes = np.@char.split(s, new byte[] { 32 });
                using var vstring = np.@char.split(t, ",", 1);
                using var objects = np.@char.split(u, new object[] { " ", null });
                try
                {
                    np.@char.split(u, " ", 1.5);
                }
                catch (TypeError)
                {
                }
                return np.@char.split(u, "  ");
            });
            E(l, "np.char.rsplit", "U with a limit, StringDType, the empty-separator error", f =>
            {
                using var u = np.array(new[] { "a,b,,c", "d" });
                using var t = np.array(new[] { "a b c" }, np.dtypes.StringDType());
                using var vstring = np.@char.rsplit(t, null, 1);
                try
                {
                    np.@char.rsplit(u, "");
                }
                catch (ValueError)
                {
                }
                return np.@char.rsplit(u, ",", 2);
            });
            E(l, "np.char.splitlines", "S / U / StringDType, keepends as a value and an array", f =>
            {
                using var u = np.array(new[] { "a\nb\r\nc", "d" });
                using var s = np.array(new object[] { new byte[] { 97, 10, 98 } });
                using var t = np.array(new[] { "x\ny" }, np.dtypes.StringDType());
                using var keep = np.array(new[] { true, false });
                using var bytes = np.@char.splitlines(s);
                using var vstring = np.@char.splitlines(t, true);
                using var kept = np.@char.splitlines(u, keep);
                return np.@char.splitlines(u);
            });
            E(l, "np.char.join", "U / S / StringDType results, a separator array, the iterable error, over a list array", f =>
            {
                using var u = np.array(new[] { "ab", "cde" });
                using var seps = np.array(new[] { "-", "" });
                using var t = np.array(new[] { "ab" }, np.dtypes.StringDType());
                using var empty = np.array(new object[] { new byte[0] });
                using var text = np.@char.join(seps, u);
                using var vstring = np.@char.join("-", t);
                using var bytes = np.@char.join(new byte[] { 45 }, empty);
                try
                {
                    np.@char.join("-", 5);
                }
                catch (TypeError)
                {
                }
                using var words = np.@char.split(u);
                try
                {
                    np.@char.join(" ", words);   // NumPy 2.4.2's ObjectDType(n) TypeError, after every list joined
                }
                catch (TypeError)
                {
                }
                using var nothing = np.zeros(new Shape(0), np.dtype("U2"));
                using var none = np.@char.split(nothing);
                using var emptyLists = np.@char.join("-", none);
                return np.@char.join("-", u);
            });
            E(l, "NDCharArray.split", "unstripped elements, a separator and a limit", f =>
            {
                using var c = np.@char.array(new[] { "a b ", "c  d" });
                using var words = c.split();
                return c.split(" ", 1);
            });
            E(l, "NDCharArray.rsplit", "a limit from the right", f =>
            {
                using var c = np.@char.array(new[] { "a b c" });
                return c.rsplit(null, 1);
            });
            E(l, "NDCharArray.splitlines", "keepends", f =>
            {
                using var c = np.@char.array(new[] { "a\nb", "c" });
                using var lines = c.splitlines();
                return c.splitlines(true);
            });
            E(l, "NDCharArray.join", "the chararray as the separators, a plain result; over a list array", f =>
            {
                using var c = np.@char.array(new[] { "-", "+" });
                using var nothing = np.zeros(new Shape(0, 2), np.dtype("U2"));
                using var none = np.@char.split(nothing);
                using var emptyLists = c.join(none);
                return c.join(new[] { "xy", "z" });
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

        /// <summary>
        ///     The entry of one of numpy.char's justify functions (<c>center</c>, <c>ljust</c>, <c>rjust</c>, numpy.strings'
        ///     own): the U, S and StringDType routes, a chararray input (a chararray view of the result), a width array, a
        ///     fill character, and the two refusals (a two-character fill, a float width: TypeError).
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <param name="api">The function's inventory id.</param>
        /// <param name="pad">The function: text, width, fill (null for the default).</param>
        private static void AddCharPad(List<LeakCase> l, string api, Func<object, object, object, NDArray> pad)
        {
            E(l, api, "U / S / StringDType, a chararray input, a width array and a fill, a StringDType fill beside a 2-d chararray and a chararray fill beside U; a long fill and a float width (TypeError), the trivial-loop copy of a short chararray (ValueError)", f =>
            {
                using var u = np.array(new[] { "ab", "cde" });
                using var s = np.array(new object[] { new byte[] { 97, 98 } });
                using var t = np.array(new[] { "xy", "z" }, np.dtypes.StringDType());
                using var c = np.@char.array(new[] { "p", "qr" });
                using var widths = np.array(new long[] { 4, 6 });
                using var flat = np.@char.array(new[] { "ab", "c", "d", "ef" });
                using var c2 = flat.reshape(2, 2);
                using var tfill = np.array(new[] { "*" }, np.dtypes.StringDType());
                using var cfill = np.@char.array(new[] { "-" });
                using var bytes = pad(s, 5, new byte[] { 42 });
                using var vstring = pad(t, 6, "*");
                using var chars = pad(c, 5, null);
                using var byWidths = pad(u, widths, "-");
                // A StringDType fill beside a 2-d chararray: NumPy's trivial loop does not copy it, a plain StringDType
                // result; a chararray fill beside a plain U text computes plain (the marks of PlainUfunc).
                using var plainT = pad(c2, 5, tfill);
                using var charFill = pad(u, 6, cfill);
                try
                {
                    pad(u, 5, "ab");
                }
                catch (TypeError)
                {
                }
                try
                {
                    pad(c, 5, tfill);
                }
                catch (ValueError)
                {
                }
                try
                {
                    pad(t, 6, cfill);
                }
                catch (ValueError)
                {
                }
                try
                {
                    pad(u, 2.5, null);
                }
                catch (TypeError)
                {
                }
                return pad(u, 7, null);
            });
        }

        /// <summary>
        ///     The entry of one of chararray's justify members (<c>center</c>, <c>ljust</c>, <c>rjust</c>): U (an owning copy,
        ///     <c>np.char.asarray</c>'s rule) and S (a view of the computed array), a width array, a fill, and a refusal.
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <param name="api">The member's inventory id.</param>
        /// <param name="pad">The member: receiver, width, fill (null for the default).</param>
        private static void AddCharArrayPad(List<LeakCase> l, string api, Func<NDCharArray, object, object, NDCharArray> pad)
        {
            E(l, api, "U (a copy) and S (a view), a width array and a fill, a StringDType fill beside a 2-d receiver (an S view); a float width (TypeError), a short receiver's trivial-loop copy (ValueError)", f =>
            {
                using var c = np.@char.array(new[] { "ab", "c" });
                using var s = np.@char.array(new object[] { new byte[] { 97 }, new byte[] { 98, 99 } });
                using var widths = np.array(new long[] { 3, 5 });
                using var flat = np.@char.array(new[] { "ab", "c", "d", "ef" });
                using var c2 = (NDCharArray)flat.reshape(2, 2);
                using var tfill = np.array(new[] { "*" }, np.dtypes.StringDType());
                using var bytes = pad(s, 4, new byte[] { 42 });
                using var byWidths = pad(c, widths, "-");
                // A StringDType fill beside a 2-d receiver: the plain StringDType result through np.char.asarray (an S view).
                using var asBytes = pad(c2, 5, tfill);
                try
                {
                    pad(c, 2.5, null);
                }
                catch (TypeError)
                {
                }
                try
                {
                    pad(c, 5, tfill);
                }
                catch (ValueError)
                {
                }
                return pad(c, 6, null);
            });
        }

        /// <summary>
        ///     The entry of one of numpy.char's search functions (<c>count</c>, <c>find</c>, <c>rfind</c>, <c>index</c>,
        ///     <c>rindex</c>, <c>startswith</c>, <c>endswith</c>, numpy.strings' own): the U, S and StringDType routes, a
        ///     chararray input (a plain result), substring and start arrays, a range, and the two refusals (a bytes substring
        ///     beside U: TypeError; a float start: the cast's ArgumentException). Every element holds every substring searched
        ///     outside the refusals, so <c>index</c> / <c>rindex</c> raise only where meant to.
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <param name="api">The function's inventory id.</param>
        /// <param name="search">The function: text, substring, start, end (null for NumPy's defaults).</param>
        private static void AddCharSearch(List<LeakCase> l, string api, Func<object, object, object, object, NDArray> search)
        {
            E(l, api, "U / S / StringDType, a chararray input, substring and start arrays, a range, a StringDType substring beside a 2-d chararray; a bytes substring beside U (TypeError), a float start (the cast error), a short chararray's trivial-loop copy (ValueError)", f =>
            {
                using var u = np.array(new[] { "ab ab", "abba" });
                using var s = np.array(new object[] { new byte[] { 97, 98 } });
                using var t = np.array(new[] { "xy", "yz" }, np.dtypes.StringDType());
                using var c = np.@char.array(new[] { "b  ", "a b" });
                using var subs = np.array(new[] { "a", "b" });
                using var starts = np.array(new long[] { 1, 0 });
                using var flat = np.@char.array(new[] { "ab", "ba", "a ", " a" });
                using var c2 = flat.reshape(2, 2);
                using var tsub = np.array(new[] { "a" }, np.dtypes.StringDType());
                using var bytes = search(s, new byte[] { 98 }, null, null);
                using var vstring = search(t, "y", null, null);
                using var chars = search(c, " ", null, null);
                using var bySubs = search(u, subs, starts, null);
                using var ranged = search(u, "b", 1, -1);
                using var plain2d = search(c2, tsub, null, null);
                try
                {
                    search(c, tsub, null, null);
                }
                catch (ValueError)
                {
                }
                try
                {
                    search(u, new byte[] { 98 }, null, null);
                }
                catch (TypeError)
                {
                }
                try
                {
                    search(u, "b", 1.5, null);
                }
                catch (ArgumentException)
                {
                }
                return search(u, "b", null, null);
            });
        }

        /// <summary>
        ///     The entry of one of chararray's search members: U and S receivers (plain results), substring and end arrays, a
        ///     range, and a refusal (a str substring beside an S receiver: TypeError). Every element holds every substring
        ///     searched outside the refusal, so <c>index</c> / <c>rindex</c> do not raise.
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <param name="api">The member's inventory id.</param>
        /// <param name="search">The member: receiver, substring, start, end (null for NumPy's defaults).</param>
        private static void AddCharArraySearch(List<LeakCase> l, string api, Func<NDCharArray, object, object, object, NDArray> search)
        {
            E(l, api, "U and S receivers, substring and end arrays, a range; a str substring beside S (TypeError)", f =>
            {
                using var c = np.@char.array(new[] { "ab ab", "abba" });
                using var s = np.@char.array(new object[] { new byte[] { 97, 98 }, new byte[] { 98, 98 } });
                using var subs = np.array(new[] { "a", "b" });
                using var ends = np.array(new long[] { 3, 4 });
                using var bytes = search(s, new byte[] { 98 }, null, null);
                using var bySubs = search(c, subs, null, ends);
                using var ranged = search(c, "b", 1, -1);
                try
                {
                    search(s, "b", null, null);
                }
                catch (TypeError)
                {
                }
                return search(c, "b", null, null);
            });
        }

        /// <summary>
        ///     The entry of one of numpy.char's strip functions (<c>strip</c>, <c>lstrip</c>, <c>rstrip</c>, numpy.strings'
        ///     own): the U, S and StringDType routes, whitespace and a set, a chararray input (a chararray view of the
        ///     result), a chararray set beside a plain text (a view too), a set array, a StringDType set beside a 2-d
        ///     chararray (a plain StringDType result), and the refusals (a short chararray's trivial-loop copy: ValueError; a
        ///     bytes set beside U: TypeError).
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <param name="api">The function's inventory id.</param>
        /// <param name="strip">The function: text, set (null for whitespace).</param>
        private static void AddCharStrip(List<LeakCase> l, string api, Func<object, object, NDArray> strip)
        {
            E(l, api, "U / S / StringDType, whitespace and a set, a chararray input and set, a set array, a StringDType set beside a 2-d chararray; a short chararray's trivial-loop copy (ValueError) and a bytes set beside U (TypeError)", f =>
            {
                using var u = np.array(new[] { " ab ", "ba " });
                using var s = np.array(new object[] { new byte[] { 32, 97, 98 } });
                using var t = np.array(new[] { " xy", "yz " }, np.dtypes.StringDType());
                using var c = np.@char.array(new[] { "b  ", " a b" });
                using var flat = np.@char.array(new[] { "ab", "ba", "a ", " b" });
                using var c2 = flat.reshape(2, 2);
                using var sets = np.array(new[] { "a ", "b" });
                using var tset = np.array(new[] { "a" }, np.dtypes.StringDType());
                using var cset = np.@char.array(new[] { "a" });
                using var bytes = strip(s, null);
                using var bytesSet = strip(s, new byte[] { 97 });
                using var vstring = strip(t, "y");
                using var chars = strip(c, null);
                using var charSet = strip(u, cset);
                using var bySets = strip(u, sets);
                using var plainT = strip(c2, tset);
                try
                {
                    strip(c, tset);
                }
                catch (ValueError)
                {
                }
                try
                {
                    strip(u, new byte[] { 98 });
                }
                catch (TypeError)
                {
                }
                return strip(u, " a");
            });
        }

        /// <summary>
        ///     The entry of one of chararray's strip members (rule 2: the function's chararray view): U and S receivers,
        ///     whitespace and a set, a set array, and the refusals — a StringDType set beside a short receiver (the
        ///     trivial-loop copy: ValueError) and beside a 2-d one (NotSupportedException: the member releases NumPy's plain
        ///     StringDType result), a str set beside S (TypeError).
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <param name="api">The member's inventory id.</param>
        /// <param name="strip">The member: receiver, set (null for whitespace).</param>
        private static void AddCharArrayStrip(List<LeakCase> l, string api, Func<NDCharArray, object, NDCharArray> strip)
        {
            E(l, api, "U and S receivers, whitespace and a set, a set array; a StringDType set beside a short (ValueError) and a 2-d receiver (NotSupportedException); a str set beside S (TypeError)", f =>
            {
                using var c = np.@char.array(new[] { " ab ", "ba " });
                using var s = np.@char.array(new object[] { new byte[] { 32, 97 }, new byte[] { 98, 32 } });
                using var flat = np.@char.array(new[] { "ab", "ba", "a ", " b" });
                using var c2 = (NDCharArray)flat.reshape(2, 2);
                using var sets = np.array(new[] { "a ", "b" });
                using var tset = np.array(new[] { "a" }, np.dtypes.StringDType());
                using var bytes = strip(s, new byte[] { 32 });
                using var bySets = strip(c, sets);
                try
                {
                    strip(c, tset);
                }
                catch (ValueError)
                {
                }
                try
                {
                    strip(c2, tset);
                }
                catch (NotSupportedException)
                {
                }
                try
                {
                    strip(s, "a");
                }
                catch (TypeError)
                {
                }
                return strip(c, null);
            });
        }

        /// <summary>
        ///     The entry of one of numpy.char's predicates (numpy.strings' ufuncs, re-exported): the U, S and StringDType
        ///     routes, a chararray input (a plain bool result), Python values (a str, a list), out= with where=, a chararray
        ///     out (a new chararray view of it), and the refusals (a number and a non-bool dtype=: TypeError; for
        ///     <c>isdecimal</c> / <c>isnumeric</c> an S operand and Python bytes too, which have no loop).
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <param name="api">The function's inventory id.</param>
        /// <param name="predicate">The function: x, out, where, dtype (null for NumPy's defaults).</param>
        /// <param name="hasBytesLoop">Whether the ufunc has an S loop (all but <c>isdecimal</c> / <c>isnumeric</c>).</param>
        private static void AddCharPredicate(List<LeakCase> l, string api, Func<object, NDArray, NDArray, DType, NDArray> predicate,
                                             bool hasBytesLoop)
        {
            E(l, api, hasBytesLoop
                ? "U / S / StringDType, a chararray input, a Python str / bytes / list, out= with where=, a chararray out; a number and a non-bool dtype (TypeError)"
                : "U / StringDType, a chararray input, a Python str / list, out= with where=, a chararray out; an S operand, Python bytes, a number and a non-bool dtype (TypeError)", f =>
            {
                using var u = np.array(new[] { "ab", "A1 " });
                using var s = np.array(new object[] { new byte[] { 97, 98 } });
                using var t = np.array(new[] { "xy", "Z" }, np.dtypes.StringDType());
                using var c = np.@char.array(new[] { "p  ", "QR" });
                using var o = np.zeros(new Shape(2), np.int32);
                using var mask = np.array(new[] { true, false });
                using var text = np.@char.array(new[] { "zzzz", "zzzz" });
                using var vstring = predicate(t, null, null, null);
                using var chars = predicate(c, null, null, null);
                using var str = predicate("Ab", null, null, null);
                using var list = predicate(new[] { "a", "B" }, null, null, null);
                using var view = predicate(c, text, null, null);
                predicate(u, o, mask, null);
                if (hasBytesLoop)
                {
                    using var bytes = predicate(s, null, null, null);
                    using var pyBytes = predicate(new byte[] { 97 }, null, null, null);
                }
                else
                {
                    try
                    {
                        predicate(s, null, null, null);
                    }
                    catch (TypeError)
                    {
                    }
                    try
                    {
                        predicate(new byte[] { 97 }, null, null, null);
                    }
                    catch (TypeError)
                    {
                    }
                }
                try
                {
                    predicate(5, null, null, null);
                }
                catch (TypeError)
                {
                }
                try
                {
                    predicate(u, null, null, np.int64);
                }
                catch (TypeError)
                {
                }
                return predicate(u, null, null, null);
            });
        }

        /// <summary>
        ///     The entry of one of chararray's predicate members (<c>isalpha(self)</c>, a plain bool result): U and S
        ///     receivers and a 0-d receiver (a 0-d array); for <c>isdecimal</c> / <c>isnumeric</c> the S receiver is the
        ///     refusal (no bytes loop: TypeError).
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <param name="api">The member's inventory id.</param>
        /// <param name="predicate">The member: receiver to result.</param>
        /// <param name="hasBytesLoop">Whether the ufunc has an S loop (all but <c>isdecimal</c> / <c>isnumeric</c>).</param>
        private static void AddCharArrayPredicate(List<LeakCase> l, string api, Func<NDCharArray, NDArray> predicate, bool hasBytesLoop)
        {
            E(l, api, hasBytesLoop ? "U and S receivers, a 0-d receiver" : "a U receiver, a 0-d receiver; an S receiver (TypeError)", f =>
            {
                using var c = np.@char.array(new[] { "ab  ", "A1" });
                using var s = np.@char.array(new object[] { new byte[] { 97, 98 }, new byte[] { 49 } });
                using var one = (NDCharArray)c["0:1"];
                using var zero = (NDCharArray)one.reshape(Shape.Scalar);
                using var scalar = predicate(zero);
                if (hasBytesLoop)
                {
                    using var bytes = predicate(s);
                }
                else
                {
                    try
                    {
                        predicate(s);
                    }
                    catch (TypeError)
                    {
                    }
                }
                return predicate(c);
            });
        }
    }
}
