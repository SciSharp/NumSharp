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
        }
    }
}
