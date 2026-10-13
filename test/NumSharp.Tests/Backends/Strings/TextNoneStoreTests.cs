using System;
using System.Linq;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Python's <c>None</c> (a C# null) and other Python objects stored into text arrays through every store entry point:
    ///     a fixed-width <c>S</c> / <c>U</c> element stores <c>str(None)</c>, <c>'None'</c> truncated to its width
    ///     (<c>STRING_setitem</c> / <c>UNICODE_setitem</c> take any object by its <c>str()</c>); a NEP 55 <c>StringDType</c>
    ///     element stores its NA object or <c>'None'</c>; <c>__setitem__</c> stores what <c>a[i] = v</c> stores; the
    ///     object-typed <c>SetData</c> takes a text, a char, bytes or null on a <c>StringDType</c> array. Expected values were
    ///     probed against <c>numpy==2.4.2</c>.
    /// </summary>
    [TestClass]
    public class TextNoneStoreTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor.</summary>
        private static DType TN => np.dtypes.StringDType(na_object: null);

        /// <summary>The elements of a fresh C-contiguous array.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values in C order.</returns>
        private static object[] V(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => a.Storage.GetAtIndex(i)).ToArray();

        /// <summary>The texts of a fixed-width <c>S</c> array, decoded as ASCII.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The texts in C order.</returns>
        private static string[] B(NDArray a) => V(a).Select(v => System.Text.Encoding.ASCII.GetString((byte[])v)).ToArray();

        /// <summary>
        ///     <c>None</c> into a <c>U2</c> array through every index form — an integer, an ellipsis, a boolean mask, an index
        ///     array, a slice — stores <c>'No'</c> (NumPy: <c>array(['No', 'cd'], dtype='&lt;U2')</c>); a <c>U8</c> array keeps the
        ///     whole <c>'None'</c>.
        /// </summary>
        [TestMethod]
        public void None_IntoUnicode_EveryIndexForm()
        {
            using (var a = np.array(new[] { "ab", "cd" })) { a[0] = null; V(a).Should().Equal("No", "cd"); }
            using (var a = np.array(new[] { "ab", "cd" })) { a[Slice.Ellipsis] = null; V(a).Should().Equal("No", "No"); }
            using (var a = np.array(new[] { "ab", "cd" }))
            using (var mask = np.array(new[] { true, false }))
            {
                a[mask] = null;
                V(a).Should().Equal("No", "cd");
            }
            using (var a = np.array(new[] { "ab", "cd" }))
            using (var index = np.array(new[] { 1 }))
            {
                a[index] = null;
                V(a).Should().Equal("ab", "No");
            }
            using (var a = np.array(new[] { "ab", "cd" }, "U8")) { a["0:1"] = null; V(a).Should().Equal("None", "cd"); }
            using (var a = np.array(new[] { "ab", "cd" }, "U8")) { a[Slice.Ellipsis] = null; V(a).Should().Equal("None", "None"); }
        }

        /// <summary><c>None</c> into an <c>S2</c> array stores <c>b'No'</c> (NumPy: <c>array([b'No', b'cd'], dtype='|S2')</c>).</summary>
        [TestMethod]
        public void None_IntoBytes()
        {
            using (var a = np.array(new[] { "ab", "cd" }, "S2")) { a[0] = null; B(a).Should().Equal("No", "cd"); }
            using (var a = np.array(new[] { "ab", "cd" }, "S2")) { a[Slice.Ellipsis] = null; B(a).Should().Equal("No", "No"); }
            using (var a = np.array(new[] { "ab", "cd" }, "S2"))
            using (var mask = np.array(new[] { true, false }))
            {
                a[mask] = null;
                B(a).Should().Equal("No", "cd");
            }
        }

        /// <summary>
        ///     <c>__setitem__</c> stores what <c>a[i] = v</c> stores: null is <c>None</c> (<c>'No'</c> into <c>U2</c>, NA into a TN
        ///     array, through an integer and a slice string), a C# integer is a Python int (range-checked: 300 into int8 is
        ///     NumPy's OverflowError, it used to wrap to 44), a C# double into an integer array truncates, an int into a T
        ///     array stores <c>'5'</c>, a text into an int64 array is parsed.
        /// </summary>
        [TestMethod]
        public void SetItem_StoresLikeTheIndexer()
        {
            using (var a = np.array(new[] { "ab", "cd" })) { a.__setitem__(0, null); V(a).Should().Equal("No", "cd"); }
            using (var a = np.array(new[] { "ab", "cd" }, TN)) { a.__setitem__(0, null); V(a).Should().Equal(new object[] { null, "cd" }); }
            using (var a = np.array(new[] { "ab", "cd" }, TN)) { a.__setitem__("...", null); V(a).Should().Equal(new object[] { null, null }); }
            using (var a = np.array(new[] { "ab", "cd" }, TN)) { a.__setitem__(1L, null); V(a).Should().Equal(new object[] { "ab", null }); }

            using (var a = np.array(new sbyte[] { 1, 2 }))
            {
                Action wide = () => a.__setitem__(0, 300);
                wide.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");
                V(a).Should().Equal((sbyte)1, (sbyte)2);
            }
            using (var a = np.array(new[] { 1, 2 })) { a.__setitem__(0, 2.9); V(a).Should().Equal(2, 2); }
            using (var a = np.array(new[] { "ab", "cd" }, T)) { a.__setitem__(0, 5); V(a).Should().Equal("5", "cd"); }
            using (var a = np.array(new long[] { 1, 2 })) { a.__setitem__(0, "12"); V(a).Should().Equal(12L, 2L); }
            using (var a = np.array(new[] { "ab", "cd" })) { a.__setitem__(0, "xyz"); V(a).Should().Equal("xy", "cd"); }
        }

        /// <summary>
        ///     The object-typed <c>SetData</c> on a <c>StringDType</c> array takes a text, a char (one character), null (the NA
        ///     object) and bytes (an <c>S</c> value, decoded) — it used to throw <c>InvalidCastException</c> casting the
        ///     descriptor to the fixed-width class — and still stores a boxed number as its <c>str()</c>.
        /// </summary>
        [TestMethod]
        public void SetDataObject_OnStringDType()
        {
            using (var a = np.array(new[] { "a", "b" }, T)) { a.SetData((object)"x", new[] { 0 }); V(a).Should().Equal("x", "b"); }
            using (var a = np.array(new[] { "a", "b" }, TN)) { a.SetData((object)null, new[] { 0 }); V(a).Should().Equal(new object[] { null, "b" }); }
            using (var a = np.array(new[] { "a", "b" }, T)) { a.SetData((object)'c', new[] { 1 }); V(a).Should().Equal("a", "c"); }
            using (var a = np.array(new[] { "a", "b" }, T)) { a.SetData((object)new byte[] { 0x71, 0x72 }, new[] { 1 }); V(a).Should().Equal("a", "qr"); }
            using (var a = np.array(new[] { "a", "b" }, T)) { a.SetData((object)5, new[] { 1 }); V(a).Should().Equal("a", "5"); }
            using (var a = np.array(new[] { "a", "b" }, T)) { a.SetData((object)"neg", new[] { -1 }); V(a).Should().Equal("a", "neg"); }
            using (var a = np.array(new[] { "a", "b" }, T))
            {
                Action outside = () => a.SetData((object)"x", new[] { 2 });
                outside.Should().Throw<IndexError>();
            }
            using (var a = np.array(new[] { "ab", "cd" })) { a.SetData((object)null, new[] { 0 }); V(a).Should().Equal("No", "cd"); }
        }

        /// <summary>
        ///     <c>None</c> into a numeric array (P1.7b): NumPy stores NaN into floats (<c>nan+nanj</c> into complex), False into
        ///     bool and raises <c>int(None)</c>'s TypeError for integers, and NumSharp now does the same through the indexer and
        ///     <c>__setitem__</c> (every other route: <see cref="NoneStoreTests"/>).
        /// </summary>
        [TestMethod]
        public void None_IntoNumbers_IsRefused()
        {
            using var f = np.array(new[] { 1.0, 2.0 });
            f[0] = null;
            f.__setitem__(1, null);
            BitConverter.DoubleToInt64Bits(f.GetDouble(0)).Should().Be(0x7FF8_0000_0000_0000L);
            BitConverter.DoubleToInt64Bits(f.GetDouble(1)).Should().Be(0x7FF8_0000_0000_0000L);

            using var i = np.array(new[] { 1, 2 });
            Action store = () => i[0] = null;
            store.Should().Throw<TypeError>()
                .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            Action setItem = () => i.__setitem__(0, null);
            setItem.Should().Throw<TypeError>();
            V(i).Should().Equal(1, 2);
        }
    }
}
