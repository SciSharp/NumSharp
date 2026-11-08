using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     <c>numpy.char</c> phase 2 (docs/plans/np-char.md): <see cref="NDCharArray"/> as data — the constructor,
    ///     <c>np.char.array</c> / <c>asarray</c> with NumPy's quirks, the finalize rule, the class kept on views and copies,
    ///     element access and iteration that strip trailing whitespace (<c>item()</c> / <c>tolist()</c> do not), the
    ///     <c>chararray(...)</c> repr and <c>view(type)</c>. Every literal expectation was probed against NumPy 2.4.2 (the
    ///     session's <c>probe_unit.py</c>); the string oracle's section <c>strchararray</c> replays the full matrices
    ///     (11,505 cases). The <see cref="MisalignedAttribute"/> tests pin the documented divergences.
    /// </summary>
    [TestClass]
    public class CharArrayTests
    {
        // ------------------------------------------------------------------ the strip tables

        /// <summary>
        ///     The <c>U</c> strip set is Python's <c>str.isspace()</c> (CPython 3.12, Unicode 15.0): every BMP code point .NET's
        ///     <see cref="char.IsWhiteSpace(char)"/> accepts plus the four information separators U+001C-U+001F, 29 in all,
        ///     and nothing past the BMP or past U+10FFFF. A drift in either direction strips a character NumPy keeps (or
        ///     keeps one it strips) at every element access.
        /// </summary>
        [TestMethod]
        public void PyWhitespace_IsPythonsStrIsspace()
        {
            var wrong = new List<string>();
            int count = 0;
            for (uint cp = 0; cp <= 0xFFFF; cp++)
            {
                bool expected = char.IsWhiteSpace((char)cp) || cp is >= 0x1C and <= 0x1F;
                if (NDCharArray.IsPyWhitespace(cp) != expected)
                    wrong.Add($"U+{cp:X4}");
                count += expected ? 1 : 0;
            }
            for (uint cp = 0x10000; cp <= 0x10FFFF; cp += 0x101)
                if (NDCharArray.IsPyWhitespace(cp))
                    wrong.Add($"U+{cp:X6}");
            wrong.Should().BeEmpty();
            count.Should().Be(29);
            NDCharArray.IsPyWhitespace(0x110000).Should().BeFalse();
            NDCharArray.IsPyWhitespace(0xFFFFFFFF).Should().BeFalse();
        }

        /// <summary>The <c>S</c> strip set is <c>bytes.rstrip()</c>'s: the six ASCII bytes <c>\t \n \v \f \r</c> and space.</summary>
        [TestMethod]
        public void BytesWhitespace_IsTheSixAsciiBytes()
        {
            Enumerable.Range(0, 256).Where(b => NDCharArray.IsBytesWhitespace((byte)b)).Should().Equal(9, 10, 11, 12, 13, 32);
        }

        // ------------------------------------------------------------------ element access and iteration

        /// <summary>
        ///     A full integer index answers the element STRIPPED with Python's rule (U+3000 and U+001C strip, a zero-width
        ///     space stays) as a read-only 0-d stand-in for NumPy's immutable <c>str</c>; <c>item()</c>, <c>tolist()</c>,
        ///     <c>take(i)</c> and the flat iterator answer the element unstripped (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void ElementAccess_StripsWithPythonsRule_ItemTolistTakeFlatDoNot()
        {
            var c = np.@char.array(new[] { "ab  ", "c\u3000", "d\x1c", "e\u200b" });

            c[0].item().Should().Be("ab");
            c[1].item().Should().Be("c");
            c[2].item().Should().Be("d");
            c[3].item().Should().Be("e\u200b");
            c[0].ndim.Should().Be(0);
            c[0].flags.writeable.Should().BeFalse();

            c.item(0).Should().Be("ab  ");
            ((IList)c.tolist()).Cast<object>().Should().Equal("ab  ", "c\u3000", "d\x1c", "e\u200b");
            c.take(0L).item().Should().Be("ab  ");
            c.flatiter[1].Should().Be("c\u3000");
        }

        /// <summary>
        ///     <c>S</c> elements strip the six ASCII bytes only: <c>\x1c</c>, <c>\x85</c> and <c>\xa0</c> stay (bytes.rstrip),
        ///     and every element access spelling gives the same stripped value.
        /// </summary>
        [TestMethod]
        public void ElementAccess_OfBytes_StripsAsciiWhitespaceOnly()
        {
            var c = np.@char.array(new object[] { Bytes("ab \t"), Bytes("c\x1c"), Bytes("d\x85"), Bytes("e\xa0 ") });

            ((byte[])c[0].item()).Should().Equal(Bytes("ab"));
            ((byte[])c[1].item()).Should().Equal(Bytes("c\x1c"));
            ((byte[])c["2"].item()).Should().Equal(Bytes("d\x85"));
            ((byte[])c[new object[] { 3 }].item()).Should().Equal(Bytes("e\xa0"));
        }

        /// <summary>
        ///     Iteration goes through the element access: a 1-d chararray yields its STRIPPED elements (exact .NET values,
        ///     an embedded NUL before stripped whitespace kept: <c>'a\0 '</c> → <c>'a\0'</c>), an N-d one yields chararray rows
        ///     (views, unstripped elements inside), a 0-d one refuses with NumPy's TypeError (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void Iteration_StrippedElements_ChararrayRows_ZeroDRefused()
        {
            // U+001C is Python whitespace but not .NET's (char.IsWhiteSpace): the enumerator must strip it too.
            var c = np.@char.array(new[] { "ab  ", "c\u3000", "a\0 ", "d\x1c" });
            var items = new List<object>();
            foreach (var x in c)
                items.Add(x);
            items.Should().Equal("ab", "c", "a\0", "d");

            var m = np.@char.array(new[,] { { "a ", "b" }, { "c", "d " } });
            var rows = new List<NDArray>();
            foreach (var row in m)
                rows.Add((NDArray)row);
            rows.Should().HaveCount(2);
            rows.Should().AllSatisfy(r => (r is NDCharArray).Should().BeTrue());
            ((IList)rows[1].tolist()).Cast<object>().Should().Equal("c", "d ");
            rows[0].flags.owndata.Should().BeFalse();

            var zero = (NDCharArray)np.array("ab ").view(typeof(NDCharArray));
            Action iterate = () => { foreach (var _ in zero) { } };
            iterate.Should().Throw<TypeError>().Where(e => e.Message == "iteration over a 0-d array");
        }

        /// <summary>
        ///     The index forms NumPy answers with an array: a row is a view; a 0-d integer array beside slices is
        ///     <c>HAS_SCALAR_ARRAY</c>'s owning copy; an index array, a mask and a scalar bool return a chararray VIEW of the
        ///     gathered array (OWNDATA false); an ellipsis is a view. On a 0-d chararray the empty index is the stripped
        ///     element and the ellipsis a 0-d chararray (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void IndexForms_ClassAndOwnership()
        {
            var m = np.@char.array(new[,] { { "a ", "b" }, { "c", "d " } });

            NDCharArray row = m[0];
            row.flags.owndata.Should().BeFalse();
            ((IList)row.tolist()).Cast<object>().Should().Equal("a ", "b");

            // One 0-d integer array per axis is a FULL integer index (mapping.c: a full integer index sees array scalars as
            // part of itself): the stripped element, as for plain integers.
            NDCharArray element = m[new object[] { NDArray.Scalar(0L), NDArray.Scalar(0L) }];
            element.ndim.Should().Be(0);
            element.item().Should().Be("a");
            element.flags.writeable.Should().BeFalse();

            NDCharArray scalarArray = m[new object[] { NDArray.Scalar(0L) }];
            scalarArray.flags.owndata.Should().BeTrue();
            ((IList)scalarArray.tolist()).Cast<object>().Should().Equal("a ", "b");
            m[new object[] { NDArray.Scalar(0L), Slice.All }].flags.owndata.Should().BeTrue();

            NDCharArray fancy = m[new object[] { np.array(new long[] { 1, 0 }) }];
            fancy.shape.Should().Equal(2L, 2L);
            fancy.flags.owndata.Should().BeFalse();

            NDCharArray masked = m[np.array(new[,] { { true, false }, { true, true } }).MakeGeneric<bool>()];
            ((IList)masked.tolist()).Cast<object>().Should().Equal("a ", "c", "d ");
            masked.flags.owndata.Should().BeFalse();

            NDCharArray yes = m[new object[] { true }];
            yes.shape.Should().Equal(1L, 2L, 2L);
            yes.flags.owndata.Should().BeFalse();
            m[new object[] { false }].shape.Should().Equal(0L, 2L, 2L);
            m[Slice.Ellipsis].flags.owndata.Should().BeFalse();

            var zero = (NDCharArray)np.array("ab ").view(typeof(NDCharArray));
            zero[""].item().Should().Be("ab");
            NDCharArray ellipsis = zero[Slice.Ellipsis];
            ellipsis.ndim.Should().Be(0);
            ellipsis.item().Should().Be("ab ");
            ellipsis.flags.writeable.Should().BeTrue();
        }

        // ------------------------------------------------------------------ the finalize rule

        /// <summary>
        ///     <c>__array_finalize__</c> accepts string data (and <c>int8</c>, char <c>'b'</c>): <c>astype(float)</c> raises,
        ///     <c>view(np.uint32)</c> keeps the class (the rule reads the dtype the array HAS) so that chararray's own views
        ///     raise while its element access answers NumPy numbers; <c>astype(int8)</c> of digits is an int8 chararray, a
        ///     plain int64 array cannot become one (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void FinalizeRule_StringDataOnly_ViewReadsTheCurrentDtype()
        {
            const string text = "Can only create a chararray from string data.";
            var u = np.@char.array(new[] { "ab", "c" });

            Action astype = () => u.astype(np.float64);
            astype.Should().Throw<ValueError>().Where(e => e.Message == text);

            NDCharArray codes = u.view(np.uint32);
            codes.dtype.Should().Be(np.uint32);
            codes.ToArray<uint>().Should().Equal(97u, 98u, 99u, 0u);
            codes[0].item().Should().Be(97u);
            codes[0].dtype.Should().Be(np.uint32);
            Action transpose = () => { var _ = codes.T; };
            transpose.Should().Throw<ValueError>().Where(e => e.Message == text);

            NDCharArray digits = np.@char.array(new[] { "1", "-5" }).astype(np.int8);
            digits.dtype.Should().Be(np.int8);
            digits.ToArray<sbyte>().Should().Equal((sbyte)1, (sbyte)-5);

            Action numeric = () => np.arange(3).view(typeof(NDCharArray));
            numeric.Should().Throw<ValueError>().Where(e => e.Message == text);
            np.arange(3).astype(np.int8).view(typeof(NDCharArray)).Should().BeOfType<NDCharArray>();
        }

        // ------------------------------------------------------------------ members

        /// <summary>
        ///     The members a chararray inherits keep the class, with NumPy's ownership (a view does not own, a copy does) and
        ///     NumPy's identities: <c>real</c>, <c>squeeze()</c> without a length-1 axis, <c>astype</c> of the same dtype with
        ///     <c>copy=False</c> and <c>to_device</c> answer the chararray itself; <c>imag</c> is a read-only chararray of empty
        ///     strings (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void Members_KeepTheClass_OwnershipAndIdentities()
        {
            var u = np.@char.array(new[] { "ab", "c" });
            ReferenceEquals(u.real, u).Should().BeTrue();
            ReferenceEquals(u.squeeze(), u).Should().BeTrue();
            ReferenceEquals(u.astype(np.dtype("U2"), copy: false), u).Should().BeTrue();
            ReferenceEquals(u.to_device("cpu"), u).Should().BeTrue();
            NDCharArray imag = u.imag;
            imag.flags.writeable.Should().BeFalse();
            imag.flags.owndata.Should().BeTrue();
            ((IList)imag.tolist()).Cast<object>().Should().Equal("", "");

            var m = np.@char.array(new[,] { { "a ", "b" }, { "c", "d " } });
            NDCharArray reshaped = m.reshape(-1);
            reshaped.flags.owndata.Should().BeFalse();
            ((IList)reshaped.tolist()).Cast<object>().Should().Equal("a ", "b", "c", "d ");
            m.flatten().flags.owndata.Should().BeTrue();
            m.copy().flags.owndata.Should().BeTrue();
            m.ravel().flags.owndata.Should().BeFalse();
            NDCharArray diagonal = m.diagonal();
            diagonal.flags.writeable.Should().BeFalse();
            ((IList)diagonal.tolist()).Cast<object>().Should().Equal("a ", "d ");
            ((IList)u.getfield(np.dtype("U1"), 4).tolist()).Cast<object>().Should().Equal("b", "");
            ((IList)u.repeat(2).tolist()).Cast<object>().Should().Equal("ab", "ab", "c", "c");
            ((IList)u.compress(np.array(new[] { true, false })).tolist()).Cast<object>().Should().Equal("ab");
            (u.T is NDCharArray).Should().BeTrue();
        }

        // ------------------------------------------------------------------ the constructor

        /// <summary>
        ///     <c>np.char.chararray(shape, itemsize, unicode, buffer, offset, strides, order)</c>: a str buffer FILLS every
        ///     element; Python bytes (<c>ReadOnlyMemory&lt;byte&gt;</c>) are a read-only buffer, a bytearray (<c>byte[]</c>) a
        ///     writable one; offset and negative strides address the buffer; NumPy's errors in its texts (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void Constructor_BufferKindsAndErrors()
        {
            var filled = np.@char.chararray(new Shape(2, 2), itemsize: 3, unicode: true, buffer: "ab");
            filled.dtype.Should().Be(np.dtype("<U3"));
            ((IList)filled.tolist()).Cast<IList>().SelectMany(r => r.Cast<object>()).Should().AllBeEquivalentTo("ab");
            filled.flags.owndata.Should().BeTrue();

            var readOnly = np.@char.chararray(new Shape(2), itemsize: 2, buffer: new ReadOnlyMemory<byte>(Bytes("abcd")));
            readOnly.flags.writeable.Should().BeFalse();
            ((IList)readOnly.tolist()).Cast<byte[]>().Should().BeEquivalentTo(new[] { Bytes("ab"), Bytes("cd") }, o => o.WithStrictOrdering());

            var writable = np.@char.chararray(new Shape(2), itemsize: 2, buffer: Bytes("abcd"));
            writable.flags.writeable.Should().BeTrue();
            writable.flags.owndata.Should().BeFalse();

            var reversed = np.@char.chararray(new Shape(2), itemsize: 1, buffer: new ReadOnlyMemory<byte>(Bytes("abcd")),
                                              offset: 3, strides: new long[] { -2 });
            ((IList)reversed.tolist()).Cast<byte[]>().Should().BeEquivalentTo(new[] { Bytes("d"), Bytes("b") }, o => o.WithStrictOrdering());

            Action small = () => np.@char.chararray(new Shape(3), itemsize: 6, buffer: new ReadOnlyMemory<byte>(Bytes("abcd")));
            small.Should().Throw<TypeError>().Where(e => e.Message == "buffer is too small for requested array");
            Action wide = () => np.@char.chararray(new Shape(0), itemsize: 1L << 31);
            wide.Should().Throw<ValueError>().Where(e => e.Message == "invalid itemsize in generic type tuple");
            Action negative = () => np.@char.chararray(new Shape(-1));
            negative.Should().Throw<Exception>().Where(e => e.Message == "negative dimensions are not allowed");
        }

        // ------------------------------------------------------------------ np.char.array / asarray

        /// <summary>
        ///     Numbers, bools, None and <c>StringDType</c> become text through NumPy's <c>tolist()</c> route, value-based:
        ///     float32 0.1 prints its double value (S19), float16 its own (S15); with <c>itemsize=0</c> the cast is to the
        ///     unsized string (int64 → S21); a list mixing str and int is U21; Python scalars give 0-d chararrays (probed
        ///     2.4.2).
        /// </summary>
        [TestMethod]
        public void Array_NumbersBoolsNoneAndStringDTypeBecomeText()
        {
            var f4 = np.@char.array(np.array(new[] { 0.1f }));
            f4.dtype.Should().Be(np.dtype("S19"));
            ((byte[])f4.item(0)).Should().Equal(Bytes("0.10000000149011612"));
            var f2 = np.@char.array(np.array(new[] { (Half)0.1 }));
            ((byte[])f2.item(0)).Should().Equal(Bytes("0.0999755859375"));
            np.@char.array(np.array(new long[] { 1, 22 }), itemsize: 0).dtype.Should().Be(np.dtype("S21"));

            var mixed = np.@char.array(new object[] { "a", 1L });
            mixed.dtype.Should().Be(np.dtype("<U21"));
            ((IList)mixed.tolist()).Cast<object>().Should().Equal("a", "1");
            var grid = np.@char.array(new object[,] { { 1L, 2L }, { 3L, 4L } });
            grid.dtype.Should().Be(np.dtype("S1"));
            grid.shape.Should().Equal(2L, 2L);

            var none = np.@char.array((object)null);
            none.ndim.Should().Be(0);
            ((byte[])none.item()).Should().Equal(Bytes("None"));
            ((byte[])np.@char.array(BigInteger.Pow(2, 70)).item()).Should().Equal(Bytes("1180591620717411303424"));
            np.@char.array(5L, unicode: true).item().Should().Be("5");

            var t = np.@char.array(np.array(new[] { "ab", "c" }, np.dtypes.StringDType()));
            t.dtype.Should().Be(np.dtype("S2"));
            var tn = np.@char.array(np.array(new object[] { "ab", null }, np.dtypes.StringDType(na_object: null)));
            ((byte[])tn.item(1)).Should().Equal(Bytes("None"));
        }

        /// <summary>
        ///     A str / bytes argument: the itemsize defaults to its length and the shape is <c>len // itemsize</c>, so an empty
        ///     one is Python's ZeroDivisionError; a str FILLS (truncated to the itemsize), bytes are a read-only buffer
        ///     (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void Array_StrAndBytesArguments()
        {
            var text = np.@char.array("abcdef", itemsize: 4);
            text.dtype.Should().Be(np.dtype("<U4"));
            ((IList)text.tolist()).Cast<object>().Should().Equal("abcd");

            var bytes = np.@char.array(Bytes("abcd"), itemsize: 2);
            bytes.flags.writeable.Should().BeFalse();
            ((IList)bytes.tolist()).Cast<byte[]>().Should().BeEquivalentTo(new[] { Bytes("ab"), Bytes("cd") }, o => o.WithStrictOrdering());

            Action empty = () => np.@char.array("");
            empty.Should().Throw<DivideByZeroException>().Where(e => e.Message == "integer division or modulo by zero");
        }

        /// <summary>
        ///     <c>np.char.asarray</c> answers an <c>S</c> chararray ITSELF; a <c>U</c> chararray is always cast (its itemsize
        ///     counts bytes, NumPy compares it with the width in characters); an <c>S</c> array with <c>unicode=True</c> and
        ///     <c>copy=False</c> stays <c>S</c> (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void Asarray_IdentityOfAnSChararray_UAlwaysCast()
        {
            var s = (NDCharArray)np.array(new object[] { Bytes("ab"), Bytes("c") }).view(typeof(NDCharArray));
            ReferenceEquals(np.@char.asarray(s), s).Should().BeTrue();
            np.@char.array(s, unicode: true, copy: false).dtype.Should().Be(np.dtype("S2"));

            var u = (NDCharArray)np.array(new[] { "ab", "c" }).view(typeof(NDCharArray));
            ReferenceEquals(np.@char.asarray(u), u).Should().BeFalse();
            ReferenceEquals(np.@char.array(u, copy: false), u).Should().BeFalse();
        }

        // ------------------------------------------------------------------ printing and view(type)

        /// <summary>
        ///     <c>repr</c> names the class and indents continuation lines past it, its elements STRIPPED (NumPy's
        ///     <c>_formatArray</c> indexes through <c>__getitem__</c>); a 0-d chararray prints unstripped (its repr reads the
        ///     plain array); <c>str</c> prints the stripped elements (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void Repr_ClassName_ContinuationIndent_StrippedElements()
        {
            np.@char.array(new[] { "ab ", "c" }).ToString(true).Should().Be("chararray(['ab', 'c'], dtype='<U3')");
            np.@char.array(new[,] { { "a", "b" }, { "c", "d" } }).ToString(true)
                .Should().Be("chararray([['a', 'b'],\n           ['c', 'd']], dtype='<U1')");
            np.array("ab ").view(typeof(NDCharArray)).ToString(true).Should().Be("chararray('ab ', dtype='<U3')");
            np.@char.array(new[] { "ab ", "c\t" }).ToString(false).Should().Be("['ab' 'c']");
            np.@char.array(new object[] { Bytes("ab "), Bytes("c") }).ToString(true).Should().Be("chararray([b'ab', b'c'], dtype='|S3')");
        }

        /// <summary>
        ///     <c>view(type)</c> checks the finalize rule against the CURRENT dtype: <c>view(int32, chararray)</c> of a <c>U</c>
        ///     array is an int32 chararray, a numeric array refuses, a type that is not an array class is NumPy's ValueError,
        ///     and <c>view(typeof(NDMatrix))</c> makes a 1-row matrix (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void ViewType_ChecksTheCurrentDtype()
        {
            var u = np.array(new[] { "ab", "c" });
            (u.view(typeof(NDCharArray)) is NDCharArray).Should().BeTrue();

            var codes = u.view(np.int32, typeof(NDCharArray));
            (codes is NDCharArray).Should().BeTrue();
            codes.ToArray<int>().Should().Equal(97, 98, 99, 0);

            Action numeric = () => np.arange(3).view((DType)null, typeof(NDCharArray));
            numeric.Should().Throw<ValueError>().Where(e => e.Message == "Can only create a chararray from string data.");
            Action notAnArrayClass = () => u.view((DType)null, typeof(long));
            notAnArrayClass.Should().Throw<ValueError>().Where(e => e.Message == "Type must be a sub-type of ndarray type");

            var matrix = u.view(typeof(NDMatrix));
            (matrix is NDMatrix).Should().BeTrue();
            matrix.shape.Should().Equal(1L, 2L);
        }

        // ------------------------------------------------------------------ documented divergences

        /// <summary>
        ///     NumPy's <c>np.char.array(obj, order=...)</c> of string data returns a PLAIN ndarray (its <c>np.asarray(obj,
        ///     order=order)</c> drops the subclass, though NumPy's own stubs declare a chararray). <c>np.@char.array</c> is typed
        ///     <see cref="NDCharArray"/>: NumSharp returns a chararray with NumPy's data, layout and flags.
        /// </summary>
        [TestMethod, Misaligned]
        public void Array_WithAnOrder_OfStringData_IsAChararray()
        {
            var u = (NDCharArray)np.array(new[] { "ab", "c" }).view(typeof(NDCharArray));
            var r = np.@char.array(u, order: 'F');
            (r is NDCharArray).Should().BeTrue();
            ((IList)r.tolist()).Cast<object>().Should().Equal("ab", "c");
        }

        /// <summary>
        ///     <c>np.char.chararray(shape, itemsize=0)</c> is NumPy's zero-width <c>S0</c> / <c>U0</c> array, which a NumSharp
        ///     string buffer cannot hold: NumSharp refuses with NotSupportedException.
        /// </summary>
        [TestMethod, Misaligned]
        public void Constructor_ItemsizeZero_IsNotSupported()
        {
            Action zero = () => np.@char.chararray(new Shape(3), itemsize: 0);
            zero.Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     A stripped element that ENDS in a NUL (<c>'a\0 '</c> → Python's <c>'a\0'</c>): the 0-d stand-in the indexer
        ///     returns is a fixed-width element, whose storage drops trailing NULs as every <c>S</c> / <c>U</c> element does, so
        ///     it reads <c>'a'</c>. The chararray's enumerator yields the exact <c>'a\0'</c>.
        /// </summary>
        [TestMethod, Misaligned]
        public void StandIn_OfAStrippedValueEndingInNul_LosesTheNul()
        {
            var c = np.@char.array(new[] { "a\0 " });
            c[0].item().Should().Be("a");
            foreach (var x in c)
                x.Should().Be("a\0");
        }

        /// <summary>The bytes of an ASCII text (Python's <c>b'...'</c> literal, one byte per character).</summary>
        /// <param name="text">The text; each character must be below U+0100.</param>
        /// <returns>The bytes.</returns>
        private static byte[] Bytes(string text) => text.Select(ch => (byte)ch).ToArray();
    }
}
