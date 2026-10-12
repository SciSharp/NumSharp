using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The casts of NEP 55 <c>StringDType</c> arrays (strings phase 1, P1.6; <see cref="NDVStringCasting"/>, NumPy's
    ///     <c>stringdtype/casts.cpp</c>): to and from <c>U</c> / <c>S</c> / bool / numbers and between <c>StringDType</c>
    ///     descriptors, their error texts, the corrupt-cell MemoryError texts, and the NumPy 2.4.2 cast bugs whose correct
    ///     answer NumSharp gives. Expected values were probed against <c>numpy==2.4.2</c>.
    /// </summary>
    [TestClass]
    public class StringDTypeCastTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor.</summary>
        private static DType TN => np.dtypes.StringDType(na_object: null);

        /// <summary>A fresh <c>StringDType(na_object=nan)</c> descriptor.</summary>
        private static DType TNaN => np.dtypes.StringDType(na_object: double.NaN);

        /// <summary>The elements of an array in C order.</summary>
        /// <param name="a">An array.</param>
        /// <returns>The values.</returns>
        private static object[] V(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => a.Storage.GetAtIndex(i)).ToArray();

        /// <summary>
        ///     The cast core's own checks: exactly one side is <c>StringDType</c> (a <c>StringDType</c> pair goes through the
        ///     copy core), the destination is writeable, both storages exist.
        /// </summary>
        [TestMethod]
        public void Core_Checks()
        {
            using var t = np.array(new[] { "1", "2" }, T);
            using var t2 = np.array(new[] { "1", "2" }, T);
            using var n = np.array(new[] { 1, 2 });
            using var n2 = np.array(new[] { 1, 2 });
            Action both = () => NDVStringCasting.Copy(t.Storage, t2.Storage);
            both.Should().Throw<ArgumentException>().WithMessage("A StringDType cast needs exactly one StringDType storage.");
            Action neither = () => NDVStringCasting.Copy(n.Storage, n2.Storage);
            neither.Should().Throw<ArgumentException>();
            Action nullDst = () => NDVStringCasting.Copy(null, t.Storage);
            nullDst.Should().Throw<ArgumentNullException>();
            Action nullSrc = () => NDVStringCasting.Copy(n.Storage, null);
            nullSrc.Should().Throw<ArgumentNullException>();
            using var ro = np.broadcast_to(n, new Shape(2, 2));
            Action readOnly = () => NDVStringCasting.Copy(ro.Storage, t.Storage);
            readOnly.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
        }

        /// <summary>
        ///     A bool element that is neither 0 nor 1 (a raw byte 2) is NumPy's <c>RuntimeError: invalid value encountered in bool
        ///     to string cast</c>; 0 and 1 are <c>'False'</c> / <c>'True'</c>.
        /// </summary>
        [TestMethod]
        public void BoolToT_InvalidByte_IsTheRuntimeError()
        {
            using var ok = np.frombuffer(new byte[] { 0, 1 }, np.dtype("?"));
            using (var r = ok.astype(T))
                V(r).Should().Equal(new object[] { "False", "True" });
            using var bad = np.frombuffer(new byte[] { 2 }, np.dtype("?"));
            Action cast = () => bad.astype(T);
            cast.Should().Throw<RuntimeError>().WithMessage("invalid value encountered in bool to string cast");
        }

        /// <summary>
        ///     A <c>U</c> element holding a surrogate code point or one above U+10FFFF has no UTF-8 form: NumPy's
        ///     <c>TypeError: Invalid unicode code point found</c>.
        /// </summary>
        [TestMethod]
        public void UToT_InvalidCodePoint_IsTheTypeError()
        {
            using var big = np.array(new uint[] { 0x110000 }).view(np.dtype("<U1"));
            Action high = () => big.astype(T);
            high.Should().Throw<TypeError>().WithMessage("Invalid unicode code point found");
            using var surrogate = np.array(new uint[] { 0xD800 }).view(np.dtype("<U1"));
            Action lone = () => surrogate.astype(T);
            lone.Should().Throw<TypeError>().WithMessage("Invalid unicode code point found");
        }

        /// <summary>
        ///     A corrupt cell (an arena string in a buffer without an arena) raises each cast loop's own MemoryError text, as
        ///     NumPy's — including the doubled "in" of <c>string_to_unicode</c> / <c>string_to_bytes</c>, whose context already
        ///     starts with "in".
        /// </summary>
        [TestMethod]
        public unsafe void CorruptCell_IsEachLoopsMemoryError()
        {
            using var a = np.empty(new Shape(1), T);
            byte* cell = a.Storage.VStringSlice.Address;
            *(ulong*)cell = 1;
            *(ulong*)(cell + 8) = 20UL | (0x40UL << 56);
            try
            {
                Action toU = () => a.astype(np.dtype("U4"));
                toU.Should().Throw<OutOfMemoryException>().WithMessage("Failed to load string in in string to unicode cast");
                Action toS = () => a.astype(np.dtype("S4"));
                toS.Should().Throw<OutOfMemoryException>().WithMessage("Failed to load string in in string to bytes cast");
                Action toBool = () => a.astype(np.dtype("?"));
                toBool.Should().Throw<OutOfMemoryException>().WithMessage("Failed to load string in string to bool cast");
                Action toInt = () => a.astype(np.dtype("i8"));
                toInt.Should().Throw<OutOfMemoryException>().WithMessage("Failed to load string for conversion to a non-nullable type");
            }
            finally
            {
                new Span<byte>(cell, 16).Clear();
            }
        }

        /// <summary>
        ///     Between <c>StringDType</c> descriptors: NA stays NA when the target has an NA object (None to nan), and becomes the
        ///     NA object's <c>str()</c> when it has none; into <c>U</c> / <c>S</c> an NA reads as its <c>str()</c> and a text is
        ///     cut by code points; into numbers a non-ASCII text is CPython's int() error quoting it; float64 NaN becomes NA
        ///     only for a NaN-like NA object, float32 / float16 NaN stay <c>'nan'</c>.
        /// </summary>
        [TestMethod]
        public void Casts_NARulesAndTexts()
        {
            using (var a = np.array(new object[] { "a", null }, TN))
            {
                using (var r = a.astype(TNaN))
                {
                    V(r)[0].Should().Be("a");
                    V(r)[1].Should().BeOfType<double>().Which.Should().Be(double.NaN);
                }
                using (var r = a.astype(T))
                    V(r).Should().Equal(new object[] { "a", "None" });
                using (var r = a.astype(np.dtype("S4")))
                    V(r).Should().Equal(new object[] { new byte[] { (byte)'a' }, "None"u8.ToArray() }, (x, y) => ((byte[])x).AsSpan().SequenceEqual((byte[])y));
            }
            using (var a = np.array(new[] { "😀😀", "ab" }, T))
            using (var r = a.astype(np.dtype("U1")))
                V(r).Should().Equal(new object[] { "😀", "a" });
            using (var a = np.array(new[] { "é" }, T))
            {
                Action parse = () => a.astype(np.dtype("i8"));
                parse.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: 'é'");
                Action sameKind = () => a.astype(np.dtype("i8"), true, 'K', "same_kind");
                sameKind.Should().Throw<Exception>().WithMessage("Cannot cast array data from StringDType() to dtype('int64') according to the rule 'same_kind'");
            }
            using (var f8 = np.array(new[] { double.NaN, 1.5 }))
            using (var r = f8.astype(TNaN))
            {
                V(r)[0].Should().BeOfType<double>("a float64 NaN is the NA of a NaN-like NA object");
                V(r)[1].Should().Be("1.5");
            }
            using (var f4 = np.array(new[] { float.NaN, 1.5f }))
            using (var r = f4.astype(TNaN))
                V(r).Should().Equal(new object[] { "nan", "1.5" });
            using (var big = np.array(new[] { "2147483648" }, T))
            {
                Action overflow = () => big.astype(np.dtype("i4"));
                overflow.Should().Throw<OverflowException>().WithMessage("Integer 2147483648 is out of bounds for int",
                    "NumSharp pins the LP64 C type name of int32 (Windows NumPy says 'long')");
            }
        }

        /// <summary>
        ///     NumPy 2.4.2's <c>StringDType → complex</c> cast writes the real part into the imaginary part too
        ///     (<c>['1+2j', '3', '-1.5j']</c> becomes <c>[1+1j, 3+3j, 0+0j]</c>); NumSharp parses each text with CPython's
        ///     <c>complex()</c>.
        /// </summary>
        [TestMethod]
        public void NumPyBug_TToComplex_IsCorrectHere()
        {
            using var a = np.array(new[] { "1+2j", "3", "-1.5j" }, T);
            using var r = a.astype(np.dtype("c16"));
            V(r).Should().Equal(new object[] { new Complex(1, 2), new Complex(3, 0), new Complex(0, -1.5) });
        }

        /// <summary>
        ///     NumPy 2.4.2's <c>bytes → StringDType</c> cast copies the bytes WITHOUT validating UTF-8 (casts.cpp
        ///     <c>bytes_to_string</c>), so invalid UTF-8 is stored and every later read raises; NumSharp raises the same
        ///     UnicodeDecodeError at the cast, so no StringDType array ever holds invalid text.
        /// </summary>
        [TestMethod]
        public void NumPyBug_SToT_InvalidUtf8_RaisesAtTheCast()
        {
            // A byte[][] without a dtype is a nested list of ints (a uint8 matrix); under an S dtype each byte[] is a bytes object.
            using var s = np.array(new[] { new byte[] { (byte)'d' }, new byte[] { 0xFF } }, np.dtype("S1"));
            Action cast = () => s.astype(T);
            cast.Should().Throw<UnicodeDecodeError>().WithMessage("'utf-8' codec can't decode byte 0xff in position 0: invalid start byte");
            using var ok = np.array(new[] { new byte[] { 0xC3, 0xA9 }, new byte[] { 0xE2, 0x82, 0xAC } }, np.dtype("S3"));
            using var r = ok.astype(T);
            V(r).Should().Equal(new object[] { "é", "€" });
        }

        /// <summary>
        ///     <c>bytes → StringDType</c> validates four-byte UTF-8 sequences with CPython's ranges: a bad second, third or
        ///     fourth byte, a value past U+10FFFF, and a sequence cut by the end of the element; a valid one is one astral code
        ///     point.
        /// </summary>
        [TestMethod]
        public void SToT_FourByteSequences_HaveCPythonsRanges()
        {
            const string Codec = "'utf-8' codec can't decode ";
            var errors = new (byte[] Bytes, string Text)[]
            {
                (new byte[] { 0xF0, 0x90, 0x28, 0xBC }, "bytes in position 0-1: invalid continuation byte"),
                (new byte[] { 0xF0, 0x90, 0x8C, 0x28 }, "bytes in position 0-2: invalid continuation byte"),
                (new byte[] { 0xF0, 0x28, 0x8C, 0xBC }, "byte 0xf0 in position 0: invalid continuation byte"),
                (new byte[] { 0xF4, 0x90, 0x80, 0x80 }, "byte 0xf4 in position 0: invalid continuation byte"),
                (new byte[] { 0xF0, 0x9F, 0x98 }, "bytes in position 0-2: unexpected end of data"),
                (new byte[] { 0xF0, 0x9F }, "bytes in position 0-1: unexpected end of data"),
                // A cut sequence that an early byte already proves invalid is a continuation error (CPython's texts).
                (new byte[] { 0xF0, 0x28 }, "byte 0xf0 in position 0: invalid continuation byte"),
                (new byte[] { 0xF0, 0x90, 0x28 }, "bytes in position 0-1: invalid continuation byte"),
                // The diagnosis walks past a valid four-byte character to the first bad byte.
                (new byte[] { 0xF0, 0x9F, 0x98, 0x80, 0xFF }, "byte 0xff in position 4: invalid start byte"),
            };
            foreach (var (bytes, text) in errors)
            {
                using var s = np.array(new[] { bytes }, np.dtype("S5"));
                Action cast = () => s.astype(T);
                cast.Should().Throw<UnicodeDecodeError>(BitConverter.ToString(bytes)).WithMessage(Codec + text);
            }
            using var valid = np.array(new[] { new byte[] { 0xF0, 0x9F, 0x98, 0x80 } }, np.dtype("S4"));
            using var r = valid.astype(T);
            r.GetValue(0).Should().Be("\U0001F600");
        }

        /// <summary>
        ///     The cast walk over an EMPTY destination makes no run call (its callers return earlier; the walk itself holds
        ///     too).
        /// </summary>
        [TestMethod]
        public unsafe void Walk_OverNothing_RunsNothing()
        {
            using var dst = np.empty(new Shape(0), T);
            using var src = np.array(new string[0], "U1");
            int calls = 0;
            NDVStringCasting.Walk(dst.Storage, src.Storage, (s, ss, d, ds, n) => calls++);
            calls.Should().Be(0);
        }

        /// <summary>
        ///     Where dynamic code is unavailable (the IL kernels switched off), the number casts into and out of
        ///     <c>StringDType</c> convert through the scalar cast loop instead of the house IL cast kernel, with the same
        ///     answers: int32 → text widens to the canonical int64 first, text → int32 narrows from it.
        /// </summary>
        [TestMethod]
        public void NumberCasts_WithoutDynamicCode_GiveTheSameAnswers()
        {
            using var ints = np.array(new[] { 1, -22, 333 });
            using var text = np.array(new[] { "4", "-55", "666" }, T);
            using var withIL = ints.astype(T);
            using var parsedWithIL = text.astype(np.int32);
            DirectILKernelGenerator.Enabled = false;
            try
            {
                using var r = ints.astype(T);
                V(r).Should().Equal(V(withIL));
                using var p = text.astype(np.int32);
                p.ToArray<int>().Should().Equal(parsedWithIL.ToArray<int>());
                using var f = np.array(new[] { 1.5f }).astype(T);
                V(f).Should().Equal(new object[] { "1.5" });
            }
            finally
            {
                DirectILKernelGenerator.Enabled = true;
            }
        }

        /// <summary>
        ///     <c>StringDType → complex</c> of a text that is not a number raises CPython's <c>complex()</c> texts, quoting the
        ///     Python str for an underscore error.
        /// </summary>
        [TestMethod]
        public void TToComplex_ParseErrors_AreCPythons()
        {
            using var x = np.array(new[] { "x" }, T);
            Action malformed = () => x.astype(np.dtype("c16"));
            malformed.Should().Throw<ValueError>().WithMessage("complex() arg is a malformed string");
            using var u = np.array(new[] { "1__0j" }, T);
            Action underscore = () => u.astype(np.dtype("c16"));
            underscore.Should().Throw<ValueError>().WithMessage("could not convert string to complex: '1__0j'");
        }
    }
}
