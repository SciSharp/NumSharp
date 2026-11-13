using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 12 (the owner approved fixes 1-9 and kept np.histogram's four <c>bins</c> forms, decision "C keep
    ///     four"): np.interp's and np.append's one signature, np.lib.stride_tricks.as_strided, np.row_stack (with
    ///     np.vstack / np.row_stack over <see cref="NDListArray"/>), the masked array's <c>base</c> / <c>device</c> /
    ///     <c>to_device</c> / <c>getfield</c> / <c>setfield</c>, and NumPy's constructors for <see cref="np.NDIndex"/>,
    ///     <see cref="np.NDIterator"/> and <see cref="NDCharArray"/>. Every expected value, dtype and text is NumPy 2.4.2's
    ///     (scratchpad dbg12/numpy12.txt and numpy12_tests.txt), for the call exactly as the C# fix makes it.
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch12Tests
    {
        /// <summary>Asserts that <paramref name="act"/> throws <typeparamref name="T"/> with NumPy's exact text.</summary>
        /// <typeparam name="T">The exception type NumPy's error maps to.</typeparam>
        /// <param name="act">The call that must fail.</param>
        /// <param name="message">NumPy 2.4.2's text, verbatim.</param>
        /// <exception cref="AssertFailedException">The call does not throw <typeparamref name="T"/> with that text.</exception>
        private static void Throws<T>(Action act, string message) where T : Exception
            => act.Should().Throw<T>().Which.Message.Should().Be(message);

        /// <summary>The bytes of an ASCII text (a Python <c>bytes</c> literal's content).</summary>
        /// <param name="text">ASCII text.</param>
        /// <returns>One byte per character.</returns>
        private static byte[] Bytes(string text) => text.Select(ch => (byte)ch).ToArray();

        /// <summary>The evidence's <c>x = np.array([-1.0])</c>: one query point below the samples.</summary>
        /// <returns>A new float64 (1,) array.</returns>
        private static NDArray Below() => np.array(new[] { -1.0 });

        /// <summary>The evidence's <c>xp = np.array([0.0, 1.0])</c>.</summary>
        /// <returns>A new float64 (2,) array.</returns>
        private static NDArray Xp() => np.array(new[] { 0.0, 1.0 });

        /// <summary>The evidence's real <c>fp = np.array([10.0, 20.0])</c>.</summary>
        /// <returns>A new float64 (2,) array.</returns>
        private static NDArray Fp() => np.array(new[] { 10.0, 20.0 });

        /// <summary>The evidence's complex <c>fpc = np.array([1j, 1+2j])</c>.</summary>
        /// <returns>A new complex128 (2,) array.</returns>
        private static NDArray Fpc() => np.array(new[] { new Complex(0, 1), new Complex(1, 2) });

        /// <summary>The evidence's <c>x = np.arange(6, dtype=np.int64)</c>.</summary>
        /// <returns>A new int64 (6,) array.</returns>
        private static NDArray Six() => np.array(new long[] { 0, 1, 2, 3, 4, 5 });

        // ------------------------------------------------------------------------------------------------------------
        // Fix 1: np.interp(x, xp, fp, left=None, right=None, period=None) - one signature, NDArray fills
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>A C# number is NumPy's Python number: <c>left=-1.0</c> and <c>left=-1</c> give <c>[-1.0]</c>,
        ///     <c>left=np.nan</c> NaN, a float32 fp with <c>left=1e40</c> the float64 <c>1e+40</c>, a bool <c>[1.0]</c>, and
        ///     <c>right=np.array(-0.0)</c> keeps the negative zero.</summary>
        [TestMethod]
        public void Interp_CSharpNumbers_AreNumPysPythonNumbers()
        {
            np.interp(Below(), Xp(), Fp(), -1.0).ToArray<double>().Should().Equal(-1.0);
            np.interp(Below(), Xp(), Fp(), -1).ToArray<double>().Should().Equal(-1.0);
            double.IsNaN(np.interp(Below(), Xp(), Fp(), double.NaN).GetDouble(0)).Should().BeTrue();
            np.interp(Below(), Xp(), Fp(), true).ToArray<double>().Should().Equal(1.0);

            var wide = np.interp(Below(), Xp(), np.array(new[] { 1f, 2f }), 1e40);
            wide.dtype.Should().Be(np.float64);
            wide.GetDouble(0).Should().Be(1e40);

            var negativeZero = np.interp(np.array(new[] { 5.0 }), Xp(), Fp(), null, np.array(-0.0));
            BitConverter.DoubleToInt64Bits(negativeZero.GetDouble(0)).Should().Be(BitConverter.DoubleToInt64Bits(-0.0));
        }

        /// <summary>A C# Complex is a Python complex: the real loop refuses it even with a zero imaginary part
        ///     (<c>must be real number, not complex</c>); the complex loop keeps both parts (<c>(1+2j)</c>), and takes a real
        ///     fill as <c>(2.5+0j)</c> and a bool as <c>(1+0j)</c>.</summary>
        [TestMethod]
        public void Interp_ComplexFills_RealLoopRefuses_ComplexLoopKeeps()
        {
            Throws<TypeError>(() => np.interp(Below(), Xp(), Fp(), new Complex(0, 1)), "must be real number, not complex");
            Throws<TypeError>(() => np.interp(Below(), Xp(), Fp(), new Complex(1, 0)), "must be real number, not complex");

            var keeps = np.interp(Below(), Xp(), Fpc(), new Complex(1, 2));
            keeps.dtype.Should().Be(np.complex128);
            keeps.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 2));
            np.interp(Below(), Xp(), Fpc(), 2.5).GetAtIndex<Complex>(0).Should().Be(new Complex(2.5, 0));
            np.interp(Below(), Xp(), Fpc(), true).GetAtIndex<Complex>(0).Should().Be(new Complex(1, 0));
        }

        /// <summary>An array fill converts as NumPy's <c>float(a)</c>: <c>np.array(-1.0)</c> gives <c>[-1.0]</c>; one or
        ///     more dimensions are refused (<c>only 0-dimensional arrays can be converted to Python scalars</c>, size 1 too);
        ///     a 0-d complex ARRAY is refused by both loops (<c>float() argument must be a string or a real number, not
        ///     'complex'</c>: CPython 3.12's complex conversion asks <c>float()</c> of anything but a complex
        ///     scalar).</summary>
        [TestMethod]
        public void Interp_ArrayFills_ConvertAsFloat()
        {
            np.interp(Below(), Xp(), Fp(), np.array(-1.0)).ToArray<double>().Should().Equal(-1.0);
            Throws<TypeError>(() => np.interp(Below(), Xp(), Fp(), np.array(new[] { 7.0 })),
                "only 0-dimensional arrays can be converted to Python scalars");
            Throws<TypeError>(() => np.interp(Below(), Xp(), Fp(), np.array(new[] { 1.0, 2.0 })),
                "only 0-dimensional arrays can be converted to Python scalars");
            Throws<TypeError>(() => np.interp(Below(), Xp(), Fp(), np.array(new Complex(1, 2))),
                "float() argument must be a string or a real number, not 'complex'");

            Throws<TypeError>(() => np.interp(Below(), Xp(), Fpc(), np.array(new Complex(1, 2))),
                "float() argument must be a string or a real number, not 'complex'");
            Throws<TypeError>(() => np.interp(Below(), Xp(), Fpc(), np.array(new[] { new Complex(1, 2) })),
                "only 0-dimensional arrays can be converted to Python scalars");
        }

        /// <summary>A NumPy scalar converts through its <c>__float__</c>: <c>np.float32(2.5)</c> gives 2.5; an element read
        ///     from a complex array (an <c>np.complex128</c> scalar) gives the real loop its REAL part (<c>[1.0]</c>, NumPy's
        ///     ComplexWarning has no channel here) and the complex loop the whole value <c>(1+2j)</c>.</summary>
        [TestMethod]
        public void Interp_NumPyScalarFills()
        {
            np.interp(Below(), Xp(), Fp(), np.float32(2.5)).ToArray<double>().Should().Equal(2.5);

            var complexArray = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            np.interp(Below(), Xp(), Fp(), complexArray[0]).ToArray<double>().Should().Equal(1.0);
            np.interp(Below(), Xp(), Fpc(), complexArray[0]).GetAtIndex<Complex>(0).Should().Be(new Complex(1, 2));
        }

        /// <summary>A C# string is a Python str, which has no <c>__float__</c>: both loops refuse it by type name
        ///     (<c>must be real number, not str</c>), before any parsing.</summary>
        [TestMethod]
        public void Interp_StrFill_IsRefusedByTypeName()
        {
            Throws<TypeError>(() => np.interp(Below(), Xp(), Fp(), "5"), "must be real number, not str");
            Throws<TypeError>(() => np.interp(Below(), Xp(), Fpc(), "5"), "must be real number, not str");
        }

        /// <summary>The fills convert AFTER the operand and length checks, left before right: a bad length reports before
        ///     a complex fill (<c>fp and xp are not of the same length.</c>), a str left before a complex right, a complex
        ///     right after a good left; and a period discards both unread (<c>period=2.0</c> with <c>left='x'</c> is
        ///     <c>[20.0]</c>, and <c>(1+2j)</c> for the complex loop).</summary>
        [TestMethod]
        public void Interp_FillErrors_ComeAfterTheOperandChecks_AndAPeriodDiscardsThem()
        {
            Throws<ValueError>(() => np.interp(Below(), Xp(), np.array(new[] { 1.0, 2.0, 3.0 }), new Complex(1, 0)),
                "fp and xp are not of the same length.");
            Throws<TypeError>(() => np.interp(Below(), Xp(), Fp(), "x", new Complex(1, 0)), "must be real number, not str");
            Throws<TypeError>(() => np.interp(Below(), Xp(), Fp(), 1.0, new Complex(1, 0)), "must be real number, not complex");

            np.interp(Below(), Xp(), Fp(), "x", null, 2.0).ToArray<double>().Should().Equal(20.0);
            np.interp(Below(), Xp(), Fpc(), "x", null, 2.0).GetAtIndex<Complex>(0).Should().Be(new Complex(1, 2));
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 2: np.append(arr, values, axis=None) - one signature; a C# number is NumPy's asarray of the Python number
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>ravel(values)</c> builds the ARRAY of a Python number, in its default dtype:
        ///     <c>append(int32[1, 2], 5)</c> and <c>append(uint8[1, 2], 300)</c> are int64, <c>append(float32[1, 2],
        ///     1.5)</c> float64 (a C# float literal too), <c>append(int64[1, 2, 3], 5)</c> int64; a Python int past int64 is
        ///     uint64, which promotes with int64 to float64 (<c>append(int64[1, 2], 2**64 - 1)</c>), and so does
        ///     <c>append(uint64[1, 2], 5)</c>.</summary>
        [TestMethod]
        public void Append_CSharpNumber_IsNumPysPythonNumber()
        {
            var int32 = np.append(np.array(new[] { 1, 2 }), 5);
            int32.dtype.Should().Be(np.int64);
            int32.ToArray<long>().Should().Equal(1L, 2L, 5L);

            var uint8 = np.append(np.array(new byte[] { 1, 2 }), 300);
            uint8.dtype.Should().Be(np.int64);
            uint8.ToArray<long>().Should().Equal(1L, 2L, 300L);

            var float32 = np.append(np.array(new[] { 1f, 2f }), 1.5f);
            float32.dtype.Should().Be(np.float64);
            float32.ToArray<double>().Should().Equal(1.0, 2.0, 1.5);
            np.append(np.array(new[] { 1f, 2f }), 1.5).dtype.Should().Be(np.float64);

            np.append(np.array(new long[] { 1, 2, 3 }), 5).ToArray<long>().Should().Equal(1L, 2L, 3L, 5L);

            var beyondInt64 = np.append(np.array(new long[] { 1, 2 }), ulong.MaxValue);
            beyondInt64.dtype.Should().Be(np.float64);
            beyondInt64.ToArray<double>().Should().Equal(1.0, 2.0, 18446744073709551615.0);

            var unsigned = np.append(np.array(new ulong[] { 1, 2 }), (ulong)5);
            unsigned.dtype.Should().Be(np.float64);
            unsigned.ToArray<double>().Should().Equal(1.0, 2.0, 5.0);
        }

        /// <summary>The other Python scalars already are NumPy's arrays of them: a bool stays bool
        ///     (<c>append(int8[1, 2], True)</c> is int8 <c>[1, 2, 1]</c>, <c>append(bool[...], True)</c> bool), a str is its
        ///     <c>U</c> array (<c>append(['a', 'b'], 'xyz')</c> is <c>&lt;U3</c>), a Python complex complex128
        ///     (<c>append(int16[1, 2], 1j)</c>), an <c>np.float16</c> float16 (<c>append(int32[1, 2], np.float16(1))</c> is
        ///     float64); and a C# number as <c>arr</c> is the Python number's array too (<c>append(5, int32[1, 2])</c> is
        ///     int64 <c>[5, 1, 2]</c>).</summary>
        [TestMethod]
        public void Append_OtherScalars_AndANumberArr()
        {
            var int8 = np.append(np.array(new sbyte[] { 1, 2 }), true);
            int8.dtype.Should().Be(np.int8);
            int8.ToArray<sbyte>().Should().Equal((sbyte)1, (sbyte)2, (sbyte)1);
            np.append(np.array(new[] { true, false }), true).ToArray<bool>().Should().Equal(true, false, true);

            var text = np.append(np.array(new[] { "a", "b" }), "xyz");
            text.dtype.Should().Be(np.dtype("<U3"));
            ((IList)text.tolist()).Cast<object>().Should().Equal("a", "b", "xyz");

            var complex = np.append(np.array(new short[] { 1, 2 }), new Complex(0, 1));
            complex.dtype.Should().Be(np.complex128);
            complex.ToArray<Complex>().Should().Equal(new Complex(1, 0), new Complex(2, 0), new Complex(0, 1));

            np.append(np.array(new[] { 1, 2 }), (Half)1).dtype.Should().Be(np.float64);

            var numberArr = np.append(5, np.array(new[] { 1, 2 }));
            numberArr.dtype.Should().Be(np.int64);
            numberArr.ToArray<long>().Should().Equal(5L, 1L, 2L);
        }

        /// <summary>Unchanged rows: an array value keeps NumPy's concatenate promotion (<c>append(int64[1, 2, 3],
        ///     [1.5])</c> is float64), and the axis form (<c>append([[1, 2], [3, 4]], [[5, 6]], axis=0)</c>).</summary>
        [TestMethod]
        public void Append_ArrayValues_Unchanged()
        {
            var promoted = np.append(np.array(new long[] { 1, 2, 3 }), new[] { 1.5 });
            promoted.dtype.Should().Be(np.float64);
            promoted.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 1.5);

            var rows = np.append(np.array(new long[,] { { 1, 2 }, { 3, 4 } }), new long[,] { { 5, 6 } }, 0);
            rows.shape.Should().Equal(3L, 2L);
            rows.ToArray<long>().Should().Equal(1L, 2L, 3L, 4L, 5L, 6L);
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 3: np.lib.stride_tricks.as_strided(x, shape=None, strides=None, subok=False, writeable=True)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>Byte strides over x's memory: <c>as_strided(x, (4, 3), (8, 8))</c> is the overlapping windows, <c>(2,
        ///     2), (0, 8)</c> repeats the first row and is WRITEABLE (NumPy protects only broadcast_to's views), and a write
        ///     through the view reaches x (<c>v[1, 1] = 99</c> gives <c>x = [0, 1, 99, 3, 4, 5]</c>).</summary>
        [TestMethod]
        public void AsStrided_WindowsAndZeroStrides()
        {
            var x = Six();
            var windows = np.lib.stride_tricks.as_strided(x, new Shape(4, 3), new long[] { 8, 8 });
            windows.shape.Should().Equal(4L, 3L);
            windows.ToArray<long>().Should().Equal(0L, 1L, 2L, 1L, 2L, 3L, 2L, 3L, 4L, 3L, 4L, 5L);

            var repeated = np.lib.stride_tricks.as_strided(x, (2, 2), new long[] { 0, 8 });
            repeated.ToArray<long>().Should().Equal(0L, 1L, 0L, 1L);
            repeated.flags.writeable.Should().BeTrue();

            windows[1, 1] = 99L;
            x.ToArray<long>().Should().Equal(0L, 1L, 99L, 3L, 4L, 5L);
        }

        /// <summary>The defaults come from x: <c>as_strided(x)</c> is x's values; a shape alone takes the C strides of the
        ///     new shape for a C-contiguous x (<c>shape=(2, 3)</c> is <c>[[0, 1, 2], [3, 4, 5]]</c>); a view starts at its
        ///     own first element (<c>x[::2]</c> with <c>(16,)</c> is <c>[0, 2, 4]</c>, <c>x[::-1]</c> with <c>(-8,)</c> is
        ///     <c>[5, 4, 3]</c>, <c>x[1:]</c> with <c>(8,)</c> is <c>[1, 2]</c>); an empty or 0-d shape works.</summary>
        [TestMethod]
        public void AsStrided_DefaultsAndViews()
        {
            var x = Six();
            np.lib.stride_tricks.as_strided(x).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            var reshaped = np.lib.stride_tricks.as_strided(x, shape: new Shape(2, 3));
            reshaped.shape.Should().Equal(2L, 3L);
            reshaped.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);

            np.lib.stride_tricks.as_strided(x["::2"], new Shape(3), new long[] { 16 }).ToArray<long>().Should().Equal(0L, 2L, 4L);
            np.lib.stride_tricks.as_strided(x["::-1"], new Shape(3), new long[] { -8 }).ToArray<long>().Should().Equal(5L, 4L, 3L);
            np.lib.stride_tricks.as_strided(x["1:"], new Shape(2), new long[] { 8 }).ToArray<long>().Should().Equal(1L, 2L);

            np.lib.stride_tricks.as_strided(x, new Shape(0, 3), new long[] { 8, 8 }).shape.Should().Equal(0L, 3L);
            var scalar = np.lib.stride_tricks.as_strided(x, new Shape(new long[0]), new long[0]);
            scalar.ndim.Should().Be(0);
            scalar.GetInt64().Should().Be(0L);
        }

        /// <summary>NumPy's flags: <c>as_strided(x, (3,), (16,))</c> is neither C nor F contiguous, OWNDATA False, WRITEABLE
        ///     True; <c>writeable=False</c> gives a read-only view; a read-only x (or a broadcast view) stays read-only
        ///     even with <c>writeable=True</c>.</summary>
        [TestMethod]
        public void AsStrided_Flags()
        {
            var x = Six();
            var flags = np.lib.stride_tricks.as_strided(x, new Shape(3), new long[] { 16 }).flags;
            flags.c_contiguous.Should().BeFalse();
            flags.f_contiguous.Should().BeFalse();
            flags.owndata.Should().BeFalse();
            flags.writeable.Should().BeTrue();

            np.lib.stride_tricks.as_strided(x, writeable: false).flags.writeable.Should().BeFalse();

            var readOnly = Six();
            readOnly.setflags(write: false);
            np.lib.stride_tricks.as_strided(readOnly, new Shape(2), new long[] { 8 }, writeable: true).flags.writeable.Should().BeFalse();

            var broadcast = np.broadcast_to(np.array(new long[] { 0, 1, 2 }), new Shape(2, 3));
            np.lib.stride_tricks.as_strided(broadcast).flags.writeable.Should().BeFalse();
        }

        /// <summary>NumPy's errors: a strides tuple of another length is refused first (<c>mismatch in length of strides
        ///     and shape</c>, even beside a negative dimension), then a negative dimension (<c>negative dimensions are not
        ///     allowed</c>).</summary>
        [TestMethod]
        public void AsStrided_NumPyErrors()
        {
            var x = Six();
            Throws<ValueError>(() => np.lib.stride_tricks.as_strided(x, new Shape(-1), new long[] { 8 }),
                "negative dimensions are not allowed");
            Throws<ValueError>(() => np.lib.stride_tricks.as_strided(x, new Shape(2, 3), new long[] { 8 }),
                "mismatch in length of strides and shape");
            Throws<ValueError>(() => np.lib.stride_tricks.as_strided(x, new Shape(2, -1), new long[] { 8 }),
                "mismatch in length of strides and shape");
        }

        /// <summary><c>subok</c> keeps the caller's class: <c>as_strided(matrix)</c> is a plain ndarray,
        ///     <c>subok=True</c> a matrix, and a 1-D view of a matrix is the (1, 6) matrix (the finalize rule).</summary>
        [TestMethod]
        public void AsStrided_Subok()
        {
            var m = np.asmatrix(Six().reshape(2, 3));
            (np.lib.stride_tricks.as_strided(m) is NDMatrix).Should().BeFalse();

            var kept = np.lib.stride_tricks.as_strided(m, subok: true);
            (kept is NDMatrix).Should().BeTrue();
            kept.shape.Should().Equal(2L, 3L);

            var row = np.lib.stride_tricks.as_strided(m, new Shape(6), new long[] { 8 }, subok: true);
            (row is NDMatrix).Should().BeTrue();
            row.shape.Should().Equal(1L, 6L);
        }

        /// <summary>
        ///     NumSharp refuses what NumPy reads unsafely: <c>as_strided(x, strides=(16,))</c> keeps x's 6 items at a 16-byte
        ///     step, which NumPy reads past the array (<c>[0, 2, 4, 99, 7022580766605312, ...]</c>, garbage), and a 3-byte
        ///     stride reads misaligned items (<c>[0, 1099511627776]</c>). NumSharp raises a ValueError for both (its strides
        ///     count whole items, and a view never leaves its buffer) - by design: NumPy's own docs call such calls unsafe.
        /// </summary>
        [TestMethod, Misaligned]
        public void AsStrided_RefusesWhatNumPyReadsUnsafely()
        {
            var x = Six();
            Throws<ValueError>(() => np.lib.stride_tricks.as_strided(x, strides: new long[] { 16 }),
                "as_strided: the view reaches outside the array's buffer (NumSharp refuses the memory NumPy would read past it)");
            Throws<ValueError>(() => np.lib.stride_tricks.as_strided(x, new Shape(2), new long[] { 3 }),
                "as_strided: a stride of 3 bytes is not a multiple of the item size 8 (NumSharp's strides count whole items)");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 4: np.row_stack (NumPy 2.4's deprecated alias of vstack), and both over NDListArray
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>vstack's answers under the old name: <c>row_stack((a, b))</c> is int64 <c>[[1, 2], [3, 4]]</c>,
        ///     <c>dtype=float32</c> float32, <c>casting='no'</c> NumPy's cast refusal (<c>Cannot cast array data from
        ///     dtype('int64') to dtype('float32') according to the rule 'no'</c>), no array <c>need at least one array to
        ///     concatenate</c>.</summary>
        [TestMethod]
        public void RowStack_IsVstack()
        {
            NDArray a = np.array(new long[] { 1, 2 }), b = np.array(new long[] { 3, 4 });
            var stacked = np.row_stack(a, b);
            stacked.dtype.Should().Be(np.int64);
            stacked.shape.Should().Equal(2L, 2L);
            stacked.ToArray<long>().Should().Equal(1L, 2L, 3L, 4L);

            var single = np.row_stack(new[] { a, b }, dtype: np.float32);
            single.dtype.Should().Be(np.float32);
            single.ToArray<float>().Should().Equal(1f, 2f, 3f, 4f);

            Throws<InvalidCastException>(() => np.row_stack(new[] { a, b }, dtype: np.float32, casting: "no"),
                "Cannot cast array data from dtype('int64') to dtype('float32') according to the rule 'no'");
            Throws<ValueError>(() => np.row_stack(), "need at least one array to concatenate");
        }

        /// <summary>
        ///     np.vstack / np.row_stack of np.char.split's object arrays of lists: <c>vstack((s1, s2))</c> is the (2, 2)
        ///     object array <c>[[['a', 'b'], ['c']], [['d'], ['e', 'f', 'g']]]</c> whose elements ARE s1's lists
        ///     (<c>vstack((s1, s2))[0, 0] is s1[0]</c>); empty rows stack to (2, 0); a 2-D part stacks with a 1-D one; no array
        ///     is NumPy's ValueError.
        /// </summary>
        [TestMethod]
        public void VStack_RowStack_OfListArrays()
        {
            using var s1 = np.@char.split(np.array(new[] { "a b", "c" }));
            using var s2 = np.@char.split(np.array(new[] { "d", "e f g" }));

            using var stacked = np.vstack(new[] { s1, s2 });
            stacked.shape.Should().Equal(2L, 2L);
            stacked.item(0, 0).Should().Equal("a", "b");
            stacked.item(0, 1).Should().Equal("c");
            stacked.item(1, 0).Should().Equal("d");
            stacked.item(1, 1).Should().Equal("e", "f", "g");
            // The same list objects: the first array's family is shared, so the element is s1's own list id.
            stacked.Ids.GetInt64(0, 0).Should().Be(s1.Ids.GetInt64(0));

            using var rows = np.row_stack(new[] { s1, s2 });
            rows.shape.Should().Equal(2L, 2L);
            rows.item(1, 1).Should().Equal("e", "f", "g");

            using var empty1 = s1["0:0"];
            using var empty2 = s2["0:0"];
            using var emptyRows = np.vstack(new[] { empty1, empty2 });
            emptyRows.shape.Should().Equal(2L, 0L);

            using var twoD = s1.reshape(1, 2);
            using var mixedRank = np.vstack(new[] { twoD, s2 });
            mixedRank.shape.Should().Equal(2L, 2L);

            Throws<ValueError>(() => np.vstack(new NDListArray[0]), "need at least one array to concatenate");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 5: MaskedArray.base / device / to_device
        // ------------------------------------------------------------------------------------------------------------

        /// <summary><c>ma.array(d).base is d</c> (no copy: the masked array views the caller's array), and not after
        ///     <c>copy=True</c>; <c>m.device</c> is <c>'cpu'</c>, <c>m.to_device('cpu') is m</c>, another device or a stream
        ///     NumPy's ValueErrors.</summary>
        [TestMethod]
        public void MaskedArray_BaseDeviceToDevice()
        {
            var d = np.array(new[] { 1.0, 2.0, 3.0 });
            ReferenceEquals(np.ma.array(d).@base, d).Should().BeTrue();
            ReferenceEquals(np.ma.array(d, copy: true).@base, d).Should().BeFalse();

            var m = np.ma.array(new[] { 1.0, 2.0, 3.0 }, mask: new[] { false, true, false });
            m.@base.ToArray<double>().Should().Equal(1.0, 2.0, 3.0);
            m.device.Should().Be("cpu");
            ReferenceEquals(m.to_device("cpu"), m).Should().BeTrue();
            Throws<ValueError>(() => m.to_device("gpu"), "Unsupported device: gpu. Only 'cpu' is accepted.");
            Throws<ValueError>(() => m.to_device("cpu", 1), "The stream argument in to_device() is not supported");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 6: MaskedArray.getfield / setfield
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>m.getfield(np.float64)</c> is the masked <c>[1, --, 3]</c>; <c>getfield(np.int32, 4)</c> the high words of
        ///     the doubles <c>[1072693248, 1073741824, 1074266112]</c> with the same mask, <c>(np.int32, 0)</c> the low words
        ///     (zeros); the mask is SHARED (masking the field's first element masks the original's), the fill value a copy
        ///     (setting the float32 field's fill to 5 leaves the original's 7.0); NumPy's offset errors.
        /// </summary>
        [TestMethod]
        public void MaskedArray_Getfield()
        {
            var m = np.ma.array(new[] { 1.0, 2.0, 3.0 }, mask: new[] { false, true, false });

            var same = m.getfield(np.float64);
            same.data.ToArray<double>().Should().Equal(1.0, 2.0, 3.0);
            np.ma.getmaskarray(same).ToArray<bool>().Should().Equal(false, true, false);

            var high = m.getfield(np.int32, 4);
            high.dtype.Should().Be(np.int32);
            high.data.ToArray<int>().Should().Equal(1072693248, 1073741824, 1074266112);
            np.ma.getmaskarray(high).ToArray<bool>().Should().Equal(false, true, false);
            m.getfield(np.int32, 0).data.ToArray<int>().Should().Equal(0, 0, 0);

            Throws<ValueError>(() => m.getfield(np.float64, 8), "new type plus offset is larger than original type");
            Throws<ValueError>(() => m.getfield(np.float64, -1), "offset is negative");

            var shared = np.ma.array(new[] { 1.0, 2.0, 3.0 }, mask: new[] { false, true, false });
            var field = shared.getfield(np.float64);
            field[0] = np.ma.masked;
            np.ma.getmaskarray(shared).ToArray<bool>().Should().Equal(true, true, false);
            np.ma.getmaskarray(field).ToArray<bool>().Should().Equal(true, true, false);

            var filled = np.ma.array(new[] { 1.0, 2.0, 3.0 }, mask: new[] { false, true, false }, fill_value: 7.0);
            var narrow = filled.getfield(np.float32);
            narrow.fill_value = 5;
            Convert.ToDouble(narrow.fill_value).Should().Be(5.0);
            Convert.ToDouble(filled.fill_value).Should().Be(7.0);
        }

        /// <summary><c>m.setfield(9.0, np.float64)</c> writes every element's data, the masked ones too (a hard mask
        ///     included), and leaves the mask as it was; an array value is stored element by element; a read-only data is
        ///     NumPy's <c>assignment destination is read-only</c>.</summary>
        [TestMethod]
        public void MaskedArray_Setfield()
        {
            var m = np.ma.array(new[] { 1.0, 2.0, 3.0 }, mask: new[] { false, true, false });
            m.setfield(9.0, np.float64);
            m.data.ToArray<double>().Should().Equal(9.0, 9.0, 9.0);
            np.ma.getmaskarray(m).ToArray<bool>().Should().Equal(false, true, false);

            var mi = np.ma.array(np.array(new long[] { 1, 2, 3 }), mask: new[] { false, true, false });
            mi.setfield(np.array(new long[] { 7, 8, 9 }), np.int64);
            mi.data.ToArray<long>().Should().Equal(7L, 8L, 9L);

            var hard = np.ma.array(np.array(new long[] { 1, 2, 3 }), mask: new[] { false, true, false }, hard_mask: true);
            hard.setfield(9, np.int64);
            hard.data.ToArray<long>().Should().Equal(9L, 9L, 9L);
            np.ma.getmaskarray(hard).ToArray<bool>().Should().Equal(false, true, false);

            var d = np.array(new[] { 1.0, 2.0, 3.0 });
            d.setflags(write: false);
            var readOnly = np.ma.array(d, mask: new[] { false, true, false });
            Throws<ValueError>(() => readOnly.setfield(9.0, np.float64), "assignment destination is read-only");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 7: new np.NDIndex(*shape)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>list(np.ndindex(2, 3))</c> through the class constructor's separate dimensions: the six
        ///     indices in C order; <c>(0, 3)</c> none; no dimension one empty index; a negative dimension NumPy's
        ///     ValueError text.</summary>
        [TestMethod]
        public void NDIndex_TakesTheDimensionsAsArguments()
        {
            new np.NDIndex(2, 3).Select(i => string.Join(",", i)).Should().Equal("0,0", "0,1", "0,2", "1,0", "1,1", "1,2");
            new np.NDIndex(new long[] { 2, 3 }).Select(i => string.Join(",", i)).Should().HaveCount(6);
            new np.NDIndex(0, 3).Should().BeEmpty();
            new np.NDIndex().Select(i => i.Length).Should().Equal(0);
            new Action(() => new np.NDIndex(2, -1).ToList()).Should().Throw<Exception>()
                .Which.Message.Should().Be("negative dimensions are not allowed");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 8: new np.NDIterator(op, flags=None, op_flags=None, op_dtypes=None, order='K', casting='safe', ...)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>np.nditer([a])</c> through the class constructor with NumPy's defaults: the six values of
        ///     <c>arange(6).reshape(2, 3)</c> in memory order, <c>itersize</c> 6, <c>dtypes</c> <c>(int64,)</c>; with
        ///     <c>['multi_index']</c> the first index is <c>(0, 0)</c>.</summary>
        [TestMethod]
        public void NDIterator_HasNumPysDefaults()
        {
            var a = Six().reshape(2, 3);
            using (var it = new np.NDIterator(new[] { a }))
            {
                it.itersize.Should().Be(6);
                it.dtypes.Should().Equal(np.int64);
                var seen = new List<long>();
                do
                {
                    seen.Add(it[0].GetInt64());
                } while (it.iternext());
                seen.Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            }

            using (var it = new np.NDIterator(new[] { a }, new[] { "multi_index" }))
                it.multi_index.Should().Equal(0L, 0L);
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 9: new NDCharArray(shape, itemsize=1, unicode=False, buffer=None, offset=0, strides=None, order='C')
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy's chararray constructor: <c>chararray((2, 3), itemsize=2)</c> is a (2, 3) <c>|S2</c> chararray,
        ///     <c>chararray(3, unicode=True)</c> a (3,) <c>&lt;U1</c> one; over <c>b'abcdef'</c> with itemsize 3
        ///     <c>[b'abc', b'def']</c>, with offset 2 and itemsize 2 <c>[b'cd', b'ef']</c>, with strides (3,) and itemsize 1
        ///     <c>[b'a', b'd']</c>; <c>order='F'</c> is F-contiguous; a negative dimension is NumPy's ValueError.</summary>
        [TestMethod]
        public void NDCharArray_IsNumPysConstructor()
        {
            var bytes = new NDCharArray(new Shape(2, 3), 2);
            bytes.shape.Should().Equal(2L, 3L);
            bytes.dtype.Should().Be(np.dtype("S2"));

            var text = new NDCharArray(3, unicode: true);
            text.shape.Should().Equal(3L);
            text.dtype.Should().Be(np.dtype("<U1"));

            var buffer = new ReadOnlyMemory<byte>(Bytes("abcdef"));
            ((IList)new NDCharArray(new Shape(2), 3, buffer: buffer).tolist()).Cast<byte[]>()
                .Should().BeEquivalentTo(new[] { Bytes("abc"), Bytes("def") }, o => o.WithStrictOrdering());
            ((IList)new NDCharArray(new Shape(2), 2, buffer: buffer, offset: 2).tolist()).Cast<byte[]>()
                .Should().BeEquivalentTo(new[] { Bytes("cd"), Bytes("ef") }, o => o.WithStrictOrdering());
            ((IList)new NDCharArray(new Shape(2), 1, buffer: buffer, strides: new long[] { 3 }).tolist()).Cast<byte[]>()
                .Should().BeEquivalentTo(new[] { Bytes("a"), Bytes("d") }, o => o.WithStrictOrdering());

            new NDCharArray(new Shape(2, 2), 1, order: 'F').flags.f_contiguous.Should().BeTrue();
            new Action(() => new NDCharArray(new Shape(-1))).Should().Throw<Exception>()
                .Which.Message.Should().Be("negative dimensions are not allowed");
        }
    }
}
