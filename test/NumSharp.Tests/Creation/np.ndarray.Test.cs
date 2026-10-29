using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     <c>np.ndarray(shape, dtype, buffer, offset, strides, order)</c> against NumPy 2.4.2's <c>ndarray.__new__</c>. Every
    ///     expected value is NumPy's answer for the same call, probed on 2026-10-05 (plan
    ///     <c>docs/plans/np-ndarray-parity.md</c>). The oracle tier <c>ndarray_ctor.jsonl</c> gates the full argument matrix;
    ///     these tests pin what the corpus cannot express: the C# buffer kinds, write-through and lifetime, flag transitions,
    ///     and the documented divergences.
    /// </summary>
    [TestClass]
    public class NdarrayConstructorTests
    {
        /// <summary>The bytes of <c>np.arange(n, dtype=float64)</c>, a buffer whose values are easy to read back.</summary>
        /// <param name="n">The element count.</param>
        /// <returns>A new byte array of <c>8 * n</c> bytes.</returns>
        private static byte[] Float64Bytes(int n)
        {
            var bytes = new byte[n * sizeof(double)];
            for (int i = 0; i < n; i++)
                BitConverter.GetBytes((double)i).CopyTo(bytes, i * sizeof(double));
            return bytes;
        }

        // ---- no buffer ------------------------------------------------------------------------------------------------

        /// <summary>NumPy's defaults are float64 and C order (NumSharp's were float32 and 'F'): <c>np.ndarray((2, 3))</c>
        ///     is float64 with strides <c>(24, 8)</c>, flags <c>C-OWA</c>.</summary>
        [TestMethod]
        public void Defaults_AreFloat64AndCOrder()
        {
            using var a = np.ndarray((2, 3));

            a.dtype.Should().Be(np.float64);
            a.shape.Should().Equal(2L, 3L);
            a.strides.Should().Equal(24L, 8L);
            a.flags.c_contiguous.Should().BeTrue();
            a.flags.f_contiguous.Should().BeFalse();
            a.flags.owndata.Should().BeTrue();
            a.flags.writeable.Should().BeTrue();
            a.flags.aligned.Should().BeTrue();
        }

        /// <summary>Without a buffer <c>order='F'</c> is honored (NumSharp ignored it): <c>(2, 3)</c> has strides
        ///     <c>(8, 16)</c> and <c>(2, 3, 4)</c> has <c>(8, 16, 48)</c>.</summary>
        [TestMethod]
        public void OrderF_WithoutBuffer_HasFStrides()
        {
            using var a = np.ndarray((2, 3), order: 'F');
            using var b = np.ndarray((2, 3, 4), order: 'F');

            a.strides.Should().Equal(8L, 16L);
            a.flags.f_contiguous.Should().BeTrue();
            a.flags.c_contiguous.Should().BeFalse();
            b.strides.Should().Equal(8L, 16L, 48L);
        }

        /// <summary><c>'A'</c> and <c>'K'</c> mean C here and lowercase is accepted (NumPy's <c>is_f_order</c> asks only
        ///     whether the order is Fortran).</summary>
        [TestMethod]
        public void Order_AandK_AreC_AndLowercaseIsAccepted()
        {
            foreach (char order in "AKcak")
            {
                using var a = np.ndarray((2, 3), order: order);
                a.strides.Should().Equal(new[] { 24L, 8L }, $"order '{order}' is the C layout");
            }

            using var f = np.ndarray((2, 3), order: 'f');
            f.strides.Should().Equal(8L, 16L);
        }

        /// <summary>An invalid order is NumPy's ValueError, raised even without a buffer (NumSharp ignored it).</summary>
        [TestMethod]
        public void Order_Invalid_RaisesNumPysValueError()
        {
            Action call = () => np.ndarray((2, 3), order: 'Q');
            call.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");
        }

        /// <summary>An array NumPy allocates empty has all-zero strides: <c>(0,)</c> → <c>(0,)</c>, <c>(2, 0, 3)</c> →
        ///     <c>(0, 0, 0)</c>, <c>(3, 0)</c> in C or F order → <c>(0, 0)</c>.</summary>
        [TestMethod]
        public void EmptyAllocation_HasAllZeroStrides()
        {
            using var a = np.ndarray(new Shape(0L));
            using var b = np.ndarray((2, 0, 3));
            using var c = np.ndarray((3, 0));
            using var d = np.ndarray((3, 0), order: 'F');

            a.strides.Should().Equal(0L);
            b.strides.Should().Equal(0L, 0L, 0L);
            c.strides.Should().Equal(0L, 0L);
            d.strides.Should().Equal(0L, 0L);
            b.flags.c_contiguous.Should().BeTrue();
            b.flags.f_contiguous.Should().BeTrue();
        }

        /// <summary><c>default(Shape)</c> is NumPy's <c>None</c>, refused by the shape converter.</summary>
        [TestMethod]
        public void DefaultShape_RaisesTypeError()
        {
            Action call = () => np.ndarray(default(Shape));
            call.Should().Throw<TypeError>().WithMessage("Use () not None as shape arguments");
        }

        /// <summary>Explicit strides without a buffer: the array owns its memory, and a stride of 0 keeps it writeable
        ///     (NumPy's <c>(2, 3)</c> with strides <c>(0, 8)</c> is <c>---WA</c> plus OWNDATA). Writing one element writes
        ///     every element that shares its memory.</summary>
        [TestMethod]
        public void ExplicitStrides_WithoutBuffer_OwnTheirMemoryAndZeroStridesStayWriteable()
        {
            using var a = np.ndarray((2, 3), np.int64, strides: new long[] { 0, 8 });

            a.strides.Should().Equal(0L, 8L);
            a.flags.owndata.Should().BeTrue();
            a.flags.writeable.Should().BeTrue();
            a.flags.c_contiguous.Should().BeFalse();

            a.fill(0L);
            a[0, 1] = 7L;
            a.GetInt64(1, 1).Should().Be(7L, "both rows are the same memory");
        }

        /// <summary>The stride checks and their texts: the length, then <c>PyArray_CheckStrides</c> against the array's
        ///     own size (no negative stride without a buffer).</summary>
        [TestMethod]
        public void Strides_WithoutBuffer_AreChecked()
        {
            Action wrongLength = () => np.ndarray((2, 3), strides: new long[] { 8 });
            Action tooFar = () => np.ndarray((2, 3), strides: new long[] { 48, 8 });
            Action negative = () => np.ndarray(new Shape(6L), strides: new long[] { -8 });

            wrongLength.Should().Throw<ValueError>().WithMessage("strides, if given, must be the same length as shape");
            tooFar.Should().Throw<ValueError>().WithMessage("strides is incompatible with shape of requested array and size of buffer");
            negative.Should().Throw<ValueError>().WithMessage("strides is incompatible with shape of requested array and size of buffer");
        }

        /// <summary>The dimension checks of <c>PyArray_NewFromDescr_int</c>, with and without a buffer.</summary>
        [TestMethod]
        public void Dimensions_AreChecked()
        {
            Action negative = () => np.ndarray((2, -3));
            Action tooBig = () => np.ndarray(new Shape(1L << 62, 1L << 62));
            Action negativeOverBuffer = () => np.ndarray((2, -1), buffer: new byte[48]);

            negative.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
            tooBig.Should().Throw<ValueError>().WithMessage("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.");
            negativeOverBuffer.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
        }

        // ---- a buffer ---------------------------------------------------------------------------------------------------

        /// <summary>A <c>byte[]</c> (NumPy's <c>bytearray</c>) is viewed, not copied: <c>[[0, 1, 2], [3, 4, 5]]</c> with
        ///     strides <c>(24, 8)</c> and flags <c>C--WA</c> (no OWNDATA, a base), and a write reaches the buffer.</summary>
        [TestMethod]
        public void ByteArrayBuffer_IsAViewThatWritesThrough()
        {
            var bytes = Float64Bytes(6);
            using var a = np.ndarray((2, 3), buffer: bytes);

            a.ToArray<double>().Should().Equal(0, 1, 2, 3, 4, 5);
            a.strides.Should().Equal(24L, 8L);
            a.flags.c_contiguous.Should().BeTrue();
            a.flags.owndata.Should().BeFalse();
            a.flags.writeable.Should().BeTrue();
            a.Storage.BaseStorage.Should().NotBeNull("NumPy's base is the buffer object");

            a[0, 1] = 42.0;
            BitConverter.ToDouble(bytes, 8).Should().Be(42.0, "the result shares the buffer's memory");
        }

        /// <summary>The layout options over a buffer: F order (<c>[[0, 2, 4], [1, 3, 5]]</c>), a byte offset
        ///     (<c>offset=16</c> as <c>(2, 2)</c> → <c>[[2, 3], [4, 5]]</c>), and a buffer larger than needed (a prefix
        ///     is read).</summary>
        [TestMethod]
        public void ByteArrayBuffer_OrderOffsetAndALargerBuffer()
        {
            using var f = np.ndarray((2, 3), buffer: Float64Bytes(6), order: 'F');
            using var off = np.ndarray((2, 2), buffer: Float64Bytes(6), offset: 16);
            using var prefix = np.ndarray(new Shape(2L), buffer: Float64Bytes(6));

            f.ToArray<double>().Should().Equal(0, 2, 4, 1, 3, 5);
            f.strides.Should().Equal(8L, 16L);
            off.ToArray<double>().Should().Equal(2, 3, 4, 5);
            prefix.ToArray<double>().Should().Equal(0, 1);
        }

        /// <summary>A multi-dimensional C# array is a buffer of its elements in memory (row-major) order:
        ///     <c>np.ndarray((3, 2), buffer=[[1, 2, 3], [4, 5, 6]])</c> is <c>[[1, 2], [3, 4], [5, 6]]</c>.</summary>
        [TestMethod]
        public void MultiDimensionalArray_IsReadInMemoryOrder()
        {
            var m = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            using var a = np.ndarray((3, 2), buffer: m);

            a.ToArray<double>().Should().Equal(1, 2, 3, 4, 5, 6);
            a.GetDouble(2, 0).Should().Be(5);
        }

        /// <summary>The dtype reads the buffer's BYTES (NumSharp used the buffer's element type): <c>int32</c> over
        ///     <c>[1.0, 2.0]</c> is <c>[0, 1072693248, 0, 1073741824]</c>.</summary>
        [TestMethod]
        public void Dtype_ReinterpretsTheBufferBytes()
        {
            using var a = np.ndarray(new Shape(4L), np.int32, new double[] { 1.0, 2.0 });

            a.dtype.Should().Be(np.int32);
            a.ToArray<int>().Should().Equal(0, 1072693248, 0, 1073741824);
        }

        /// <summary>The NumSharp-only dtypes read bytes too: <c>decimal</c> over a <c>decimal[]</c>, <c>char</c> over the
        ///     UTF-16 code units of a <c>char[]</c> read as bytes.</summary>
        [TestMethod]
        public void NumSharpOnlyDtypes_ReadTheBufferBytes()
        {
            using var d = np.ndarray(new Shape(2L), NPTypeCode.Decimal, new decimal[] { 1.5m, -2m });
            using var c = np.ndarray(new Shape(2L), NPTypeCode.Char, new byte[] { (byte)'h', 0, (byte)'i', 0 });

            d.ToArray<decimal>().Should().Equal(1.5m, -2m);
            c.ToArray<char>().Should().Equal('h', 'i');
        }

        /// <summary>The one-dimensional shape <c>-1</c> takes every whole element after the offset: 48 bytes at offset 8
        ///     are 5 float64.</summary>
        [TestMethod]
        public void ShapeMinusOne_TakesEveryWholeElementAfterTheOffset()
        {
            using var a = np.ndarray(-1, buffer: new byte[48], offset: 8);
            using var floor = np.ndarray(-1, np.int16, new byte[7]);

            a.shape.Should().Equal(5L);
            floor.shape.Should().Equal(3L);
        }

        /// <summary>Negative strides over a buffer: <c>(6,)</c> with stride <c>-8</c> from offset 40 reads
        ///     <c>arange(6.)</c> backwards.</summary>
        [TestMethod]
        public void ReversedStrides_FromAnOffset_ReadBackwards()
        {
            using var src = np.arange(6.0);
            using var a = np.ndarray(new Shape(6L), buffer: src, offset: 40, strides: new long[] { -8 });

            a.ToArray<double>().Should().Equal(5, 4, 3, 2, 1, 0);
            a.strides.Should().Equal(-8L);
        }

        /// <summary>A stride of 0 over a buffer aliases elements and the array stays writeable (NumPy <c>---WA</c>); a write
        ///     to <c>[0, 0]</c> is seen at <c>[1, 0]</c>.</summary>
        [TestMethod]
        public void ZeroStrides_OverABuffer_AliasAndStayWriteable()
        {
            using var a = np.ndarray((2, 3), buffer: Float64Bytes(3), strides: new long[] { 0, 8 });

            a.ToArray<double>().Should().Equal(0, 1, 2, 0, 1, 2);
            a.flags.writeable.Should().BeTrue();
            a.flags.c_contiguous.Should().BeFalse();

            a[0, 0] = 9.0;
            a.GetDouble(1, 0).Should().Be(9.0);
        }

        // ---- ALIGNED ----------------------------------------------------------------------------------------------------

        /// <summary>
        ///     A byte offset that splits the alignment makes the data misaligned, which NumPy reports on the array AND on
        ///     every view of it; a copy is aligned again, <c>setflags(align=True)</c> raises, and <c>np.require(…, 'A')</c>
        ///     copies. NumPy: <c>CF-W-</c> for the array, <c>[1:]</c> and <c>.T</c>; <c>CFOWA</c> for the copy.
        /// </summary>
        [TestMethod]
        public void UnalignedOffset_IsNotAligned_OnEveryView()
        {
            using var a = np.ndarray(new Shape(3L), buffer: new byte[32], offset: 3);
            using var tail = a["1:"];
            using var t = a.T;
            using var copy = a.copy();

            a.flags.aligned.Should().BeFalse();
            tail.flags.aligned.Should().BeFalse();
            t.flags.aligned.Should().BeFalse();
            copy.flags.aligned.Should().BeTrue();

            Action align = () => a.setflags(align: true);
            align.Should().Throw<ValueError>().WithMessage("cannot set aligned flag of mis-aligned array to True");
            a.flags.aligned.Should().BeFalse();

            using var required = np.require(a, requirements: new[] { "A" });
            required.flags.aligned.Should().BeTrue();
            required.flags.owndata.Should().BeTrue("require copies an array that is not aligned");
        }

        /// <summary>An empty array and a 1-byte dtype are aligned at any offset (NumPy: a zero-size array is aligned;
        ///     uint8 needs no alignment).</summary>
        [TestMethod]
        public void EmptyOrOneByteArrays_AreAlignedAtAnyOffset()
        {
            using var empty = np.ndarray(new Shape(0L), buffer: new byte[16], offset: 3);
            using var bytes = np.ndarray(new Shape(4L), np.uint8, new byte[16], offset: 3);

            empty.flags.aligned.Should().BeTrue();
            bytes.flags.aligned.Should().BeTrue();
        }

        /// <summary><c>np.require(a, requirements='A')</c> copies an array whose ALIGNED flag was cleared by
        ///     <c>setflags(align=False)</c>, as NumPy's <c>arr.flags[prop]</c> test does.</summary>
        [TestMethod]
        public void RequireAligned_CopiesAnAlignClearedArray()
        {
            using var a = np.arange(4.0);
            a.setflags(align: false);

            using var r = np.require(a, requirements: new[] { "A" });

            r.flags.aligned.Should().BeTrue();
            np.shares_memory(a, r).Should().BeFalse();
        }

        // ---- writeability -----------------------------------------------------------------------------------------------

        /// <summary>A <see cref="ReadOnlyMemory{T}"/> (NumPy's <c>bytes</c>) gives a read-only array (NumPy <c>CF--A</c>)
        ///     that <c>setflags(write=True)</c> cannot make writeable.</summary>
        [TestMethod]
        public void ReadOnlyMemoryBuffer_IsReadOnlyForGood()
        {
            using var a = np.ndarray(new Shape(2L), buffer: new ReadOnlyMemory<byte>(new byte[16]));

            a.flags.writeable.Should().BeFalse();
            Action write = () => a.setflags(write: true);
            write.Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");
        }

        /// <summary>A read-only array as the buffer gives a read-only result whose base is that array (NumPy
        ///     <c>CF--A</c>, <c>base is ro</c>), and <c>setflags(write=True)</c> is refused there too.</summary>
        [TestMethod]
        public void ReadOnlyNDArrayBuffer_GivesAReadOnlyResult()
        {
            using var ro = np.arange(6.0);
            ro.setflags(write: false);
            using var a = np.ndarray(new Shape(6L), buffer: ro);

            a.flags.writeable.Should().BeFalse();
            a.Storage.BaseStorage.Should().BeSameAs(ro.Storage);
            Action write = () => a.setflags(write: true);
            write.Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");
        }

        // ---- NumSharp arrays as buffers ---------------------------------------------------------------------------------

        /// <summary>The base collapses to the first owner, as <c>PyArray_SetBaseObject</c> does: a buffer that is a view
        ///     of <c>a</c> gives a result whose base is <c>a</c>.</summary>
        [TestMethod]
        public void NDArrayBuffer_BaseCollapsesToTheOwner()
        {
            using var a = np.arange(6.0);
            using var v = a["2:"];
            using var r = np.ndarray(new Shape(4L), buffer: v);

            r.Storage.BaseStorage.Should().BeSameAs(a.Storage);
            r.ToArray<double>().Should().Equal(2, 3, 4, 5);
        }

        /// <summary>The result keeps the buffer array's memory alive: disposing the buffer array while the result lives
        ///     does not free the memory under it.</summary>
        [TestMethod]
        public void NDArrayBuffer_KeepsTheMemoryAliveAfterTheBufferIsDisposed()
        {
            var src = np.arange(6.0);
            using var r = np.ndarray((2, 3), buffer: src);
            src.Dispose();

            // A same-size allocation would reuse a freed buffer; the result must still read its own values.
            using var other = np.full(new Shape(6L), -1.0);
            r.ToArray<double>().Should().Equal(0, 1, 2, 3, 4, 5);
        }

        /// <summary>A <see cref="np.MemoryView"/> (<c>ndarray.data</c>) of an F-contiguous array is read in memory order:
        ///     <c>[0, 3, 1, 4, 2, 5]</c> for <c>asfortranarray(arange(6.).reshape(2, 3))</c>.</summary>
        [TestMethod]
        public void MemoryViewOfAnFArray_IsReadInMemoryOrder()
        {
            using var f = np.asfortranarray(np.arange(6.0).reshape(2, 3));
            using var a = np.ndarray(new Shape(6L), buffer: f.data);

            a.ToArray<double>().Should().Equal(0, 3, 1, 4, 2, 5);
        }

        /// <summary>A non-contiguous array is not a buffer: NumPy's ValueError for the array itself, CPython's BufferError
        ///     for its memoryview.</summary>
        [TestMethod]
        public void NonContiguousBuffers_AreRefused()
        {
            using var stepped = np.arange(6.0)["::2"];
            Action array = () => np.ndarray(new Shape(3L), buffer: stepped);
            Action view = () => np.ndarray(new Shape(3L), buffer: stepped.data);

            array.Should().Throw<ValueError>().WithMessage("ndarray is not contiguous");
            view.Should().Throw<BufferError>().WithMessage("memoryview: underlying buffer is not contiguous");
        }

        // ---- other C# buffers -------------------------------------------------------------------------------------------

        /// <summary><see cref="ArraySegment{T}"/>, <see cref="Memory{T}"/> and <see cref="IArraySlice"/> are buffers.</summary>
        [TestMethod]
        public void SegmentMemoryAndSlice_AreBuffers()
        {
            var bytes = Float64Bytes(6);
            using var seg = np.ndarray(new Shape(2L), buffer: new ArraySegment<byte>(bytes, 16, 16));
            using var mem = np.ndarray(new Shape(3L), buffer: new Memory<byte>(bytes, 24, 24));
            using var src = np.arange(4.0);
            using var slice = np.ndarray(new Shape(2L), np.float64, src.Storage.InternalArray, offset: 16);

            seg.ToArray<double>().Should().Equal(2, 3);
            mem.ToArray<double>().Should().Equal(3, 4, 5);
            slice.ToArray<double>().Should().Equal(2, 3);
            mem.flags.writeable.Should().BeTrue();
        }

        /// <summary>The C# values NumSharp treats as NumPy scalars (Half, char, decimal) export a read-only copy of their
        ///     bytes, as NumPy's scalars do: <c>np.float16(1.5)</c> as uint8 is <c>[0, 62]</c>, <c>np.uint16(97)</c> is
        ///     <c>[97, 0]</c>, and the result is <c>CF--A</c>.</summary>
        [TestMethod]
        public void NumPyScalarValues_AreReadOnlyBuffers()
        {
            using var h = np.ndarray(new Shape(1L), np.float16, (Half)1.5);
            using var hb = np.ndarray(new Shape(2L), np.uint8, (Half)1.5);
            using var cb = np.ndarray(new Shape(2L), np.uint8, 'a');
            using var d = np.ndarray(new Shape(1L), NPTypeCode.Decimal, 2.25m);

            h.GetValue<Half>(0).Should().Be((Half)1.5);
            h.flags.writeable.Should().BeFalse();
            hb.ToArray<byte>().Should().Equal(0, 62);
            cb.ToArray<byte>().Should().Equal(97, 0);
            d.GetValue<decimal>(0).Should().Be(2.25m);
        }

        /// <summary>Anything else is NumPy's TypeError with the Python type name of the C# value.</summary>
        [TestMethod]
        public void NotABuffer_RaisesTypeErrorWithThePythonTypeName()
        {
            var cases = new (object buffer, string name)[]
            {
                ((1, 2), "tuple"), (new Dictionary<int, int> { [1] = 2 }, "dict"), (new HashSet<int> { 1 }, "set"),
                (1.5, "float"), (true, "bool"), (new Complex(0, 1), "complex"), (7, "int"), (7L, "int"),
                ("ab", "str"), (new object[] { 1 }, "list"), (new List<double> { 1 }, "list"), (new[] { "a" }, "list"),
                (new double[][] { new double[] { 1 } }, "list"),
            };
            foreach (var (buffer, name) in cases)
            {
                Action call = () => np.ndarray(new Shape(1L), buffer: buffer);
                call.Should().Throw<TypeError>().WithMessage($"a bytes-like object is required, not '{name}'");
            }
        }

        /// <summary>The buffer is checked before the order (NumPy's converter order): a list with a bad order reports the
        ///     list.</summary>
        [TestMethod]
        public void BufferIsCheckedBeforeOrder()
        {
            Action call = () => np.ndarray((2, 3), buffer: new object[] { 1 }, order: 'Q');
            call.Should().Throw<TypeError>().WithMessage("a bytes-like object is required, not 'list'");
        }

        // ---- lifetime -------------------------------------------------------------------------------------------------

        /// <summary>A managed array stays pinned only while an array over it lives: once the result is disposed the
        ///     array is collectable (a pin is a strong root, so a leaked pin keeps it alive).</summary>
        [TestMethod]
        public void ManagedArrayPin_IsReleasedWithTheResult()
        {
            GcQuiescence.WaitCollected(WrapAndDispose()).Should().BeTrue("disposing the result frees the pin");

            [MethodImpl(MethodImplOptions.NoInlining)]
            static WeakReference WrapAndDispose()
            {
                var bytes = new byte[64];
                using (var a = np.ndarray(new Shape(8L), buffer: bytes))
                    a[0] = 1.0;
                return new WeakReference(bytes);
            }
        }

        /// <summary>A result that is never disposed releases the pin when it is collected.</summary>
        [TestMethod]
        public void ManagedArrayPin_IsReleasedWhenTheResultIsCollected()
        {
            GcQuiescence.WaitCollected(WrapAndDrop()).Should().BeTrue("the finalizers release the pin");

            [MethodImpl(MethodImplOptions.NoInlining)]
            static WeakReference WrapAndDrop()
            {
                var bytes = new byte[64];
                var a = np.ndarray(new Shape(8L), buffer: bytes);
                a[0] = 1.0;
                return new WeakReference(bytes);
            }
        }

        /// <summary>A call that fails after the buffer was pinned (here: a bad order, checked after the buffer) frees the
        ///     pin at once.</summary>
        [TestMethod]
        public void ManagedArrayPin_IsReleasedWhenTheCallFails()
        {
            GcQuiescence.WaitCollected(FailAfterPinning()).Should().BeTrue("the failed call releases its pin");

            [MethodImpl(MethodImplOptions.NoInlining)]
            static WeakReference FailAfterPinning()
            {
                var bytes = new byte[64];
                Action call = () => np.ndarray(new Shape(8L), buffer: bytes, order: 'Q');
                call.Should().Throw<ValueError>();
                return new WeakReference(bytes);
            }
        }

        // ---- the old signature ------------------------------------------------------------------------------------------

        /// <summary>
        ///     The old fourth positional parameter was the order, and a <c>char</c> converts silently to the new
        ///     <c>long offset</c>. An overload taking a <c>char</c> fourth argument is marked obsolete AS AN ERROR, so
        ///     <c>np.ndarray(shape, dtype, buffer, 'F')</c> no longer compiles; a named <c>order:</c> does.
        /// </summary>
        [TestMethod]
        public void OldPositionalOrder_DoesNotCompile()
        {
            var guard = typeof(np).GetMethod(nameof(np.ndarray), new[] { typeof(Shape), typeof(DType), typeof(object), typeof(char) });

            guard.Should().NotBeNull();
            var obsolete = guard!.GetCustomAttribute<ObsoleteAttribute>();
            obsolete.Should().NotBeNull();
            obsolete!.IsError.Should().BeTrue();
        }

        // ---- documented divergences -------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy lays <c>np.ndarray((2,), buffer=bytearray(48), offset=-8)</c> out from 8 bytes BEFORE the buffer (its
        ///     checks do not cover a negative offset). NumSharp checks the exact byte range and refuses with
        ///     <c>np.frombuffer</c>'s offset text.
        /// </summary>
        [TestMethod, Misaligned]
        public void NegativeOffset_IsRefused()
        {
            Action call = () => np.ndarray(new Shape(2L), buffer: new byte[48], offset: -8);
            call.Should().Throw<ValueError>().WithMessage("offset must be non-negative and no greater than buffer length (48)");
        }

        /// <summary>
        ///     NumPy checks the strides while the shape is still <c>-1</c>, so <c>np.ndarray(-1, buffer=bytearray(48),
        ///     strides=(-8,))</c> reads 40 bytes before the buffer; an offset near 2**63 wraps NumPy's size check; and strides
        ///     over an empty buffer are checked against the array's own size. NumSharp refuses all three.
        /// </summary>
        [TestMethod, Misaligned]
        public void LayoutsOutsideTheBuffer_AreRefused()
        {
            Action minusOne = () => np.ndarray(-1, buffer: new byte[48], strides: new long[] { -8 });
            Action hugeOffset = () => np.ndarray(new Shape(2L), buffer: new byte[48], offset: long.MaxValue);
            Action emptyBuffer = () => np.ndarray(new Shape(2L), buffer: Array.Empty<byte>(), strides: new long[] { 8 });

            minusOne.Should().Throw<ValueError>().WithMessage("strides is incompatible with shape of requested array and size of buffer");
            hugeOffset.Should().Throw<TypeError>().WithMessage("buffer is too small for requested array");
            emptyBuffer.Should().Throw<ValueError>().WithMessage("strides is incompatible with shape of requested array and size of buffer");
        }

        /// <summary>
        ///     NumPy accepts byte strides that split an element (<c>int32</c> with strides <c>(6, 2)</c>, an unaligned array).
        ///     NumSharp stores strides in whole elements and cannot represent it.
        /// </summary>
        [TestMethod, Misaligned]
        public void StrideThatSplitsAnElement_IsNotSupported()
        {
            Action call = () => np.ndarray((2, 3), np.int32, new byte[24], strides: new long[] { 6, 2 });
            call.Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     A stride that addresses no element (the axis of extent 1 in <c>(2, 1, 3)</c>) may split an element; NumSharp
        ///     accepts it and stores the truncated whole-element stride, so <c>strides</c> reports <c>(24, 0, 8)</c> where
        ///     NumPy reports <c>(24, 4, 8)</c>. Every element address is the same.
        /// </summary>
        [TestMethod, Misaligned]
        public void UnusedStrideThatSplitsAnElement_IsReportedTruncated()
        {
            using var a = np.ndarray((2, 1, 3), buffer: Float64Bytes(6), strides: new long[] { 24, 4, 8 });

            a.strides.Should().Equal(24L, 0L, 8L);
            a.ToArray<double>().Should().Equal(0, 1, 2, 3, 4, 5);
        }

        /// <summary>
        ///     A non-native dtype over a buffer: NumPy views the bytes as big-endian (<c>dtype('&gt;i4')</c>, no OWNDATA).
        ///     NumSharp has no big-endian storage, so it DECODES the bytes into a new native array: the same values
        ///     <c>[1, 2]</c>, dtype <c>&lt;i4</c>, OWNDATA.
        /// </summary>
        [TestMethod, Misaligned]
        public void NonNativeDtype_OverABuffer_IsDecodedIntoANewArray()
        {
            using var a = np.ndarray(new Shape(2L), np.dtype(">i4"), new byte[] { 0, 0, 0, 1, 0, 0, 0, 2 });

            a.ToArray<int>().Should().Equal(1, 2);
            a.dtype.str.Should().Be("<i4");
            a.flags.owndata.Should().BeTrue();
        }
    }
}
