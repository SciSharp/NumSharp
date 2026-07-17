using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Generic;
using NumSharp.IO;

namespace NumSharp.UnitTest.Documentation
{
    /// <summary>
    ///     Executable coverage for every code example in <c>docs/website-src/docs/ndarray.md</c>
    ///     (the "NumSharp's ndarray is NDArray!" guide). Each test mirrors one documented snippet and
    ///     asserts the behaviour the doc claims, so the page cannot silently drift from the library.
    ///     Section headers below match the doc's headings; the observed values were captured by running
    ///     the snippets against this branch.
    /// </summary>
    [TestClass]
    public class NDArrayDocExamplesTests
    {
        private string _dir;

        [TestInitialize]
        public void Setup()
        {
            _dir = Path.Combine(Path.GetTempPath(), "ns_ndarray_doc_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_dir);
        }

        [TestCleanup]
        public void Cleanup()
        {
            try { Directory.Delete(_dir, recursive: true); }
            catch (IOException) { /* a leaked handle fails the test that leaked it, not this one */ }
        }

        private string At(string name) => Path.Combine(_dir, name);

        // ── Creating an NDArray ────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Creating_TheUsualWays()
        {
            np.array(new[] { 1, 2, 3 }).shape.Should().Equal(new long[] { 3 });
            np.array(new[] { 1, 2, 3 }).typecode.Should().Be(NPTypeCode.Int32);
            np.array(new int[,] { { 1, 2 }, { 3, 4 } }).shape.Should().Equal(new long[] { 2, 2 });

            np.zeros((3, 4)).shape.Should().Equal(new long[] { 3, 4 });
            np.zeros((3, 4)).typecode.Should().Be(NPTypeCode.Double);
            np.ones(5).shape.Should().Equal(new long[] { 5 });
            np.full((2, 2), 7).ToArray<int>().Should().Equal(7, 7, 7, 7);
            np.full(new Shape(2, 2), 7).shape.Should().Equal(new long[] { 2, 2 });
            np.empty((3, 3)).shape.Should().Equal(new long[] { 3, 3 });
            np.eye(4).shape.Should().Equal(new long[] { 4, 4 });
            np.identity(4).shape.Should().Equal(new long[] { 4, 4 });

            np.arange(10).shape.Should().Equal(new long[] { 10 });
            np.arange(10).typecode.Should().Be(NPTypeCode.Int64, "arange defaults to int64 (NumPy 2.x)");
            np.arange(0, 1, 0.1).typecode.Should().Be(NPTypeCode.Double);
            np.linspace(0, 1, 11).shape.Should().Equal(new long[] { 11 });
            ((double)np.linspace(0, 1, 11)[0]).Should().Be(0.0);
            ((double)np.linspace(0, 1, 11)[10]).Should().Be(1.0);

            np.random.rand(3, 4).shape.Should().Equal(new long[] { 3, 4 });
            np.random.randn(100).shape.Should().Equal(new long[] { 100 });
        }

        [TestMethod]
        public void Creating_ShapeConversionForms_AllProduce3x4()
        {
            np.zeros((3, 4)).shape.Should().Equal(new long[] { 3, 4 });
            np.zeros(new[] { 3, 4 }).shape.Should().Equal(new long[] { 3, 4 });
            np.zeros(new Shape(3, 4)).shape.Should().Equal(new long[] { 3, 4 });
            np.zeros(new Shape(new[] { 3L, 4L })).shape.Should().Equal(new long[] { 3, 4 });

            np.zeros(5).shape.Should().Equal(new long[] { 5 }, "a bare int is the 1-D length overload");
        }

        [TestMethod]
        public void Creating_ScalarsFlowInImplicitly()
        {
            NDArray a = 42;
            NDArray b = 3.14;
            NDArray c = Half.One;
            NDArray d = NDArray.Scalar(100.123m);
            NDArray e = NDArray.Scalar<long>(1);

            a.typecode.Should().Be(NPTypeCode.Int32);
            a.ndim.Should().Be(0);
            b.typecode.Should().Be(NPTypeCode.Double);
            c.typecode.Should().Be(NPTypeCode.Half);
            d.typecode.Should().Be(NPTypeCode.Decimal);
            e.typecode.Should().Be(NPTypeCode.Int64);
        }

        // ── Wrapping Existing Buffers — np.frombuffer ──────────────────────────────────────────────

        [TestMethod]
        public void Frombuffer_ByteBuffer_OffsetAndCount()
        {
            byte[] buffer = new byte[16];
            Buffer.BlockCopy(new[] { 1f, 2f, 3f, 4f }, 0, buffer, 0, 16);

            np.frombuffer(buffer, typeof(float)).ToArray<float>().Should().Equal(1f, 2f, 3f, 4f);
            // offset is in BYTES
            np.frombuffer(buffer, typeof(float), offset: 4).ToArray<float>().Should().Equal(2f, 3f, 4f);
            // count is in ELEMENTS
            np.frombuffer(buffer, typeof(float), count: 2, offset: 4).ToArray<float>().Should().Equal(2f, 3f);
        }

        [TestMethod]
        public void Frombuffer_ReinterpretTypedArrayAsBytes()
        {
            int[] ints = { 1, 2, 3, 4 };
            var bytes = np.frombuffer<int>(ints, typeof(byte));   // little-endian on x86/x64

            bytes.size.Should().Be(16);
            bytes.ToArray<byte>().Take(8).Should().Equal((byte)1, 0, 0, 0, 2, 0, 0, 0);
        }

        [TestMethod]
        public void Frombuffer_SegmentMemoryAndSpan()
        {
            byte[] buffer = new byte[12];
            Buffer.BlockCopy(new[] { 11, 22, 33 }, 0, buffer, 0, 12);

            np.frombuffer(new ArraySegment<byte>(buffer, 0, 12), typeof(int)).ToArray<int>()
              .Should().Equal(11, 22, 33);
            np.frombuffer((Memory<byte>)buffer, typeof(int)).ToArray<int>()
              .Should().Equal(11, 22, 33);

            Span<byte> tmp = stackalloc byte[8];
            BitConverter.TryWriteBytes(tmp, 7);
            BitConverter.TryWriteBytes(tmp.Slice(4), 9);
            ReadOnlySpan<byte> span = tmp;
            np.frombuffer(span, typeof(int)).ToArray<int>().Should().Equal(7, 9);  // spans always copy
        }

        [TestMethod]
        public void Frombuffer_NativeMemory_BorrowedAndOwned()
        {
            // Borrowed — caller frees after the view is done.
            IntPtr borrowed = Marshal.AllocHGlobal(sizeof(float) * 3);
            try
            {
                Marshal.Copy(new[] { 1f, 2f, 3f }, 0, borrowed, 3);
                np.frombuffer(borrowed, sizeof(float) * 3, typeof(float)).ToArray<float>()
                  .Should().Equal(1f, 2f, 3f);
            }
            finally { Marshal.FreeHGlobal(borrowed); }

            // Owned — NumSharp frees on GC via the dispose callback (we must NOT free it too).
            IntPtr owned = Marshal.AllocHGlobal(sizeof(float) * 2);
            Marshal.Copy(new[] { 5f, 6f }, 0, owned, 2);
            var arr1 = np.frombuffer(owned, sizeof(float) * 2, typeof(float),
                dispose: () => Marshal.FreeHGlobal(owned));
            arr1.ToArray<float>().Should().Equal(5f, 6f);
        }

        [TestMethod]
        public void Frombuffer_Endianness_BigVsLittle()
        {
            byte[] networkData = { 0, 0, 0, 1 };
            np.frombuffer(networkData, ">i4").GetInt32(0).Should().Be(1, "big-endian reads 0x00000001");
            np.frombuffer(networkData, "<i4").GetInt32(0).Should().Be(16777216, "little-endian reads 0x01000000");
        }

        // ── Core Properties ────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void CoreProperties_Values()
        {
            var a = np.arange(12).reshape(3, 4);   // int64
            a.shape.Should().Equal(new long[] { 3, 4 });
            a.ndim.Should().Be(2);
            a.size.Should().Be(12);
            a.dtype.Should().Be(typeof(long));
            a.typecode.Should().Be(NPTypeCode.Int64);
            a.T.shape.Should().Equal(new long[] { 4, 3 });
            (a.@base is null).Should().BeFalse("reshape returns a view onto the 1-D arange buffer");

            var b = a["1:, :2"];
            (b.@base is null).Should().BeFalse("b is a view of a");
        }

        // ── Indexing & Slicing ─────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Indexing_SliceNotation_Shapes()
        {
            var a = np.arange(20).reshape(4, 5);

            a[0].shape.Should().Equal(new long[] { 5 }, "integer index reduces a dimension");
            a[-1].GetInt64(0).Should().Be(15L, "last row starts at 15");
            ((long)a[1, 2]).Should().Be(7L);
            a["1:3"].shape.Should().Equal(new long[] { 2, 5 });
            a["1:3, :2"].shape.Should().Equal(new long[] { 2, 2 });
            a["::2"].shape.Should().Equal(new long[] { 2, 5 });
            a["::-1"][0].GetInt64(0).Should().Be(15L, "reversed first axis");
            a["..., -1"].ToArray<long>().Should().Equal(4L, 9L, 14L, 19L);
        }

        [TestMethod]
        public void Indexing_BooleanAndFancy()
        {
            var arr = np.array(new[] { 10, 20, 30, 40, 50 });

            var mask = arr > 20;
            mask.ToArray<bool>().Should().Equal(false, false, true, true, true);
            arr[mask].ToArray<int>().Should().Equal(30, 40, 50);

            var idx = np.array(new[] { 0, 2, 4 });
            arr[idx].ToArray<int>().Should().Equal(10, 30, 50);
        }

        [TestMethod]
        public void Indexing_Assignment()
        {
            var a = np.arange(20).reshape(4, 5);

            a[1, 2] = 99;                    // scalar write
            ((long)a[1, 2]).Should().Be(99L);

            a[0] = np.zeros(5);              // row write (double broadcast into the int64 row)
            a["0"].ToArray<long>().Should().Equal(0L, 0L, 0L, 0L, 0L);

            a[a > 10] = -1;                  // masked write
            a.ToArray<long>().Should().OnlyContain(x => x <= 10);
        }

        // ── Views vs Copies ────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Views_ShareMemory_CopyDoesNot()
        {
            var a = np.arange(10);
            var v = a["2:5"];                // view — shares memory
            v[0] = 999;
            ((long)a[2]).Should().Be(999L, "mutating the view mutates the parent");

            var c = a["2:5"].copy();         // explicit copy — independent
            c[0] = 0;
            ((long)a[2]).Should().Be(999L, "the copy does not touch the parent");
        }

        // ── Operators ──────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Operators_Arithmetic()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 4, 5, 6 });

            (a + b).ToArray<int>().Should().Equal(5, 7, 9);
            (b - a).ToArray<int>().Should().Equal(3, 3, 3);
            (a * b).ToArray<int>().Should().Equal(4, 10, 18);

            var div = np.array(new[] { 1, 2, 3 }) / np.array(new[] { 2, 2, 2 });
            div.typecode.Should().Be(NPTypeCode.Double, "int / int returns a float dtype");
            div.ToArray<double>().Should().Equal(0.5, 1.0, 1.5);

            // result sign follows the divisor (Python/NumPy convention)
            (np.array(new[] { -7, 7 }) % np.array(new[] { 3, 3 })).ToArray<int>().Should().Equal(2, 1);

            (-np.array(new[] { 1, -2 })).ToArray<int>().Should().Equal(-1, 2);

            var p = +a;                      // unary plus returns a copy
            p.ToArray<int>().Should().Equal(1, 2, 3);
            ReferenceEquals(p, a).Should().BeFalse();

            // object on either side works
            (10 - np.array(new[] { 1, 2, 3 })).ToArray<int>().Should().Equal(9, 8, 7);
        }

        [TestMethod]
        public void Operators_BitwiseAndShift()
        {
            (np.array(new[] { 12 }) & np.array(new[] { 10 })).GetInt32(0).Should().Be(8);
            (np.array(new[] { 12 }) | np.array(new[] { 10 })).GetInt32(0).Should().Be(14);
            (np.array(new[] { 12 }) ^ np.array(new[] { 10 })).GetInt32(0).Should().Be(6);
            (~np.array(new[] { 0, -1 })).ToArray<int>().Should().Equal(-1, 0);

            (np.array(new[] { 1, 2, 3 }) << 2).ToArray<int>().Should().Equal(4, 8, 12);
            (np.array(new[] { 4, 8, 16 }) >> 2).ToArray<int>().Should().Equal(1, 2, 4);

            // bool arrays: & is logical AND, | is logical OR
            var t = np.array(new[] { true, true, false });
            var u = np.array(new[] { true, false, false });
            (t & u).ToArray<bool>().Should().Equal(true, false, false);
            (t | u).ToArray<bool>().Should().Equal(true, true, false);
        }

        [TestMethod]
        public void Operators_Comparison()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 3, 2, 1 });

            (a == b).ToArray<bool>().Should().Equal(false, true, false);
            (a != b).ToArray<bool>().Should().Equal(true, false, true);
            (a < b).ToArray<bool>().Should().Equal(true, false, false);
            (a <= b).ToArray<bool>().Should().Equal(true, true, false);
            (a > b).ToArray<bool>().Should().Equal(false, false, true);
            (a >= b).ToArray<bool>().Should().Equal(false, true, true);

            // comparisons with NaN are always false (IEEE 754)
            var n = np.array(new[] { double.NaN });
            ((bool)(n == n)[0]).Should().BeFalse();
        }

        [TestMethod]
        public void Operators_LogicalNot()
        {
            var mask = np.array(new[] { 1, 3 }) > 2;   // [false, true]
            (!mask).ToArray<bool>().Should().Equal(true, false);
        }

        [TestMethod]
        public void Operators_FunctionForms_ForMissingCSharpOperators()
        {
            // a ** b
            np.power(np.array(new[] { 2, 3 }), 2).astype(NPTypeCode.Int64).ToArray<long>()
              .Should().Equal(4L, 9L);
            // a // b
            np.floor_divide(np.array(new[] { 7 }), np.array(new[] { 3 })).astype(NPTypeCode.Int64)
              .ToArray<long>().Should().Equal(2L);
            // a @ b — 1-D dot product
            ((int)np.dot(np.array(new[] { 1, 2, 3 }), np.array(new[] { 4, 5, 6 }))).Should().Be(32);
            // a @ b — 2-D matmul against the identity is a no-op
            var m = np.array(new int[,] { { 5, 6 }, { 7, 8 } });
            np.array_equal(np.matmul(np.array(new int[,] { { 1, 0 }, { 0, 1 } }), m), m).Should().BeTrue();
            // abs(a)
            np.abs(np.array(new[] { -1, 2, -3 })).ToArray<int>().Should().Equal(1, 2, 3);
            // divmod(a, b) == (floor_divide, mod)
            var (q, r) = (np.floor_divide(np.array(new[] { 7 }), np.array(new[] { 3 })),
                          np.array(new[] { 7 }) % np.array(new[] { 3 }));
            q.astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(2L);
            r.ToArray<int>().Should().Equal(1);
        }

        [TestMethod]
        public void Operators_ShiftQuirk_RhsForms()
        {
            var arr = np.array(new[] { 1, 2, 3 });
            (arr << 2).ToArray<int>().Should().Equal(4, 8, 12);           // int RHS

            object rhs = 2;
            (np.array(new[] { 1, 2, 3 }) << rhs).ToArray<int>().Should().Equal(4, 8, 12);   // object RHS

            // 2 << arr is a C# compile error; the function form is the way
            np.left_shift(2, np.array(new[] { 1, 2, 3 })).ToArray<int>().Should().Equal(4, 8, 16);
        }

        [TestMethod]
        public void Operators_CompoundAssignment_IsNotInPlace()
        {
            var x = np.array(new[] { 1, 2, 3 });
            var alias = x;
            x += 10;                         // synthesized as x = x + 10 → new array
            x.ToArray<int>().Should().Equal(11, 12, 13);
            // the alias still sees the original — unlike NumPy
            alias.ToArray<int>().Should().Equal(1, 2, 3);
        }

        // ── Dtype Conversion ───────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void DtypeConversion_AstypeAndScalarCasts()
        {
            var a = np.array(new[] { 1, 2, 3 });
            a.astype(np.float64).typecode.Should().Be(NPTypeCode.Double);
            a.astype(NPTypeCode.Int64).typecode.Should().Be(NPTypeCode.Int64);

            NDArray scalar = NDArray.Scalar(42);
            ((int)scalar).Should().Be(42);
            ((double)scalar).Should().Be(42.0);
            ((Half)scalar).Should().Be((Half)42);
            var cx = (Complex)scalar;
            cx.Real.Should().Be(42);
            cx.Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void DtypeConversion_ComplexToNonComplex_Throws()
        {
            // "Complex → non-complex throws TypeError (mirroring Python's int(1+2j))"
            NDArray c = NDArray.Scalar(new Complex(1, 2));
            Action act = () => { int _ = (int)c; };
            act.Should().Throw<Exception>();
        }

        // ── Scalars (0-d Arrays) ───────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Scalars_ZeroDimensional()
        {
            var s1 = NDArray.Scalar(42);
            NDArray s2 = 42;
            s1.ndim.Should().Be(0);
            s1.size.Should().Be(1);
            ((int)s1).Should().Be(42);
            ((int)s2).Should().Be(42);
        }

        // ── Reading & Writing Elements ─────────────────────────────────────────────────────────────

        [TestMethod]
        public void ReadWrite_FourWays()
        {
            var a = np.arange(12).reshape(3, 4);   // int64

            // reads
            NDArray elem = a[1, 2];
            ((long)elem).Should().Be(6L);
            a.item<long>(6).Should().Be(6L);
            a.item(6).Should().Be(6L);             // boxed long
            a.GetValue<long>(1, 2).Should().Be(6L);
            a.GetAtIndex<long>(6).Should().Be(6L);

            // writes
            a[1, 2] = 99;
            ((long)a[1, 2]).Should().Be(99L);
            a.SetValue(99L, 2, 0);
            a.GetValue<long>(2, 0).Should().Be(99L);
            a.SetAtIndex(99L, 0);
            a.GetAtIndex<long>(0).Should().Be(99L);
        }

        [TestMethod]
        public void ReadWrite_ItemNoArgs_OnSize1()
        {
            NDArray.Scalar(5).item().Should().Be(5);          // 0-d
            np.array(new[] { 7 }).item().Should().Be(7);      // 1-element 1-d

            Action act = () => np.array(new[] { 1, 2 }).item();
            act.Should().Throw<Exception>("item() requires a size-1 array");
        }

        // ── Iterating (foreach) ────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Iterating_ForeachIsAxis0_FlatIsElementwise()
        {
            var m = np.arange(6).reshape(2, 3);

            int rows = 0;
            foreach (NDArray row in m)
            {
                row.shape.Should().Equal(new long[] { 3 }, "each row is a (3,) view");
                rows++;
            }
            rows.Should().Be(2);

            int flat = 0;
            foreach (var _ in m.flat) flat++;
            flat.Should().Be(6);
        }

        // ── Common Patterns ────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Patterns_FlattenReshapeTranspose()
        {
            var a = np.arange(12).reshape(3, 4);

            a.ravel().shape.Should().Equal(new long[] { 12 });
            a.flatten().shape.Should().Equal(new long[] { 12 });

            np.arange(12).reshape(3, 4).shape.Should().Equal(new long[] { 3, 4 });
            np.arange(12).reshape(-1).shape.Should().Equal(new long[] { 12 });
            np.arange(12).reshape(-1, 4).shape.Should().Equal(new long[] { 3, 4 });

            var t = np.arange(24).reshape(2, 3, 4);
            t.T.shape.Should().Equal(new long[] { 4, 3, 2 });
            t.transpose(new[] { 1, 0, 2 }).shape.Should().Equal(new long[] { 3, 2, 4 });
            np.swapaxes(t, 0, 1).shape.Should().Equal(new long[] { 3, 2, 4 });
            np.moveaxis(t, 0, -1).shape.Should().Equal(new long[] { 3, 4, 2 });
        }

        // ── Generic NDArray<T> ─────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Generic_TypedWrappers()
        {
            NDArray<double> a = np.zeros(10).MakeGeneric<double>();
            double first = a[0];
            first.Should().Be(0.0);
            a[0] = 3.14;
            a[0].Should().Be(3.14);

            np.zeros(3).AsGeneric<double>()[0].Should().Be(0.0);          // never allocates; throws on mismatch
            np.array(new[] { 1, 2, 3 }).AsOrMakeGeneric<double>()[0].Should().Be(1.0); // astype when dtype differs
        }

        // ── Saving, Loading, and Interop ───────────────────────────────────────────────────────────

        [TestMethod]
        public void SaveLoad_Npy_RoundTrip()
        {
            var arr = np.arange(10).reshape(2, 5);
            np.save(At("arr.npy"), arr);
            NDArray a = np.load_npy(At("arr.npy"));
            np.array_equal(a, arr).Should().BeTrue();
        }

        [TestMethod]
        public void SaveLoad_Npz_Positional()
        {
            var x = np.arange(3);
            var y = np.arange(4.0);
            np.savez(At("bundle.npz"), x, y);           // positional → "arr_0", "arr_1"

            using NpzFile npz = np.load_npz(At("bundle.npz"));
            npz.Files.Should().Equal("arr_0", "arr_1");
            npz["arr_0"].ToArray<long>().Should().Equal(0L, 1L, 2L);
            npz["arr_1"].ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0);
        }

        [TestMethod]
        public void SaveLoad_Npz_DictionaryKeys()
        {
            var w = np.arange(6).reshape(2, 3);
            var b = np.zeros(3);
            np.savez(At("bundle.npz"), new Dictionary<string, NDArray> { ["w"] = w, ["b"] = b });

            using NpzFile npz = np.load_npz(At("bundle.npz"));
            npz.Files.Should().BeEquivalentTo(new[] { "w", "b" });
            npz["w"].shape.Should().Equal(new long[] { 2, 3 });
        }

        [TestMethod]
        public void SaveLoad_Npz_Compressed_IsSmaller()
        {
            var big = np.zeros(20_000);
            byte[] stored = np.savez(big);
            byte[] deflated = np.savez_compressed(big);
            deflated.Length.Should().BeLessThan(stored.Length, "zeros compress well");
        }

        [TestMethod]
        public void SaveLoad_NpzFile_LazyCachedDisposableAccess()
        {
            np.savez(At("bundle.npz"), new Dictionary<string, NDArray> { ["w"] = np.arange(3), ["b"] = np.zeros(2) });

            using NpzFile npz = np.load_npz(At("bundle.npz"));
            npz["w"].Should().BeSameAs(npz["w.npy"], "\"w\" and \"w.npy\" name the same member");
            npz["w"].Should().BeSameAs(npz["w"], "a member is cached after first access");

            NDArray w2 = npz.f.w;                        // dot access, like NumPy's npz.f
            w2.ToArray<long>().Should().Equal(0L, 1L, 2L);

            npz.Files.Should().BeEquivalentTo(new[] { "w", "b" }, "'.npy' is stripped from Files");
        }

        [TestMethod]
        public void SaveLoad_Load_ReturnsObject_DispatchedOnMagicBytes()
        {
            np.save(At("one.npy"), np.arange(3));
            np.savez(At("many.npz"), np.arange(3));

            object fromNpy = np.load(At("one.npy"));
            object fromNpz = np.load(At("many.npz"));

            fromNpy.Should().BeOfType<NDArray>();
            fromNpz.Should().BeOfType<NpzFile>();
            ((NpzFile)fromNpz).Dispose();
        }

        [TestMethod]
        public void SaveLoad_RawBinary_TofileFromfile()
        {
            var arr = np.array(new[] { 1.0, 2.0, 3.0 });
            arr.tofile(At("data.bin"));
            np.fromfile(At("data.bin"), np.float64).ToArray<double>().Should().Equal(1.0, 2.0, 3.0);
        }

        [TestMethod]
        public void Interop_DotNetArrays()
        {
            var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } });

            int[,] md = (int[,])arr.ToMuliDimArray<int>();
            md[1, 1].Should().Be(4);

            int[][] jag = (int[][])arr.ToJaggedArray<int>();
            jag[1][0].Should().Be(3);

            np.array(md).shape.Should().Equal(new long[] { 2, 2 });
        }

        // ── When Two Arrays Are "The Same" ─────────────────────────────────────────────────────────

        [TestMethod]
        public void Sameness_ComparisonHelpers()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 1, 2, 3 });

            (a == b).ToArray<bool>().Should().Equal(true, true, true);   // element-wise
            np.array_equal(a, b).Should().BeTrue();                      // same shape AND all equal
            np.allclose(np.array(new[] { 1.0, 2.0 }), np.array(new[] { 1.0, 2.0 })).Should().BeTrue();
            ReferenceEquals(a, b).Should().BeFalse();                    // different C# objects

            var v = a["1:"];
            (v.@base is not null).Should().BeTrue("v is a view");
            (a.@base is not null).Should().BeFalse("a owns its data");
        }

        // ── Troubleshooting ────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Troubleshooting_Snippets()
        {
            // "My array changed when I modified a slice!" → copy to detach
            var a = np.arange(10);
            var detached = a["1:3"].copy();
            detached[0] = -5;
            ((long)a[1]).Should().Be(1L, "the copy is independent");

            // "ReadOnlyArrayException writing to my slice" → copy the broadcast view first
            var b = np.arange(3);
            var writable = b.copy();
            writable[0] = 100;
            ((long)writable[0]).Should().Be(100L);

            // "ScalarConversionException on (int)arr" → index/GetAtIndex first
            var arr = np.array(new[] { 10, 20, 30 });
            arr.GetAtIndex<int>(0).Should().Be(10);

            // "10 << arr doesn't compile" → np.left_shift
            np.left_shift(10, np.array(new[] { 1, 2 })).ToArray<int>().Should().Equal(20, 40);

            // "a += 1 didn't update another reference" → write directly for in-place
            var c = np.array(new[] { 1, 2, 3 });
            c["..."] = c + 1;
            c.ToArray<int>().Should().Equal(2, 3, 4);
        }
    }
}
