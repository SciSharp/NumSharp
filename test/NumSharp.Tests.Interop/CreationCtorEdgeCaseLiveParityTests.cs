using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     <b>Edge-case and rare-scenario live parity (embedded CPython numpy 2.4.2) for the seven
    ///     AuditV2 Tier-1 creation/ctor fixes.</b> The boundary and degenerate inputs the happy-path
    ///     <see cref="CreationCtorLiveParityTests"/> does not reach — size-1 / empty / higher-rank F
    ///     layouts across every dtype; negative-, fractional-step and single-arg <c>arange(bool)</c>;
    ///     negative / oversized / empty-input <c>ndmin</c>; <c>frombuffer</c> <c>count</c>/<c>offset</c>,
    ///     every supported dtype, big-endian byte-swap, and the buffer-size error; and the
    ///     <c>find_common_type</c> / <c>_can_coerce_all</c> promotion across start offsets. Every case is
    ///     compared byte-for-byte against a REAL numpy running in-process, and the deliberate divergences
    ///     (big-endian dtype, buffer-size <c>ArgumentException</c>) are pinned against numpy's live behaviour.
    /// </summary>
    [TestClass]
    public class CreationCtorEdgeCaseLiveParityTests : InteropTestBase
    {
        /// <summary><c>str(e)</c> of whatever exception the one-line numpy statement raises ("" if none).</summary>
        private string NumpyErrorText(string oneLineStmt)
        {
            PyExec("__e=''\ntry:\n    " + oneLineStmt + "\nexcept Exception as e:\n    __e=str(e)");
            return PyStr("__e");
        }

        /// <summary>Invariant python-literal spelling of a double (so 1.5 stays "1.5", not the current culture's "1,5").</summary>
        private static string Py(double d) => d.ToString("R", CultureInfo.InvariantCulture);

        // ============================================================ T1.44 — F-order ctor edges

        [TestMethod]
        public void T1_44_FOrderCtor_Size1Dims_FlagsStridesBytes_Match_Numpy()
        {
            // Size-1 dims are the subtle case: numpy flags them BOTH C- and F-contiguous. NumSharp must
            // agree flag-for-flag AND stride-for-stride against numpy's own order='F' array.
            var shapes = new (long[] dims, string npShape)[]
            {
                (new long[] { 1, 6 }, "(1,6)"),
                (new long[] { 6, 1 }, "(6,1)"),
                (new long[] { 1, 1, 6 }, "(1,1,6)"),
                (new long[] { 2, 1, 3 }, "(2,1,3)"),
                (new long[] { 6, 1, 1 }, "(6,1,1)"),
            };
            foreach (var (dims, npShape) in shapes)
            {
                long n = dims.Aggregate(1L, (a, b) => a * b);
                var buf = Enumerable.Range(0, (int)n).Select(x => (long)x).ToArray();
                var nsF = new NDArray(buf, new Shape(dims, 'C'), 'F');

                using (Gil())
                {
                    using PyObject npF = Python.np.evalStmts(
                        $"b = np.arange({n}, dtype='<i8')\n" +
                        $"r = np.ndarray({npShape}, dtype='<i8', buffer=b.tobytes(), order='F')", "r");

                    ByteContract.AssertSameBytes(nsF, npF, $"F {npShape} values");
                    nsF.Shape.IsFContiguous.Should().Be(
                        Python.np.truthy("a.flags['F_CONTIGUOUS']", ("a", npF)), $"F {npShape} F-contig flag");
                    nsF.Shape.IsContiguous.Should().Be(
                        Python.np.truthy("a.flags['C_CONTIGUOUS']", ("a", npF)), $"F {npShape} C-contig flag");
                    nsF.Shape.Strides.Should().Equal(
                        npF.strides_bytes().Select(s => s / 8L).ToArray(), $"F {npShape} element strides");
                }
            }
        }

        [TestMethod]
        public void T1_44_FOrderCtor_Empty_1D_HigherRank_Match_Numpy()
        {
            // Empty (0,3): trivially both-contiguous; 0 bytes; strides are meaningless so not asserted.
            var e = new NDArray(new long[0], new Shape(new long[] { 0, 3 }, 'C'), 'F');
            using (Gil())
            {
                using PyObject npE = Python.np.evalStmts(
                    "r = np.ndarray((0,3), dtype='<i8', buffer=b'', order='F')", "r");
                ByteContract.AssertSameBytes(e, npE, "empty F values");
                e.Shape.IsFContiguous.Should().BeTrue("empty arrays are trivially F-contiguous");
                e.size.Should().Be(0);
                e.shape.Select(x => (long)x).Should().Equal(npE.shape_dims());
            }

            // 1-D 'F' == 'C' (a 1-D array is both).
            var oneD = new NDArray(new long[] { 5, 6, 7 }, new Shape(new long[] { 3 }, 'C'), 'F');
            oneD.Shape.IsFContiguous.Should().BeTrue();
            oneD.Shape.IsContiguous.Should().BeTrue();
            using (Gil())
            {
                using PyObject np1 = Python.np.eval("np.array([5,6,7], dtype='<i8')");
                ByteContract.AssertSameBytes(oneD, np1, "1-D F==C values");
            }

            // 4-D higher-rank F.
            var buf4 = Enumerable.Range(0, 24).Select(x => (long)x).ToArray();
            var ns4 = new NDArray(buf4, new Shape(new long[] { 2, 3, 2, 2 }, 'C'), 'F');
            using (Gil())
            {
                using PyObject np4 = Python.np.evalStmts(
                    "b = np.arange(24, dtype='<i8')\n" +
                    "r = np.ndarray((2,3,2,2), dtype='<i8', buffer=b.tobytes(), order='F')", "r");
                ByteContract.AssertSameBytes(ns4, np4, "4-D F values");
                ns4.Shape.IsFContiguous.Should().BeTrue();
                ns4.Shape.Strides.Should().Equal(np4.strides_bytes().Select(s => s / 8L).ToArray());
            }
        }

        [TestMethod]
        public void T1_44_FOrderCtor_AllDtypes_ByteExact_Match_Numpy()
        {
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => (sbyte)x).ToArray(), "i1");
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => (byte)x).ToArray(), "u1");
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => (short)x).ToArray(), "i2");
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => (ushort)x).ToArray(), "u2");
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => x).ToArray(), "i4");
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => (uint)x).ToArray(), "u4");
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => (long)x).ToArray(), "i8");
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => (ulong)x).ToArray(), "u8");
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => (Half)x).ToArray(), "f2");
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => (float)x).ToArray(), "f4");
            AssertFOrderDtype(Enumerable.Range(0, 6).Select(x => (double)x).ToArray(), "f8");
        }

        /// <summary>An F-order (2,3) of arange(6) in dtype <paramref name="npDtype"/>, compared byte-for-byte to numpy.</summary>
        private void AssertFOrderDtype(Array buf, string npDtype)
        {
            var nsF = new NDArray(buf, new Shape(new long[] { 2, 3 }, 'C'), 'F');
            using (Gil())
            {
                using PyObject npF = Python.np.evalStmts(
                    $"b = np.arange(6, dtype='{npDtype}')\n" +
                    $"r = np.ndarray((2,3), dtype='{npDtype}', buffer=b.tobytes(), order='F')", "r");
                ByteContract.AssertSameBytes(nsF, npF, $"F-order dtype '{npDtype}'");
                nsF.Shape.IsFContiguous.Should().BeTrue($"F-order dtype '{npDtype}' F-contig");
            }
        }

        [TestMethod]
        public void T1_44_FOrderCtor_WriteThrough_FStrided_Match_Numpy()
        {
            // A buffer-owned F array is writeable, and a coordinate write must land through the F strides.
            var nsF = new NDArray(Enumerable.Range(0, 6).Select(x => (long)x).ToArray(),
                new Shape(new long[] { 2, 3 }, 'C'), 'F');
            nsF.Shape.IsWriteable.Should().BeTrue("a buffer-owned F array is writeable");

            WriteAt<long>(nsF, 99L, 1, 0);            // uses Shape.Strides — i.e. the F strides
            ReadAt<long>(nsF, 1, 0).Should().Be(99L);

            using (Gil())
            {
                // F(2,3) of [0..5] is [[0,2,4],[1,3,5]]; writing [1,0]=99 -> [[0,2,4],[99,3,5]].
                using PyObject exp = Python.np.eval("np.array([[0,2,4],[99,3,5]], dtype='<i8')");
                ByteContract.AssertSameBytes(nsF, exp, "F-strided write lands at the correct physical slot");
            }
        }

        [TestMethod]
        public void T1_44_FOrderCtor_IArraySliceOverload_Match_Numpy()
        {
            // The sibling (IArraySlice, Shape, 'F') ctor got the same fix — verify it too.
            var slice = ArraySlice.FromArray(Enumerable.Range(0, 6).Select(x => (long)x).ToArray(), true);
            var nsF = new NDArray(slice, new Shape(new long[] { 2, 3 }, 'C'), 'F');
            nsF.Shape.IsFContiguous.Should().BeTrue();
            using (Gil())
            {
                using PyObject npF = Python.np.evalStmts(
                    "b = np.arange(6, dtype='<i8')\n" +
                    "r = np.ndarray((2,3), dtype='<i8', buffer=b.tobytes(), order='F')", "r");
                ByteContract.AssertSameBytes(nsF, npF, "IArraySlice F ctor values");
                nsF.Shape.Strides.Should().Equal(npF.strides_bytes().Select(s => s / 8L).ToArray());
            }
        }

        // ============================================================ T1.50 — arange(bool) rare steps

        [TestMethod]
        public void T1_50_ArangeBool_RareSteps_ByteExact_Match_Numpy()
        {
            // Negative / fractional / empty step producing length <= 2 — all byte-exact vs numpy.
            var triples = new (double s, double e, double st)[]
            {
                (2, 0, -1),      // [True, True]
                (0, 1, 0.5),     // [False, True]
                (5, 5, 1),       // []  (empty)
                (-2, 0, 1),      // [True, True]
                (0, 2, 1.5),     // [False, True]  (ceil(2/1.5)=2)
                (3, 1, -1),      // [True, True]
            };
            foreach (var (s, e, st) in triples)
            {
                var ns = np.arange(s, e, st, NPTypeCode.Boolean);
                using (Gil())
                {
                    using PyObject npv = Python.np.eval($"np.arange({Py(s)}, {Py(e)}, {Py(st)}, dtype=bool)");
                    ByteContract.AssertSameBytes(ns, npv, $"arange({s},{e},{st},bool)");
                }
            }

            // Single-argument overload arange(stop, bool) — legal lengths.
            foreach (int stop in new[] { 0, 1, 2 })
            {
                var ns = np.arange(stop, NPTypeCode.Boolean);
                using (Gil())
                {
                    using PyObject npv = Python.np.eval($"np.arange({stop}, dtype=bool)");
                    ByteContract.AssertSameBytes(ns, npv, $"arange({stop},bool) single-arg");
                }
            }
        }

        [TestMethod]
        public void T1_50_ArangeBool_RareSteps_Over2_BothRaise_VerbatimMessage()
        {
            var triples = new (double s, double e, double st)[]
            {
                (3, 0, -1),      // length 3
                (0, 1.5, 0.5),   // length 3
                (0, 10, 1),      // length 10
                (-3, 0, 1),      // length 3
            };
            foreach (var (s, e, st) in triples)
            {
                Action nsAct = () => np.arange(s, e, st, NPTypeCode.Boolean);
                string nsMsg = nsAct.Should().Throw<NumSharp.TypeError>().Which.Message;
                string npMsg = NumpyErrorText($"np.arange({Py(s)}, {Py(e)}, {Py(st)}, dtype=bool)");
                npMsg.Should().NotBeEmpty();
                nsMsg.Should().Be(npMsg, $"arange({s},{e},{st},bool) message must be numpy's verbatim");
            }

            // Single-argument overload over length 2 also raises on both sides.
            ((Action)(() => np.arange(3, NPTypeCode.Boolean))).Should().Throw<NumSharp.TypeError>();
            NumpyErrorText("np.arange(3, dtype=bool)").Should().Contain("at most length 2");
        }

        // ============================================================ T1.56 — np.array ndmin rare values

        [TestMethod]
        public void T1_56_NpArray_Ndmin_RareValues_ShapesAndBytes_Match_Numpy()
        {
            // 1-D input: ndmin=-1 (no-op), ndmin=0, ndmin=5 (prepend four 1s).
            foreach (int ndmin in new[] { -1, 0, 5 })
                AssertNdminParity(new int[] { 1, 2, 3 }, "[1,2,3]", ndmin);

            // 2-D input: ndmin below rank is a no-op; above prepends 1s.
            foreach (int ndmin in new[] { 0, 3 })
                AssertNdminParity(new int[,] { { 1, 2 }, { 3, 4 } }, "[[1,2],[3,4]]", ndmin);

            // Empty input: ndmin=2 -> (1,0); ndmin=0 -> (0,).
            AssertNdminParity(new int[0], "[]", 2);
            AssertNdminParity(new int[0], "[]", 0);
        }

        private void AssertNdminParity(Array data, string npLit, int ndmin)
        {
            var ns = np.array(data, dtype: null, ndmin: ndmin);
            using (Gil())
            {
                using PyObject npv = Python.np.eval($"np.array({npLit}, dtype='<i4', ndmin={ndmin})");
                ns.shape.Select(x => (long)x).Should().Equal(npv.shape_dims(), $"{npLit} ndmin={ndmin} shape");
                ByteContract.AssertSameBytes(ns, npv, $"{npLit} ndmin={ndmin} values");
            }
        }

        // ============================================================ T1.54 — frombuffer edges

        [TestMethod]
        public void T1_54_Frombuffer_CountOffset_ByteExact_Match_Numpy()
        {
            using (Gil())
            {
                using PyObject src = Python.np.eval("np.array([10,20,30,40,50], dtype='<i4')");
                byte[] raw = src.bytes_c();

                using (PyObject exp = Python.np.with("a[:3]", ("a", src)))       // count=3
                    ByteContract.AssertSameBytes(np.frombuffer(raw, "<i4", count: 3), exp, "count=3");
                using (PyObject exp = Python.np.with("a[2:]", ("a", src)))       // offset=8 bytes = 2 int32
                    ByteContract.AssertSameBytes(np.frombuffer(raw, "<i4", offset: 8), exp, "offset=8");
                using (PyObject exp = Python.np.with("a[1:3]", ("a", src)))      // count=2, offset=4
                    ByteContract.AssertSameBytes(np.frombuffer(raw, "<i4", count: 2, offset: 4), exp, "count=2 offset=4");
            }

            // Empty buffer -> empty array.
            var nsEmpty = np.frombuffer(new byte[0], "<i4");
            nsEmpty.size.Should().Be(0);
            using (Gil())
            {
                using PyObject expEmpty = Python.np.eval("np.frombuffer(b'', dtype='<i4')");
                ByteContract.AssertSameBytes(nsEmpty, expEmpty, "empty buffer");
            }
        }

        [TestMethod]
        public void T1_54_Frombuffer_AllSupportedDtypes_ByteExact_Match_Numpy()
        {
            AssertFrombufferParity("?",   "np.array([1,0,1,0], dtype='?')");
            AssertFrombufferParity("i1",  "np.array([-128,-1,0,127], dtype='i1')");
            AssertFrombufferParity("u1",  "np.array([0,1,254,255], dtype='u1')");
            AssertFrombufferParity("i2",  "np.array([-32768,-1,0,32767], dtype='<i2')");
            AssertFrombufferParity("u2",  "np.array([0,1,65534,65535], dtype='<u2')");
            AssertFrombufferParity("i4",  "np.array([-2147483648,-1,0,2147483647], dtype='<i4')");
            AssertFrombufferParity("u4",  "np.array([0,1,4294967294,4294967295], dtype='<u4')");
            AssertFrombufferParity("i8",  "np.array([-1,0,1,4611686018427387904], dtype='<i8')");
            AssertFrombufferParity("u8",  "np.array([0,1,9223372036854775808,18446744073709551615], dtype='<u8')");
            AssertFrombufferParity("f2",  "np.array([1.5,-2.0,0.0,65504.0], dtype='<f2')");
            AssertFrombufferParity("f4",  "np.array([1.5,-2.25,3.4028235e38,0.0], dtype='<f4')");
            AssertFrombufferParity("f8",  "np.array([1.5,-2.25,3.141592653589793,0.0], dtype='<f8')");
            AssertFrombufferParity("c16", "np.array([1+2j,3-4j,0j,-5+0j], dtype='<c16')");
        }

        /// <summary>numpy builds the reference array; NumSharp frombuffer's its exact bytes; byte-compare.</summary>
        private void AssertFrombufferParity(string nsDtype, string npLiteralExpr)
        {
            using (Gil())
            {
                using PyObject src = Python.np.eval(npLiteralExpr);
                byte[] raw = src.bytes_c();
                var ns = np.frombuffer(raw, nsDtype);
                ByteContract.AssertSameBytes(ns, src, $"frombuffer('{nsDtype}') must reproduce numpy's array");
                ns.size.Should().Be((int)src.shape_dims()[0], $"frombuffer('{nsDtype}') element count");
            }
        }

        [TestMethod]
        public void T1_54_Frombuffer_BigEndian_ByteSwapsToNative_Match_Numpy()
        {
            // Big-endian input: numpy keeps the '>' dtype, NumSharp byte-swaps to native — so the VALUES
            // match (compared against numpy's native-cast reference) even though numpy's own bytes differ.
            AssertBigEndianParity("[1,2,3,-5]", "i2");
            AssertBigEndianParity("[1,2,3,-5]", "i4");
            AssertBigEndianParity("[1,2,3,9223372036854775807]", "i8");
            AssertBigEndianParity("[1,2,65535,40000]", "u2");
            AssertBigEndianParity("[1.5,-2.25,3.141592653589793,0.0]", "f8");
        }

        private void AssertBigEndianParity(string values, string dtype)
        {
            using (Gil())
            {
                using PyObject beSrc = Python.np.eval($"np.array({values}, dtype='>{dtype}')");
                byte[] beBytes = beSrc.bytes_c();                       // big-endian bytes
                var ns = np.frombuffer(beBytes, ">" + dtype);          // NumSharp swaps to native
                using PyObject neRef = Python.np.eval($"np.array({values}, dtype='<{dtype}')");  // native reference
                ByteContract.AssertSameBytes(ns, neRef,
                    $"frombuffer('>{dtype}') byte-swaps to native (values match numpy)");
            }
        }

        [TestMethod]
        public void T1_54_Frombuffer_SizeNotMultipleOfItemsize_BothRaise_SameMessage()
        {
            // NumSharp raises ArgumentException with numpy's verbatim text (numpy raises ValueError).
            Action ns = () => np.frombuffer(new byte[3], "<i4");
            ns.Should().Throw<ArgumentException>()
                .WithMessage("*buffer size must be a multiple of element size*");

            string npErr = NumpyErrorText("np.frombuffer(b'\\x00\\x00\\x00', dtype='<i4')");
            npErr.Should().Contain("buffer size must be a multiple of element size",
                "NumSharp reproduces numpy's error text (as ArgumentException — the house convention)");
        }

        // ============================================================ T1.47 — promotion across starts

        [TestMethod]
        public void T1_47_FindCommonType_ManyCombos_Match_Numpy_ResultType()
        {
            // find_common_type is removed in numpy 2.x, but its END result still equals numpy's modern
            // result_type for these same-kind and int->float widening combos (verified live below).
            AssertPromoParity(new[] { typeof(sbyte), typeof(short) }, "np.int8, np.int16");
            AssertPromoParity(new[] { typeof(short), typeof(int), typeof(long) }, "np.int16, np.int32, np.int64");
            AssertPromoParity(new[] { typeof(byte), typeof(ushort), typeof(uint) }, "np.uint8, np.uint16, np.uint32");
            AssertPromoParity(new[] { typeof(float), typeof(double) }, "np.float32, np.float64");
            AssertPromoParity(new[] { typeof(int), typeof(float), typeof(double) }, "np.int32, np.float32, np.float64");
            AssertPromoParity(new[] { typeof(long), typeof(double) }, "np.int64, np.float64");
            AssertPromoParity(new[] { typeof(int), typeof(float) }, "np.int32, np.float32");
            AssertPromoParity(new[] { typeof(int), typeof(int) }, "np.int32, np.int32");
        }

        private void AssertPromoParity(Type[] types, string npScalars)
        {
            NPTypeCode got = np.find_common_type(types, new Type[0]);
            string npName = PyStr($"np.result_type({npScalars})");
            got.AsNumpyDtypeName().Should().Be(npName,
                $"find_common_type({string.Join(",", types.Select(t => t.Name))}) == numpy result_type");
        }

        [TestMethod]
        public void T1_47_CanCoerceAll_VariousStarts_NoThrow_Match_Numpy_ResultType()
        {
            var arr = new[] { NPTypeCode.Int32, NPTypeCode.Single, NPTypeCode.Double };

            // start 0/1/2 -> promote the tail sub-list; cross-checked against numpy result_type of that tail.
            AssertCanCoerce(arr, 0, "np.int32, np.float32, np.float64");
            AssertCanCoerce(arr, 1, "np.float32, np.float64");
            AssertCanCoerce(arr, 2, "np.float64");

            // start == length -> empty tail -> Empty (no numpy analog; assert no-crash + Empty).
            MethodInfo mArr = CanCoerceArrayMethod();
            object empty = null;
            ((Action)(() => empty = mArr.Invoke(null, new object[] { arr, 3 })))
                .Should().NotThrow("start==length must not crash");
            ((NPTypeCode)empty).Should().Be(NPTypeCode.Empty);

            // The List<> sibling overload with start>0 must also not throw and give the same result.
            MethodInfo mList = typeof(np).GetMethod("_can_coerce_all",
                BindingFlags.NonPublic | BindingFlags.Static, null,
                new[] { typeof(List<NPTypeCode>), typeof(int) }, null);
            mList.Should().NotBeNull("internal _can_coerce_all(List<NPTypeCode>, int) overload");
            object listResult = null;
            ((Action)(() => listResult = mList.Invoke(null, new object[] { new List<NPTypeCode>(arr), 1 })))
                .Should().NotThrow("the List<> overload's uninitialized-slot bug must be fixed");
            ((NPTypeCode)listResult).Should().Be(NPTypeCode.Double);
        }

        private static MethodInfo CanCoerceArrayMethod() =>
            typeof(np).GetMethod("_can_coerce_all", BindingFlags.NonPublic | BindingFlags.Static, null,
                new[] { typeof(NPTypeCode[]), typeof(int) }, null);

        private void AssertCanCoerce(NPTypeCode[] arr, int start, string npTailScalars)
        {
            MethodInfo m = CanCoerceArrayMethod();
            object result = null;
            ((Action)(() => result = m.Invoke(null, new object[] { arr, start })))
                .Should().NotThrow($"_can_coerce_all(start={start}) must not throw");
            string npName = PyStr($"np.result_type({npTailScalars})");
            ((NPTypeCode)result).AsNumpyDtypeName().Should().Be(npName,
                $"_can_coerce_all(start={start}) == numpy result_type of the tail");
        }
    }
}
