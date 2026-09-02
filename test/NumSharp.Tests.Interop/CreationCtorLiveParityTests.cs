using System;
using System.Linq;
using System.Reflection;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     <b>Live parity of the seven AuditV2 Tier-1 creation/ctor fixes against embedded CPython
    ///     numpy 2.4.2.</b> Each case computes a value in NumSharp and again in the SAME in-process
    ///     numpy over identical bytes, and asserts them <see cref="ByteContract">byte-identical</see>
    ///     — the strongest equality (it catches dtype width, endianness and layout linearization a
    ///     value compare silently accepts). The companion offline unit gate is
    ///     <c>AuditV2_NDArrayCreation</c> (probed-and-hardcoded); this suite proves the same claims
    ///     against a REAL numpy running beside NumSharp.
    ///
    ///     <para>Two of the seven fixes are DELIBERATE divergences, not parity, and their tests pin
    ///     that honestly: numpy has no equivalent of the removed <c>find_common_type</c> (T1.47), and
    ///     NumSharp has no 32-bit complex so it rejects <c>c8</c>/<c>F</c> where numpy reads complex64
    ///     (T1.54). Those two run live numpy to record exactly what numpy does and assert NumSharp's
    ///     documented, self-consistent departure. The other five (T1.44/50/56) are strict byte-parity.</para>
    /// </summary>
    [TestClass]
    public class CreationCtorLiveParityTests : InteropTestBase
    {
        /// <summary>
        ///     <c>str(e)</c> of whatever exception the one-line numpy statement raises ("" if none).
        ///     Runs in <see cref="InteropTestBase.Scope"/> (which already has <c>np</c> imported), so
        ///     the raising cases assert against numpy's ACTUAL message text, not a paraphrase.
        /// </summary>
        private string NumpyErrorText(string oneLineStmt)
        {
            PyExec("__e=''\ntry:\n    " + oneLineStmt + "\nexcept Exception as e:\n    __e=str(e)");
            return PyStr("__e");
        }

        // ============================================================ T1.44 — F-order ctor honours 'F'
        // NumPy: np.ndarray(shape, buffer=, order='F') reinterprets the row-major buffer as column-major
        // via the F-stride recurrence strides[0]=itemsize; strides[i]=strides[i-1]*dims[i-1]. NumSharp's
        // NDArray(buffer, shape, 'F') now does the same (ComputeFContiguousStrides), so the values, the
        // F_CONTIGUOUS/C_CONTIGUOUS flags and the strides all match numpy.

        [TestMethod]
        public void T1_44_FOrderCtor_2D_ByteExact_FlagsAndStrides_Match_Numpy()
        {
            var buf = new long[] { 1, 2, 3, 4, 5, 6 };
            var nsF = new NDArray(buf, new Shape(2, 3), 'F');
            var nsC = new NDArray(buf, new Shape(2, 3), 'C');

            using (Gil())
            {
                using PyObject npF = Python.np.evalStmts(
                    "b = np.array([1,2,3,4,5,6], dtype='<i8')\n" +
                    "r = np.ndarray((2,3), dtype='<i8', buffer=b.tobytes(), order='F')", "r");
                using PyObject npC = Python.np.evalStmts(
                    "b = np.array([1,2,3,4,5,6], dtype='<i8')\n" +
                    "r = np.ndarray((2,3), dtype='<i8', buffer=b.tobytes(), order='C')", "r");

                // (1) values — canonical C-order linearization matches numpy on BOTH orders.
                ByteContract.AssertSameBytes(nsF, npF, "ndarray(buffer, order='F') values must match numpy");
                ByteContract.AssertSameBytes(nsC, npC, "ndarray(buffer, order='C') values must match numpy");

                // (2) numpy's own F array: F- not C-contiguous, byte strides (8,16).
                npF.shape_dims().Should().Equal(new long[] { 2, 3 });
                npF.strides_bytes().Should().Equal(new long[] { 8, 16 });
                Python.np.truthy("a.flags['F_CONTIGUOUS'] and not a.flags['C_CONTIGUOUS']", ("a", npF))
                    .Should().BeTrue("numpy F array is F- not C-contiguous");

                // (3) NumSharp side agrees flag-for-flag and stride-for-stride (element strides).
                nsF.Shape.IsFContiguous.Should().BeTrue("NumSharp ctor must produce an F-contiguous shape");
                nsF.Shape.IsContiguous.Should().BeFalse();
                nsF.Shape.Strides.Should().Equal(npF.strides_bytes().Select(s => s / 8L).ToArray(),
                    "NumSharp element strides == numpy byte strides / itemsize");
                nsC.Shape.IsContiguous.Should().BeTrue("the 'C' path is unchanged");
            }
        }

        [TestMethod]
        public void T1_44_FOrderCtor_3D_ByteExact_Match_Numpy()
        {
            var buf = Enumerable.Range(0, 24).Select(x => (long)x).ToArray();
            var nsF = new NDArray(buf, new Shape(2, 3, 4), 'F');

            using (Gil())
            {
                using PyObject npF = Python.np.evalStmts(
                    "b = np.arange(24, dtype='<i8')\n" +
                    "r = np.ndarray((2,3,4), dtype='<i8', buffer=b.tobytes(), order='F')", "r");

                ByteContract.AssertSameBytes(nsF, npF, "3-D F ctor values must match numpy");
                Python.np.truthy("a.flags['F_CONTIGUOUS']", ("a", npF)).Should().BeTrue();
                nsF.Shape.IsFContiguous.Should().BeTrue();
                nsF.Shape.Strides.Should().Equal(npF.strides_bytes().Select(s => s / 8L).ToArray());
            }
        }

        // ============================================================ T1.50 — arange(bool) length gate
        // NumPy PyArray_Arange sets buf[0]=start, returns at len==1, sets buf[1]=start+step, returns at
        // len==2, else calls fill() — which for bool is BOOL_fill and raises TypeError. NumSharp now
        // matches the boundary AND the verbatim message.

        [TestMethod]
        public void T1_50_ArangeBool_LegalLengths_ByteExact_Match_Numpy()
        {
            // Every legal (length <= 2) case — including value-varied starts — byte-exact against numpy.
            var cases = new (int start, int stop)[] { (0, 0), (0, 1), (0, 2), (1, 3), (-1, 1), (2, 4), (5, 5) };
            foreach (var (start, stop) in cases)
            {
                var ns = np.arange(start, stop, 1, NPTypeCode.Boolean);
                using (Gil())
                {
                    using PyObject npv = Python.np.eval($"np.arange({start}, {stop}, 1, dtype=bool)");
                    ByteContract.AssertSameBytes(ns, npv, $"arange({start},{stop},1,bool) values");
                    npv.dtype_str().Should().Be("|b1", "numpy bool dtype str");
                }
            }
        }

        [TestMethod]
        public void T1_50_ArangeBool_LengthOver2_BothRaise_VerbatimMessage()
        {
            foreach (int stop in new[] { 3, 5, 10 })
            {
                // NumSharp raises TypeError.
                Action nsAct = () => np.arange(0, stop, 1, NPTypeCode.Boolean);
                string nsMsg = nsAct.Should().Throw<NumSharp.TypeError>().Which.Message;

                // numpy raises TypeError too (surfaced as PythonException) ...
                using (Gil())
                {
                    Action pyAct = () => { using var _ = Python.np.eval($"np.arange(0, {stop}, 1, dtype=bool)"); };
                    pyAct.Should().Throw<PythonException>("numpy rejects arange(bool) beyond length 2");
                }

                // ... with a message NumSharp reproduces character-for-character.
                string npMsg = NumpyErrorText($"np.arange(0, {stop}, 1, dtype=bool)");
                npMsg.Should().NotBeEmpty();
                nsMsg.Should().Be(npMsg, "NumSharp's arange(bool) TypeError text must be numpy's verbatim");
            }
        }

        // ============================================================ T1.56 — np.array default ndmin=0
        // NumPy's np.array default is ndmin=0 (only prepends 1s when actual ndim < ndmin). NumSharp's
        // (Array,...) overload default is now 0 too; behaviour is byte-identical for any rank>=1 input.

        [TestMethod]
        public void T1_56_NpArray_Ndmin_DefaultAndSweep_ShapesAndBytes_Match_Numpy()
        {
            var data = new int[] { 1, 2, 3 };

            // Default ndmin (omitted) — matches numpy's default ndmin=0.
            var nsDefault = np.array((Array)data);
            using (Gil())
            {
                using PyObject npDefault = Python.np.eval("np.array([1,2,3], dtype='<i4')");
                nsDefault.shape.Select(x => (long)x).Should().Equal(npDefault.shape_dims(), "default ndmin shape");
                ByteContract.AssertSameBytes(nsDefault, npDefault, "np.array default ndmin values");
            }

            // Explicit ndmin sweep — the prepend-ones semantics must match numpy exactly.
            foreach (int ndmin in new[] { 0, 1, 2, 3 })
            {
                var ns = np.array((Array)data, dtype: null, ndmin: ndmin);
                using (Gil())
                {
                    using PyObject npv = Python.np.eval($"np.array([1,2,3], dtype='<i4', ndmin={ndmin})");
                    ns.shape.Select(x => (long)x).Should().Equal(npv.shape_dims(), $"ndmin={ndmin} shape");
                    ByteContract.AssertSameBytes(ns, npv, $"ndmin={ndmin} values");
                }
            }

            // A 2-D input keeps its rank under the new default (max(actual, 0) == actual).
            var data2 = new int[,] { { 1, 2 }, { 3, 4 } };
            var ns2 = np.array(data2);
            using (Gil())
            {
                using PyObject np2 = Python.np.eval("np.array([[1,2],[3,4]], dtype='<i4')");
                ns2.shape.Select(x => (long)x).Should().Equal(np2.shape_dims());
                ByteContract.AssertSameBytes(ns2, np2, "2-D input default ndmin values");
            }
        }

        // ============================================================ T1.54 — frombuffer dtypes
        // The SUPPORTED dtypes are true byte-parity. 'c8'/'F' (complex64) are a DELIBERATE divergence:
        // NumSharp has no 32-bit complex, so it rejects them (consistent with np.dtype()) where numpy
        // reads complex64 — this test runs live numpy to pin exactly that.

        [TestMethod]
        public void T1_54_Frombuffer_SupportedDtypes_ByteExact_Match_Numpy()
        {
            AssertFrombufferParity("<i4",  "np.array([1,-2,3,-4], dtype='<i4')");
            AssertFrombufferParity("<u1",  "np.array([0,1,254,255], dtype='<u1')");
            AssertFrombufferParity("<f8",  "np.array([1.5,-2.25,3.125,0.0], dtype='<f8')");
            AssertFrombufferParity("<c16", "np.array([1+2j, 3-4j], dtype='<c16')");
        }

        /// <summary>numpy builds the reference array; NumSharp <c>frombuffer</c>s its exact bytes; byte-compare.</summary>
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
        public void T1_54_Frombuffer_c8_F_NumpyReadsComplex64_NumSharpDeliberatelyRejects()
        {
            // Live numpy: 'c8' / 'F' ARE complex64 — 16 bytes -> 2 elements. NumSharp cannot reproduce
            // this (it has no 32-bit complex type).
            using (Gil())
            {
                PyExec("c8 = np.frombuffer(np.array([1.,2.,3.,4.], dtype='<f4').tobytes(), dtype='c8')");
                PyStr("c8.dtype.name").Should().Be("complex64", "numpy: 'c8' is complex64");
                PyLong("c8.size").Should().Be(2, "numpy: 16 bytes / 8-byte complex64 = 2 elements");
                PyStr("np.dtype('F').name").Should().Be("complex64", "numpy: 'F' typechar is complex64");
            }

            // NumSharp: deliberate, self-consistent divergence — reject across frombuffer AND np.dtype.
            var bytes16 = new byte[16];
            ((Action)(() => np.frombuffer(bytes16, "c8"))).Should().Throw<NotSupportedException>();
            ((Action)(() => np.frombuffer(bytes16, "F"))).Should().Throw<NotSupportedException>();
            ((Action)(() => np.frombuffer(bytes16, "complex64"))).Should().Throw<NotSupportedException>();
            ((Action)(() => np.dtype("c8"))).Should().Throw<NotSupportedException>("frombuffer must agree with np.dtype");
            ((Action)(() => np.dtype("F"))).Should().Throw<NotSupportedException>();

            // The 128-bit sibling still resolves and is byte-exact — only the 32-bit width is refused.
            using (Gil())
            {
                using PyObject src = Python.np.eval("np.array([1+2j, 3-4j], dtype='<c16')");
                byte[] raw = src.bytes_c();
                var ns = np.frombuffer(raw, "c16");
                ByteContract.AssertSameBytes(ns, src, "c16 (complex128) is supported while c8 is rejected");
            }
        }

        // ============================================================ T1.47 — find_common_type crash fix
        // numpy REMOVED find_common_type in 2.0 (there is no flow to mirror). NumSharp keeps the legacy
        // helper; the fix is the internal _can_coerce_all(start>0) crash. This test pins the removal live
        // and cross-checks NumSharp's END result against numpy's modern replacement, result_type.

        [TestMethod]
        public void T1_47_FindCommonType_RemovedInNumpy_NumSharp_EndResult_Matches_ResultType()
        {
            using (Gil())
            {
                // numpy 2.x has no find_common_type — it raises AttributeError pointing at the removal.
                string removed = NumpyErrorText("np.find_common_type([np.int32, np.float32], [])");
                removed.Should().Contain("find_common_type").And.Contain("removed",
                    "numpy 2.0 removed find_common_type");

                // numpy's modern replacement for array-dtype promotion is result_type.
                PyStr("str(np.result_type(np.int32, np.float32, np.float64))").Should().Be("float64");
                PyStr("str(np.result_type(np.float32, np.float64))").Should().Be("float64");
            }

            // NumSharp keeps the legacy helper; its END result equals numpy's result_type.
            np.find_common_type(new[] { typeof(int), typeof(float), typeof(double) }, new Type[0])
                .Should().Be(NPTypeCode.Double, "== numpy result_type(int32,float32,float64) == float64");

            // The T1.47 core defect: internal _can_coerce_all(arr, start>0) must not crash (Array.Copy
            // destIndex bug), and its result must equal numpy's result_type of the tail sub-list.
            var m = typeof(np).GetMethod("_can_coerce_all", BindingFlags.NonPublic | BindingFlags.Static, null,
                new[] { typeof(NPTypeCode[]), typeof(int) }, null);
            m.Should().NotBeNull("internal _can_coerce_all(NPTypeCode[], int) overload");

            object result = null;
            ((Action)(() => result = m.Invoke(null,
                    new object[] { new[] { NPTypeCode.Int32, NPTypeCode.Single, NPTypeCode.Double }, 1 })))
                .Should().NotThrow("start>0 must not throw ArgumentException via the wrong Array.Copy destIndex");
            ((NPTypeCode)result).Should().Be(NPTypeCode.Double,
                "the tail {Single,Double} promotes to Double == numpy result_type(float32,float64)");
        }
    }
}
