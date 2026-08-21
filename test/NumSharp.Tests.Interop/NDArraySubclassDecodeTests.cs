using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     The <c>numpy.ndarray</c>-subclass gate (<c>NumpyCodec</c>'s <c>__mro__</c> walk) and the
    ///     dtypes NumSharp has to decline — the two things a type-level <c>CanDecode</c> can and cannot
    ///     decide.
    ///
    ///     <para><b>Why a subclass needs the <c>__mro__</c> walk:</b> a subclass's <c>tp_name</c> is its
    ///     OWN (<c>matrix</c>, <c>MaskedArray</c>, a user name), never <c>numpy.ndarray</c>, so the fast
    ///     name check misses it. In the default mode it is still caught as a buffer exporter, but with
    ///     <c>DecodeAnyBuffer=false</c> (numpy-arrays-only) the walk is the ONLY thing that keeps a
    ///     matrix/memmap/masked/user subclass decodable — which is exactly what these tests pin.</para>
    ///
    ///     <para><b>Why object/structured/datetime can't be gated at <c>CanDecode</c>:</b> pythonnet hands
    ///     <c>CanDecode</c> the <c>PyType</c>, not the instance, so it cannot see the dtype — every
    ///     <c>numpy.ndarray</c> must pass <c>CanDecode</c> and the real refusal happens in <c>TryDecode</c>
    ///     (which returns <c>false</c>) and in the verbs (which throw <c>NotSupportedException</c>). These
    ///     tests pin that graceful, at-the-instance decline.</para>
    ///
    ///     <para>Codecs are exercised as LOCAL instances (never the process-global registration), so this
    ///     never perturbs other tests.</para>
    /// </summary>
    [TestClass]
    public class NdarraySubclassDecodeTests : InteropTestBase
    {
        private static PyType TypeOf(PyObject o) => new PyType(o.GetPythonType());

        // ==============================  subclasses DO decode (the __mro__ walk works)  =========

        [TestMethod]
        public void Subclass_UserNdarraySubclass_DecodesLikeItsBase()
        {
            using (Gil())
            {
                Scope.Exec("class MyArr(np.ndarray):\n    pass\n" +
                           "usub = np.arange(3, dtype='<f8').view(MyArr)");
                using PyObject sub = Scope.Eval("usub");

                using (NDArray copy = sub.ToNDArray())
                {
                    copy.typecode.Should().Be(NPTypeCode.Double);
                    copy.ndim.Should().Be(1);
                    copy.shape[0].Should().Be(3);
                    ByteContract.AssertSameBytes(copy, sub, "a user ndarray subclass copies like its base");
                }

                using NDArray view = sub.AsNDArray();   // shares memory, like any ndarray
                ByteContract.AssertSameBytes(view, sub, "a user ndarray subclass views like its base");
            }
        }

        [TestMethod]
        public void Subclass_NumpyMatrix_And_MaskedArray_Decode()
        {
            using (Gil())
            {
                using (PyObject mx = Scope.Eval("np.matrix('1 2; 3 4').astype('<i8')"))
                using (NDArray nd = mx.ToNDArray())
                {
                    nd.ndim.Should().Be(2, "numpy.matrix is always 2-D");
                    nd.shape[0].Should().Be(2);
                    nd.shape[1].Should().Be(2);
                    nd.GetAtIndex<long>(3).Should().Be(4, "matrix decodes via the __mro__ walk");
                }

                using (PyObject masked = Scope.Eval("np.ma.array([10.0, 20.0, 30.0], mask=[0,1,0])"))
                using (NDArray nd = masked.ToNDArray())
                    nd.typecode.Should().Be(NPTypeCode.Double, "a MaskedArray is an ndarray subclass and decodes");
            }
        }

        // The one behavior worth stating out loud: decoding a MaskedArray yields its UNDERLYING data with
        // the mask DROPPED — identical to what numpy's own np.asarray does to a masked array. NumSharp has
        // no masked dtype, so this is the consistent choice, not a loss unique to the interop.
        [TestMethod]
        [TestCategory("Misaligned")]
        public void MaskedArray_DecodesToUnderlyingData_MaskDropped_MatchingAsarray()
        {
            using (Gil())
            {
                using PyObject masked = Scope.Eval("np.ma.array([10.0, 20.0, 30.0], mask=[0, 1, 0])");
                using NDArray nd = masked.ToNDArray();

                nd.GetAtIndex<double>(1).Should().Be(20.0, "the masked element decodes as its underlying value, not a fill");
                using PyObject asarr = Python.np.asarray(masked);   // np.asarray drops the mask too
                ByteContract.AssertSameBytes(nd, asarr, "NumSharp's masked-array decode equals np.asarray(masked)");
            }
        }

        // ==============================  the mode where the __mro__ walk is load-bearing  =======

        [TestMethod]
        public void CanDecode_NumpyOnlyMode_AcceptsSubclasses_RejectsPlainBuffers()
        {
            // DecodeAnyBuffer=false => a bare bytes/bytearray/memoryview is NOT a numpy array and must be
            // declined, but an ndarray SUBCLASS (whose tp_name is not "numpy.ndarray") must still be
            // accepted — and the only thing that can accept it here is IsNdarraySubclass.
            var numpyOnly = new NumpyCodec(new NumpyCodecOptions { DecodeAnyBuffer = false, DecodeArrayLike = false });

            using (Gil())
            {
                Scope.Exec("class MyArr2(np.ndarray):\n    pass\nusub2 = np.arange(3).view(MyArr2)");

                foreach (string subExpr in new[] { "np.matrix('1 2; 3 4')", "usub2", "np.ma.array([1,2,3], mask=[0,1,0])" })
                    using (PyObject o = Scope.Eval(subExpr))
                        numpyOnly.CanDecode(TypeOf(o), typeof(NDArray))
                                 .Should().BeTrue($"{subExpr} is an ndarray subclass — accepted via the __mro__ walk");

                foreach (string plainExpr in new[] { "b'abcd'", "bytearray(4)", "memoryview(b'ab')", "[1, 2, 3]" })
                    using (PyObject o = Scope.Eval(plainExpr))
                        numpyOnly.CanDecode(TypeOf(o), typeof(NDArray))
                                 .Should().BeFalse($"{plainExpr} is not a numpy array — declined when DecodeAnyBuffer/ArrayLike are off");
            }
        }

        [TestMethod]
        public void IsNdarraySubclass_DoesNotFalsePositive_OnUnrelatedTypes()
        {
            // A plain Python object whose MRO never reaches numpy.ndarray must not be mistaken for one.
            var numpyOnly = new NumpyCodec(new NumpyCodecOptions { DecodeAnyBuffer = false, DecodeArrayLike = false });
            using (Gil())
                foreach (string expr in new[] { "object()", "dict(a=1)", "'a string'", "np.int64(5)" })
                    using (PyObject o = Scope.Eval(expr))
                        numpyOnly.CanDecode(TypeOf(o), typeof(NDArray))
                                 .Should().BeFalse($"{expr} is not an ndarray subclass");
            }

        // ==============================  numpy-only dtypes: CanDecode true, decode declines  ====

        [TestMethod]
        public void ObjectDtype_CanDecodeTrue_ButTryDecodeFalse_AndVerbsThrow()
        {
            // The crux: an object array's TYPE is numpy.ndarray, so CanDecode CANNOT reject it — the
            // decline has to happen at the instance. Both a full and a 0-d object array behave the same.
            var codec = new NumpyCodec();
            using (Gil())
                foreach (string expr in new[] { "np.array([1, 'x', None], dtype=object)", "np.array(None, dtype=object)" })
                {
                    using PyObject obj = Scope.Eval(expr);

                    codec.CanDecode(TypeOf(obj), typeof(NDArray)).Should().BeTrue("its type is numpy.ndarray");
                    codec.TryDecode<NDArray>(obj, out NDArray decoded).Should().BeFalse("...but object dtype has no NumSharp representation");
                    decoded.Should().BeNull();

                    ((Action)(() => { using var _ = obj.ToNDArray(); }))
                        .Should().Throw<NotSupportedException>($"ToNDArray refuses {expr}");
                    ((Action)(() => { using var _ = NDArrayPythonInterop.ToNDArrayView(obj, allowReadonly: true); }))
                        .Should().Throw<NotSupportedException>($"ToNDArrayView refuses {expr}");
                }
        }

        [TestMethod]
        public void StructuredVoidAndDatetimeDtypes_Decline_AtTheInstance()
        {
            var codec = new NumpyCodec();
            using (Gil())
                foreach (string expr in new[]
                {
                    "np.zeros(3, dtype=[('a','<i4'),('b','<f8')])",   // structured / record
                    "np.zeros(3, dtype='V8')",                        // raw void
                    "np.array(['2021-01-01'], dtype='<M8[D]')",       // datetime64
                    "np.array([1, 2], dtype='<m8[s]')",               // timedelta64
                })
                {
                    using PyObject obj = Scope.Eval(expr);
                    codec.CanDecode(TypeOf(obj), typeof(NDArray)).Should().BeTrue("type is numpy.ndarray");
                    codec.TryDecode<NDArray>(obj, out NDArray decoded).Should().BeFalse($"{expr} has no NumSharp dtype");
                    decoded.Should().BeNull();
                    ((Action)(() => { using var _ = obj.ToNDArray(); }))
                        .Should().Throw<NotSupportedException>($"ToNDArray refuses {expr}");
                }
        }

        // ==============================  engine stays healthy after the declines  ================

        [TestMethod]
        public void FailedDecodes_LeaveTheEngineHealthy()
        {
            var codec = new NumpyCodec();
            using (Gil())
            {
                // Hammer the decline path, then prove numpy still computes and a normal array still decodes.
                for (int i = 0; i < 25; i++)
                {
                    using PyObject bad = Scope.Eval("np.array([object()], dtype=object)");
                    codec.TryDecode<NDArray>(bad, out _).Should().BeFalse();
                }

                using PyObject good = Scope.Eval("np.arange(4, dtype='<f8') + 1.0");
                using NDArray nd = good.ToNDArray();
                nd.GetAtIndex<double>(3).Should().Be(4.0, "the engine and the decode path are unharmed by prior failures");
            }
        }
    }
}
