using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     <see cref="TupleCodec"/>: C# tuples cross as Python tuples and back. pythonnet has no tuple
    ///     conversion of its own — a shape spelled <c>numpy.zeros((2, 3))</c> used to reach numpy as an
    ///     opaque wrapped <c>System.ValueTuple</c> — so this codec is what makes the idiomatic numpy call
    ///     shapes (shapes, axes, multi-indices, value/index pairs) read the same through <c>dynamic</c>.
    ///     Registered by <see cref="NDArrayPythonInterop.RegisterCodec()"/>; every test here goes through
    ///     pythonnet's own conversion pipeline (a <c>dynamic</c> call, <c>scope.Set</c>, <c>As&lt;T&gt;()</c>,
    ///     an implicit <c>dynamic</c> conversion), never through the codec object directly.
    /// </summary>
    [TestClass]
    public class TupleCodecTests : InteropTestBase
    {
        [TestInitialize]
        public void RegisterCodec() => NDArrayPythonInterop.RegisterCodec();

        // ============================  encode: C# tuple -> Python tuple  ==========================

        [TestMethod]
        public void Encode_ValueTupleShape_ReachesNumpyAsATuple()
        {
            using (Gil())
            {
                dynamic numpy = Py.Import("numpy");
                dynamic a = numpy.zeros((2, 3));                     // the very call that failed without the codec
                ((string)a.shape.ToString()).Should().Be("(2, 3)");
                ((string)a.strides.ToString()).Should().Be("(24, 8)");

                dynamic r = numpy.arange(6).reshape((3, 2));
                ((string)r.shape.ToString()).Should().Be("(3, 2)");
            }
        }

        [TestMethod]
        public void Encode_BoundIntoAScope_IsAPythonTuple_NestedAndMixed()
        {
            using (Gil())
            {
                Scope.Set("t", ((1, 2), 3.5, "x", (string)null));
            }

            PyStr("type(t).__name__").Should().Be("tuple");
            PyStr("repr(t)").Should().Be("((1, 2), 3.5, 'x', None)", "elements convert recursively; null is None");
            PyStr("type(t[0]).__name__").Should().Be("tuple", "a nested C# tuple is a nested Python tuple");
        }

        [TestMethod]
        public void Encode_ReferenceTuple_AndEightPlusElements_RideTheRestSlot()
        {
            using (Gil())
            {
                Scope.Set("rt", Tuple.Create(4, 5));
                Scope.Set("eight", (1, 2, 3, 4, 5, 6, 7, 8));
                Scope.Set("ten", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
            }

            PyStr("repr(rt)").Should().Be("(4, 5)", "System.Tuple encodes like ValueTuple");
            PyStr("repr(eight)").Should().Be("(1, 2, 3, 4, 5, 6, 7, 8)", "ITuple flattens the Rest chain");
            PyLong("len(ten)").Should().Be(10);
        }

        [TestMethod]
        public void Encode_NDArrayElement_CrossesAsAZeroCopyNumpyView()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Gil())
            {
                Scope.Set("pair", (nd, 2));
            }

            PyStr("type(pair[0]).__name__").Should().Be("ndarray", "the element went through the numpy codec");
            PyBool("pair[0].flags['OWNDATA']").Should().BeFalse("...as a view over NumSharp's buffer");
            PyExec("pair[0][1] = 42.0");
            ReadAt<double>(nd, 1).Should().Be(42.0, "the view shares memory");

            PyExec("del pair");
            nd.Dispose();
        }

        [TestMethod]
        public void Encode_AsACallArgument_NumpyTakesTheShape_AndBroadcastToGetsATuple()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Gil())
            {
                dynamic numpy = Py.Import("numpy");
                dynamic b = numpy.broadcast_to(nd, (2, 4));           // an NDArray AND a tuple in one call
                ((string)b.shape.ToString()).Should().Be("(2, 4)");
                ((string)b.strides.ToString()).Should().Be("(0, 8)");
                ((double)b.item((1, 3))).Should().Be(3.0, "item((1, 3)) takes a tuple multi-index");

                dynamic z = numpy.zeros((2, 3));
                z.__setitem__((1, 2), 9.5);                            // a tuple multi-index write
                ((double)z.item(1, 2)).Should().Be(9.5);
            }

            nd.Dispose();
        }

        [TestMethod]
        public void Encode_LeavesNoDanglingReference_OnTheElements()
        {
            using (Gil())
            {
                Scope.Exec("import sys");
                Scope.Set("t", (1234567, 7654321));                   // ints outside the small-int cache: fresh objects
            }

            // the tuple holds one reference to each element; the C# element wrappers were disposed
            // right after the PyTuple took its own reference — no leak, no premature free.
            PyLong("sys.getrefcount(t[0])").Should().BeLessThan(6);
            PyLong("t[0] + t[1]").Should().Be(1234567 + 7654321, "the elements are alive and correct");
        }

        // ============================  decode: Python tuple -> C# tuple  ==========================

        [TestMethod]
        public void Decode_ShapeAndStrides_IntoValueTuples_ThroughDynamic()
        {
            using (Gil())
            {
                dynamic numpy = Py.Import("numpy");
                dynamic a = numpy.zeros((2, 3));
                (long, long) shape = a.shape;                          // the implicit dynamic conversion asks the decoder
                (int, int) shapeInts = a.shape;
                (long, long) strides = a.strides;

                shape.Should().Be((2L, 3L));
                shapeInts.Should().Be((2, 3));
                strides.Should().Be((24L, 8L));
            }
        }

        [TestMethod]
        public void Decode_MixedNestedAndReferenceTuples_ByElementType()
        {
            using (Gil())
            {
                using PyObject mixed = Scope.Eval("(1, 'x', 2.5)");
                mixed.As<(long, string, double)>().Should().Be((1L, "x", 2.5));

                using PyObject nested = Scope.Eval("((1, 2), 3)");
                nested.As<((long, long), long)>().Should().Be(((1L, 2L), 3L));

                using PyObject reference = Scope.Eval("(1, 2)");
                Tuple<long, long> rt = reference.As<Tuple<long, long>>();
                rt.Item1.Should().Be(1);
                rt.Item2.Should().Be(2);

                using PyObject ten = Scope.Eval("tuple(range(10))");
                ten.As<(int, int, int, int, int, int, int, int, int, int)>().Should().Be((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "8+ elements decode through the Rest slot");

                using PyObject empty = Scope.Eval("()");
                empty.As<ValueTuple>().Should().Be(default(ValueTuple));
            }
        }

        [TestMethod]
        public void Decode_NDArrayElement_GoesThroughTheNumpyCodec()
        {
            (NDArray, long) pair;
            using (Gil())
            {
                using PyObject t = Scope.Eval("(np.arange(3, dtype='f8'), 7)");
                pair = t.As<(NDArray, long)>();
            }

            pair.Item2.Should().Be(7);
            pair.Item1.typecode.Should().Be(NPTypeCode.Double);
            pair.Item1.size.Should().Be(3);
            pair.Item1.Dispose();
        }

        [TestMethod]
        public void Decode_NamedTuple_IsATupleSubclass_AndDecodes()
        {
            PyExec("from collections import namedtuple\nPoint = namedtuple('Point', 'x y')\np = Point(3, 4)");
            using (Gil())
            {
                using PyObject p = Scope.Get("p");
                p.As<(long, long)>().Should().Be((3L, 4L));
            }
        }

        [TestMethod]
        public void Decode_Declines_OnArityMismatch_AndOnALIst()
        {
            using (Gil())
            {
                using PyObject three = Scope.Eval("(1, 2, 3)");
                new Action(() => three.As<(long, long)>()).Should().Throw<Exception>("a tuple decodes only into a C# tuple of the SAME arity — never truncated");

                using PyObject list = Scope.Eval("[1, 2]");
                new Action(() => list.As<(long, long)>()).Should().Throw<Exception>("a list is not a tuple");
            }
        }

        [TestMethod]
        public void Decode_ScopeGet_AndModuleGetT_ReadATupleVariable()
        {
            PyExec("shape2 = (3, 4)");
            using (Gil())
            {
                Scope.Get<(long, long)>("shape2").Should().Be((3L, 4L));
                dynamic py = Scope;
                (long, long) viaDynamic = py.shape2;
                viaDynamic.Should().Be((3L, 4L));
            }
        }

        // ============================  registration  ==============================================

        [TestMethod]
        public void Options_ConvertTuples_DefaultsOn_AndTheStandaloneCodecAnswersOnlyForTuples()
        {
            NumpyCodecOptions.Default.ConvertTuples.Should().BeTrue();
            new NumpyCodecOptions { ConvertTuples = false }.ConvertTuples.Should().BeFalse();

            var codec = TupleCodec.Instance;
            codec.CanEncode(typeof((int, int))).Should().BeTrue();
            codec.CanEncode(typeof(Tuple<int, int>)).Should().BeTrue();
            codec.CanEncode(typeof(int[])).Should().BeFalse();
            codec.CanEncode(typeof(NDArray)).Should().BeFalse("arrays belong to the numpy codec");

            using (Gil())
            {
                using PyObject t = Scope.Eval("(1, 2)");
                using PyObject l = Scope.Eval("[1, 2]");
                using PyObject tt = t.GetPythonType();
                using PyObject lt = l.GetPythonType();
                using var tupleType = new PyType(tt);
                using var listType = new PyType(lt);
                codec.CanDecode(tupleType, typeof((long, long))).Should().BeTrue();
                codec.CanDecode(listType, typeof((long, long))).Should().BeFalse("only tuples decode");
                codec.CanDecode(tupleType, typeof(long[])).Should().BeFalse("only tuple targets");
            }
        }
    }
}
