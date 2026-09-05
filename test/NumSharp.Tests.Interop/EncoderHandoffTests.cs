using System;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     <see cref="EncoderHandoff"/>: the wrapper an encoder hands pythonnet is disposed by the NEXT encode
    ///     on the same thread, so an implicit encode (<c>scope.Set("x", nd)</c>, <c>py.x = nd</c>, an
    ///     <see cref="NDArray"/> argument to a <c>dynamic</c> call) does not keep its export pinned until the
    ///     CLR finalizer runs. Found by the telemetry-ring scenario of the examples: 150 rebindings of one
    ///     name through the codec left 150 live exports where the explicit <c>using PyObject</c> spelling
    ///     holds one.
    /// </summary>
    [TestClass]
    public class EncoderHandoffTests : InteropTestBase
    {
        [TestInitialize]
        public void RegisterCodec() => NDArrayPythonInterop.RegisterCodec();

        [TestMethod]
        public void RebindingANameThroughTheCodec_KeepsTheLiveExportCountBounded()
        {
            int baseline = NDArrayPythonInterop.LiveExports;
            int peak = RebindManyTimes(150);
            (peak - baseline).Should().BeLessThanOrEqualTo(2, "each encode disposes the previous encode's wrapper; the export dies with the rebinding");
            PyExec("del w");
            WaitFor(() => NDArrayPythonInterop.LiveExports == baseline).Should().BeTrue();
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private int RebindManyTimes(int rounds)
        {
            int peak = 0;
            using (Gil())
            {
                dynamic py = Scope;
                for (int i = 0; i < rounds; i++)
                using (NDScope.Open())
                {
                    NDArray window = np.arange(16).astype(NPTypeCode.Double) + i;
                    py.w = window;                                      // an implicit encode: the codec's wrapper is handed to pythonnet
                    peak = Math.Max(peak, NDArrayPythonInterop.LiveExports);
                }
            }

            return peak;
        }

        [TestMethod]
        public void TheHandedOffWrapperIsRedundant_PythonnetHoldsItsOwnReference()
        {
            using var scope = NDScope.Open();
            NDArray first = np.arange(3).astype(NPTypeCode.Double);
            NDArray second = np.arange(4).astype(NPTypeCode.Double);
            using (Gil())
            {
                Scope.Exec("import sys");
                Scope.Set("a", first);                                   // the handoff slot now holds a's wrapper
                Scope.Set("b", second);                                  // ...which this encode disposes
                PyLong("sys.getrefcount(a)").Should().Be(2, "the namespace and getrefcount's own argument — no lingering wrapper");
                PyLong("sys.getrefcount(b)").Should().Be(3, "the namespace, getrefcount's argument, and the one wrapper still in the slot");
                PyStr("a.tolist()").Should().Be("[0.0, 1.0, 2.0]", "a is alive and intact: pythonnet's reference carried it, not the wrapper's");
                Scope.Exec("a[1] = 9.0");
                first.GetDouble(1).Should().Be(9.0, "and it is still the shared view");
                Scope.Exec("del a, b");
            }
        }

        [TestMethod]
        public void TupleEncodes_AreHandedOffToo()
        {
            using (Gil())
            {
                Scope.Exec("import sys");
                Scope.Set("t1", (1234567, 7654321));
                Scope.Set("t2", (2345678, 8765432));                     // disposes t1's wrapper
                PyLong("sys.getrefcount(t1)").Should().Be(2);
                PyStr("repr(t1)").Should().Be("(1234567, 7654321)");
                Scope.Exec("del t1, t2");
            }
        }

        [TestMethod]
        public void DynamicCallArguments_AreHandedOffToo()
        {
            int baseline = NDArrayPythonInterop.LiveExports;
            int peak = CallManyTimes(100);
            (peak - baseline).Should().BeLessThanOrEqualTo(2, "an NDArray argument's wrapper is released by the next encode");
            WaitFor(() => NDArrayPythonInterop.LiveExports == baseline).Should().BeTrue();
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private int CallManyTimes(int rounds)
        {
            int peak = 0;
            using (Gil())
            {
                dynamic numpy = Py.Import("numpy");
                for (int i = 0; i < rounds; i++)
                using (NDScope.Open())
                {
                    NDArray batch = np.arange(8).astype(NPTypeCode.Double) + i;
                    double s = numpy.sum(batch);                         // the argument is encoded implicitly; nothing keeps the view but the call
                    s.Should().Be(28.0 + 8 * i);
                    peak = Math.Max(peak, NDArrayPythonInterop.LiveExports);
                }
            }

            return peak;
        }
    }
}
