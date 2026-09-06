using System;
using System.Threading;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/01-bootstrap.cs</c>, section by section.</summary>
    [TestClass]
    public class Example01_Bootstrap : ExampleTestBase
    {
        [TestMethod]
        public void Section1_ThePairing_EveryPythonnetHardCapsThePythonItCanDrive()
        {
            Version pythonnet = typeof(PythonEngine).Assembly.GetName().Version;
            Version python = Version.Parse(PythonEngine.Version.Split(' ')[0]);
            Version floor = PythonEngine.MinSupportedVersion;
            Version ceiling = PythonEngine.MaxSupportedVersion;

            PythonEngine.IsInitialized.Should().BeTrue();
            pythonnet.Major.Should().Be(3);
            floor.Should().Be(new Version(3, 7), "the same for the whole pythonnet 3 line");
            ceiling.Should().BeGreaterThanOrEqualTo(new Version(3, 13), "the pinned 3.0.5 drives up to 3.13");
            (python >= new Version(floor.Major, floor.Minor)).Should().BeTrue();
            (new Version(python.Major, python.Minor) <= ceiling).Should().BeTrue("the interop re-checks this range on first use");
        }

        [TestMethod]
        public void Section2_CodecRegistration_IsIdempotentPerEngineSession()
        {
            // The script's PythonHost.Start(codec: false) lets it see the first registration return true; here the
            // suite registered already, so the idempotency half is pinned (the ordering trap is Example07's subject).
            NDArrayPythonInterop.RegisterCodec().Should().BeFalse("already registered for this engine session");
            NDArrayPythonInterop.RegisterCodec().Should().BeFalse();
        }

        [TestMethod]
        public void Section3_OneBufferBothSides()
        {
            using var scope = NDScope.Open();
            NDArray nd = np.arange(6).reshape(2, 3);
            NDArray copy, view;
            using (Gil())
            {
                py.x = nd;                                           // zero-copy: the codec ran ToNumpy
                py.Exec("x[1, 2] = 99");
                nd.GetInt64(1, 2).Should().Be(99, "Python's write landed in NumSharp");

                py.Exec("r = np.sin(x / 3.0)");
                using PyObject r = py.r;
                copy = r.ToNDArray();
                view = r.AsNDArray();
                copy.typecode.Should().Be(NPTypeCode.Double);
                copy.shape.Should().Equal(2, 3);
                view.Shape.IsWriteable.Should().BeTrue("a view of a fresh numpy result is writable");

                view[0, 0] = -1.0;
                double r00 = py.Eval("r[0, 0]");
                r00.Should().Be(-1.0, "numpy sees the write through the view");
            }
        }

        [TestMethod]
        public void Section4_BeginAllowThreads_LetsAnyThreadConvert()
        {
            double sum = double.NaN;
            Exception failure = null;
            var worker = new Thread(() =>
            {
                try
                {
                    using NDArray batch = np.arange(4).astype(np.float64);   // outside the main thread's scope: disposed here
                    using (Py.GIL())
                    {
                        dynamic p = batch.ToNumpy();
                        sum = p.sum();
                    }
                }
                catch (Exception e) { failure = e; }
            });
            worker.Start();
            worker.Join(30_000).Should().BeTrue();
            failure.Should().BeNull("a thread that never touched Python converts under its own Py.GIL()");
            sum.Should().Be(6.0);
        }

        [TestMethod]
        public void Section5_WhoFreesWhat_TheScopeThenTheNames()
        {
            int exports0 = NDArrayPythonInterop.LiveExports, imports0 = NDArrayPythonInterop.LiveImports;
            using (NDScope.Open())
            using (Gil())
            {
                NDArray nd = np.arange(6).reshape(2, 3);
                py.x = nd;
                py.Exec("r = np.sin(x / 3.0)");
                using PyObject r = py.r;
                NDArray view = r.AsNDArray();
                NDArrayPythonInterop.LiveExports.Should().Be(exports0 + 1, "one export pin while x is bound");
                NDArrayPythonInterop.LiveImports.Should().Be(imports0 + 1, "one lease while the view lives");
            }                                                        // the scope released nd and view

            PyExec("del x, r");                                      // the names died
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports0 && NDArrayPythonInterop.LiveImports == imports0)
                .Should().BeTrue("both counters return to where they were");
        }
    }
}
