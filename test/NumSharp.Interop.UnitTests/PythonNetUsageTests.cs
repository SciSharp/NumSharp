using System;
using System.Collections.Concurrent;
using System.Threading;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     The unit-test-proven Python.NET cookbook. Every section of the docs page
    ///     "Interoperability → Python.NET" (docs/website-src/docs/interop/python-net.md) corresponds
    ///     1:1 to a test here — if a documented pattern breaks, this suite fails. The patterns are
    ///     idiomatic Python.NET (scopes, dynamic modules, runtime-defined functions and classes,
    ///     iterators, exceptions, threads) with NumSharp arrays flowing through them; the base-class
    ///     gate additionally proves every pattern leaks nothing.
    /// </summary>
    [TestClass]
    public class PythonNetUsageTests : InteropTestBase
    {
        [TestMethod]
        public void Embedding_ScopeVariables_FlowBothWays()
        {
            var nd = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);

            using (Gil())
            {
                using (PyObject x = nd.ToNumpy())
                    Scope.Set("x", x);

                Scope.Exec("total = float(x.sum())\nx[0, 0] = 100.0");
            }

            PyFloat("total").Should().BeApproximately(15.0, 1e-9, "python computed over the shared buffer");
            ReadAt<double>(nd, 0, 0).Should().BeApproximately(100.0, 1e-12, "python's write landed in NumSharp's memory");

            WriteAt(nd, -2.5, 1, 2);
            PyFloat("float(x[1, 2])").Should().BeApproximately(-2.5, 1e-12, "NumSharp's write is visible to the scope variable");
        }

        [TestMethod]
        public void DynamicModules_NumpyFunctions_OnNumSharpArrays()
        {
            CodecTests.EnsureCodec();

            var a = np.arange(4).astype(NPTypeCode.Double).reshape(2, 2);
            var b = (np.arange(4).astype(NPTypeCode.Double) + 1).reshape(2, 2);

            NDArray product;
            using (Gil())
            {
                dynamic numpy = Py.Import("numpy");
                product = ((PyObject)numpy.matmul(a, b)).As<NDArray>();   // NDArrays in, NDArray out
            }

            var expected = np.matmul(a, b);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    ReadAt<double>(product, i, j).Should().BeApproximately(ReadAt<double>(expected, i, j), 1e-9,
                        "python's matmul must agree with NumSharp's own matmul");
        }

        [TestMethod]
        public void PythonStdlib_JsonRoundtrip()
        {
            var nd = (np.arange(5).astype(NPTypeCode.Double) * 1.5).reshape(5);
            ExportTo("jx", nd);

            PyExec("import json\npayload = json.dumps(jx.tolist())");
            PyStr("payload").Should().Be("[0.0, 1.5, 3.0, 4.5, 6.0]");

            var back = ImportOf("np.asarray(json.loads(payload), dtype='f8')");
            back.typecode.Should().Be(NPTypeCode.Double);
            for (int i = 0; i < 5; i++)
                ReadAt<double>(back, i).Should().BeApproximately(1.5 * i, 1e-12, "the JSON round trip must be lossless");
        }

        [TestMethod]
        public void RuntimeDefinedFunction_TypedResults()
        {
            CodecTests.EnsureCodec();
            PyExec("def zscore(a):\n    return (a - a.mean()) / a.std()");

            var samples = np.arange(8).astype(NPTypeCode.Double);
            NDArray scored;
            using (Gil())
            {
                dynamic zscore = Scope.Get("zscore");
                scored = ((PyObject)zscore(samples)).As<NDArray>();
            }

            var expected = (samples - np.mean(samples)) / np.std(samples);
            for (int i = 0; i < 8; i++)
                ReadAt<double>(scored, i).Should().BeApproximately(ReadAt<double>(expected, i), 1e-9);
        }

        [TestMethod]
        public void PythonClass_StoresArrays_SeesLaterMutations()
        {
            CodecTests.EnsureCodec();
            PyExec(
                "class Accumulator:\n" +
                "    def __init__(self):\n" +
                "        self.batches = []\n" +
                "    def add(self, arr):\n" +
                "        self.batches.append(arr)\n" +          // stores the SHARED view long-term
                "    def total(self):\n" +
                "        return float(sum(b.sum() for b in self.batches))\n");

            var first = np.arange(4).astype(NPTypeCode.Double);        // sum 6
            var second = np.arange(4).astype(NPTypeCode.Double) * 10;  // sum 60

            using (Gil())
            {
                dynamic acc = Scope.Eval("Accumulator()");
                acc.add(first);
                acc.add(second);
                ((double)acc.total()).Should().BeApproximately(66.0, 1e-9);

                // The instance stored VIEWS: a later C#-side mutation is visible on the next call.
                WriteAt(first, 100.0, 0);
                ((double)acc.total()).Should().BeApproximately(166.0, 1e-9,
                    "the python object holds live views, not snapshots");

                Scope.Set("acc", (PyObject)acc);
            }

            PyExec("del acc");   // dropping the instance releases the stored views (leak gate verifies)
        }

        [TestMethod]
        public void PythonExceptions_SurfaceAsPythonException_EngineStaysUsable()
        {
            ExportTo("ex", np.arange(3).astype(NPTypeCode.Double));

            ((Action)(() => PyExec("raise ValueError('bad shape: ' + str(ex.shape))")))
                .Should().Throw<PythonException>().WithMessage("*bad shape: (3,)*");

            // The failed call must leave the engine and the conversion fully usable.
            PyExec("ex[0] = 7.5");
            PyFloat("float(ex.sum())").Should().BeApproximately(10.5, 1e-9);
        }

        [TestMethod]
        public void Generators_StreamChunksIntoNumSharp()
        {
            PyExec(
                "def chunks(n):\n" +
                "    for i in range(n):\n" +
                "        yield np.arange(3, dtype='f8') + 10 * i\n");

            var rows = new System.Collections.Generic.List<NDArray>();
            using (Gil())
            {
                using PyObject generator = Scope.Eval("chunks(4)");
                using var iterator = PyIter.GetIter(generator);
                while (iterator.MoveNext())
                {
                    using PyObject chunk = iterator.Current;
                    rows.Add(PythonConvert.ToNDArray(chunk));   // each yielded array copied out
                }
            }

            rows.Count.Should().Be(4);
            var stacked = np.vstack(rows.ToArray());
            stacked.shape[0].Should().Be(4);
            stacked.shape[1].Should().Be(3);
            ReadAt<double>(stacked, 3, 2).Should().BeApproximately(32.0, 1e-12, "last chunk, last element = 30 + 2");
        }

        [TestMethod]
        public void ManualPyObjectWork_ListsAndDicts_MixedWithConversions()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);

            using (Gil())
            {
                using var record = new PyDict();
                using (PyObject arr = nd.ToNumpy())
                    record["data"] = arr;
                using (var tag = new PyString("run-42"))
                    record["tag"] = tag;

                Scope.Set("record", record);
            }

            PyStr("record['tag']").Should().Be("run-42");
            PyFloat("float(record['data'].sum())").Should().BeApproximately(6.0, 1e-9);

            PyExec("record['data'][1] = 41.0");
            ReadAt<double>(nd, 1).Should().BeApproximately(41.0, 1e-12, "the dict entry is the same shared buffer");
        }

        [TestMethod]
        public void ModuleCache_StoredDynamicReference_ReusedAcrossCalls()
        {
            PyObject numpyModule;
            using (Gil())
                numpyModule = Py.Import("numpy");   // stored once, reused many times (typical app pattern)

            try
            {
                for (int i = 1; i <= 5; i++)
                {
                    var nd = np.arange(i * 3).astype(NPTypeCode.Double);
                    using (Gil())
                    {
                        dynamic numpy = numpyModule;
                        using PyObject arr = nd.ToNumpy();
                        double mean = (double)numpy.mean(arr);
                        mean.Should().BeApproximately((i * 3 - 1) / 2.0, 1e-9, $"call {i} through the cached module");
                    }
                }
            }
            finally
            {
                using (Gil())
                    numpyModule.Dispose();
            }
        }

        [TestMethod]
        public void InPlaceUfuncs_WriteThroughTheSharedView()
        {
            var nd = np.arange(6).astype(NPTypeCode.Double);
            ExportTo("u", nd);

            using (Gil())
            {
                dynamic numpy = Py.Import("numpy");
                dynamic u = Scope.Get("u");
                numpy.add(u, 1.0, @out: u);        // classic numpy in-place call, straight into NumSharp memory
                numpy.clip(u, 2.0, 5.0, @out: u);
            }

            double[] expected = { 2, 2, 3, 4, 5, 5 };
            for (int i = 0; i < 6; i++)
                ReadAt<double>(nd, i).Should().BeApproximately(expected[i], 1e-12, $"element {i} after add+clip in place");
        }

        [TestMethod]
        public void PythonLists_ToAndFromArrays()
        {
            // python list -> NDArray (through numpy's asarray)
            var fromList = ImportOf("np.asarray([2.5, 3.5, 4.5])");
            fromList.typecode.Should().Be(NPTypeCode.Double);
            ReadAt<double>(fromList, 2).Should().BeApproximately(4.5, 1e-12);

            // NDArray -> python list
            ExportTo("pl", np.arange(4).astype(NPTypeCode.Int64) * 2);
            PyExec("as_list = pl.tolist()");
            PyLong("len(as_list)").Should().Be(4);
            PyStr("as_list").Should().Be("[0, 2, 4, 6]");
        }

        [TestMethod]
        public void WorkerThreads_CallPythonWithTheirOwnArrays()
        {
            CodecTests.EnsureCodec();
            PyExec("def weigh(a):\n    return float((a * 0.5).sum())");

            var failures = new ConcurrentQueue<Exception>();
            var threads = new Thread[3];
            for (int t = 0; t < threads.Length; t++)
            {
                int id = t;
                threads[t] = new Thread(() =>
                {
                    try
                    {
                        for (int k = 0; k < 10; k++)
                        {
                            var batch = np.arange(6).astype(NPTypeCode.Double) + id;   // sum 15 + 6*id
                            double weighed;
                            using (Py.GIL())
                            {
                                dynamic weigh = Scope.Get("weigh");
                                weighed = (double)weigh(batch);
                            }

                            weighed.Should().BeApproximately((15.0 + 6 * id) * 0.5, 1e-9);
                        }
                    }
                    catch (Exception e)
                    {
                        failures.Enqueue(e);
                    }
                }) { IsBackground = true };
            }

            foreach (var t in threads) t.Start();
            foreach (var t in threads) t.Join(30_000).Should().BeTrue("python calls from workers must not deadlock");
            failures.Should().BeEmpty(string.Join(" | ", failures));
        }
    }
}
