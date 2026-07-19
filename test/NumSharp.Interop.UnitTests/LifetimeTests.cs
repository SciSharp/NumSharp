using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     The rock-solid core: memory must stay alive exactly as long as EITHER side can still see it,
    ///     and must be released (observably, via <see cref="NDArrayPythonInterop.LiveExports"/> /
    ///     <see cref="NDArrayPythonInterop.LiveImports"/>) once neither can. Collection-dependent lifecycles
    ///     run inside NoInlining helpers — see <see cref="InteropTestBase"/> for why.
    /// </summary>
    [TestClass]
    public class LifetimeTests : InteropTestBase
    {
        [TestMethod]
        public void OrphanedExport_SurvivesFullClrCollection()
        {
            int e0 = NDArrayPythonInterop.LiveExports;

            [MethodImpl(MethodImplOptions.NoInlining)]
            void CreateOrphan()
            {
                var nd = np.arange(10).astype(NPTypeCode.Double);
                ExportTo("orph", nd);
                // nd and the PyObject wrapper both die here — only python's reference remains.
            }

            CreateOrphan();
            Pump(); Pump(); Pump();

            PyFloat("float(orph.sum())").Should().BeApproximately(45.0, 1e-9,
                "the numpy view must still read valid NumSharp memory after every C# reference is gone");
            NDArrayPythonInterop.LiveExports.Should().Be(e0 + 1, "the buffer is rooted for python's sake");

            PyExec("del orph");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0).Should().BeTrue(
                "dropping the last python reference must release the NumSharp buffer pin");
        }

        [TestMethod]
        public void Export_IsRootedByDerivedPythonViews_NotJustTheReturnedArray()
        {
            int e0 = NDArrayPythonInterop.LiveExports;

            [MethodImpl(MethodImplOptions.NoInlining)]
            void CreateAndSlicePythonSide()
            {
                ExportTo("x", np.arange(10).astype(NPTypeCode.Double));
                PyExec("v = x[4:]");   // derived numpy view chains .base to the export's buffer object
                PyExec("del x");       // the ORIGINAL exported array dies; only the derived view remains
            }

            CreateAndSlicePythonSide();
            Pump(); Pump();

            NDArrayPythonInterop.LiveExports.Should().Be(e0 + 1, "a derived python view must keep the buffer rooted");
            PyStr("v.tolist()").Should().Be("[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]", "and its data must still be valid");

            PyExec("del v");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0).Should().BeTrue("the last derived view releases the pin");
        }

        [TestMethod]
        public void ImportLease_IsExtendedByDerivedNumSharpViews()
        {
            int i0 = NDArrayPythonInterop.LiveImports;
            PyExec("src = np.arange(8, dtype='f8')");

            [MethodImpl(MethodImplOptions.NoInlining)]
            NDArray MakeDerived()
            {
                var original = ViewOf("src");
                return original["2:5"];   // original dies with this frame; the slice shares its memory block
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            void UseDerivedAlone()
            {
                var derived = MakeDerived();
                Pump(); Pump();
                NDArrayPythonInterop.LiveImports.Should().Be(i0 + 1,
                    "the derived slice alone must keep the Python buffer leased after the original NDArray is collected");
                WriteAt(derived, -7.5, 0);
                PyFloat("float(src[2])").Should().BeApproximately(-7.5, 1e-12, "the derived view still aliases python memory");
            }

            UseDerivedAlone();
            WaitFor(() => NDArrayPythonInterop.LiveImports == i0).Should().BeTrue(
                "collecting the last NumSharp view must release the lease");
        }

        [TestMethod]
        public void ImportLease_KeepsPythonExporterAlive_AfterPythonForgetsIt()
        {
            int i0 = NDArrayPythonInterop.LiveImports;
            PyExec("import weakref\nkeep = np.arange(6, dtype='f8') * 1.5\nwr = weakref.ref(keep)");

            [MethodImpl(MethodImplOptions.NoInlining)]
            void HoldWhilePythonForgets()
            {
                var view = ViewOf("keep");
                PyExec("del keep");
                Pump(); Pump();

                PyBool("wr() is not None").Should().BeTrue(
                    "the lease must keep the exporter alive after python dropped its last reference");
                for (int k = 0; k < 6; k++)
                    ReadAt<double>(view, k).Should().BeApproximately(k * 1.5, 1e-12,
                        "the memory must NOT have been freed while NumSharp can still see it");
            }

            HoldWhilePythonForgets();
            WaitFor(() => NDArrayPythonInterop.LiveImports == i0).Should().BeTrue();
            Pump();
            PyBool("wr() is None").Should().BeTrue(
                "once the last NumSharp view is gone the exporter must actually be freed — no leak");
        }

        [TestMethod]
        public void ExplicitDispose_ReleasesTheLeaseDeterministically()
        {
            int i0 = NDArrayPythonInterop.LiveImports;
            PyExec("dv = np.arange(4, dtype='i8')");

            var nd = ViewOf("dv");
            NDArrayPythonInterop.LiveImports.Should().Be(i0 + 1);

            nd.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == i0, 4000).Should().BeTrue(
                "Dispose must release the lease without waiting for a garbage collection");
        }

        [TestMethod]
        public void NumSharpResize_IsBlockedByTheExportPin()
        {
            int e0 = NDArrayPythonInterop.LiveExports;
            var rz = np.arange(4);
            ExportTo("rz", rz);

            // NumSharp's ndarray.resize refcheck consults the same ARC refcount the export pins,
            // mirroring numpy's "referenced by another array" guard.
            ((Action)(() => rz.resize(new Shape(8))))
                .Should().Throw<Exception>().Which.GetType().Name.Should().Be("IncorrectShapeException");

            PyExec("del rz");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0).Should().BeTrue();

            rz.resize(new Shape(8));
            rz.size.Should().Be(8, "after python released the pin, resize must succeed");
        }

        [TestMethod]
        public void GcHammer_EverythingReturnsToBaseline()
        {
            int e0 = NDArrayPythonInterop.LiveExports, i0 = NDArrayPythonInterop.LiveImports;

            [MethodImpl(MethodImplOptions.NoInlining)]
            void HammerOnce(int i)
            {
                var a = np.arange(32).astype(NPTypeCode.Double);
                using (Py.GIL())
                {
                    using var p = NDArrayPythonInterop.ToNumpy(a);      // export; wrapper dropped immediately
                }

                PyExec("h = np.arange(16, dtype='i8')");
                var view = ViewOf("h");
                var sub = view["4:12"];
                WriteAt(sub, (long)i, 0);

                var c = np.arange(8).astype(NPTypeCode.Complex);
                using (Py.GIL())
                {
                    using var pc = NDArrayPythonInterop.ToNumpyCopy(c);
                }
            }

            for (int i = 0; i < 120; i++)
            {
                HammerOnce(i);
                if (i % 40 == 39) Pump();
            }

            PyExec("del h");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0 && NDArrayPythonInterop.LiveImports == i0, 20_000)
                .Should().BeTrue($"hammer must not leak (exports {NDArrayPythonInterop.LiveExports}, imports {NDArrayPythonInterop.LiveImports})");
        }

        [TestMethod]
        public void ConcurrentConversions_AreThreadSafe()
        {
            int e0 = NDArrayPythonInterop.LiveExports, i0 = NDArrayPythonInterop.LiveImports;
            var failures = new ConcurrentQueue<Exception>();

            void Worker(int id)
            {
                try
                {
                    for (int k = 0; k < 30; k++)
                    {
                        var nd = np.arange(16).astype(NPTypeCode.Double) + id;
                        ExportTo($"t{id}", nd);
                        PyFloat($"float(t{id}.sum())").Should().BeApproximately(120 + 16.0 * id, 1e-9);

                        PyExec($"s{id} = np.arange(8, dtype='f8') * {id + 1}");
                        var view = ViewOf($"s{id}");
                        WriteAt(view, -1.0, 0);
                        PyFloat($"float(s{id}[0])").Should().BeApproximately(-1.0, 1e-12);
                        view.Dispose();

                        var back = ImportOf($"t{id}");
                        ReadAt<double>(back, 1).Should().BeApproximately(1 + id, 1e-12);
                    }
                }
                catch (Exception e)
                {
                    failures.Enqueue(e);
                }
            }

            var threads = new Thread[4];
            for (int t = 0; t < threads.Length; t++)
            {
                int id = t;
                threads[t] = new Thread(() => Worker(id)) { IsBackground = true };
                threads[t].Start();
            }

            foreach (var t in threads)
                t.Join(60_000).Should().BeTrue("workers must not deadlock");

            failures.Should().BeEmpty(string.Join(" | ", failures));
            PyExec("del t0, t1, t2, t3, s0, s1, s2, s3");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0 && NDArrayPythonInterop.LiveImports == i0, 20_000)
                .Should().BeTrue($"concurrent churn must drain (exports {NDArrayPythonInterop.LiveExports}, imports {NDArrayPythonInterop.LiveImports})");
        }

        [TestMethod]
        public void LiveCounters_TrackConversionsExactly()
        {
            int e0 = NDArrayPythonInterop.LiveExports, i0 = NDArrayPythonInterop.LiveImports;

            var nd = np.arange(3);
            ExportTo("c1", nd);
            NDArrayPythonInterop.LiveExports.Should().Be(e0 + 1);

            PyExec("c2 = np.arange(3, dtype='i8')");
            var view = ViewOf("c2");
            NDArrayPythonInterop.LiveImports.Should().Be(i0 + 1);

            view.Dispose();
            PyExec("del c1");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0 && NDArrayPythonInterop.LiveImports == i0)
                .Should().BeTrue();
        }
    }
}
