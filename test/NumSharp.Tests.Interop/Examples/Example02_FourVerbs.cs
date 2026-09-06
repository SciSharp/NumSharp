using System;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/02-four-verbs.cs</c>, section by section.</summary>
    [TestClass]
    public class Example02_FourVerbs : ExampleTestBase
    {
        [TestMethod]
        public void ToNumpy_IsAView_ToNumpyCopy_IsACopy()
        {
            using var scope = NDScope.Open();
            NDArray nd = np.arange(4).astype(np.float64);
            using (Gil())
            {
                py.v = nd.ToNumpy();
                py.c = nd.ToNumpyCopy();
                py.Exec("v[0] = 11.0; c[1] = 22.0");
                nd.GetDouble(0).Should().Be(11.0, "the view's write arrived");
                nd.GetDouble(1).Should().Be(1.0, "the copy's did not");

                bool shared = py.np.shares_memory(py.v, py.c);
                bool viewOwnsData = py.v.flags.owndata;
                bool copyOwnsData = py.c.flags.owndata;
                shared.Should().BeFalse();
                viewOwnsData.Should().BeFalse("numpy agrees the view's memory is NumSharp's");
                copyOwnsData.Should().BeTrue();

                nd[2] = -5.0;
                double v2 = py.v[2];
                double c2 = py.c[2];
                v2.Should().Be(-5.0, "the view shows NumSharp's write");
                c2.Should().Be(2.0, "the copy does not");
            }
        }

        [TestMethod]
        public void TheAliases_ToPythonIsTheView_CopyTrueIsTheCopy()
        {
            using var scope = NDScope.Open();
            NDArray nd = np.arange(4).astype(np.float64);
            using (Gil())
            {
                py.v = nd.ToNumpy();
                py.p = nd.ToPython();
                py.r = nd.ToNumpy(copy: true);
                bool pIsTheView = py.np.shares_memory(py.p, py.v);
                bool rIsACopy = py.np.shares_memory(py.r, py.v);
                pIsTheView.Should().BeTrue();
                rIsACopy.Should().BeFalse();
            }
        }

        [TestMethod]
        public void TheEverydaySpelling_BindingByNameIsToNumpy()
        {
            using var scope = NDScope.Open();
            NDArray nd = np.arange(4).astype(np.float64);
            using (Gil())
            {
                py.v = nd.ToNumpy();
                py.w = nd;                                           // no verb, no PyObject
                bool wIsTheView = py.np.shares_memory(py.w, py.v);
                wIsTheView.Should().BeTrue("the codec exported the view");
            }
        }

        [TestMethod]
        public void ToNDArray_IsACopy_AsNDArray_IsAView()
        {
            using var scope = NDScope.Open();
            NDArray copied, leased;
            using (Gil())
            {
                py.Exec("src = np.arange(6, dtype='f8').reshape(2, 3) * 0.5");
                using PyObject src = py.src;
                copied = src.ToNDArray();
                leased = src.AsNDArray();
                copied.Shape.IsContiguous.Should().BeTrue();
                copied.typecode.Should().Be(NPTypeCode.Double);

                copied[0, 0] = -9.0;
                double afterCopyWrite = py.Eval("src[0, 0]");
                afterCopyWrite.Should().Be(0.0, "the copy never reaches numpy");
                leased[0, 0] = -9.0;
                double afterViewWrite = py.Eval("src[0, 0]");
                afterViewWrite.Should().Be(-9.0, "the view does");
                py.Exec("src[1, 2] = 42.0");
                leased.GetDouble(1, 2).Should().Be(42.0, "numpy's write is visible through the view");
                NDArrayPythonInterop.LiveImports.Should().BeGreaterThanOrEqualTo(1, "one lease while the view is alive");
            }
        }

        [TestMethod]
        public void TheEverydaySpelling_ReadingByNameIsAsNDArray()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("src = np.arange(6, dtype='f8').reshape(2, 3) * 0.5\nsrc[1, 2] = 42.0");
                NDArray viaCodec = py.src;
                viaCodec.GetDouble(1, 2).Should().Be(42.0);
                viaCodec[0, 0] = -3.0;
                double src00 = py.Eval("src[0, 0]");
                src00.Should().Be(-3.0, "Auto mode took a view");
            }
        }

        [TestMethod]
        public void NonContiguousSources_CopiesLinearize_ViewsKeepStrides()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("src = np.arange(6, dtype='f8').reshape(2, 3) * 0.5\nsrc[1, 2] = 42.0");
                using PyObject t = py.src.T;
                NDArray tCopy = t.ToNDArray();
                NDArray strided = t.AsNDArray();
                tCopy.Shape.IsContiguous.Should().BeTrue();
                tCopy.GetDouble(2, 1).Should().Be(42.0, "logical order preserved");
                strided.shape.Should().Equal(3, 2);
                strided.Shape.Strides.Should().Equal(new long[] { 1, 3 }, "the layout numpy has, in elements");
            }
        }

        [TestMethod]
        public void ReadOnlySources_RefusedByDefault_NonWriteableWithAllowReadonly()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject ro = py.Eval("b'\\x01\\x02\\x03\\x04'");
                Exception refused = Throws(() => ro.AsNDArray());
                refused.Should().BeOfType<InvalidOperationException>();
                refused.Message.Should().Contain("read-only").And.Contain("allowReadonly:true");

                NDArray view = ro.AsNDArray(allowReadonly: true);
                view.Shape.IsWriteable.Should().BeFalse();
                view.GetByte(1).Should().Be(2);
                Exception guarded = Throws(() => view[0] = (byte)7);
                guarded.Should().NotBeNull("a guarded write throws instead of corrupting Python's bytes");
                guarded.Message.Should().Contain("read-only");

                NDArray owned = ro.ToNDArray();
                owned.Shape.IsWriteable.Should().BeTrue("a copy is an owning, writable array");
            }
        }

        [TestMethod]
        public void AnImportView_OwnsNothing()
        {
            using var scope = NDScope.Open();
            NDArray leased;
            using (Gil())
            {
                py.Exec("src = np.arange(6, dtype='f8').reshape(2, 3) * 0.5");
                leased = py.src;
            }

            Exception cannotResize = Throws(() => leased.resize(new Shape(12)));
            cannotResize.Should().NotBeNull();
            cannotResize.Message.Should().Contain("does not own its data");

            NDArray owning = np.require(leased, (Type)null, "O");
            owning[0, 0] = 123.0;
            PyFloat("src[0, 0]").Should().Be(0.0, "the owning copy is detached");
        }

        [TestMethod]
        public void ArrayLikes_FromArrayLike_GoesThroughNumpyAsarray()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject list = py.Eval("[[1, 2, 3], [4, 5, 6]]");
                NDArray fromList = list.FromArrayLike();
                fromList.typecode.Should().Be(NPTypeCode.Int64);
                fromList.shape.Should().Equal(2, 3);
                fromList.GetInt64(1, 2).Should().Be(6);

                using PyObject scalar = py.Eval("2.5");
                NDArray fromScalar = scalar.FromArrayLike();
                fromScalar.ndim.Should().Be(0);
                fromScalar.GetDouble().Should().Be(2.5);

                Throws(() => list.AsNDArray()).Should().BeOfType<NotSupportedException>("a list exports no PEP 3118 buffer");

                NDArray implicitly = py.Eval("[[1, 2, 3], [4, 5, 6]]");
                implicitly.Shape.IsWriteable.Should().BeTrue("the codec's array-like decode is always an owning copy");
                implicitly.GetInt64(1, 2).Should().Be(6);
            }
        }

        [TestMethod]
        public void WhoFreesWhat_TheScopeReleasesTheLeases_TheNamesReleaseThePins()
        {
            int exports0 = NDArrayPythonInterop.LiveExports, imports0 = NDArrayPythonInterop.LiveImports;
            CrossAndLetTheScopeClose(exports0, imports0);
            PyExec("del v, w, src");                                // the names died
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports0 && NDArrayPythonInterop.LiveImports == imports0)
                .Should().BeTrue("the scope released the lease, the names released the pins");
        }

        /// <summary>
        ///     Its own frame on purpose: the CLR wrappers a `dynamic` read hands out (`py.src`, the temporary from
        ///     `nd.ToNumpy()`) hold Python references until the GC collects them, and a Debug JIT keeps a frame's
        ///     temporaries alive until the method returns — so the pump in the caller can only free them once this
        ///     method has returned.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private void CrossAndLetTheScopeClose(int exports0, int imports0)
        {
            using (NDScope.Open())
            using (Gil())
            {
                NDArray nd = np.arange(4).astype(np.float64);
                py.v = nd.ToNumpy();
                py.w = nd;
                py.Exec("src = np.arange(6, dtype='f8')");
                NDArray leased = py.src;
                NDArrayPythonInterop.LiveExports.Should().Be(exports0 + 2);
                NDArrayPythonInterop.LiveImports.Should().Be(imports0 + 1);
            }
        }
    }
}
