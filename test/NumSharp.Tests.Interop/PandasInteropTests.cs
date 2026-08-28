using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Happy-path proof against Pandas 3.0.5. Pandas objects enter through
    ///     <see cref="PandasPythonArrayAdapter"/>, while all dtype/layout/lifetime behavior remains in
    ///     the ordinary <see cref="NDArrayPythonInterop"/> memory bridge.
    /// </summary>
    [TestClass]
    public class PandasInteropTests : InteropTestBase
    {
        [TestMethod]
        public void Runtime_IsLatestStablePandas305()
        {
            RequirePandas();
            PyStr("pd.__version__").Should().Be(PandasTestGate.ValidatedPandasVersion);
        }

        [TestMethod]
        public void BuiltInAdapter_RecognizesFrameSeriesIndexExtensionAndSubclasses_NotScalars()
        {
            RequirePandas();
            var codec = new NumpyCodec();
            PythonArrayAdapterRegistry.Register(PandasPythonArrayAdapter.Instance).Should().BeFalse(
                "the Pandas adapter is built in");
            PyExec(
                "class NumSharpPandasFrame(pd.DataFrame):\n" +
                "    pass\n" +
                "pd_adapter_objects = [\n" +
                "    pd.DataFrame({'a': [1]}), pd.Series([1]), pd.Index([1]),\n" +
                "    pd.RangeIndex(2), pd.array([1, 2], dtype='Int64'),\n" +
                "    NumSharpPandasFrame({'a': [1]})]\n" +
                "pd_adapter_scalar = pd.Timestamp('2020-01-01')\n" +
                "pd_adapter_impostor = type('DataFrame', (), {'__module__': 'pandas_fake'})()");

            using (Gil())
            using (PyObject objects = Scope.Get("pd_adapter_objects"))
            using (var list = new PyList(objects))
            {
                for (int i = 0; i < list.Length(); i++)
                {
                    using PyObject value = list[i];
                    using PyObject typeObject = value.GetPythonType();
                    using var type = new PyType(typeObject);
                    codec.CanDecode(type, typeof(NDArray)).Should().BeTrue($"adapter object {i}");
                }

                using PyObject scalar = Scope.Get("pd_adapter_scalar");
                using PyObject scalarTypeObject = scalar.GetPythonType();
                using var scalarType = new PyType(scalarTypeObject);
                codec.CanDecode(scalarType, typeof(NDArray)).Should().BeFalse(
                    "Pandas scalar labels are not arrays");

                using PyObject impostor = Scope.Get("pd_adapter_impostor");
                using PyObject impostorTypeObject = impostor.GetPythonType();
                using var impostorType = new PyType(impostorTypeObject);
                codec.CanDecode(impostorType, typeof(NDArray)).Should().BeFalse(
                    "a module merely prefixed with pandas must not impersonate the real package");
            }
        }

        [TestMethod]
        public unsafe void NumSharpToPandas_ImplicitEncoderSharesInput_WhenCopyFalseIsExplicit()
        {
            RequirePandas();
            CodecTests.EnsureCodec();
            using NDArray source = np.arange(6).astype(NPTypeCode.Double);

            using (Gil())
            {
                Scope.Set("pd_implicit_source", source); // registered encoder: NDArray -> NumPy
                Scope.Exec("pd_implicit_series = pd.Series(pd_implicit_source, copy=False)");
            }

            PyLong("pd_implicit_series.to_numpy().ctypes.data").Should().Be((long)source.Storage.Address);
            PyBool("pd_implicit_series.to_numpy().flags.writeable").Should().BeFalse(
                "Pandas 3 Copy-on-Write exposes a shared NumPy projection as read-only");

            WriteAt(source, 81.5, 2);
            PyFloat("pd_implicit_series.iloc[2]").Should().Be(81.5,
                "Pandas initially observes the NumSharp-owned storage");

            PyExec("pd_implicit_series.iloc[2] = -7.25");
            ReadAt<double>(source, 2).Should().Be(-7.25,
                "copy=False explicitly opts an external NumPy buffer into shared mutation");
            PyFloat("pd_implicit_series.iloc[2]").Should().Be(-7.25);
        }

        [TestMethod]
        public unsafe void NumericSeries_UsesExistingReadonlyViewBridge_WithExactPointerAndLayout()
        {
            RequirePandas();
            PyExec("pd_numeric_series = pd.Series(np.arange(6, dtype='i8'))");

            using (Gil())
            using (PyObject series = Scope.Get("pd_numeric_series"))
            {
                new Action(() => { using var _ = series.AsNDArray(requireGIL: false); })
                    .Should().Throw<InvalidOperationException>().WithMessage("*read-only*allowReadonly:true*");

                using NDArray view = series.AsNDArray(allowReadonly: true, requireGIL: false);
                view.typecode.Should().Be(NPTypeCode.Int64);
                view.shape.Should().Equal(6);
                view.Shape.Strides.Should().Equal(1L);
                view.Shape.IsWriteable.Should().BeFalse();
                ((long)view.Storage.Address + view.Shape.Offset * view.dtypesize)
                    .Should().Be(PyLong("pd_numeric_series.to_numpy().ctypes.data"));
                ReadAt<long>(view, 4).Should().Be(4);

                using NDArray replacement = np.zeros(new Shape(6), NPTypeCode.Int64);
                new Action(() => np.copyto(view, replacement))
                    .Should().Throw<NumSharpException>().WithMessage("*read-only*");
            }
        }

        [TestMethod]
        public unsafe void HomogeneousDataFrame_ImplicitDecoderPreservesFortranStridesAndPointer()
        {
            RequirePandas();
            CodecTests.EnsureCodec();
            PyExec("pd_homogeneous = pd.DataFrame({'a': [1,2,3], 'b': [10,20,30]}, dtype='i8')");

            using (Gil())
            using (PyObject frame = Scope.Get("pd_homogeneous"))
            using (NDArray view = frame.As<NDArray>())
            using (NDArray viaManagedObject = (NDArray)frame.AsManagedObject(typeof(NDArray)))
            {
                view.typecode.Should().Be(NPTypeCode.Int64);
                view.shape.Should().Equal(3, 2);
                view.Shape.Strides.Should().Equal(1L, 3L);
                view.Shape.IsFContiguous.Should().BeTrue();
                view.Shape.IsWriteable.Should().BeFalse();
                ((long)view.Storage.Address + view.Shape.Offset * view.dtypesize)
                    .Should().Be(PyLong("pd_homogeneous.to_numpy().ctypes.data"));
                ReadAt<long>(view, 0, 0).Should().Be(1);
                ReadAt<long>(view, 2, 1).Should().Be(30);
                ((long)viaManagedObject.Storage.Address +
                 viaManagedObject.Shape.Offset * viaManagedObject.dtypesize)
                    .Should().Be((long)view.Storage.Address + view.Shape.Offset * view.dtypesize);

                Scope.Exec("pd_homogeneous.iloc[1,1] = 77");
                ReadAt<long>(view, 1, 1).Should().Be(77,
                    "Pandas writes remain visible through the read-only external projection");
                ReadAt<long>(viaManagedObject, 1, 1).Should().Be(77);
            }
        }

        [TestMethod]
        public void IndexAndNullableExtensionWithoutMissingValues_AreStableViews()
        {
            RequirePandas();
            PyExec(
                "pd_numeric_index = pd.Index([4, 5, 6], dtype='i8')\n" +
                "pd_nullable_array = pd.array([7, 8, 9], dtype='Int64')");

            using (Gil())
            using (PyObject index = Scope.Get("pd_numeric_index"))
            using (PyObject extension = Scope.Get("pd_nullable_array"))
            using (NDArray indexView = index.AsNDArray(allowReadonly: true, requireGIL: false))
            using (NDArray extensionView = extension.AsNDArray(allowReadonly: true, requireGIL: false))
            {
                indexView.Shape.IsWriteable.Should().BeFalse("Pandas Index is immutable");
                ReadAt<long>(indexView, 2).Should().Be(6);

                extensionView.typecode.Should().Be(NPTypeCode.Int64);
                extensionView.Shape.IsWriteable.Should().BeTrue(
                    "an IntegerArray without missing values exposes its stable numeric data buffer");
                WriteAt(extensionView, 71L, 1);
                PyLong("int(pd_nullable_array[1])").Should().Be(71);
            }
        }

        [TestMethod]
        public void MixedNumericFrame_OrdinaryCopyAndImplicitAutoAreDetachedAndUpcastByPandas()
        {
            RequirePandas();
            CodecTests.EnsureCodec();
            PyExec(
                "pd_mixed_numeric = pd.DataFrame({\n" +
                "    'small': np.array([1,2], dtype='i4'),\n" +
                "    'real': np.array([3.5,4.5], dtype='f8')})");

            using (Gil())
            using (PyObject frame = Scope.Get("pd_mixed_numeric"))
            using (NDArray directCopy = frame.ToNDArray(requireGIL: false))
            using (NDArray implicitAuto = frame.As<NDArray>())
            {
                directCopy.typecode.Should().Be(NPTypeCode.Double);
                implicitAuto.typecode.Should().Be(NPTypeCode.Double);
                directCopy.shape.Should().Equal(2, 2);
                ReadAt<double>(directCopy, 1, 0).Should().Be(2.0);
                ReadAt<double>(directCopy, 0, 1).Should().Be(3.5);
                implicitAuto.Shape.IsWriteable.Should().BeTrue(
                    "Auto must fall back to the owning bridge when Pandas materializes mixed blocks");

                WriteAt(directCopy, -100.0, 0, 0);
                WriteAt(implicitAuto, -200.0, 0, 1);
                PyFloat("float(pd_mixed_numeric.iloc[0,0])").Should().Be(1.0);
                PyFloat("pd_mixed_numeric.iloc[0,1]").Should().Be(3.5);
            }
        }

        private void RequirePandas() => PandasTestGate.Require(Scope);
    }

    internal static class PandasTestGate
    {
        internal const string ValidatedPandasVersion = "3.0.5";

        internal static void Require(PyModule scope)
        {
            string version = null;
            using (Py.GIL())
            {
                try
                {
                    scope.Exec("import pandas as pd");
                    using PyObject value = scope.Eval("str(pd.__version__)");
                    version = value.As<string>();
                }
                catch (PythonException)
                {
                    // Report outside the catch; Assert.Inconclusive throws a control-flow exception.
                }
            }

            if (version is null)
                Assert.Inconclusive("python package 'pandas' is not installed");
            if (version != ValidatedPandasVersion)
                Assert.Inconclusive(
                    $"Pandas adapter claims are pinned to {ValidatedPandasVersion}; host has {version}");
        }
    }
}
