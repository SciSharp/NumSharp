using System;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/10-pandas.cs</c>, section by section.</summary>
    [TestClass]
    public class Example10_Pandas : ExampleTestBase
    {
        [TestInitialize]
        public void RequirePandas()
        {
            SkipUnless("pandas");
            PyExec("import pandas as pd");
        }

        [TestMethod]
        public void Series_TheOrdinaryVerbs_ThroughTheAdapter()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("s = pd.Series(np.arange(6, dtype='i8'))");
                using PyObject s = py.s;
                Throws(() => s.AsNDArray()).Should().BeOfType<InvalidOperationException>().Which.Message.Should().Contain("allowReadonly:true");
                NDArray view = s.AsNDArray(allowReadonly: true);
                view.typecode.Should().Be(NPTypeCode.Int64);
                view.shape.Should().Equal(6);
                view.Shape.IsWriteable.Should().BeFalse("Pandas 3 exposes the shared projection read-only");
                py.Exec("s.iloc[2] = 77");
                view.GetInt64(2).Should().Be(77, "a Pandas write is visible through the view");
                Exception guarded = Throws(() => np.copyto(view, np.zeros(new Shape(6), np.int64)));
                guarded.Should().NotBeNull();
                guarded.Message.Should().Contain("read-only");
                NDArray copy = s.ToNDArray();
                copy.Shape.IsWriteable.Should().BeTrue();
                copy[0] = 999L;
                PyLong("int(s.iloc[0])").Should().Be(0, "the snapshot is detached");
                NDArray viaCodec = py.s;
                viaCodec.Shape.IsWriteable.Should().BeFalse("Auto took the read-only view by itself");
                viaCodec.GetInt64(2).Should().Be(77);
                py.Exec("del s");
            }
        }

        [TestMethod]
        public void DataFrame_AHomogeneousFrameIsOneFStridedBlock()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]}, dtype='i8')");
                NDArray view = py.df;
                view.shape.Should().Equal(3, 2);
                view.Shape.Strides.Should().Equal(new long[] { 1, 3 }, "Fortran order, as Pandas stores blocks");
                view.Shape.IsFContiguous.Should().BeTrue();
                view.Shape.IsWriteable.Should().BeFalse("Copy-on-Write");
                view.GetInt64(2, 1).Should().Be(30);
                py.Exec("df.iloc[1, 1] = 77");
                view.GetInt64(1, 1).Should().Be(77);
                py.Exec("del df");
            }
        }

        [TestMethod]
        public void Index_RangeIndex_AndNullableExtensionArrays()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("idx = pd.Index([4, 5, 6], dtype='i8')\nrng = pd.RangeIndex(5)\nnullable = pd.array([7, 8, 9], dtype='Int64')");
                NDArray iv = py.idx;
                iv.Shape.IsWriteable.Should().BeFalse("an Index is immutable");
                iv.GetInt64(2).Should().Be(6);
                NDArray rv = py.rng;
                rv.typecode.Should().Be(NPTypeCode.Int64);
                rv.GetInt64(4).Should().Be(4, "RangeIndex materializes through the Index MRO");
                NDArray nv = py.nullable;
                nv.typecode.Should().Be(NPTypeCode.Int64);
                nv.Shape.IsWriteable.Should().BeTrue("a nullable array without missing values exposes its stable typed buffer");
                nv[1] = 71L;
                PyLong("int(nullable[1])").Should().Be(71);
                py.Exec("del idx, rng, nullable");
            }
        }

        [TestMethod]
        public void WhatMaterializes_MixedBlocks_MissingValues_Categoricals_Text()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("mixed = pd.DataFrame({'small': np.array([1, 2], dtype='i4'), 'real': np.array([3.5, 4.5], dtype='f8')})\n" +
                        "with_na = pd.array([1, None, 3], dtype='Int64')\n" +
                        "cat = pd.Series(pd.Categorical([3, 1, 3]))\n" +
                        "text = pd.Series(['a', 'b'])");
                using PyObject mixed = py.mixed;
                Throws(() => mixed.AsNDArray(allowReadonly: true)).Should().BeOfType<NotSupportedException>().Which.Message.Should().Contain("cannot be proven");
                NDArray upcast = mixed.ToNDArray();
                upcast.typecode.Should().Be(NPTypeCode.Double, "Pandas' common-dtype upcast");
                upcast.GetDouble(1, 0).Should().Be(2.0);
                upcast.GetDouble(0, 1).Should().Be(3.5);
                NDArray auto = py.mixed;
                auto.Shape.IsWriteable.Should().BeTrue("Auto fell back to the owning copy");
                auto.typecode.Should().Be(NPTypeCode.Double);

                using PyObject withNa = py.with_na;
                Throws(() => withNa.AsNDArray(allowReadonly: true)).Should().BeOfType<NotSupportedException>();
                NDArray naCopy = withNa.ToNDArray();
                naCopy.typecode.Should().Be(NPTypeCode.Double);
                double.IsNaN(naCopy.GetDouble(1)).Should().BeTrue("NaN for the missing value");

                using PyObject cat = py.cat;
                Throws(() => cat.AsNDArray(allowReadonly: true)).Should().BeOfType<NotSupportedException>();
                NDArray catCopy = cat.ToNDArray();
                catCopy.GetInt64(0).Should().Be(3);
                catCopy.GetInt64(1).Should().Be(1);

                using PyObject text = py.text;
                Throws(() => text.ToNDArray()).Should().BeOfType<NotSupportedException>("object/string columns have no NumSharp dtype");
                py.Exec("del mixed, with_na, cat, text");
            }
        }

        [TestMethod]
        public void LabelsAreMetadata_OnlyTheValueMatrixCrosses()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("labelled = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2)]), columns=['dup', 'dup'])");
                NDArray values = py.labelled;
                values.shape.Should().Equal(2, 2);
                values.GetDouble(1, 0).Should().Be(3.0);
                py.Exec("del labelled");
            }
        }

        [TestMethod]
        public void NumSharpToPandas_TheNumpyEncoderPlusCopyFalse()
        {
            using var scope = NDScope.Open();
            NDArray nd = np.arange(6).astype(np.float64);
            using (Gil())
            {
                py.values = nd;
                py.Exec("series = pd.Series(values, copy=False)\nframe = pd.DataFrame(values.reshape(-1, 2), copy=False)");
                bool onNumSharpMemory = py.Eval("series.to_numpy().ctypes.data == values.ctypes.data");
                bool writeable = py.Eval("series.to_numpy().flags.writeable");
                onNumSharpMemory.Should().BeTrue("pd.Series(values, copy=False) sits on NumSharp's memory");
                writeable.Should().BeFalse("Pandas 3 marks the shared projection read-only");
                nd[2] = 81.5;
                PyFloat("series.iloc[2]").Should().Be(81.5);
                PyFloat("frame.iloc[1, 0]").Should().Be(81.5);
                py.Exec("series.iloc[2] = -7.25");
                nd.GetDouble(2).Should().Be(-7.25, "copy=False is a deliberate shared-memory opt-in");
                py.Exec("default_copy = pd.Series(values)");
                PyBool("default_copy.to_numpy().ctypes.data == values.ctypes.data").Should().BeFalse("without copy=False Pandas 3 copies");
                py.Exec("del values, series, frame, default_copy");
            }
        }

        [TestMethod]
        public void Lifetime_ADerivedViewKeepsPandasStorageAlive()
        {
            int imports0 = NDArrayPythonInterop.LiveImports;
            using var scope = NDScope.Open();
            NDArray slice = TakeASliceAndLetPythonForget();
            Drain();
            slice.GetDouble(0).Should().Be(3.0);
            slice.GetDouble(2).Should().Be(6.0);
            NDArrayPythonInterop.LiveImports.Should().Be(imports0 + 1, "the lease holds the projection");
            slice.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == imports0).Should().BeTrue();
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private NDArray TakeASliceAndLetPythonForget()
        {
            NDArray slice;
            using (Gil())
            {
                py.Exec("tmp = pd.Series(np.arange(8, dtype='f8') * 1.5)");
                using (var inner = NDScope.Open())
                {
                    using PyObject tmp = py.tmp;
                    NDArray whole = tmp.AsNDArray(allowReadonly: true);
                    slice = inner.Returns(whole["2:5"]);
                }

                py.Exec("del tmp\nimport gc; gc.collect()");
            }

            return slice;
        }

        [TestMethod]
        public void Registry_TheAdapterIsBuiltIn()
        {
            PythonArrayAdapterRegistry.Register(PandasPythonArrayAdapter.Instance).Should().BeFalse("already registered under 'pandas'");
            PandasPythonArrayAdapter.Instance.Name.Should().Be("pandas");
            var noAdapters = new NumpyCodec(new NumpyCodecOptions { DecodeArrayAdapters = false });
            using (Gil())
            {
                py.Exec("s = pd.Series(np.arange(6, dtype='i8'))");
                using PyObject s = py.s;
                using PyObject t = s.GetPythonType();
                using var seriesType = new PyType(t);
                noAdapters.CanDecode(seriesType, typeof(NDArray)).Should().BeFalse("DecodeArrayAdapters = false");
                py.Exec("del s");
            }
        }
    }
}
