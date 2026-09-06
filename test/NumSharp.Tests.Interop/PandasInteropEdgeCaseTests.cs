using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Failure and rare-scenario matrix for the built-in Pandas adapter: materializing projections,
    ///     unsupported semantic dtypes, degenerate shapes, arbitrary labels, strides, all NumSharp
    ///     dtypes, lifetime ownership, codec policy, explicit GIL ownership and concurrent churn.
    /// </summary>
    [TestClass]
    public class PandasInteropEdgeCaseTests : InteropTestBase
    {
        [TestMethod]
        public void MaterializingPandasObjects_RejectView_ButNumericCopiesPreservePandasValues()
        {
            RequirePandas();
            PyExec(
                "pd_materialized = [\n" +
                "  pd.DataFrame({'i': np.array([1,2], dtype='i4'), 'f': np.array([3.5,4.5], dtype='f8')}),\n" +
                "  pd.Series([1, pd.NA], dtype='Int64'),\n" +
                "  pd.Series([1,2,1], dtype='category'),\n" +
                "  pd.Series([0,1,0], dtype='Sparse[int64]')] ");

            using (Gil())
            using (PyObject objects = Scope.Get("pd_materialized"))
            using (var list = new PyList(objects))
            {
                for (int i = 0; i < list.Length(); i++)
                {
                    using PyObject value = list[i];
                    new Action(() =>
                        {
                            using var _ = value.AsNDArray(allowReadonly: true, requireGIL: false);
                        })
                        .Should().Throw<NotSupportedException>()
                        .WithMessage("*materialized*stable shared-memory*ToNDArray*");
                }

                using PyObject mixed = list[0];
                using PyObject nullable = list[1];
                using PyObject categorical = list[2];
                using PyObject sparse = list[3];
                using NDArray mixedCopy = mixed.ToNDArray(requireGIL: false);
                using NDArray nullableCopy = nullable.ToNDArray(requireGIL: false);
                using NDArray categoricalCopy = categorical.ToNDArray(requireGIL: false);
                using NDArray sparseCopy = sparse.ToNDArray(requireGIL: false);

                mixedCopy.typecode.Should().Be(NPTypeCode.Double);
                mixedCopy.shape.Should().Equal(2, 2);
                ReadAt<double>(mixedCopy, 1, 1).Should().Be(4.5);
                nullableCopy.typecode.Should().Be(NPTypeCode.Double);
                ReadAt<double>(nullableCopy, 0).Should().Be(1.0);
                double.IsNaN(ReadAt<double>(nullableCopy, 1)).Should().BeTrue();
                categoricalCopy.typecode.Should().Be(NPTypeCode.Int64);
                ReadAt<long>(categoricalCopy, 1).Should().Be(2);
                sparseCopy.typecode.Should().Be(NPTypeCode.Int64);
                ReadAt<long>(sparseCopy, 1).Should().Be(1);
            }
        }

        [TestMethod]
        public void ObjectStringDatetimeTimedeltaAndNestedIndex_AreDirectedUnsupportedFailures()
        {
            RequirePandas();
            int imports = NDArrayPythonInterop.LiveImports;
            PyExec(
                "pd_unsupported = [\n" +
                "  pd.Series(['a','b'], dtype='str'),\n" +
                "  pd.DataFrame({'a': [1,2], 'b': ['x','y']}),\n" +
                "  pd.Series(pd.date_range('2020-01-01', periods=2)),\n" +
                "  pd.Series(pd.date_range('2020-01-01', periods=2, tz='UTC')),\n" +
                "  pd.Series(pd.to_timedelta([1,2], unit='D')),\n" +
                "  pd.Series([True, pd.NA], dtype='boolean'),\n" +
                "  pd.Series(pd.period_range('2020-01', periods=2, freq='M')),\n" +
                "  pd.MultiIndex.from_tuples([(1,'a'), (2,'b')]),\n" +
                "  pd.IntervalIndex.from_breaks([0, 1, 2])]\n" +
                "pd_scalar_timestamp = pd.Timestamp('2020-01-01')");

            using (Gil())
            using (PyObject objects = Scope.Get("pd_unsupported"))
            using (var list = new PyList(objects))
            {
                for (int i = 0; i < list.Length(); i++)
                {
                    using PyObject value = list[i];
                    new Action(() => { using var _ = value.ToNDArray(requireGIL: false); })
                        .Should().Throw<NotSupportedException>($"unsupported Pandas semantic dtype {i}");
                    new Action(() =>
                        {
                            using var _ = value.AsNDArray(allowReadonly: true, requireGIL: false);
                        })
                        .Should().Throw<NotSupportedException>();
                }

                using PyObject timestamp = Scope.Get("pd_scalar_timestamp");
                new Action(() => { using var _ = timestamp.ToNDArray(requireGIL: false); })
                    .Should().Throw<NotSupportedException>().WithMessage("*does not export a PEP 3118 buffer*");
            }

            NDArrayPythonInterop.LiveImports.Should().Be(imports, "failed conversions must not retain leases");
        }

        [TestMethod]
        public void Complex64ViewDeclines_AndCopyWidensToNumSharpComplex128()
        {
            RequirePandas();
            PyExec("pd_complex64 = pd.Series(np.array([1+2j, -3+4j], dtype='c8'))");

            using (Gil())
            using (PyObject series = Scope.Get("pd_complex64"))
            {
                new Action(() =>
                    {
                        using var _ = series.AsNDArray(allowReadonly: true, requireGIL: false);
                    })
                    .Should().Throw<NotSupportedException>().WithMessage("*complex64*copy*");

                using NDArray copy = series.ToNDArray(requireGIL: false);
                copy.typecode.Should().Be(NPTypeCode.Complex);
                ReadAt<Complex>(copy, 0).Should().Be(new Complex(1, 2));
                ReadAt<Complex>(copy, 1).Should().Be(new Complex(-3, 4));
            }
        }

        [TestMethod]
        public void EmptyAndDegenerateShapes_PreserveRankShapeAndResolvedDtype()
        {
            RequirePandas();
            PyExec(
                "pd_empty_values = [\n" +
                "  pd.Series([], dtype='i4'),\n" +
                "  pd.DataFrame({'a': pd.Series([], dtype='i8'), 'b': pd.Series([], dtype='i8')}),\n" +
                "  pd.DataFrame(index=range(3)),\n" +
                "  pd.RangeIndex(0),\n" +
                "  pd.DataFrame({'a': [42]})]");

            using (Gil())
            using (PyObject objects = Scope.Get("pd_empty_values"))
            using (var list = new PyList(objects))
            using (PyObject s = list[0])
            using (PyObject rows = list[1])
            using (PyObject cols = list[2])
            using (PyObject index = list[3])
            using (PyObject one = list[4])
            using (NDArray sView = s.AsNDArray(allowReadonly: true, requireGIL: false))
            using (NDArray rowsView = rows.AsNDArray(allowReadonly: true, requireGIL: false))
            using (NDArray colsView = cols.AsNDArray(allowReadonly: true, requireGIL: false))
            using (NDArray indexView = index.AsNDArray(allowReadonly: true, requireGIL: false))
            using (NDArray oneView = one.AsNDArray(allowReadonly: true, requireGIL: false))
            {
                sView.typecode.Should().Be(NPTypeCode.Int32);
                sView.shape.Should().Equal(0);
                rowsView.typecode.Should().Be(NPTypeCode.Int64);
                rowsView.shape.Should().Equal(0, 2);
                colsView.typecode.Should().Be(NPTypeCode.Double);
                colsView.shape.Should().Equal(3, 0);
                indexView.typecode.Should().Be(NPTypeCode.Int64);
                indexView.shape.Should().Equal(0);
                oneView.shape.Should().Equal(1, 1);
                ReadAt<long>(oneView, 0, 0).Should().Be(42);
            }
        }

        [TestMethod]
        public void PositiveAndNegativeStridedSeries_PreserveLogicalOffsetsAndStrides()
        {
            RequirePandas();
            PyExec(
                "pd_stride_base = pd.Series(np.arange(8, dtype='f8'))\n" +
                "pd_stride_positive = pd_stride_base.iloc[1::2]\n" +
                "pd_stride_negative = pd_stride_base.iloc[::-1]");

            using (Gil())
            using (PyObject positive = Scope.Get("pd_stride_positive"))
            using (PyObject negative = Scope.Get("pd_stride_negative"))
            using (NDArray p = positive.AsNDArray(allowReadonly: true, requireGIL: false))
            using (NDArray n = negative.AsNDArray(allowReadonly: true, requireGIL: false))
            {
                p.shape.Should().Equal(4);
                p.Shape.Strides.Should().Equal(2L);
                ReadAt<double>(p, 0).Should().Be(1.0);
                ReadAt<double>(p, 3).Should().Be(7.0);

                n.shape.Should().Equal(8);
                n.Shape.Strides.Should().Equal(-1L);
                ReadAt<double>(n, 0).Should().Be(7.0);
                ReadAt<double>(n, 7).Should().Be(0.0);
                n.Shape.IsWriteable.Should().BeFalse();
            }
        }

        [TestMethod]
        public void All15NumSharpDtypes_RoundTripThroughPandasNumericSeries()
        {
            RequirePandas();
            var cases = new (string name, NDArray source, NPTypeCode returnedType)[]
            {
                ("bool", np.array(new[] { false, true }), NPTypeCode.Boolean),
                ("u8", np.array(new byte[] { 0, byte.MaxValue }), NPTypeCode.Byte),
                ("i8", np.array(new[] { sbyte.MinValue, (sbyte)-1, sbyte.MaxValue }), NPTypeCode.SByte),
                ("i16", np.array(new[] { short.MinValue, (short)-1, short.MaxValue }), NPTypeCode.Int16),
                ("u16", np.array(new ushort[] { 0, ushort.MaxValue }), NPTypeCode.UInt16),
                ("i32", np.array(new[] { int.MinValue, -1, int.MaxValue }), NPTypeCode.Int32),
                ("u32", np.array(new uint[] { 0, uint.MaxValue }), NPTypeCode.UInt32),
                ("i64", np.array(new[] { long.MinValue, -1L, long.MaxValue }), NPTypeCode.Int64),
                ("u64", np.array(new ulong[] { 0, ulong.MaxValue }), NPTypeCode.UInt64),
                ("char", np.array(new[] { '\0', '\ud800', '\uffff' }), NPTypeCode.UInt16),
                ("f16", np.array(new[]
                {
                    BitConverter.UInt16BitsToHalf(0x8000),
                    BitConverter.UInt16BitsToHalf(0x7c00),
                    BitConverter.UInt16BitsToHalf(0xfe55),
                }), NPTypeCode.Half),
                ("f32", np.array(new[]
                {
                    -0.0f,
                    float.PositiveInfinity,
                    BitConverter.Int32BitsToSingle(unchecked((int)0xffc12345)),
                }), NPTypeCode.Single),
                ("f64", np.array(new[]
                {
                    -0.0,
                    double.NegativeInfinity,
                    BitConverter.Int64BitsToDouble(unchecked((long)0xfff8123456789abc)),
                }), NPTypeCode.Double),
                ("decimal", np.array(new[] { decimal.MinValue, 1.234567890123456789m, decimal.MaxValue }), NPTypeCode.Double),
                ("complex", np.array(new[]
                {
                    new Complex(-0.0, double.PositiveInfinity),
                    new Complex(BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000042)), -0.0),
                }), NPTypeCode.Complex),
            };

            try
            {
                using (Gil())
                {
                    foreach (var (name, source, returnedType) in cases)
                    {
                        using PyObject array = source.ToNumpy(requireGIL: false);
                        Scope.Set("pd_dtype_input", array);
                        using PyObject series = Scope.Eval("pd.Series(pd_dtype_input, copy=False)");
                        using NDArray back = series.ToNDArray(requireGIL: false);
                        back.typecode.Should().Be(returnedType, name);

                        if (source.typecode == NPTypeCode.Decimal)
                        {
                            using NDArray expected = source.astype(NPTypeCode.Double);
                            ByteContract.NsBytes(back).Should().Equal(ByteContract.NsBytes(expected), name);
                        }
                        else
                        {
                            ByteContract.NsBytes(back).Should().Equal(ByteContract.NsBytes(source), name);
                        }
                    }
                }
            }
            finally
            {
                foreach (var item in cases)
                    item.source.Dispose();
            }
        }

        [TestMethod]
        public void AxisLabelsDuplicatesAndMultiIndex_AreMetadataOnly_ValuesKeepTwoDimensionalOrder()
        {
            RequirePandas();
            PyExec(
                "pd_labels = pd.DataFrame(\n" +
                "  np.array([[1,10],[2,20],[3,30]], dtype='i8'),\n" +
                "  index=pd.MultiIndex.from_tuples([('x',1),('x',2),('y',1)]),\n" +
                "  columns=['dup','dup'])");

            using (Gil())
            using (PyObject frame = Scope.Get("pd_labels"))
            using (NDArray values = frame.AsNDArray(allowReadonly: true, requireGIL: false))
            {
                values.shape.Should().Equal(3, 2);
                ReadAt<long>(values, 0, 0).Should().Be(1);
                ReadAt<long>(values, 0, 1).Should().Be(10);
                ReadAt<long>(values, 2, 0).Should().Be(3);
                ReadAt<long>(values, 2, 1).Should().Be(30);
            }
        }

        [TestMethod]
        public void DerivedNumSharpView_KeepsPandasStorageAliveAfterAllPythonWrappersDie()
        {
            RequirePandas();
            PyExec("pd_lifetime = pd.Series(np.arange(8, dtype='f8'))");

            NDArray root;
            using (Gil())
            using (PyObject series = Scope.Get("pd_lifetime"))
                root = series.AsNDArray(allowReadonly: true, requireGIL: false);

            using NDArray derived = root["2::2"];
            root.Dispose();
            PyExec("del pd_lifetime; gc.collect()");

            derived.shape.Should().Equal(3);
            ReadAt<double>(derived, 0).Should().Be(2.0);
            ReadAt<double>(derived, 2).Should().Be(6.0);
        }

        [TestMethod]
        public void CodecModes_RespectPandasViewCopyAutoAndExplicitAdapterDisable()
        {
            RequirePandas();
            var auto = new NumpyCodec();
            var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View });
            var copyOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy });
            var disabled = new NumpyCodec(new NumpyCodecOptions { DecodeArrayAdapters = false });
            PyExec(
                "pd_codec_homogeneous = pd.DataFrame({'a':[1,2], 'b':[3,4]}, dtype='i8')\n" +
                "pd_codec_mixed = pd.DataFrame({'a':np.array([1,2],dtype='i4'), 'b':[3.5,4.5]})\n" +
                "pd_codec_object = pd.Series(['a','b'], dtype='str')");

            using (Gil())
            using (PyObject homogeneous = Scope.Get("pd_codec_homogeneous"))
            using (PyObject mixed = Scope.Get("pd_codec_mixed"))
            using (PyObject objects = Scope.Get("pd_codec_object"))
            {
                using PyObject typeObject = homogeneous.GetPythonType();
                using var type = new PyType(typeObject);
                auto.CanDecode(type, typeof(NDArray)).Should().BeTrue();
                disabled.CanDecode(type, typeof(NDArray)).Should().BeFalse();
                disabled.TryDecode(homogeneous, out NDArray declined).Should().BeFalse();
                declined.Should().BeNull();

                viewOnly.TryDecode(homogeneous, out NDArray view).Should().BeTrue();
                copyOnly.TryDecode(homogeneous, out NDArray copy).Should().BeTrue();
                using (view)
                using (copy)
                {
                    view.Shape.IsWriteable.Should().BeFalse();
                    copy.Shape.IsWriteable.Should().BeTrue();
                    WriteAt(copy, 99L, 0, 0);
                    PyLong("int(pd_codec_homogeneous.iloc[0,0])").Should().Be(1);
                }

                viewOnly.TryDecode(mixed, out NDArray impossibleView).Should().BeFalse();
                impossibleView.Should().BeNull();
                auto.TryDecode(mixed, out NDArray autoCopy).Should().BeTrue();
                using (autoCopy)
                    autoCopy.typecode.Should().Be(NPTypeCode.Double);

                auto.TryDecode(objects, out NDArray unsupported).Should().BeFalse();
                unsupported.Should().BeNull();
            }
        }

        [TestMethod]
        public void ExplicitNoGilPolicy_WorksInsideOneOuterGil_ForPandasViewAndCopy()
        {
            RequirePandas();
            PyExec("pd_gil = pd.Series(np.arange(4, dtype='f8'))");

            using (Gil())
            using (PyObject series = Scope.Get("pd_gil"))
            using (NDArray view = series.AsNDArray(allowReadonly: true, requireGIL: false))
            using (NDArray copy = series.ToNDArray(requireGIL: false))
            {
                ReadAt<double>(view, 3).Should().Be(3.0);
                ReadAt<double>(copy, 2).Should().Be(2.0);
            }
        }

        [TestMethod]
        public void ConcurrentPandasRoundTrips_AreThreadSafeAndDrainEveryLifetimeCounter()
        {
            RequirePandas();
            int exports = NDArrayPythonInterop.LiveExports;
            int imports = NDArrayPythonInterop.LiveImports;
            var failures = new ConcurrentQueue<Exception>();

            void Worker(int id)
            {
                try
                {
                    for (int k = 0; k < 10; k++)
                    {
                        using NDArray source = np.arange(12).astype(NPTypeCode.Double) + id;
                        using (Py.GIL())
                        using (PyObject pandas = Py.Import("pandas"))
                        using (PyObject constructor = pandas.GetAttr("Series"))
                        using (PyObject array = source.ToNumpy(requireGIL: false))
                        using (PyObject series = constructor.Invoke(array))
                        using (NDArray view = series.AsNDArray(allowReadonly: true, requireGIL: false))
                        using (NDArray copy = series.ToNDArray(requireGIL: false))
                        {
                            ReadAt<double>(view, 5).Should().Be(5 + id);
                            ReadAt<double>(copy, 11).Should().Be(11 + id);
                        }
                    }
                }
                catch (Exception e)
                {
                    failures.Enqueue(e);
                }
            }

            var threads = new Thread[4];
            for (int i = 0; i < threads.Length; i++)
            {
                int id = i;
                threads[i] = new Thread(() => Worker(id)) { IsBackground = true };
                threads[i].Start();
            }

            foreach (Thread thread in threads)
                thread.Join(60_000).Should().BeTrue("Pandas interop workers must not deadlock");

            failures.Should().BeEmpty(string.Join(" | ", failures));
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports &&
                          NDArrayPythonInterop.LiveImports == imports, 20_000).Should().BeTrue(
                $"all Pandas crossings must drain (exports={NDArrayPythonInterop.LiveExports}, " +
                $"imports={NDArrayPythonInterop.LiveImports})");
        }

        [TestMethod]
        public void DisposedAndBrokenPandasObjects_FailCleanlyWithoutLeasingMemory()
        {
            RequirePandas();
            int imports = NDArrayPythonInterop.LiveImports;
            PyExec(
                "class NumSharpBrokenSeries(pd.Series):\n" +
                "    def to_numpy(self, *args, **kwargs):\n" +
                "        raise RuntimeError('intentional pandas projection failure')\n" +
                "pd_broken = NumSharpBrokenSeries([1,2,3])");

            PyObject disposed;
            using (Gil())
            {
                disposed = Scope.Get("pd_broken");
                disposed.Dispose();
            }

            new Action(() => { using var _ = disposed.ToNDArray(); })
                .Should().Throw<PythonException>().WithMessage("*null argument to internal routine*");

            using (Gil())
            using (PyObject broken = Scope.Get("pd_broken"))
            {
                new Action(() => { using var _ = broken.ToNDArray(requireGIL: false); })
                    .Should().Throw<PythonException>().WithMessage("*intentional pandas projection failure*");
                new Action(() =>
                    {
                        using var _ = broken.AsNDArray(allowReadonly: true, requireGIL: false);
                    })
                    .Should().Throw<PythonException>().WithMessage("*intentional pandas projection failure*");
            }

            NDArrayPythonInterop.LiveImports.Should().Be(imports);
        }

        private void RequirePandas() => PandasTestGate.Require(Scope);
    }
}
