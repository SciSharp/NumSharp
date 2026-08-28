using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Additional adversarial coverage for Pandas 3.0.5: complete layout round-trips, extension
    ///     dtype families, cross-column promotion, byte order, ownership, both-direction lifetime,
    ///     mutation visibility, malformed projections, and two-axis negative strides.
    /// </summary>
    [TestClass]
    public class PandasInteropRareScenarioTests : InteropTestBase
    {
        [TestMethod]
        public unsafe void NumSharpLayouts_RoundTripThroughPandasWithoutLosingPointerShapeOrStrides()
        {
            RequirePandas();
            using NDArray vectorOwner = np.arange(12).astype(NPTypeCode.Double);
            using NDArray positive = vectorOwner["1::2"];
            using NDArray negative = vectorOwner["::-1"];
            using NDArray matrixOwner = np.arange(30).astype(NPTypeCode.Double).reshape(5, 6);
            using NDArray offset = matrixOwner["1:4, 1::2"];
            using NDArray fortran = np.asfortranarray(
                np.arange(12).astype(NPTypeCode.Double).reshape(3, 4));
            using NDArray broadcastBase = np.arange(4).astype(NPTypeCode.Double);
            using NDArray broadcast = np.broadcast_to(broadcastBase, new Shape(3, 4));

            var cases = new (string name, NDArray source, bool frame)[]
            {
                ("positive-stride", positive, false),
                ("negative-stride", negative, false),
                ("offset-matrix", offset, true),
                ("fortran", fortran, true),
                ("broadcast", broadcast, true),
            };

            using (Gil())
            {
                foreach (var (name, source, frame) in cases)
                {
                    using PyObject array = source.ToNumpy(requireGIL: false);
                    Scope.Set("pd_layout_input", array);
                    using PyObject pandas = Scope.Eval(
                        frame
                            ? "pd.DataFrame(pd_layout_input, copy=False)"
                            : "pd.Series(pd_layout_input, copy=False)");
                    using NDArray back = pandas.AsNDArray(allowReadonly: true, requireGIL: false);

                    back.typecode.Should().Be(source.typecode, name);
                    back.shape.Should().Equal(source.shape, name);
                    back.Shape.Strides.Should().Equal(source.Shape.Strides, name);
                    long sourceFirst = (long)source.Storage.Address + source.Shape.Offset * source.dtypesize;
                    long backFirst = (long)back.Storage.Address + back.Shape.Offset * back.dtypesize;
                    backFirst.Should().Be(sourceFirst, $"{name} must return to the identical first element");
                    ByteContract.NsBytes(back).Should().Equal(ByteContract.NsBytes(source), name);
                    if (name == "broadcast")
                    {
                        back.Shape.IsBroadcasted.Should().BeTrue();
                        back.Shape.IsWriteable.Should().BeFalse();
                    }
                }
            }
        }

        [TestMethod]
        public void NullableExtensionDtypesWithoutMissingValues_ExposeEveryStableTypedBuffer()
        {
            RequirePandas();
            (string pandasType, NPTypeCode numSharpType, string values)[] cases =
            {
                ("Int8", NPTypeCode.SByte, "[1, 2]"),
                ("Int16", NPTypeCode.Int16, "[1, 2]"),
                ("Int32", NPTypeCode.Int32, "[1, 2]"),
                ("Int64", NPTypeCode.Int64, "[1, 2]"),
                ("UInt8", NPTypeCode.Byte, "[1, 2]"),
                ("UInt16", NPTypeCode.UInt16, "[1, 2]"),
                ("UInt32", NPTypeCode.UInt32, "[1, 2]"),
                ("UInt64", NPTypeCode.UInt64, "[1, 2]"),
                ("Float32", NPTypeCode.Single, "[1.5, 2.5]"),
                ("Float64", NPTypeCode.Double, "[1.5, 2.5]"),
                ("boolean", NPTypeCode.Boolean, "[True, False]"),
            };

            using (Gil())
            {
                foreach (var (pandasType, numSharpType, values) in cases)
                {
                    using PyObject series = Scope.Eval($"pd.Series({values}, dtype='{pandasType}')");
                    using NDArray view = series.AsNDArray(allowReadonly: true, requireGIL: false);
                    view.typecode.Should().Be(numSharpType, pandasType);
                    view.shape.Should().Equal(2);
                    view.Shape.IsWriteable.Should().BeTrue(
                        $"{pandasType} without a mask exposes its stable extension-data buffer");
                    Scope.Set("pd_clean_extension", series);
                    PyBool("np.shares_memory(pd_clean_extension.to_numpy(), " +
                           "pd_clean_extension.array.to_numpy())").Should().BeTrue(pandasType);
                }
            }
        }

        [TestMethod]
        public void NullableExtensionDtypesWithMissingValues_RejectViewAndResolveCopyDtypePrecisely()
        {
            RequirePandas();
            (string pandasType, NPTypeCode copiedType)[] numeric =
            {
                ("Int8", NPTypeCode.Double),
                ("Int16", NPTypeCode.Double),
                ("Int32", NPTypeCode.Double),
                ("Int64", NPTypeCode.Double),
                ("UInt8", NPTypeCode.Double),
                ("UInt16", NPTypeCode.Double),
                ("UInt32", NPTypeCode.Double),
                ("UInt64", NPTypeCode.Double),
                ("Float32", NPTypeCode.Single),
                ("Float64", NPTypeCode.Double),
            };

            using (Gil())
            {
                foreach (var (pandasType, copiedType) in numeric)
                {
                    using PyObject series = Scope.Eval($"pd.Series([1, pd.NA], dtype='{pandasType}')");
                    new Action(() =>
                        {
                            using var _ = series.AsNDArray(allowReadonly: true, requireGIL: false);
                        })
                        .Should().Throw<NotSupportedException>().WithMessage("*materialized*");
                    using NDArray copy = series.ToNDArray(requireGIL: false);
                    copy.typecode.Should().Be(copiedType, pandasType);
                    if (copiedType == NPTypeCode.Single)
                        float.IsNaN(ReadAt<float>(copy, 1)).Should().BeTrue(pandasType);
                    else
                        double.IsNaN(ReadAt<double>(copy, 1)).Should().BeTrue(pandasType);
                }

                using PyObject boolean = Scope.Eval("pd.Series([True, pd.NA], dtype='boolean')");
                new Action(() => { using var _ = boolean.ToNDArray(requireGIL: false); })
                    .Should().Throw<NotSupportedException>(
                        "Pandas resolves nullable boolean with missing data to object dtype");
            }
        }

        [TestMethod]
        public void MixedColumnPromotion_FollowsPandasCommonDtypeIncludingPrecisionBoundaries()
        {
            RequirePandas();
            PyExec(
                "pd_promote_floats = pd.DataFrame({\n" +
                " 'a': np.array([1,2], dtype='f2'), 'b': np.array([3,4], dtype='f4')})\n" +
                "pd_promote_unsigned = pd.DataFrame({\n" +
                " 'a': np.array([-1,2], dtype='i8'), 'b': np.array([2**64-1,0], dtype='u8')})\n" +
                "pd_promote_complex = pd.DataFrame({\n" +
                " 'a': np.array([1,2], dtype='f8'), 'b': np.array([3+4j,5+6j], dtype='c16')})\n" +
                "pd_promote_object = pd.DataFrame({'a':[True,False], 'b':[1,2]})");

            using (Gil())
            using (PyObject floats = Scope.Get("pd_promote_floats"))
            using (PyObject unsigned = Scope.Get("pd_promote_unsigned"))
            using (PyObject complex = Scope.Get("pd_promote_complex"))
            using (PyObject objects = Scope.Get("pd_promote_object"))
            using (NDArray floatCopy = floats.ToNDArray(requireGIL: false))
            using (NDArray unsignedCopy = unsigned.ToNDArray(requireGIL: false))
            using (NDArray complexCopy = complex.ToNDArray(requireGIL: false))
            {
                floatCopy.typecode.Should().Be(NPTypeCode.Single);
                ReadAt<float>(floatCopy, 1, 1).Should().Be(4f);

                unsignedCopy.typecode.Should().Be(NPTypeCode.Double);
                ReadAt<double>(unsignedCopy, 0, 0).Should().Be(-1.0);
                ReadAt<double>(unsignedCopy, 0, 1).Should().Be(18446744073709551616.0,
                    "Pandas/NumPy common dtype rounds UInt64.MaxValue to float64 2^64");

                complexCopy.typecode.Should().Be(NPTypeCode.Complex);
                ReadAt<Complex>(complexCopy, 0, 1).Should().Be(new Complex(3, 4));
                new Action(() => { using var _ = objects.ToNDArray(requireGIL: false); })
                    .Should().Throw<NotSupportedException>("bool + int resolves to object dtype");
            }
        }

        [TestMethod]
        public void BigEndianNumericSeries_RejectView_AndCopyByteSwapsEveryScalarComponent()
        {
            RequirePandas();
            PyExec(
                "pd_big_i4 = pd.Series(np.array([1, -2], dtype='>i4'), copy=False)\n" +
                "pd_big_f8 = pd.Series(np.array([1.5, -2.25], dtype='>f8'), copy=False)\n" +
                "pd_big_c16 = pd.Series(np.array([1+2j, -3+4j], dtype='>c16'), copy=False)");

            using (Gil())
            using (PyObject ints = Scope.Get("pd_big_i4"))
            using (PyObject floats = Scope.Get("pd_big_f8"))
            using (PyObject complex = Scope.Get("pd_big_c16"))
            {
                foreach (PyObject value in new[] { ints, floats, complex })
                    new Action(() =>
                        {
                            using var _ = value.AsNDArray(allowReadonly: true, requireGIL: false);
                        })
                        .Should().Throw<NotSupportedException>().WithMessage("*big-endian*Byte-swap*");

                using NDArray intCopy = ints.ToNDArray(requireGIL: false);
                using NDArray floatCopy = floats.ToNDArray(requireGIL: false);
                using NDArray complexCopy = complex.ToNDArray(requireGIL: false);
                intCopy.typecode.Should().Be(NPTypeCode.Int32);
                ReadAt<int>(intCopy, 1).Should().Be(-2);
                floatCopy.typecode.Should().Be(NPTypeCode.Double);
                ReadAt<double>(floatCopy, 1).Should().Be(-2.25);
                complexCopy.typecode.Should().Be(NPTypeCode.Complex);
                ReadAt<Complex>(complexCopy, 0).Should().Be(new Complex(1, 2));
                ReadAt<Complex>(complexCopy, 1).Should().Be(new Complex(-3, 4));
            }
        }

        [TestMethod]
        public void ImportedPandasViewCannotResizeOrDetach_WhileOwningCopyCan()
        {
            RequirePandas();
            PyExec("pd_resize = pd.Series(np.arange(4, dtype='i8'))");

            using (Gil())
            using (PyObject series = Scope.Get("pd_resize"))
            using (NDArray view = series.AsNDArray(allowReadonly: true, requireGIL: false))
            using (NDArray copy = series.ToNDArray(requireGIL: false))
            {
                new Action(() => view.resize(new Shape(8)))
                    .Should().Throw<Exception>().WithMessage("*does not own its data*");
                copy.resize(new Shape(8));
                copy.size.Should().Be(8);
                PyLong("len(pd_resize)").Should().Be(4);
            }
        }

        [TestMethod]
        public void PandasObject_KeepsNumSharpExportAliveAfterOriginalArrayIsDisposed()
        {
            RequirePandas();
            int exports = NDArrayPythonInterop.LiveExports;
            NDArray source = np.arange(6).astype(NPTypeCode.Double);

            using (Gil())
            {
                using PyObject array = source.ToNumpy(requireGIL: false);
                Scope.Set("pd_export_input", array);
                Scope.Exec(
                    "pd_export_owner = pd.Series(pd_export_input, copy=False)\n" +
                    "del pd_export_input");
            }

            source.Dispose();
            NDArrayPythonInterop.LiveExports.Should().Be(exports + 1,
                "the live Pandas object must retain NumSharp's export keeper");
            PyFloat("float(pd_export_owner.iloc[5])").Should().Be(5.0);
            PyExec("pd_export_owner.iloc[1] = 91.5");
            PyFloat("float(pd_export_owner.iloc[1])").Should().Be(91.5);

            PyExec("del pd_export_owner");
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports).Should().BeTrue();
        }

        [TestMethod]
        public void PandasMutationUpdatesLeasedView_ButOwningCopyRemainsDetached()
        {
            RequirePandas();
            PyExec("pd_mutation = pd.Series(np.array([1.0,2.0,3.0], dtype='f8'))");

            using (Gil())
            using (PyObject series = Scope.Get("pd_mutation"))
            using (NDArray view = series.AsNDArray(allowReadonly: true, requireGIL: false))
            using (NDArray copy = series.ToNDArray(requireGIL: false))
            {
                Scope.Exec("pd_mutation.iloc[1] = -25.5");
                ReadAt<double>(view, 1).Should().Be(-25.5,
                    "read-only applies to the external consumer; Pandas may mutate its own buffer");
                ReadAt<double>(copy, 1).Should().Be(2.0);

                WriteAt(copy, 700.0, 2);
                PyFloat("float(pd_mutation.iloc[2])").Should().Be(3.0);
            }
        }

        [TestMethod]
        public void MalformedToNumpyReturnTypes_FailWithoutAcquiringImportLeases()
        {
            RequirePandas();
            int imports = NDArrayPythonInterop.LiveImports;
            PyExec(
                "class NumSharpListSeries(pd.Series):\n" +
                "    def to_numpy(self, *args, **kwargs): return [1,2,3]\n" +
                "class NumSharpNoneSeries(pd.Series):\n" +
                "    def to_numpy(self, *args, **kwargs): return None\n" +
                "pd_bad_list = NumSharpListSeries([1,2,3])\n" +
                "pd_bad_none = NumSharpNoneSeries([1,2,3])");

            using (Gil())
            using (PyObject badList = Scope.Get("pd_bad_list"))
            using (PyObject badNone = Scope.Get("pd_bad_none"))
            {
                foreach (PyObject value in new[] { badList, badNone })
                {
                    new Action(() => { using var _ = value.ToNDArray(requireGIL: false); })
                        .Should().Throw<Exception>();
                    new Action(() =>
                        {
                            using var _ = value.AsNDArray(allowReadonly: true, requireGIL: false);
                        })
                        .Should().Throw<Exception>();
                }
            }

            NDArrayPythonInterop.LiveImports.Should().Be(imports);
        }

        [TestMethod]
        public void DescendingRangeIndexAndTwoAxisReorderedFrame_PreserveUnusualLogicalOrder()
        {
            RequirePandas();
            PyExec(
                "pd_descending_range = pd.RangeIndex(10, -2, -3)\n" +
                "pd_reordered_frame = pd.DataFrame(np.arange(20, dtype='i8').reshape(5,4))" +
                ".iloc[::-1, [3,1]]");

            using (Gil())
            using (PyObject range = Scope.Get("pd_descending_range"))
            using (PyObject frame = Scope.Get("pd_reordered_frame"))
            using (NDArray rangeView = range.AsNDArray(allowReadonly: true, requireGIL: false))
            using (NDArray frameView = frame.AsNDArray(allowReadonly: true, requireGIL: false))
            {
                rangeView.shape.Should().Equal(4);
                ReadAt<long>(rangeView, 0).Should().Be(10);
                ReadAt<long>(rangeView, 3).Should().Be(1);

                frameView.shape.Should().Equal(5, 2);
                frameView.Shape.Strides.Should().Equal(-1L, -10L);
                ReadAt<long>(frameView, 0, 0).Should().Be(19);
                ReadAt<long>(frameView, 0, 1).Should().Be(17);
                ReadAt<long>(frameView, 4, 0).Should().Be(3);
                ReadAt<long>(frameView, 4, 1).Should().Be(1);
            }
        }

        private void RequirePandas() => PandasTestGate.Require(Scope);
    }
}
