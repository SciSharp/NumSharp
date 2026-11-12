using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     <c>AsTensor(dtype:)</c> over every dtype the bridge writes, from every shared and converted source dtype: the tensor
    ///     holds NumSharp's own <c>astype</c> wherever NumPy's casting rule allows the pair, and NumPy's refusal text where it
    ///     does not; the torch dtypes the bridge never writes are argument errors naming the ten it does.
    /// </summary>
    [TestClass]
    public class ExplicitDtypeTests : TorchTestBase
    {
        /// <summary>The ten dtypes the bridge converts into, with their NumSharp layouts.</summary>
        private static readonly (ScalarType target, DType dtype)[] Targets =
        {
            (ScalarType.Bool, np.@bool), (ScalarType.Byte, np.uint8), (ScalarType.Int8, np.int8), (ScalarType.Int16, np.int16),
            (ScalarType.Int32, np.int32), (ScalarType.Int64, np.int64), (ScalarType.Float16, np.float16),
            (ScalarType.Float32, np.float32), (ScalarType.Float64, np.float64), (ScalarType.ComplexFloat64, np.complex128),
        };

        /// <summary>Every explicit target, one test case each.</summary>
        public static IEnumerable<object[]> TargetCases => Targets.Select(t => new object[] { t.target.ToString() });

        /// <summary>
        ///     One target from every source dtype under "unsafe" (every pair converts, as astype does) and "same_kind"
        ///     (NumPy's rule decides, the bridge never second-guesses it).
        /// </summary>
        /// <param name="targetName">The target's ScalarType name.</param>
        [TestMethod]
        [DynamicData(nameof(TargetCases))]
        public void ExplicitTarget_ConvertsAsAstypeUnderNumPysCasting(string targetName)
        {
            var (target, dtype) = Targets.Single(t => t.target.ToString() == targetName);
            var failures = new StateMatrixSupport.Failures();
            foreach (string source in StateCatalog.SharedDtypes.Concat(new[] { "uint16", "uint32", "uint64", "Char", "Decimal" }))
            {
                using NDArray nd = StateCatalog.Ramp(source, 2, 3);
                foreach (string casting in new[] { "unsafe", "same_kind" })
                {
                    string at = $"{source} -> {targetName} ({casting})";
                    bool allowed = casting == "unsafe" || np.can_cast(nd.dtype, dtype, casting);
                    try
                    {
                        using Tensor t = nd.AsTensor(dtype: target, casting: casting);
                        failures.Check(allowed, $"{at}: converted although NumPy's rule refuses the pair");
                        failures.Check(t.dtype == target, $"{at}: dtype {t.dtype}");
                        using NDArray expected = nd.astype(dtype);
                        failures.Check(SameBits(expected, t), $"{at}: the values are not NumSharp's astype");
                        bool same = nd.typecode == dtype.GetTypeCode();
                        failures.Check(nd.MayShareMemory(t) == same, $"{at}: shares {nd.MayShareMemory(t)}, expected {same} (naming the array's own dtype converts nothing)");
                    }
                    catch (InvalidCastException ex)
                    {
                        failures.Check(!allowed, $"{at}: refused a pair NumPy's rule allows: {ex.Message}");
                        failures.Check(ex.Message.StartsWith("Cannot cast array data from dtype("), $"{at}: not NumPy's refusal text: {ex.Message}");
                    }
                }
            }
            failures.None.Should().BeTrue(failures.ToString());
        }

        /// <summary>The torch dtypes the bridge never writes are argument errors that name the ten it does (none says NumSharp lacks a uint16).</summary>
        [TestMethod]
        public void NonTargetDtypes_AreArgumentErrorsNamingTheTargets()
        {
            using NDArray nd = np.arange(4).astype(np.uint16);
            foreach (ScalarType refused in new[] { ScalarType.BFloat16, ScalarType.ComplexFloat32, (ScalarType)12, (ScalarType)27, (ScalarType)28, (ScalarType)29 })
            {
                Action act = () => nd.AsTensor(dtype: refused);
                act.Should().Throw<ArgumentException>()
                    .WithMessage($"*dtype {NDArrayTorchInterop.Name(refused)} is not one the bridge converts into*Bool, Byte, Int8, Int16, Int32, Int64, Float16, Float32, Float64, ComplexFloat64*")
                    .Which.ParamName.Should().Be("dtype");
            }
        }

        /// <summary>A casting rule outside NumPy's five is refused even when nothing would convert (NumPy validates the keyword).</summary>
        [TestMethod]
        public void UnknownCasting_IsRefusedWithoutAConversion()
        {
            using NDArray nd = np.arange(3).astype(np.float32);
            Action act = () => nd.AsTensor(TorchCodecMode.View, casting: "SAFE");
            act.Should().Throw<ArgumentException>().Which.ParamName.Should().Be("casting");
        }
    }
}
