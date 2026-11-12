using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using static NumSharp.Tests.Interop.TorchSharp.StateMatrixSupport;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     The five NumSharp dtypes torch lacks (uint16, uint32, uint64, char, decimal) across the layouts that decide an
    ///     export: whatever the state, Auto and Copy give an independent tensor of the NEP 50 dtype holding NumSharp's own
    ///     <c>astype</c> of the values, and View refuses with the conversion as the reason - before any layout rule (a
    ///     converted copy is never a view).
    /// </summary>
    [TestClass]
    public class ConvertedDtypeStateTests : TorchTestBase
    {
        /// <summary>The converted dtypes by the names NumSharp parses, with the torch dtype each crosses as.</summary>
        private static readonly (string name, ScalarType crossesAs)[] Converted =
        {
            ("uint16", ScalarType.Int32), ("uint32", ScalarType.Int64), ("uint64", ScalarType.Float64),
            ("Char", ScalarType.Int32), ("Decimal", ScalarType.Float64),
        };

        /// <summary>The layouts that decide an export (sharable, reversed, read-only, broadcast, 0-d, empty, strided).</summary>
        private static readonly string[] Layouts =
        {
            "c_2d", "f_2d_owner", "transposed_3d", "stepped_offset_3d", "negstride_2d", "broadcast_row", "broadcast_of_readonly",
            "readonly_owner", "zerod_owner", "zerod_view", "empty_2d", "highrank_8d_permuted", "overlapping_windows",
        };

        /// <summary>Every converted dtype, one test case each.</summary>
        public static IEnumerable<object[]> Dtypes => Converted.Select(c => new object[] { c.name });

        /// <summary>One converted dtype over every deciding layout, in every mode.</summary>
        /// <param name="dtype">The NumSharp dtype name.</param>
        [TestMethod]
        [DynamicData(nameof(Dtypes))]
        public void ConvertedDtype_CopiesAsNep50InEveryLayout(string dtype)
        {
            ScalarType crossesAs = Converted.Single(c => c.name == dtype).crossesAs;
            var failures = new Failures();
            foreach (string layout in Layouts)
            {
                string where = $"{layout} [{dtype}]";
                try
                {
                    using var scope = Scope();
                    NDArray nd = StateCatalog.ArrayStates[layout](dtype);
                    using NDArray expected = nd.astype(NDArrayTorchInterop.ToDType(crossesAs));
                    foreach (TorchCodecMode mode in Modes)
                    {
                        string at = $"{where} {mode}";
                        if (mode == TorchCodecMode.View)
                        {
                            try
                            {
                                using Tensor refused = nd.AsTensor(mode);
                                failures.Check(false, $"{at}: a converted copy was returned as a view");
                            }
                            catch (InvalidOperationException ex)
                            {
                                failures.Check(ex.Message.Contains("NEP 50"), $"{at}: the refusal does not name the conversion: {ex.Message}");
                            }
                            continue;
                        }
                        using Tensor t = nd.AsTensor(mode);
                        failures.Check(t.dtype == crossesAs, $"{at}: dtype {t.dtype}, expected {crossesAs}");
                        failures.Check(SameBits(expected, t), $"{at}: the values are not NumSharp's astype");
                        failures.Check(!nd.MayShareMemory(t), $"{at}: a converted copy shares memory");
                        failures.Check(t.is_contiguous(), $"{at}: a converted copy is not contiguous");
                    }
                }
                catch (Exception ex)
                {
                    failures.Unexpected(where, ex);
                }
            }
            failures.None.Should().BeTrue(failures.ToString());
        }
    }
}
