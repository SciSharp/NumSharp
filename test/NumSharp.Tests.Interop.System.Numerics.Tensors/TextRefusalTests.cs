using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Text and the System.Numerics.Tensors bridge: a BCL tensor holds an UNMANAGED element type (every verb is
    ///     <c>where T : unmanaged</c>), and NumSharp's text dtypes — fixed-width <c>U</c> / <c>S</c> and
    ///     <c>StringDType</c>, which share the <see cref="NPTypeCode.String"/> storage lane — have no such element type.
    ///     So text never crosses: every export verb refuses every text array, of every class, width and layout, with a
    ///     <see cref="NotSupportedException"/> before anything is read, leased or pinned, and no import verb can even be
    ///     instantiated for <c>string</c>.
    /// </summary>
    /// <remarks>
    ///     <para>WHICH text the refusal carries is an open bug
    ///     (<see cref="TextExports_RefuseWithATextMessage_NotTheStorageLaneError"/>): the verbs hand the array's type code
    ///     to the dtype map, whose parameter is a <see cref="DType"/>, and the implicit <c>NPTypeCode → DType</c> conversion
    ///     throws Core's storage-lane error first ("… asked for a dtype by storage lane …"), which the ONNX Runtime and
    ///     ML.NET bridges replaced with a text refusal on 2026-11-02 (<c>da1deae2</c>).</para>
    ///     <para>NumSharp's 16-bit <see cref="NPTypeCode.Char"/> is NOT text and crosses as <see cref="char"/>
    ///     (<c>DtypeMapTests</c>); the 0-d and negative-stride cells prove the dtype refusal comes FIRST, before the layout
    ///     checks a numeric array of the same layout would meet.</para>
    /// </remarks>
    [TestClass]
    public class TextRefusalTests : TensorsTestBase
    {
        /// <summary>
        ///     The text arrays: <c>U1</c>, <c>U5</c>, <c>S1</c>, <c>S5</c>, <c>StringDType()</c> and
        ///     <c>StringDType(na_object=None)</c> with an NA, each in C order, F order, transposed, reversed (a negative
        ///     stride, which the BCL forbids), stepped, broadcast, 0-d and empty.
        /// </summary>
        /// <returns>(label, array) pairs.</returns>
        private static IEnumerable<(string Name, NDArray Array)> TextArrays()
        {
            string[] texts = { "alpha", "be", "", "x", "zz", "q" };
            var bases = new (string, NDArray)[]
            {
                ("U1", np.array(texts).astype(np.dtype("U1"))),
                ("U5", np.array(texts)),
                ("S1", np.array(texts, np.dtype("S1"))),
                ("S5", np.array(texts, np.dtype("S5"))),
                ("StringDType()", np.array(texts, np.dtypes.StringDType())),
                ("StringDType(na_object=None) + NA", np.array(new object[] { "alpha", null, "", "x", "zz", "q" }, np.dtypes.StringDType((object)null))),
            };
            foreach (var (name, b) in bases)
            {
                NDArray grid = b.reshape(2, 3);
                yield return ($"{name} C", grid);
                yield return ($"{name} F", np.asfortranarray(grid));
                yield return ($"{name} transposed", grid.T);
                yield return ($"{name} reversed", b["::-1"]);
                yield return ($"{name} stepped", b["1::2"]);
                yield return ($"{name} broadcast", np.broadcast_to(b["0:3"], new Shape(2, 3)));
                yield return ($"{name} 0-d", b["2"]);
                yield return ($"{name} empty", b["3:3"]);
            }
        }

        /// <summary>The export verbs and element types a caller might try on a text array.</summary>
        /// <param name="text">The text array.</param>
        /// <returns>(verb label, call) pairs; each call disposes what it made, should it succeed.</returns>
        private static IEnumerable<(string Verb, Action Call)> ExportVerbs(NDArray text) => new (string, Action)[]
        {
            ("AsTensorSpan<char>", () => { using var _ = text.AsTensorSpan<char>(); }),
            ("AsTensorSpan<byte>", () => { using var _ = text.AsTensorSpan<byte>(); }),
            ("AsTensorSpan<int>", () => { using var _ = text.AsTensorSpan<int>(); }),
            ("AsTensorSpan<ushort>", () => { using var _ = text.AsTensorSpan<ushort>(); }),
            ("ToTensor<char>", () => text.ToTensor<char>()),
            ("ToTensor<byte>", () => text.ToTensor<byte>()),
            ("ToTensor<int>", () => text.ToTensor<int>()),
        };

        /// <summary>
        ///     Every text array × every export verb and element type a caller might try (<c>AsTensorSpan&lt;T&gt;</c> /
        ///     <c>ToTensor&lt;T&gt;</c> for <see cref="char"/>, <see cref="byte"/>, <see cref="int"/>, <see cref="ushort"/>)
        ///     is refused with <see cref="NotSupportedException"/> — for every class, width and layout, the negative-stride
        ///     and 0-d views included (the dtype refusal precedes the layout checks) — and leases nothing.
        /// </summary>
        [TestMethod]
        public void TextArrays_EveryClassLayoutAndElementType_RefusedByEveryExportVerb()
        {
            int exports = NDArrayTensorsInterop.LiveExports;
            int cells = 0;
            foreach (var (name, text) in TextArrays())
            {
                foreach (var (verb, call) in ExportVerbs(text))
                {
                    cells++;
                    call.Should().Throw<NotSupportedException>($"{name} / {verb}");
                }

                NDArrayTensorsInterop.LiveExports.Should().Be(exports, $"{name}: a refusal leases nothing");
            }

            cells.Should().Be(6 * 8 * 7, "every class × layout × verb ran");
        }

        /// <summary>
        ///     OPEN BUG (found by these tests, 2026-11-05): a text array's export refusal is Core's storage-lane conversion
        ///     error — "This operation does not support fixed-width string arrays: it asked for a dtype by storage lane
        ///     (NPTypeCode.String) … Use the array's own dtype (a.dtype), a dtype string such as 'U5' …" — because
        ///     <c>EnsureElementType</c> passes the array's TYPE CODE to <c>ToTensorElementClrType(DType)</c> and the implicit
        ///     conversion throws before the map's own refusal runs. The text tells the caller to pass a dtype they never
        ///     asked for and calls a <c>StringDType</c> array fixed-width; the ONNX Runtime and ML.NET bridges answer text with
        ///     their own refusal that names text (<c>da1deae2</c>). Expected: a refusal that names text (and the class the
        ///     array is), never the storage lane.
        /// </summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void TextExports_RefuseWithATextMessage_NotTheStorageLaneError()
        {
            foreach (var (name, text) in TextArrays())
            foreach (var (verb, call) in ExportVerbs(text))
            {
                string message = call.Should().Throw<NotSupportedException>($"{name} / {verb}").Which.Message;
                message.Should().NotContain("storage lane", $"{name} / {verb}: Core's storage-lane error leaked out");
                message.Should().Contain("text", $"{name} / {verb}: the refusal names text");
                if (text.dtype.kind == 'T')
                    message.Should().NotContain("fixed-width", $"{name} / {verb}: a StringDType array is not fixed-width");
            }
        }

        /// <summary>
        ///     The dtype map's own refusal of a text DESCRIPTOR (<c>U5</c>, <c>S3</c>, <c>StringDType()</c>) — reached when a
        ///     caller asks the map with the array's dtype — is "NumSharp dtype String has no CLR element type.", while
        ///     NumSharp's 16-bit <see cref="NPTypeCode.Char"/> (a number, not text) maps to <see cref="char"/>.
        /// </summary>
        [TestMethod]
        public void DtypeMap_RefusesTextDescriptors_CharIsANumber()
        {
            foreach (DType text in new[] { np.dtype("U5"), np.dtype("S3"), np.dtypes.StringDType() })
                new Action(() => NDArrayTensorsInterop.ToTensorElementClrType(text)).Should().Throw<NotSupportedException>(text.ToString())
                    .WithMessage("NumSharp dtype String has no CLR element type.");
            NDArrayTensorsInterop.ToTensorElementClrType(NPTypeCode.Char).Should().Be(typeof(char), "NumSharp's char is a number, not text");
        }

        /// <summary>
        ///     The import side cannot take text at all: every public verb that builds an NDArray from a BCL tensor or span is
        ///     generic over <c>T : unmanaged</c>, so even reflection cannot instantiate one for <see cref="string"/> —
        ///     <see cref="MethodInfo.MakeGenericMethod"/> refuses the constraint violation before any call.
        /// </summary>
        [TestMethod]
        public void ImportVerbs_CannotBeInstantiatedForString()
        {
            MethodInfo[] imports = typeof(NDArrayTensorsInterop).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.IsGenericMethodDefinition && (m.Name == "ToNDArray" || m.Name == "AsNDArray"))
                .ToArray();
            imports.Should().NotBeEmpty();
            foreach (MethodInfo import in imports)
            {
                import.GetGenericArguments()[0].GenericParameterAttributes
                    .HasFlag(GenericParameterAttributes.NotNullableValueTypeConstraint)
                    .Should().BeTrue($"{import}: T is constrained to an unmanaged (value) type");
                new Action(() => import.MakeGenericMethod(typeof(string))).Should().Throw<ArgumentException>(import.ToString());
            }
        }
    }
}
