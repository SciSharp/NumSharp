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
    ///     Text and the System.Numerics.Tensors bridge's UNMANAGED verbs: <c>AsTensorSpan&lt;T&gt;</c> / <c>ToTensor&lt;T&gt;</c>
    ///     are <c>where T : unmanaged</c>, and NumSharp's text dtypes — fixed-width <c>U</c> / <c>S</c> and
    ///     <c>StringDType</c>, which share the <see cref="NPTypeCode.String"/> storage lane — have no such element type. So
    ///     every unmanaged export verb refuses every text array, of every class, width and layout, with a
    ///     <see cref="NotSupportedException"/> before anything is read, leased or pinned, naming text, the array's class and
    ///     <c>ToStringTensor</c>, which copies text into a <c>Tensor&lt;string&gt;</c> (<c>TextCrossingTests</c>).
    /// </summary>
    /// <remarks>
    ///     <para>The dtype map and the verbs throw ONE text (<c>NDArrayTensorsInterop.TextRefusal</c>), the ML.NET bridge's
    ///     rule. The verbs used to hand the array's type code to the map, and the implicit <c>NPTypeCode → DType</c>
    ///     conversion threw Core's storage-lane error first ("… asked for a dtype by storage lane …"), which told the caller
    ///     to pass a dtype they never asked for and called a <c>StringDType</c> array fixed-width; the owner chose "Shared
    ///     text refusal" on 2026-11-05.</para>
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
        ///     A text array's export refusal names TEXT, the array's own class — "a fixed-width text array (&lt;U5)", "a
        ///     StringDType array (StringDType(na_object=None))", never calling a <c>StringDType</c> array fixed-width — and
        ///     <c>ToStringTensor()</c>, the verb that copies it; never Core's storage-lane error ("… asked for a dtype by storage
        ///     lane …"), which these verbs used to leak by handing the map the array's type code (fixed 2026-11-05, the owner's
        ///     "Shared text refusal").
        /// </summary>
        [TestMethod]
        public void TextExports_RefuseWithATextMessage_NotTheStorageLaneError()
        {
            foreach (var (name, text) in TextArrays())
            foreach (var (verb, call) in ExportVerbs(text))
            {
                string message = call.Should().Throw<NotSupportedException>($"{name} / {verb}").Which.Message;
                message.Should().NotContain("storage lane", $"{name} / {verb}: Core's storage-lane error leaked out");
                message.Should().Contain("text", $"{name} / {verb}: the refusal names text");
                message.Should().Contain("nd.ToStringTensor()", $"{name} / {verb}: the refusal names the verb that crosses text");
                message.Should().Contain(text.dtype.kind == 'T' ? $"a StringDType array ({text.dtype})" : $"a fixed-width text array ({text.dtype.str})",
                    $"{name} / {verb}: the refusal names the array's class");
                if (text.dtype.kind == 'T')
                    message.Should().NotContain("fixed-width", $"{name} / {verb}: a StringDType array is not fixed-width");
            }
        }

        /// <summary>
        ///     The dtype map refuses a text DESCRIPTOR (<c>U5</c>, <c>S3</c>, <c>StringDType()</c>) with the same text the verbs
        ///     throw — the class and what <c>ToStringTensor()</c> does with it (copies; decodes an <c>S</c> array's bytes as
        ///     UTF-8; turns a <c>StringDType</c> NA into null) — while NumSharp's 16-bit <see cref="NPTypeCode.Char"/> (a
        ///     number, not text) maps to <see cref="char"/>.
        /// </summary>
        [TestMethod]
        public void DtypeMap_RefusesTextDescriptors_CharIsANumber()
        {
            const string Head = "NumSharp text does not cross into System.Numerics.Tensors' unmanaged element types: ";
            const string Mid = ", and the bridge maps the numeric and boolean dtypes to their own CLR types. nd.ToStringTensor() ";
            var expected = new (DType Dtype, string Message)[]
            {
                (np.dtype("U5"), Head + "this is a fixed-width text array (<U5)" + Mid + "copies its texts into a Tensor<string>."),
                (np.dtype("S3"), Head + "this is a fixed-width text array (|S3)" + Mid + "decodes its bytes as UTF-8 (np.strings.decode) into a Tensor<string>."),
                (np.dtypes.StringDType(), Head + "this is a StringDType array (StringDType())" + Mid + "copies its texts into a Tensor<string>, an NA as null."),
            };
            foreach (var (text, message) in expected)
                new Action(() => NDArrayTensorsInterop.ToTensorElementClrType(text)).Should().Throw<NotSupportedException>(text.ToString())
                    .Which.Message.Should().Be(message, text.ToString());
            NDArrayTensorsInterop.ToTensorElementClrType(NPTypeCode.Char).Should().Be(typeof(char), "NumSharp's char is a number, not text");
        }

        /// <summary>
        ///     Text never reaches the GENERIC import verbs: every <c>ToNDArray&lt;T&gt;</c> / <c>AsNDArray&lt;T&gt;</c> is
        ///     <c>where T : unmanaged</c>, so even reflection cannot instantiate one for <see cref="string"/> —
        ///     <see cref="MethodInfo.MakeGenericMethod"/> refuses the constraint violation before any call. A
        ///     <c>Tensor&lt;string&gt;</c> imports through the non-generic <c>ToNDArray</c> overloads instead (a copy into
        ///     <c>StringDType</c>; <c>TextCrossingTests</c>), and there is no <c>AsNDArray</c> for it: no string can be viewed.
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
