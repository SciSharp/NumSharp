using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Layering
{
    /// <summary>
    ///     Unit tests of <see cref="LayeringScanner.Strip"/>: the two ratchets count only code, so comments and literal
    ///     contents must disappear while every newline stays (a count's line number is quoted in failure messages).
    /// </summary>
    [TestClass]
    public class LayeringScannerTests
    {
        /// <summary>Line and block comments are blanked; the code around them and every newline stay.</summary>
        [TestMethod]
        [TestCategory("Layering")]
        public void Strip_RemovesComments_KeepsNewlines()
        {
            string source = "a(); // np.sum(x)\n/* NDStringUfuncs.Guard(\n */ b();\n/// <see cref=\"np.max\"/>\nc();";
            string stripped = LayeringScanner.Strip(source);
            Assert.AreEqual(source.Count(ch => ch == '\n'), stripped.Count(ch => ch == '\n'), "newlines must be kept");
            StringAssert.Contains(stripped, "a();");
            StringAssert.Contains(stripped, "b();");
            StringAssert.Contains(stripped, "c();");
            Assert.IsFalse(stripped.Contains("np.sum"), "a line comment must be blanked");
            Assert.IsFalse(stripped.Contains("NDStringUfuncs"), "a block comment must be blanked");
            Assert.IsFalse(stripped.Contains("np.max"), "a doc comment must be blanked");
        }

        /// <summary>Regular, verbatim, interpolated and raw strings and char literals keep their quotes and lose their contents.</summary>
        [TestMethod]
        [TestCategory("Layering")]
        public void Strip_BlanksLiteralContents()
        {
            string source = "x(\"np.sum(a)\", @\"NPTypeCode.String \"\" more\", $\"{np.zeros(3)}\", '\\'', 'q');\n" +
                            "y(\"\"\"np.copyto(b)\"\"\", \"esc \\\" np.max(c)\");\nnp.real(z);";
            string stripped = LayeringScanner.Strip(source);
            Assert.AreEqual(2, stripped.Count(ch => ch == '\n'));
            Assert.IsFalse(stripped.Contains("np.sum"), "a regular string's contents must be blanked");
            Assert.IsFalse(stripped.Contains("NPTypeCode.String"), "a verbatim string's contents must be blanked");
            Assert.IsFalse(stripped.Contains("np.zeros"), "an interpolated string's contents must be blanked");
            Assert.IsFalse(stripped.Contains("np.copyto"), "a raw string's contents must be blanked");
            Assert.IsFalse(stripped.Contains("np.max"), "an escaped quote must not end the string early");
            StringAssert.Contains(stripped, "np.real(z);", "code after the literals must survive");
        }

        /// <summary>A verbatim identifier (<c>np.@char</c>) is code, not a string prefix.</summary>
        [TestMethod]
        [TestCategory("Layering")]
        public void Strip_KeepsVerbatimIdentifiers()
        {
            string stripped = LayeringScanner.Strip("var u = np.@char.upper(a);\n");
            StringAssert.Contains(stripped, "np.@char.upper(a);");
        }

        /// <summary>
        ///     The engine scan counts module calls (<c>np.linalg.*</c>, <c>np.@char.*</c>) and generic calls, and the facade
        ///     scan counts both kinds of dtype test, over a synthetic tree laid out like <c>src/NumSharp.Core</c>.
        /// </summary>
        [TestMethod]
        [TestCategory("Layering")]
        public void Scans_CountWhatThePatternsDescribe()
        {
            string root = Path.Combine(Path.GetTempPath(), "layering_scan_" + Guid.NewGuid().ToString("N"));
            try
            {
                string core = Path.Combine(root, "src", "NumSharp.Core");
                Directory.CreateDirectory(Path.Combine(core, "Backends", "Strings"));
                Directory.CreateDirectory(Path.Combine(core, "Backends", "Default"));
                Directory.CreateDirectory(Path.Combine(core, "Math"));
                File.WriteAllText(Path.Combine(core, "Backends", "Strings", "NDStringUfuncs.cs"),
                    "internal static class NDStringUfuncs { }\ninternal enum Op { A }\n");
                File.WriteAllText(Path.Combine(core, "Backends", "Default", "Default.Thing.cs"),
                    "var a = np.zeros(3);\nvar b = np.linalg.inv(a) + np.@char.upper(a) + np.array<int>(c);\n// np.sum(a)\n" +
                    "var s = \"np.max(a)\";\n");
                File.WriteAllText(Path.Combine(core, "Math", "np.thing.cs"),
                    "if (NDStringUfuncs.IsText(a)) x();\nif (a.typecode == NPTypeCode.Double) y();\n" +
                    "switch (a.typecode) {\ncase NPTypeCode.Half: break; }\nvar t = a.astype(NPTypeCode.Double);\n" +
                    "if (a.typecode == NPTypeCode.String) z();\nvar o = Op.A;\n");

                var engine = LayeringScanner.CountEngineUpCalls(core, out _);
                Assert.AreEqual((2, 2), engine["Backends/Default/Default.Thing.cs"],
                    "np.zeros + np.array<int> are structural; np.linalg.inv + np.@char.upper are computations; " +
                    "the comment and the string are not calls");

                var facade = LayeringScanner.CountFacadeDtypeTests(core, out var lines);
                Assert.AreEqual((2, 3), facade["Math/np.thing.cs"],
                    "text: the backend call and NPTypeCode.String; numeric: the comparison, the switch header, the case arm; " +
                    "the astype use and the short type name Op are not tests. Lines:\n" + string.Join("\n", lines["Math/np.thing.cs"]));
            }
            finally
            {
                if (Directory.Exists(root))
                    Directory.Delete(root, recursive: true);
            }
        }

        /// <summary>
        ///     The facade scan reads the <c>np.*</c> files and the <c>NDArray.*</c> files of the ndarray methods, but not the
        ///     array object's homes (<see cref="LayeringScanner.ArrayObjectHomes"/>), whose statements are NumPy's.
        /// </summary>
        [TestMethod]
        [TestCategory("Layering")]
        public void FacadeScan_LeavesOutTheArrayObjectHomes()
        {
            string root = Path.Combine(Path.GetTempPath(), "layering_homes_" + Guid.NewGuid().ToString("N"));
            try
            {
                string core = Path.Combine(root, "src", "NumSharp.Core");
                // the scan reads Backends/Strings for the text-backend type names, so the tree needs the folder
                Directory.CreateDirectory(Path.Combine(core, "Backends", "Strings"));
                Directory.CreateDirectory(Path.Combine(core, "Creation"));
                Directory.CreateDirectory(Path.Combine(core, "Manipulation"));
                const string test = "if (a.typecode == NPTypeCode.Double) y();\n";
                File.WriteAllText(Path.Combine(core, "Creation", "np.thing.cs"), test);
                File.WriteAllText(Path.Combine(core, "Creation", "NDArray.Multiarray.cs"), test);
                File.WriteAllText(Path.Combine(core, "Manipulation", "NDArray.FunctionBase.cs"), test);
                File.WriteAllText(Path.Combine(core, "Manipulation", "NDArray.fill.cs"), test);

                var files = LayeringScanner.FacadeFiles(core).ToList();
                CollectionAssert.AreEqual(new[] { "Creation/np.thing.cs", "Manipulation/NDArray.fill.cs" }, files,
                    "the homes must be left out, a facade and an ndarray method's file kept: " + string.Join(", ", files));
                var counts = LayeringScanner.CountFacadeDtypeTests(core, out _);
                Assert.AreEqual((0, 1), counts["Creation/np.thing.cs"]);
                Assert.IsFalse(counts.ContainsKey("Creation/NDArray.Multiarray.cs"), "a home must not be counted");
            }
            finally
            {
                if (Directory.Exists(root))
                    Directory.Delete(root, recursive: true);
            }
        }
    }
}
