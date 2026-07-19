using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.PythonNet;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Holds the interop documentation's <i>prose</i> to the code, where the other
    ///     <c>DocExamples_*</c> classes hold its <i>examples</i> to the code.
    ///
    ///     <para>Two things rot silently in documentation that no build can see: a "proven by
    ///     <c>SomeTest</c>" citation whose test has since been renamed, and a quoted suite size that
    ///     stopped being true three commits ago. Both are checked here against the assembly itself.</para>
    ///
    ///     <para>The pages are copied into the test output by the csproj. If the copy is missing these
    ///     tests <b>fail</b> rather than skip — a doc gate that quietly asserts nothing when its inputs
    ///     vanish is worse than no gate at all.</para>
    /// </summary>
    [TestClass]
    public class DocExamples_DocIntegrity
    {
        private static string DocsDirectory
            => Path.Combine(Path.GetDirectoryName(typeof(DocExamples_DocIntegrity).Assembly.Location), "docs", "interop");

        private static IReadOnlyList<(string File, string Text)> Pages()
        {
            Directory.Exists(DocsDirectory).Should().BeTrue(
                $"the interop doc pages must be copied to '{DocsDirectory}' (see the csproj None rule)");

            var pages = Directory.GetFiles(DocsDirectory, "*.md")
                                 .Select(f => (File: Path.GetFileName(f), Text: File.ReadAllText(f)))
                                 .ToList();
            pages.Should().NotBeEmpty("the doc-integrity gate must never run against zero pages");
            return pages;
        }

        private static HashSet<string> TestMethodNames()
            => typeof(DocExamples_DocIntegrity).Assembly
                   .GetTypes()
                   .Where(t => t.GetCustomAttribute<TestClassAttribute>() is not null)
                   .SelectMany(t => t.GetMethods(BindingFlags.Public | BindingFlags.Instance))
                   .Where(m => m.GetCustomAttribute<TestMethodAttribute>() is not null)
                   .Select(m => m.Name)
                   .ToHashSet(StringComparer.Ordinal);

        /// <summary>
        ///     Every test name cited by a page — "Proven by <c>X</c>", "see <c>Class.Method</c>" — must
        ///     resolve to a real <c>[TestMethod]</c> in this assembly.
        /// </summary>
        [TestMethod]
        public void EveryTestNameCitedByTheDocs_ExistsInThisAssembly()
        {
            var known = TestMethodNames();

            // A citation is a backticked identifier shaped like a test name: PascalCase with at least
            // one underscore-joined segment, optionally qualified by its class.
            var citation = new Regex(@"`(?<name>(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)`",
                                     RegexOptions.Compiled);

            var missing = new List<string>();
            int checkedCount = 0;

            foreach (var (file, text) in Pages())
            foreach (Match m in citation.Matches(text))
            {
                string cited = m.Groups["name"].Value;
                string method = cited.Contains('.') ? cited[(cited.LastIndexOf('.') + 1)..] : cited;

                // Only names the docs present as tests — every citation in these pages is one, but be
                // explicit rather than clever: skip anything the assembly has no chance of owning.
                if (!known.Contains(method) && !LooksLikeATestCitation(text, m.Index))
                    continue;

                checkedCount++;
                if (!known.Contains(method))
                    missing.Add($"{file}: `{cited}`");
            }

            checkedCount.Should().BeGreaterThan(0,
                "the docs cite tests by name; finding none means the citation pattern stopped matching");
            missing.Should().BeEmpty(
                "a renamed test leaves the documentation citing something that no longer exists");
        }

        /// <summary>A citation is one if the sentence introducing it says so.</summary>
        private static bool LooksLikeATestCitation(string text, int index)
        {
            int from = Math.Max(0, index - 60);
            string lead = text.Substring(from, index - from);
            return lead.Contains("Proven by", StringComparison.OrdinalIgnoreCase)
                || lead.Contains("corresponds", StringComparison.OrdinalIgnoreCase)
                || lead.Contains("test", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        ///     The Testing section of <c>pythonnet.md</c> quotes the suite size. Reflection knows the
        ///     real number, so the sentence cannot drift.
        /// </summary>
        [TestMethod]
        public void TheSuiteSizeQuotedByPythonnetMd_IsTheRealTestCount()
        {
            var page = Pages().SingleOrDefault(p => p.File == "pythonnet.md");
            page.Text.Should().NotBeNull("pythonnet.md must be among the copied pages");

            var quoted = Regex.Match(page.Text, @"\((?<n>\d+) tests on each of `net8\.0` and `net10\.0`\)");
            quoted.Success.Should().BeTrue("pythonnet.md's Testing section states the suite size");

            int actual = TestMethodNames().Count;
            int.Parse(quoted.Groups["n"].Value).Should().Be(actual,
                $"the docs quote a suite size; this assembly declares {actual} test methods");
        }

        /// <summary>
        ///     Every <c>*Interop</c> type name the documentation mentions must exist in the shipped
        ///     assembly.
        ///
        ///     <para>This exists because it already happened: the package README kept telling users to
        ///     call <c>NDArrayInterop.RegisterCodec()</c> for several commits after the type was renamed
        ///     to <c>NDArrayPythonInterop</c> — samples that cannot compile, shipped inside the NuGet
        ///     package. A rename is invisible to a markdown file, so the check has to be explicit.</para>
        /// </summary>
        [TestMethod]
        public void EveryInteropTypeNameMentionedByTheDocs_ExistsInTheShippedAssembly()
        {
            var shipped = typeof(NumSharp.Interop.PythonNet.NDArrayPythonInterop).Assembly
                              .GetTypes().Select(t => t.Name).ToHashSet(StringComparer.Ordinal);

            // Deliberately narrow: names ending in "Interop" are this package's own types, so any one
            // the docs use is either real or a leftover from a rename.
            var mention = new Regex(@"\b(?<name>[A-Z][A-Za-z0-9]*Interop)\b", RegexOptions.Compiled);

            var stale = new List<string>();
            int seen = 0;

            foreach (var (file, text) in AllDocuments())
            foreach (Match m in mention.Matches(text))
            {
                string name = m.Groups["name"].Value;
                if (name.EndsWith("UnitTests", StringComparison.Ordinal))
                    continue;   // the test assembly's own namespace, not a package type

                seen++;
                if (!shipped.Contains(name))
                    stale.Add($"{file}: {name}");
            }

            seen.Should().BeGreaterThan(0, "the docs name this package's types; matching none means the pattern broke");
            stale.Distinct().Should().BeEmpty(
                "documentation naming a type that no longer exists ships samples that cannot compile");
        }

        /// <summary>
        ///     <c>pythonnet.md</c> quotes the version guard's error in a fenced block, and repeats the
        ///     same advice as an XML snippet in the paragraph above it. Both name a pythonnet version —
        ///     the drift-prone part — and both must agree with what the guard would actually say.
        ///
        ///     <para>Checked from the outside, against the mapping the guard uses, rather than by
        ///     reshaping the guard so its message can be built for a pairing this process cannot be in.</para>
        /// </summary>
        [TestMethod]
        public void TheVersionAdviceQuotedByPythonnetMd_IsWhatTheGuardWouldSay()
        {
            string page = Pages().Single(p => p.File == "pythonnet.md").Text;

            var quoted = Regex.Match(page,
                @"Python (?<python>\d+\.\d+) is not supported by the loaded pythonnet [\d.]+[\s\S]*?" +
                @"Upgrade pythonnet to (?<advised>[\d.]+) or later: " +
                @"<PackageReference Include=""pythonnet"" Version=""(?<pinned>[\d.]+)"" />");
            quoted.Success.Should().BeTrue("pythonnet.md reproduces the guard's error in a fenced block");

            string advised = quoted.Groups["advised"].Value;
            string expected = PythonRuntimeInterop.MinimumPythonnetFor(Version.Parse(quoted.Groups["python"].Value));

            advised.Should().Be(expected,
                "the error block tells users to install a version; it must be the one the guard names");
            quoted.Groups["pinned"].Value.Should().Be(advised,
                "the prose and the PackageReference inside the same message must not disagree");

            // The Installation section repeats that PackageReference on its own — same version.
            Regex.Matches(page, @"<PackageReference Include=""pythonnet"" Version=""(?<v>[\d.]+)"" />")
                 .Select(m => m.Groups["v"].Value).Distinct()
                 .Should().ContainSingle(v => v == advised,
                     "every pythonnet PackageReference the page shows should name the same version");
        }

        /// <summary>The website pages plus the README that ships inside the NuGet package.</summary>
        private static IReadOnlyList<(string File, string Text)> AllDocuments()
        {
            var all = Pages().ToList();

            string packageReadme = Path.Combine(
                Path.GetDirectoryName(typeof(DocExamples_DocIntegrity).Assembly.Location), "docs", "package", "README.md");
            File.Exists(packageReadme).Should().BeTrue(
                $"the package README must be copied to '{packageReadme}' (see the csproj None rule)");
            all.Add(("package/README.md", File.ReadAllText(packageReadme)));

            return all;
        }

        /// <summary>
        ///     Both index pages promise that "every sample maps 1:1 to a test". That promise is only
        ///     credible if each page with samples actually has a companion class here.
        /// </summary>
        [TestMethod]
        public void EveryInteropPageWithSamples_HasACompanionTestClass()
        {
            var companions = new Dictionary<string, string>
            {
                ["pythonnet.md"] = nameof(DocExamples_PythonnetPage),
                ["zero-copy-model.md"] = nameof(DocExamples_ZeroCopyModelPage),
                ["index.md"] = nameof(DocExamples_InteropIndexPage),
                ["numpy-net.md"] = nameof(NumpyNetInteropTests),
            };

            var present = Pages().Select(p => p.File).ToHashSet(StringComparer.OrdinalIgnoreCase);
            var types = typeof(DocExamples_DocIntegrity).Assembly.GetTypes().Select(t => t.Name).ToHashSet(StringComparer.Ordinal);

            foreach (var (page, companion) in companions)
            {
                present.Should().Contain(page, "the interop docs are expected to ship this page");
                types.Should().Contain(companion, $"{page} promises its samples are proven by tests");
            }

            // ...and no page slipped in without one.
            present.Where(p => !companions.ContainsKey(p))
                   .Should().BeEmpty("a new interop page needs a companion DocExamples_* class");
        }
    }
}
