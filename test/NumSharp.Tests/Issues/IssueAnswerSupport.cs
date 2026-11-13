using System;
using System.IO;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Shared helpers for the tests that back the answers posted to GitHub issues in the 2026-11-08 triage after the
    ///     0.80.0 release (one <c>Issue&lt;number&gt;</c> class per answered issue). Each of those classes turns the claims
    ///     of one posted answer into assertions, so a regression that would make an answer untrue fails a test that names
    ///     the issue.
    /// </summary>
    internal static class IssueAnswerSupport
    {
        /// <summary>
        ///     Finds the repository root (the folder that holds <c>ARCHITECTURE.md</c> and <c>src/NumSharp.Core</c>) by
        ///     walking up from the test assembly's folder, so a test can check the documents an answer links to.
        /// </summary>
        /// <returns>The absolute path of the repository root.</returns>
        /// <exception cref="AssertInconclusiveException">
        ///     The tests run outside a checkout (for example from a copied test folder), so there is no repository to read:
        ///     the documentation claims cannot be checked there, which is not a failure of NumSharp.
        /// </exception>
        public static string RepoRoot()
        {
            for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
            {
                // Both markers, because a test output folder can contain a copied ARCHITECTURE.md but never the source tree.
                if (File.Exists(Path.Combine(dir.FullName, "ARCHITECTURE.md")) &&
                    Directory.Exists(Path.Combine(dir.FullName, "src", "NumSharp.Core")))
                    return dir.FullName;
            }

            Assert.Inconclusive("the tests are not running inside a NumSharp checkout, so the repository documents cannot be read");
            return null; // unreachable: Assert.Inconclusive throws
        }

        /// <summary>Reads a repository file as text, failing with the file name when an answer's link would be dead.</summary>
        /// <param name="relativePath">The path below the repository root, with forward slashes.</param>
        /// <returns>The file's text.</returns>
        /// <exception cref="AssertFailedException">The file does not exist, so a link posted in an answer is broken.</exception>
        public static string ReadRepoFile(string relativePath)
        {
            var path = Path.Combine(RepoRoot(), relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.IsTrue(File.Exists(path), $"{relativePath} is linked from a posted answer but does not exist");
            return File.ReadAllText(path);
        }

        /// <summary>
        ///     Asserts the platform facts several "not planned" answers rest on (#424, #430, #455, #498, #511): the NumSharp
        ///     assembly is built for .NET 8 or .NET 10 only (no .NET Framework, Mono or netstandard build), and its kernels are
        ///     <c>DynamicMethod</c>s emitted at run time, so it needs a runtime that can JIT dynamic code (not IL2CPP or Native
        ///     AOT).
        /// </summary>
        /// <exception cref="AssertFailedException">A fact the answers state no longer holds.</exception>
        public static void AssertNet8Or10_WithRuntimeEmittedKernels()
        {
            var framework = typeof(np).Assembly.GetCustomAttribute<TargetFrameworkAttribute>()?.FrameworkName;
            framework.Should().Match(f => f == ".NETCoreApp,Version=v8.0" || f == ".NETCoreApp,Version=v10.0",
                "NumSharp ships net8.0 and net10.0 assemblies only");

            RuntimeFeature.IsDynamicCodeSupported.Should().BeTrue("NumSharp needs a runtime that can emit and JIT code");
            var kernel = DirectILKernelGenerator.GetMixedTypeKernel(
                new MixedTypeKernelKey(NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double, BinaryOp.Add, ExecutionPath.SimdFull));
            kernel.Should().NotBeNull();
            kernel.Method.Should().BeAssignableTo<DynamicMethod>("binary kernels are IL emitted at run time");
        }

        /// <summary>The IEEE-754 bit pattern of a float32, for assertions that are bit-exact rather than approximate.</summary>
        /// <param name="value">The value.</param>
        /// <returns>Its 32 bits.</returns>
        public static uint Bits(float value) => BitConverter.SingleToUInt32Bits(value);

        /// <summary>The IEEE-754 bit pattern of a float64, for assertions that are bit-exact rather than approximate.</summary>
        /// <param name="value">The value.</param>
        /// <returns>Its 64 bits.</returns>
        public static ulong Bits(double value) => BitConverter.DoubleToUInt64Bits(value);
    }
}
