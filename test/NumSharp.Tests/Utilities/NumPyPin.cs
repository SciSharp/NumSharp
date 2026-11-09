using System;
using System.Linq;
using System.Reflection;

namespace NumSharp.Tests
{
    /// <summary>
    ///     The NumPy release every committed corpus of this repository is generated with: the one line of
    ///     <c>numpy-version.txt</c> at the repository root, compiled into this assembly as the <c>NumPyVersion</c>
    ///     <see cref="AssemblyMetadataAttribute"/> (<c>NumSharp.Tests.csproj</c>). Compare a corpus's recorded NumPy
    ///     against <see cref="Version"/> instead of a literal, so a NumPy bump stays one edit of the pin file and a corpus
    ///     nobody regenerated fails its gate.
    /// </summary>
    /// <remarks>
    ///     The value is fixed at BUILD time: after editing <c>numpy-version.txt</c>, rebuild before running the gates (a
    ///     <c>--no-build</c> run still compares against the release the assembly was built with).
    /// </remarks>
    internal static class NumPyPin
    {
        /// <summary>
        ///     The pinned NumPy release, for example <c>"2.5.4"</c>.
        /// </summary>
        /// <exception cref="InvalidOperationException">On first read, when the assembly carries no non-empty
        ///     <c>NumPyVersion</c> metadata (a build that bypassed the csproj's pin check); surfaced through the type
        ///     initializer as a <see cref="TypeInitializationException"/>.</exception>
        public static string Version { get; } = Read();

        /// <summary>
        ///     Reads the <c>NumPyVersion</c> assembly metadata the csproj compiles in from <c>numpy-version.txt</c>.
        /// </summary>
        /// <returns>The pinned release, trimmed (the csproj already trims the file's line ending).</returns>
        /// <exception cref="InvalidOperationException">The metadata is missing or empty: failing loudly here keeps every
        ///     version gate from comparing against an empty string and reporting a misleading corpus mismatch.</exception>
        private static string Read()
        {
            string version = typeof(NumPyPin).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
                .FirstOrDefault(a => a.Key == "NumPyVersion")?.Value?.Trim();
            if (string.IsNullOrEmpty(version))
                throw new InvalidOperationException(
                    "NumSharp.Tests carries no NumPyVersion assembly metadata: build it through NumSharp.Tests.csproj, which reads numpy-version.txt.");
            return version;
        }
    }
}
