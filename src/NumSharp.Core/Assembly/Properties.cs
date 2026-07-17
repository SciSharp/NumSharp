using System.Runtime.CompilerServices;

// Friend assemblies. Every name carries PublicKey= because NumSharp is strong-named (see
// Directory.Build.props at the repo root): the compiler rejects a keyless InternalsVisibleTo from a
// signed assembly with CS1726, and the signing configuration used to sit in a `Publish` config that
// nothing built — so these went keyless and unnoticed, and `Publish` did not compile at all.
//
// The key is Microsoft's PUBLISHED open-source key (token cc7b13ffcd2ddd51). Its private half is
// public by design, so this list is NOT an access control — anyone can build an assembly with a
// matching name and key. It is a compile-time convenience, exactly as it was when unsigned.
//
// The literal lives once, in NumSharpFriendKey.PublicKey below; `"name, PublicKey=" + const` is a
// compile-time constant, which is what an attribute argument requires. Global attributes must
// precede type declarations, so the holder class follows them.

[assembly: InternalsVisibleTo("NumSharp.UnitTest, PublicKey=" + NumSharpFriendKey.PublicKey)]

// Optional out-of-box backends: they subclass DefaultEngine and so need the same internal view of
// Shape (strides/offset) and the promotion helpers that the built-in kernels have.
[assembly: InternalsVisibleTo("NumSharp.Interop.OpenBLAS, PublicKey=" + NumSharpFriendKey.PublicKey)]

// The Python.NET interop bridge reads the same internal view of Shape (strides/offset) to convert
// NDArray <-> numpy (and any PEP 3118 buffer exporter) zero-copy, both ways.
[assembly: InternalsVisibleTo("NumSharp.Interop.pythonnet, PublicKey=" + NumSharpFriendKey.PublicKey)]

// The Python.NET interop test assembly exercises the same internals.
[assembly: InternalsVisibleTo("NumSharp.Interop.UnitTests, PublicKey=" + NumSharpFriendKey.PublicKey)]

[assembly: InternalsVisibleTo("NumSharp.Benchmark, PublicKey=" + NumSharpFriendKey.PublicKey)]

// Cross-repo: TensorFlow.NET signs with the identical key, which is the reason NumSharp keeps using
// Microsoft's open key rather than minting its own — a NumSharp-owned key would break this line.
[assembly: InternalsVisibleTo("TensorFlowNET.UnitTest, PublicKey=" + NumSharpFriendKey.PublicKey)]

// `dotnet run` file-based scripts that override AssemblyName to this (see .claude/CLAUDE.md ->
// "Scripting with dotnet run"). A script UNDER the repo root inherits Directory.Build.props and is
// signed automatically; one run from elsewhere must pass the signing properties itself, or it will
// build unsigned and see no internals.
[assembly: InternalsVisibleTo("NumSharp.DotNetRunScript, PublicKey=" + NumSharpFriendKey.PublicKey)]

/// <summary>
///     The public half of the strong-name key NumSharp is signed with, as the hex literal
///     <see cref="InternalsVisibleToAttribute"/> expects.
/// </summary>
/// <remarks>
///     Mirrors the <c>NumSharpPublicKey</c> property in the repo-root <c>Directory.Build.props</c>.
///     Both are pinned against the actual compiled output by
///     <c>test/NumSharp.UnitTest/Assembly/StrongNameTests.cs</c>, so the two copies cannot drift
///     apart silently — a mismatch turns that gate red rather than degrading into a friend
///     reference that no longer resolves.
/// </remarks>
internal static class NumSharpFriendKey
{
    internal const string PublicKey =
        "00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34" +
        "bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbb" +
        "ad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8" +
        "fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb";
}
