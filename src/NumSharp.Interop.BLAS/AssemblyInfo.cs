using System.Runtime.CompilerServices;

// Same friend pattern as NumSharp.Core/Assembly/Properties.cs: the fuzz gate replays the
// host-pinned matmul_parity corpus through this engine, and NumSharp.DotNetRunScript lets ad-hoc
// `dotnet run` probes reach internals by overriding their AssemblyName.
//
// PublicKey= is required because this assembly is strong-named too (repo-root
// Directory.Build.props) — a keyless name is CS1726 from a signed assembly. The key is the same
// published Microsoft open key NumSharp.Core uses; see that file for why it is identity, not
// access control. The literal is duplicated rather than shared because NumSharp.Core's copy is
// `internal` to Core, and this project reaching for it would invert the dependency it is
// declaring. StrongNameTests pins both against the compiled output.
[assembly: InternalsVisibleTo("NumSharp.UnitTest, PublicKey=" + BlasFriendKey.PublicKey)]
[assembly: InternalsVisibleTo("NumSharp.DotNetRunScript, PublicKey=" + BlasFriendKey.PublicKey)]

internal static class BlasFriendKey
{
    internal const string PublicKey =
        "00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34" +
        "bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbb" +
        "ad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8" +
        "fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb";
}
