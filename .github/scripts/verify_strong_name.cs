#:property PublishAot=false
// -----------------------------------------------------------------------------------------------
// Verifies that every managed assembly inside a built .nupkg is strong-named with NumSharp's key.
//
// WHY THIS EXISTS. Signing used to be configured in a `Publish|AnyCPU` PropertyGroup while CI built
// and packed `Release`, so it never applied and every published NumSharp shipped with
// `PublicKeyToken=null` — for six years, with nothing red. That failure mode produces output which
// builds cleanly, tests cleanly and packs cleanly; it is invisible to every gate that is not
// specifically looking at the identity of the packed artifact. So the release pipeline looks at it.
//
// This runs against the PACKED PACKAGE, not the build output, because the package is the thing
// users install and it is the last point where a stray `<SignAssembly>false</SignAssembly>`, an
// overriding csproj property, or a lost key file can still be caught.
//
// Cross-platform and dependency-free: reads the PE metadata directly out of the zip, so it needs
// no `sn.exe` (Windows-only) and no temp files. CI runs it on ubuntu-latest.
//
//   dotnet run .github/scripts/verify_strong_name.cs -- <dir-or-nupkg> [more...]     (default: artifacts/nuget)
//
// Exit 0 = every assembly signed with the expected key. Exit 1 = anything else, INCLUDING finding
// nothing to check: a verifier that passes vacuously is worse than no verifier, because it reads as
// a green check. The repo has been bitten by exactly that before (NumpyCompatibilityTests silently
// asserting nothing on a clean checkout), so absence is a failure here, not a skip.
// -----------------------------------------------------------------------------------------------
using System.IO.Compression;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Security.Cryptography;

const string ExpectedToken = "cc7b13ffcd2ddd51";
const string ExpectedPublicKey =
    "00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34" +
    "bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbb" +
    "ad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8" +
    "fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb";

string[] inputs = args.Length > 0 ? args : new[] { "artifacts/nuget" };

var packages = new List<string>();
foreach (var input in inputs)
{
    if (Directory.Exists(input)) packages.AddRange(Directory.GetFiles(input, "*.nupkg", SearchOption.AllDirectories));
    else if (File.Exists(input)) packages.Add(input);
    else { Console.Error.WriteLine($"::error::verify_strong_name: input not found: {input}"); return 1; }
}
// .snupkg carries only symbols; it has no lib/ assemblies and would look like an empty package.
packages = packages.Where(p => !p.EndsWith(".snupkg", StringComparison.OrdinalIgnoreCase))
                   .Distinct().OrderBy(p => p).ToList();

if (packages.Count == 0)
{
    Console.Error.WriteLine($"::error::verify_strong_name: no .nupkg found under [{string.Join(", ", inputs)}]. " +
                            "Refusing to pass vacuously — if the pack step produced nothing, that IS the failure.");
    return 1;
}

int checkedCount = 0, failures = 0;
Console.WriteLine($"Expecting PublicKeyToken={ExpectedToken}\n");

foreach (var pkg in packages)
{
    Console.WriteLine($"{Path.GetFileName(pkg)}");
    using var zip = ZipFile.OpenRead(pkg);

    // NumSharp.Build is a TOOLS-ONLY package BY DESIGN: build/ targets + a tools/ payload, no lib/
    // and no dependency entries — installing it changes the consumer's BUILD, never its dependency
    // graph (gate: tools/verify_weaver_package.sh). Its payload is exempt from the key check — the
    // tool never loads into a user's app (tools/ assemblies are neither referenced nor copied), and
    // Mono.Cecil beside it carries Cecil's key, not NumSharp's. What IS asserted is the payload
    // itself, so a mis-pack cannot ride the exemption and ship an empty shell.
    if (Path.GetFileName(pkg).StartsWith("NumSharp.Build.", StringComparison.OrdinalIgnoreCase))
    {
        bool hasTool = zip.Entries.Any(e => e.FullName.Equals("tools/net8.0/any/NumSharp.Build.dll", StringComparison.OrdinalIgnoreCase));
        bool hasCecil = zip.Entries.Any(e => e.FullName.Equals("tools/net8.0/any/Mono.Cecil.dll", StringComparison.OrdinalIgnoreCase));
        bool hasTargets = zip.Entries.Any(e => e.FullName.Equals("build/NumSharp.Build.targets", StringComparison.OrdinalIgnoreCase));
        bool hasLib = zip.Entries.Any(e => e.FullName.StartsWith("lib/", StringComparison.OrdinalIgnoreCase));

        checkedCount++;
        if (hasTool && hasCecil && hasTargets && !hasLib)
        {
            Console.WriteLine("  OK   tools-only weaver package (build/ + tools/ payload verified; key check n/a)");
        }
        else
        {
            Console.Error.WriteLine($"  ::error::FAIL {Path.GetFileName(pkg)} — tools-only package malformed " +
                                    $"(tool={hasTool}, cecil={hasCecil}, targets={hasTargets}, lib-present={hasLib})");
            failures++;
        }
        continue;
    }

    // Only lib/ — a package may also carry runtimes/<rid>/native/*.dll (NumSharp.Interop.OpenBLAS ships
    // OpenBLAS), which are unmanaged and have no assembly identity to check.
    var libDlls = zip.Entries
        .Where(e => e.FullName.StartsWith("lib/", StringComparison.OrdinalIgnoreCase)
                    && e.FullName.EndsWith(".dll", StringComparison.OrdinalIgnoreCase))
        .OrderBy(e => e.FullName).ToList();

    if (libDlls.Count == 0)
    {
        Console.Error.WriteLine($"  ::error::no lib/**/*.dll inside {Path.GetFileName(pkg)} — a shipping " +
                                 "package with no managed assembly is a packaging failure, not a pass.");
        failures++;
        continue;
    }

    foreach (var entry in libDlls)
    {
        // PEReader needs a seekable stream; a zip entry stream is not.
        using var ms = new MemoryStream();
        using (var es = entry.Open()) es.CopyTo(ms);
        ms.Position = 0;

        string token;
        string? problem = null;
        bool strongNameSigned;
        try
        {
            using var pe = new PEReader(ms);
            if (!pe.HasMetadata) { Console.WriteLine($"  --   {entry.FullName}  (native, skipped)"); continue; }

            var md = pe.GetMetadataReader();
            var def = md.GetAssemblyDefinition();
            byte[] pubKey = md.GetBlobBytes(def.PublicKey);

            // The StrongNameSigned CorFlag separates a REAL signature from a delay-signed image,
            // which carries the public key (so the token looks right) and an empty signature blob.
            strongNameSigned = (pe.PEHeaders.CorHeader.Flags & CorFlags.StrongNameSigned) != 0;

            if (pubKey.Length == 0) { token = "<unsigned>"; problem = "no public key"; }
            else
            {
                string hex = Convert.ToHexString(pubKey).ToLowerInvariant();
                // token = low 8 bytes of SHA-1(publicKey), reversed
                token = Convert.ToHexString(SHA1.HashData(pubKey)[^8..].Reverse().ToArray()).ToLowerInvariant();
                if (hex != ExpectedPublicKey) problem = "public key is not NumSharp's";
                else if (token != ExpectedToken) problem = $"token {token} != {ExpectedToken}";
                else if (!strongNameSigned) problem = "delay-signed (public key present, signature blank)";
            }
        }
        catch (Exception ex) { token = "<error>"; problem = ex.Message; }

        checkedCount++;
        if (problem is null) Console.WriteLine($"  OK   {entry.FullName}  {token}");
        else { Console.Error.WriteLine($"  ::error::FAIL {entry.FullName}  {token}  — {problem}"); failures++; }
    }
}

Console.WriteLine($"\nassemblies checked: {checkedCount}   packages: {packages.Count}   failures: {failures}");
if (failures > 0)
{
    Console.Error.WriteLine("::error::verify_strong_name: NumSharp would ship unsigned or wrongly-signed " +
                            "assemblies. Check SignAssembly in Directory.Build.props and that Open.snk was " +
                            "checked out (it is committed and marked `binary` in .gitattributes).");
    return 1;
}
Console.WriteLine("verify_strong_name: OK");
return 0;
