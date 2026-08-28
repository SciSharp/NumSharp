using Mono.Cecil;

namespace NumSharp.Build;

/// <summary>
///     Assembly resolver for cross-module weaving: when the assembly being woven is NOT NumSharp
///     itself (a consumer project that installed the <c>NumSharp.Build</c> package), <c>NDScope</c> /
///     <c>NDArray</c> / <c>INDArrayCarrier</c> live in the REFERENCED NumSharp assembly, and
///     classification must chase base-type chains (<c>NDArray&lt;T&gt; : NDArray</c>) and interface
///     lists through it. The MSBuild target hands the compiler's own reference list
///     (<c>@(ReferencePathWithRefAssemblies)</c>, one path per line) via <c>--refs</c>, so resolution
///     sees exactly what the compile saw — reference assemblies included, which is fine: the weaver
///     imports member SIGNATURES, never bodies.
/// </summary>
/// <remarks>
///     Exact paths win over directory probing (two packages can carry same-named assemblies; the
///     compile's pick is authoritative). The woven assembly's own directory remains a probe fallback
///     for the bare self-weave invocation — NumSharp.Core's inline <c>NDScopeWeave</c> target passes
///     no <c>--refs</c> because everything it needs is in-module. Resolved references are cached and
///     disposed with the resolver; they are opened read-only, never written.
/// </remarks>
internal sealed class WeaverAssemblyResolver : DefaultAssemblyResolver
{
    private readonly Dictionary<string, string> _pathByName = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, AssemblyDefinition> _resolved = new(StringComparer.OrdinalIgnoreCase);

    public WeaverAssemblyResolver(string assemblyDir, IReadOnlyList<string> referencePaths)
    {
        if (!string.IsNullOrEmpty(assemblyDir))
            AddSearchDirectory(assemblyDir);

        foreach (var raw in referencePaths)
        {
            string full;
            try
            {
                full = Path.GetFullPath(raw.Trim());
            }
            catch
            {
                continue; // a malformed rsp line must not kill the weave; probing still works
            }

            if (!File.Exists(full))
                continue;

            // First mention wins, matching the compiler's handling of duplicate simple names.
            _pathByName.TryAdd(Path.GetFileNameWithoutExtension(full), full);
        }
    }

    public override AssemblyDefinition Resolve(AssemblyNameReference name)
    {
        if (_resolved.TryGetValue(name.Name, out var hit))
            return hit;

        if (_pathByName.TryGetValue(name.Name, out var path))
        {
            var assembly = AssemblyDefinition.ReadAssembly(path, new ReaderParameters { AssemblyResolver = this });
            _resolved[name.Name] = assembly;
            return assembly;
        }

        return base.Resolve(name); // search-dir probing; DefaultAssemblyResolver caches its own hits
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (var assembly in _resolved.Values)
                assembly.Dispose();
            _resolved.Clear();
        }

        base.Dispose(disposing);
    }
}
