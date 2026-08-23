using Mono.Cecil;

namespace NumSharp.Weaver;

/// <summary>
///     NumSharp's NDScope IL weaver — post-compile injection of the ambient-reclamation-scope
///     pattern into methods marked <c>[NDScoped]</c>, so source files keep their 100% original
///     bodies while the built assembly carries the exact IL the hand-written pattern produces:
///     <code>
///     using var scope = NDScope.Open();
///     ...original body...
///     return scope.Returns(result);     // and out-NDArray params escape before each return
///     </code>
///     Invoked by the <c>NDScopeWeave</c> target in NumSharp.Core.csproj after CoreCompile,
///     for each TargetFramework, on the intermediate assembly. Re-signs with the repo's
///     strong-name key (IL rewriting invalidates the compile-time signature) and preserves
///     the portable PDB (instructions are only inserted/mutated in place, so original
///     sequence points survive and source stepping still lands on the right lines).
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        string assemblyPath = null;
        string snkPath = null;
        var verbose = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--snk":
                    if (++i >= args.Length) return Usage("missing value for --snk");
                    snkPath = args[i];
                    break;
                case "--verbose":
                case "-v":
                    verbose = true;
                    break;
                case "--help":
                case "-h":
                    return Usage(null);
                default:
                    if (assemblyPath != null) return Usage($"unexpected argument '{args[i]}'");
                    assemblyPath = args[i];
                    break;
            }
        }

        if (assemblyPath is null)
            return Usage("no assembly path given");
        if (!File.Exists(assemblyPath))
        {
            Console.Error.WriteLine($"NumSharp.Weaver : error NDW000: assembly not found: {assemblyPath}");
            return 2;
        }

        byte[] snkBlob = null;
        if (snkPath != null)
        {
            if (!File.Exists(snkPath))
            {
                Console.Error.WriteLine($"NumSharp.Weaver : error NDW000: strong-name key not found: {snkPath}");
                return 2;
            }

            snkBlob = File.ReadAllBytes(snkPath);
        }

        try
        {
            var result = ScopeWeaver.WeaveAssembly(assemblyPath, snkBlob, verbose, Console.Out, Console.Error);
            if (result.Errors > 0)
                return 1;

            Console.Out.WriteLine(
                $"NumSharp.Weaver: {Path.GetFileName(assemblyPath)} — woven {result.Woven}, " +
                $"already-scoped {result.Skipped}{(snkBlob != null ? ", re-signed" : "")}" +
                $"{(result.SymbolsWritten ? ", symbols updated" : "")}");
            return 0;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"NumSharp.Weaver : error NDW099: unhandled failure weaving '{assemblyPath}': {e}");
            return 2;
        }
    }

    private static int Usage(string problem)
    {
        if (problem != null)
            Console.Error.WriteLine($"NumSharp.Weaver : error NDW000: {problem}");
        Console.Error.WriteLine("usage: NumSharp.Weaver <assembly.dll> [--snk <keyfile.snk>] [--verbose]");
        return problem is null ? 0 : 2;
    }
}
