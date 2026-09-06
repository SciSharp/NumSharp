using System.Runtime.CompilerServices;

// The analyzer test project reaches the internal symbol-side helpers (ScopeInheritance, KnownTypes)
// for the weaver-vs-analyzer PARITY gate (WeaverAnalyzerParityTests): both layers must classify a
// method's scoping identically, and the only way to assert that directly is to run both resolvers
// over the same fixture. This assembly is deliberately unsigned (a compiler-host component, see the
// csproj), so the grant is keyless — a simple-name friend, identity rather than access control.
[assembly: InternalsVisibleTo("NumSharp.Tests.Build.Analyzer")]
