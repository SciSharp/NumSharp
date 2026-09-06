using System.Runtime.CompilerServices;

// The weaver's in-process test harness (test/NumSharp.Tests.Build.Analyzer → WeaverTestHarness)
// drives ScopeWeaver.WeaveAssembly and ScopeInheritance directly on freshly compiled fixture
// assemblies — the fast, fine-grained twin of the real-build gates (WeaverInheritanceBuildTests,
// tools/verify_build_package.sh, tools/stress_weaver.sh). This assembly is deliberately unsigned
// (a build tool, see the csproj), so the grant is keyless: it names the friend by simple name
// only — identity, not access control (Directory.Build.props).
[assembly: InternalsVisibleTo("NumSharp.Tests.Build.Analyzer")]
