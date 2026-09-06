; Unshipped analyzer release
; https://github.com/dotnet/roslyn-analyzers/blob/main/src/Microsoft.CodeAnalysis.Analyzers/ReleaseTrackingAnalyzers.Help.md

### New Rules

Rule ID | Category | Severity | Notes
--------|----------|----------|-------
NDW002 | NumSharp.Build | Error | Scoped method has a hidden 'ref' NDArray egress
NDW003 | NumSharp.Build | Error | Scoped method returns an unsupported carrier
NDW005 | NumSharp.Build | Error | Scoped method has no body
NDW006 | NumSharp.Build | Error | Scoped attribute on a setter-only property
NDW009 | NumSharp.Build | Error | async / Task-returning method marked [NDScoped]
NDW010 | NumSharp.Build | Error | synchronous method / iterator marked [NDScopedAsync]
NDW011 | NumSharp.Build | Error | method carries both [NDScoped] and [NDScopedAsync]
NDW012 | NumSharp.Build | Warning | NDArray is created but never disposed, returned, or scoped (leaked to the finalizer)
NDW013 | NumSharp.Build | Warning | Scope attribute or [NDScopedExit] present, but the weaver is not installed
NDW015 | NumSharp.Build | Error | Scoped method has an unsupported 'out' NDArray-carrying parameter
NDW016 | NumSharp.Build | Warning | Type stores NDArrays but is not disposable
NDW017 | NumSharp.Build | Warning | NDArray-holding member is never disposed
