# NumSharp.Tests.Interop.OnnxRuntime.Android

The ONNX Runtime bridge's execution providers end to end on an Android device or emulator. Desktop ONNX Runtime
packages carry only the CPU provider; XNNPACK (and NNAPI) exist only in the mobile packages, so this host is where an
XNNPACK session's outputs, which live in XNNPACK's own allocator (`XnnpackExecutionProvider`), are imported for real.

It is an app with no activity, only an instrumentation (`ProviderCheck.cs`), running
`../NumSharp.Tests.Interop.OnnxRuntime/ProviderEndToEnd.cs` (the code the desktop suite's `ProviderEndToEndTests` runs)
over `matmul_const_f32.onnx` for every provider it knows: the output must come from the provider's allocator, and
`AsNDArray`, `ToNDArray` and `session.Run(…, zeroCopyOutputs: true)` must all give X @ W exactly. It passes when no
provider fails and both the CPU provider and XNNPACK ran. Not in `NumSharp.sln`: CI machines have no Android workload.

```bash
# needs the .NET 10 SDK's android workload and an Android SDK; a device or emulator in `adb devices`
dotnet build test/NumSharp.Tests.Interop.OnnxRuntime.Android -c Release -p:TestAbis=android-arm64   # android-x64 for an emulator; never -p:RuntimeIdentifiers (it reaches the analyzer project)
export MSYS_NO_PATHCONV=1   # Git Bash: keep adb's device paths
adb -s <serial> install -r test/NumSharp.Tests.Interop.OnnxRuntime.Android/bin/Release/net10.0-android/dev.numsharp.tests.ortproviders-Signed.apk
adb -s <serial> shell am instrument -w dev.numsharp.tests.ortproviders/dev.numsharp.tests.ortproviders.ProviderCheck
adb -s <serial> uninstall dev.numsharp.tests.ortproviders
```

`-p:OnnxRuntimeVersion=1.16.0` runs it on the package's floor (default: 1.31.1, the CI's current version). The report
is the instrumentation's result (`INSTRUMENTATION_RESULT: report=…`, code -1 on success) and logcat tag `NumSharpOrt`.
