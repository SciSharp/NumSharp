using System.Text;
using Android.App;
using Android.OS;
using Android.Runtime;
using Android.Util;
using Microsoft.ML.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime.Android;

/// <summary>
///     Runs <see cref="ProviderEndToEnd"/> for every provider it knows on the device and finishes with the report:
///     <c>adb shell am instrument -w dev.numsharp.tests.ortproviders/dev.numsharp.tests.ortproviders.ProviderCheck</c>.
/// </summary>
/// <remarks>
///     Each provider's line is <c>ok: …</c>, <c>FAIL: …</c>, <c>absent: …</c> (the loaded ONNX Runtime does not carry
///     it) or <c>unavailable: …</c> (a GPU provider the build carries but the device cannot run). CPU-side providers must
///     import their outputs end to end; GPU providers (<see cref="ProviderEndToEnd.DeviceMemoryProviders"/>) must have
///     their device-bound output refused. The result code is OK (-1) when nothing failed and at least the CPU provider and
///     XNNPACK ran, CANCELED (0) otherwise. Every line also goes to logcat (tag NumSharpOrt). An Instrumentation, not an Activity, so nothing is
///     shown on the device.
/// </remarks>
[Instrumentation(Name = "dev.numsharp.tests.ortproviders.ProviderCheck", Label = "NumSharp ORT providers", TargetPackage = "dev.numsharp.tests.ortproviders")]
public sealed class ProviderCheck : Instrumentation
{
    /// <summary> The constructor Android's runtime calls through JNI when it instantiates the instrumentation. </summary>
    /// <param name="handle"> The Java object's handle. </param>
    /// <param name="transfer"> Who owns the handle. </param>
    public ProviderCheck(IntPtr handle, JniHandleOwnership transfer) : base(handle, transfer) { }

    /// <summary> Starts the instrumentation thread, which calls <see cref="OnStart"/>. </summary>
    /// <param name="arguments"> The <c>-e</c> arguments (none are used). </param>
    public override void OnCreate(Bundle arguments)
    {
        base.OnCreate(arguments);
        Start();
    }

    /// <summary> Runs every provider in turn off the main thread, holding a partial wake lock, and finishes with the
    /// report as the result bundle's <c>report</c>. </summary>
    /// <remarks> A provider's exception is caught and reported as its FAIL line, so one provider cannot hide the others;
    /// the wake lock is released in every case. </remarks>
    public override void OnStart()
    {
        base.OnStart();
        var power = (PowerManager)TargetContext.GetSystemService(global::Android.Content.Context.PowerService)!;
        var awake = power.NewWakeLock(WakeLockFlags.Partial, "NumSharpOrt:providers")!;
        awake.Acquire(10 * 60 * 1000);
        var report = new StringBuilder();
        bool failed = false;
        var ran = new HashSet<string>();
        try
        {
            Line(report, $"device: {Build.Manufacturer} {Build.Model}, Android {Build.VERSION.Release} (API {(int)Build.VERSION.SdkInt}); ONNX Runtime {OrtEnv.Instance().GetVersionString()}; providers {string.Join(", ", OrtEnv.Instance().GetAvailableProviders())}");
            byte[] model;
            using (var asset = TargetContext.Assets!.Open("matmul_const_f32.onnx"))
            using (var buffer = new MemoryStream())
            {
                asset.CopyTo(buffer);
                model = buffer.ToArray();
            }
            var available = ProviderEndToEnd.Available();
            foreach (var (provider, _, _) in ProviderEndToEnd.Known)
            {
                if (!available.Contains(provider))
                {
                    Line(report, $"absent: {provider} (not in this ONNX Runtime build)");
                    continue;
                }
                try
                {
                    Line(report, "ok: " + ProviderEndToEnd.Run(model, provider));
                    ran.Add(provider);
                }
                catch (Exception e)
                {
                    failed = true;
                    Line(report, $"FAIL: {provider}: {e.GetType().Name}: {e.Message}");
                }
            }
            var all = OrtEnv.Instance().GetAvailableProviders();
            foreach (var (provider, _, _, _) in ProviderEndToEnd.DeviceMemoryProviders)
            {
                if (!all.Contains(provider))
                {
                    Line(report, $"absent: {provider} (not in this ONNX Runtime build)");
                    continue;
                }
                try
                {
                    Line(report, "ok: " + ProviderEndToEnd.RunRefused(model, provider));
                }
                catch (OnnxRuntimeException e)
                {
                    // The build lists the provider but this device cannot run it (no suitable GPU adapter, say):
                    // nothing reached the bridge, so it is reported, not failed.
                    Line(report, $"unavailable: {provider} on this device: {e.Message.Split('\n')[0]}");
                }
                catch (Exception e)
                {
                    failed = true;
                    Line(report, $"FAIL: {provider}: {e.GetType().Name}: {e.Message}");
                }
            }
            bool covered = ran.Contains("CPUExecutionProvider") && ran.Contains("XnnpackExecutionProvider");
            Line(report, failed ? "result: FAIL" : covered ? "result: PASS" : "result: INCOMPLETE (CPU and XNNPACK must both run)");
            failed |= !covered;
        }
        finally
        {
            if (awake.IsHeld) awake.Release();
        }
        var results = new Bundle();
        results.PutString("report", report.ToString());
        Finish(failed ? Result.Canceled : Result.Ok, results);
    }

    /// <summary> Appends one report line and writes it to logcat. </summary>
    /// <param name="report"> The report. </param>
    /// <param name="line"> The line. </param>
    private static void Line(StringBuilder report, string line)
    {
        report.AppendLine(line);
        Log.Info("NumSharpOrt", line);
    }
}
