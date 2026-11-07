using System.Runtime.InteropServices;

namespace NumSharp.Build;

/// <summary>
///     Opts the weaver process out of Windows' execution-speed power throttling ("EcoQoS") before it does any work, so a
///     build schedules it like the foreground step it is instead of like a background task.
/// </summary>
/// <remarks>
///     <para><b>Why.</b> MSBuild starts the weaver as a windowless child process. On a hybrid CPU (Intel 12th generation
///     and later: performance + efficiency cores) Windows 11 treats such a process as background work and lets the
///     scheduler keep it on the efficiency cores at reduced clocks. Measured on an i9-13900K: weaving NumSharp itself
///     took 1.65 s pinned to the performance cores and 3.4-3.8 s as a build step - the placement alone doubled the
///     weave. Explicitly DISABLING execution-speed throttling (<c>PROCESS_POWER_THROTTLING_EXECUTION_SPEED</c> in the
///     control mask, clear in the state mask) is the documented way for a process to declare it is not background work;
///     a foreground application gets it implicitly. The test hosts do the same (<c>TestHostQualityOfService</c>).</para>
///     <para><b>Scope and safety.</b> Windows only (a no-op elsewhere); a failed call (a Windows without the
///     information class) is ignored - scheduling is an optimization, never a correctness input. It changes nothing
///     the weave produces: no priority class, no affinity, no timer resolution. The in-process callers of
///     <see cref="ScopeWeaver.WeaveAssembly"/> (the test harness, the benchmarks) never run it: only
///     <see cref="Program"/> does, as a separate process.</para>
/// </remarks>
internal static class ToolQualityOfService
{
    /// <summary><c>PROCESS_INFORMATION_CLASS.ProcessPowerThrottling</c>.</summary>
    private const int ProcessPowerThrottling = 4;

    /// <summary><c>PROCESS_POWER_THROTTLING_CURRENT_VERSION</c>.</summary>
    private const uint PowerThrottlingCurrentVersion = 1;

    /// <summary><c>PROCESS_POWER_THROTTLING_EXECUTION_SPEED</c>: the EcoQoS policy bit.</summary>
    private const uint PowerThrottlingExecutionSpeed = 0x1;

    /// <summary>The native <c>PROCESS_POWER_THROTTLING_STATE</c>.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct PowerThrottlingState
    {
        /// <summary>Structure version; must be <see cref="PowerThrottlingCurrentVersion"/>.</summary>
        public uint Version;

        /// <summary>Which policies this call controls (the rest stay system-managed).</summary>
        public uint ControlMask;

        /// <summary>For each controlled policy: set = throttling ON, clear = throttling OFF.</summary>
        public uint StateMask;
    }

    /// <summary>Sets a piece of process information; here, the power-throttling state.</summary>
    /// <param name="process">The process handle (the current-process pseudo handle).</param>
    /// <param name="informationClass">The information class (<see cref="ProcessPowerThrottling"/>).</param>
    /// <param name="information">The state to apply.</param>
    /// <param name="size">The size of <paramref name="information"/> in bytes.</param>
    /// <returns>Nonzero on success.</returns>
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetProcessInformation(IntPtr process, int informationClass,
                                                     ref PowerThrottlingState information, uint size);

    /// <summary>The current-process pseudo handle (never needs closing).</summary>
    /// <returns><c>(HANDLE)-1</c>.</returns>
    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentProcess();

    /// <summary>Disables execution-speed throttling for this process on Windows; does nothing elsewhere, and never throws.</summary>
    /// <remarks>
    ///     Every exception is swallowed: a weaver that cannot be re-scheduled must still weave exactly as before (a missing
    ///     entry point on an old Windows, a marshalling refusal).
    /// </remarks>
    public static void OptOutOfEfficiencyMode()
    {
        if (!OperatingSystem.IsWindows())
            return;
        try
        {
            var state = new PowerThrottlingState
            {
                Version = PowerThrottlingCurrentVersion,
                ControlMask = PowerThrottlingExecutionSpeed,
                StateMask = 0, // throttling OFF for the controlled policy
            };
            SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, ref state,
                                  (uint)Marshal.SizeOf<PowerThrottlingState>());
        }
        catch
        {
            // Entry point missing or marshalling refused: keep the default scheduling.
        }
    }
}
