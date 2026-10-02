using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Opts the test host process out of Windows' execution-speed power throttling ("EcoQoS") the moment the test
    ///     assembly loads, so a local run is scheduled like the foreground work it is instead of like a background task.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why.</b> VSTest's <c>testhost</c> is a windowless child of <c>dotnet test</c>. On a hybrid CPU (Intel
    ///     12th-gen and later: performance + efficiency cores) Windows 11 treats such a process as background work and
    ///     lets the scheduler run it on the efficiency cores at reduced clocks. Measured 2026-09-24 on an i9-13900K
    ///     (net10.0, CI filter): NumSharp.Tests 26.6 s pinned to the P-cores vs ~47 s unpinned, the Oracle suite 12 s vs
    ///     22 s — the placement alone cost ~45 % of every local run and made run-to-run timings swing with whatever else
    ///     the desktop was doing. Explicitly DISABLING execution-speed throttling
    ///     (<c>PROCESS_POWER_THROTTLING_EXECUTION_SPEED</c> in the control mask, clear in the state mask) is the
    ///     documented way for a process to declare it is not background work; it is what a foreground application gets
    ///     implicitly.</para>
    ///     <para><b>Scope and safety.</b> Windows only (a no-op elsewhere, including CI's Linux/macOS legs); a failed
    ///     call (an older Windows without the information class) is ignored — scheduling is an optimization, never a
    ///     correctness input. It changes nothing the tests observe: no priority class, no affinity, no timer
    ///     resolution; the process may still use every core. Linked into every test project whose host runs long enough
    ///     to matter (see their csproj files).</para>
    /// </remarks>
    internal static class TestHostQualityOfService
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

        /// <summary>
        ///     Runs when the test assembly is loaded (before any test): disables execution-speed throttling for this
        ///     process on Windows.
        /// </summary>
        /// <remarks>
        ///     A module initializer rather than <c>[AssemblyInitialize]</c> so it needs no MSTest hook and runs even
        ///     for a filtered run; every exception is swallowed because a host that cannot be re-scheduled must still run
        ///     its tests exactly as before.
        /// </remarks>
        [ModuleInitializer]
        internal static void OptOutOfEfficiencyMode()
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
                // Entry point missing (pre-1709 Windows) or marshalling refused: keep the default scheduling.
            }
        }
    }
}
