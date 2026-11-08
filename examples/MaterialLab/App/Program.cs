using System;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;

namespace NumSharp.Examples.MaterialLab.App
{
    /// <summary>
    /// Entry point of the NumSharp Material Lab — a real-time MLS-MPM simulator of liquids, grains and solids
    /// whose physics runs on NumSharp. Opens a window covering 2/3 of the primary monitor (F11 toggles
    /// borderless fullscreen).
    /// </summary>
    internal static class Program
    {
        /// <summary>
        /// Parses the command line and runs the lab.
        /// <list type="bullet">
        /// <item><description><c>--fullscreen</c> start fullscreen; <c>--fraction 0.66</c> window size as a fraction of the screen.</description></item>
        /// <item><description><c>--scene N</c> start on scene N (1-based, as the F-keys); <c>--quality low|medium|high|ultra</c>
        /// pins the grid resolution (without it the lab starts at High and steps down on its own if a scene cannot hold 60 FPS).</description></item>
        /// <item><description><c>--novsync</c> uncapped presentation.</description></item>
        /// <item><description><c>--screenshot file.png --frames K [--hidden] [--nohud]</c> render K frames, save, exit.</description></item>
        /// <item><description><c>--bench K</c> run K frames uncapped and print timing.</description></item>
        /// </list>
        /// </summary>
        /// <param name="args">Command-line arguments.</param>
        /// <returns>0 on success, 1 on a fatal error (also shown in a dialog and written to MaterialLab-error.log).</returns>
        [STAThread]
        public static int Main(string[] args)
        {
            try
            {
                var o = Parse(args);
                using var app = new LabApp(o);
                return app.Run();
            }
            catch (Exception e)
            {
                string msg = e.ToString();
                try { File.WriteAllText("MaterialLab-error.log", msg); } catch { /* best effort */ }
                Console.Error.WriteLine(msg);
                if (OperatingSystem.IsWindows())
                    MessageBoxW(0, e.Message + "\n\n(details in MaterialLab-error.log)", "NumSharp Material Lab", 0x10);
                return 1;
            }
        }

        /// <summary>Parses the options (unknown flags are ignored).</summary>
        /// <param name="args">Arguments.</param>
        /// <returns>The options.</returns>
        /// <exception cref="ArgumentException">A value is malformed (e.g. a non-numeric frame count).</exception>
        private static LabOptions Parse(string[] args)
        {
            var o = new LabOptions();
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{a} needs a value");
                switch (a)
                {
                    case "--fullscreen": o.Fullscreen = true; break;
                    case "--fraction": o.ScreenFraction = Math.Clamp(double.Parse(Next(), CultureInfo.InvariantCulture), 0.2, 1.0); break;
                    case "--scene": o.Scene = int.Parse(Next(), CultureInfo.InvariantCulture) - 1; break;
                    case "--quality":
                        o.GridRows = Next().ToLowerInvariant() switch { "low" => 72, "medium" => 90, "ultra" => 126, _ => 108 };
                        o.AutoQuality = false;   // an explicit choice is never overridden
                        break;
                    case "--novsync": o.VSync = false; break;
                    case "--screenshot": o.ScreenshotPath = Next(); if (o.Frames == 0) o.Frames = 120; break;
                    case "--frames": o.Frames = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--hidden": o.Hidden = true; break;
                    case "--nohud": o.NoHud = true; break;
                    case "--bench": o.Bench = true; o.Frames = int.Parse(Next(), CultureInfo.InvariantCulture); o.VSync = false; break;
                }
            }
            return o;
        }

        /// <summary>Shows a modal error dialog — a windowed app has no console for the user to read a stack trace in.</summary>
        /// <param name="hwnd">Owner window (0 = none).</param>
        /// <param name="text">Message.</param>
        /// <param name="caption">Title.</param>
        /// <param name="type">MB_* flags (0x10 = error icon).</param>
        /// <returns>The button pressed.</returns>
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern int MessageBoxW(nint hwnd, string text, string caption, uint type);
    }
}
