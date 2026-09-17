using System;
using System.Linq;
using System.Text;
using System.Threading;
using System.Runtime.InteropServices;
using NumSharp.Examples.FallingSand.Simulation;

namespace NumSharp.Examples.FallingSand.Player
{
    /// <summary>
    /// A terminal front-end that makes the falling-sand game playable without Unity. It drives the very
    /// same NumSharp <see cref="FallingSandWorld"/> the Unity view runs and draws the grid live in the
    /// console using 24-bit ANSI colour and half-block characters (two grid cells per character cell, so
    /// the picture is twice as tall as the text). Two modes:
    /// <list type="bullet">
    /// <item><description><b>interactive</b> (a real terminal): move a cursor, pick a material, paint, place faucets;</description></item>
    /// <item><description><b>demo</b> (<c>--demo</c>, or when input is redirected): a self-driving showcase that pours sand, water, oil and smoke and prints labelled snapshots.</description></item>
    /// </list>
    /// The physics is identical to the Unity game and is separately machine-verified; this file is only
    /// input and rendering.
    /// </summary>
    public static class Program
    {
        // Terminal-friendly resolution: wide enough to read, and an even height so half-blocks pair cleanly.
        private const int W = 90;
        private const int H = 50;

        // Material → RGB, taken straight from the shared Cell colour table so the terminal matches Unity.
        private static readonly (byte r, byte g, byte b)[] Lut = BuildLut();

        // When set (via --ascii), the demo renders plain characters instead of colour blocks, for
        // terminals (or captured logs) that don't render ANSI. Glyph per material id.
        private static bool UseAscii;
        private static readonly char[] Glyph = { ' ', '#', '.', '~', 'o', '"' };   // Empty Wall Sand Water Oil Smoke

        // ANSI control strings used by the renderer.
        private const string Home = "\x1b[H";        // move cursor to top-left (redraw in place)
        private const string ClearScreen = "\x1b[2J";
        private const string HideCursor = "\x1b[?25l";
        private const string ShowCursor = "\x1b[?25h";
        private const string Reset = "\x1b[0m";

        /// <summary>Entry point: enables ANSI, then runs the demo or the interactive game.</summary>
        /// <param name="args"><c>--demo</c> forces the showcase; <c>--play</c> forces interactive; default auto-detects a real terminal.</param>
        public static void Main(string[] args)
        {
            EnableAnsi();

            // Auto-detect: a redirected stdin (e.g. run from a script) can't read keys, so fall back to the
            // self-driving demo. An explicit flag overrides.
            UseAscii = args.Contains("--ascii");
            bool wantDemo = args.Contains("--demo") || UseAscii;
            bool wantPlay = args.Contains("--play");
            bool interactive = wantPlay || (!wantDemo && !Console.IsInputRedirected);

            if (interactive) RunInteractive();
            else RunDemo();
        }

        /// <summary>Builds the material→colour lookup (0..1 floats → 0..255 bytes) from <see cref="Cell.Color"/>.</summary>
        private static (byte, byte, byte)[] BuildLut()
        {
            var lut = new (byte, byte, byte)[Cell.Count];
            for (int id = 0; id < Cell.Count; id++)
            {
                var c = Cell.Color[id];
                lut[id] = ((byte)(c.r * 255), (byte)(c.g * 255), (byte)(c.b * 255));
            }
            return lut;
        }

        /// <summary>
        /// Renders one grid snapshot to a colour string using the upper-half-block glyph '▀': the glyph's
        /// FOREGROUND colour is the upper cell and its BACKGROUND is the lower cell, so each text row shows
        /// two grid rows. An optional cursor cell is drawn white.
        /// </summary>
        /// <param name="snap">Flat C-order grid (<c>Height*Width</c>).</param>
        /// <param name="curRow">Cursor row to highlight, or −1 for none.</param>
        /// <param name="curCol">Cursor column to highlight, or −1 for none.</param>
        /// <returns>A multi-line ANSI string (no trailing newline after the last row).</returns>
        private static string RenderColor(int[] snap, int curRow, int curCol)
        {
            var sb = new StringBuilder(H * W * 24);
            for (int t = 0; t < (H + 1) / 2; t++)
            {
                int top = 2 * t, bot = 2 * t + 1;
                for (int c = 0; c < W; c++)
                {
                    var (tr, tg, tb) = CellColor(snap, top, c, curRow, curCol);
                    var (br, bg, bb) = bot < H ? CellColor(snap, bot, c, curRow, curCol) : ((byte)5, (byte)5, (byte)8);
                    // One SGR sets both fg (upper cell) and bg (lower cell); '▀' fills the upper half.
                    sb.Append("\x1b[38;2;").Append(tr).Append(';').Append(tg).Append(';').Append(tb)
                      .Append(";48;2;").Append(br).Append(';').Append(bg).Append(';').Append(bb).Append('m').Append('▀');
                }
                sb.Append(Reset).Append('\n');
            }
            return sb.ToString();
        }

        /// <summary>Colour of one cell, or white when it is the cursor cell.</summary>
        private static (byte, byte, byte) CellColor(int[] snap, int row, int col, int curRow, int curCol)
        {
            if (row == curRow && col == curCol) return (255, 255, 255);
            int id = snap[row * W + col];
            return (uint)id < (uint)Lut.Length ? Lut[id] : Lut[0];
        }

        /// <summary>Renders a snapshot as plain characters (one per cell, full height), for non-ANSI terminals or captured logs.</summary>
        /// <param name="snap">Flat C-order grid.</param>
        /// <returns>A multi-line ASCII string.</returns>
        private static string RenderAscii(int[] snap)
        {
            var sb = new StringBuilder((W + 1) * H);
            for (int r = 0; r < H; r++)
            {
                for (int c = 0; c < W; c++)
                {
                    int id = snap[r * W + c];
                    sb.Append((uint)id < (uint)Glyph.Length ? Glyph[id] : '?');
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        /// <summary>The self-driving showcase: pours each material in turn and prints labelled snapshots so the physics is visible even without an interactive terminal.</summary>
        private static void RunDemo()
        {
            var world = new FallingSandWorld(H, W, seed: 1, withBoundary: true);
            int cx = W / 2;
            Console.WriteLine("NumSharp Falling-Sand — self-driving demo (run with --play to control it yourself)\n");

            void Advance(int frames) { for (int i = 0; i < frames; i++) world.Step(2); }
            void Snap(string label)
            {
                Console.WriteLine($"== {label} ==");
                Console.Write(UseAscii ? RenderAscii(world.Grid.Snapshot()) : RenderColor(world.Grid.Snapshot(), -1, -1));
                Console.WriteLine(Reset);
            }

            world.AddEmitter(3, cx, Cell.Sand);
            Advance(45); Snap("sand pours from the centre and piles into a cone (~45°)");

            world.AddEmitter(3, 12, Cell.Water);
            Advance(60); Snap("water pours on the left, flows around the pile and levels flat");

            world.AddEmitter(3, W - 12, Cell.Oil);
            Advance(70); Snap("oil is added on the right — lighter than water, it floats on top");

            world.ClearEmitters();
            world.BrushMaterial = Cell.Smoke; world.BrushRadius = 5;
            world.Paint(H - 6, cx);
            Advance(45); Snap("a puff of smoke released near the floor — lighter than air, it rises");

            // Final scene: wipe to walls, fill the basin with a uniform RANDOM mix of all four movable
            // materials, and let it settle. This is the clean proof of density stratification — the mix
            // separates into horizontal layers, smoke over oil over water over sand.
            world.Clear(keepWalls: true);
            var rng = new Random(7);
            int[] mix = { Cell.Water, Cell.Water, Cell.Oil, Cell.Sand, Cell.Smoke };
            for (int r = 3; r < H - 3; r++)
                for (int c = 3; c < W - 3; c++)
                    world.Grid.SetCell(r, c, mix[rng.Next(mix.Length)]);
            Snap("a random mix of sand, water, oil and smoke, before settling");
            Advance(400);
            Snap("SETTLED into density layers: smoke (top) · oil · water · sand (bottom)");
            Console.WriteLine("Layers by mean height (smaller = higher):");
            Console.WriteLine(HeightReport(world));
        }

        /// <summary>A one-line readout of each material's mean row, evidence the density stratification is real.</summary>
        private static string HeightReport(FallingSandWorld world)
        {
            var snap = world.Grid.Snapshot();
            string One(int mat)
            {
                long s = 0; int n = 0;
                for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) if (snap[r * W + c] == mat) { s += r; n++; }
                return n == 0 ? $"{Cell.Name[mat]}:-" : $"{Cell.Name[mat]}:{(double)s / n:F1}";
            }
            return "  " + One(Cell.Smoke) + "  " + One(Cell.Oil) + "  " + One(Cell.Water) + "  " + One(Cell.Sand);
        }

        /// <summary>The interactive game loop: read keys, paint, step, and redraw ~30 times a second.</summary>
        private static void RunInteractive()
        {
            var world = new FallingSandWorld(H, W, seed: 1, withBoundary: true);
            world.BrushMaterial = Cell.Sand;
            int brush = 2; world.BrushRadius = brush;
            int curR = H / 2, curC = W / 2;
            bool pen = false, paused = false;
            int subSteps = 2;

            Console.CancelKeyPress += (_, e) => Console.Write(ShowCursor + Reset);   // restore terminal on Ctrl+C
            Console.Write(HideCursor + ClearScreen);
            try
            {
                while (true)
                {
                    // Drain all pending key presses this frame so input stays responsive.
                    while (Console.KeyAvailable)
                    {
                        var key = Console.ReadKey(true);
                        if (key.Key == ConsoleKey.Q || key.Key == ConsoleKey.Escape) return;
                        HandleKey(key, world, ref curR, ref curC, ref pen, ref paused, ref brush, ref subSteps);
                    }

                    if (pen) world.Paint(curR, curC);      // pen-down paints continuously as the cursor moves
                    if (!paused) world.Step(subSteps);

                    var frame = RenderColor(world.Grid.Snapshot(), curR, curC);
                    Console.Write(Home + frame + StatusLine(world, brush, subSteps, pen, paused));
                    Thread.Sleep(33);
                }
            }
            finally
            {
                Console.Write(ShowCursor + Reset + "\n");
            }
        }

        /// <summary>Applies one key press to the game state.</summary>
        private static void HandleKey(ConsoleKeyInfo key, FallingSandWorld world,
            ref int curR, ref int curC, ref bool pen, ref bool paused, ref int brush, ref int subSteps)
        {
            switch (key.Key)
            {
                case ConsoleKey.W: case ConsoleKey.UpArrow: curR = Math.Max(0, curR - 2); break;
                case ConsoleKey.S: case ConsoleKey.DownArrow: curR = Math.Min(H - 1, curR + 2); break;
                case ConsoleKey.A: case ConsoleKey.LeftArrow: curC = Math.Max(0, curC - 2); break;
                case ConsoleKey.D: case ConsoleKey.RightArrow: curC = Math.Min(W - 1, curC + 2); break;
                case ConsoleKey.Spacebar: pen = !pen; break;                 // toggle pen-down
                case ConsoleKey.D1: world.BrushMaterial = Cell.Sand; break;
                case ConsoleKey.D2: world.BrushMaterial = Cell.Water; break;
                case ConsoleKey.D3: world.BrushMaterial = Cell.Oil; break;
                case ConsoleKey.D4: world.BrushMaterial = Cell.Smoke; break;
                case ConsoleKey.D5: world.BrushMaterial = Cell.Wall; break;
                case ConsoleKey.D6: world.BrushMaterial = Cell.Empty; break;  // eraser
                case ConsoleKey.E: world.AddEmitter(curR, curC, world.BrushMaterial); break;
                case ConsoleKey.F: world.ClearEmitters(); break;
                case ConsoleKey.P: paused = !paused; break;
                case ConsoleKey.C: world.Clear(keepWalls: true); break;
                case ConsoleKey.X: world.Clear(keepWalls: false); break;
                case ConsoleKey.Oem4: brush = Math.Max(0, brush - 1); world.BrushRadius = brush; break;   // [
                case ConsoleKey.Oem6: brush = Math.Min(12, brush + 1); world.BrushRadius = brush; break;  // ]
                case ConsoleKey.OemComma: subSteps = Math.Max(1, subSteps - 1); break;
                case ConsoleKey.OemPeriod: subSteps = Math.Min(8, subSteps + 1); break;
            }
        }

        /// <summary>The one-line status/help bar drawn beneath the grid.</summary>
        private static string StatusLine(FallingSandWorld world, int brush, int subSteps, bool pen, bool paused)
        {
            string mat = Cell.Name[world.BrushMaterial];
            string state = paused ? " [PAUSED]" : (pen ? " [PEN DOWN]" : "");
            return Reset +
                $"material:{mat}  brush:{brush}  substeps:{subSteps}  faucets:{world.EmitterCount}{state}\n" +
                "WASD/arrows move · Space pen · 1-6 material · E faucet · F clear-faucets · [ ] brush · P pause · C clear · X wipe · Q quit   ";
        }

        // --- Windows ANSI enablement (no-op elsewhere) ---
        [DllImport("kernel32.dll", SetLastError = true)] private static extern IntPtr GetStdHandle(int nStdHandle);
        [DllImport("kernel32.dll")] private static extern bool GetConsoleMode(IntPtr handle, out uint mode);
        [DllImport("kernel32.dll")] private static extern bool SetConsoleMode(IntPtr handle, uint mode);

        /// <summary>
        /// Turns on virtual-terminal (ANSI) processing so escape codes render on Windows consoles.
        /// Best-effort: harmless if it fails (Windows Terminal, VS Code and Unix terminals handle ANSI
        /// natively) and skipped entirely off Windows.
        /// </summary>
        private static void EnableAnsi()
        {
            try
            {
                Console.OutputEncoding = Encoding.UTF8;   // the '▀' block needs UTF-8 output
                if (!OperatingSystem.IsWindows()) return;
                const int STD_OUTPUT_HANDLE = -11;
                const uint ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;
                var h = GetStdHandle(STD_OUTPUT_HANDLE);
                if (GetConsoleMode(h, out uint mode))
                    SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            }
            catch { /* rendering still works on any ANSI-aware terminal */ }
        }
    }
}
