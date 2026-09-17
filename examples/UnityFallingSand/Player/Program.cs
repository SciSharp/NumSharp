using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading;
using System.Runtime.InteropServices;
using NumSharp.Examples.FallingSand.Simulation;

namespace NumSharp.Examples.FallingSand.Player
{
    /// <summary>
    /// A Unity-free front-end that makes the falling-sand game playable and viewable at real resolution.
    /// It drives the very same NumSharp <see cref="FallingSandWorld"/> the Unity view runs, and offers
    /// several ways to see it:
    /// <list type="bullet">
    /// <item><description><b>--png [file]</b>: render the sim at NATIVE resolution (default 1000×500) to real PNG images — the true high-pixel output;</description></item>
    /// <item><description><b>--play</b> (or a real terminal): interactive — move a cursor, pick a material, paint, place faucets;</description></item>
    /// <item><description><b>--demo</b> / <b>--ascii</b>: a self-driving terminal showcase (colour half-blocks, or plain text);</description></item>
    /// <item><description><b>--bench</b>: time the raw step cost at the chosen resolution.</description></item>
    /// </list>
    /// A terminal can't show 1000×500 as text, so the terminal modes DOWNSCALE the grid to a readable
    /// preview; only <c>--png</c> (and the Unity game) render at full resolution.
    /// </summary>
    public static class Program
    {
        // Simulation resolution in cells (W wide × H tall). Set per mode from --width/--height; the
        // high-resolution modes (--png, --bench) default to 1000×500, terminal modes to a readable size.
        private static int W = 1000;
        private static int H = 500;

        // Material → RGB, from the shared Cell colour table so every renderer matches Unity.
        private static readonly (byte r, byte g, byte b)[] Lut = BuildLut();

        // Glyphs for --ascii (Empty Wall Sand Water Oil Smoke).
        private static readonly char[] Glyph = { ' ', '#', '.', '~', 'o', '"' };
        private static bool UseAscii;

        // Terminal preview budget (characters). Half-blocks pack two grid rows per text row.
        private const int MaxCols = 180;
        private const int MaxTextRows = 55;

        private const string Home = "\x1b[H";
        private const string ClearScreen = "\x1b[2J";
        private const string HideCursor = "\x1b[?25l";
        private const string ShowCursor = "\x1b[?25h";
        private const string Reset = "\x1b[0m";

        /// <summary>Entry point: parses options, picks a resolution appropriate to the mode, and runs it.</summary>
        /// <param name="args">See the class summary for the flags (<c>--png</c>, <c>--play</c>, <c>--demo</c>, <c>--ascii</c>, <c>--bench</c>, <c>--width</c>, <c>--height</c>, <c>--scale</c>).</param>
        public static void Main(string[] args)
        {
            EnableAnsi();
            int argW = ArgInt(args, "--width", -1);
            int argH = ArgInt(args, "--height", -1);

            if (args.Contains("--bench"))
            {
                W = argW > 0 ? argW : 1000; H = argH > 0 ? argH : 500;
                RunBench();
                return;
            }
            if (args.Contains("--png"))
            {
                W = argW > 0 ? argW : 1000; H = argH > 0 ? argH : 500;   // native high-res image
                RunPng(ArgStr(args, "--png", "sand.png"), Math.Max(1, ArgInt(args, "--scale", 1)));
                return;
            }

            // Terminal modes: default to a readable resolution; a bigger --width/--height still works and
            // is shown via a downscaled preview.
            W = argW > 0 ? argW : 200; H = argH > 0 ? argH : 110;
            UseAscii = args.Contains("--ascii");
            bool wantDemo = args.Contains("--demo") || UseAscii;
            bool wantPlay = args.Contains("--play");
            // Only go interactive on a genuine two-way terminal; under a redirected/automated shell fall
            // back to the demo so the process can never block forever waiting for keystrokes.
            bool interactive = wantPlay || (!wantDemo && !Console.IsInputRedirected && !Console.IsOutputRedirected);

            if (interactive) RunInteractive();
            else RunDemo();
        }

        /// <summary>Reads an integer option like <c>--width 1000</c>, or returns <paramref name="def"/>.</summary>
        private static int ArgInt(string[] a, string key, int def)
        {
            int i = Array.IndexOf(a, key);
            return (i >= 0 && i + 1 < a.Length && int.TryParse(a[i + 1], out int v)) ? v : def;
        }

        /// <summary>Reads a string option like <c>--png out.png</c>, or returns <paramref name="def"/>.</summary>
        private static string ArgStr(string[] a, string key, string def)
        {
            int i = Array.IndexOf(a, key);
            return (i >= 0 && i + 1 < a.Length && !a[i + 1].StartsWith("--")) ? a[i + 1] : def;
        }

        /// <summary>Times the raw per-step cost at the current resolution (no rendering).</summary>
        private static void RunBench()
        {
            var world = new FallingSandWorld(H, W, seed: 1, withBoundary: true);
            SeedRandomMix(world);
            world.Step(1);   // warm the JIT / kernel cache
            var sw = Stopwatch.StartNew();
            const int n = 40;
            for (int i = 0; i < n; i++) world.Step(1);
            sw.Stop();
            Console.WriteLine($"{W}x{H} = {(long)W * H:N0} cells : {sw.Elapsed.TotalMilliseconds / n:F1} ms/step  ({n} steps in {sw.ElapsedMilliseconds} ms)");
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

        // ---------------- shared scene timeline ----------------

        /// <summary>
        /// The self-driving scene both the terminal demo and the PNG renderer play: pour sand, then water,
        /// then oil, let smoke rise, then settle a random mix into density layers. <paramref name="onSnap"/>
        /// is invoked at each milestone so a caller can render it however it likes. Frame counts scale with
        /// the grid height so the timing looks the same at any resolution.
        /// </summary>
        /// <param name="world">The world to drive.</param>
        /// <param name="onSnap">Called with a label at each milestone worth capturing.</param>
        private static void RunTimeline(FallingSandWorld world, Action<string> onSnap)
        {
            int cx = W / 2;
            int u = Math.Max(1, H / 55);   // time scale: base counts are tuned for a ~55-tall grid
            void Adv(int frames) { for (int i = 0; i < frames; i++) world.Step(2); }

            world.AddEmitter(3, cx, Cell.Sand);
            world.AddEmitter(3, Math.Max(3, W / 6), Cell.Water);
            world.AddEmitter(3, W - Math.Max(4, W / 6), Cell.Oil);
            Adv(24 * u);
            onSnap("pouring sand (centre), water (left) and oil (right)");

            world.ClearEmitters();
            world.BrushMaterial = Cell.Smoke; world.BrushRadius = Math.Max(3, W / 40);
            world.Paint(H - Math.Max(4, H / 10), cx);
            Adv(16 * u);
            onSnap("a puff of smoke rises through the liquids");

            // Wipe to walls and fill the basin with a uniform random mix, then let it settle: the clean
            // proof of density stratification — smoke over oil over water over sand.
            world.Clear(keepWalls: true);
            world.Grid.Blit(3, 3, RandomMix(H - 6, W - 6, seed: 7));
            onSnap("a random mix of all four materials, before settling");
            Adv(Math.Max(120, H));   // enough steps for material to fall the full height and level
            onSnap("SETTLED into density layers: smoke · oil · water · sand");
        }

        /// <summary>Builds an <c>[h,w]</c> block of a uniform random mix of the four movable materials.</summary>
        private static int[,] RandomMix(int h, int w, int seed)
        {
            var rng = new Random(seed);
            int[] mix = { Cell.Water, Cell.Water, Cell.Oil, Cell.Sand, Cell.Smoke };
            var block = new int[h, w];
            for (int r = 0; r < h; r++)
                for (int c = 0; c < w; c++)
                    block[r, c] = mix[rng.Next(mix.Length)];
            return block;
        }

        /// <summary>Seeds a world with a random mix filling most of the basin (used by the benchmark).</summary>
        private static void SeedRandomMix(FallingSandWorld world)
        {
            if (H > 6 && W > 6) world.Grid.Blit(3, 3, RandomMix(H - 6, W - 6, seed: 1));
        }

        /// <summary>A one-line readout of each material's mean row — evidence the stratification is real (smaller row = higher).</summary>
        private static string HeightReport(FallingSandWorld world)
        {
            var snap = world.Grid.Snapshot();
            string One(int mat)
            {
                long s = 0; int n = 0;
                for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) if (snap[r * W + c] == mat) { s += r; n++; }
                return n == 0 ? $"{Cell.Name[mat]}:-" : $"{Cell.Name[mat]}:{(double)s / n:F1}";
            }
            return $"{One(Cell.Smoke)}  {One(Cell.Oil)}  {One(Cell.Water)}  {One(Cell.Sand)}";
        }

        // ---------------- terminal rendering ----------------

        /// <summary>The terminal showcase: plays the timeline, printing a labelled (downscaled) frame at each milestone.</summary>
        private static void RunDemo()
        {
            var world = new FallingSandWorld(H, W, seed: 1, withBoundary: true);
            Console.WriteLine($"NumSharp Falling-Sand — self-driving demo at {W}x{H} (preview downscaled to fit; run with --play to control it, or --png for a full-resolution image)\n");
            RunTimeline(world, label =>
            {
                Console.WriteLine($"== {label} ==");
                var (buf, bw, bh, _) = Preview(world.Grid.Snapshot());
                Console.Write(UseAscii ? RenderAsciiBuf(buf, bw, bh) : RenderColorBuf(buf, bw, bh, -1, -1));
                Console.WriteLine(Reset);
            });
            Console.WriteLine("Layers by mean height (smaller = higher):  " + HeightReport(world));
        }

        /// <summary>Downsamples the native <c>W×H</c> grid to a terminal-sized buffer, keeping the highest-priority (most solid) material in each block so thin features stay visible.</summary>
        /// <param name="snap">The native grid snapshot.</param>
        /// <returns>The preview buffer and its dimensions, plus the integer downscale factor.</returns>
        private static (int[] buf, int bw, int bh, int scale) Preview(int[] snap)
        {
            int scale = ScaleFor();
            int bw = (W + scale - 1) / scale, bh = (H + scale - 1) / scale;
            var buf = new int[bw * bh];
            for (int pr = 0; pr < bh; pr++)
            {
                int r1 = Math.Min(H, pr * scale + scale);
                for (int pc = 0; pc < bw; pc++)
                {
                    int c1 = Math.Min(W, pc * scale + scale);
                    int best = Cell.Empty, bestPri = -1;
                    for (int r = pr * scale; r < r1; r++)
                        for (int c = pc * scale; c < c1; c++)
                        {
                            int pri = Priority(snap[r * W + c]);
                            if (pri > bestPri) { bestPri = pri; best = snap[r * W + c]; }
                        }
                    buf[pr * bw + pc] = best;
                }
            }
            return (buf, bw, bh, scale);
        }

        /// <summary>The integer downscale factor that makes the grid fit the terminal preview budget (1 = no downscale).</summary>
        private static int ScaleFor() =>
            Math.Max(1, Math.Max((W + MaxCols - 1) / MaxCols, (H + MaxTextRows * 2 - 1) / (MaxTextRows * 2)));

        /// <summary>Rendering priority for downsampling: prefer solids/liquids over gas over empty so features pop in a preview.</summary>
        private static int Priority(int id) => id switch
        {
            Cell.Wall => 5,
            Cell.Sand => 4,
            Cell.Water => 3,
            Cell.Oil => 2,
            Cell.Smoke => 1,
            _ => 0,
        };

        /// <summary>Renders a buffer as 24-bit colour half-blocks ('▀' = upper cell fg, lower cell bg → two grid rows per text row). Cursor cell (in buffer coords) is drawn white; pass −1 for none.</summary>
        private static string RenderColorBuf(int[] buf, int bw, int bh, int curRow, int curCol)
        {
            var sb = new StringBuilder(bw * bh * 20);
            for (int t = 0; t < (bh + 1) / 2; t++)
            {
                int top = 2 * t, bot = 2 * t + 1;
                for (int c = 0; c < bw; c++)
                {
                    var (tr, tg, tb) = ColOf(buf, bw, top, c, curRow, curCol);
                    var (br, bg, bb) = bot < bh ? ColOf(buf, bw, bot, c, curRow, curCol) : ((byte)5, (byte)5, (byte)8);
                    sb.Append("\x1b[38;2;").Append(tr).Append(';').Append(tg).Append(';').Append(tb)
                      .Append(";48;2;").Append(br).Append(';').Append(bg).Append(';').Append(bb).Append('m').Append('▀');
                }
                sb.Append(Reset).Append('\n');
            }
            return sb.ToString();
        }

        /// <summary>Colour of one buffer cell, or white when it is the cursor.</summary>
        private static (byte, byte, byte) ColOf(int[] buf, int bw, int row, int col, int curRow, int curCol)
        {
            if (row == curRow && col == curCol) return (255, 255, 255);
            int id = buf[row * bw + col];
            return (uint)id < (uint)Lut.Length ? Lut[id] : Lut[0];
        }

        /// <summary>Renders a buffer as plain characters (one per cell), for non-ANSI terminals or captured logs.</summary>
        private static string RenderAsciiBuf(int[] buf, int bw, int bh)
        {
            var sb = new StringBuilder((bw + 1) * bh);
            for (int r = 0; r < bh; r++)
            {
                for (int c = 0; c < bw; c++)
                {
                    int id = buf[r * bw + c];
                    sb.Append((uint)id < (uint)Glyph.Length ? Glyph[id] : '?');
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        // ---------------- PNG rendering (real high-res images) ----------------

        /// <summary>
        /// Plays the timeline and writes a real PNG at NATIVE resolution (optionally integer-upscaled by
        /// <paramref name="scale"/>) at each milestone. This is the true high-pixel output — a 1000×500 grid
        /// becomes a 1000×500 (or larger) image file.
        /// </summary>
        /// <param name="path">Base output path; frames are written as <c>name-0.png</c>, <c>name-1.png</c>, …</param>
        /// <param name="scale">Integer pixel upscale (1 = one pixel per cell).</param>
        private static void RunPng(string path, int scale)
        {
            var world = new FallingSandWorld(H, W, seed: 1, withBoundary: true);
            Console.WriteLine($"Rendering the falling-sand sim at {W}x{H} (×{scale} → {W * scale}x{H * scale} px)…");
            int idx = 0;
            RunTimeline(world, label =>
            {
                string p = FrameName(path, idx++);
                WritePng(p, ToPixels(world.Grid.Snapshot(), scale), W * scale, H * scale);
                Console.WriteLine($"  wrote {p}  —  {label}");
            });
            Console.WriteLine("Layers by mean height (smaller = higher):  " + HeightReport(world));
        }

        /// <summary>Inserts a frame index before the file extension: <c>sand.png</c> → <c>sand-2.png</c>.</summary>
        private static string FrameName(string path, int idx)
        {
            string dir = Path.GetDirectoryName(path) ?? "";
            string stem = Path.GetFileNameWithoutExtension(path);
            string ext = Path.GetExtension(path);
            if (string.IsNullOrEmpty(ext)) ext = ".png";
            return Path.Combine(dir, $"{stem}-{idx}{ext}");
        }

        /// <summary>Converts a grid snapshot to a top-to-bottom RGB pixel buffer, each cell drawn as a <paramref name="scale"/>×<paramref name="scale"/> block.</summary>
        private static byte[] ToPixels(int[] snap, int scale)
        {
            int pw = W * scale, ph = H * scale;
            var px = new byte[pw * ph * 3];
            for (int r = 0; r < H; r++)
                for (int c = 0; c < W; c++)
                {
                    int id = snap[r * W + c];
                    var (cr, cg, cb) = (uint)id < (uint)Lut.Length ? Lut[id] : Lut[0];
                    for (int dy = 0; dy < scale; dy++)
                    {
                        int rowBase = ((r * scale + dy) * pw + c * scale) * 3;
                        for (int dx = 0; dx < scale; dx++)
                        {
                            int o = rowBase + dx * 3;
                            px[o] = cr; px[o + 1] = cg; px[o + 2] = cb;
                        }
                    }
                }
            return px;
        }

        /// <summary>Writes a minimal 8-bit RGB PNG (no external image library — keeps the sample dependency-free like NumSharp itself).</summary>
        /// <param name="path">Output file path.</param>
        /// <param name="rgb">Top-to-bottom RGB pixels, <c>width*height*3</c> bytes.</param>
        /// <param name="width">Image width in pixels.</param>
        /// <param name="height">Image height in pixels.</param>
        private static void WritePng(string path, byte[] rgb, int width, int height)
        {
            using var fs = File.Create(path);
            fs.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });   // PNG signature

            var ihdr = new byte[13];
            WriteBE(ihdr, 0, width); WriteBE(ihdr, 4, height);
            ihdr[8] = 8;    // bit depth
            ihdr[9] = 2;    // colour type 2 = truecolour RGB
            // ihdr[10..12] = 0 (compression/filter/interlace)
            WriteChunk(fs, "IHDR", ihdr);

            // Raw image data: each scanline is a filter byte (0 = none) followed by width*3 RGB bytes.
            var raw = new byte[height * (1 + width * 3)];
            int p = 0;
            for (int y = 0; y < height; y++)
            {
                raw[p++] = 0;
                Array.Copy(rgb, y * width * 3, raw, p, width * 3);
                p += width * 3;
            }
            byte[] compressed;
            using (var ms = new MemoryStream())
            {
                using (var z = new ZLibStream(ms, CompressionLevel.Optimal, leaveOpen: true)) z.Write(raw, 0, raw.Length);
                compressed = ms.ToArray();
            }
            WriteChunk(fs, "IDAT", compressed);
            WriteChunk(fs, "IEND", Array.Empty<byte>());
        }

        /// <summary>Writes one PNG chunk (length, type, data, CRC-32 over type+data).</summary>
        private static void WriteChunk(Stream s, string type, byte[] data)
        {
            var len = new byte[4]; WriteBE(len, 0, data.Length); s.Write(len);
            var t = Encoding.ASCII.GetBytes(type); s.Write(t);
            s.Write(data);
            uint crc = Crc32(t, data);
            var c = new byte[4]; WriteBE(c, 0, (int)crc); s.Write(c);
        }

        /// <summary>Writes a 32-bit big-endian integer.</summary>
        private static void WriteBE(byte[] b, int o, int v)
        {
            b[o] = (byte)(v >> 24); b[o + 1] = (byte)(v >> 16); b[o + 2] = (byte)(v >> 8); b[o + 3] = (byte)v;
        }

        private static uint[] _crcTable;
        /// <summary>Standard PNG CRC-32 over the chunk type followed by its data.</summary>
        private static uint Crc32(byte[] type, byte[] data)
        {
            if (_crcTable == null)
            {
                _crcTable = new uint[256];
                for (uint n = 0; n < 256; n++)
                {
                    uint c = n;
                    for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                    _crcTable[n] = c;
                }
            }
            uint crc = 0xFFFFFFFFu;
            foreach (byte b in type) crc = _crcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
            foreach (byte b in data) crc = _crcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
            return crc ^ 0xFFFFFFFFu;
        }

        // ---------------- interactive ----------------

        /// <summary>The interactive game loop: read keys, paint, step, redraw ~30×/s. The world is shown via the downscaled preview; painting happens at native resolution under the cursor.</summary>
        private static void RunInteractive()
        {
            var world = new FallingSandWorld(H, W, seed: 1, withBoundary: true);
            world.BrushMaterial = Cell.Sand;
            int brush = Math.Max(1, W / 60); world.BrushRadius = brush;
            int curR = H / 2, curC = W / 2;
            bool pen = false, paused = false;
            int subSteps = 2;
            int scale = ScaleFor();
            int move = Math.Max(1, scale);   // move the cursor ~one preview cell per key press

            Console.CancelKeyPress += (_, e) => Console.Write(ShowCursor + Reset);
            Console.Write(HideCursor + ClearScreen);
            try
            {
                while (true)
                {
                    while (Console.KeyAvailable)
                    {
                        var key = Console.ReadKey(true);
                        if (key.Key == ConsoleKey.Q || key.Key == ConsoleKey.Escape) return;
                        HandleKey(key, world, ref curR, ref curC, ref pen, ref paused, ref brush, ref subSteps, move);
                    }
                    if (pen) world.Paint(curR, curC);
                    if (!paused) world.Step(subSteps);

                    var (buf, bw, bh, _) = Preview(world.Grid.Snapshot());
                    Console.Write(Home + RenderColorBuf(buf, bw, bh, curR / scale, curC / scale)
                                 + StatusLine(world, brush, subSteps, pen, paused));
                    Thread.Sleep(33);
                }
            }
            finally { Console.Write(ShowCursor + Reset + "\n"); }
        }

        /// <summary>Applies one key press to the interactive game state.</summary>
        private static void HandleKey(ConsoleKeyInfo key, FallingSandWorld world,
            ref int curR, ref int curC, ref bool pen, ref bool paused, ref int brush, ref int subSteps, int move)
        {
            switch (key.Key)
            {
                case ConsoleKey.W: case ConsoleKey.UpArrow: curR = Math.Max(0, curR - move); break;
                case ConsoleKey.S: case ConsoleKey.DownArrow: curR = Math.Min(H - 1, curR + move); break;
                case ConsoleKey.A: case ConsoleKey.LeftArrow: curC = Math.Max(0, curC - move); break;
                case ConsoleKey.D: case ConsoleKey.RightArrow: curC = Math.Min(W - 1, curC + move); break;
                case ConsoleKey.Spacebar: pen = !pen; break;
                case ConsoleKey.D1: world.BrushMaterial = Cell.Sand; break;
                case ConsoleKey.D2: world.BrushMaterial = Cell.Water; break;
                case ConsoleKey.D3: world.BrushMaterial = Cell.Oil; break;
                case ConsoleKey.D4: world.BrushMaterial = Cell.Smoke; break;
                case ConsoleKey.D5: world.BrushMaterial = Cell.Wall; break;
                case ConsoleKey.D6: world.BrushMaterial = Cell.Empty; break;
                case ConsoleKey.E: world.AddEmitter(curR, curC, world.BrushMaterial); break;
                case ConsoleKey.F: world.ClearEmitters(); break;
                case ConsoleKey.P: paused = !paused; break;
                case ConsoleKey.C: world.Clear(keepWalls: true); break;
                case ConsoleKey.X: world.Clear(keepWalls: false); break;
                case ConsoleKey.Oem4: brush = Math.Max(0, brush - 1); world.BrushRadius = brush; break;
                case ConsoleKey.Oem6: brush = Math.Min(40, brush + 1); world.BrushRadius = brush; break;
                case ConsoleKey.OemComma: subSteps = Math.Max(1, subSteps - 1); break;
                case ConsoleKey.OemPeriod: subSteps = Math.Min(8, subSteps + 1); break;
            }
        }

        /// <summary>The one-line status/help bar under the grid.</summary>
        private static string StatusLine(FallingSandWorld world, int brush, int subSteps, bool pen, bool paused)
        {
            string state = paused ? " [PAUSED]" : (pen ? " [PEN DOWN]" : "");
            return Reset +
                $"{W}x{H}  material:{Cell.Name[world.BrushMaterial]}  brush:{brush}  substeps:{subSteps}  faucets:{world.EmitterCount}{state}\n" +
                "WASD move · Space pen · 1-6 material · E faucet · F clear-faucets · [ ] brush · P pause · C clear · X wipe · Q quit   ";
        }

        // ---------------- Windows ANSI enablement ----------------
        [DllImport("kernel32.dll", SetLastError = true)] private static extern IntPtr GetStdHandle(int nStdHandle);
        [DllImport("kernel32.dll")] private static extern bool GetConsoleMode(IntPtr handle, out uint mode);
        [DllImport("kernel32.dll")] private static extern bool SetConsoleMode(IntPtr handle, uint mode);

        /// <summary>Turns on virtual-terminal (ANSI) processing so escape codes render on Windows consoles. Best-effort; harmless if it fails or off Windows.</summary>
        private static void EnableAnsi()
        {
            try
            {
                Console.OutputEncoding = Encoding.UTF8;
                if (!OperatingSystem.IsWindows()) return;
                const int STD_OUTPUT_HANDLE = -11;
                const uint ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;
                var h = GetStdHandle(STD_OUTPUT_HANDLE);
                if (GetConsoleMode(h, out uint mode)) SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            }
            catch { }
        }
    }
}
